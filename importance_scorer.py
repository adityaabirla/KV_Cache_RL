"""
KV Cache Importance Scorer + RL State/Action Space
===================================================
Milestone 3: Block-level importance tracking and RL formulation
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# ──────────────────────────────────────────────
# 1. BLOCK STATE  (extended feature set)
# ──────────────────────────────────────────────

@dataclass
class BlockState:
    """
    Full per-block feature set tracked throughout the simulation.
    Tier legend: 0=GPU, 1=CPU, 2=Disk
    """
    block_id: int
    position: int          # token index in the sequence
    layer: int             # transformer layer index

    # --- mutable fields ---
    tier: int = 0                          # current location
    access_count: int = 0
    last_access_step: int = 0
    creation_step: int = 0
    num_promotions: int = 0

    # access history for reuse-gap computation
    _access_history: List[int] = field(default_factory=list, repr=False)

    # ── Core features ─────────────────────────────────────────────────

    def record_access(self, current_step: int) -> None:
        """Call every time this block is read by an attention head."""
        self._access_history.append(current_step)
        self.access_count += 1
        self.last_access_step = current_step

    def promote(self, target_tier: int) -> None:
        """Move block to a higher-memory (lower number) tier."""
        if target_tier < self.tier:
            self.num_promotions += 1
        self.tier = target_tier

    # ── Derived features ──────────────────────────────────────────────

    def age(self, current_step: int) -> int:
        return max(1, current_step - self.creation_step)

    def recency(self, current_step: int) -> int:
        return current_step - self.last_access_step

    def access_frequency(self, current_step: int) -> float:
        return self.access_count / self.age(current_step)

    def reuse_gap(self) -> float:
        """
        Average number of steps between consecutive accesses.
        Returns 0 if fewer than 2 accesses have been recorded.
        """
        if len(self._access_history) < 2:
            return 0.0
        gaps = [
            self._access_history[i] - self._access_history[i - 1]
            for i in range(1, len(self._access_history))
        ]
        return sum(gaps) / len(gaps)

    def feature_vector(self, current_step: int, num_layers: int, seq_len: int, global_stats: dict | None = None) -> np.ndarray:
        """
        Returns the 10-dimensional normalised feature vector used as the
        RL agent's observation for this single block.

        Index  Feature                  Normalisation
        -----  -----------------------  ---------------------------
        0      tier                     / 2  (GPU→0, Disk→1)
        1      access_count             log1p / 10
        2      last_access_step         / current_step
        3      creation_step            / current_step
        4      num_promotions           log1p / 5
        5      age (derived)            log1p / log1p(current_step)
        6      recency (derived)        / current_step
        7      access_frequency         log1p
        8      reuse_gap (derived)      log1p / log1p(max_gap)
        9      position (structural)    / seq_len
        10     layer (structural)       / num_layers
        """

        eps = 1e-8

        age_val   = self.age(current_step)
        recency_v = self.recency(current_step)
        freq_v    = self.access_frequency(current_step)
        gap_v     = self.reuse_gap()

        # ---- relative context ----
        rel_recency = 0.0
        rel_freq    = 0.0

        if global_stats:
            rel_recency = recency_v / (global_stats["avg_recency"] + eps)
            rel_freq    = freq_v    / (global_stats["avg_freq"] + eps)

        vec = np.array([
            self.tier / 2.0,
            math.log1p(self.access_count) / 10.0,
            self.last_access_step / max(1, current_step),
            self.creation_step    / max(1, current_step),
            math.log1p(self.num_promotions) / 5.0,

            math.log1p(age_val)   / max(eps, math.log1p(current_step)),
            recency_v             / max(1, current_step),
            math.log1p(freq_v) / math.log1p(10.0),

            math.log1p(gap_v)     / max(eps, math.log1p(max(1, current_step))),
            self.position         / max(1, seq_len - 1),
            self.layer            / max(1, num_layers - 1),

            # 🔥 NEW FEATURES
            min(1.0, rel_recency),
            min(1.0, rel_freq),

        ], dtype=np.float32)

        return np.clip(vec, 0.0, 1.0)


# ──────────────────────────────────────────────
# 2. IMPORTANCE SCORER
# ──────────────────────────────────────────────

class ImportanceScorer:
    """
    Computes a scalar importance score for each KV-cache block.

    The score blends four signals with configurable weights:

        importance = w_freq  * frequency_score
                   + w_rec   * recency_score
                   + w_pos   * positional_score
                   + w_layer * layer_score

    Lower score → better eviction candidate (LFU-style: evict least important).

    Usage
    -----
        scorer = ImportanceScorer(num_layers=32, seq_len=2048)
        score  = scorer.score(block, current_step=500)
        blocks.sort(key=lambda b: scorer.score(b, step))
        evict = blocks[0]   # lowest importance
    """

    def __init__(
        self,
        num_layers: int,
        seq_len: int,
        w_freq:  float = 0.40,
        w_rec:   float = 0.30,
        w_pos:   float = 0.15,
        w_layer: float = 0.15,
    ):
        self.num_layers = num_layers
        self.seq_len    = seq_len
        # weights must sum to 1.0
        total = w_freq + w_rec + w_pos + w_layer
        self.w_freq  = w_freq  / total
        self.w_rec   = w_rec   / total
        self.w_pos   = w_pos   / total
        self.w_layer = w_layer / total

    # ── Individual signals (all → [0, 1], higher = keep) ─────────────

    def _frequency_score(self, block: BlockState, step: int) -> float:
        """
        Access frequency normalised with log smoothing.
        High frequency → high score → keep.
        """
        freq = block.access_frequency(step)
        return min(1.0, math.log1p(freq) / math.log1p(10.0))

    def _recency_score(self, block: BlockState, step: int) -> float:
        """
        Exponential decay instead of linear ratio.
        More stable and closer to attention patterns.
        """
        recency = block.recency(step)
        tau = max(5.0, block.reuse_gap() + 1.0)
        return math.exp(-recency / tau)

    def _positional_score(self, block: BlockState) -> float:
        """
        Early positions (small index) are accessed more in attention
        (BOS tokens, system prompt, etc.) → higher score.
        Score decays from 1.0 (pos=0) toward 0.5 (pos=seq_len-1).
        """
        norm_pos = block.position / max(1, self.seq_len - 1)
        return 1.0 - 0.5 * norm_pos

    def _layer_score(self, block: BlockState) -> float:
        """
        Middle layers typically show the highest attention entropy
        → they are reused most unpredictably, so bias toward keeping them.
        Score is a bell curve peaking at the middle layer.
        """
        mid   = (self.num_layers - 1) / 2.0
        sigma = self.num_layers / 4.0
        norm  = block.layer / max(1, self.num_layers - 1)
        mid_n = 0.5
        score = math.exp(-0.5 * ((norm - mid_n) / (sigma / self.num_layers)) ** 2)
        return score

    # ── Combined importance ────────────────────────────────────────────

    def score(self, block: BlockState, current_step: int) -> float:
        """
        Returns a score in [0, 1].
        Higher score  →  more important  →  keep.
        Lower  score  →  evict candidate.
        """
        s_freq  = self._frequency_score(block, current_step)
        s_rec   = self._recency_score(block, current_step)
        s_pos   = self._positional_score(block)
        s_layer = self._layer_score(block)

        return (
            self.w_freq  * s_freq
          + self.w_rec   * s_rec
          + self.w_pos   * s_pos
          + self.w_layer * s_layer
        )

    def rank_by_eviction_priority(
        self, blocks: List[BlockState], current_step: int
    ) -> List[Tuple[float, BlockState]]:
        """
        Returns blocks sorted ascending by importance score.
        The first element is the *best eviction candidate*.
        """
        scored = [(self.score(b, current_step), b) for b in blocks]
        return sorted(scored, key=lambda x: x[0])

    def evict_candidate(
        self, blocks: List[BlockState], current_step: int
    ) -> BlockState:
        """Return the single best block to evict (lowest importance)."""
        ranked = self.rank_by_eviction_priority(blocks, current_step)
        return ranked[0][1]
        
    def update_weights_from_experience(
        self,
        feature_vector: np.ndarray,   # saved at eviction time
        reward: float,
        lr: float = 0.02,
    ) -> None:
        """
        Online weight nudge from a deferred RL reward.

        reward == 0  → perfect eviction (block never re-accessed), no update.
        reward  < 0  → costly eviction (block was needed soon from CPU/Disk).
                    We increase the weights for whichever signals were
                    strongest in the wrongly-evicted block, so similar
                    blocks score higher — and survive eviction — next time.

        Feature-index → weight mapping (matches feature_vector layout):
            [1] log(access_count), [7] log(freq)      → w_freq
            [2] last_access,       [6] recency         → w_rec
            [9] position                               → w_pos
            [10] layer                                 → w_layer
        """
        if reward >= 0:
            return   # good eviction — nothing to learn

        penalty = abs(reward)

        freq_signal  = (feature_vector[1] + feature_vector[7]) / 2.0
        rec_signal   = (feature_vector[2] + feature_vector[6]) / 2.0
        pos_signal   = float(feature_vector[9])
        layer_signal = float(feature_vector[10])

        self.w_freq  += lr * penalty * freq_signal
        self.w_rec   += lr * penalty * rec_signal
        self.w_pos   += lr * penalty * pos_signal
        self.w_layer += lr * penalty * layer_signal

        total = self.w_freq + self.w_rec + self.w_pos + self.w_layer
        if total > 0:
            self.w_freq  /= total
            self.w_rec   /= total
            self.w_pos   /= total
            self.w_layer /= total


# ──────────────────────────────────────────────
# 3. RL STATE / ACTION SPACE
# ──────────────────────────────────────────────

class RLState:
    """
    Packages the agent's observation at eviction time.

    The state is a 2-D array:
        shape = (k, FEATURE_DIM)
        k     = number of candidate blocks presented to the agent
                (all GPU blocks, or a sampled subset for scalability)

    FEATURE_DIM = 13  (see BlockState.feature_vector)
    """

    FEATURE_DIM: int = 13

    def __init__(
        self,
        candidates: List[BlockState],
        current_step: int,
        num_layers: int,
        seq_len: int,
        max_candidates: int = 64,
    ):
        self.block_ids: List[int] = []
        self.matrix: np.ndarray  = self._build(
            candidates, current_step, num_layers, seq_len, max_candidates
        )

    def _build(
        self,
        candidates: List[BlockState],
        step: int,
        num_layers: int,
        seq_len: int,
        k: int,
    ) -> np.ndarray:

        # ---- compute global stats ----
        recencies = [b.recency(step) for b in candidates]
        freqs     = [b.access_frequency(step) for b in candidates]

        global_stats = {
            "avg_recency": np.mean(recencies) if recencies else 1.0,
            "avg_freq":    np.mean(freqs)     if freqs else 1.0,
        }

        # ---- sample if needed ----
        if len(candidates) > k:
            rng = np.random.default_rng()
            indices = rng.choice(len(candidates), size=k, replace=False)
            sampled = [candidates[i] for i in indices]
        else:
            sampled = list(candidates)

        # ---- sort AFTER sampling (stabilizes learning) ----
        sampled = sorted(
            sampled,
            key=lambda b: (b.recency(step), -b.access_count)
        )

        self.block_ids = [b.block_id for b in sampled]

        mat = np.stack([
            b.feature_vector(step, num_layers, seq_len, global_stats)
            for b in sampled
        ])

        return mat.astype(np.float32)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.matrix.shape          # (k, 13)

    def flat(self) -> np.ndarray:
        """Flattened observation for MLP-based agents."""
        return self.matrix.flatten()


class RLAction:
    """
    Defines the action space and translates agent output to a block.

    Option A – Discrete
    -------------------
    action ∈ {0, 1, ..., k-1}
    Agent picks the index of the block to evict directly.
    Good for DQN with small k.

    Option B – Scoring
    ------------------
    Agent outputs a score vector of shape (k,).
    Block with the *lowest* score is evicted.
    Closer to optimal; suitable for PPO / actor-critic.
    """

    @staticmethod
    def discrete_to_block(
        action_idx: int,
        state: RLState,
        candidate_map: dict,   # block_id → BlockState
    ) -> BlockState:
        """Option A: map integer action index → BlockState."""
        block_id = state.block_ids[action_idx]
        return candidate_map[block_id]

    @staticmethod
    def scores_to_block(
        scores: np.ndarray,    # shape (k,) – lower = evict
        state: RLState,
        candidate_map: dict,
    ) -> BlockState:
        """Option B: map score vector → lowest-scored BlockState."""
        best_local_idx = int(np.argmin(scores))
        block_id       = state.block_ids[best_local_idx]
        return candidate_map[block_id]


class RLReward:
    """
    Computes the deferred reward for an eviction decision.

    reward = − latency_incurred_by_eviction

    If the evicted block is re-accessed later:
        from GPU  →  0          (shouldn't happen after eviction, but guard)
        from CPU  →  -0.005
        from Disk →  -0.020
    If never re-accessed:
        reward = 0              (perfect eviction)

    The reward is recorded *at the time of re-access*, then associated
    back to the eviction step via the replay buffer.
    """

    LATENCY: dict = {
        0: 0.000,   # GPU hit  (no penalty)
        1: 0.005,   # CPU hit
        2: 0.020,   # Disk hit
    }

    def __init__(self):
        # pending_evictions[block_id] = eviction_step
        self._pending: dict[int, tuple[int, BlockState]] = {}

    def record_eviction(self, block: BlockState, eviction_step: int) -> None:
        self._pending[block.block_id] = (eviction_step, block)

    def resolve(
        self,
        block_id: int,
        retrieved_from_tier: int,
        current_step: int,
    ) -> Optional[float]:
        """
        Call when a previously-evicted block is retrieved.
        Returns the reward (negative latency) and removes the pending entry.
        Returns None if block_id was not being tracked.
        NEW: penalize early reuse more to encourage better eviction choices.
        """
        if block_id not in self._pending:
            return None

        eviction_step, block = self._pending[block_id]
        del self._pending[block_id]

        latency_penalty = self.LATENCY[retrieved_from_tier]

        time_gap = current_step - eviction_step
        expected_gap = max(1.0, block.reuse_gap())

        relative_gap = time_gap / expected_gap
        urgency_penalty = math.exp(-relative_gap)

        return -(latency_penalty + 0.3 * urgency_penalty)

    def sweep_unreaccessed(
        self,
        current_step: int,
        horizon: int = 500,
    ) -> List[Tuple[int, float]]:
        """
        At the end of a generation run, blocks never re-accessed get reward=0.
        Returns list of (block_id, 0.0) and clears them from pending.
        """
        resolved = []

        stale = [
            bid for bid, (step, _) in self._pending.items()
            if (current_step - step) > horizon
        ]

        for bid in stale:
            del self._pending[bid]
            resolved.append((bid, 0.0))

        return resolved


# ──────────────────────────────────────────────
# 4. INTEGRATION HELPER  (wires scorer into RL)
# ──────────────────────────────────────────────

class CacheEvictionAgent:
    """
    Thin wrapper that combines ImportanceScorer + RLState/Action.
    Drop-in replacement for existing heuristic policies.

    Phase 1 (imitation):  use `heuristic_action` to generate labels.
    Phase 2 (RL):         replace `select_eviction` with a learned policy.
    """

    def __init__(
        self,
        num_layers: int,
        seq_len: int,
        max_candidates: int = 64,
    ):
        self.scorer        = ImportanceScorer(num_layers, seq_len)
        self.reward_signal = RLReward()
        self.num_layers    = num_layers
        self.seq_len       = seq_len
        self.max_candidates = max_candidates

    def build_state(
        self,
        candidates: List[BlockState],
        current_step: int,
    ) -> RLState:
        return RLState(
            candidates, current_step,
            self.num_layers, self.seq_len, self.max_candidates
        )

    def heuristic_action(
        self,
        state: RLState,
        candidates: List[BlockState],
        current_step: int,
    ) -> int:
        """
        Phase-1 teacher signal.
        Returns the local index (within state.block_ids) of the block the
        importance scorer would evict — used as the imitation label.
        """
        scores = np.array([
            self.scorer.score(b, current_step) for b in candidates
            if b.block_id in state.block_ids
        ])
        # map to the index order in state.block_ids
        id_to_score = {b.block_id: self.scorer.score(b, current_step)
                       for b in candidates}
        local_scores = np.array([id_to_score[bid] for bid in state.block_ids])
        return int(np.argmin(local_scores))   # index of least important block

    def select_eviction(
        self,
        candidates: List[BlockState],
        current_step: int,
        policy_fn=None,        # callable(RLState) → np.ndarray | int
        mode: str = "heuristic",
    ) -> BlockState:
        """
        Unified eviction selector.

        mode='heuristic'  → use ImportanceScorer (Phase 1 / baseline)
        mode='rl'         → call policy_fn(state) and decode action
        """
        if mode == "heuristic" or policy_fn is None:
            return self.scorer.evict_candidate(candidates, current_step)

        state       = self.build_state(candidates, current_step)
        candidate_map = {b.block_id: b for b in candidates}
        action      = policy_fn(state)          # int or np.ndarray

        if isinstance(action, (int, np.integer)):
            return RLAction.discrete_to_block(int(action), state, candidate_map)
        else:
            return RLAction.scores_to_block(np.asarray(action), state, candidate_map)


# ──────────────────────────────────────────────
# 5. QUICK SMOKE TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import random

    NUM_LAYERS, SEQ_LEN = 32, 2048
    rng = random.Random(42)

    # Build 10 fake blocks
    blocks: List[BlockState] = []
    for i in range(10):
        b = BlockState(
            block_id=i,
            position=rng.randint(0, SEQ_LEN - 1),
            layer=rng.randint(0, NUM_LAYERS - 1),
            tier=rng.choice([0, 1, 2]),
            creation_step=rng.randint(0, 100),
        )
        # simulate some access history
        for step in sorted(rng.sample(range(1, 500), k=rng.randint(1, 10))):
            b.record_access(step)
        blocks.append(b)

    current_step = 500
    agent = CacheEvictionAgent(num_layers=NUM_LAYERS, seq_len=SEQ_LEN)

    print("=" * 60)
    print("IMPORTANCE SCORES  (higher = keep, lower = evict)")
    print("=" * 60)
    ranked = agent.scorer.rank_by_eviction_priority(blocks, current_step)
    for rank, (score, b) in enumerate(ranked):
        print(
            f"  rank {rank+1:>2}  block {b.block_id:>2}"
            f"  tier={'GPU' if b.tier==0 else 'CPU' if b.tier==1 else 'Dsk'}"
            f"  pos={b.position:>4}  layer={b.layer:>2}"
            f"  accesses={b.access_count:>2}"
            f"  score={score:.4f}"
        )

    evict_block = ranked[0][1]
    print(f"\n→ Evict candidate: block {evict_block.block_id}  (score={ranked[0][0]:.4f})")

    print("\n" + "=" * 60)
    print("RL STATE  shape:", agent.build_state(blocks, current_step).shape)
    print("RL STATE  first block feature vector:")
    state = agent.build_state(blocks, current_step)
    fv = blocks[0].feature_vector(current_step, NUM_LAYERS, SEQ_LEN)
    labels = [
        "tier", "access_count", "last_access", "creation",
        "promotions", "age", "recency", "frequency", "reuse_gap",
        "position", "layer",
    ]
    for lbl, val in zip(labels, fv):
        print(f"    {lbl:<16} {val:.4f}")

    print("\n" + "=" * 60)
    print("HEURISTIC ACTION (imitation label for Phase-1 training)")
    action_idx = agent.heuristic_action(state, blocks, current_step)
    print(f"  Local index to evict: {action_idx}  (block_id={state.block_ids[action_idx]})")

    print("\n" + "=" * 60)
    print("REWARD SIGNAL demo")
    reward_mgr = RLReward()
    reward_mgr.record_eviction(evict_block, eviction_step=current_step)
    # simulate re-access from CPU
    r = reward_mgr.resolve(evict_block.block_id, retrieved_from_tier=1, current_step=current_step)
    print(f"  Evicted block re-accessed from CPU → reward = {r}")
