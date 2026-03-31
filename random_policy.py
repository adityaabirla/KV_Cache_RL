"""
random_policy.py — Random eviction policy.

Evicts a uniformly random block from the tracked set.
"""

from __future__ import annotations

import random


class RandomPolicy:
    """
    Track GPU-resident blocks and evict one at random.

    * ``access(block_id)`` — register the block.
    * ``evict()``          — return (and forget) a *random* block.
    * ``remove(block_id)`` — stop tracking a block.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._blocks: list[str] = []
        self._index: dict[str, int] = {}   # block_id → position in _blocks
        self._rng = random.Random(seed)

    # ── EvictionPolicy interface ─────────────────────────────────────

    def access(self, block_id: str) -> None:
        """Register *block_id*. No-op if already tracked."""
        if block_id not in self._index:
            self._index[block_id] = len(self._blocks)
            self._blocks.append(block_id)

    def evict(self, candidates: list[str] | None = None) -> str:
        """Pop and return a random block_id (O(1) swap-and-pop).
        
        If candidates is provided, only consider blocks in that list."""
        if not self._blocks:
            raise RuntimeError("RandomPolicy: nothing to evict (empty)")
        
        # Filter to only candidate blocks if provided
        if candidates is not None:
            candidate_set = set(candidates)
            valid_idxs = [i for i, bid in enumerate(self._blocks) if bid in candidate_set]
            if not valid_idxs:
                raise RuntimeError("RandomPolicy: no valid candidates to evict")
            idx = self._rng.choice(valid_idxs)
        else:
            idx = self._rng.randrange(len(self._blocks))
        
        # Swap with last element for O(1) removal
        last = self._blocks[-1]
        self._blocks[idx] = last
        self._index[last] = idx
        victim = self._blocks.pop()
        del self._index[victim]
        return victim

    def remove(self, block_id: str) -> None:
        """Stop tracking *block_id*."""
        if block_id not in self._index:
            return
        idx = self._index[block_id]
        last = self._blocks[-1]
        self._blocks[idx] = last
        self._index[last] = idx
        self._blocks.pop()
        del self._index[block_id]

    # ── helpers ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        return f"RandomPolicy(tracked={len(self)})"
