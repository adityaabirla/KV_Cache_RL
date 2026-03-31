from importance_scorer import BlockState, ImportanceScorer


class ImportancePolicy:
    def __init__(self, num_layers=12, seq_len=2048):
        self.scorer = ImportanceScorer(num_layers, seq_len)

        self.blocks = {}       # str block_id -> BlockState
        self._id_to_str = {}   # int block_id -> str block_id  (reverse lookup)
        self.step = 0
        self.next_id = 0

    # ── step counter ───────────────────────────

    def tick(self, step: int) -> None:
        """Called by main.py each generation step to advance the clock."""
        self.step = step

    # ── required interface ─────────────────────

    def access(self, block_id: str, tier: int = 0) -> None:
        """
        Record an access to block_id.
        tier: current storage tier of the block (0=GPU, 1=CPU, 2=Disk).
        Creating a new block defaults tier to GPU (0).
        """
        if block_id not in self.blocks:
            layer, pos = self._parse_block_id(block_id)
            int_id = self.next_id
            self.next_id += 1

            self.blocks[block_id] = BlockState(
                block_id=int_id,
                position=pos,
                layer=layer,
                tier=tier,
                creation_step=self.step,
            )
            self._id_to_str[int_id] = block_id
        else:
            # update tier in case block moved since last access
            self.blocks[block_id].tier = tier

        self.blocks[block_id].record_access(self.step)

    def evict(self, candidates: list[str] | None = None) -> str:
        """
        Choose and return the least-important block.
        If candidates provided, only consider blocks in that list.
        Otherwise, use blocks currently in the GPU (tier 0).
        """
        if candidates is not None:
            # Filter to only tracked blocks that are in candidates (string block IDs)
            candidate_blocks = [self.blocks[bid] for bid in candidates if bid in self.blocks]
        else:
            # Default: ONLY look at blocks currently in Tier 0 (GPU)
            candidate_blocks = [b for b in self.blocks.values() if b.tier == 0]
        
        if not candidate_blocks:
            raise RuntimeError("ImportancePolicy: no blocks to evict")

        evict_block = self.scorer.evict_candidate(candidate_blocks, current_step=self.step)

        str_bid = self._id_to_str.get(evict_block.block_id)
        if str_bid is None:
            # Fallback: use first candidate (fallback should map to string block_id)
            str_bid = candidates[0] if candidates else list(self.blocks.keys())[0]

        # REMOVED: self.remove(str_bid) 
        # We want to keep tracking it as it moves to CPU/Disk!
        return str_bid

    def remove(self, block_id: str) -> None:
        """Remove a block from tracking (called after eviction or explicit drop)."""
        block = self.blocks.pop(block_id, None)
        if block is not None:
            self._id_to_str.pop(block.block_id, None)

    def notify_tier_change(self, block_id: str, new_tier: int) -> None:
        """
        Called when storage moves a block between tiers (GPU→CPU, CPU→Disk, etc.)
        so the importance scorer always sees the correct tier signal.
        """
        if block_id in self.blocks:
            old_tier = self.blocks[block_id].tier
            self.blocks[block_id].tier = new_tier

            if new_tier < old_tier:
                self.blocks[block_id].num_promotions += 1

    # ── helper ─────────────────────────────────

    def _parse_block_id(self, bid: str):
        # expects "L{layer}_P{position}"
        parts = bid.split("_")
        layer = int(parts[0][1:])
        pos   = int(parts[1][1:])
        return layer, pos

    def __len__(self):
        return len(self.blocks)