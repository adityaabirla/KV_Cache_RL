"""
lfu.py — Least-Frequently-Used eviction policy.

Evicts the block with the lowest access count.
Ties are broken by evicting the least-recently-used among the
lowest-frequency blocks.
"""

from __future__ import annotations

from collections import defaultdict, OrderedDict


class LFUPolicy:
    """
    Track access frequency for GPU-resident blocks.

    * ``access(block_id)`` — increment frequency counter.
    * ``evict()``          — return (and forget) the *least frequently used* block.
    * ``remove(block_id)`` — stop tracking a block.
    """

    def __init__(self) -> None:
        # block_id → frequency
        self._freq: dict[str, int] = {}
        # frequency → OrderedDict of block_ids (insertion-ordered for tie-breaking)
        self._freq_to_blocks: defaultdict[int, OrderedDict[str, None]] = defaultdict(OrderedDict)
        self._min_freq: int = 0

    # ── EvictionPolicy interface ─────────────────────────────────────

    def access(self, block_id: str) -> None:
        """Increment *block_id*'s frequency counter."""
        if block_id in self._freq:
            old_freq = self._freq[block_id]
            self._freq_to_blocks[old_freq].pop(block_id)
            if not self._freq_to_blocks[old_freq]:
                del self._freq_to_blocks[old_freq]
                if self._min_freq == old_freq:
                    self._min_freq = old_freq + 1
            new_freq = old_freq + 1
            self._freq[block_id] = new_freq
            self._freq_to_blocks[new_freq][block_id] = None
        else:
            self._freq[block_id] = 1
            self._freq_to_blocks[1][block_id] = None
            self._min_freq = 1

    def evict(self) -> str:
        """Pop and return the least-frequently-used block_id (LRU tie-break)."""
        if not self._freq:
            raise RuntimeError("LFUPolicy: nothing to evict (empty)")
        # Pop the oldest (LRU) block from the lowest-frequency bucket
        bucket = self._freq_to_blocks[self._min_freq]
        block_id, _ = bucket.popitem(last=False)
        if not bucket:
            del self._freq_to_blocks[self._min_freq]
        del self._freq[block_id]
        return block_id

    def remove(self, block_id: str) -> None:
        """Stop tracking *block_id*."""
        if block_id not in self._freq:
            return
        freq = self._freq.pop(block_id)
        self._freq_to_blocks[freq].pop(block_id)
        if not self._freq_to_blocks[freq]:
            del self._freq_to_blocks[freq]
        # Recalculate min_freq if needed
        if self._freq_to_blocks:
            self._min_freq = min(self._freq_to_blocks)
        else:
            self._min_freq = 0

    # ── helpers ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._freq)

    def __repr__(self) -> str:
        return f"LFUPolicy(tracked={len(self)})"
