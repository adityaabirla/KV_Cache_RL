"""
fifo.py — First-In-First-Out eviction policy.

Evicts the block that was *inserted* earliest, regardless of
how recently or frequently it was accessed.
"""

from __future__ import annotations

from collections import OrderedDict


class FIFOPolicy:
    """
    Track insertion order for GPU-resident blocks.

    * ``access(block_id)`` — register the block (does NOT reorder).
    * ``evict()``          — return (and forget) the *oldest inserted* block.
    * ``remove(block_id)`` — stop tracking a block.
    """

    def __init__(self) -> None:
        self._order: OrderedDict[str, None] = OrderedDict()

    # ── EvictionPolicy interface ─────────────────────────────────────

    def access(self, block_id: str) -> None:
        """Register *block_id*. Does NOT move it — FIFO ignores re-access."""
        if block_id not in self._order:
            self._order[block_id] = None
        # Intentionally no move_to_end — that's what makes it FIFO, not LRU.

    def evict(self) -> str:
        """Pop and return the oldest-inserted block_id."""
        if not self._order:
            raise RuntimeError("FIFOPolicy: nothing to evict (empty)")
        block_id, _ = self._order.popitem(last=False)
        return block_id

    def remove(self, block_id: str) -> None:
        """Stop tracking *block_id*."""
        self._order.pop(block_id, None)

    # ── helpers ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._order)

    def __repr__(self) -> str:
        return f"FIFOPolicy(tracked={len(self)})"
