"""
lru.py — Least-Recently-Used eviction policy.

This is the *baseline* that the future RL agent must beat.
"""

from __future__ import annotations

from collections import OrderedDict


class LRUPolicy:
    """
    Track access order for GPU-resident blocks.

    * ``access(block_id)`` — mark as most recently used.
    * ``evict()``          — return (and forget) the *least* recently used block.
    * ``remove(block_id)`` — stop tracking a block.
    """

    def __init__(self) -> None:
        self._order: OrderedDict[str, None] = OrderedDict()

    # ── EvictionPolicy interface ─────────────────────────────────────

    def access(self, block_id: str) -> None:
        """Move *block_id* to the most-recently-used end."""
        if block_id in self._order:
            self._order.move_to_end(block_id)
        else:
            self._order[block_id] = None

    def evict(self, candidates: list[str] | None = None) -> str:
        """Pop and return the least-recently-used block_id.
        
        If candidates is provided, only consider blocks in that list."""
        if not self._order:
            raise RuntimeError("LRUPolicy: nothing to evict (empty)")
        
        # Filter to only candidate blocks if provided
        if candidates is not None:
            candidate_set = set(candidates)
            # Iterate through _order (oldest first) and find first candidate
            matched = []
            for block_id in list(self._order.keys()):
                matched.append((block_id, block_id in candidate_set))
                if block_id in candidate_set:
                    del self._order[block_id]
                    return block_id
            # No valid candidate found - provide detailed debugging info
            tracked = list(self._order.keys())
            raise RuntimeError(
                f"LRUPolicy: no valid candidates to evict. "
                f"Tracked={len(tracked)}, Candidates={len(candidates)}, "
                f"FirstMatches={matched[:5]}"
            )
        
        block_id, _ = self._order.popitem(last=False)
        return block_id
        
        block_id, _ = self._order.popitem(last=False)
        return block_id

    def remove(self, block_id: str) -> None:
        """Stop tracking *block_id*."""
        self._order.pop(block_id, None)

    # ── helpers ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._order)

    def __repr__(self) -> str:
        return f"LRUPolicy(tracked={len(self)})"
