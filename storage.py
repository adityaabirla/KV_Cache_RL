"""
storage.py — SimulatedStorage: a three-tier (GPU / CPU / Disk) block store
with configurable eviction policy and artificial latency.

The eviction policy is *injected*, so the same storage class works with the
baseline LRU and with a future RL agent.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Protocol


# ── Eviction-policy interface ────────────────────────────────────────

class EvictionPolicy(Protocol):
    """Minimal interface that any eviction strategy must satisfy."""

    def access(self, block_id: str) -> None:
        """Record that *block_id* was just accessed / stored."""
        ...

    def evict(self) -> str:
        """Return the block_id that should be evicted next."""
        ...

    def remove(self, block_id: str) -> None:
        """Stop tracking *block_id* (e.g. it was deleted entirely)."""
        ...


# ── Tier dataclass ───────────────────────────────────────────────────

class _Tier:
    """One storage tier (GPU, CPU, or Disk)."""

    def __init__(self, name: str, capacity: int | float):
        self.name = name
        self.capacity = capacity
        self.store: OrderedDict[str, Any] = OrderedDict()

    @property
    def full(self) -> bool:
        return len(self.store) >= self.capacity

    def put(self, block_id: str, data: Any) -> None:
        self.store[block_id] = data

    def pop(self, block_id: str) -> Any:
        return self.store.pop(block_id)

    def __contains__(self, block_id: str) -> bool:
        return block_id in self.store

    def __len__(self) -> int:
        return len(self.store)

    def __repr__(self) -> str:
        cap = "∞" if math.isinf(self.capacity) else str(int(self.capacity))
        return f"{self.name}[{len(self)}/{cap}]"


# ── SimulatedStorage ─────────────────────────────────────────────────

class SimulatedStorage:
    """
    Three-tier block store that **artificially slows down** data access
    depending on which tier the requested block currently lives in.

    Parameters
    ----------
    policy : EvictionPolicy
        Strategy object that decides *which* block to evict when a tier
        is full (e.g. LRU, RL agent, …).
    gpu_capacity : int
        Max blocks on the GPU tier.
    cpu_capacity : int
        Max blocks on the CPU tier.
    cpu_latency : float
        Seconds to sleep when fetching from CPU.
    disk_latency : float
        Seconds to sleep when fetching from Disk.
    verbose : bool
        If True, print every eviction / promotion event.
    """

    def __init__(
        self,
        policy: EvictionPolicy,
        gpu_capacity: int = 20,
        cpu_capacity: int = 50,
        # cpu_latency: float = 0.05,
        # disk_latency: float = 0.5,
        cpu_latency=0.001,   # 1 ms instead of 50 ms
        disk_latency=0.005,  # 5 ms instead of 500 ms
        verbose: bool = False,
    ):
        self.policy = policy
        self.gpu = _Tier("GPU", gpu_capacity)
        self.cpu = _Tier("CPU", cpu_capacity)
        self.disk = _Tier("Disk", math.inf)

        self.cpu_latency = cpu_latency
        self.disk_latency = disk_latency
        self.verbose = verbose

        # stats
        self.gpu_hits = 0
        self.cpu_hits = 0
        self.disk_hits = 0
        self.total_latency = 0.0   # accumulated *simulated* latency

    # ── internal helpers ─────────────────────────────────────────────

    def _evict_gpu_to_cpu(self) -> None:
        """Demote one block from GPU → CPU (using the policy)."""
        victim = self.policy.evict()
        data = self.gpu.pop(victim)

        if self.cpu.full:
            self._evict_cpu_to_disk()

        self.cpu.put(victim, data)
        if self.verbose:
            print(f"    [storage] evict GPU→CPU  block={victim}  {self.gpu} {self.cpu}")

    def _evict_cpu_to_disk(self) -> None:
        """Demote the LRU block from CPU → Disk."""
        # CPU eviction is always LRU among CPU-resident blocks
        # (the policy only tracks GPU-resident blocks).
        victim, data = self.cpu.store.popitem(last=False)   # oldest
        self.disk.put(victim, data)
        if self.verbose:
            print(f"    [storage] evict CPU→Disk block={victim}  {self.cpu} {self.disk}")

    def _promote_to_gpu(self, block_id: str, data: Any) -> None:
        """Bring *block_id* into GPU, evicting if necessary."""
        if self.gpu.full:
            self._evict_gpu_to_cpu()
        self.gpu.put(block_id, data)
        self.policy.access(block_id)

    # ── public API ───────────────────────────────────────────────────

    def store(self, block_id: str, data: Any) -> None:
        """Insert a **new** block (e.g. freshly computed KV-cache slice)."""
        self._promote_to_gpu(block_id, data)

    def get_data(self, block_id: str, promote: bool = True) -> Any:
        """
        Retrieve *block_id* with simulated latency:

        * GPU → instant
        * CPU → cpu_latency s, then optionally promote to GPU
        * Disk → disk_latency s, then optionally promote to GPU

        If *promote* is False the block stays in its current tier
        (avoids thrashing the GPU with cold reads).
        """
        # --- GPU hit ---
        if block_id in self.gpu:
            self.gpu_hits += 1
            self.policy.access(block_id)
            return self.gpu.store[block_id]

        # --- CPU hit ---
        if block_id in self.cpu:
            self.cpu_hits += 1
            self.total_latency += self.cpu_latency
            if promote:
                data = self.cpu.pop(block_id)
                self._promote_to_gpu(block_id, data)
            else:
                data = self.cpu.store[block_id]
            return data

        # --- Disk hit ---
        if block_id in self.disk:
            self.disk_hits += 1
            self.total_latency += self.disk_latency
            if promote:
                data = self.disk.pop(block_id)
                self._promote_to_gpu(block_id, data)
            else:
                data = self.disk.store[block_id]
            return data

        raise KeyError(f"Block {block_id!r} not found in any tier")

    def report(self) -> dict:
        """Print and return summary statistics."""
        total = self.gpu_hits + self.cpu_hits + self.disk_hits
        stats = {
            "total_accesses": total,
            "gpu_hits": self.gpu_hits,
            "cpu_hits": self.cpu_hits,
            "disk_hits": self.disk_hits,
            "simulated_latency_s": round(self.total_latency, 4),
            "gpu_occupancy": len(self.gpu),
            "cpu_occupancy": len(self.cpu),
            "disk_occupancy": len(self.disk),
        }
        print("\n[storage] ══════ Storage Report ══════")
        for k, v in stats.items():
            print(f"  {k:.<30s} {v}")
        print("[storage] ══════════════════════════\n")
        return stats
