#!/usr/bin/env python3
"""
test_tiers.py — Tests that exercise ALL three storage tiers (GPU / CPU / Disk)
and prove we get a healthy mix of hits.

Test 1 — Pure storage unit test (no model, instant, deterministic)
Test 2 — Model integration with capacities tuned for mixed hits
"""

from __future__ import annotations

import time


# ═══════════════════════════════════════════════════════════════════════
# Test 1  — Pure storage / LRU unit test  (no model, runs instantly)
# ═══════════════════════════════════════════════════════════════════════

def test_storage_unit():
    """
    Tiny capacity so we can hand-trace every eviction and verify
    GPU / CPU / disk hits precisely.

    Capacity: GPU=3, CPU=3  →  total fast storage = 6 blocks.

    Sequence:
      1. Store blocks A, B, C, D, E, F, G, H  (8 blocks)
         - GPU always holds the 3 most recent stores.
         - Evictions cascade GPU→CPU→Disk.
      2. Read back specific blocks and assert the tier they come from.
    """
    from kv_cache_rl.lru import LRUPolicy
    from kv_cache_rl.storage import SimulatedStorage

    policy = LRUPolicy()
    s = SimulatedStorage(
        policy=policy,
        gpu_capacity=3,
        cpu_capacity=3,
        cpu_latency=0.0,     # no actual sleep — we just count hits
        disk_latency=0.0,
        verbose=True,
    )

    # --- store 8 blocks one by one ---
    # After each store the GPU holds the 3 most recent.
    # Evictions: GPU→CPU when GPU is full, CPU→Disk when CPU is full.
    for name in list("ABCDEFGH"):
        s.store(name, f"data-{name}")

    # After storing A..H:
    #   GPU : F, G, H   (most recent 3)
    #   CPU : C, D, E   (middle 3)
    #   Disk: A, B       (oldest, spilled when CPU filled)
    print(f"\n  GPU  = {list(s.gpu.store.keys())}")
    print(f"  CPU  = {list(s.cpu.store.keys())}")
    print(f"  Disk = {list(s.disk.store.keys())}")

    # --- read-back ---
    s.get_data("H")   # GPU hit  (most recent)
    s.get_data("F")   # GPU hit
    s.get_data("D")   # CPU hit  → promoted to GPU, evicts G (LRU) to CPU
    s.get_data("A")   # Disk hit → promoted to GPU, evicts H (LRU) to CPU (CPU evicts C to disk)
    s.get_data("H")   # CPU hit  (H was just demoted to CPU by the A promotion)

    print(f"\n  GPU  hits : {s.gpu_hits}")
    print(f"  CPU  hits : {s.cpu_hits}")
    print(f"  Disk hits : {s.disk_hits}")
    print(f"  Total     : {s.gpu_hits + s.cpu_hits + s.disk_hits}")

    assert s.gpu_hits  == 2, f"Expected 2 GPU hits, got {s.gpu_hits}"
    assert s.cpu_hits  == 2, f"Expected 2 CPU hits, got {s.cpu_hits}"
    assert s.disk_hits == 1, f"Expected 1 Disk hit, got {s.disk_hits}"

    print("\n  ✅ Test 1 PASSED — all three tiers register hits.\n")


# ═══════════════════════════════════════════════════════════════════════
# Test 2  — Model integration: capacity tuned for a good hit-mix
# ═══════════════════════════════════════════════════════════════════════

def test_model_mixed_hits():
    """
    GPT-2 small: 12 layers, 7-token prompt → 84 blocks at step 0.
    Each new token adds 12 blocks.

    With gpu_capacity=100 and cpu_capacity=100 the first few steps
    fit entirely on GPU (= GPU hits during read-back).  Once total
    blocks exceed 100 some spill to CPU, then to disk after 200.

    We generate 15 tokens (total blocks = (7+15)×12 = 264) so all
    three tiers are exercised.

    Latencies are set to 0.001 / 0.002 so the test finishes fast.
    """
    from kv_cache_rl.lru import LRUPolicy
    from kv_cache_rl.model import generate_tokens, load_model
    from kv_cache_rl.storage import SimulatedStorage

    model, tokenizer, device = load_model()

    policy = LRUPolicy()
    storage = SimulatedStorage(
        policy=policy,
        gpu_capacity=100,
        cpu_capacity=100,
        cpu_latency=0.001,
        disk_latency=0.002,
        verbose=False,
    )

    num_layers = model.config.n_layer
    from kv_cache_rl import DEFAULT_PROMPT
    prompt = DEFAULT_PROMPT
    max_new_tokens = 15

    def _bid(layer, pos):
        return f"L{layer}_P{pos}"

    generated_tokens = []
    prev_positions = 0
    wall_start = time.perf_counter()

    for step, token_id, past_key_values in generate_tokens(
        model, tokenizer, prompt, max_new_tokens, device
    ):
        generated_tokens.append(token_id)

        total_positions = past_key_values[0][0].shape[2]

        # Store new blocks
        for pos in range(prev_positions, total_positions):
            for layer_idx in range(num_layers):
                bid = _bid(layer_idx, pos)
                k = past_key_values[layer_idx][0][:, :, pos:pos + 1, :].cpu()
                v = past_key_values[layer_idx][1][:, :, pos:pos + 1, :].cpu()
                storage.store(bid, (k, v))
        prev_positions = total_positions

        # Read-back all blocks (simulates model needing full cache)
        for pos in range(total_positions):
            for layer_idx in range(num_layers):
                bid = _bid(layer_idx, pos)
                storage.get_data(bid)

        elapsed = time.perf_counter() - wall_start
        tok_text = tokenizer.decode([token_id])
        total_blocks = total_positions * num_layers
        print(
            f"  step {step:>2d}  tok={tok_text!r:>14s}  "
            f"blocks={total_blocks:>4d}  "
            f"gpu_hits={storage.gpu_hits}  cpu_hits={storage.cpu_hits}  "
            f"disk_hits={storage.disk_hits}  wall={elapsed:.2f}s"
        )

    # Final report
    stats = storage.report()

    # Print generated text
    full_ids = tokenizer.encode(prompt) + generated_tokens
    print(f"  Text: {tokenizer.decode(full_ids)}\n")

    # Final KV-cache shape
    print("  Final past_key_values shape:")
    for i, (k, v) in enumerate(past_key_values):
        print(f"    Layer {i:>2d}  key={k.shape}  value={v.shape}")

    # Assertions
    assert stats["gpu_hits"]  > 0, "Expected some GPU hits"
    assert stats["cpu_hits"]  > 0, "Expected some CPU hits"
    assert stats["disk_hits"] > 0, "Expected some Disk hits"
    assert len(generated_tokens) == max_new_tokens

    gpu_pct  = 100 * stats["gpu_hits"]  / stats["total_accesses"]
    cpu_pct  = 100 * stats["cpu_hits"]  / stats["total_accesses"]
    disk_pct = 100 * stats["disk_hits"] / stats["total_accesses"]
    print(f"\n  Hit distribution:  GPU {gpu_pct:.1f}%  |  CPU {cpu_pct:.1f}%  |  Disk {disk_pct:.1f}%")
    print(f"  ✅ Test 2 PASSED — all three tiers register hits with real model.\n")


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 64)
    print("  TEST 1 — Pure Storage Unit Test (no model)")
    print("=" * 64)
    test_storage_unit()

    print("=" * 64)
    print("  TEST 2 — Model Integration (mixed tier hits)")
    print("=" * 64)
    test_model_mixed_hits()

    print("=" * 64)
    print("  ALL TESTS PASSED")
    print("=" * 64)
