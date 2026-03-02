#!/usr/bin/env python3
"""Quick end-to-end test with tiny latencies to validate the full pipeline."""

from kv_cache_rl.lru import LRUPolicy
from kv_cache_rl.model import generate_tokens, load_model
from kv_cache_rl.storage import SimulatedStorage
import time

def _block_id(layer: int, position: int) -> str:
    return f"L{layer}_P{position}"


def main():
    model, tokenizer, device = load_model()

    policy = LRUPolicy()
    storage = SimulatedStorage(
        policy=policy,
        gpu_capacity=20,
        cpu_capacity=50,
        cpu_latency=0.001,   # 1 ms instead of 50 ms
        disk_latency=0.005,  # 5 ms instead of 500 ms
        verbose=False,
    )

    from kv_cache_rl import DEFAULT_PROMPT
    prompt = DEFAULT_PROMPT
    num_layers = model.config.n_layer
    max_new_tokens = 10

    generated_tokens = []
    prev_positions = 0
    wall_start = time.perf_counter()

    for step, token_id, past_key_values in generate_tokens(
        model, tokenizer, prompt, max_new_tokens, device
    ):
        generated_tokens.append(token_id)

        total_positions = past_key_values[0][0].shape[2]

        for pos in range(prev_positions, total_positions):
            for layer_idx in range(num_layers):
                bid = _block_id(layer_idx, pos)
                k = past_key_values[layer_idx][0][:, :, pos:pos+1, :].cpu()
                v = past_key_values[layer_idx][1][:, :, pos:pos+1, :].cpu()
                storage.store(bid, (k, v))

        prev_positions = total_positions

        for pos in range(total_positions):
            for layer_idx in range(num_layers):
                bid = _block_id(layer_idx, pos)
                storage.get_data(bid)

        elapsed = time.perf_counter() - wall_start
        tok_text = tokenizer.decode([token_id])
        print(f"  step {step:>3d}  tok={tok_text!r:>12s}  "
              f"cache_positions={total_positions}  "
              f"wall={elapsed:.2f}s  sim_lat={storage.total_latency:.4f}s")

    wall_total = time.perf_counter() - wall_start

    # Decoded text
    full_ids = tokenizer.encode(prompt) + generated_tokens
    decoded = tokenizer.decode(full_ids)
    print(f"\n[test] Generated text:\n{decoded}\n")

    # Final KV-cache shape
    print("[test] Final past_key_values shape:")
    for i, (k, v) in enumerate(past_key_values):
        print(f"  Layer {i:>2d}  key={k.shape}  value={v.shape}")

    # Storage report
    stats = storage.report()
    print(f"[test] Total wall-clock time: {wall_total:.2f}s")

    # Assertions
    assert stats["total_accesses"] > 0, "Should have some accesses"
    assert stats["cpu_hits"] + stats["disk_hits"] > 0, "Should have tier spills (cpu/disk)"
    assert stats["gpu_occupancy"] == 20, f"GPU should be full at capacity 20"
    assert len(generated_tokens) == max_new_tokens, f"Expected {max_new_tokens} tokens"
    print("\n✅ All assertions passed!")


if __name__ == "__main__":
    main()
