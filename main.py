#!/usr/bin/env python3
"""
main.py - Milestone 3 entry-point.
 
Loads GPT-2, generates tokens, and routes every KV-cache block
through SimulatedStorage (GPU -> CPU -> Disk tiers) with pluggable
eviction policies: LRU, LFU, FIFO, Random, Importance.
 
Runs all policies sequentially on the same token sequence and prints
a comparison summary at the end.
"""
 
from __future__ import annotations
 
import time
 
import torch
 
from __init__ import DEFAULT_PROMPT
from lru import LRUPolicy
from lfu import LFUPolicy
from fifo import FIFOPolicy
from random_policy import RandomPolicy
from model import generate_tokens, load_model
from storage import SimulatedStorage
from importance_policy import ImportancePolicy
 
 
# -- Block-ID scheme --
# One "block" = one layer's (key, value) pair for one token position.
#   block_id = "L{layer}_P{position}"
# GPT-2 small has 12 layers, so each new token creates 12 new blocks.
 
 
def _block_id(layer: int, position: int) -> str:
    return f"L{layer}_P{position}"
 
 
def _tier_of(storage: SimulatedStorage, bid: str) -> int:
    """Return the current storage tier of a block (0=GPU, 1=CPU, 2=Disk)."""
    if bid in storage.gpu:
        return 0
    if bid in storage.cpu:
        return 1
    return 2
 
 
# -- Policy registry --
 
POLICIES = {
    "LRU":        LRUPolicy,
    "LFU":        LFUPolicy,
    "FIFO":       FIFOPolicy,
    "Random":     lambda: RandomPolicy(seed=42),
    "Importance": ImportancePolicy,
}
 
 
def _make_policy(name: str):
    """Instantiate a policy by name."""
    factory = POLICIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown policy {name!r}. Choose from: {list(POLICIES)}")
    return factory()
 
 
# -- Single-policy simulation --
 
def run_one(
    policy_name: str,
    model,
    tokenizer,
    device,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 100,
    gpu_capacity: int = 200,
    cpu_capacity: int = 400,
    cpu_latency: float = 0.005,
    disk_latency: float = 0.02,
    verbose: bool = False,
) -> dict:
    """Run the simulation with one eviction policy and return stats."""
 
    policy = _make_policy(policy_name)
    is_importance = isinstance(policy, ImportancePolicy)
 
    storage = SimulatedStorage(
        policy=policy,
        gpu_capacity=gpu_capacity,
        cpu_capacity=cpu_capacity,
        cpu_latency=cpu_latency,
        disk_latency=disk_latency,
        verbose=verbose,
    )
 
    num_layers = model.config.n_layer          # 12 for gpt2
 
    generated_tokens: list[int] = []
    prev_positions = 0
    wall_start = time.perf_counter()
 
    print(f"\n{'=' * 70}")
    print(f"  Policy: {policy_name}")
    print(f"{'=' * 70}")
    print(f"  Prompt : {prompt!r}")
    print(f"  Tokens to generate : {max_new_tokens}")
    print(f"  GPU cap={gpu_capacity}  CPU cap={cpu_capacity}")
    print(f"  CPU latency={cpu_latency}s  Disk latency={disk_latency}s")
    print(f"  Layers={num_layers}  -> {num_layers} new blocks per step\n")
 
    for step, token_id, past_key_values in generate_tokens(
        model, tokenizer, prompt, max_new_tokens, device
    ):
        # Advance the importance policy clock via the proper method.
        # For other policies, fall back to the old attribute assignment.
        if is_importance:
            policy.tick(step)
        elif hasattr(policy, "step"):
            policy.step = step
 
        generated_tokens.append(token_id)
 
        total_positions = past_key_values[0][0].shape[2]
 
        # -- Store NEW blocks --
        for pos in range(prev_positions, total_positions):
            for layer_idx in range(num_layers):
                bid = _block_id(layer_idx, pos)
                k = past_key_values[layer_idx][0][:, :, pos:pos+1, :].cpu()
                v = past_key_values[layer_idx][1][:, :, pos:pos+1, :].cpu()
                storage.store(bid, (k, v))
 
        old_positions = prev_positions
        prev_positions = total_positions
 
        # Snapshot counters for per-step deltas
        gpu_before  = storage.gpu_hits
        cpu_before  = storage.cpu_hits
        disk_before = storage.disk_hits
        lat_before  = storage.total_latency
 
        # -- Read back OLD blocks --
        for layer_idx in range(num_layers):
            for pos in range(old_positions):
                bid = _block_id(layer_idx, pos)
                storage.get_data(bid)
 
        step_gpu  = storage.gpu_hits  - gpu_before
        step_cpu  = storage.cpu_hits  - cpu_before
        step_disk = storage.disk_hits - disk_before
        step_lat  = storage.total_latency - lat_before
 
        if step < 30 or step % 10 == 0 or step == max_new_tokens - 1:
            elapsed = time.perf_counter() - wall_start
            tok_text = tokenizer.decode([token_id])
            total_blocks = total_positions * num_layers
            print(
                f"  step {step:>3d}  tok={tok_text!r:>12s}  "
                f"blocks={total_blocks:>4d}  "
                f"gpu={step_gpu:>4d}  cpu={step_cpu:>4d}  disk={step_disk:>4d}  "
                f"gpu_occ={len(storage.gpu):>4d}  cpu_occ={len(storage.cpu):>4d}  "
                f"disk_occ={len(storage.disk):>4d}  "
                f"step_sim={step_lat:.4f}s  total_sim={storage.total_latency:.4f}s  "
                f"wall={elapsed:.2f}s"
            )
 
    wall_total = time.perf_counter() - wall_start
 
    # Decoded text
    full_ids = tokenizer.encode(prompt) + generated_tokens
    decoded  = tokenizer.decode(full_ids)
    print(f"\n  -- Generated text {'-' * 40}")
    print(f"  {decoded}")
    print(f"  {'-' * 60}\n")
 
    # Report
    stats = storage.report()
    stats["policy"]      = policy_name
    stats["wall_time_s"] = round(wall_total, 4)
    print(f"  [{policy_name}] Wall-clock time: {wall_total:.2f}s")
    return stats
 
 
# -- Run all policies --
 
def run(
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 100,
    gpu_capacity: int = 200,
    cpu_capacity: int = 400,
    cpu_latency: float = 0.005,
    disk_latency: float = 0.02,
    verbose: bool = False,
) -> list[dict]:
    """Run the simulation with all policies and return a list of stats."""
 
    model, tokenizer, device = load_model()
 
    all_stats: list[dict] = []
    for name in POLICIES:
        stats = run_one(
            policy_name=name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            gpu_capacity=gpu_capacity,
            cpu_capacity=cpu_capacity,
            cpu_latency=cpu_latency,
            disk_latency=disk_latency,
            verbose=verbose,
        )
        all_stats.append(stats)
 
    # -- Comparison summary --
    print(f"\n{'=' * 80}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"  {'Policy':<12s}  {'GPU hits':>10s}  {'CPU hits':>10s}  "
        f"{'Disk hits':>10s}  {'Sim latency':>12s}  {'Wall time':>10s}"
    )
    print(
        f"  {'-' * 12}  {'-' * 10}  {'-' * 10}  "
        f"{'-' * 10}  {'-' * 12}  {'-' * 10}"
    )
    for s in all_stats:
        print(
            f"  {s['policy']:<12s}  {s['gpu_hits']:>10d}  {s['cpu_hits']:>10d}  "
            f"{s['disk_hits']:>10d}  {s['simulated_latency_s']:>11.4f}s  "
            f"{s['wall_time_s']:>9.2f}s"
        )
    print(f"{'=' * 80}\n")
 
    return all_stats
 
 
# -- CLI --
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(
        description="Milestone 3 - KV-cache simulation with importance-scored eviction"
    )
    parser.add_argument("--prompt",    default=DEFAULT_PROMPT)
    parser.add_argument("--tokens",    type=int,   default=100)
    parser.add_argument("--gpu-cap",   type=int,   default=200)
    parser.add_argument("--cpu-cap",   type=int,   default=400)
    parser.add_argument("--cpu-lat",   type=float, default=0.005,
                        help="Simulated CPU-fetch latency in seconds")
    parser.add_argument("--disk-lat",  type=float, default=0.02,
                        help="Simulated Disk-fetch latency in seconds")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()
 
    run(
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        gpu_capacity=args.gpu_cap,
        cpu_capacity=args.cpu_cap,
        cpu_latency=args.cpu_lat,
        disk_latency=args.disk_lat,
        verbose=args.verbose,
    )
 