# Milestone 2 — KV-Cache Simulation Infrastructure

**Project:** kv_cache_rl  
**Date:** March 2026  
**Environment:** Python 3.9, conda env `gpt2_env`, Apple Silicon (MPS)  
**Model:** GPT-2 Small (124,439,808 parameters)

---

## 1. Objective

Build a complete, modular KV-cache simulation infrastructure that:

1. Runs GPT-2 token-by-token generation, exposing the full `past_key_values` cache.
2. Routes every KV-cache block through a **three-tier storage hierarchy** (GPU → CPU → Disk) with configurable capacities and simulated latencies.
3. Supports **pluggable eviction policies** so that a future RL agent can be dropped in as a replacement.
4. Provides **four baseline eviction policies** (LRU, LFU, FIFO, Random) with head-to-head comparison output.

This milestone establishes the environment and baselines that the Milestone 3 RL agent must beat.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    main.py (entry-point)             │
│  POLICIES registry → run_one() per policy → run()    │
│  Comparison summary table at the end                 │
└────────────┬─────────────────────────┬───────────────┘
             │                         │
     ┌───────▼───────---┐       ┌────────▼────────-┐
     │   model.py       │       │   storage.py     │
     │  GPT-2 wrapper.  │       │  SimulatedStorage│
     │  token-by-       │       │  3-tier block    │
     │  token generation│       │  store + stats   │
     └───────────────---┘       └───────┬──────────┘
                                     │ accepts any
                                     │ EvictionPolicy
                   ┌─────────────────┼───────────────────┐
                   │                 │                   │
           ┌───────▼──┐   ┌────────▼────┐   ┌──────────▼──────┐
           │  lru.py  │   │  lfu.py     │   │ random_policy.py│
           │  fifo.py │   │             │   │                 │
           └──────────┘   └─────────────┘   └─────────────────┘
```

### 2.1 File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 4 | Package init; single source of truth for `DEFAULT_PROMPT` |
| `model.py` | 108 | GPT-2 wrapper: `load_model()`, `generate_tokens()` generator |
| `storage.py` | 210 | `SimulatedStorage`: 3-tier block store with injected eviction policy |
| `lru.py` | 51 | LRU eviction (OrderedDict, move_to_end) |
| `lfu.py` | 82 | LFU eviction (frequency buckets + LRU tie-break) |
| `fifo.py` | 51 | FIFO eviction (insertion-order, access never reorders) |
| `random_policy.py` | 65 | Random eviction (O(1) swap-and-pop, seeded RNG) |
| `main.py` | 247 | Entry-point: runs all 4 policies, prints comparison |
| `test_quick.py` | 88 | Quick integration test (LRU, tiny caps) |
| `test_tiers.py` | 197 | Unit test + model integration test for all 3 tiers |
| `requirements.txt` | 2 | `transformers>=4.38`, `torch>=2.1` |

---

## 3. Block Scheme

Each KV-cache "block" represents **one layer's (key, value) pair for one token position**.

```
block_id = "L{layer}_P{position}"
```

- GPT-2 Small has **12 layers**, so each new token creates **12 new blocks**.
- A 7-token prompt ("The history of the Roman Empire is") seeds **84 blocks** on step 0.
- After 100 generated tokens, the cache holds **1,272 blocks** total (106 positions × 12 layers).

### 3.1 Per-Block Shape

```
key  : torch.Size([1, 12, 1, 64])   # (batch, heads, 1_position, head_dim)
value: torch.Size([1, 12, 1, 64])
```

Each block is extracted from the model's `past_key_values` via:
```python
k = past_key_values[layer][0][:, :, pos:pos+1, :].cpu()
v = past_key_values[layer][1][:, :, pos:pos+1, :].cpu()
```

---

## 4. Three-Tier Storage Hierarchy

### 4.1 Design

| Tier | Default Capacity | Simulated Latency | Real Analogue |
|------|-----------------|-------------------|---------------|
| **GPU** | 200 blocks | 0 s (instant) | GPU HBM / VRAM |
| **CPU** | 400 blocks | 0.005 s per access | System DRAM |
| **Disk** | ∞ | 0.02 s per access | NVMe SSD / swap |

### 4.2 Parameter Reasoning

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `gpu_capacity = 200` | ~16.7 token positions worth | Forces eviction pressure by step 10 (7 prompt + ~10 generated tokens = 204 blocks > 200). Small enough to expose policy differences, large enough to not be trivially full. |
| `cpu_capacity = 400` | 2× GPU | Realistic hierarchy ratio. CPU fills by step ~50, after which Disk spill begins. |
| `cpu_latency = 0.005s` | 5 ms simulated | Order-of-magnitude approximation of PCIe DMA transfer latency (real: 1–10 ms depending on payload). |
| `disk_latency = 0.02s` | 20 ms simulated | 4× CPU latency. Approximates SSD random-read latency (real: 10–100 µs for NVMe, but we inflate to make the penalty visible in totals). |

### 4.3 Eviction Cascade

When the GPU tier is full and a new block must be stored:

1. **GPU → CPU**: The eviction policy picks a victim from GPU; it is demoted to CPU.
2. **CPU → Disk**: If CPU is also full, the oldest CPU-resident block (always LRU among CPU blocks, regardless of GPU policy) spills to Disk.
3. **Disk → ∞**: Disk never evicts; it absorbs all overflow.

When a block is read from a lower tier, it is **promoted back to GPU** (with an eviction cascade if GPU is full). This models real systems where hot data migrates up.

### 4.4 Virtual Latency (No Real Sleeping)

All `time.sleep()` calls have been removed. Latency is tracked as a **virtual accumulator** (`storage.total_latency`), so the simulation runs at full speed (~1.5 s wall-clock for 100 tokens × 4 policies) while still producing meaningful simulated timing data.

---

## 5. Eviction Policies

All four policies implement the same `EvictionPolicy` protocol:

```python
class EvictionPolicy(Protocol):
    def access(self, block_id: str) -> None: ...
    def evict(self) -> str: ...
    def remove(self, block_id: str) -> None: ...
```

### 5.1 LRU — Least Recently Used (`lru.py`)

**Data Structure:** `OrderedDict[str, None]`

| Operation | Complexity | Implementation |
|-----------|-----------|----------------|
| `access()` | O(1) | `move_to_end(block_id)` |
| `evict()` | O(1) | `popitem(last=False)` — pops the least-recently-used |
| `remove()` | O(1) | `pop(block_id, None)` |

**Behaviour:** Every access (read or store) moves the block to the MRU end. Eviction always picks the block that hasn't been touched the longest.

**Known weakness:** Suffers from **thrashing** when the working set exceeds GPU capacity. Each promoted block evicts the next block that will be needed, causing a "domino effect" cascade of CPU/Disk reads.

### 5.2 LFU — Least Frequently Used (`lfu.py`)

**Data Structures:**
- `_freq: dict[str, int]` — block → access count
- `_freq_to_blocks: defaultdict[int, OrderedDict[str, None]]` — frequency bucket → insertion-ordered set
- `_min_freq: int` — tracks the current minimum frequency for O(1) eviction

| Operation | Complexity | Implementation |
|-----------|-----------|----------------|
| `access()` | O(1) | Increment frequency, move block to new bucket |
| `evict()` | O(1) | Pop oldest from `_freq_to_blocks[_min_freq]` |
| `remove()` | O(1) | Remove from freq dict + bucket, recalculate `_min_freq` |

**Behaviour:** Evicts the block with the fewest total accesses. Ties are broken by LRU within the same frequency bucket. This is the classic O(1) LFU algorithm.

**Advantage:** Retains blocks that are accessed repeatedly (e.g., early prompt positions touched every step) even if they weren't accessed most recently.

### 5.3 FIFO — First In, First Out (`fifo.py`)

**Data Structure:** `OrderedDict[str, None]`

| Operation | Complexity | Implementation |
|-----------|-----------|----------------|
| `access()` | O(1) | Insert if new; **never** `move_to_end()` |
| `evict()` | O(1) | `popitem(last=False)` — pops the oldest-inserted |
| `remove()` | O(1) | `pop(block_id, None)` |

**Behaviour:** Evicts the block that was inserted (or promoted) earliest, completely ignoring subsequent accesses. The critical difference from LRU is that `access()` does NOT reorder—this is what makes it FIFO.

### 5.4 Random (`random_policy.py`)

**Data Structures:**
- `_blocks: list[str]` — flat list of tracked block IDs
- `_index: dict[str, int]` — block → index in `_blocks`
- `_rng: random.Random(seed=42)` — seeded for reproducibility

| Operation | Complexity | Implementation |
|-----------|-----------|----------------|
| `access()` | O(1) | Append if new |
| `evict()` | O(1) | `randrange(len)` → swap-and-pop from list |
| `remove()` | O(1) | Swap-and-pop |

**Behaviour:** Picks a uniformly random block to evict. Seeded at 42 for deterministic, reproducible runs. The swap-and-pop technique ensures O(1) removal without shifting list elements.

---

## 6. Simulation Results (100 Generated Tokens)

### 6.1 Configuration

```
Model          : GPT-2 Small (124M params, 12 layers, 12 heads, 64 head_dim)
Device         : MPS (Apple Silicon)
Prompt         : "The history of the Roman Empire is" (7 tokens)
Tokens to gen  : 100
GPU capacity   : 200 blocks
CPU capacity   : 400 blocks
CPU latency    : 0.005 s (simulated)
Disk latency   : 0.02 s (simulated)
Total accesses : 66,528 per policy
Final blocks   : 1,272 (106 positions × 12 layers)
```

### 6.2 Comparison Summary

| Policy | GPU Hits | CPU Hits | Disk Hits | Sim Latency (s) | Wall Time (s) |
|--------|----------|----------|-----------|-----------------|---------------|
| **LRU** | 1,323 | 13,174 | 52,031 | 1,106.49 | 1.68 |
| **LFU** | 17,388 | 7,174 | 41,966 | 875.19 | 1.46 |
| **FIFO** | 2,803 | 11,694 | 52,031 | 1,099.09 | 1.46 |
| **Random** | 19,091 | 6,533 | 40,904 | 850.75 | 1.46 |

### 6.3 Hit Rate Analysis

| Policy | GPU Hit % | CPU Hit % | Disk Hit % |
|--------|-----------|-----------|------------|
| LRU | 1.99% | 19.80% | 78.21% |
| LFU | 26.14% | 10.78% | 63.08% |
| FIFO | 4.21% | 17.58% | 78.21% |
| Random | 28.69% | 9.82% | 61.49% |

### 6.4 Relative Latency (lower is better)

```
Random   ████████████████████████████████████████ 850.75 s  (best)
LFU      ██████████████████████████████████████████ 875.19 s  (+2.9%)
FIFO     ████████████████████████████████████████████████████ 1,099.09 s  (+29.2%)
LRU      █████████████████████████████████████████████████████ 1,106.49 s  (+30.1%, worst)
```

---

## 7. Analysis

### 7.1 LRU Thrashing (The Domino Effect)

LRU performs worst because of a classic **thrashing cascade**:

1. **Steps 0–9** (blocks ≤ 192, GPU cap = 200): Everything fits in GPU. 100% GPU hit rate.
2. **Step 10** (204 blocks > 200 cap): GPU is full. The 12 new blocks each evict the least-recently-used block. But the evicted blocks are the very next blocks the read-back loop will need.
3. **Step 11 onward**: Every read-back promotes a block from CPU→GPU, which evicts another block that is about to be needed. This creates a **domino cascade** where GPU hits drop to **zero** by step 11.

```
step  10: gpu=135, cpu= 57   ← first eviction pressure, partial thrashing
step  11: gpu=  0, cpu=204   ← complete thrashing begins
step  19: gpu=  0, cpu=300   ← all reads from CPU
step  50: gpu=  0, disk=672  ← CPU overflows too, all reads from Disk
```

Once the working set exceeds GPU capacity, LRU's "promote on every access" strategy guarantees that the most recently promoted block is the last one the scan will revisit—exactly the wrong priority.

### 7.2 Why Random Wins (Counterintuitively)

Random eviction outperforms LRU, FIFO, and even LFU because:

- It **avoids the systematic worst-case** patterns that LRU/FIFO create. When the working set is larger than the cache, any deterministic scan-order policy tends to evict exactly the blocks that will be needed soonest.
- With random eviction, there is always a **~200/total_blocks probability** that the needed block is still in GPU. This probability is low but nonzero and consistent, yielding ~199 GPU hits per step even when total blocks far exceed capacity.
- This is the well-known result that **random replacement performs comparably to optimal** for certain access patterns (sequential scans through data larger than cache).

### 7.3 LFU's Strength

LFU retains the most-frequently-accessed blocks (typically the early prompt positions, which are read on every step). Once those blocks accumulate high frequency counts, they become "sticky" in GPU and resist eviction. This gives LFU a 26% GPU hit rate vs LRU's 2%.

### 7.4 FIFO ≈ LRU

FIFO performs nearly identically to LRU (4.21% vs 1.99% GPU hits) because both policies evict the oldest/least-recently-used blocks—the only difference is that LRU counts re-access and FIFO doesn't. When every block is accessed in a sequential scan, the distinction is minimal.

### 7.5 All Policies Hit Disk

After step ~50, **all four policies** are reading from Disk because:
- Total blocks (684 at step 50) exceed GPU + CPU capacity (200 + 400 = 600).
- Disk spill is unavoidable once the sequence is long enough.
- The key differentiator is **how many steps it takes before GPU becomes useless** (LRU: step 11, Random: never fully useless).

---

## 8. Generated Text

All four policies produce identical text (greedy decoding, deterministic):

> The history of the Roman Empire is a fascinating one. The Roman Empire was founded by the Romans, and the Roman Empire was founded by the Romans. The Romans were the first to establish a state of war, and the Romans were the first to establish a state of peace. The Romans were the first to establish a state of war, and the Romans were the first to establish a state of peace. The Romans were the first to establish a state of war, and the Romans were the first to establish a state of peace. The Romans…

The repetitive output is expected from GPT-2 Small with greedy decoding (no sampling, no temperature). Text quality is not the target of this milestone—correctness of the cache simulation is.

---

## 9. How to Run

```bash
# Activate environment
conda activate gpt2_env

# Default run: 100 tokens, all 4 policies
python -m kv_cache_rl.main

# Custom run
python -m kv_cache_rl.main --tokens 50 --gpu-cap 100 --cpu-cap 200 --verbose

# CLI options
python -m kv_cache_rl.main --help
```

### Available CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | `DEFAULT_PROMPT` | Input prompt text |
| `--tokens` | 100 | Number of tokens to generate |
| `--gpu-cap` | 200 | GPU tier capacity (blocks) |
| `--cpu-cap` | 400 | CPU tier capacity (blocks) |
| `--cpu-lat` | 0.005 | Simulated CPU-fetch latency (seconds) |
| `--disk-lat` | 0.02 | Simulated Disk-fetch latency (seconds) |
| `--verbose` | off | Print every eviction/promotion event |

### Changing the Prompt

Edit the single line in `__init__.py`:

```python
DEFAULT_PROMPT = "The history of the Roman Empire is"
```

This propagates to `main.py`, `model.py`, `test_quick.py`, and `test_tiers.py` automatically.
