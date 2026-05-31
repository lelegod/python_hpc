# Week 6 — Parallelism Part 2: Reductions & Shared Memory

## Overview

Week 6 covers two major topics: parallel reductions (combining N elements into 1 using associative operators) and how hardware architecture — specifically NUMA (Non-Uniform Memory Access) — interferes with the expected scaling behavior. The practical exercise builds a parallel mean-face computation over 200,000 celebrity images using `mp.RawArray` shared memory and a binary tree reduction strategy.

---

## Theory & Concepts

### Reduction Operations

A reduction collapses an array of N elements into a single value using a binary operator applied repeatedly:

- **sum**: `sum([4,9,1,2,...])` — the canonical example
- **product**: multiply all elements
- **min / max**: keep running extreme value
- **logical AND / OR**: fold boolean values
- **bitwise XOR**: commutative and associative — valid
- **matrix multiplication**: NOT a valid reduction operator — it is not commutative (`AB != BA`), so the order of operands cannot be freely reordered across parallel chunks

A reduction operator must satisfy two algebraic properties:

1. **Associative**: `(a O b) O c = a O (b O c)` — required to split into subproblems
2. **Commutative**: `a O b = b O a` — required in practice because parallel implementations may reorder inputs during processing

The lecture explicitly shows XOR (`^`) as a counterexample: sequential `1^4^8^3^2^4` gives a different result when the grouping is changed, demonstrating that non-associative operators cannot be parallelized this way.

---

### Parallel Reduction Algorithms

#### Simple (Flat) Reduction

Split the array into T chunks, reduce each chunk in parallel (N/T operations per chunk), then sum the T partial results serially (T operations):

```
sum(x) = sum(x[:N/T]) + sum(x[N/T:2*N/T]) + sum(x[2*N/T:3*N/T]) + ...
```

- Total time: N/T + T
- Speed-up: N / (N/T + T) = 1 / (1/T + T/N)
- The speed-up peaks when the two terms are equal: T = N/T → T^2 = N → **T_opt = sqrt(N)**
- Maximum speed-up (simple method): **sqrt(N) / 2**

For N=100, the theoretical max is ~5x at T≈10 tasks, after which the serial final summation (T operations) grows faster than the parallel savings.

#### Binary Tree Reduction

Each level halves the number of active elements. Each task does exactly 1 operation (add neighbor into current position). The array is updated in-place.

Level pattern (step s doubling each round):
```
# Step s=1: pair adjacent elements
for i in range(0, N, 2*s):
    x[i] += x[i+s]

# Step s=2: pair elements 2 apart
for i in range(0, N, 2*s):
    x[i] += x[i+s]

# ... continue for ceil(log2(N)) rounds
```

- Number of levels: **log2(N)**
- Total operations: N (same work as serial)
- Critical path depth: **log2(N)** — the theoretical span
- Speed-up: **N / log2(N)** — dramatically better than sqrt(N)/2 for large N

Comparing the two: for N=100,000 the simple method gives ~158x theoretical max, the binary tree gives ~5,900x. The binary tree is clearly superior.

---

### Shared Memory Parallelism

To avoid copying data between processes (which would dominate the runtime), all processes share a single `mp.RawArray`. The array is backed by OS shared memory — all worker processes see the same physical bytes.

Key pattern:
- Main process allocates `mp.RawArray(ctypes.c_float, data.size)` and fills it
- Worker processes receive the array handle via the pool initializer (`init` function sets a module-level global `shared_arr`)
- Workers reinterpret the raw bytes as a numpy array via `np.frombuffer`, reshape, and operate in-place
- No data serialization happens across process boundaries — just index arithmetic

Race conditions are avoided by design: each reduction step assigns non-overlapping write targets (only even-indexed positions at stride 2s are written), so no two workers write to the same element in the same step.

---

### NUMA Architecture

**NUMA = Non-Uniform Memory Access**

Modern multi-socket servers (like the DTU cluster's Xeon Gold 6226R nodes) have 2 or more physical CPU sockets, each with its own bank of local RAM. Accessing local RAM is fast; accessing the other socket's RAM requires crossing an inter-socket link and is significantly slower.

**Without numactl:** When Python allocates memory, all pages land on socket 0's RAM. Cores on socket 0 get fast local access. Cores on socket 1 must cross the link for every memory access. Result: speedup improves up to ~50% of total cores (those on socket 0), then plateaus or drops as socket 1 cores saturate the inter-socket link.

**With `numactl --interleave=all`:** Memory pages are distributed round-robin across all NUMA nodes. All cores get approximately equal (average) memory access latency. Result: speedup continues to grow with every additional core.

---

## Mathematical Content

### Simple Reduction Speed-up

```
Speed-up = N / (N/T + T) = 1 / (1/T + T/N)

Optimal T: dSpeedup/dT = 0
  N/T = T  →  T^2 = N  →  T_opt = sqrt(N)

Max speed-up = 1 / (1/sqrt(N) + sqrt(N)/N)
             = 1 / (2/sqrt(N))
             = sqrt(N) / 2
```

For N=100: max speed-up ≈ 5x at T=10 tasks.

### Binary Tree Reduction Speed-up

```
Number of levels = ceil(log2(N))
Each level: N/2 parallel additions (halved each round)
Total work: N operations (same as serial)
Critical path: log2(N) sequential steps

Speed-up (ideal, unlimited processors) = N / log2(N)
```

For N=100,000: ~6,000x theoretical; simple method gives only ~158x.

### When Reduction Overhead Exceeds Benefit

The simple reduction crossover point: when T > sqrt(N), adding more tasks makes the serial final summation the bottleneck. This is why the speed-up curve peaks before reaching the processor count and then falls.

---

## Key Code Examples

### reduction1.py — Single Binary Tree Step

```python
import ctypes
import multiprocessing as mp
import sys
from time import perf_counter as time
import numpy as np
from PIL import Image


def init(shared_arr_):
    global shared_arr
    shared_arr = shared_arr_


def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr, dtype='float32')


def reduce_step(args):
    b, e, s, elemshape = args
    arr = tonumpyarray(shared_arr).reshape((-1,) + elemshape)
    # One step of binary tree reduction: add neighbor at offset +1
    if b + 1 < len(arr):
        arr[b] += arr[b + 1]


if __name__ == '__main__':
    n_processes = 1
    chunk = 2

    data = np.load(sys.argv[1])
    elemshape = data.shape[1:]
    shared_arr = mp.RawArray(ctypes.c_float, data.size)
    arr = tonumpyarray(shared_arr).reshape(data.shape)
    np.copyto(arr, data)
    del data

    t = time()
    pool = mp.Pool(n_processes, initializer=init, initargs=(shared_arr,))

    # Submit one reduction step: each task handles one pair (stride=chunk=2)
    pool.map(reduce_step,
             [(i, i + chunk, 1, elemshape) for i in range(0, len(arr), chunk)],
             chunksize=1)

    print(time() - t)
    final_image = arr[0]
    # final_image /= len(arr)  # For mean
    Image.fromarray(
        (255 * final_image.astype(float)).astype('uint8')
    ).save('result.png')
```

Key point: `reduce_step` receives `(b, e, s, elemshape)`. The index `b` is the write target; `b+1` is the neighbor to add. The result accumulates in `arr[b]` in-place in shared memory.

### reduction_full.py — Complete Binary Tree Reduction

```python
def reduce_step(args):
    b, e, s, elemshape = args
    arr = tonumpyarray(shared_arr).reshape((-1,) + elemshape)
    # Add neighbor at stride s into position b
    if b + s < len(arr):
        arr[b] += arr[b + s]


if __name__ == '__main__':
    # ...setup same as above...

    t = time()
    pool = mp.Pool(n_processes, initializer=init, initargs=(shared_arr,))

    # Binary tree: ceil(log2(N)) rounds, stride doubles each round
    for j in range(int(np.ceil(np.log2(len(arr))))):
        s = 2**j
        pool.map(reduce_step,
                [(i, 0, s, elemshape) for i in range(0, len(arr), 2*s)],
                chunksize=1)

    print(time() - t)
    final_image = arr[0]
    final_image /= len(arr)  # Divide by N for mean
    Image.fromarray(
        (255 * final_image.astype(float)).astype('uint8')
    ).save('result.png')
```

The outer loop runs `ceil(log2(N))` times. In round `j`, stride `s = 2^j`. Tasks are generated for indices `i = 0, 2s, 4s, ...` — each task adds `arr[i + s]` into `arr[i]`. After all rounds, `arr[0]` holds the total sum.

---

## numactl Usage

From `numactl_notes.md`:

```bash
# Run with memory interleaved across all NUMA nodes
numactl --interleave=all python my_script.py
```

**What it does:** Spreads memory allocation across all NUMA nodes in round-robin. Does NOT control which cores run the code — only where memory pages are allocated.

**Why it matters on the DTU cluster (XeonGold6226R):**

| Scenario | Socket 0 cores | Socket 1 cores | Observed behavior |
|---|---|---|---|
| Without numactl | Fast (local RAM) | Slow (remote RAM) | Speedup plateaus at ~50% of cores |
| With numactl --interleave=all | Average latency | Average latency | Speedup scales with all cores |

**Quiz answers from numactl_notes.md:**
1. Before numactl: speedup improves until 50% of threads are added, then decreases
2. After numactl: speedup now increases with every core added
3. Effect: ensures memory allocation is spread to both CPU sockets

---

## Quiz/Exercise Highlights

The in-lecture Vevox quiz (from Quiz_week06.pdf) tests:

**Q1: What is the issue with the simple reduction?**
- Options: "Does not scale to many processors" / "Does not scale with small data sizes" / "Only scales with many processors"
- Correct answer (86% of class): **Does not scale to many processors** — the serial final summation of T partial results becomes the bottleneck as T grows, giving max speed-up of only sqrt(N)/2.

**Q2: Theoretical max speed-up for the simple method given array of length N?**
- Options: N / 1/N / sqrt(N)/2
- Correct answer (98%): **sqrt(N)/2** — derived from setting T = sqrt(N) as the optimal number of tasks.

**Q3: How many steps does the binary tree reduction need for an array of length N?**
- Options: log2(N) / N / sqrt(N)
- Correct answer (99%): **log2(N)** — one level of the binary tree per step, halving active elements each time.

**Q4: Theoretical max speed-up for the binary tree reduction given array of length N?**
- Options: log2(N) / log2(N)/N / N/log2(N)
- Correct answer (94%): **N/log2(N)** — N total operations completed in log2(N) parallel steps.

**Q5: Can multiply be a reduction operator?**
- Answer: **Yes** — multiplication is both commutative (a*b = b*a) and associative ((a*b)*c = a*(b*c)), so it satisfies both required properties.

---

## Key Takeaways

1. **Reductions are not embarrassingly parallel.** Unlike element-wise operations (multiply array by 2), reductions have data dependencies between elements that require a structured parallel strategy.

2. **Binary tree is far better than flat partitioning.** Simple reduction caps at sqrt(N)/2 speed-up; binary tree achieves N/log2(N). For 100K images the difference is 158x vs 5,900x theoretical.

3. **The reduction operator must be commutative AND associative.** Associativity lets you split into subproblems; commutativity covers implementation details where inputs may be reordered. Matrix multiplication fails commutativity — it cannot be parallelized this way.

4. **Shared memory (`mp.RawArray`) eliminates serialization overhead.** All worker processes operate directly on the same memory-mapped buffer. Without this, pickling/unpickling 200K images across process pipes would dominate the runtime.

5. **NUMA is the hidden wall.** On a dual-socket machine without numactl, you get a speedup cliff at ~50% of cores. Interleaving memory with `numactl --interleave=all` resolves this and allows full scaling across all available cores. This is a hardware effect invisible in the code.

6. **Chunk size tuning matters.** Empirically, chunk=64 was found to give good performance for the 100K image dataset. Too small creates excessive process scheduling overhead; too large underutilizes the pool.

7. **Even the optimized parallel code only beats `np.sum` by 2-4x.** NumPy's built-in operations are highly vectorized and cache-optimized. The parallel reduction wins through multi-core parallelism but carries Python overhead per task.
