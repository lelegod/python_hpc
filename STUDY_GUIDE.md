# 02613 Python HPC — Master Study Guide

> **Root files:** [STUDY_GUIDE](STUDY_GUIDE.md) · [Exam Review](exam_review.md) · [Cheat Sheet](master_cheat_sheet.md) · [Tips & Pitfalls](tips_and_tricks.md) · [README](README.md)

One file to rule them all. Read top-to-bottom before the exam. Every major topic, every formula, every trap.

---

## Contents

| # | Topic |
|---|---|
| 1 | [LSF / BSUB](#topic-1--lsf--bsub-hpc-job-scheduling) |
| 2 | [Profiling — cProfile, kernprof, `time`](#topic-2--profiling-cprofile-kernprof-time) |
| 3 | [Cache & Memory Layout](#topic-3--cache--memory-layout) |
| 4 | [NumPy: Broadcasting & Reshaping](#topic-4--numpy-broadcasting--reshaping) |
| 5 | [Amdahl's Law](#topic-5--amdahls-law-every-exam-has-multiple-questions) |
| 6 | [GIL, Threads vs Processes](#topic-6--gil-threads-vs-processes) |
| 7 | [Parallel Reduction](#topic-7--parallel-reduction) |
| 8 | [Pandas & Memory Optimization](#topic-8--pandas--memory-optimization) |
| 9 | [Memory-Mapped Files & Zarr](#topic-9--memory-mapped-files--zarr) |
| 10 | [Numba CPU JIT](#topic-10--numba-cpu-jit) |
| 11 | [CUDA / GPU Computing](#topic-11--cuda--gpu-computing) |
| 12 | [CuPy & GPU Shared Memory](#topic-12--cupy--gpu-shared-memory-week-10) |
| 13 | [HPC Workflows — Job Arrays & Dependencies](#topic-13--hpc-workflows-job-arrays--dependencies) |
| 14 | [HPC Pitfalls](#topic-14--hpc-pitfalls-week-13) |
| — | [Exam Quick Decision Guide](#exam-quick-decision-guide) |
| — | [Formulas Cheat Sheet](#formulas-cheat-sheet) |
| — | [All Past Exam Answers](#all-past-exam-answers-quick-reference) |

---

## TOPIC 1 — LSF / BSUB (HPC Job Scheduling)

### Core Rules

| Directive | Meaning |
|---|---|
| `#BSUB -W HH:MM` | Wall-clock time limit |
| `#BSUB -n N` | Request N CPU cores/slots |
| `#BSUB -R "rusage[mem=XGB]"` | Memory **per core** (not total!) |
| `#BSUB -R "span[hosts=1]"` | All cores on one machine (required for shared-memory programs) |
| `#BSUB -J name[1-N]` | Job array: N jobs with indices 1..N |
| `#BSUB -w "ended(name)"` | Dependency: wait until all jobs with that name have *finished* |
| `#BSUB -w "done(name)"` | Dependency: wait until all jobs finished *successfully* |
| `#BSUB -o file_%J.out` | `%J` = job ID, `%I` = array index |
| `#BSUB -gpu "num=1:mode=exclusive_process"` | Request 1 GPU (also needs `-q gpuv100` or `gpua100`) |
| `$LSB_JOBINDEX` | Array index inside the job script |

### The One Calculation You Must Know

```
rusage[mem] = Total_memory / n_cores
```

**Example:** 16 GB total, 8 cores → `rusage[mem=2GB]`  
**Trap:** Writing `16GB` instead of `2GB` wastes 4× cluster resources.

### BSUB Traps

1. **rusage is per-core** — always divide total by n.
2. **`ended` vs `done`**: "finished regardless of outcome" = `ended`. "finished successfully" = `done`. If any job may fail and you still want the post-job to run → `ended`.
3. **Typos matter**: `#BSUG` (not `#BSUB`) silently ignores the directive.
4. **span[hosts=1] is required** whenever you use shared-memory (`multiprocessing`, `RawArray`, etc.) — without it, cores may land on different machines.
5. **GPU queue change**: CPU job uses `-q hpc`. GPU job needs `-q gpuv100` (or `gpua100`) AND `#BSUB -gpu "num=1:mode=exclusive_process"`.

### Job Array Pattern

```bash
#BSUB -J myjob[1-12]
#BSUB -o myjob_%I_%J.out   # %I = array index, %J = job ID

python process.py $LSB_JOBINDEX data/
```

Post-processing job that runs after array finishes:
```bash
#BSUB -w "ended(myjob)"    # NOT done(myjob) if you want to run even if some fail
```

---

## TOPIC 2 — Profiling (cProfile, kernprof, `time`)

### cProfile Output

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
    10    1.266   0.127    5.055    0.505   ourlib.py:13(process_sample)
```

| Column | Meaning |
|---|---|
| `ncalls` | How many times the function was called |
| `tottime` | Time inside this function only (not counting callees) |
| `cumtime` | Total time including all sub-function calls |

- **"What takes the most time overall?"** → Use `cumtime`.
- **"What is this function itself slow at?"** → Use `tottime`.
- **"How many samples/items were processed?"** → Read `ncalls` of the per-item function.

### Extrapolating to Larger Workloads

Separate functions into:
- **Scales with data**: called once per item (ncalls = dataset size) → multiply per-call time × new size
- **Fixed cost**: called once (setup, load, save) → stays constant regardless of data size

**Trap:** If profiling used 10 samples but normal workload is 1000, do NOT multiply total runtime by 100. Multiply only the scaling part.

**Example from 2024 exam:**
- `prep_conds`: 2.0 s (called once — fixed, does not scale)
- `process_sample`: 0.505 s/call × 1000 = 505 s (per sample — scales)
- Total at 1000: 2.0 + 505 = **507 s ≈ 507 s** (not 3.27 × 100 = 327 s)

### kernprof (line_profiler) Output

```
Line #   Hits        Time  Per Hit   % Time  Line Contents
   99       1  2005064.0    2e+06     61.2       conds = prep_conds(params)
  101    1001     740.0      0.7      0.0       for x in data:
  102    1000  1266748.0   1266.7     38.7           y = process_single(x, conds)
```

- Timer unit is shown at top (usually `1e-06 s` = microseconds).
- `Hits` = how many times that line executed. The for-loop header hits once more than the body (the final termination check).
- Dataset size = Hits on the loop body = **1000** (not 1001).

**Extrapolation formula:**
```
loop_time_at_N = (time_per_hit_body × N) + (fixed_lines_time)
```

### `time` Command: real vs user vs sys

```
real    0m6.03s    ← wall clock (how long you waited)
user    0m12.00s   ← CPU seconds spent in user space, SUMMED across all cores
sys     0m0.034s   ← CPU seconds in kernel, summed across all cores
```

With **perfect parallelism on 2 cores**:
- `real` → halved (6 s instead of 12 s)
- `user` → stays the same (2 cores × 6 s = 12 s of CPU time)
- `sys` → stays the same

**Trap:** Students think `user` halves when you double the cores. It does not — `user` is aggregate CPU time. Only `real` changes.

---

## TOPIC 3 — Cache & Memory Layout

### Memory Hierarchy (fastest to slowest)

```
Registers → L1 cache (fast, ~1 ns) → L2 cache → L3 cache → DRAM (~100 ns) → Disk
```

Each level is ~10× slower than the one above. When you get a cache miss, the CPU waits for data from the next level down. Random access over large data = constant misses = constant DRAM access = slow.

### Strides and Loop Ordering

Arrays are stored **row-major (C order)**: last index changes fastest in memory. Stride = number of bytes between adjacent elements along that axis.

**Rule: innermost loop should index the axis with the shortest (smallest) stride.**

```python
# Given strides (600, 40, 8, 200) for axes (0, 1, 2, 3):
# Sort by stride ascending: 8, 40, 200, 600 → axes 2, 1, 3, 0
# Innermost = axis 2 (stride 8), Outermost = axis 0 (stride 600)

for i in ...:         # axis 0, stride 600
    for l in ...:     # axis 3, stride 200
        for j in ...: # axis 1, stride 40
            for k in ...: # axis 2, stride 8 ← innermost, stride 1
                x = arr[i, j, k, l]
```

**Trap:** "Reversing the loop order" is NOT the same as sorting by stride. If strides are (600, 40, 200, 8), reversing gives innermost = stride 600 (bad). Sorting gives innermost = stride 8 (good).

### HWC vs CHW for Image Data

- **CPU (sequential access)**: innermost loop = channel axis → put channels **last**: shape `(H, W, C)`.
- **GPU (warp-level coalesced access)**: adjacent threads (j-varying) must hit adjacent memory → channel last axis must be width W → put channels **first**: shape `(C, H, W)`.

This is the most common CPU vs GPU layout question. The answer flips between CPU and GPU.

### Cache Behavior With Random Access

If a program accesses a large data structure randomly → **a slower part of the memory hierarchy is being used** (DRAM instead of cache). The root cause is not "caches too small" — even infinite caches wouldn't help if there's no locality.

---

## TOPIC 4 — NumPy: Broadcasting & Reshaping

### Broadcasting Rules

NumPy aligns shapes **from the right** and expands size-1 dimensions.

```
images: (N, H, W, 3)
mim:         (H, W)    → need to become (H, W, 1) or (1, H, W, 1)
```

To subtract `mim` from every image and every channel:
```python
images - mim[:, :, None]   # (H, W, 1) → broadcasts to (N, H, W, 3) ✓
```

To subtract per-image means `mean_pixels` of shape `(N, 3)` from images `(N, H, W, 3)`:
```python
images - mean_pixels[:, None, None]   # (N, 1, 1, 3) → broadcasts to (N, H, W, 3) ✓
```

**Strategy:** Identify which axes are "already right" and insert `None` (=`np.newaxis`) for missing middle axes.

### reshape(-1) Flattening

`a.reshape(-1)` flattens to 1D in **row-major order** (read row 0 left-to-right, then row 1, etc.).

```
Array:   [[1, 5, 43, 51, 32],
          [73, 2,  4, 67, 37],
          [ 9, 3, 54,  8, 22]]

Flat:    [1, 5, 43, 51, 32, 73, 2, 4, 67, 37, 9, 3, 54, 8, 22]
Index:    0  1   2   3   4   5  6  7   8   9 10 11  12 13  14
```

`a.reshape(-1)[8]` = **67** (row 1, col 3).

### dtype Sizes

| dtype | bytes | range |
|---|---|---|
| int8 | 1 | -128 to 127 |
| uint8 | 1 | 0 to 255 |
| int16 | 2 | -32,768 to 32,767 |
| uint16 | 2 | 0 to 65,535 |
| int32 | 4 | ±2.1 billion |
| int64 | 8 | ±9.2×10¹⁸ |
| float16 | 2 | resolution ≈ 0.001 relative |
| float32 | 4 | |
| float64 | 8 | |
| datetime64 | 8 | compact date/time |

### float16 Precision Trap

float16 has **relative** resolution of ~0.001. Absolute precision at magnitude M ≈ M × 0.001.

At M = 10,000: gap between representable values ≈ 10. So `10000 + 1` rounds back to `10000` — the addition is a no-op.

**General rule:** At magnitude M, values closer than M × 0.001 are indistinguishable in float16.

---

## TOPIC 5 — Amdahl's Law (Every Exam Has Multiple Questions)

### Core Formulas

```
S(p) = 1 / ((1 - F) + F/p)     # speedup with p processors
S_max = 1 / (1 - F)             # theoretical max (p → ∞)
```

Where:
- `F` = parallel fraction (0 to 1)
- `1 - F` = serial fraction
- `p` = number of processors

### Solving for F from measured speedup S at p processors

```
1/S = (1 - F) + F/p
F = (1 - 1/S) / (1 - 1/p)
  = p(1 - 1/S) / (p - 1)
```

**Example:** S(3) = 2.5 → F = 3 × (1 - 1/2.5) / (3 - 1) = 3 × 0.6 / 2 = 0.9 → S_max = 10

### Solving for F from S_max (speedup plot saturation)

```
S_max = 1 / (1 - F)   →   F = 1 - 1/S_max
```

**Example:** Speedup curve saturates at 5 → F = 1 - 1/5 = **0.8** (not 0.2, which is the serial fraction).

### Reversing to find T(1) from T(p)

```
T(1) = T(p) × S(p)
```

**Example:** T(4) = 10 min, F = 0.8 → S(4) = 1/(0.2 + 0.2) = 2.5 → T(1) = 10 × 2.5 = **25 min**

### Changing only the serial portion

Split T(1) into components:
```
T_serial = (1 - F) × T(1)
T_parallel = F × T(1)
T(p) = T_serial + T_parallel / p
```

Modify only T_serial, keep T_parallel unchanged:
```
T_new(p) = (T_serial ± change) + T_parallel / p
```

### Incomplete information

If you only know one parallelizable portion (say 16 s of a 32 s program), you cannot conclude a hard ceiling. The remaining portion might also be parallelizable. Answer: **"further analysis needed"** — not "cannot reach target."

### Amdahl's Law Traps Summary

| Trap | Correct |
|---|---|
| F = S_max (confusing F with S_max) | F = 1 - 1/S_max |
| Use S_max to answer "can we reach X with 8 cores?" | Use S(8), not S_max |
| "serial fraction is 0.2" → "F = 0.2" | F = 1 - 0.2 = 0.8 |
| Solve with wrong hardware limit | Always evaluate S(p) at actual p, not ∞ |

---

## TOPIC 6 — GIL, Threads vs Processes

### The Rule

| Code type | Threading works? | Why |
|---|---|---|
| Pure Python loop (CPU-bound) | No | GIL serializes threads |
| NumPy operations | Yes | NumPy releases the GIL |
| Numba `@jit(nogil=True)` | Yes | Numba releases the GIL |
| I/O operations | Yes | I/O releases the GIL |

**When to use each:**
- `ThreadPool` → NumPy-heavy or I/O-bound code, or Numba with `nogil=True`
- `Pool` (multiprocessing) → pure Python CPU-bound code
- **Never** use threads for pure Python CPU work

### Sequential Dependencies

A loop is parallelizable ONLY if iteration i+1 does NOT depend on the output of iteration i.

```python
# Sequential dependency — CANNOT parallelize
for i in range(n):
    scene = advance_scene(scene, dt)   # scene depends on previous scene

# No dependency — CAN parallelize
for scene in all_scenes:
    frame = render_scene(scene)        # independent per item
```

### Static vs Dynamic Scheduling

- **Static**: divide work evenly upfront. Best when all tasks take roughly equal time (StdDev/Avg ≈ 0).
- **Dynamic**: hand out tasks on demand. Best when task runtimes vary (StdDev/Avg is large).

**Signal in profiler**: if kernel StdDev ≈ 0 → static fine. If StdDev >> 0 → use dynamic.

### Parallel Fraction Scope

**Can parallelize the outer loop** if calls are independent. Cannot parallelize the inner loop if there's a sequential state update. Example:

```python
def simulate(n, x0s, step):
    for j in range(m):         # ← parallelize THIS (m independent sims)
        r = simulate_single(x0s[j], n, step)   # inner loop has x→x dependency, can't parallelize
```

Use `ThreadPoolExecutor` if function has `nogil=True`. Use `multiprocessing.Pool` otherwise.

---

## TOPIC 7 — Parallel Reduction

### Requirement for Reduction Trees

The combining function must be **both commutative and associative**:
- **Commutative**: f(a, b) == f(b, a)
- **Associative**: f(f(a, b), c) == f(a, f(b, c))

**Passes:** sum, product, min, max, set intersection (`∩`), bitwise AND/OR/XOR

**Fails:** abssum `abs(x+y)`, subtraction, division, set difference

### Proving non-associativity (exam technique)

Find a counterexample with sign cancellation:
```
abssum(abssum(1, 2), -3) = abssum(3, -3) = |0| = 0
abssum(1, abssum(2, -3)) = abssum(1, 1)  = |2| = 2
0 ≠ 2  → not associative
```

**Fix**: perform a plain sum reduction, take abs only at the very end.

### Set Intersection is Valid

- Commutative: A ∩ B = B ∩ A ✓
- Associative: (A ∩ B) ∩ C = A ∩ (B ∩ C) ✓ (element x in result iff x is in all sets, regardless of grouping)

---

## TOPIC 8 — Pandas & Memory Optimization

### Dtype Reduction Strategy

For each column, ask:
1. Is it `object`? → Almost always recode.
2. Is it a date/time stored as string? → Convert to `datetime64`.
3. Is it a string with few unique values? → Convert to **categorical** (huge win: stores strings once, uses uint index per row).
4. Is it integer? → Pick the smallest int type whose range covers [min, max].
5. Is it float? → Rarely worth converting (check precision requirements).

### Integer Type Ranges (memorize)

- `int8`: -128 to 127
- `uint8`: 0 to 255
- `int16`: -32,768 to 32,767
- `int32`: -2,147,483,648 to ~2.1 billion

**Watch for negatives**: range [-1, 5730] → cannot use `uint16`, must use `int16` (max 32,767 ✓).

### Rows-per-Chunk Calculation

```
bytes_per_row = sum of bytes per dtype per column
max_rows = available_memory_bytes / bytes_per_row
```

**Example:** 3 columns of int64 (8 bytes each) = 24 bytes/row. 24 MB available = 24,000,000 bytes.  
max_rows = 24,000,000 / 24 = 1,000,000. Pick largest option ≤ 1,000,000.

**Trap:** 1 MB = 1,048,576 bytes (binary) or 1,000,000 bytes (SI). The answer will be "around X" so either gives the same order of magnitude.

### Pandas Query Speed

For "find rows matching a date" operations repeated many times:
```python
df = df.set_index('date').sort_index()
# Now df.loc['2024-05-28'] uses binary search: O(log n) instead of O(n)
```

The one-time sort cost is amortized over many queries.

---

## TOPIC 9 — Memory-Mapped Files & Zarr

### numpy.memmap

```python
x = np.memmap('bigarray.raw', mode='r', dtype='uint8', shape=10_000_000_000)
```

- Does **not** load the file into RAM.
- OS lazily fetches pages only when accessed.
- Memory footprint = only what you actually materialize into a regular array.

**Example:** `x[::100_000]` from 10B elements = 10B/100K = **100,000 elements**. At uint8 (1 byte each) = **100 KB** in RAM. Not 10 GB.

### Zarr Chunk Sizing

Match chunk shape to your access pattern. If you access row by row:

| Chunk | Loads per `a[i]` | Total loads for all rows |
|---|---|---|
| (1, 1024) | 1 (one complete row per chunk) | 1,024 total |
| (32, 32) | 32 (32 chunks span one row) | 32,768 total |
| (1024, 1) | 1024 (each column is a chunk) | 1,048,576 total |

**Rule:** chunk boundaries should align with your access pattern. Row access → one chunk = one row.

---

## TOPIC 10 — Numba CPU JIT

```python
from numba import jit

@jit(nopython=True)       # compile to native code, no Python objects
def func(x):
    ...

@jit(nopython=True, nogil=True)   # also releases GIL (enables Python threading)
def func(x):
    ...
```

- **First call is slow** (compilation). Always warm up before timing.
- Compilation overhead is constant (not proportional to array size).
- Shines for loops that NumPy cannot vectorize.
- Speedup: typically 50–500× over pure Python loops.

### Cache-Efficient Loop Order for JIT

For a triple-nested matmul, `ikj` order is cache-friendly:
```python
for i in range(n):
    for k in range(n):
        for j in range(n):        # innermost = j → accesses B[k,j] along row ✓
            C[i,j] += A[i,k] * B[k,j]
```

The original `ijk` has innermost `k` → accesses `B[k,j]` along a column (non-contiguous) → cache misses.

---

## TOPIC 11 — CUDA / GPU Computing

### Thread Hierarchy

```
Grid of blocks → each block has threads → threads grouped into warps of 32
```

- Max threads per block: 1024
- Typical block sizes: 128, 256, 512 (must be multiple of 32)
- One thread handles one output element

### Grid Size Formula

```python
bpg = (N + tpb - 1) // tpb          # 1D
bpg = (ceil(rows/tpb[0]), ceil(cols/tpb[1]))   # 2D
```

**Always bounds-check inside kernel** — grid may have more threads than elements.

### 2D Kernel Pattern

```python
@cuda.jit
def my_kernel(arr):
    j, i = cuda.grid(2)             # j=column (x-dim), i=row (y-dim)
    if i < arr.shape[0] and j < arr.shape[1]:
        arr[i, j] = ...

tpb = (16, 16)
bpg = (ceil(W/16), ceil(H/16))      # (x-blocks, y-blocks)
my_kernel[bpg, tpb](arr)
```

**Note:** `cuda.grid(2)` returns `(x, y)` → `j, i = cuda.grid(2)` means `j` varies with `threadIdx.x` (across columns).

### Memory Coalescing

Threads in the same warp execute simultaneously. For a **warp** to do a coalesced memory access, adjacent threads must access **adjacent memory addresses**.

In a 2D kernel with `j, i = cuda.grid(2)`, threads with adjacent `j` values are in the same warp. For `arr[i, j]` (row-major), adjacent `j` → adjacent memory (last axis, stride 1) → **coalesced** ✓.

For `vol[vi, vj, vk]` — adjacent `j` threads vary in `vj` (controlled by `v_step`):
- `v_step = [0, 0, 1]` → adjacent threads differ in `vk` (axis 2, stride 1) → **coalesced** ✓
- `v_step = [0, 1, 0]` → adjacent threads differ in `vj` (axis 1, stride = dim2) → **non-coalesced** ✗ (worst)

### Memory Transfers

```python
# Explicit (efficient)
d_x = cuda.to_device(x)             # HtoD
d_out = cuda.device_array(n)        # allocate on device
kernel[bpg, tpb](d_x, d_out)
result = d_out.copy_to_host()       # DtoH

# Numba automatic (with plain NumPy arrays)
kernel[bpg, tpb](x, out)           # Numba does 2 HtoD + 2 DtoH (copies ALL args both ways)
```

**Numba auto-transfer trap:** Calling `@cuda.jit` with NumPy arrays → Numba copies **all** arguments both to GPU (before) and back (after), even write-only outputs and read-only inputs. That's **2 HtoD + 2 DtoH** for a 2-argument kernel.

**Optimal:** 1 HtoD (input only) + 1 DtoH (output only) — use `cuda.to_device` and `cuda.device_array`.

### nsys Profiler Output

```
** CUDA GPU Kernel Summary:
   Time (%)  Total Time (sec)  Name
   100.0          0.5000        my_kernel

** GPU MemOps Summary (by Time):
   Time (%)  Total Time (sec)  Operation
      83.3          2.5000      [CUDA memcpy HtoD]
      16.7          0.5000      [CUDA memcpy DtoH]

** GPU MemOps Summary (by Size):
   Total (MB)  Operation
   25000.000   [CUDA memcpy HtoD]
    1000.000   [CUDA memcpy DtoH]
```

**Bandwidth = Total Size / Total Time:**
```
HtoD: 25,000 MB / 2.5 s = 10,000 MB/s ≈ 10 GB/s
```

**Trap:** "Average size per transfer" (12,500 MB) is NOT bandwidth. Use total size / total time.

**Total GPU pipeline time** = HtoD + kernel + DtoH. Compare THIS to CPU time for real speedup.

**Example:**
- CPU: 7 s
- GPU: HtoD 2.5 + kernel 0.5 + DtoH 0.5 = **3.5 s**
- Speedup = 7 / 3.5 = **2×** (not 7/0.5 = 14×, which ignores transfers)

### Thread Block Count

Grid size is determined by the **output** array shape, NOT the input/volume shape.

```
Output: 200×200, Block: 16×16 → ceil(200/16) = 13 → Grid: 13×13 blocks
```

**Trap:** Using volume size (500) instead of output size (200) → gives 32×32 (wrong).

### GPU Break-Even Point

If GPU has a fixed transfer overhead and lower per-iteration cost:
```
GPU time = transfer_overhead + per_iter_cost × n
CPU time = per_iter_cost_cpu × n

Break-even: n = transfer_overhead / (per_iter_cost_cpu - per_iter_cost_gpu)
```

**Example:** GPU: 0.60 s overhead + 0.05 s/iter. CPU: 0.10 s/iter.
Break-even = 0.60 / (0.10 - 0.05) = **12 iterations**. For n > 12, GPU wins.

**Trap:** Don't evaluate GPU vs CPU at n=5 if break-even is at n=12 — the benchmark gives a misleading result.

---

## TOPIC 12 — CuPy & GPU Shared Memory (Week 10)

### Shared Memory Pattern

```python
from numba import cuda, float32

@cuda.jit
def reduction_kernel(x, out):
    shared = cuda.shared.array(shape=256, dtype=float32)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    
    shared[tid] = x[i] if i < x.shape[0] else 0.0
    cuda.syncthreads()          # REQUIRED: wait for all threads to load
    
    # Reduction tree
    step = 1
    while step < cuda.blockDim.x:
        if tid % (2 * step) == 0:
            shared[tid] += shared[tid + step]
        step *= 2
        cuda.syncthreads()      # REQUIRED between reduction steps
    
    if tid == 0:
        out[cuda.blockIdx.x] = shared[0]
```

**Key rules:**
- `cuda.shared.array` is per-block shared memory (fast, limited size)
- `cuda.syncthreads()` must be called after each collective write to shared memory
- Warp divergence (threads taking different branches) serializes execution

---

## TOPIC 13 — HPC Workflows (Job Arrays & Dependencies)

### Complete Job Array Setup

```bash
#!/bin/bash
#BSUB -J myjob[1-12]
#BSUB -q hpc
#BSUB -W 02:30
#BSUB -R "rusage[mem=250MB]"
#BSUB -n 1
#BSUB -o myjob_%I_%J.out    # %I = array index, %J = job ID

python myanalysis.py $LSB_JOBINDEX data/
```

Post-processing job:
```bash
#!/bin/bash
#BSUB -J postproc
#BSUB -w "ended(myjob)"     # ended = finished (regardless of success)
#BSUB -q hpc
python combine_results.py
```

### Dependency Conditions

| Condition | Meaning |
|---|---|
| `done(jobname)` | All jobs with that name finished **successfully** (exit code 0) |
| `ended(jobname)` | All jobs with that name finished, **success or failure** |
| `done(jobname[i])` | Specific array element i finished successfully |

**Common exam question:** "Run post-job after array, regardless of individual failures." → `ended`, not `done`.

---

## TOPIC 14 — HPC Pitfalls (Week 13)

### Thread Environment Variables

When using NumPy/OpenBLAS, BLAS may spawn its own threads. If you also use multiprocessing, you get P × T threads competing for P cores (T = BLAS threads per process).

```bash
export OMP_NUM_THREADS=1      # or:
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Set these to 1 when doing manual multiprocessing to avoid oversubscription.

### I/O Redirection in Job Scripts

```bash
# Wrong: parallel processes all write to the same file → corruption
for i in $(seq 1 8); do
    python process.py $i >> output.txt &
done

# Right: each process writes to its own file
for i in $(seq 1 8); do
    python process.py $i > output_$i.txt &
done
wait  # wait for all background processes
```

### Orphan Processes

If you kill the parent job, child processes may keep running and consuming cluster resources. Always use `wait` or proper process management.

---

## EXAM QUICK DECISION GUIDE

**Q: "How much memory in rusage?"** → `total_memory / n_cores`

**Q: "Should we parallelize?" (Amdahl)** → Compute S(p_actual), compare to threshold. Not S_max.

**Q: "Threads or processes?"** → Pure Python CPU-bound → processes. NumPy/Numba nogil → threads OK.

**Q: "Static or dynamic scheduling?"** → Check StdDev/Avg. Near zero → static. Large → dynamic.

**Q: "Is X suitable for parallel reduction?"** → Test f(f(a,b),c) == f(a,f(b,c)) with a concrete counterexample.

**Q: "Best array layout for CPU?"** → innermost loop axis has shortest stride → channels last for channel loop.

**Q: "Best array layout for GPU?"** → j-adjacent threads (warp) need adjacent memory → j must index last axis → spatial dims last → channels first.

**Q: "How many HtoD/DtoH with NumPy arrays?"** → Numba does 2+2 (all args both ways). Optimal = 1+1.

**Q: "Bandwidth from nsys?"** → total_size_MB / total_time_s. Not avg_size.

**Q: "Total GPU time for speedup?"** → HtoD + kernel + DtoH. Never just kernel.

**Q: "GPU or not worth it?"** → Find break-even iterations. If benchmarks are below break-even, GPU wins at realistic scale.

**Q: "ended vs done?"** → "Regardless of success" = `ended`. "Only if successful" = `done`.

**Q: "dtype for column?"** → Check range, pick smallest that fits. Dates as object → datetime64. Few unique strings → categorical.

**Q: "Memory footprint of memmap?"** → Only materialized slices count. The map itself uses no RAM.

**Q: "Best Zarr chunk?"** → One chunk = one access pattern unit (e.g., one row for row-by-row access).

**Q: "cProfile bottleneck for large workload?"** → Separate one-time functions from per-item functions. Scale only the per-item ones.

---

## FORMULAS CHEAT SHEET

```
Amdahl speedup:     S(p) = 1 / ((1-F) + F/p)
Max speedup:        S_max = 1 / (1-F)
F from S_max:       F = 1 - 1/S_max
F from S(p):        F = p(1 - 1/S) / (p - 1)
T serial:           T(1) = T(p) × S(p)

rusage memory:      mem_per_core = total_mem / n_cores
Grid size (1D):     bpg = (N + tpb - 1) // tpb
Grid size (2D):     bpg = (ceil(rows/tpb[0]), ceil(cols/tpb[1]))

GPU bandwidth:      bandwidth = total_size / total_time
GPU speedup (real): cpu_time / (HtoD + kernel + DtoH)
GPU break-even:     n_iters = fixed_overhead / (cpu_iter - gpu_iter)

Rows in memory:     max_rows = available_bytes / bytes_per_row
float16 gap at M:   gap ≈ M × 0.001
```

---

## ALL PAST EXAM ANSWERS (quick reference)

### 2024 Exam Answers
| Q | Answer | Topic |
|---|---|---|
| 1 | `-n 8`, `rusage[mem=2GB]`, `span[hosts=1]`, `-W 00:15` | BSUB correction |
| 2 | C) 0.8 | F from S_max = 5 |
| 3 | No, S(8)=3.33 < 4 | Amdahl at hardware limit |
| 4 | B) Multiprocessing | GIL, pure Python |
| 5 | Dynamic scheduling | Variable task time (n varies 1 to 100M) |
| 6 | abssum not associative; use plain sum + abs at end | Reduction associativity |
| 7 | B) `images - mim[:,:,None]` | Broadcasting |
| 8 | Reorder by stride: k(8)→j(40)→l(200)→i(600) outermost | Loop order from strides |
| 9 | C) 10 | ncalls of process_sample |
| 10 | process_sample | Per-item function scales to 1000 samples |
| 11 | Channels last h×w×c | CPU innermost=channel loop |
| 12 | Channels first c×h×w | GPU warp coalescing |
| 13 | B) ~10 GB/s | 25,000 MB / 2.5 s |
| 14 | 2× faster | CPU 7s vs GPU 3.5s (HtoD+kernel+DtoH) |
| 15 | date→datetime64; location→categorical; mach_id→int16; units→int32 | dtype reduction |
| 16 | Set index on date, sort_index() | Pandas sorted index lookup |
| 17 | B) ~10,000,000 | 200 MB / 20 bytes/row |
| 18 | `#BSUB -w "ended(power)"` | ended not done |
| 19 | Multithread outer loop; nogil=True allows threads; S≈min(t,m) | nogil Numba threading |

### F25 Exam Answers
| Q | A | Topic |
|---|---|---|
| 1 | A (25 GB) | rusage = 100/4 |
| 2 | B (10000) | float16 gap at 10000 = 10, adding 1 is no-op |
| 3 | A (Yes) | Set intersection is commutative and associative |
| 4 | A (67) | reshape(-1)[8] = row1[3] = 67 |
| 5 | A (`[:, None, None]`) | (N,3)→(N,1,1,3) for (N,H,W,3) |
| 6 | A (render_scene) | Highest cumtime = 8.841s |
| 7 | B (second loop only) | First loop has sequential dependency |
| 8 | C (~10×) | S(3)=2.5→F=0.9→S_max=10 |
| 9 | B (real halves, user same) | Parallel: real↓, user constant |
| 10 | B (kernel1 needs dynamic) | StdDev=40 (high var) vs StdDev=0.05 (uniform) |
| 11 | A (~14.7 s) | prep_conds fixed (2s) + loop×10 (12.69s) |
| 12 | A (v_step=[0,1,0] worst) | v_step=[0,1,0]→axis-1 stride=dim2 (non-coalesced) |
| 13 | A (13×13) | ceil(200/16)=13, not 32 (volume size irrelevant) |
| 14 | B (HtoD dominates) | 26.8 ms > 13.8 ms kernel > 0.16 ms DtoH |
| 15 | A (2 HtoD + 2 DtoH; optimal 1+1) | Numba auto-transfers all args |
| 16 | A (slower memory hierarchy) | Random access → cache misses → DRAM |
| 17 | C (smaller integer type) | 0-42 fits in uint8 |
| 18 | A (800,000) | 24MB/24bytes=1M rows; 800K is largest choice ≤1M |
| 19 | D (~100 KB) | 10B/100K = 100K elements × 1 byte |
| 20 | A ((1,1024)) | One chunk = one row = 1 load per row |
| 21 | D (loop swap DOES impact) | Loop order affects cache; "no impact" is false |
| 22 | B (multiprocessing + dynamic) | GIL blocks threads; variable time needs dynamic |
| 23 | D (GPU wins at more iterations) | GPU 0.05s/iter < CPU 0.1s/iter; break-even=12 |
| 24 | B (need further analysis) | Other 16s is unknown — may be parallelizable |

### Re-exam 2024 Key Answers
| Q | Answer |
|---|---|
| 1 | Wall-time=2h, cores=8, memory=32GB total (4GB×8) |
| 2 | Change queue to `gpuv100`; add `#BSUB -gpu "num=1:mode=exclusive_process"` |
| 3 | 25 min (T(1) = T(4) × S(4) = 10 × 2.5) |
| 4 | 7 min (T_serial=2, T_parallel/4=5) |

---

*End of master study guide. Good luck on the exam.*
