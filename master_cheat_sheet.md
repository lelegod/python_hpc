# 02613 Python HPC — Master Cheat Sheet

> **Root files:** [STUDY_GUIDE](STUDY_GUIDE.md) · [Exam Review](exam_review.md) · [Cheat Sheet](master_cheat_sheet.md) · [Tips & Pitfalls](tips_and_tricks.md) · [README](README.md)

---

## Contents

- [1. HPC System & Job Submission](#1-hpc-system--job-submission)
  - [All BSUB Directives](#all-bsub-directives)
  - [Job Arrays](#job-arrays)
  - [Job Dependencies](#job-dependencies)
  - [LSF Commands](#lsf-commands)
- [2. Performance Measurement](#2-performance-measurement)
  - [Standard Timing Pattern](#standard-timing-pattern)
  - [MFLOP/s Formula](#mflops-formula)
  - [Bandwidth Formula](#bandwidth-formula)
  - [Profiling Tools](#profiling-tools)
  - [Benchmarking Rules](#benchmarking-rules)
  - [`time` Command: Wall vs CPU Time](#time-command-wall-vs-cpu-time)
- [3. Memory Hierarchy & Cache](#3-memory-hierarchy--cache)
  - [Hierarchy](#hierarchy-fastest--slowest)
  - [Row vs Column Access](#row-vs-column-access-numpy-is-c-order--row-major)
  - [Loop Ordering Rule](#loop-ordering-rule)
  - [NumPy Strides](#numpy-strides)
- [4. NumPy Vectorization & Broadcasting](#4-numpy-vectorization--broadcasting)
  - [Broadcasting Rules](#broadcasting-rules-right-to-left)
  - [Key Broadcasting Patterns](#key-broadcasting-patterns)
  - [Views vs Copies](#views-vs-copies)
- [5. Parallelism — Amdahl's Law & Multiprocessing](#5-parallelism--amdahls-law--multiprocessing)
  - [Amdahl's Law](#amdahls-law)
  - [multiprocessing.Pool API](#multiprocessingpool-api)
  - [Static vs Dynamic Scheduling](#when-to-use-static-vs-dynamic-scheduling)
  - [Granularity: Critical Rule](#granularity-critical-rule)
- [6. Parallel Reductions](#6-parallel-reductions)
  - [Requirements](#requirements)
  - [Simple (Flat) Reduction](#simple-flat-reduction)
  - [Binary Tree Reduction](#binary-tree-reduction)
  - [Shared Memory for Reductions](#shared-memory-for-reductions)
- [7. Data Formats — Pandas & Apache Arrow](#7-data-formats--pandas--apache-arrow)
  - [Pandas Memory Optimization](#pandas-memory-optimization)
  - [PyArrow API](#pyarrow-api)
  - [File Format Comparison](#file-format-comparison-table)
- [8. Memory-Mapped Files](#8-memory-mapped-files)
  - [numpy.memmap API](#numpymemmap-api)
  - [shared_memory API](#multiprocessingshared_memory-api)
  - [Zarr API](#zarr-api)
  - [Pandas Chunking](#pandas-chunking)
- [9. Numba & GPU/CUDA](#9-numba--gpucuda)
  - [Numba CPU JIT](#numba-cpu-jit)
  - [CUDA Kernel Structure](#cuda-kernel-structure)
  - [Grid/Block Calculation](#gridblock-calculation)
  - [Thread Index Variables](#thread-index-variables)
  - [Host-Device Memory Transfers](#host-device-memory-transfers)
  - [CUDA Warp Coalescing](#cuda-warp-coalescing-memory-access-patterns)
  - [GPU Profiling with nsys](#gpu-profiling-with-nsys)
  - [CuPy](#cupy-gpu-numpy-drop-in)
  - [GPU Reduction Kernel](#gpu-reduction-kernel-with-shared-memory)
- [10. HPC Workflows](#10-hpc-workflows)
  - [Job Array Patterns](#job-array-patterns)
  - [Pitfalls to Avoid](#pitfalls-to-avoid)
  - [Thread Count Environment Variables](#thread-count-environment-variables)
- [11. Common Formulas Quick Reference](#11-common-formulas-quick-reference)
- [12. Top 10 Pitfalls](#12-top-10-pitfalls-one-liner-each)

---

## 1. HPC System & Job Submission

### All BSUB Directives

```bash
#!/bin/bash
#BSUB -J jobname              # Job name (shown in bjobs/bstat)
#BSUB -q hpc                  # Queue: hpc (CPU) | gpuv100 | gpua100
#BSUB -W HH:MM                # Wall-clock limit (-W 2 = 2 min, -W 1:30 = 1h30m)
#BSUB -n N                    # Number of CPU cores
#BSUB -R "span[hosts=1]"      # REQUIRED for shared-memory parallelism
#BSUB -R "rusage[mem=XGB]"    # Memory PER CORE (total = X * n)
#BSUB -R "select[model==XeonGold6226R]"  # Pin to specific CPU model
#BSUB -o name_%J.out          # Stdout (%J = job ID)
#BSUB -e name_%J.err          # Stderr
#BSUB -o name_%J_%I.out       # Per-array-element output (%I = array index)
#BSUB -u user@dtu.dk          # Email address
#BSUB -N                      # Email on job end
#BSUB -B                      # Email on job begin
#BSUB -gpu "num=1:mode=exclusive_process"  # GPU allocation (add with GPU queue)

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Good practice: redirect output manually (26x faster than LSF -o channel)
python -u script.py \
    1> /work3/02613/output_${LSB_JOBID}.txt \
    2> /work3/02613/error_${LSB_JOBID}.txt
```

**CRITICAL TRAP:** `rusage[mem=XGB]` is per core. For 16 GB total on 8 cores → `rusage[mem=2GB]`

### Job Arrays

```bash
#BSUB -J name[1-10]           # Range: indices 1..10
#BSUB -J name[1,4,7]          # Explicit list
#BSUB -J name[1-100:2]        # Step: 1,3,5,...99
#BSUB -J name[1,4,10-14,20-30:2]  # Mixed

# Inside job: use $LSB_JOBINDEX (1-based!)
IDX=${LSB_JOBINDEX}
python script.py ${IDX}       # script converts 1-based to 0-based

# Each element gets its own log file:
#BSUB -o name_%J_%I.out       # ALWAYS do this for arrays
```

### Job Dependencies

```bash
#BSUB -w "done(jobname)"      # Wait for ALL jobs named jobname → DONE (success only)
#BSUB -w "ended(jobname)"     # Wait for ALL jobs → DONE or EXIT (any termination)
#BSUB -w "done(array[3])"     # Wait for specific array element
#BSUB -w "done(array[*])"     # Element-wise: new[i] waits for old[i]

# Or pass at submission time:
bsub -w "done(job1)" < job2.sh
```

**TRAP:** `done()` requires ALL jobs to succeed. If any job is in EXIT state, dependent job **never starts**. Use `ended()` for cleanup/always-run jobs.

### LSF Commands

```bash
bsub < submit.sh              # Submit job
bstat                         # Your jobs (compact)
bjobs                         # Your jobs (shows TIME_LEFT column)
bjobs -p                      # Why is job PEND? (diagnose)
bjobs -A                      # Array summary (one row per array, state counts)
bjobs -l JOBID                # Detailed job info
bpeek JOBID                   # Live stdout of running job
bpeek JOBID[3]                # Live stdout of array element
bkill JOBID                   # Kill job/array
bkill JOBID[1-5]              # Kill array subset
bkill 0                       # Kill ALL your jobs
nodestat -f hpc               # Available nodes and CPU models
bhist                         # Job history
```

**Job states:** `PEND → RUN → DONE` (success) or `EXIT` (failure/kill)

**Never run computation on login node.** Use `linuxsh` for interactive work.

---

## 2. Performance Measurement

### Standard Timing Pattern

```python
from time import perf_counter

t_start = perf_counter()
# ... work ...
elapsed = perf_counter() - t_start   # seconds, float
```

Use `perf_counter()` not `time.time()` — highest resolution, unaffected by clock adjustments.

**For fast operations, repeat and divide:**
```python
n_repeat = 1000
t = perf_counter()
for _ in range(n_repeat):
    result = fast_operation(x)
elapsed = (perf_counter() - t) / n_repeat
```

### MFLOP/s Formula

```
MFLOP/s = num_operations / (time_seconds * 1_000_000)
```

Count FLOPs manually — there is no automatic tool. Example: `2 * mat[0, :]` on SIZE elements = SIZE FLOPs.

### Bandwidth Formula

```
Bandwidth (B/s) = bytes_transferred / time_seconds
```

For float64: 1 element = 8 bytes. For float32: 4 bytes. For uint8: 1 byte.

**Matrix size in bytes:** `rows * cols * dtype_bytes`

### SI Prefixes

| Prefix | Symbol | Factor | Example |
|--------|--------|--------|---------|
| kilo   | k      | 10³    | 1 KB = 1,000 bytes |
| mega   | M      | 10⁶    | 1 MB = 1,000,000 bytes |
| giga   | G      | 10⁹    | 1 GHz = 1 billion cycles/s |
| tera   | T      | 10¹²   | 1 TB/s = peak memory bandwidth |
| peta   | P      | 10¹⁵   | 1 PFLOP/s = top supercomputer |
| milli  | m      | 10⁻³   | 1 ms = 0.001 s |
| micro  | µ      | 10⁻⁶   | 1 µs = 0.000001 s |
| nano   | n      | 10⁻⁹   | 1 ns ≈ 1 CPU clock cycle at 1 GHz |
| pico   | p      | 10⁻¹²  | 1 pJ = energy per GPU operation |

> **Binary vs decimal:** KB/MB/GB are loosely used — strictly 1 KiB = 1024 bytes, 1 GiB = 1024³ bytes. OS tools often report in GiB but label it GB.

### Profiling Tools

```bash
# Function-level profiling
python -m cProfile -s cumulative script.py args

# Line-level profiling (add @profile decorator first)
kernprof -l -v script.py args
```

**cProfile columns:** `ncalls | tottime (own only) | percall | cumtime (inclusive) | filename`

**line_profiler columns:** `Hits | Time | Per Hit | % Time | Line Contents`
- `Hits` tells you how many loop iterations ran → what the input size was
- Scale to production: `percall × expected_ncalls`

**FLOP/s from line profiler:**
```
FLOP/s = FLOPs_per_iteration * n_iterations / total_time_seconds
```

### Benchmarking Rules

- Always benchmark via a batch job on a dedicated compute node — never on the login node
- For cache effect plots: use `np.logspace(1, 4.5, 30)` for SIZE, plot on loglog axes
- Mark cache boundaries as vertical lines: L1 (~32 KB), L2 (~1 MB), L3 (~19 MB)
- `os.sync()` required before I/O benchmarks to flush OS page cache

### `time` Command: Wall vs CPU Time

```bash
time python script.py
# real  = wall-clock time (decreases with parallelism)
# user  = CPU time summed over all cores (stays same or INCREASES with parallelism)
# sys   = OS system call time
```

With 2 cores: real ~halves, user stays ~same.

---

## 3. Memory Hierarchy & Cache

### Hierarchy (fastest → slowest)

| Level | Size | Latency |
|-------|------|---------|
| Registers | bytes | ~0.25 ns |
| L1 Cache | ~32 KB/core | ~1 ns |
| L2 Cache | ~256 KB–1 MB/core | ~4 ns |
| L3 Cache | ~8–20 MB shared | ~10 ns |
| RAM | GBs | ~100 ns |
| Disk | TBs | ms |

**Memory wall:** RAM is ~200x slower than CPU. Caches bridge this gap.

**On XeonGold6226R (DTU cluster):**
- L1d: 32 KB | L2: 1 MB | L3: 19.7 MB

```bash
lscpu | grep cache    # Read cache sizes on any node
```

### Cache Lines = 64 bytes = 8 float64 values

When you access `x[0]`, the CPU **automatically fetches x[0]..x[7]** into L1 cache. The next 7 accesses are free cache hits — but only if you access **sequentially**.

### Spatial vs Temporal Locality

- **Spatial:** accessing nearby addresses → cache line prefetching helps → access arrays sequentially
- **Temporal:** reusing recently accessed data → stays warm in cache → reuse before eviction

### Row vs Column Access (NumPy is C-order / row-major)

```python
mat = np.random.rand(N, N)   # row-major storage

mat[0, :]    # ROW ACCESS: contiguous in memory → FAST (cache-friendly)
mat[:, 0]    # COL ACCESS: stride-N in memory  → SLOW (every element = cache miss)
```

**Real difference on 18K×18K image:** row = 1.67s, column = 20.15s → **12x slower**

**Performance staircase:** MFLOP/s drops at each cache level boundary. Plot MFLOP/s vs data size (bytes) on loglog axes to see steps at L1, L2, L3.

### Loop Ordering Rule

**Innermost loop → axis with smallest stride (last axis in C-order)**

Example: strides `(600, 40, 8, 200)` → sort ascending → axis 2 (8) < axis 1 (40) < axis 3 (200) < axis 0 (600) → loop order inner→outer: `k, j, l, i`

### NumPy Strides

```python
a = np.ones((2, 3))   # float64 (8 bytes/elem)
a.strides             # (24, 8) — 24 bytes to next row, 8 bytes to next col

# Byte offset for a[i, j] = i * strides[0] + j * strides[1]
```

- `a.T` → view, strides reversed
- `a[1, :]` → view (row slice)
- `a[[0, 2], :]` → always a copy (fancy indexing)
- `np.shares_memory(a, b)` → True if same buffer

---

## 4. NumPy Vectorization & Broadcasting

### Broadcasting Rules (right-to-left)

1. Right-align shapes; prepend 1s to shorter shape
2. Each dimension: must be **equal** OR one must be **1**
3. Output shape = max of each dimension
4. Mismatch → `ValueError`

```
(2, 3) + (3,)     → (2, 3)   ✓  [broadcast 3→2,3]
(2, 1, 3) + (4, 1) → (2, 4, 3) ✓
(2, 3) + (2,)     → ERROR    ✗  [need b[:, None] → (2,1)]
```

### Key Broadcasting Patterns

```python
# Outer product: (n,) × (m,) → (n, m)
x[:, None] * y          # x becomes (n,1), broadcasts with y (m,)

# Pairwise distances: (n,) vs (m,) → (n, m)
abs(x[:, None] - y)

# Haversine distance matrix: (N,2) × (M,2) → (N, M)
p1[:, None, :] - p2[None, :, :]    # (N,1,2) - (1,M,2) = (N,M,2)

# Image mean subtraction: images (N,H,W,3), mean (N,3) → (N,H,W,3)
images - mean_pixels[:, None, None]  # mean becomes (N,1,1,3)

# Standardize rows: data (n,d), mean (d,) → (n,d)
(data - mean) / std    # mean/std broadcast as (1,d) → (n,d)
```

### Views vs Copies

| Operation | Result |
|-----------|--------|
| `a.T` | View (strides reversed) |
| `a[1, :]` | View |
| `a.reshape(...)` on C-order | View if possible |
| `a.reshape(...)` on F-order | Copy |
| `a[[0, 2], :]` | Always copy |

### Avoid Python Loops

```python
# BAD: Python loop, 100x slower
for i in range(N):
    for j in range(M):
        D[i, j] = haversine(p1[i], p2[j])

# GOOD: one-loop (M elements per NumPy call)
for i in range(N):
    D[i, :] = haversine_row(p1[i], p2)   # ~80x speedup

# BETTER: no loop (full broadcasting, watch memory!)
D = haversine_matrix(p1, p2)              # creates (N, M, 2) intermediates
```

No-loop is not always faster — large intermediate arrays exhaust cache. Profile to find crossover.

---

## 5. Parallelism — Amdahl's Law & Multiprocessing

### Amdahl's Law

```
S(p) = 1 / (s + (1 - s) / p)
     = 1 / ((1 - F) + F/p)
```

Where: `F = parallel fraction`, `1-F = serial fraction`, `p = processors`

**Maximum speedup (p → ∞):**
```
S_max = 1 / (1 - F)
```

**Solve for F given measured speedup:**
```
F = p(1 - 1/S(p)) / (p - 1)
```

**Example:** S(3) = 2.5 → F = 3(1 - 1/2.5)/(3-1) = 3(0.4)/2 = 0.9 → S_max = 10

**Efficiency:**
```
E(p) = S(p) / p    (ideal = 1.0)
```

**Key insight:** 10% serial → max 10x speedup regardless of core count. Reducing the serial fraction often beats adding cores.

### Speedup Definition

```
S(p) = T(1) / T(p)    # wall-clock time, not CPU time
```

### multiprocessing.Pool API

```python
from multiprocessing import Pool

# pool.map: apply function to each element
with Pool(n_proc) as pool:
    results = pool.map(func, iterable)
    results = pool.map(func, iterable, chunksize=100)  # tune chunk size

# pool.starmap: multiple args per call
with Pool(n_proc) as pool:
    results = pool.starmap(func, [(a1, b1), (a2, b2), ...])

# pool.imap_unordered: dynamic scheduling (results as they complete)
with Pool(n_proc) as pool:
    results = list(pool.imap_unordered(func, iterable))

# pool.apply_async: submit individually, collect later
pool = Pool(n_proc)
futures = [pool.apply_async(func, (arg,)) for arg in args]
results = [f.get() for f in futures]
pool.close(); pool.join()
```

**ALWAYS include guard:**
```python
if __name__ == '__main__':    # Required on macOS/Windows to prevent infinite spawning
    with Pool(n) as pool:
        ...
```

### ThreadPool (when NumPy releases GIL)

```python
from multiprocessing.pool import ThreadPool
with ThreadPool(n_threads) as pool:
    results = pool.map(np.sum, arrays)   # Works! NumPy releases GIL
```

GIL rules:
- Pure Python loops → GIL blocks threads → use **multiprocessing**
- NumPy operations → GIL released → **ThreadPool** works (avoids copy overhead)
- `@jit(nogil=True)` → GIL released → ThreadPool works

### When to Use Static vs Dynamic Scheduling

- **Static** (`pool.map` with large chunksize): task times are uniform, low overhead
- **Dynamic** (`pool.imap_unordered` or `chunksize=1`): task times vary significantly (high stddev), prevents load imbalance

### Granularity: Critical Rule

```python
# BAD: 1,000,000 tasks = 1,000,000 inter-process messages → SLOWER than serial
results = pool.map(sample_one_point, range(1_000_000))

# GOOD: 10 tasks, each does 100,000 samples → only 10 messages
chunk = total // n_proc
results = pool.map(sample_chunk, [chunk] * n_proc)
```

---

## 6. Parallel Reductions

### Requirements

A reduction operator must be:
1. **Associative:** `(a op b) op c == a op (b op c)` — required to split into subproblems
2. **Commutative:** `a op b == b op a` — required because parallel implementations may reorder

Valid: sum, product, min, max, AND, OR, XOR, set intersection

Invalid: `abs(x + y)` — commutative but NOT associative. Counterexample: `|(1+2)+(-3)| = 0 ≠ |1+(2+(-3))| = 2`. Matrix multiply — NOT commutative.

**Fix for invalid operators:** do the valid parallel operation first, apply invalid transformation at the end (e.g., parallel sum → take abs of final result).

### Simple (Flat) Reduction

```
Split N elements into T chunks → reduce T chunks in parallel (N/T ops) → sum T results serially (T ops)

Total time = N/T + T
Optimal T = sqrt(N)
Max speedup = sqrt(N) / 2
```

### Binary Tree Reduction

```python
# ceil(log2(N)) rounds, stride doubles each round
for j in range(int(np.ceil(np.log2(len(arr))))):
    s = 2**j
    # Each task: arr[i] += arr[i + s]
    for i in range(0, len(arr), 2*s):
        if i + s < len(arr):
            arr[i] += arr[i + s]
```

```
Levels = ceil(log2(N))
Work per level = N/2 operations (in parallel)
Critical path depth = log2(N)
Max speedup = N / log2(N)
```

**Comparison:** N=100,000: flat=~158x, binary tree=~5,900x theoretical.

### Shared Memory for Reductions

```python
import ctypes
import multiprocessing as mp
import numpy as np

def init(shared_arr_):
    global shared_arr
    shared_arr = shared_arr_

# Allocate shared memory
data = np.load('data.npy')
shared_arr = mp.RawArray(ctypes.c_float, data.size)
arr = np.frombuffer(shared_arr, dtype='float32').reshape(data.shape)
np.copyto(arr, data)

# Workers access via module global
pool = mp.Pool(n, initializer=init, initargs=(shared_arr,))
```

No pickling/serialization — all processes share the same memory buffer.

### NUMA Warning

On dual-socket servers (XeonGold6226R): without numactl, all memory lands on socket 0. Speedup plateaus at ~50% of cores.

```bash
numactl --interleave=all python script.py   # Spread memory across all NUMA nodes
```

---

## 7. Data Formats — Pandas & Apache Arrow

### Pandas Memory Optimization

```python
import pandas as pd

# Inspect memory usage
df.info(memory_usage='deep')              # Summary with total
df.memory_usage(deep=True)                # Per-column bytes
df.memory_usage(deep=True).sum()          # Total bytes
# deep=True required for object columns (shows actual string heap, not just pointers)

# Convert timestamp strings (huge savings)
df['col'] = pd.to_datetime(df['col'], format="ISO8601")   # object → datetime64[ns]

# Convert low-cardinality strings
df['col'] = df['col'].astype('category')  # object → int codes + lookup table

# Downcast numeric types
df['col'] = pd.to_numeric(df['col'], downcast='integer')  # int64 → smallest int
df['col'] = pd.to_numeric(df['col'], downcast='float')    # float64 → float32

# dtype size ranges (choose smallest that fits min/max):
# int8:   -128 to 127
# uint8:  0 to 255
# int16:  -32768 to 32767
# int32:  ±2.1×10^9
# float32: ~7 decimal digits (check for overflow before downcasting)
```

**Memory reduction example (DMI dataset, 8M rows):**

| Column | Before | After | Change |
|--------|--------|-------|--------|
| Timestamps (object) | 597 MB | 48 MB (datetime64) | 12x |
| parameterId (object, 47 unique) | 546 MB | 6 MB (category) | 90x |
| stationId (int64) | 62 MB | 16 MB (int16) | 4x |
| value (float64) | 62 MB | 31 MB (float32) | 2x |
| **Total** | **~2045 MB** | **~200 MB** | **10x** |

### PyArrow API

```python
from pyarrow import csv
import pyarrow.parquet as pq
import pyarrow as pa

# Faster CSV loading (multi-threaded, ~3.7x faster than pd.read_csv)
table = csv.read_csv('file.csv')
df = table.to_pandas()   # Convert to Pandas

# With type optimization on load
convert_options = csv.ConvertOptions(
    column_types={
        'value': pa.float32(),
        'parameterId': pa.dictionary(pa.int32(), pa.string()),  # category
    }
)
table = csv.read_csv('file.csv', convert_options=convert_options)

# Parquet write
table = pa.Table.from_pandas(df)
pq.write_table(table, 'file.parquet')

# Parquet read
df = pd.read_parquet('file.parquet')          # Pandas (slower)
table = pq.read_table('file.parquet')         # PyArrow (faster ~400ms vs ~1s)

# Chunked Parquet (for large files)
pf = pq.ParquetFile('file.parquet')
pf.num_row_groups                             # Number of row groups
group = pf.read_row_group(i, columns=['a', 'b'])   # Column pruning!
df = group.to_pandas()
```

### Convert CSV → Chunked Parquet

```python
writer = None
for chunk in pd.read_csv('big.csv.zip', chunksize=100_000):
    table = pa.Table.from_pandas(chunk)
    if writer is None:
        writer = pq.ParquetWriter('big.parquet', schema=table.schema)
    writer.write_table(table)
writer.close()
```

### Vectorized Pandas Operations

```python
# Slow: Python loop (do not use)
for i in range(len(df)):
    if df.iloc[i]['col'] == val:
        total += df.iloc[i]['value']

# Fast: vectorized boolean indexing (~1000x faster)
total = df[df['col'] == val]['value'].sum()

# Fastest: indexed lookup (O(log n), worth it for many queries)
df_idx = df.set_index('col').sort_index()
total = df_idx.loc[val]['value'].sum()   # ~15,000x vs raw loop
```

### File Format Comparison Table

| Format | File Size | Read Speed | Write Speed | Use When |
|--------|-----------|------------|-------------|----------|
| CSV | ~1.2 GB | slow (~12s) | slow | human-readable, sharing |
| CSV.zip | compressed | medium (~11s) | slow | disk savings |
| Parquet | **86 MB** | fast (~1s Pandas, ~400ms Arrow) | medium (~1.6s) | repeated analysis |
| Arrow/Feather | small | fastest | fast | same-session transfer |

Parquet is ~14x smaller than CSV, ~10-30x faster to read.

---

## 8. Memory-Mapped Files

### numpy.memmap API

```python
import numpy as np

# Create new file (w+ = create or overwrite)
mm = np.memmap('data.raw', mode='w+', shape=(1000, 1000), dtype='float64')
mm[:, :] = computed_data      # Writes go directly to disk

# Open existing for reading
mm = np.memmap('data.raw', mode='r', dtype='float64', shape=(1000, 1000))

# Open existing for read/write
mm = np.memmap('data.raw', mode='r+', dtype='float64', shape=(1000, 1000))

# Copy-on-write (changes NOT saved to disk)
mm = np.memmap('data.raw', mode='c', dtype='float64', shape=(1000, 1000))
```

**Modes:**

| Mode | Creates | Read | Write | Persisted |
|------|---------|------|-------|-----------|
| `w+` | Yes | Yes | Yes | Yes |
| `r` | No | Yes | No | N/A |
| `r+` | No | Yes | Yes | Yes |
| `c` | No | Yes | Yes (copy) | No |

**CRITICAL:** Always specify both `dtype` AND `shape` when opening. Default dtype=uint8 will silently misinterpret data. A 10×10 float64 file opened without dtype shows as shape (800,) of garbage.

```python
# Only read accessed pages (downsampling uses tiny fraction of RAM)
big_mm = np.memmap('huge.raw', mode='r', dtype='int32', shape=(4000, 4000))
small = big_mm[::4, ::4]   # Only loads touched pages from disk
```

### multiprocessing.shared_memory API

```python
from multiprocessing import shared_memory
import numpy as np

# Create shared memory block
shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
np.copyto(shared_arr, arr)

# In worker processes (attach to existing block)
existing_shm = shared_memory.SharedMemory(name=shm.name)
worker_arr = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

# Cleanup
shm.close()
shm.unlink()
```

### When to Use Each

| Technique | Use When |
|-----------|----------|
| `np.memmap` | Large numerical arrays, disk-backed, parallel writes to different regions |
| `mp.RawArray` + `np.frombuffer` | Shared memory between processes, no disk persistence |
| `shared_memory.SharedMemory` | Modern Python 3.8+ shared memory, cleaner API |
| Zarr | Compressed storage, chunk-based parallel writes, interoperable |

### Zarr API

```python
import zarr

# Create Zarr array
z = zarr.open('data.zarr', mode='w', shape=(1000, 1000),
              chunks=(50, 50), dtype='int32')
z[0:50, 0:50] = data_chunk   # Write one chunk

# Different processes can write different chunks simultaneously (no locking)

# Read
z = zarr.open('data.zarr', mode='r')
data = z[100:200, :]

# Nested storage (avoid too many files in one dir)
store = zarr.storage.NestedDirectoryStore('data.zarr')
z = zarr.open(store, mode='w', shape=shape, chunks=chunks)
```

**Chunk size tuning (1000×1000 Mandelbrot, 24 cores):**

| Chunk size | Runtime | Stored size |
|------------|---------|-------------|
| 10×10 | 24.2s (slow) | 285 MB |
| 50×50 | 1.1s (optimal) | 9.5 MB |
| 200×200 | 4.3s | 656 KB (smallest) |

Too-small chunks: overhead. Too-large chunks: poor parallelism. Optimal chunk exists in the middle.

### Pandas Chunking

```python
import pandas as pd

# CSV too large for RAM
dfc = pd.read_csv('big.csv.zip', chunksize=100_000)
total = 0.0
for chunk in dfc:   # Only one chunk in RAM at a time
    total += chunk[chunk['col'] == val]['value'].sum()
# Cannot do len(dfc), random access, or re-iterate without reopening
```

---

## 9. Numba & GPU/CUDA

### Numba CPU JIT

```python
from numba import jit, njit

@jit(nopython=True)   # Equivalent to @njit
def fast_func(A, B):
    result = 0.0
    for i in range(len(A)):
        result += A[i] * B[i]
    return result

# ALWAYS warm up before timing (first call includes compilation)
fast_func(A, B)    # triggers compilation, slow

# Time subsequent calls
t = perf_counter()
result = fast_func(A, B)   # fast
elapsed = perf_counter() - t

# With GIL release (for ThreadPool parallelism)
@jit(nopython=True, nogil=True)
def fast_func_nogil(arr):
    ...

# Cache-efficient loop ordering (ikj not ijk for matmul)
@njit
def matmul(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for k in range(A.shape[1]):      # ikj order → B accessed row-wise
            for j in range(B.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C
```

**JIT speedup:** Pure Python loop: 21.5ms → JIT: 0.1ms → ~200x speedup

### CUDA Kernel Structure

```python
from numba import cuda
import numpy as np

# 1D kernel
@cuda.jit
def kernel_1d(x, y, out):
    i = cuda.grid(1)             # Global thread index
    if i < x.shape[0]:          # ALWAYS bounds check
        out[i] = x[i] + y[i]

# 2D kernel (matrix operations)
@cuda.jit
def kernel_2d(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = float32(0.)
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

# Device helper (callable from kernel only, not from host)
@cuda.jit(device=True)
def helper(a, b):
    return a * b
```

### Grid/Block Calculation

```python
# 1D grid (always round UP)
tpb = 256                          # threads per block (multiple of 32!)
bpg = (n + tpb - 1) // tpb        # blocks per grid = ceil(n / tpb)
kernel_1d[bpg, tpb](x, y, out)

# 2D grid
tpb = (16, 16)
bpg = (
    (A.shape[0] + tpb[0] - 1) // tpb[0],
    (A.shape[1] + tpb[1] - 1) // tpb[1]
)
kernel_2d[bpg, tpb](A, B, C)
```

**Max threads per block: 1024. Warp size = 32. Use multiples of 32 (128, 256, 512).**

### Thread Index Variables

| Variable | Meaning |
|----------|---------|
| `cuda.threadIdx.x` | Index within block |
| `cuda.blockIdx.x` | Block index in grid |
| `cuda.blockDim.x` | Threads per block |
| `cuda.gridDim.x` | Blocks per grid |
| `cuda.grid(1)` | Global 1D index = `blockIdx.x * blockDim.x + threadIdx.x` |
| `cuda.grid(2)` | Global (row, col) = `(i, j)` |

### Host-Device Memory Transfers

```python
from numba import cuda

# EXPLICIT transfers (avoid implicit for performance)
d_x = cuda.to_device(x)          # Host → Device (HtoD)
d_y = cuda.to_device(y)
d_out = cuda.device_array(n)     # Allocate on device (no copy)
d_out = cuda.device_array_like(x)

# Run kernel (no transfer overhead in timing)
kernel[bpg, tpb](d_x, d_y, d_out)
cuda.synchronize()               # REQUIRED before reading results or timing

# Device → Host (DtoH)
result = d_out.copy_to_host()

# Pinned memory (faster DMA)
with cuda.pinned(x, y, out):
    kernel[bpg, tpb](x, y, out)
    cuda.synchronize()

# Numba auto-transfers (IMPLICIT — expensive!)
# Calling kernel with NumPy arrays: transfers ALL args HtoD before, ALL args DtoH after
# → 2 HtoD + 2 DtoH for a 2-arg kernel
# → Optimal: 1 HtoD per input, 1 DtoH per output
```

### CUDA Warp Coalescing (Memory Access Patterns)

**Adjacent threads in a warp differ by 1 in threadIdx.x (x-dim = FIRST return value of `cuda.grid(2)`).**

With `row, col = cuda.grid(2)`: row=x-dim, col=y-dim.
→ Adjacent threads differ by 1 in **row** (not col).
→ For coalesced access of `x[row, col]`, you need **col to vary** → requires blockDim.x=1.

**Block shape determines which index varies in the warp** (Thread ID = threadIdx.x + threadIdx.y * blockDim.x):

```
Array shape (rows, cols), row-major: last axis (col) has smallest stride
→ Want col to vary across warp → need blockDim.x = 1

Block shape comparison for row, col = cuda.grid(2):
  (1, 256): blockDim.x=1 → threadIdx.x=0 always → col varies  → COALESCED   [FASTEST]
  (16, 16): blockDim.x=16 → row varies 0..15 twice → col changes once  [PARTIAL]
  (256, 1): blockDim.x=256 → row varies 0..31 → col never changes   → NON-COALESCED [SLOWEST]
```

**DTU lecture convention:** Use `j, i = cuda.grid(2)` (swap names) so j=x-dim varies in warp, then j indexes col (last axis) → coalesced without needing blockDim.x=1. Grid must also be swapped: `bpg = (cols//tpb[0], rows//tpb[1])`.

**CPU vs CUDA layout — opposite rules:**
- **CPU:** inner loop (sequential) should be last dimension → channels last = HWC
- **CUDA:** all warp threads access same channel simultaneously → channels first = CHW

### GPU Profiling with nsys

```bash
nsys profile -o profile_name python script.py args
nsys stats profile_name.nsys-rep
```

**Key sections:**
- `gpukernsum` → kernel execution times
- `gpumemtimesum` → HtoD and DtoH transfer times
- `gpumemsizesum` → transfer volumes
- `cudaapisum` → CPU-side CUDA API calls (includes `cuLaunchKernel` overhead)

**Transfer bandwidth:** `total_size_MB / total_time_s = MB/s`

### CuPy (GPU NumPy Drop-in)

```python
import cupy as cp   # Replace numpy with cupy

# Load directly to GPU
data = cp.loadtxt('file.csv', delimiter=',', skiprows=1)

# Same API as NumPy — all ops run on GPU
result = cp.sin(data) + cp.cos(data)
result_cpu = cp.asnumpy(result)   # or result.get()

# Rule: AVOID Python loops — each iteration = separate kernel launch overhead
# BAD (Python loop → many small kernels):
for i in range(N):
    y[i] = cp.sin(x[i])

# GOOD (vectorized → one large kernel):
y = cp.sin(x)
```

**CuPy speedup (Haversine, 5000 locations):**
- NumPy one-loop: 10.42s
- CuPy one-loop: 3.29s (3x faster, minimal code change)
- CuPy no-loop: 1.00s (10x faster)

### GPU Reduction Kernel (with shared memory)

```python
from numba import cuda

TPB = 64

@cuda.jit
def reduce_kernel(data, out, n):
    sdata = cuda.shared.array(TPB, dtype=data.dtype)  # Shared memory
    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    sdata[tid] = data[i] if i < n else 0.0   # Load global → shared
    cuda.syncthreads()                        # Wait for all loads

    s = 1
    while s < cuda.blockDim.x:
        idx = 2 * s * tid                    # Strided index (no warp divergence)
        if idx < cuda.blockDim.x:
            sdata[idx] += sdata[idx + s]
        s *= 2
        cuda.syncthreads()

    if tid == 0:
        out[cuda.blockIdx.x] = sdata[0]     # Block result → global

def gpu_sum(x):
    n = len(x)
    bpg = (n + TPB - 1) // TPB
    out = cuda.device_array(bpg, dtype=x.dtype)
    while bpg > 1:
        reduce_kernel[bpg, TPB](x, out, n)
        n = bpg
        bpg = (n + TPB - 1) // TPB
        x[:n] = out[:n]
    reduce_kernel[1, TPB](x, out, n)
    return out
```

**Warp divergence:** `if tid % (2*s) == 0` → half threads idle per warp. Fix: `idx = 2*s*tid` (strided index) → active threads are contiguous → ~2.3x speedup.

---

## 10. HPC Workflows

### Job Array Patterns

```bash
# Map-Reduce workflow
# 1. Submit map jobs
#BSUB -J map[1-203]
#BSUB -o output/map_%J_%I.out   # Per-element output
python process.py $LSB_JOBINDEX  # 1-indexed → script converts to 0-indexed with -1

# 2. Submit reduce job (automatically waits for ALL map jobs to succeed)
#BSUB -J reduce
#BSUB -w done(map)               # Waits for ALL map[1-203] to reach DONE
python aggregate.py
```

### Python Script for Array Jobs

```python
import sys
import os

def main():
    idx = int(sys.argv[1]) - 1    # Convert 1-based LSB_JOBINDEX to 0-based
    all_inputs = sorted(os.listdir('/path/to/data'))
    this_input = all_inputs[idx]
    result = process(this_input)
    np.save(f'result_{idx}.npy', result)

if __name__ == '__main__':
    main()
```

### Pitfalls to Avoid

```bash
# DO NOT: email with large arrays
# BAD (generates 364 emails for 182-element array):
#BSUB -N
#BSUB -J name[1-182]

# DO NOT: spam scheduler
watch bstat           # BAD (every 2s)
watch -n0.1 bstat     # TERRIBLE

# GOOD:
watch -n60 bstat      # once per minute max

# DO NOT: leave orphaned background processes
mymonitor &
python script.py
# job waits until wall time!

# GOOD:
mymonitor &
MONITOR_PID=$!
python script.py
kill $MONITOR_PID; wait $MONITOR_PID

# DO NOT: use tee (duplicates I/O)
python script.py | tee output.txt    # BAD

# GOOD: choose one output path
python script.py > /work3/02613/output.txt
```

### Thread Count Environment Variables

```bash
# Set these to match #BSUB -n to prevent oversubscription
NUM_THREADS=8
OMP_NUM_THREADS=$NUM_THREADS
MKL_NUM_THREADS=$NUM_THREADS
OPENBLAS_NUM_THREADS=$NUM_THREADS
MPI_NUM_THREADS=$NUM_THREADS

python script.py   # NumPy will use exactly 8 threads
```

**Oversubscription trap:** `multiprocessing.Pool()` uses `os.cpu_count()` (hardware cores), not your LSF allocation. ThreadPool workers calling multi-threaded NumPy → threads multiply! Disable NumPy threading when using your own pool.

---

## 11. Common Formulas Quick Reference

### Amdahl's Law

```
S(p) = 1 / ((1 - F) + F/p)

S_max = 1 / (1 - F)      # as p → ∞

F = p(1 - 1/S(p)) / (p - 1)   # solve for F from measured speedup

Efficiency E(p) = S(p) / p
```

**Memory trick:** if speedup saturates at value X → S_max = X → F = 1 - 1/X

### Speedup

```
S(p) = T(1) / T(p)    (wall-clock time only, never CPU time)
```

### MFLOP/s

```
MFLOP/s = num_FLOPs / (time_seconds × 1,000,000)
```

### Bandwidth

```
Bandwidth = bytes_transferred / time_seconds
```

### Simple Reduction

```
Time = N/T + T
Optimal T = sqrt(N)
Max speedup = sqrt(N) / 2
```

### Binary Tree Reduction

```
Levels = ceil(log2(N))
Max speedup = N / log2(N)
```

### CUDA Grid Dimensions

```
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block   # always round up
```

### Memory Footprint

```
bytes_per_row = sum(dtype_bytes for each column)
max_chunk_rows = available_bytes / bytes_per_row
```

**dtype sizes:** uint8=1, int16=2, int32=4, float32=4, int64=8, float64=8

### GPU Transfer Bandwidth

```
bandwidth = total_MB_transferred / total_seconds   (from nsys output)
```

### FLOP/s from Line Profiler

```
FLOP/s = FLOPs_per_iteration × n_iterations / total_time_seconds
```

### Scaling from Profiler to Production

```
production_time ≈ percall × expected_ncalls  (for scaling functions)
               + fixed_overhead              (for fixed-cost functions)
```

---

## 12. Top 10 Pitfalls (one-liner each)

1. **rusage memory is per core, not total** — divide total needed by number of cores: 16 GB on 8 cores = `rusage[mem=2GB]`

2. **`-w done()` requires ALL jobs to succeed** — if any job is in EXIT state, the dependent job never starts; use `-w ended()` for cleanup jobs

3. **Numba transfers ALL args HtoD AND DtoH** — calling a kernel with k NumPy arrays does 2k transfers; optimal is 1 HtoD per input + 1 DtoH per output

4. **No-loop vectorization is not always faster** — large intermediate arrays exhaust cache; one-loop can beat no-loop for large N (profile to find crossover)

5. **CPU and CUDA have opposite optimal array layouts** — CPU: channels last (HWC) for sequential inner loop; CUDA: channels first (CHW) for coalesced warp access

6. **Always warm up JIT-compiled functions before timing** — first call includes compilation; subsequent calls are fast

7. **Pool() uses hardware core count, not LSF allocation** — set OMP/MKL/OPENBLAS_NUM_THREADS or you'll oversubscribe; ThreadPool + multi-threaded NumPy = thread count multiplies

8. **`done(jobname)` is by name, not ID** — if a second array with the same name exists, behavior is unexpected; `-w ended(jobname)` requires any-exit including failures

9. **Parallel reduction requires associativity** — `abs(x+y)` is commutative but not associative: `|(1+2)+(-3)|=0 ≠ |1+(2+(-3))|=2`; test with a concrete counterexample

10. **float16 resolution is relative** — at value 10000, smallest representable increment is `10000 × 0.001 = 10`; `float16(10000) + 1 == 10000`
