# Week 3 — Cache Effects & Efficient Data Storage

## Overview

Week 3 covers the **memory hierarchy** and how it affects computational performance. The central insight is that FLOP/s (floating point operations per second) is not constant — it degrades as data grows beyond each cache level. The lecture uses a motivating example of an 18K x 18K astronomical image to show why accessing data in the wrong order can be 12x slower. The second half covers **Blosc compression**, showing that for I/O-bound workloads, compressing data before writing it can actually make reads and writes faster.

Topics:
- What performance actually means (wall time vs CPU time vs FLOP/s)
- The memory hierarchy: registers, L1, L2, L3 cache, RAM, virtual memory
- Cache lines and spatial locality
- Row vs column access patterns in NumPy (C-order arrays)
- Blosc: compressed I/O that can outperform uncompressed I/O

---

## Theory & Concepts

### Measuring Performance

Three distinct ways to measure performance:

| Metric | Definition | Use |
|---|---|---|
| Wall-clock time | How long you waited (`time() - t`) | Always measure this |
| CPU time | How much the CPU actually worked | Less than wall time for single-threaded; can be more than wall time for multi-core (summed across cores) |
| FLOP/s | Floating point operations per second | Measures computational throughput |

```
FLOP/s = #FLOPs / wall_time
```

There is no automatic tool to count FLOPs — you must read the code and count manually.

**Important:** FLOP/s is not constant. The fantasy model assumes constant throughput regardless of array size. The reality is a staircase: performance drops sharply each time the working set exceeds a cache level.

To measure accurately with small operations, repeat many times and divide:
```python
t = time()
for _ in range(100):
    s = sum(x)
t = (time() - t) / 100
```

### Memory Hierarchy

From fastest/smallest to slowest/largest:

```
Registers  (bytes, ~0.25 ns)
    |
L1 Cache   (~32 KB per core, ~1 ns)
    |
L2 Cache   (~256 KB – 1 MB per core, ~4 ns)
    |
L3 Cache   (~8–20 MB shared, ~10 ns)
    |
RAM        (GBs, ~100 ns)
    |
Virtual Memory / Disk  (TBs, ms)
```

Key numbers from the lecture (Hennesy & Patterson):
- Processor at 2 GHz: **0.5 ns per operation**
- Memory (RAM) access: **100 ns per operation** — a 200x gap

This gap is why caches exist and why cache-aware programming matters.

**On the HPC cluster (Intel Xeon Gold 6126):**
- L1d cache: 32 KB
- L2 cache: 1024 KB (1 MB)
- L3 cache: 19712 KB (~19.25 MB)

Read cache sizes on the cluster with:
```bash
lscpu | grep cache
```

### Cache Lines

When the CPU fetches data from RAM, it does not fetch a single value — it fetches an entire **cache line** (typically 64 bytes = 8 float64 values). The entire cache line is loaded into L1 cache.

This is the mechanism behind spatial locality:
- If you access `x[0]`, the hardware also loads `x[1]` through `x[7]` for free.
- The next 7 accesses are instant cache hits.
- This only helps if you access memory **sequentially**.

**Cache hit:** data found in cache — fast.
**Cache miss:** data not in cache — must fetch from the next level (L2, L3, or RAM) — slow.

### Spatial Locality vs Temporal Locality

- **Spatial locality:** accessing memory addresses near each other (e.g., sequential row access). Benefits from cache line prefetching.
- **Temporal locality:** accessing the same memory address repeatedly in a short time. Benefits from data staying warm in cache.

### Row vs Column Access in NumPy

NumPy arrays are **row-major (C order)** by default. A 2D array `mat[i, j]` stores row `i` contiguously in memory:

```
Row 0: mat[0,0], mat[0,1], mat[0,2], ...   <- contiguous in memory
Row 1: mat[1,0], mat[1,1], mat[1,2], ...   <- contiguous in memory
```

**Row access `mat[0, :]`:** reads contiguous memory. Each cache line load brings in multiple useful values. Very cache-friendly.

**Column access `mat[:, 0]`:** reads every `SIZE`-th element in memory. Each access is to a different cache line. For large matrices, every access is a cache miss.

**Real measured difference from the lecture (18K x 18K image):**
```python
# Row-wise downsampling (fast):  1.669 seconds
small[i, :] = im[i*2, :] + im[i*2+1, :]

# Column-wise downsampling (slow):  20.15 seconds
small[:, i] = im[:, i*2] + im[:, i*2+1]
```

That is a **12x slowdown** from column access alone.

---

## Mathematical Formulas

**MFLOP/s** (mega floating point operations per second):
```
MFLOP/s = #operations / (time_seconds * 1_000_000)
```

For `2 * mat[0, :]` with `SIZE` elements: `#operations = SIZE` (one multiply per element).

**Matrix size in bytes** (float64 = 8 bytes):
```
size_bytes = SIZE * SIZE * 8
```

**Bandwidth** (bytes transferred per second):
```
Bandwidth = bytes_transferred / time_seconds
```

**Effect of cache on performance:**
- When working set fits in L1: highest throughput (close to theoretical peak)
- When working set fits in L2 only: ~4x slower
- When working set fits in L3 only: ~10x slower
- When working set spills to RAM: ~100x slower than peak

The performance vs array size plot is a **staircase on a loglog plot**, with steps aligned to L1, L2, and L3 cache boundaries.

---

## Key Code Examples

### cache.py — Row vs Column Access Timing

```python
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

n = 100
SIZE_LIST = np.logspace(1, 4, n)
matrix_size = SIZE_LIST**2 * 8   # size in bytes (float64)
MLOP_S_row = np.zeros(n)
MLOP_S_col = np.zeros(n)

for i, SIZE in enumerate(SIZE_LIST):
    mat = np.random.rand(int(SIZE), int(SIZE))
    operations = SIZE

    # Column access (cache-unfriendly: strides across rows)
    start = perf_counter()
    double_column = 2 * mat[:, 0]
    end = perf_counter()
    MLOP_S_col[i] = operations / (end - start) / 1000000

    # Row access (cache-friendly: sequential memory)
    start = perf_counter()
    double_row = 2 * mat[0, :]
    end = perf_counter()
    MLOP_S_row[i] = operations / (end - start) / 1000000

# Cache sizes for the HPC node used in cache.py
L1 = 512 * 1024       # 512 KB
L2 = 16 * 1024**2     # 16 MB
L3 = 22 * 1024**2     # 22 MB

plt.subplots(1, 1, figsize=(10, 6))
plt.plot(matrix_size, MLOP_S_row, color='blue', label='row')
plt.plot(matrix_size, MLOP_S_col, color='orange', label='col')
plt.axvline(L1, linestyle='--', label='L1')
plt.axvline(L2, linestyle='--', label='L2')
plt.axvline(L3, linestyle='--', label='L3')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig("cache.jpg")
```

Key points:
- x-axis: matrix size in bytes (logscale)
- y-axis: MFLOP/s (logscale)
- Vertical dashed lines mark L1/L2/L3 cache boundaries
- Row performance stays high; column performance drops as matrix grows past each cache level

### Solution: Timing with Repetitions (from solutions)

```python
from time import perf_counter as time
import numpy as np

SIZE = 100
n_repeat = int(1e3)
mat = np.random.rand(SIZE, SIZE)

trow = time()
for _ in range(n_repeat):
    mat[0, :] * 1.01
trow = time() - trow

tcol = time()
for _ in range(n_repeat):
    mat[:, 0] * 1.01
tcol = time() - tcol

print('trow =', trow / n_repeat)
print('tcol =', tcol / n_repeat)
```

### Solution: Sweeping SIZE with logspace

```python
ns = np.round(np.logspace(1, 4.5, 30))
trows, tcols = [], []
n_repeat = int(1e3)

for n in ns:
    n = int(n)
    mat = np.random.rand(n, n)
    trow = time()
    for _ in range(n_repeat):
        mat[0, :] * 1.01
    trow = (time() - trow) / n_repeat
    tcol = time()
    for _ in range(n_repeat):
        mat[:, 0] * 1.01
    tcol = (time() - tcol) / n_repeat
    trows.append(trow)
    tcols.append(tcol)
```

---

## Plotting Performance (loglog plots)

The canonical plot for cache effects is MFLOP/s vs data size on a loglog scale, with vertical lines marking cache boundaries:

```python
import matplotlib.pyplot as plt
import numpy as np

# After collecting MLOP_S_row, MLOP_S_col, ns arrays:
matrix_sizes_kb = np.array(ns)**2 * 8 / 1024  # float64 bytes -> KB

plt.figure(figsize=(10, 6))
plt.loglog(matrix_sizes_kb, MLOP_S_row, label='Row access')
plt.loglog(matrix_sizes_kb, MLOP_S_col, label='Col access')

# Mark cache boundaries (Intel Xeon Gold 6126)
plt.axvline(32,    linestyle='--', color='gray', label='L1 (32 KB)')
plt.axvline(1024,  linestyle='--', color='gray', label='L2 (1 MB)')
plt.axvline(19712, linestyle='--', color='gray', label='L3 (19.7 MB)')

plt.xlabel('Matrix size (KB)')
plt.ylabel('MFLOP/s')
plt.legend()
plt.title('Row vs Column Access Performance')
plt.savefig('cache_plot.png')
```

**What to expect in the plot:**
- Row access: relatively flat (high throughput) across all sizes
- Column access: staircase downward — drops at L1, L2, L3 boundaries
- Performance ratio (row/col) can reach 10–100x for large matrices
- When array fits in L1: both access patterns perform similarly (the entire row fits in cache regardless)

---

## Blosc API Reference

Blosc is a meta-compressor that is often **faster than raw I/O** for large arrays because it reduces the number of bytes that must be transferred from disk. This is the I/O-bound vs compute-bound tradeoff.

```python
import blosc
import numpy as np

# Pack a NumPy array into compressed bytes
packed = blosc.pack_array(arr, cname='lz4')   # returns bytes object

# Write to file
with open('data.bl', 'wb') as f:
    f.write(packed)

# Read back
with open('data.bl', 'rb') as f:
    packed = f.read()

# Unpack back to NumPy array
arr = blosc.unpack_array(packed)
```

**Available compression algorithms (`cname`):**

| Algorithm | Speed | Compression ratio | Use case |
|---|---|---|---|
| `lz4` | Very fast | Moderate | Default — good balance |
| `lz4hc` | Slower compress, fast decompress | Better | When write speed matters less |
| `zstd` | Slower than lz4 | Better | Best ratio, higher CPU cost |
| `blosclz` | Fast | Moderate | Blosc native |
| `snappy` | Fast | Low | Speed priority |

### blosc_ex.py — Full Read/Write Comparison

```python
import sys
import os
import blosc
from time import perf_counter
import numpy as np
from functools import wraps

def time_it(func):
    @wraps(func)
    def wrapper(*args):
        start = perf_counter()
        result = func(*args)
        end = perf_counter()
        print(f"{end - start}")
        return result
    return wrapper

@time_it
def write_numpy(arr, file_name):
    np.save(f"{file_name}.npy", arr)
    if hasattr(os, 'sync'):
        os.sync()   # flush OS I/O buffers for fair comparison

@time_it
def write_blosc(arr, file_name, cname="lz4"):
    b_arr = blosc.pack_array(arr, cname=cname)
    with open(f"{file_name}.bl", "wb") as w:
        w.write(b_arr)
    if hasattr(os, 'sync'):
        os.sync()

@time_it
def read_numpy(file_name):
    return np.load(f"{file_name}.npy")

@time_it
def read_blosc(file_name):
    with open(f"{file_name}.bl", "rb") as r:
        b_arr = r.read()
    return blosc.unpack_array(b_arr)

def main():
    n = int(sys.argv[1])
    A = np.zeros((n, n, n), dtype='uint8')
    write_numpy(A, 'write')
    write_blosc(A, 'write')
    read_numpy('write')
    read_blosc('write')

if __name__ == '__main__':
    main()
```

`os.sync()` is called after each write to flush the OS page cache, ensuring the benchmark measures actual disk I/O and not cached writes.

### blosc_quiz.py — Three Data Patterns

```python
def zero(n):
    A = np.zeros((n, n, n), dtype='uint8')
    write_read(A)                  # All zeros: extreme compressibility

def tiled(n):
    tiled_array = np.tile(
        np.arange(256, dtype='uint8'),
        (n // 256) * n * n,
    ).reshape(n, n, n)
    write_read(tiled_array)        # Repeating pattern: high compressibility

def random(n):
    random_array = np.random.randint(
        low=1, high=256, size=(n, n, n), dtype='uint8'
    )
    write_read(random_array)       # Maximum entropy: nearly incompressible
```

---

## Exercise Highlights

### Exercise 1: Cache Effects

The exercises build the full cache visualization workflow:

1. **Time row vs column access** for SIZE=100 with 1000 repetitions.
2. **Submit a batch job** to the `hpc` queue targeting an Intel Xeon Gold 6126/6142/6226R node, 1 core. This is critical — cache sizes vary by hardware and noise on shared systems distorts results.
3. **Sweep SIZE from 10^1 to 10^4.5** using `np.logspace(1, 4.5, 30)`.
4. **Plot MFLOP/s vs matrix size (KB) on loglog axes** with vertical lines at L1/L2/L3 boundaries. Also plot the ratio (row MFLOP/s) / (col MFLOP/s) to make the divergence obvious.
5. **Row vector experiment:** use `mat = np.random.rand(1, SIZE)` and sweep SIZE from 10^2 to 10^8 with 100 repetitions. This isolates spatial locality from the matrix structure and gives a clean staircase plot aligned with cache boundaries.

**Key findings from the solutions:**
- When the array fits in L1: row and column perform similarly
- As the matrix grows past L1: column access degrades because it accesses nearly the entire matrix (all cache lines), while the single row stays in L1
- The performance ratio widens through L2 and L3 transitions
- The staircase in MFLOP/s on the row vector plot aligns precisely with cache boundaries

### Exercise 2: Efficient Data Storage with Blosc

The exercises benchmark write and read times for three array types at n = 256, 512, 1024 (3D arrays of shape `n x n x n`, `dtype='uint8'`):

1. **Zero array** — all zeros. Trivially compressible. Blosc achieves near-zero compressed size.
2. **Tiled array** — repeating `0..255` pattern. Highly compressible.
3. **Random array** — random integers 1–255. High entropy, nearly incompressible.

**When to use Blosc:**
- Use it when data is compressible (zeros, structured patterns, repeated values) AND the operation is I/O-bound
- For random data, compression overhead exceeds I/O savings — raw numpy is faster
- Blosc wins when: `time_compress + time_write_compressed < time_write_raw` AND `time_read_compressed + time_decompress < time_read_raw`

**lz4 vs zstd:**
- `zstd` achieves better compression ratios than `lz4`, especially for tiled/structured data
- The cost: `zstd` is slower to compress (higher CPU time for writes)
- Decompression speed is similar; read times may still be competitive or better due to smaller files

**File sizes to compare:**
- `write.npy`: always `n^3` bytes (uncompressed)
- `write.bl`: depends on compressibility — can be orders of magnitude smaller for zero/tiled arrays

---

## Key Takeaways

1. **FLOP/s is not constant.** It degrades as your working set grows past each cache level. The performance profile is a staircase, not a flat line.

2. **Cache lines are 64 bytes.** When you access one element, the hardware fetches 8 adjacent float64 values. Sequential (row) access reuses this; strided (column) access wastes it.

3. **NumPy is row-major.** Always prefer `mat[i, :]` over `mat[:, j]` for performance. The difference can be 12x or more for large matrices (as shown with the 18K image: 1.67s vs 20.15s).

4. **Measure on the actual hardware.** Cache sizes differ by CPU. Always benchmark from a batch job on the target node — don't trust results from a shared login node.

5. **Use `np.logspace` + loglog plots** to visualize cache effects. The steps in the MFLOP/s curve align with L1, L2, L3 boundaries and are immediately visible on a log scale.

6. **Blosc is not always faster.** It wins when data is compressible and I/O is the bottleneck. For random (high-entropy) data, the compression overhead makes it slower. Always benchmark with your actual data.

7. **`os.sync()` is required for fair I/O benchmarks.** Without it, the OS page cache can make writes appear much faster than they are, giving misleading results.

8. **Wall time is what users experience.** CPU time and FLOP/s are diagnostic tools — wall time is the number you ultimately want to minimize.

---

## Common Misconception: Stride-8 vs Sequential Access

**Intuition that seems right but isn't:**
> "stride-8 reads fewer elements → less work → must be faster"

**Why it's wrong — the cache line trap:**

A cache line is 64 bytes = 8 float64 values. The CPU always loads a full cache line — you cannot load just one float64.

```
Sequential x[::1] summing 10M elements:
  Cache lines loaded = 10M / 8 = 1,250,000
  Elements used per cache line = 8  → 100% utilisation

Stride-8 x[::8] summing 1.25M elements:
  Cache lines loaded = 1.25M / 1 = 1,250,000
  Elements used per cache line = 1  → 12.5% utilisation
```

**Same number of cache line loads. Same memory traffic (80 MB). But stride-8 does 8× less arithmetic.**

So why is sequential still faster? Two reasons:

**1. SIMD vectorisation.** Sequential access lets the CPU process 4–8 float64s per instruction (AVX2). Stride-8 breaks vectorisation — each element is an isolated scalar load processed one at a time.

**2. Hardware prefetcher.** The CPU detects sequential patterns and pre-loads the next cache line before you request it, hiding latency. Stride-8 is harder to prefetch effectively.

**The full picture:**

| Pattern | Memory traffic | Arithmetic | Vectorised | Faster? |
|---|---|---|---|---|
| `x[::1]` | 80 MB | 10M ops | ✅ SIMD | ✅ |
| `x[::8]` | 80 MB | 1.25M ops | ❌ scalar | ❌ |

**Bandwidth** = maximum bytes per second the memory bus can deliver (e.g. 50 GB/s). It's a fixed pipe. Both patterns saturate it equally — so bandwidth alone doesn't explain the difference. The advantage of sequential access comes from SIMD and prefetching on top of equal memory traffic.
