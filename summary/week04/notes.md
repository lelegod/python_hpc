# Week 4 — Profiling and High-Performance NumPy

## Overview

Week 4 covers two tightly linked topics: **NumPy internals** (arrays, strides, views vs copies, broadcasting) and **profiling** (finding where time is actually spent, then systematically eliminating that bottleneck). The central case study is the Haversine distance matrix — rewritten from two nested Python loops down to zero loops, with profiling guiding every step.

The lecture is dated 25.02.2025. This week picks up from Week 3 (caches and memory hierarchy) and connects those low-level ideas directly to NumPy's memory model.

---

## Theory & Concepts

### NumPy Array Internals

A NumPy array is a Python object wrapping a flat, contiguous block of memory. The object stores metadata separately from the data itself:

- `shape` — the logical dimensions (e.g. `(2, 3)`)
- `dtype` — element type (e.g. `int32`, `float64`) and therefore element size in bytes
- `strides` — how many **bytes** to step in memory to move one position along each axis
- `data` — a pointer to the start of the raw buffer

For `a = np.array([[1,2,3],[4,5,6]])` with `dtype=int32` (4 bytes/element):
- `shape = (2, 3)`, `strides = (12, 4)`
- To access `a[i, j]`: byte offset = `i * 12 + j * 4`

For the same data stored column-major (`order='F'`, Fortran style):
- `strides = (4, 8)` — moving down a column steps only 4 bytes, moving across a row steps 8 bytes

**Cache efficiency:** The direction with the smallest stride is most cache-friendly because elements are contiguous in memory. For C-order (row-major) arrays, iterating along the last axis (columns) is fastest.

**Stride quiz answers from lecture:**
- `np.ones((2,3)).strides` → `(24, 8)` (float64, 8 bytes/element)
- `np.ones((4,2,5)).strides` → `(80, 40, 8)`
- `a = np.ones((3,4)); b = a[1, :]` → `b.strides = (8,)` — a 1D view into row 1
- `a = np.ones((5,6)); b = a[1, :]` → `b.strides = (48,)` (stride inherits from a's column stride)

**Indexing formula:**
```python
a = np.zeros((32, 3, 64, 64))   # strides: (98304, 32768, 512, 8)
# a[2, 1, 4, 7] lives at byte offset:
# 2*98304 + 1*32768 + 4*512 + 7*8
```

### Views vs Copies

Many NumPy operations return a **view** — a new array object pointing to the **same underlying data buffer**, with different shape/strides metadata. No data is copied; mutations to the view affect the original.

| Operation | View or Copy? |
|---|---|
| `a.T` (transpose) | View — strides are reversed |
| `a[1, :]` (row slice) | View |
| `a[:, ::−1]` (reverse columns) | View — strides become negative |
| `a.reshape(...)` on C-order array | View **if possible**, otherwise copy |
| `a.reshape(...)` on F-order array | Copy (because reshape needs row-major traversal) |
| Fancy indexing `a[[0, 2], :]` | Always a copy |

Key check: `np.shares_memory(a, b)` returns `True` if they share a buffer.

**Gotcha with reshape:** Reshaping a Fortran-order array cannot be expressed as a stride change, so NumPy silently makes a copy and reorders data to row-major. The resulting `b` is C-contiguous even though `a` was F-contiguous.

**Diagonal via strides trick:**
```python
diag = a.reshape(-1)[::4]   # stride = 16 bytes for a 3x3 int32 array
# Gives array([1, 5, 9]) — the diagonal as a view
```

**Flipping and rotating via indexing (Mentimeter quiz answers):**
- Flip horizontally: `a[:, ::-1]` — correct
- Rotate 90 degrees: `a.T[:, ::-1]` — correct

### Broadcasting Rules

Broadcasting lets NumPy operate on arrays with different shapes without copying data. The rules are applied from the **trailing (rightmost) dimension** inward:

1. If arrays have different numbers of dimensions, prepend 1s to the shape of the smaller array.
2. Along each dimension: sizes must be **equal** OR one of them must be **1**.
3. A dimension of size 1 is "stretched" (virtually repeated) to match the other.
4. If neither condition holds, NumPy raises a `ValueError`.

**Shape inference examples (from Mentimeter quizzes):**

| a.shape | b.shape | Result shape | Compatible? |
|---|---|---|---|
| `(2, 3)` | `(3,)` | `(2, 3)` | Yes — `(3,)` becomes `(1,3)` |
| `(2, 1, 3)` | `(4, 1)` | `(2, 4, 3)` | Yes |
| `(2, 1, 3)` | `(8, 4, 1)` | Won't broadcast | No — 2 vs 8 conflict |
| `(2, 3)` | `(2,)` | Won't broadcast | No — need `b[:, None]` to fix |

**Fixing incompatible shapes:** When `a.shape = (2,3)` and `b.shape = (2,)`, broadcasting fails because the trailing dimensions 3 and 2 are not equal and neither is 1. The fix is `b[:, None]` which makes `b.shape = (2, 1)`, then broadcasting gives `(2, 3)`. From the Mentimeter: `a[:, :, None]` makes `a` broadcastable with `b` in that scenario.

**None / np.newaxis:** Inserting `None` (or `np.newaxis`) adds a size-1 dimension at that position, enabling broadcasting along a new axis.

---

## Key Code Examples

### bradscasting1.py — Row Standardization

Subtract a per-feature mean and divide by per-feature std from a 2D data matrix. Broadcasting handles the (n, d) - (d,) subtraction automatically:

```python
import numpy as np

def standardize_rows(data, mean, std):
    return (data - mean) / std

if __name__ == "__main__":
    data = np.array([[1,2,3], [4,5,6]])
    mean = np.array([.5,1,3])
    std = np.array([1,2,3])
    print(standardize_rows(data, mean, std))
    # Output:
    # [[0.5  0.5  0. ]
    #  [3.5  2.   1. ]]
```

`data` has shape `(2, 3)`, `mean` and `std` have shape `(3,)`. Broadcasting aligns them on the trailing dimension: `(2,3) - (3,)` works because `(3,)` is treated as `(1,3)` and stretched to `(2,3)`.

### bradscasting2.py — Outer Product

Compute the outer product of two 1D vectors without `np.outer`. The key is `x[:, None]` which reshapes `x` from `(n,)` to `(n, 1)`, then multiplies against `y` of shape `(m,)` to get `(n, m)`:

```python
import numpy as np

def outer(x, y):
    return x[:, None] * y

if __name__ == "__main__":
    x = np.array([1,3])
    y = np.array([1,2,3])
    print(outer(x, y))
    # Output:
    # [[1 2 3]
    #  [3 6 9]]
```

Shape trace: `x[:, None]` is `(2, 1)`, `y` is `(3,)` treated as `(1, 3)` → result is `(2, 3)`.

### bradscasting3.py — 1D Distance Matrix

Compute the pairwise absolute differences between two vectors, producing an n×m matrix. Same `[:, None]` trick:

```python
import numpy as np

def distmat_1d(x, y):
    return abs(x[:, None] - y)

if __name__ == "__main__":
    x = np.array([1,3])
    y = np.array([1,2,3])
    print(distmat_1d(x, y))
    # Output:
    # [[0 1 2]
    #  [2 1 0]]
```

`x[:, None]` is `(2,1)`, `y` is `(3,)` → subtraction gives `(2,3)`, then `abs` is element-wise.

---

## The Haversine Distance Matrix

The **Haversine formula** computes the great-circle distance between two points on a sphere given their latitude/longitude coordinates. For points `p1[i]` and `p2[j]` (in radians):

```
dsin2 = sin(0.5 * (p1[i] - p2[j]))^2       # element-wise for [lat, lon]
cosprod = cos(p1[i, 0]) * cos(p2[j, 0])    # product of latitude cosines
a = dsin2[0] + cosprod * dsin2[1]
distance = 2 * arctan2(sqrt(a), sqrt(1-a)) * 6371  # km
```

`haversine.py` implements three versions of `distance_matrix` and leaves two commented out for reference. The active (no-loop) version:

```python
def distance_matrix(p1, p2):
    p1, p2 = np.radians(p1), np.radians(p2)

    # No loop — fully vectorized with broadcasting
    dsin2 = np.sin(0.5 * (p1[:, None, :] - p2[None, :, :])) ** 2
    cosprod = np.cos(p1[:, None, 0]) * np.cos(p2[None, :, 0])
    a = dsin2[:, :, 0] + cosprod * dsin2[:, :, 1]
    D = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    D *= 6371  # Earth radius in km
    return D
```

**Shape analysis for the no-loop version:**
- `p1` has shape `(N, 2)`, `p2` has shape `(M, 2)`
- `p1[:, None, :]` → `(N, 1, 2)`, `p2[None, :, :]` → `(1, M, 2)`
- Subtraction broadcasts to `(N, M, 2)` — all pairwise lat/lon differences at once
- `dsin2` has shape `(N, M, 2)`
- `p1[:, None, 0]` → `(N, 1)`, `p2[None, :, 0]` → `(1, M)` → `cosprod` is `(N, M)`
- `a = dsin2[:,:,0] + cosprod * dsin2[:,:,1]` → `(N, M)`
- Final `D` is `(N, M)` — the full distance matrix

**Commented-out two-loop version (for comparison):**
```python
for i in range(len(p1)):
    for j in range(len(p2)):
        dsin2 = np.sin(0.5 * (p1[i] - p2[j])) ** 2
        cosprod = np.cos(p1[i, 0]) * np.cos(p2[j, 0])
        a = dsin2[0] + cosprod * dsin2[1]
        D[i, j] = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
```

**Commented-out one-loop version:**
```python
for i in range(len(p1)):
    dsin2 = np.sin(0.5 * (p1[i] - p2)) ** 2      # (M, 2) — whole row at once
    cosprod = np.cos(p1[i, 0]) * np.cos(p2[:, 0]) # (M,)
    a = dsin2[:, 0] + cosprod * dsin2[:, 1]        # (M,)
    D[i, :] = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
```

**Statistics on the result:**
```python
def distance_stats(D):
    assert D.shape[0] == D.shape[1]
    idx = np.triu_indices(D.shape[0], k=1)   # upper triangle, no diagonal
    distances = D[idx]
    return {'mean': float(distances.mean()), 'std': float(distances.std()),
            'max': float(distances.max()),   'min': float(distances.min())}
```

---

## Two-Loop vs One-Loop vs No-Loop

| Version | Strategy | Typical speedup vs two-loop |
|---|---|---|
| Two-loop | Plain Python loops, one scalar per iteration | baseline |
| One-loop | Outer loop over rows, inner computation is a NumPy row operation | ~80x faster |
| No-loop | Full broadcasting over 3D arrays | faster for small N, can be slower for large N |

The one-loop version is faster than two loops because:
- Each inner NumPy call operates on a whole row (M elements) rather than one scalar
- NumPy dispatches to C-level loops with SIMD instructions
- Python interpreter overhead is paid only N times instead of N*M times

The no-loop version is not always faster than the optimized one-loop because:
- It creates large intermediate arrays (`(N, M, 2)` for `dsin2`)
- For large N and M, these intermediates exceed cache size
- Memory bandwidth becomes the bottleneck, not compute

**Key insight from the exercise solutions:** The crossover between no-loop being faster and one-loop being faster happens approximately when the distance matrix no longer fits in L2 cache. This is a direct application of the Week 3 cache hierarchy material.

---

## Profiling Workflow

The exercises use profiling to guide optimization rather than guessing:

### Step 1 — cProfile (function-level)

Run the whole program under cProfile:
```bash
python -m cProfile -s cumulative points.py input.csv
```

This reveals which functions consume the most total time. After the two-loop version, `distance_matrix` dominates. After switching to one-loop, it becomes ~80x faster but is still the bottleneck.

### Step 2 — line_profiler (line-level)

For finer detail, decorate the function with `@profile` and run with `kernprof`:
```bash
kernprof -l -v points.py input.csv
```

Example output for the one-loop version (Intel Xeon Gold 6226R, 500 locations):
```
Line #   % Time  Line Contents
    27    52.4%   dsin2 = np.sin(0.5 * (p1[i] - p2)) ** 2
    28    21.0%   cosprod = np.cos(p1[i, 0]) * np.cos(p2[:, 0])
    30    14.4%   row = np.arctan2(np.sqrt(a), np.sqrt(1-a))
```

This points to two optimizations:
1. `np.cos(p2[:, 0])` is recomputed every loop iteration — pre-compute it once before the loop
2. `arctan2(sqrt(a), sqrt(1-a))` computes two square roots; use the identity `arcsin(sqrt(a))` instead to compute only one

Result: these optimizations give ~25% further speedup on top of the one-loop version.

### Step 3 — MFLOP/S measurement

To compare versions at different problem sizes, time them across N from 10 to 10,000 and plot performance in MFLOP/S vs problem size in KB. Including cache boundaries on the plot shows the cache-size crossover directly.

---

## Exercise Highlights

### Section 1 — Broadcasting

Three Autolab-graded exercises building up in complexity:

1. **standardize_rows** — `(data - mean) / std` with shape `(n,d) - (d,)`. Solved in one line without any NumPy functions other than arithmetic operators.

2. **outer product** — `x[:, None] * y` with shapes `(n,1) * (m,)` → `(n,m)`. The `[:, None]` reshape is the entire solution.

3. **distmat_1d** — `abs(x[:, None] - y)` with shapes `(n,1) - (m,)` → `(n,m)`. No loops, only `abs` permitted as a NumPy function.

### Section 2 — High Performance Haversine

Eight-step progression:
1. Write job script for HPC queue, single core, fixed CPU model
2. Profile with cProfile, identify `distance_matrix` as bottleneck
3. Rewrite with one loop — ~80x speedup observed
4. Re-profile to confirm improvement, observe shift in hotspots
5. Use `line_profiler` to find line-level bottlenecks within the loop
6. Apply two micro-optimizations: pre-compute cosines, replace arctan2 with arcsin — ~25% further gain
7. Eliminate all loops — may or may not be faster depending on problem size
8. Benchmark across problem sizes, plot MFLOP/S vs KB, overlay cache sizes to explain crossover

---

## Key Takeaways

1. **NumPy arrays are flat buffers plus metadata.** Shape, strides, and dtype are just numbers stored in the array object; no data is moved when you transpose, slice, or reshape (when possible).

2. **Strides control cache efficiency.** Iterating along the axis with the smallest stride (the last axis in C-order) is fastest because elements are contiguous in memory.

3. **Views share memory; copies do not.** Transpose and basic slicing give views. Fancy indexing always copies. `reshape` gives a view when the memory layout allows it, otherwise a copy.

4. **The `[:, None]` idiom is the key to 2D broadcasting.** Inserting a size-1 dimension converts a 1D vector into a column vector, enabling pairwise operations between two vectors to produce a matrix result.

5. **Broadcasting rules go right-to-left.** Prepend 1s, then check each pair: equal or one is 1. The output shape takes the max along each dimension.

6. **Vectorization eliminates Python loop overhead.** Python's interpreter overhead is O(N*M) for nested loops but O(1) for a single NumPy call — even if the underlying arithmetic is the same.

7. **More vectorization is not always faster.** Fully eliminating loops can create large intermediate arrays that exhaust cache. For the Haversine case, the no-loop version is only faster when the working set fits in L2 cache.

8. **Measure, don't guess.** cProfile finds the bottleneck function; line_profiler finds the bottleneck line; timing benchmarks across problem sizes reveal the memory vs compute crossover. Intuition about what should be fast is frequently wrong.
