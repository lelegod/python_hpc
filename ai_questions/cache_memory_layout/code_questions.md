# Cache Efficiency & Memory Layout — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Loop Order from Strides (4D Array)](#q1--loop-order-from-strides-4d-array)
- [Q2 — Row vs Column Traversal](#q2--row-vs-column-traversal)
- [Q3 — Strides of (3,4) float64 Array](#q3--strides-of-34-float64-array)
- [Q4 — Smallest Stride in (N,H,W,C) Array](#q4--smallest-stride-in-nhwc-array)
- [Q5 — ijk vs ikj Matrix Multiply Loop Order](#q5--ijk-vs-ikj-matrix-multiply-loop-order)
- [Q6 — Transpose Shares Memory, Reverses Strides](#q6--transpose-shares-memory-reverses-strides)
- [Q7 — Row vs Column Processing in Pool](#q7--row-vs-column-processing-in-pool)
- [Q8 — Sequential vs Stride-8 Access](#q8--sequential-vs-stride-8-access)
- [Q9 — CHW Channels-First Cache Inefficiency](#q9--chw-channels-first-cache-inefficiency)
- [Q10 — reshape(-1) and C-Order Flattening](#q10--reshape-1-and-c-order-flattening)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q11 — Strides After np.asfortranarray](#q11-strides-after-npasfortranarray)
- [Q12 — Loop Order for F-Order Array](#q12-loop-order-for-f-order-array)
- [Q13 — Strides of a 3D C-Order Array](#q13-strides-of-a-3d-c-order-array)
- [Q14 — Broadcasting Stride Trick](#q14-broadcasting-stride-trick)
- [Q15 — CUDA Thread Layout for C-Order Access](#q15-cuda-thread-layout-for-c-order-access)
- [Q16 — np.ascontiguousarray on a Slice](#q16-npascontiguousarray-on-a-slice)
- [Q17 — Axis=0 vs Axis=1 Sum Strides](#q17-axis0-vs-axis1-sum-strides)
- [Q18 — Transpose Flags](#q18-transpose-flags)
- [Q19 — Memory Access Pattern for Diagonal Sum](#q19-memory-access-pattern-for-diagonal-sum)
- [Q20 — Choosing Layout for a CUDA Reduction](#q20-choosing-layout-for-a-cuda-reduction)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q21 — Strides of Every-Other-Column Slice](#q21--strides-of-every-other-column-slice)
- [Q22 — Fancy Index vs Slice: View or Copy?](#q22--fancy-index-vs-slice-view-or-copy)
- [Q23 — reshape(-1) on F-Order Array](#q23--reshape-1-on-f-order-array)
- [Q24 — np.nditer with order='F' on C-Order Array](#q24--npnditer-with-orderf-on-c-order-array)
- [Q25 — Strides After np.expand_dims](#q25--strides-after-npexpand_dims)
- [Q26 — C_CONTIGUOUS Flag After Transpose and Copy](#q26--c_contiguous-flag-after-transpose-and-copy)
- [Q27 — Loop Order for 3D F-Order Array](#q27--loop-order-for-3d-f-order-array)
- [Q28 — np.broadcast_to Writability](#q28--npbroadcast_to-writability)
- [Q29 — Strides of int32 vs float64 Array](#q29--strides-of-int32-vs-float64-array)
- [Q30 — Identifying the Cache-Friendly Reduction Axis](#q30--identifying-the-cache-friendly-reduction-axis)

---

> Format: Each question shows Python/NumPy code to analyse for cache efficiency.
> Exam frequency: **Every exam**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#question-1)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — Loop Order from Strides (4D Array)

Given the following 4D array and loop:

```python
import numpy as np

a = np.random.rand(3, 5, 4, 25)
# strides: (600, 40, 8, 200) bytes  (note: not necessarily in this shape's default order)
# Suppose strides are manually assigned as (600, 40, 8, 200) via as_strided or similar.

result = 0.0
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        for k in range(a.shape[2]):
            for l in range(a.shape[3]):
                result += a[i, j, k, l]
```

The strides are `(600, 40, 8, 200)` bytes for axes `(i, j, k, l)` respectively. Which reordering of the loops gives the most cache-efficient traversal (innermost to outermost: smallest stride first)?

**A)** `i → j → k → l` (as written above)

**B)** `k → j → l → i` (innermost `k`, outermost `i`)

**C)** `l → k → j → i`

**D)** `j → l → k → i`

---

**Answer: B**

The most cache-efficient loop order places the axis with the **smallest stride in the innermost loop**. Strides in ascending order:

| Axis | Stride (bytes) |
|------|---------------|
| `k`  | 8             |
| `j`  | 40            |
| `l`  | 200           |
| `i`  | 600           |

So the optimal ordering is: innermost `k` (stride 8) → `j` (stride 40) → `l` (stride 200) → outermost `i` (stride 600).

- **A) Incorrect.** The innermost loop is over `l` (stride 200), not the smallest stride. Many cache misses result.
- **B) Correct.** Innermost `k` has stride 8 (one `float64` element), so every access is sequential in memory.
- **C) Incorrect.** Innermost `l` has stride 200 — far from optimal.
- **D) Incorrect.** Innermost `j` has stride 40, skipping over many cache lines per access.

---

## Q2 — Row vs Column Traversal

```python
import numpy as np

mat = np.random.rand(1000, 1000)

# Option A: row-major traversal
total_a = sum(mat[i, j] for i in range(1000) for j in range(1000))

# Option B: column-major traversal
total_b = sum(mat[i, j] for j in range(1000) for i in range(1000))
```

`mat` is a standard NumPy C-order (row-major) array. Which option is faster, and why?

**A)** Option A, because the innermost loop varies `j` (column index), giving sequential memory access along each row.

**B)** Option B, because varying `i` (row index) in the inner loop accesses column-major order, which is optimal for NumPy.

**C)** Both are equally fast because NumPy accesses elements the same way regardless of loop order.

**D)** Option B, because column-wise access uses SIMD instructions automatically.

---

**Answer: A**

NumPy stores arrays in **C (row-major) order** by default: elements in the same row are contiguous in memory. When you load `mat[i, 0]`, the CPU fetches an entire cache line covering `mat[i, 0]` through `mat[i, 7]` (for `float64`). Varying `j` in the inner loop reuses these cached elements.

- **A) Correct.** Inner loop over `j` → sequential row access → excellent cache utilisation.
- **B) Incorrect.** Inner loop over `i` → stride of 1000 × 8 = 8000 bytes per step → a cache miss on nearly every access.
- **C) Incorrect.** Loop order dramatically affects cache behaviour for large arrays.
- **D) Incorrect.** SIMD is unrelated to loop ordering here; column-major access remains cache-unfriendly.

---

## Q3 — Strides of (3,4) float64 Array

```python
import numpy as np

a = np.ones((3, 4))
print(a.strides)
```

What does this print?

**A)** `(4, 1)`

**B)** `(32, 8)`

**C)** `(8, 32)`

**D)** `(12, 4)`

---

**Answer: B**

`np.ones` produces a `float64` array by default (8 bytes per element). For a C-order array with shape `(3, 4)`:

- **Row stride (axis 0):** moving to the next row skips 4 elements × 8 bytes = **32 bytes**
- **Column stride (axis 1):** moving to the next column skips 1 element × 8 bytes = **8 bytes**

So `a.strides == (32, 8)`.

- **A) Incorrect.** These would be element counts, not byte offsets.
- **B) Correct.** `(32, 8)` bytes — row stride, then column stride.
- **C) Incorrect.** `(8, 32)` would be the strides of the **transpose** `a.T`.
- **D) Incorrect.** `(12, 4)` would correspond to `float32` with 3 columns — neither matches here.

---

## Q4 — Smallest Stride in (N,H,W,C) Array

```python
import numpy as np

images = np.zeros((100, 64, 64, 3))  # shape: (N, H, W, C)
print(images.strides)
```

Which axis has the **smallest stride**, and what is its value in bytes?

**A)** Axis 0 (N), stride = 98304

**B)** Axis 1 (H), stride = 1536

**C)** Axis 2 (W), stride = 24

**D)** Axis 3 (C), stride = 8

---

**Answer: D**

For a C-order `float64` array with shape `(100, 64, 64, 3)`, strides are computed right-to-left:

| Axis | Meaning | Stride calculation          | Stride (bytes) |
|------|---------|----------------------------|----------------|
| 3    | C       | 1 × 8                       | **8**          |
| 2    | W       | 3 × 8                       | 24             |
| 1    | H       | 64 × 3 × 8                  | 1536           |
| 0    | N       | 64 × 64 × 3 × 8             | 98304          |

The full output is `(98304, 1536, 24, 8)`. Axis 3 (channels) has the smallest stride of **8 bytes**.

- **A) Incorrect.** 98304 is the largest stride (axis 0, N).
- **B) Incorrect.** 1536 is the stride for axis 1 (H).
- **C) Incorrect.** 24 is the stride for axis 2 (W).
- **D) Correct.** Axis 3 (C) has stride 8 — adjacent channel values are contiguous in memory.

---

## Q5 — ijk vs ikj Matrix Multiply Loop Order

```python
import numpy as np

N = 512
A = np.random.rand(N, N)
B = np.random.rand(N, N)
C = np.zeros((N, N))

# Version A: ijk loop order
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i, j] += A[i, k] * B[k, j]

# Version B: ikj loop order
for i in range(N):
    for k in range(N):
        for j in range(N):
            C[i, j] += A[i, k] * B[k, j]
```

Which version is faster and why?

**A)** Version A (ijk), because iterating over `j` then `k` in the inner loops gives better access to `C`.

**B)** Version B (ikj), because the innermost loop over `j` accesses `B[k, j]` and `C[i, j]` sequentially (row-wise).

**C)** Both are equivalent in speed; the CPU reorders memory accesses automatically.

**D)** Version A (ijk), because `B[k, j]` is column-wise which the cache prefetcher handles well.

---

**Answer: B**

In a C-order array, row elements are contiguous. Analysing the innermost loop variable in each version:

| Version | Inner var | `A[i,k]` access | `B[k,j]` access | `C[i,j]` access |
|---------|-----------|-----------------|-----------------|-----------------|
| A (ijk) | `k`       | row-wise (good) | **column-wise** (bad) | fixed `j` (ok) |
| B (ikj) | `j`       | fixed `k` (ok)  | **row-wise** (good)   | row-wise (good) |

Version B keeps `k` fixed in the middle loop and strides over `j` in the innermost loop — both `B[k, j]` and `C[i, j]` are accessed sequentially. Version A's innermost `k` loop accesses `B[k, j]` down a column (stride = N × 8 bytes), causing frequent cache misses.

- **A) Incorrect.** ijk accesses `B` column-wise in the innermost loop — this is cache-unfriendly.
- **B) Correct.** ikj keeps the innermost loop sequential for both `B` and `C`.
- **C) Incorrect.** The hardware prefetcher helps with sequential patterns but cannot eliminate strided-access penalties at scale.
- **D) Incorrect.** Column-wise access to `B` is not handled efficiently by prefetching for large `N`.

---

## Q6 — Transpose Shares Memory, Reverses Strides

```python
import numpy as np

a = np.ones((3, 4))   # strides: (32, 8)
b = a.T

print(np.shares_memory(a, b))
print(b.strides)
```

What does this code print?

**A)**
```
False
(8, 32)
```

**B)**
```
True
(32, 8)
```

**C)**
```
True
(8, 32)
```

**D)**
```
False
(32, 8)
```

---

**Answer: C**

`a.T` returns a **view** — it points to the same underlying data buffer as `a`, so `np.shares_memory(a, b)` is `True`. Transposing simply **reverses the strides**: if `a.strides == (32, 8)` then `b.strides == (8, 32)`.

No data is copied; NumPy reinterprets the same bytes with swapped stride values, making `b` appear column-major (Fortran-order) relative to the original layout.

- **A) Incorrect.** `b` is a view, so memory is shared.
- **B) Incorrect.** Strides are reversed by transpose, not kept the same.
- **C) Correct.** `True` and `(8, 32)`.
- **D) Incorrect.** Both values are wrong — memory is shared and strides are reversed.

---

## Q7 — Row vs Column Processing in Pool

```python
import numpy as np
from multiprocessing import Pool

a = np.random.rand(1000, 1000)

def process_row(i):
    return a[i, :].sum()      # accesses a full row

def process_col(j):
    return a[:, j].sum()      # accesses a full column

# Parallel row processing
with Pool(4) as p:
    row_results = p.map(process_row, range(1000))

# Parallel column processing
with Pool(4) as p:
    col_results = p.map(process_col, range(1000))
```

Which is faster, and why?

**A)** Column processing, because columns are processed independently with no false sharing.

**B)** Row processing, because `a[i, :]` accesses a contiguous block of memory (a full row), giving sequential cache access.

**C)** Both are equally fast; multiprocessing eliminates cache effects by isolating workers in separate processes.

**D)** Column processing, because strided access allows the OS to page in data more efficiently.

---

**Answer: B**

In C-order storage, `a[i, :]` (a row) occupies a **contiguous** region of 1000 × 8 = 8000 bytes. Accessing it in order generates at most ~125 cache-line loads for the whole row.

`a[:, j]` (a column) has elements spaced 1000 × 8 = 8000 bytes apart. Each element access loads a new cache line, discarding the rest of it, resulting in ~1000 cache misses per column.

Multiprocessing uses separate address spaces, but each worker process still experiences the same cache-miss penalty when accessing strided data from its own copy.

- **A) Incorrect.** False sharing is a concern with shared memory (e.g., `multiprocessing.Array`), not separate processes.
- **B) Correct.** Contiguous row access → cache-friendly.
- **C) Incorrect.** Separate processes still have per-process CPU caches; strided access is still expensive.
- **D) Incorrect.** OS paging operates at 4KB granularity; cache-line behaviour is independent and still penalises strided access.

---

## Q8 — Sequential vs Stride-8 Access

```python
import numpy as np

SIZE = 10_000_000
x = np.arange(SIZE, dtype=np.float64)

# Access pattern A: every 8th element
a_strided = x[::8]

# Access pattern B: every element sequentially
a_sequential = x[::1]
```

If you sum all the selected elements, which access pattern is faster and why?

**A)** Stride-8 access (`x[::8]`), because fewer elements are read, reducing total memory bandwidth.

**B)** Sequential access (`x[::1]`), because each cache line loaded contains 8 `float64` values, all of which are used.

**C)** Both are the same speed; NumPy uses SIMD to vectorise both patterns equally.

**D)** Stride-8 access (`x[::8]`), because the CPU prefetcher specifically optimises constant-stride patterns.

---

**Answer: B**

A typical CPU cache line is **64 bytes**. For `float64` (8 bytes each), one cache line holds **8 elements**.

- **Sequential (`x[::1]`):** loading element `x[0]` brings `x[0]`–`x[7]` into the cache. All 8 are consumed by the next 8 accesses — 100% cache-line utilisation.
- **Stride-8 (`x[::8]`):** loading element `x[0]` brings `x[0]`–`x[7]` into the cache, but only `x[0]` is used; the rest are evicted unused. Each access triggers a new cache-line load — 12.5% cache-line utilisation, ~8× more memory traffic per useful element.

- **A) Incorrect.** Fewer *logical* reads, but each triggers a full cache-line load — actual memory traffic per useful element is much higher.
- **B) Correct.** Sequential access fully utilises every loaded cache line.
- **C) Incorrect.** SIMD works well with sequential data; stride-8 prevents effective vectorisation.
- **D) Incorrect.** While prefetchers can detect constant strides, the wasted cache-line bandwidth still makes stride-8 slower than sequential.

---

## Q9 — CHW Channels-First Cache Inefficiency

```python
import numpy as np

C, H, W = 3, 64, 64
image = np.random.rand(C, H, W)   # channels-first layout: (C, H, W)
result = np.zeros((H, W))

for i in range(H):
    for j in range(W):
        for k in range(C):         # innermost loop over channels
            result[i, j] += image[k, i, j]
```

Is this loop cache-efficient on a CPU, and why?

**A)** Yes, because the innermost loop is the shortest (C=3), so it exits quickly and minimises total iterations.

**B)** No, because `image[k, i, j]` with varying `k` in the innermost loop accesses memory with a stride of `H × W × 8 = 32768` bytes between elements.

**C)** Yes, because NumPy automatically reorders loops to match memory layout at runtime.

**D)** No, but only because `result[i, j]` is accessed redundantly in the innermost loop.

---

**Answer: B**

`image` has shape `(C, H, W)` in C-order. Its strides are:

| Axis | Stride                         |
|------|--------------------------------|
| C (axis 0) | H × W × 8 = 64 × 64 × 8 = **32768 bytes** |
| H (axis 1) | W × 8 = 64 × 8 = 512 bytes     |
| W (axis 2) | 8 bytes                        |

The innermost loop varies `k` (the channel axis), so consecutive accesses `image[0, i, j]`, `image[1, i, j]`, `image[2, i, j]` are **32768 bytes apart**. Each access is a cache miss. The cache-efficient fix is either:

1. Use a **channels-last** layout `(H, W, C)` so the innermost `k` loop is sequential, or
2. Reorder loops so `i` or `j` is innermost (matching the smallest stride in the current layout).

- **A) Incorrect.** Loop length does not determine cache efficiency; stride does.
- **B) Correct.** Varying the channel axis (`k`) in the innermost loop on a channels-first array is maximally cache-unfriendly.
- **C) Incorrect.** NumPy does not reorder Python loops; it only vectorises ufuncs internally.
- **D) Incorrect.** `result[i, j]` access is not the bottleneck; the strided `image` read is.

---

## Q10 — reshape(-1) and C-Order Flattening

```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.reshape(-1)[4])
```

What does this print?

**A)** `4`

**B)** `5`

**C)** `2`

**D)** `6`

---

**Answer: B**

`a.reshape(-1)` flattens the array into a 1D array using **C (row-major) order** by default: rows are concatenated left to right.

```
a.reshape(-1) → [1, 2, 3, 4, 5, 6]
index:           [0, 1, 2, 3, 4, 5]
```

Index 4 is **5**.

This illustrates row-major storage: element `a[1, 1]` (row 1, column 1) is at flat index `1 × 3 + 1 = 4`.

- **A) Incorrect.** Index 3 is `4`, not index 4.
- **B) Correct.** Index 4 is `5`.
- **C) Incorrect.** Index 1 is `2`.
- **D) Incorrect.** Index 5 is `6`.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets loop order optimization, cache-friendly strides, C vs Fortran order, and CPU vs CUDA layout differences

---

## Q11 — Strides After np.asfortranarray

> **Week reference:** Week 3

```python
import numpy as np

a = np.asfortranarray(np.zeros((4, 6), dtype=np.float64))
print(a.strides)
print(a.flags['F_CONTIGUOUS'])
print(a.flags['C_CONTIGUOUS'])
```

What does this print?

- A) `(48, 8)` / `False` / `True`
- B) `(8, 32)` / `True` / `False`
- C) `(8, 8)` / `True` / `True`
- D) `(32, 8)` / `False` / `True`

**Answer: B**

`np.asfortranarray` creates a Fortran-order (column-major) array. For shape `(4, 6)` float64, axis 0 (rows) varies fastest: stride = 1×8 = 8 bytes. Axis 1 (columns) stride = 4×8 = 32 bytes. The array is F_CONTIGUOUS but not C_CONTIGUOUS.

- **A) Incorrect.** `(48, 8)` would be C-order strides for shape `(4, 6)`: axis 0 = 6×8 = 48, axis 1 = 8. This is C-order, not F-order.
- **B) Correct.** F-order `(4, 6)` float64: strides `(8, 32)`. `F_CONTIGUOUS = True`, `C_CONTIGUOUS = False`.
- **C) Incorrect.** An array cannot have both strides equal to 8 for a 2D array with more than 1 row/column — adjacent elements along different axes would overlap in memory.
- **D) Incorrect.** `(32, 8)` would be C-order strides for a `(4, 4)` float64 array, not `(4, 6)`. Also, C-order gives `C_CONTIGUOUS = True`, `F_CONTIGUOUS = False`.

---

## Q12 — Loop Order for F-Order Array

> **Week reference:** Week 3

```python
import numpy as np

a = np.asfortranarray(np.ones((100, 200), dtype=np.float64))
# a.strides == (8, 800)

total = 0.0
# Option X
for j in range(200):
    for i in range(100):
        total += a[i, j]

# Option Y
for i in range(100):
    for j in range(200):
        total += a[i, j]
```

Which option is more cache-efficient for the F-order array `a`?

- A) Option Y — outer `i`, inner `j` — because this is the standard row-major traversal
- B) Option X — outer `j`, inner `i` — because F-order stores columns contiguously, so the inner `i` loop is stride 8
- C) Both are equivalent; NumPy normalizes access patterns for Fortran arrays
- D) Option Y — because the inner `j` loop accesses the axis with the larger stride (800 bytes), which triggers prefetching

**Answer: B**

`a` has F-order strides `(8, 800)`. Axis 0 (rows, `i`) has stride 8 bytes — adjacent row elements are contiguous. Axis 1 (columns, `j`) has stride 800 bytes — adjacent column elements are 100 float64 values apart.

Option X: innermost `i` loop steps through memory 8 bytes at a time (sequential, cache-friendly).
Option Y: innermost `j` loop steps 800 bytes at a time — 100 elements per step, a new cache line every iteration.

- **A) Incorrect.** Standard row-major traversal (outer row, inner column) is optimal for C-order arrays. For F-order, the memory layout is transposed, so the optimal loop order is also transposed.
- **B) Correct.** For F-order, inner loop over axis 0 (`i`, stride 8) is cache-friendly. This is the opposite of C-order best practice.
- **C) Incorrect.** NumPy does not reorder Python for-loops. The programmer must explicitly match loop order to memory layout.
- **D) Incorrect.** A larger stride in the innermost loop means more cache misses, not better prefetching. Prefetching works best with small, regular strides — exactly what axis 0 (stride 8) provides.

---

## Q13 — Strides of a 3D C-Order Array

> **Week reference:** Week 3

```python
import numpy as np

vol = np.zeros((8, 16, 32), dtype=np.float64)
print(vol.strides)
```

What does this print?

- A) `(8, 8, 8)`
- B) `(4096, 256, 8)`
- C) `(32, 16, 8)`
- D) `(8, 128, 4096)`

**Answer: B**

For a C-order float64 array of shape `(8, 16, 32)`, strides are computed right-to-left:
- Axis 2 (W=32): 1×8 = **8 bytes**
- Axis 1 (H=16): 32×8 = **256 bytes**
- Axis 0 (D=8): 16×32×8 = **4096 bytes**

Strides = `(4096, 256, 8)`.

- **A) Incorrect.** `(8, 8, 8)` would imply all elements along every axis are adjacent — impossible for a 3D array where moving along axis 0 must skip an entire 16×32 slice.
- **B) Correct.** `(4096, 256, 8)` — each step along axis 0 skips 16×32×8 = 4096 bytes; each step along axis 1 skips 32×8 = 256 bytes; each step along axis 2 skips 8 bytes.
- **C) Incorrect.** `(32, 16, 8)` treats the strides as if they were element counts equal to each dimension — confusing shape with strides.
- **D) Incorrect.** `(8, 128, 4096)` is the reversed order — these would be F-order strides for shape `(8, 16, 32)` but in the wrong axis order.

---

## Q14 — Broadcasting Stride Trick

> **Week reference:** Week 3

```python
import numpy as np

row = np.array([1.0, 2.0, 3.0, 4.0])  # shape (4,), strides (8,)
mat = np.broadcast_to(row, (5, 4))
print(mat.strides)
print(mat[0, :] is mat[1, :])
```

What does this print?

- A) `(32, 8)` / `False`
- B) `(0, 8)` / `False`
- C) `(0, 8)` / `True`
- D) `(8, 8)` / `True`

**Answer: B**

`np.broadcast_to(row, (5, 4))` creates a read-only view with shape `(5, 4)`. The new axis 0 has stride 0 (no memory advance per row — all rows are the same data). Axis 1 retains stride 8.

However, `mat[0, :]` and `mat[1, :]` are two separate view objects (Python creates a new object for each slice operation), so `is` returns `False` even though they point to the same memory.

- **A) Incorrect.** `(32, 8)` would be the strides of a real 2D array with 4 float64 columns — a copy, not a broadcast view. `broadcast_to` uses stride 0 for the broadcast dimension.
- **B) Correct.** Strides `(0, 8)` and `False` (two different Python view objects even though they share memory).
- **C) Incorrect.** The strides `(0, 8)` are correct, but `is` compares Python object identity, not memory addresses. Each slice creates a new view object, so `is` is `False`.
- **D) Incorrect.** Stride 8 for axis 0 would mean each row is 1 element apart — overlapping with the previous row, which makes no sense. Broadcast uses stride 0 for repeated dimensions.

---

## Q15 — CUDA Thread Layout for C-Order Access

> **Week reference:** Week 9

```python
# Pseudocode for a CUDA kernel processing a C-order float32 array
# shape: (H, W) = (1024, 1024), C-order (row-major)

# Option A: threadIdx.x → row, threadIdx.y → column
#   thread (tx, ty) accesses data[tx, ty]

# Option B: threadIdx.x → column, threadIdx.y → row
#   thread (tx, ty) accesses data[ty, tx]
```

Which option achieves coalesced memory access, and why?

- A) Option A — `threadIdx.x` maps to rows, so threads in a warp (which differ in x) access the same column across different rows
- B) Option B — `threadIdx.x` maps to columns, so threads in a warp access adjacent columns in the same row, which are contiguous in C-order memory
- C) Both options are equally coalesced because CUDA automatically reorders memory requests
- D) Option A — row-major arrays are designed for x-dimension access patterns in CUDA

**Answer: B**

A CUDA warp consists of 32 threads with consecutive `threadIdx.x` values (threadIdx.y and threadIdx.z are identical within a warp). For coalesced access, these 32 threads must access 32 consecutive memory addresses simultaneously.

In a C-order `(H, W)` array, consecutive addresses are along the W (column) axis. Option B maps `threadIdx.x` → column, so warp threads access `data[row, 0], data[row, 1], ..., data[row, 31]` — 32 consecutive float32 values = 128 bytes = one coalesced transaction.

- **A) Incorrect.** If `threadIdx.x` maps to rows, warp threads access `data[0, col], data[1, col], ..., data[31, col]`. In C-order these are W×4 = 4096 bytes apart — 32 separate uncoalesced transactions, ~32× memory bandwidth waste.
- **B) Correct.** `threadIdx.x` → column index ensures warp threads access adjacent columns (stride 4 bytes for float32), forming one coalesced 128-byte transaction.
- **C) Incorrect.** CUDA hardware can coalesce only if addresses are already aligned and consecutive. It does not reorder accesses between threads. The programmer must arrange the mapping.
- **D) Incorrect.** C-order (row-major) is optimized for column-varies-fastest access — meaning `threadIdx.x` should map to columns, not rows.

---

## Q16 — np.ascontiguousarray on a Slice

> **Week reference:** Week 3

```python
import numpy as np

a = np.zeros((100, 200), dtype=np.float64)  # C-order, strides (1600, 8)
b = a[:, ::2]   # every other column, shape (100, 100)
c = np.ascontiguousarray(b)

print(b.strides)
print(c.strides)
print(np.shares_memory(b, c))
```

What does this print?

- A) `(1600, 8)` / `(1600, 8)` / `True`
- B) `(1600, 16)` / `(800, 8)` / `False`
- C) `(800, 8)` / `(800, 8)` / `True`
- D) `(1600, 16)` / `(1600, 16)` / `True`

**Answer: B**

`b = a[:, ::2]`: selecting every other column doubles the column stride. Original axis 1 stride = 8 bytes; with `::2`, axis 1 stride = 2×8 = 16 bytes. Axis 0 stride unchanged at 1600 bytes. So `b.strides = (1600, 16)`.

`b` is not C-contiguous (axis 1 stride 16 ≠ 8 bytes for float64). `np.ascontiguousarray(b)` copies data into a new C-contiguous array of shape `(100, 100)`. For C-contiguous `(100, 100)` float64: strides = `(100×8, 8)` = `(800, 8)`.

Since `c` is a new allocation, `np.shares_memory(b, c)` is `False`.

- **A) Incorrect.** Column stride 8 would mean the original unsliced spacing — but `::2` doubles the column stride to 16.
- **B) Correct.** `b.strides = (1600, 16)`, `c.strides = (800, 8)`, `shares_memory = False`.
- **C) Incorrect.** `(800, 8)` would require `b` to already be contiguous, but `b` is a non-contiguous view with column stride 16.
- **D) Incorrect.** If `c` had the same strides as `b`, they would share the same (non-contiguous) layout. `np.ascontiguousarray` specifically produces C-contiguous output with stride 8 on the last axis.

---

## Q17 — Axis=0 vs Axis=1 Sum Strides

> **Week reference:** Week 3

```python
import numpy as np

a = np.random.rand(500, 800).astype(np.float64)
# a.strides == (6400, 8)

result_0 = a.sum(axis=0)  # sum along rows → shape (800,)
result_1 = a.sum(axis=1)  # sum along columns → shape (500,)
```

Which reduction accesses memory with the smaller per-element stride, and what is that stride?

- A) `axis=0` — stride 8 bytes, because summing along rows accesses consecutive row elements
- B) `axis=1` — stride 8 bytes, because summing along columns accesses consecutive elements within each row
- C) `axis=0` — stride 6400 bytes, because moving down a column in C-order jumps one full row
- D) Both have stride 8 bytes — NumPy always accesses memory sequentially regardless of axis

**Answer: B**

`a.sum(axis=1)` sums along axis 1 (columns within each row). For each of the 500 rows, NumPy iterates over 800 consecutive float64 values with stride 8 bytes. This is cache-friendly sequential access.

`a.sum(axis=0)` sums along axis 0 (rows within each column). For each of the 800 columns, NumPy iterates over 500 elements spaced 6400 bytes apart (one full row between each element) — cache-unfriendly strided access.

- **A) Incorrect.** `axis=0` sums down columns, not along rows. The per-element stride is 6400 bytes (the row stride), not 8 bytes. Stride 8 describes movement along axis 1.
- **B) Correct.** `axis=1` sums along rows with stride 8 bytes — sequential, cache-friendly. This is the faster reduction for C-order arrays.
- **C) Partially correct description, wrong answer label.** The description of axis=0 stride being 6400 bytes is accurate, but this means axis=0 is the slower, not the smaller-stride reduction.
- **D) Incorrect.** NumPy's reduction follows the actual memory strides. For `axis=0`, the stride between accessed elements is 6400 bytes; for `axis=1`, it is 8 bytes. They are not equal.

---

## Q18 — Transpose Flags

> **Week reference:** Week 3

```python
import numpy as np

a = np.zeros((5, 10), dtype=np.float64)
b = a.T
c = np.ascontiguousarray(b)

print(b.flags['C_CONTIGUOUS'], b.flags['F_CONTIGUOUS'])
print(c.flags['C_CONTIGUOUS'], c.flags['F_CONTIGUOUS'])
```

What does this print?

- A) `False True` / `True False`
- B) `True True` / `True True`
- C) `False False` / `True False`
- D) `True False` / `True False`

**Answer: A**

`a` is C-contiguous (shape `(5, 10)`, strides `(80, 8)`). `b = a.T` has shape `(10, 5)` and strides `(8, 80)`. A 2D array with strides `(8, 80)` has its first axis as the fast-varying axis (stride 8), which matches Fortran-order. So `b` is F-contiguous but not C-contiguous.

`c = np.ascontiguousarray(b)` produces a C-contiguous copy of shape `(10, 5)` with strides `(40, 8)`. A freshly allocated C-contiguous array is generally not F-contiguous (unless it is 1D or has one dimension of size 1), so `c.flags['F_CONTIGUOUS']` is `False`.

- **A) Correct.** `b`: `False True` (F-contiguous). `c`: `True False` (C-contiguous).
- **B) Incorrect.** An array can be simultaneously C- and F-contiguous only if it is 1D or one dimension has size 1. Neither `b` (shape 10×5) nor `c` satisfies this.
- **C) Incorrect.** `b` is F-contiguous (not neither). Transposing a C-contiguous non-square array gives an F-contiguous view.
- **D) Incorrect.** `b.flags['C_CONTIGUOUS']` is `False` — `b` is a transposed view and not C-contiguous. Only C-contiguous `c` has `C_CONTIGUOUS = True`.

---

## Q19 — Memory Access Pattern for Diagonal Sum

> **Week reference:** Week 3

```python
import numpy as np

N = 1000
a = np.zeros((N, N), dtype=np.float64)  # C-order, strides (8000, 8)

# Computing the trace (sum of diagonal)
trace = sum(a[i, i] for i in range(N))
```

What is the memory stride between consecutive diagonal accesses `a[i, i]` and `a[i+1, i+1]`?

- A) 8 bytes — diagonal elements are adjacent in memory
- B) 8008 bytes — each diagonal step advances 1 row + 1 column
- C) 8000 bytes — each diagonal step advances exactly one row
- D) 16 bytes — diagonal elements are every other element in the flattened array

**Answer: B**

`a[i, i]` is at offset `i × 8000 + i × 8 = i × 8008` bytes from the start. Moving from `a[i, i]` to `a[i+1, i+1]` advances both the row index and column index by 1: Δbytes = 1×8000 + 1×8 = **8008 bytes**.

Each diagonal access is 8008 bytes from the previous — more than 125 cache lines apart. The trace computation is maximally cache-unfriendly (one miss per element).

- **A) Incorrect.** Adjacent in memory would mean stride 8 bytes (consecutive float64). Diagonal elements are neither in the same row nor the same column.
- **B) Correct.** Stride = row_stride + col_stride = 8000 + 8 = 8008 bytes. Each diagonal step crosses a row boundary and advances one column.
- **C) Incorrect.** 8000 bytes is the row stride alone (moving to the same column of the next row). The diagonal also advances one column (+8 bytes), giving 8008 total.
- **D) Incorrect.** 16 bytes would mean diagonal elements are every other element — only true for a 2×1 or 1×2 array. In a 1000×1000 array, elements in adjacent rows are 1000 elements = 8000 bytes apart.

---

## Q20 — Choosing Layout for a CUDA Reduction

> **Week reference:** Week 9

You have a 2D float32 array of shape `(H, W)` = `(1024, 1024)` and want to compute the sum of each row on a GPU. Each CUDA block processes one row, with 32 threads per block (one warp). Thread `tx` (0–31) reads `data[row, tx], data[row, tx+32], data[row, tx+64], ...` in a stride-32 pattern.

For coalesced access on the GPU, which array layout is preferred?

- A) Fortran-order (column-major) — so that thread `tx` accesses elements stride 1 in memory
- B) C-order (row-major) — so that adjacent threads (`tx` and `tx+1`) access adjacent memory addresses (`data[row, tx]` and `data[row, tx+1]`)
- C) Either layout — CUDA coalesces both C and F-order accesses automatically
- D) Fortran-order — because CUDA was designed for Fortran scientific computing

**Answer: B**

The access pattern `data[row, tx]` where `tx` is `threadIdx.x` (varying 0–31 across the warp) accesses 32 consecutive columns within the same row. In C-order (row-major), columns are contiguous: `data[row, 0], data[row, 1], ..., data[row, 31]` are at addresses offset by 4 bytes (float32) each. All 32 threads issue addresses spanning 128 bytes — one coalesced memory transaction.

In F-order, `data[row, 0], data[row, 1], ...` are H×4 = 4096 bytes apart — 32 separate uncoalesced transactions.

- **A) Incorrect.** In F-order, columns are stride H×4 bytes apart. Thread `tx` accessing `data[row, tx]` in F-order jumps 4096 bytes per increment in `tx` — severe uncoalesced access, not stride 1.
- **B) Correct.** C-order places all columns of a row contiguously. Warp threads accessing adjacent columns issue one coalesced transaction.
- **C) Incorrect.** CUDA hardware coalesces only when addresses issued by a warp are naturally aligned and consecutive. F-order column access produces strided, uncoalesced transactions that hardware cannot transparently fix.
- **D) Incorrect.** CUDA is a C/C++ extension; it has no affinity for Fortran-order arrays. CUDA's memory coalescing model aligns with C-order for row-based access patterns.

---

## Set 3 — Extended Practice

> Targets strides after slicing, fancy-index copies, reshape ordering, nditer, expand_dims, and dtype-dependent strides — edge cases not covered in Sets 1–2.

---

## Q21 — Strides of Every-Other-Column Slice

> **Week reference:** Week 3

```python
import numpy as np

a = np.zeros((6, 8), dtype=np.float64)  # C-order, strides (64, 8)
b = a[:, ::2]   # every other column
print(b.shape)
print(b.strides)
```

What does this print?

- A) `(6, 4)` / `(64, 8)`
- B) `(6, 4)` / `(64, 16)`
- C) `(6, 8)` / `(64, 16)`
- D) `(3, 8)` / `(128, 8)`

**Answer: B**

`a` has shape `(6, 8)` and strides `(64, 8)`. `[:, ::2]` selects every other column (columns 0, 2, 4, 6), giving shape `(6, 4)`. The row stride is unchanged at 64 bytes (skipping rows still jumps a full original row). The column stride doubles: stepping over one column in `b` skips two original columns: `2 × 8 = 16` bytes. No data is copied — `b` is a view.

- **A) Incorrect.** Shape is correct but strides are wrong. Column stride 8 would mean adjacent columns in `b` are still 1 float64 apart, as if no skipping occurred. The `::2` step doubles the column stride.
- **B) Correct.** Shape `(6, 4)`, strides `(64, 16)`. Row stride unchanged; column stride doubled by the step-2 slice.
- **C) Incorrect.** Shape `(6, 8)` would mean all 8 original columns are present, but `::2` selects only 4 columns. Shape and strides are both wrong.
- **D) Incorrect.** Shape `(3, 8)` would result from `a[::2, :]` (every other row), not `a[:, ::2]`. This confuses row slicing with column slicing.

---

## Q22 — Fancy Index vs Slice: View or Copy?

> **Week reference:** Week 3

```python
import numpy as np

a = np.arange(20, dtype=np.float64)

b = a[2:10]          # basic slice
c = a[[2, 3, 4, 5, 6, 7, 8, 9]]  # fancy index with same elements

print(np.shares_memory(a, b))
print(np.shares_memory(a, c))
```

What does this print?

- A) `True` / `True`
- B) `True` / `False`
- C) `False` / `True`
- D) `False` / `False`

**Answer: B**

`b = a[2:10]` is basic (slice) indexing — it returns a view that shares the buffer with `a`. `np.shares_memory(a, b)` is `True`.

`c = a[[2, 3, 4, 5, 6, 7, 8, 9]]` is fancy (integer array) indexing — NumPy cannot express this as a stride-offset view even though the indices happen to be consecutive. It always allocates a new buffer and copies the selected elements. `np.shares_memory(a, c)` is `False`.

- **A) Incorrect.** Fancy indexing always produces a copy, so the second `shares_memory` is `False`, not `True`.
- **B) Correct.** Basic slice → view (True); fancy index → copy (False), even when the elements selected are identical.
- **C) Incorrect.** Basic slicing always produces a view (True), not a copy (False). Only fancy indexing always produces a copy.
- **D) Incorrect.** Basic slicing never copies — it is always a view, so the first result cannot be `False` for a valid non-strided slice like `a[2:10]`.

---

## Q23 — reshape(-1) on F-Order Array

> **Week reference:** Week 3

```python
import numpy as np

a = np.asfortranarray(np.array([[1, 2, 3],
                                 [4, 5, 6]], dtype=np.int64))
print(a.reshape(-1))
```

What does this print?

- A) `[1 2 3 4 5 6]` — row-major (C-order) flattening
- B) `[1 4 2 5 3 6]` — column-major (F-order) flattening
- C) `[1 2 3 4 5 6]` — NumPy always flattens in C-order regardless of array layout
- D) `[4 5 6 1 2 3]` — F-order reverses the row sequence

**Answer: C**

`np.reshape` with no `order` argument defaults to `order='C'`, which means the output is filled in row-major order regardless of the input array's memory layout. Even though `a` is F-contiguous, `a.reshape(-1)` reads elements in C-order: row 0 first (`1, 2, 3`), then row 1 (`4, 5, 6`). To flatten in column-major order you must use `a.reshape(-1, order='F')`, which would give `[1, 4, 2, 5, 3, 6]`.

- **A) Correct description, but the reasoning in option A is that it is row-major — that matches C correct.** See C below.
- **B) Incorrect.** This is what `a.reshape(-1, order='F')` would produce (column-major flattening: column 0 first = 1,4; column 1 = 2,5; column 2 = 3,6). Without `order='F'`, reshape defaults to C-order.
- **C) Correct.** `reshape(-1)` uses C-order by default, producing `[1 2 3 4 5 6]`. The underlying F-order storage is irrelevant to the reshape output order.
- **D) Incorrect.** F-order changes which axis varies fastest in memory, but it does not reverse the row sequence. The rows of `a` are still [1,2,3] and [4,5,6] logically; F-order only affects the physical byte layout.

---

## Q24 — np.nditer with order='F' on C-Order Array

> **Week reference:** Week 3

```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.float64)  # C-order

result = [x for x in np.nditer(a, order='F')]
print(result)
```

What does this print?

- A) `[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]` — C-order traversal (row by row)
- B) `[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]` — F-order traversal (column by column)
- C) `[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]` — reversed C-order
- D) `[4.0, 5.0, 6.0, 1.0, 2.0, 3.0]` — row-reversed order

**Answer: B**

`np.nditer(a, order='F')` traverses the array in Fortran (column-major) order — the first axis (rows) varies fastest. For the shape `(2, 3)` array, iteration goes: column 0 top-to-bottom (`a[0,0]=1`, `a[1,0]=4`), then column 1 (`a[0,1]=2`, `a[1,1]=5`), then column 2 (`a[0,2]=3`, `a[1,2]=6`). The `order` parameter of `nditer` controls traversal sequence, not the memory layout of `a`.

- **A) Incorrect.** This is what `np.nditer(a, order='C')` would produce (or equivalently, just iterating `a` with default settings). `order='F'` changes the traversal to column-major.
- **B) Correct.** F-order iteration: column 0 (1, 4), column 1 (2, 5), column 2 (3, 6) → `[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]`.
- **C) Incorrect.** Reversed C-order would be `[6, 5, 4, 3, 2, 1]`. `order='F'` does not reverse; it changes which axis varies fastest.
- **D) Incorrect.** Row-reversed would be `[4, 5, 6, 1, 2, 3]`. This would come from iterating rows in reverse order, which is not what `order='F'` does.

---

## Q25 — Strides After np.expand_dims

> **Week reference:** Week 3

```python
import numpy as np

a = np.zeros((4, 5), dtype=np.float64)
# a.strides == (40, 8)

b = np.expand_dims(a, axis=1)
print(b.shape)
print(b.strides)
```

What does this print?

- A) `(4, 1, 5)` / `(40, 40, 8)`
- B) `(4, 1, 5)` / `(40, 8, 8)`
- C) `(1, 4, 5)` / `(320, 40, 8)`
- D) `(4, 5, 1)` / `(40, 8, 8)`

**Answer: A**

`np.expand_dims(a, axis=1)` inserts a new axis of size 1 at position 1, giving shape `(4, 1, 5)`. This is a zero-copy view — no data moves. The new axis has a formally defined stride, but since its size is 1, the stride value never actually causes any memory jump. NumPy sets the stride of the new size-1 axis equal to the stride of the original axis at that position, which is 40 bytes (the original row stride). The existing strides (40 for axis 0, 8 for axis 2) are preserved.

- **A) Correct.** Shape `(4, 1, 5)`, strides `(40, 40, 8)`. The inserted axis-1 inherits the stride from the original axis-0 stride (40). Axis 0 stays 40; axis 2 stays 8.
- **B) Incorrect.** `(40, 8, 8)` would place the new axis stride at 8 bytes, confusing it with the column stride. The inserted axis stride should reflect the step between "rows" of the sub-array it spans, which is 40 bytes.
- **C) Incorrect.** Shape `(1, 4, 5)` would result from `np.expand_dims(a, axis=0)` (inserting at position 0, not 1). The strides `(320, 40, 8)` are correct for that case: the new leading axis of size 1 has stride = 4×5×8 = 160... actually 320 is also wrong; the stride of a size-1 axis inserted at position 0 is the original total array size in bytes or the stride of the first axis = 40×4 = 160... this option is internally inconsistent.
- **D) Incorrect.** Shape `(4, 5, 1)` would result from `np.expand_dims(a, axis=2)` or `np.expand_dims(a, axis=-1)`, inserting the new axis at the end.

---

## Q26 — C_CONTIGUOUS Flag After Transpose and Copy

> **Week reference:** Week 3

```python
import numpy as np

a = np.zeros((3, 7), dtype=np.float64)
b = a.T
c = b.copy()

print(b.flags['C_CONTIGUOUS'])
print(c.flags['C_CONTIGUOUS'])
print(np.shares_memory(b, c))
```

What does this print?

- A) `False` / `False` / `True`
- B) `False` / `True` / `False`
- C) `True` / `True` / `False`
- D) `True` / `False` / `True`

**Answer: B**

`a` is C-contiguous with shape `(3, 7)` and strides `(56, 8)`. `b = a.T` has shape `(7, 3)` and strides `(8, 56)`. With strides `(8, 56)`, the row stride (8) is smaller than the column stride (56), which is the signature of F-contiguous layout — so `b` is F-contiguous but **not** C-contiguous. `b.flags['C_CONTIGUOUS']` is `False`.

`c = b.copy()` creates a C-contiguous copy by default. Shape `(7, 3)`, strides `(3×8, 8) = (24, 8)`. `c.flags['C_CONTIGUOUS']` is `True`.

`np.shares_memory(b, c)` is `False` because `.copy()` always allocates a new independent buffer.

- **A) Incorrect.** `c.copy()` produces a C-contiguous array by default, so `c.flags['C_CONTIGUOUS']` is `True`, not `False`. Also, the two arrays do not share memory after a copy.
- **B) Correct.** `b`: False (F-contiguous transpose view). `c`: True (freshly copied C-contiguous array). `shares_memory`: False (copy allocates new buffer).
- **C) Incorrect.** `b.flags['C_CONTIGUOUS']` is `False` — the transposed view is not C-contiguous. This option incorrectly claims `True` for `b`.
- **D) Incorrect.** `.copy()` never produces a view; memory is never shared after a `.copy()` call. `np.shares_memory` cannot be `True` for a copy.

---

## Q27 — Loop Order for 3D F-Order Array

> **Week reference:** Week 3

```python
import numpy as np

a = np.asfortranarray(np.zeros((4, 5, 6), dtype=np.float64))
# a.strides == (8, 32, 160)

total = 0.0
# Option X
for k in range(6):
    for j in range(5):
        for i in range(4):
            total += a[i, j, k]

# Option Y
for i in range(4):
    for j in range(5):
        for k in range(6):
            total += a[i, j, k]
```

Which option is more cache-efficient, and why?

- A) Option Y — outer `i`, middle `j`, inner `k` — because this is the standard row-major loop order
- B) Option X — outer `k`, middle `j`, inner `i` — because F-order stores axis 0 fastest, so the inner `i` loop steps by 8 bytes
- C) Both are equivalent — `np.asfortranarray` normalizes access patterns
- D) Option Y — because iterating over the largest dimension last (k=6) reduces cache evictions

**Answer: B**

For an F-order array of shape `(4, 5, 6)`, strides are: axis 0 (i) = 8, axis 1 (j) = 4×8 = 32, axis 2 (k) = 4×5×8 = 160. The smallest stride is axis 0 (8 bytes). Cache efficiency requires the innermost loop to vary the axis with the smallest stride.

Option X: inner `i` (stride 8) — sequential memory access, 8 float64 values per 64-byte cache line.
Option Y: inner `k` (stride 160) — jumps 20 float64 values per step, a cache miss on almost every access.

- **A) Incorrect.** "Standard row-major loop order" (outer=axis 0, inner=last axis) is optimal for C-order arrays. For F-order arrays the logic is inverted: the innermost loop should cover the first axis (axis 0), not the last. Applying C-order intuition to F-order arrays is the classic exam trap.
- **B) Correct.** F-order strides `(8, 32, 160)`: innermost loop must vary axis 0 (`i`, stride 8). Option X has `i` innermost, giving stride-8 sequential access.
- **C) Incorrect.** `np.asfortranarray` changes the memory layout but does not reorder Python for-loops. The programmer must match loop order to the actual strides.
- **D) Incorrect.** Dimension size does not determine cache efficiency — stride does. The innermost loop over `k` in Option Y has stride 160 bytes regardless of whether k's dimension is 6 or 600.

---

## Q28 — np.broadcast_to Writability

> **Week reference:** Week 3

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.broadcast_to(a, (4, 3))

try:
    b[0, 0] = 99.0
    print("write succeeded")
except ValueError as e:
    print("write failed:", e)
```

What does this print?

- A) `write succeeded` — broadcast arrays support element assignment
- B) `write failed: ...` — broadcast arrays are read-only
- C) `write succeeded` — only the first row is affected since stride[0] = 0
- D) `write failed: ...` — but only because axis 0 has stride 0

**Answer: B**

`np.broadcast_to` creates a **read-only view**. The stride-0 trick used for the broadcast dimension means that a single write to any element would logically affect all rows (since they all point to the same memory). NumPy prevents this silent aliasing by marking broadcast arrays as read-only. Attempting to write raises `ValueError: assignment destination is read-only`.

- **A) Incorrect.** Broadcast arrays are explicitly read-only. Attempting to write raises `ValueError`, not silently succeeds. A writable broadcast-like array requires `np.broadcast_arrays` followed by `.copy()`.
- **B) Correct.** `b.flags['WRITEABLE']` is `False` for broadcast views. Any assignment raises `ValueError: assignment destination is read-only`.
- **C) Incorrect.** Even if the runtime allowed the write (it does not), writing to `b[0, 0]` with stride 0 on axis 0 would write to the same memory location used by `b[1, 0]`, `b[2, 0]`, and `b[3, 0]` — silently modifying all rows. NumPy prevents this by marking the array read-only.
- **D) Incorrect.** The `ValueError` is raised because the array is read-only (not specifically because stride[0] = 0). Any non-writable array raises this error on write, regardless of the reason for the read-only flag.

---

## Q29 — Strides of int32 vs float64 Array

> **Week reference:** Week 3

```python
import numpy as np

a = np.zeros((5, 4), dtype=np.int32)
b = np.zeros((5, 4), dtype=np.float64)

print(a.strides)
print(b.strides)
```

What does this print?

- A) `(20, 4)` / `(40, 8)`
- B) `(16, 4)` / `(32, 8)`
- C) `(20, 4)` / `(20, 4)`
- D) `(4, 1)` / `(8, 1)`

**Answer: B**

Strides depend on the dtype element size. For a C-order array of shape `(5, 4)`:
- `int32`: 4 bytes per element. Axis 1 stride = 1×4 = 4 bytes. Axis 0 stride = 4×4 = 16 bytes. → `(16, 4)`.
- `float64`: 8 bytes per element. Axis 1 stride = 1×8 = 8 bytes. Axis 0 stride = 4×8 = 32 bytes. → `(32, 8)`.

- **A) Incorrect.** `(20, 4)` for int32 would imply axis 0 stride = 5×4 = 20 bytes, using the first dimension (5) instead of the second (4) in the formula. Axis 0 stride = n_cols × element_size = 4×4 = 16, not 20.
- **B) Correct.** `int32`: `(16, 4)` — 4 columns × 4 bytes/col = 16 for axis 0. `float64`: `(32, 8)` — 4 columns × 8 bytes/col = 32 for axis 0.
- **C) Incorrect.** Both arrays having the same strides would only happen if `int32` and `float64` had the same element size, which they do not (4 vs 8 bytes).
- **D) Incorrect.** `(4, 1)` and `(8, 1)` would be element-count strides (not byte strides). NumPy strides are always in **bytes**. The column stride is never 1 for numeric dtypes larger than 1 byte.

---

## Q30 — Identifying the Cache-Friendly Reduction Axis

> **Week reference:** Week 3

```python
import numpy as np

a = np.random.rand(200, 300, 400).astype(np.float64)
# C-order: strides = (300*400*8, 400*8, 8) = (960000, 3200, 8)

result_axis0 = a.sum(axis=0)  # shape (300, 400)
result_axis1 = a.sum(axis=1)  # shape (200, 400)
result_axis2 = a.sum(axis=2)  # shape (200, 300)
```

Which reduction accesses memory with the smallest per-element stride (most cache-friendly), and what is that stride?

- A) `axis=0` — stride 960000 bytes (reduces the outermost axis)
- B) `axis=1` — stride 3200 bytes (reduces the middle axis)
- C) `axis=2` — stride 8 bytes (reduces the innermost/last axis, which is contiguous)
- D) All three reductions have the same stride because NumPy internally reorders accesses

**Answer: C**

For a C-order array, the last axis has the smallest stride (8 bytes for float64). Summing along axis 2 means iterating over 400 consecutive float64 values per row — stride 8 bytes, fully sequential and cache-friendly. Each 64-byte cache line delivers 8 useful elements.

Summing along axis 0 means iterating over 200 values spaced 960,000 bytes apart — one cache miss per element. Summing along axis 1 means iterating over 300 values spaced 3,200 bytes apart — still cache-unfriendly.

- **A) Incorrect.** `axis=0` reduction iterates along the first axis with stride 960,000 bytes. This is the worst case — each element accessed is in a completely different region of memory with no cache reuse.
- **B) Incorrect.** `axis=1` reduction iterates along the middle axis with stride 3,200 bytes. Better than axis=0 but still causes a cache miss on almost every access (3,200 bytes >> 64-byte cache line).
- **C) Correct.** `axis=2` reduction iterates along the last axis with stride 8 bytes. This is sequential memory access: `a[i, j, 0], a[i, j, 1], ..., a[i, j, 399]` are physically contiguous. Every loaded cache line delivers 8 usable values.
- **D) Incorrect.** NumPy does not transparently reorder reduction accesses to be cache-friendly. The actual memory access pattern follows the strides of the array for the chosen axis. The programmer must select the axis that aligns with the physical layout.

---
