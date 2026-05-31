# Cache Efficiency & Memory Layout — Code-Based MCQ Practice

> Format: Each question shows Python/NumPy code to analyse for cache efficiency.
> Exam frequency: **Every exam**.

---

## Question 1

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

## Question 2

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

## Question 3

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

## Question 4

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

## Question 5

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

## Question 6

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

## Question 7

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

## Question 8

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

## Question 9

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

## Question 10

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
