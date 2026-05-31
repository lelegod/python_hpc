# Cache Efficiency & Memory Layout — MCQ Practice

> Topics: Strides, loop ordering, row vs column access, cache hierarchy, spatial/temporal locality.
> Exam frequency: **Every exam**.

---

## Q1 — Optimal Loop Order from Strides
> **Week reference:** Week 3

A 4D NumPy float64 array has strides `(600, 40, 8, 200)` bytes. To maximise cache efficiency, what should be the order of the loop variables from **innermost to outermost**, where `i, j, k, l` correspond to axes 0, 1, 2, 3?

- A) `i, j, k, l`
- B) `k, j, l, i`
- C) `l, j, k, i`
- D) `k, l, j, i`

**Answer: B**

- A) Incorrect — axis 0 has the largest stride (600), making it the worst choice for the innermost loop.
- B) Correct — sorting strides ascending: 8(axis2=k) < 40(axis1=j) < 200(axis3=l) < 600(axis0=i); innermost to outermost = k, j, l, i.
- C) Incorrect — placing l (stride 200) innermost skips many cache lines per iteration.
- D) Incorrect — places l before j, reversing the correct j < l ordering.

---

## Q2 — C-Order Stride Layout
> **Week reference:** Week 3

For a C-order (row-major) float64 array of shape `(5, 6, 7)`, which axis has the **smallest** stride?

- A) Axis 0 (size 5)
- B) Axis 1 (size 6)
- C) Axis 2 (size 7)
- D) All axes have the same stride

**Answer: C**

- A) Incorrect — in C-order, axis 0 has the largest stride because it spans the most contiguous memory.
- B) Incorrect — axis 1 has an intermediate stride, not the smallest.
- C) Correct — in C-order, the last axis (axis 2) always has the smallest stride (equal to the element size, 8 bytes for float64).
- D) Incorrect — strides differ; they equal the product of all later dimension sizes times the element size.

---

## Q3 — Row Access vs Column Access
> **Week reference:** Week 3

You have a large float64 matrix `mat` of shape `(1000, 1000)` stored in C-order. Which access pattern is faster, and why?

- A) `mat[:,0]` — column access, because it reads a contiguous block of memory
- B) `mat[0,:]` — row access, because consecutive elements are adjacent in memory
- C) Both are equally fast because the total number of elements read is the same
- D) `mat[:,0]` — column access, because NumPy optimises vertical reads automatically

**Answer: B**

- A) Incorrect — column access in a row-major array is strided; each element is 8000 bytes apart, causing a cache miss for every access.
- B) Correct — row access reads consecutive float64 values, loading 8 elements per 64-byte cache line with full reuse.
- C) Incorrect — cache efficiency depends on access pattern, not just element count; column access is ~10x slower in practice.
- D) Incorrect — NumPy does not transparently reorder memory for column reads; column access remains cache-unfriendly.

---

## Q4 — Cache Line Size and Prefetching
> **Week reference:** Week 3

A CPU cache line is 64 bytes. When you access `x[0]` in a float64 array, how many consecutive elements are loaded into the cache for free?

- A) 1
- B) 4
- C) 8
- D) 16

**Answer: C**

- A) Incorrect — only one element would be loaded if there were no spatial prefetching, but cache lines load the entire 64-byte block.
- B) Incorrect — 4 float64 values = 32 bytes, which is only half a cache line.
- C) Correct — 64 bytes / 8 bytes per float64 = 8 elements loaded per cache line access.
- D) Incorrect — 16 float64 values = 128 bytes, which spans two cache lines, not one.

---

## Q5 — Spatial Locality
> **Week reference:** Week 3

Which of the following access patterns demonstrates the **best** spatial locality for a float64 array `a` of shape `(1024,)`?

- A) `a[0], a[512], a[256], a[768]` — accessing every 512th element
- B) `a[0], a[8], a[16], a[24]` — accessing every 8th element (stride 8 elements = 64 bytes)
- C) `a[0], a[1], a[2], a[3]` — accessing consecutive elements
- D) `a[1023], a[511], a[255], a[127]` — reverse strided access

**Answer: C**

- A) Incorrect — stride of 512 elements = 4096 bytes; each access loads a fresh cache line with no reuse.
- B) Incorrect — stride of 8 elements = 64 bytes; each access lands on a new cache line boundary, wasting the prefetched elements.
- C) Correct — consecutive access fully exploits the 8 float64 values loaded per cache line, maximising spatial reuse.
- D) Incorrect — reverse strided access still loads each element from a different cache line direction, giving poor spatial locality.

---

## Q6 — Temporal Locality and Loop Tiling
> **Week reference:** Week 3

Loop tiling (blocking) improves performance primarily by exploiting:

- A) Spatial locality — accessing memory in sequential order within each tile
- B) Temporal locality — reusing data in the cache before it is evicted
- C) Parallelism — running multiple tiles simultaneously on different cores
- D) Vectorisation — fitting tile data into SIMD registers

**Answer: B**

- A) Incorrect — tiling can improve spatial locality too, but its primary purpose is keeping a working set hot in cache across repeated accesses.
- B) Correct — tiling ensures that data loaded into cache is accessed multiple times (e.g., for matrix multiply, each tile element is reused across the inner loop) before being evicted.
- C) Incorrect — tiling is a sequential optimisation; it does not inherently add parallelism.
- D) Incorrect — vectorisation is a separate concern; tiling operates at the loop-structure level, not at the register level.

---

## Q7 — Infer Shape from Strides
> **Week reference:** Week 3

A float64 array has strides `(80, 8)` bytes. What is its shape in elements?

- A) `(80, 8)`
- B) `(8, 80)`
- C) `(10, 10)`
- D) Cannot be determined from strides alone

**Answer: D**

- A) Incorrect — strides are in bytes, not elements; shape cannot be read directly from stride values.
- B) Incorrect — confuses bytes with elements and swaps axes; still doesn't determine row count.
- C) Incorrect — n_cols = 80/8 = 10 is correct, but n_rows is not encoded in strides; arrays of shape (5,10), (10,10), or (20,10) all produce identical strides (80, 8).
- D) Correct — strides establish n_cols = 10 (from 80/8), but n_rows cannot be inferred; any array with 10 columns and any number of rows would have the same (80, 8) strides.

---

## Q8 — Effect of Transposing an Array
> **Week reference:** Week 3

You have a C-order float64 array `a` of shape `(100, 200)`. After computing `b = a.T`, which statement is true?

- A) `b` is a new array with a Fortran-order (column-major) memory layout
- B) `b` has the same memory buffer as `a`, but with reversed strides
- C) `b` is contiguous in C-order because NumPy copies data during transpose
- D) `b.strides` equals `a.strides` since the underlying data is unchanged

**Answer: B**

- A) Incorrect — `a.T` does not produce a Fortran-order copy; it is a view with swapped strides, not a layout change.
- B) Correct — `a.T` returns a view sharing `a`'s data buffer; only the strides are reversed from `(1600, 8)` to `(8, 1600)`.
- C) Incorrect — `a.T` is not C-contiguous; `np.ascontiguousarray(a.T)` would be needed for a contiguous copy.
- D) Incorrect — `b.strides` is `(8, 1600)`, the reverse of `a.strides` `(1600, 8)`.

---

## Q9 — Optimal Array Layout for CPU Convolution
> **Week reference:** Week 4

You are writing a CPU-optimised image convolution where the innermost loop iterates over **channels**. Which array layout should you use for the image data?

- A) CHW (channels first) — because CUDA GPUs use this format
- B) HWC (channels last) — because channels in the innermost position have the smallest stride
- C) HCW (height × channels × width) — to balance stride across all dimensions
- D) Either CHW or HWC; the CPU treats them identically

**Answer: B**

- A) Incorrect — CHW is optimal for CUDA GPUs, not CPUs; placing channels first means the innermost channel loop strides across large memory gaps.
- B) Correct — HWC puts channels in the last (innermost) dimension with stride 8 bytes, so the inner channel loop reads contiguous memory and reuses cache lines.
- C) Incorrect — HCW is a non-standard layout that provides no cache benefit for the standard convolution inner loop over channels.
- D) Incorrect — CPU and CUDA have opposite optimal layouts (HWC vs CHW); they are not equivalent.

---

## Q10 — Performance Staircase
> **Week reference:** Week 3

When you plot MFLOP/s against working set size for a CPU-bound loop, you observe a **staircase pattern** with distinct performance drops. What causes each step down?

- A) The loop hits Python's GIL at each new array allocation threshold
- B) The working set exceeds each successive cache level (L1, L2, L3), forcing slower memory accesses
- C) NumPy switches from vectorised to scalar code for larger arrays
- D) Memory bandwidth saturates at a fixed threshold regardless of cache size

**Answer: B**

- A) Incorrect — the GIL affects threading, not single-threaded memory access patterns; it does not produce a performance staircase.
- B) Correct — when the working set fits in L1 the throughput is highest; exceeding L1 drops to L2 speed, then L3, then DRAM, creating the staircase shape.
- C) Incorrect — NumPy's vectorisation does not toggle at specific array size boundaries.
- D) Incorrect — memory bandwidth saturation is a flat ceiling, not a staircase; the staircase reflects discrete cache-level transitions.

---

## Q11 — Loop Order and Cache Efficiency
> **Week reference:** Week 3

Consider a C-order float64 array `a` of shape `(N, M)` and the loop:

```python
for i in range(N):
    for j in range(M):
        process(a[j, i])
```

Is this cache-efficient?

- A) Yes — the outer loop over `i` covers rows, so memory is accessed sequentially
- B) No — `a[j, i]` accesses column `i` across all rows, which is strided in row-major storage
- C) Yes — NumPy automatically reorders accesses for cache efficiency
- D) It depends on whether N > M or M > N

**Answer: B**

- A) Incorrect — the loop variable `i` is in the outer position, but the array index `a[j, i]` uses `i` as the column; the inner loop over `j` walks down a column.
- B) Correct — the inner loop increments `j` while holding `i` fixed, accessing `a[0,i], a[1,i], a[2,i], ...`; consecutive accesses are M×8 bytes apart, causing a cache miss on each one.
- C) Incorrect — NumPy does not reorder loop-level memory accesses; the programmer must write cache-friendly loop order.
- D) Incorrect — the access pattern is column-major regardless of the relative sizes of N and M.

---

## Q12 — Stride Calculation for 2D Array
> **Week reference:** Week 3

What are the strides (in bytes) for a C-order float64 array of shape `(3, 4)`?

- A) `(4, 1)`
- B) `(8, 4)`
- C) `(32, 8)`
- D) `(24, 8)`

**Answer: C**

- A) Incorrect — these would be strides in elements, not bytes; for float64 each element is 8 bytes.
- B) Incorrect — axis 1 stride should be 8 bytes (one float64), not 4; and axis 0 stride should span a full row of 4×8=32 bytes.
- C) Correct — axis 1 (innermost) stride = 1 element × 8 bytes = 8; axis 0 stride = 4 elements × 8 bytes = 32; so strides = (32, 8).
- D) Incorrect — 24 would be correct for a shape (3,3) array (3×8=24), not (3,4).

---
