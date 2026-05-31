# Cache Efficiency & Memory Layout — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Optimal Loop Order from Strides](#q1-optimal-loop-order-from-strides)
- [Q2 — C-Order Stride Layout](#q2-c-order-stride-layout)
- [Q3 — Row Access vs Column Access](#q3-row-access-vs-column-access)
- [Q4 — Cache Line Size and Prefetching](#q4-cache-line-size-and-prefetching)
- [Q5 — Spatial Locality](#q5-spatial-locality)
- [Q6 — Temporal Locality and Loop Tiling](#q6-temporal-locality-and-loop-tiling)
- [Q7 — Infer Shape from Strides](#q7-infer-shape-from-strides)
- [Q8 — Effect of Transposing an Array](#q8-effect-of-transposing-an-array)
- [Q9 — Optimal Array Layout for CPU Convolution](#q9-optimal-array-layout-for-cpu-convolution)
- [Q10 — Performance Staircase](#q10-performance-staircase)
- [Q11 — Loop Order and Cache Efficiency](#q11-loop-order-and-cache-efficiency)
- [Q12 — Stride Calculation for 2D Array](#q12-stride-calculation-for-2d-array)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q13 — Fortran-Order Innermost Loop](#q13-fortran-order-innermost-loop)
- [Q14 — np.ascontiguousarray Behavior](#q14-npascontiguousarray-behavior)
- [Q15 — Stride of 0 and Broadcasting](#q15-stride-of-0-and-broadcasting)
- [Q16 — CUDA Coalescing vs CPU Cache](#q16-cuda-coalescing-vs-cpu-cache)
- [Q17 — Column Sum vs Row Sum Performance](#q17-column-sum-vs-row-sum-performance)
- [Q18 — 3D Array Optimal Inner Loop](#q18-3d-array-optimal-inner-loop)
- [Q19 — np.asfortranarray Strides](#q19-npasfortranarray-strides)
- [Q20 — Cache Efficiency of np.sum vs Python Loop](#q20-cache-efficiency-of-npsum-vs-python-loop)
- [Q21 — Strides After Slicing](#q21-strides-after-slicing)
- [Q22 — C-Order vs F-Order for Matrix Multiply](#q22-c-order-vs-f-order-for-matrix-multiply)

---

> Topics: Strides, loop ordering, row vs column access, cache hierarchy, spatial/temporal locality.
> Exam frequency: **Every exam**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--optimal-loop-order-from-strides)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — Optimal Loop Order from Strides
> **Week reference:** Week 3

**Mental Model:** Innermost loop = smallest stride — sort axes by stride ascending and assign innermost first — the trap is sorting in the wrong direction (largest stride innermost) or confusing axis index with stride value.

A 4D NumPy float64 array has strides `(600, 40, 8, 200)` bytes. To maximise cache efficiency, what should be the order of the loop variables from **innermost to outermost**, where `i, j, k, l` correspond to axes 0, 1, 2, 3?

- A) `i, j, k, l`
- B) `k, j, l, i`
- C) `l, j, k, i`
- D) `k, l, j, i`

**Answer: B**

- A) Incorrect — axis 0 has the largest stride (600 bytes), so placing i innermost means every iteration jumps 600 bytes in memory. That evicts the prefetched cache line immediately and causes a miss on every access.
- B) Correct — rank axes by stride ascending: axis 2 (k, 8 bytes) < axis 1 (j, 40 bytes) < axis 3 (l, 200 bytes) < axis 0 (i, 600 bytes). Innermost to outermost: k, j, l, i. The innermost k loop walks through 8-byte steps, fitting 8 float64 elements per 64-byte cache line.
- C) Incorrect — placing l (stride 200) as the innermost loop jumps 200 bytes per iteration, loading a new cache line every 64/200 < 1 iteration and wasting most of the prefetched data.
- D) Incorrect — this puts k innermost (correct) but then l before j, reversing the ascending stride order for the middle two axes: j has stride 40 < l's stride 200, so j should be closer to the inner loop.

---

## Q2 — C-Order Stride Layout
> **Week reference:** Week 3

**Mental Model:** C-order = last axis varies fastest = smallest stride — stride of last axis = element size (8 bytes for float64), each earlier axis multiplies by the next dimension size — the trap is thinking axis 0 (outermost) has the smallest stride.

For a C-order (row-major) float64 array of shape `(5, 6, 7)`, which axis has the **smallest** stride?

- A) Axis 0 (size 5)
- B) Axis 1 (size 6)
- C) Axis 2 (size 7)
- D) All axes have the same stride

**Answer: C**

- A) Incorrect — in C-order, axis 0 is the outermost dimension and has the largest stride. Stride for axis 0 = 6 × 7 × 8 = 336 bytes; moving one step along axis 0 skips an entire 6×7 sub-array.
- B) Incorrect — axis 1 has an intermediate stride of 7 × 8 = 56 bytes; moving one step along axis 1 skips a full row of 7 elements.
- C) Correct — in C-order, the last axis (axis 2) has the smallest stride: 1 element × 8 bytes = 8 bytes. Consecutive elements along axis 2 are physically adjacent in memory, so iterating axis 2 in the innermost loop is maximally cache-friendly.
- D) Incorrect — strides are (336, 56, 8) for shape (5,6,7) in C-order. Each axis's stride = product of all later dimension sizes × element size. They differ by factors of 6 and 7 respectively.

---

## Q3 — Row Access vs Column Access
> **Week reference:** Week 3

**Mental Model:** C-order stores rows contiguously — row access = stride 8 bytes (sequential), column access = stride 8000 bytes (1 cache miss per element) — the trap is assuming equal element count means equal speed.

You have a large float64 matrix `mat` of shape `(1000, 1000)` stored in C-order. Which access pattern is faster, and why?

- A) `mat[:,0]` — column access, because it reads a contiguous block of memory
- B) `mat[0,:]` — row access, because consecutive elements are adjacent in memory
- C) Both are equally fast because the total number of elements read is the same
- D) `mat[:,0]` — column access, because NumPy optimises vertical reads automatically

**Answer: B**

- A) Incorrect — column access in a C-order array is strided, not contiguous. mat[0,0], mat[1,0], mat[2,0] are each 1000×8 = 8000 bytes apart. Every element access is a cache miss; no prefetched data is reused.
- B) Correct — row access (mat[0,:]) reads mat[0,0], mat[0,1], ..., mat[0,999], which are physically consecutive at 8-byte intervals. A single 64-byte cache line load delivers 8 elements, giving 8:1 spatial reuse.
- C) Incorrect — cache efficiency depends entirely on access pattern and stride, not element count. Column access causes ~1000 cache misses for 1000 elements; row access causes ~125 cache misses. In practice the difference is ~10× in throughput.
- D) Incorrect — NumPy does not transparently reorder or vectorise column reads to be cache-friendly. It accesses memory exactly as indexed. Column access remains cache-unfriendly regardless of NumPy version.

---

## Q4 — Cache Line Size and Prefetching
> **Week reference:** Week 3

**Mental Model:** Cache line = 64 bytes loaded atomically — for float64 (8 bytes each), one cache miss loads 64/8 = 8 consecutive elements for free — the trap is thinking only the requested element is loaded.

A CPU cache line is 64 bytes. When you access `x[0]` in a float64 array, how many consecutive elements are loaded into the cache for free?

- A) 1
- B) 4
- C) 8
- D) 16

**Answer: C**

- A) Incorrect — if only one element were loaded per access there would be no spatial prefetching benefit. The entire point of the cache line is to amortize memory latency by loading a contiguous block at once.
- B) Incorrect — 4 float64 values = 4 × 8 = 32 bytes, which is only half a cache line. The CPU always loads the full 64-byte line aligned to a 64-byte boundary, so 4 is too few.
- C) Correct — 64 bytes ÷ 8 bytes per float64 = 8 elements. Accessing x[0] loads x[0] through x[7] into cache simultaneously. Subsequent accesses to x[1]…x[7] are cache hits at near-zero cost.
- D) Incorrect — 16 float64 values = 16 × 8 = 128 bytes, which spans two 64-byte cache lines. A single cache miss loads exactly one 64-byte line, not two.

---

## Q5 — Spatial Locality
> **Week reference:** Week 3

**Mental Model:** Best spatial locality = smallest stride between consecutive accesses — consecutive elements (stride 1) maximize reuse of the 8 elements loaded per cache line — any stride ≥ 8 elements means a new cache line on every access.

Which of the following access patterns demonstrates the **best** spatial locality for a float64 array `a` of shape `(1024,)`?

- A) `a[0], a[512], a[256], a[768]` — accessing every 512th element
- B) `a[0], a[8], a[16], a[24]` — accessing every 8th element (stride 8 elements = 64 bytes)
- C) `a[0], a[1], a[2], a[3]` — accessing consecutive elements
- D) `a[1023], a[511], a[255], a[127]` — reverse strided access

**Answer: C**

- A) Incorrect — stride of 512 elements = 512 × 8 = 4096 bytes. Each access is in a completely different cache line with no overlap. Zero spatial reuse: 4 accesses require 4 cache line loads.
- B) Incorrect — stride of 8 elements = 64 bytes, which is exactly one cache line width. Each access lands at the start of a fresh cache line, discarding the 7 other elements that were prefetched. Still zero reuse.
- C) Correct — consecutive elements are 8 bytes apart. Accessing a[0] loads a[0]–a[7] into the cache. The next three accesses (a[1], a[2], a[3]) are already cached. 4 accesses cost only 1 cache line load — 8× better than options A or B.
- D) Incorrect — the elements are accessed in reverse order, but the key issue is the strides between them: 512, 256, 128 elements respectively. Each is in a different cache line, providing no spatial locality regardless of direction.

---

## Q6 — Temporal Locality and Loop Tiling
> **Week reference:** Week 3

**Mental Model:** Tiling keeps the working set inside cache across multiple passes — data loaded once is reused many times before eviction — the trap is confusing it with spatial locality (which is about sequential access, not reuse).

Loop tiling (blocking) improves performance primarily by exploiting:

- A) Spatial locality — accessing memory in sequential order within each tile
- B) Temporal locality — reusing data in the cache before it is evicted
- C) Parallelism — running multiple tiles simultaneously on different cores
- D) Vectorisation — fitting tile data into SIMD registers

**Answer: B**

- A) Incorrect — spatial locality is about accessing memory sequentially to reuse cache lines. Tiling can incidentally improve spatial locality, but that is not its primary purpose. The main benefit is keeping the working set hot across the inner-loop iterations that revisit the same data.
- B) Correct — in matrix multiplication without tiling, a full row of A and column of B must stay resident while iterating the inner loop, often exceeding L1/L2 cache. Tiling reduces the working set to a tile-sized block that fits in cache, so each element of A and B is reused many times (across the tile's inner loop) before being evicted.
- C) Incorrect — tiling is a single-threaded cache optimization. It restructures loop execution order but does not create parallel work. It must be combined with separate parallelism techniques to exploit multiple cores.
- D) Incorrect — vectorisation (SIMD) is a hardware-register level optimization applied by the compiler to individual loop iterations. Tiling operates at the loop-structure level, restructuring which iterations run together; it is independent of vectorisation.

---

## Q7 — Infer Shape from Strides
> **Week reference:** Week 3

**Mental Model:** Strides tell you element spacing, not array extent — stride of last axis gives bytes-per-element (confirming dtype), stride of axis 0 gives bytes per row (so n_cols = stride[0]/stride[1]), but n_rows is unconstrained — the trap is reading shape directly from stride values.

A float64 array has strides `(80, 8)` bytes. What is its shape in elements?

- A) `(80, 8)`
- B) `(8, 80)`
- C) `(10, 10)`
- D) Cannot be determined from strides alone

**Answer: D**

- A) Incorrect — strides are in bytes, not element counts. Axis 1 stride = 8 bytes = 1 float64 element, confirming float64 dtype. Axis 0 stride = 80 bytes = 10 float64 elements per row, confirming 10 columns. But shape (80, 8) conflates byte values with dimension sizes.
- B) Incorrect — this swaps the axes and still treats bytes as element counts. n_cols = 80/8 = 10, not 8; and n_rows is not 80.
- C) Incorrect — n_cols = 80/8 = 10 is correct (the number of columns can be derived). But n_rows cannot be inferred from strides: arrays of shape (5,10), (10,10), (100,10), or any (N,10) all produce identical strides (80, 8). Only the column count is fixed.
- D) Correct — strides encode only the step size between elements along each axis. n_cols = stride[0]/stride[1] = 80/8 = 10 is derivable, but n_rows (how many rows exist) is not encoded in the stride tuple at all. You need the shape attribute separately.

---

## Q8 — Effect of Transposing an Array
> **Week reference:** Week 3

**Mental Model:** a.T is a zero-copy view — it reverses the strides tuple without touching the data buffer — the trap is thinking transpose copies or reorders data in memory.

You have a C-order float64 array `a` of shape `(100, 200)`. After computing `b = a.T`, which statement is true?

- A) `b` is a new array with a Fortran-order (column-major) memory layout
- B) `b` has the same memory buffer as `a`, but with reversed strides
- C) `b` is contiguous in C-order because NumPy copies data during transpose
- D) `b.strides` equals `a.strides` since the underlying data is unchanged

**Answer: B**

- A) Incorrect — a.T does not produce a copy with a new memory layout. No data movement occurs. It returns a view object pointing to the same buffer; calling it "Fortran-order" is misleading because the bytes on disk are unchanged — only the stride metadata differs.
- B) Correct — a has shape (100,200) and strides (1600, 8): axis 0 jumps 200×8=1600 bytes, axis 1 jumps 8 bytes. b = a.T has shape (200,100) and strides (8, 1600): the strides are simply reversed. Same buffer, same data, different traversal metadata.
- C) Incorrect — a.T returns a non-contiguous view, not a C-contiguous copy. b.flags['C_CONTIGUOUS'] is False. To get a contiguous copy you must call np.ascontiguousarray(a.T) or a.T.copy(), which do perform data movement.
- D) Incorrect — b.strides = (8, 1600) while a.strides = (1600, 8). They are the reversal of each other. The underlying data buffer is shared, but the stride metadata is different, which is precisely how NumPy knows to traverse the array differently.

---

## Q9 — Optimal Array Layout for CPU Convolution
> **Week reference:** Week 4

**Mental Model:** CPU wants the innermost loop dimension to be contiguous in memory — for a channel-inner loop, channels must be in the last array axis (HWC) — CUDA is the opposite (CHW), so CPU and GPU have opposite optimal layouts.

You are writing a CPU-optimised image convolution where the innermost loop iterates over **channels**. Which array layout should you use for the image data?

- A) CHW (channels first) — because CUDA GPUs use this format
- B) HWC (channels last) — because channels in the innermost position have the smallest stride
- C) HCW (height × channels × width) — to balance stride across all dimensions
- D) Either CHW or HWC; the CPU treats them identically

**Answer: B**

- A) Incorrect — CHW is optimal for CUDA GPUs because of how GPU memory coalescing works (threads in a warp access the same channel across adjacent pixels). On a CPU, CHW means the innermost channel loop strides by H×W×8 bytes per step — enormous stride, constant cache misses.
- B) Correct — in HWC layout, channels are the last axis with stride = 8 bytes (one float64). The innermost channel loop steps through consecutive memory addresses, loading 8 channels per 64-byte cache line with full spatial reuse. This is why PyTorch defaults to NHWC for CPU inference.
- C) Incorrect — HCW is a non-standard layout. The innermost channel dimension would not be last (W comes after C), so the channel loop still has a stride of W×8 bytes rather than 8 bytes. No cache benefit for the channel-inner loop.
- D) Incorrect — CPU and CUDA have strictly opposite optimal layouts. On CPU: HWC (channels last, stride 8). On CUDA: CHW (channels first, coalesced across the spatial dimension). Treating them identically ignores the fundamentally different memory access patterns.

---

## Q10 — Performance Staircase
> **Week reference:** Week 3

**Mental Model:** Cache hierarchy produces discrete performance tiers — L1 → L2 → L3 → DRAM transitions create visible drops when the working set crosses each capacity threshold — the trap is attributing the staircase to software (GIL, vectorisation) rather than hardware.

When you plot MFLOP/s against working set size for a CPU-bound loop, you observe a **staircase pattern** with distinct performance drops. What causes each step down?

- A) The loop hits Python's GIL at each new array allocation threshold
- B) The working set exceeds each successive cache level (L1, L2, L3), forcing slower memory accesses
- C) NumPy switches from vectorised to scalar code for larger arrays
- D) Memory bandwidth saturates at a fixed threshold regardless of cache size

**Answer: B**

- A) Incorrect — the GIL affects multi-threaded Python execution by serializing threads. For a single-threaded CPU-bound loop (which this benchmarking scenario is), the GIL is not involved. It produces contention artifacts, not a staircase in single-thread throughput.
- B) Correct — L1 cache (~32 KB) gives maximum throughput; once the working set exceeds L1 it spills to L2 (~256 KB, ~4× slower). Exceeding L2 spills to L3 (~several MB, ~10× slower than L1). Exceeding L3 goes to DRAM (~100× slower than L1). Each boundary creates one visible step down in MFLOP/s.
- C) Incorrect — NumPy's vectorisation (SIMD path) is determined by array dtype and operation type, not array size. There is no size threshold at which NumPy degrades to scalar code. SIMD instructions operate on the same code path for all sizes.
- D) Incorrect — memory bandwidth saturation would appear as a flat plateau at the bandwidth ceiling, not a staircase. The staircase has multiple distinct levels corresponding to the three or four layers of the cache hierarchy; a single bandwidth ceiling would produce only one drop.

---

## Q11 — Loop Order and Cache Efficiency
> **Week reference:** Week 3

**Mental Model:** The array index pattern determines cache behavior, not the loop variable name — a[j, i] with j in the inner loop walks down a column (bad for C-order) even though i looks like the row variable — always trace what the inner loop does to the array subscript.

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

- A) Incorrect — the variable i is in the outer loop, but the array indexing is a[j, i], which uses i as the column index, not the row index. The inner loop increments j while i is fixed, so each iteration accesses a different row at the same column — that is column traversal, not row traversal.
- B) Correct — the inner loop increments j while i is fixed: a[0,i], a[1,i], a[2,i], ..., a[N-1,i]. In C-order, consecutive elements along axis 0 (different rows) are M×8 bytes apart. With M=1000, that is 8000 bytes between accesses — far beyond a 64-byte cache line. Every inner-loop iteration is a cache miss.
- C) Incorrect — NumPy does not analyze or reorder loop-level memory access patterns at runtime. The programmer is fully responsible for writing cache-friendly index ordering. NumPy only optimizes within individual vectorised operations (like ufuncs), not user-written loops.
- D) Incorrect — the access pattern is column-major regardless of the relative sizes of N and M. Whether N=10,M=1000 or N=1000,M=10, the inner j loop always walks down a column (axis 0) with stride M×8 bytes. The shape affects severity, not the direction of the problem.

---

## Q12 — Stride Calculation for 2D Array
> **Week reference:** Week 3

**Mental Model:** For C-order, stride[axis] = (product of all later dimension sizes) × element_size — for a (3,4) float64 array: axis 1 stride = 1×8=8, axis 0 stride = 4×8=32 — the trap is using element counts instead of bytes or confusing the formula direction.

What are the strides (in bytes) for a C-order float64 array of shape `(3, 4)`?

- A) `(4, 1)`
- B) `(8, 4)`
- C) `(32, 8)`
- D) `(24, 8)`

**Answer: C**

- A) Incorrect — (4, 1) would be the strides in elements (element counts, not bytes). For float64 each element is 8 bytes, so byte strides are 8× larger: (4×8, 1×8) = (32, 8). Strides in NumPy are always in bytes.
- B) Incorrect — axis 1 stride should be 8 bytes (one float64 element), not 4. And axis 0 stride should span one full row of 4 elements: 4×8=32 bytes, not 8. This answer swaps the correct values.
- C) Correct — axis 1 (innermost, last axis) stride = 1 element × 8 bytes/element = 8 bytes. Axis 0 stride = 4 elements per row × 8 bytes/element = 32 bytes. Strides = (32, 8). Verify: a[1,0] is at offset 1×32=32 bytes from a[0,0]; a[0,1] is at offset 1×8=8 bytes.
- D) Incorrect — 24 = 3×8, which would be the axis 0 stride for a shape (3,3) array (3 columns × 8 bytes = 24 bytes per row). For a (3,4) array the row is 4 elements wide, giving axis 0 stride = 4×8 = 32, not 24.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets loop order optimization, cache-friendly strides, C vs Fortran order, and CPU vs CUDA layout differences

---

## Q13 — Fortran-Order Innermost Loop

> **Week reference:** Week 3

A NumPy float64 array `a` of shape `(4, 5)` is created with `np.asfortranarray`. Which loop order is most cache-efficient when iterating over all elements?

- A) Outer loop over rows (axis 0), inner loop over columns (axis 1): `for i in range(4): for j in range(5)`
- B) Outer loop over columns (axis 1), inner loop over rows (axis 0): `for j in range(5): for i in range(4)`
- C) Either order is equally efficient because NumPy handles Fortran arrays transparently
- D) Outer loop over rows, inner loop over rows again — use np.nditer for automatic ordering

**Answer: B**

Fortran-order (column-major) stores elements column by column: `a[0,0], a[1,0], a[2,0], a[3,0], a[0,1], ...`. The first axis (rows) varies fastest, so its stride = 8 bytes (one float64). The second axis (columns) has stride = 4×8 = 32 bytes.

- A) Incorrect — with F-order, the innermost loop over axis 1 (columns) has a stride of 32 bytes. Each step skips 4 elements, causing cache misses. This is the optimal pattern for C-order, not F-order.
- B) Correct — iterating axis 0 (rows) in the innermost loop steps through memory 8 bytes at a time. A single 64-byte cache line covers 8 consecutive row elements, giving maximum spatial reuse.
- C) Incorrect — NumPy does not reorder Python loops. The programmer must choose the loop order to match the array's memory layout. `np.nditer` with `order='F'` can help, but plain nested loops must be ordered manually.
- D) Incorrect — `np.nditer` can iterate in F-order, but the question asks about plain Python loops. Repeating axis 0 in both loops makes no sense and is not a valid traversal strategy.

---

## Q14 — np.ascontiguousarray Behavior

> **Week reference:** Week 3

After `b = a.T` where `a` is a C-order float64 array of shape `(100, 200)`, which statement about `np.ascontiguousarray(b)` is true?

- A) It returns `b` itself unchanged, because `b` is already C-contiguous
- B) It returns a new array with shape `(200, 100)` and strides `(800, 8)`, copying data
- C) It returns a view of `b` with updated strides — no copy occurs
- D) It raises a ValueError because transposed arrays cannot be made contiguous

**Answer: B**

`b = a.T` has shape `(200, 100)` and strides `(8, 1600)` — it is not C-contiguous (the first axis stride of 8 is smaller than the second axis stride of 1600, which violates C-order). `np.ascontiguousarray(b)` detects this and creates a new C-contiguous array with the same shape `(200, 100)`.

For a C-contiguous `(200, 100)` float64 array: strides = `(100×8, 8)` = `(800, 8)`.

- A) Incorrect — `b` is not C-contiguous. `b.flags['C_CONTIGUOUS']` is False. `np.ascontiguousarray` will make a copy.
- B) Correct — the output has shape `(200, 100)` and C-order strides `(800, 8)`, produced by a data copy.
- C) Incorrect — a view cannot change the physical memory layout. Making an array contiguous always requires copying data to a new buffer when the source is non-contiguous.
- D) Incorrect — `np.ascontiguousarray` never raises on a valid array. It always succeeds, returning a copy if needed or the original if already C-contiguous.

---

## Q15 — Stride of 0 and Broadcasting

> **Week reference:** Week 3

```python
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.broadcast_to(a, (4, 3))
print(b.strides)
```

What does this print, and what does a stride of 0 mean?

- A) `(0, 8)` — the row stride is 0, meaning no new memory is allocated per row; all rows share the same data
- B) `(24, 8)` — the row stride is 24 bytes (3 float64 elements), a normal C-order 2D array
- C) `(8, 0)` — the column stride is 0, meaning all columns share the same element
- D) `(0, 0)` — all strides are 0 because broadcast arrays have no real memory layout

**Answer: A**

`np.broadcast_to` creates a read-only view that expands dimensions without copying data. The original array `a` has 3 elements in memory. Broadcasting to `(4, 3)` repeats the same row 4 times by setting the row stride to 0: moving one step along axis 0 adds 0 bytes to the pointer, so each "row" points to the same memory.

- A) Correct — strides `(0, 8)`: axis 1 steps 8 bytes (normal float64), axis 0 steps 0 bytes (same row repeated). Zero stride = virtual repetition with no data copy.
- B) Incorrect — strides `(24, 8)` would describe a real 2D array with 3 float64 columns. `broadcast_to` does not allocate new rows; it uses stride 0 for the broadcast dimension.
- C) Incorrect — a stride of 0 on axis 1 would mean all columns within a row are the same element (broadcasting a scalar across columns). Here the broadcast is across rows, so axis 0 stride is 0.
- D) Incorrect — only the broadcast axis (axis 0) has stride 0. The original data axis (axis 1) retains its normal stride of 8 bytes.

---

## Q16 — CUDA Coalescing vs CPU Cache

> **Week reference:** Week 9

A CUDA kernel processes a 2D float array stored in C-order (row-major) with shape `(H, W)`. For **coalesced memory access**, adjacent threads in a warp should access adjacent memory addresses. How should thread indices map to array indices?

- A) `threadIdx.x` → row index, `threadIdx.y` → column index — threads in a warp access different rows
- B) `threadIdx.x` → column index, `threadIdx.y` → row index — threads in a warp access adjacent columns in the same row
- C) Layout does not matter for CUDA; the L2 cache handles coalescing automatically
- D) `threadIdx.x` → row index only; column access should always be sequential within a single thread

**Answer: B**

In CUDA, threads within a warp execute in lockstep. A warp consists of 32 threads that differ only in `threadIdx.x`. For coalesced access, these 32 threads should access 32 consecutive memory addresses simultaneously. In a C-order array, consecutive addresses are along the last axis (columns). Therefore `threadIdx.x` should map to the column index so adjacent threads in a warp read adjacent columns — 32 consecutive float32/float64 values = one or two 128-byte cache lines fetched in one transaction.

- A) Incorrect — if `threadIdx.x` maps to rows, threads in a warp access `array[0,col]`, `array[1,col]`, `array[2,col]` etc. In C-order these are W×4 bytes apart — strided access, one memory transaction per thread, 32× memory traffic overhead.
- B) Correct — `threadIdx.x` → column means warp threads access `array[row, 0]`, `array[row, 1]`, ..., `array[row, 31]`: 32 consecutive bytes, fully coalesced into a single memory transaction.
- C) Incorrect — the L2 cache helps but cannot eliminate the penalty of uncoalesced access. Uncoalesced warps issue up to 32 separate memory transactions instead of 1, and this cannot be transparently fixed by hardware.
- D) Incorrect — restricting column access to a single thread eliminates all parallelism across the column dimension. The whole point of GPU parallelism is that many threads access different columns simultaneously.

---

## Q17 — Column Sum vs Row Sum Performance

> **Week reference:** Week 3

For a C-contiguous float64 array `a` of shape `(1000, 1000)`, which operation is faster?

- A) `a.sum(axis=0)` — summing along axis 0 produces a row vector by visiting each column
- B) `a.sum(axis=1)` — summing along axis 1 produces a column vector by visiting each row
- C) Both are equally fast because NumPy uses BLAS routines for both reductions
- D) `a.sum(axis=0)` — DRAM bandwidth is higher for vertical access patterns

**Answer: B**

`a.sum(axis=1)` reduces along axis 1 (columns within each row). For each of the 1000 rows, NumPy reads 1000 consecutive float64 elements — sequential access with stride 8 bytes. Each 64-byte cache line delivers 8 useful values.

`a.sum(axis=0)` reduces along axis 0 (rows within each column). For each of the 1000 columns, NumPy reads `a[0,j], a[1,j], ..., a[999,j]` — elements spaced 1000×8 = 8000 bytes apart. Each cache line load yields only 1 useful element.

- A) Incorrect — axis=0 sum accesses each column in sequence, requiring a stride-8000-byte jump per element. This is cache-unfriendly and causes ~125,000 extra cache misses compared to row-wise access.
- B) Correct — axis=1 sum iterates along rows (stride 8 bytes), maximally reusing every loaded cache line. This is the cache-friendly direction for C-order arrays.
- C) Incorrect — NumPy does use optimized reduction code, but it cannot eliminate the fundamental stride difference. The axis=0 reduction still accesses memory with a large stride regardless of the underlying routine.
- D) Incorrect — DRAM bandwidth is symmetric; there is no hardware advantage for vertical access. The bottleneck is cache-line utilization, not DRAM direction.

---

## Q18 — 3D Array Optimal Inner Loop

> **Week reference:** Week 3

You have a C-contiguous float64 array `vol` of shape `(D, H, W)` = `(32, 64, 128)`. You want to compute the sum of all elements with a triple nested loop. Which loop order is most cache-efficient?

- A) Outermost: W, middle: H, innermost: D
- B) Outermost: D, middle: H, innermost: W
- C) Outermost: H, middle: D, innermost: W
- D) Outermost: W, middle: D, innermost: H

**Answer: B**

For a C-contiguous array of shape `(D, H, W)`, strides are `(H×W×8, W×8, 8)`. The last axis (W) has the smallest stride (8 bytes). Cache-efficient traversal places the innermost loop over the axis with the smallest stride.

Strides: axis 0 (D) = 64×128×8 = 65536 bytes, axis 1 (H) = 128×8 = 1024 bytes, axis 2 (W) = 8 bytes.

Optimal order innermost to outermost: W (stride 8) → H (stride 1024) → D (stride 65536).

- A) Incorrect — placing W outermost and D innermost means the innermost loop strides 65536 bytes per step — one cache miss per element, the worst possible case.
- B) Correct — outermost D, middle H, innermost W. The innermost W loop walks through memory at 8-byte intervals, loading 8 elements per 64-byte cache line with perfect spatial reuse.
- C) Incorrect — innermost W is correct, but middle D has a larger stride (65536 bytes) than H (1024 bytes). The correct middle axis is H, not D.
- D) Incorrect — innermost H has stride 1024 bytes, skipping 128 elements per step. This is much worse than innermost W (stride 8 bytes).

---

## Q19 — np.asfortranarray Strides

> **Week reference:** Week 3

For a float64 array `a = np.asfortranarray(np.zeros((3, 5)))`, what are `a.strides`?

- A) `(40, 8)` — row stride = 5 elements × 8 bytes, column stride = 8 bytes (C-order)
- B) `(8, 24)` — row stride = 8 bytes (one element), column stride = 3 elements × 8 bytes (F-order)
- C) `(8, 8)` — both strides are 8 bytes because all elements are contiguous
- D) `(5, 3)` — strides equal the shape dimensions

**Answer: B**

Fortran-order (column-major) stores elements column by column: `a[0,0], a[1,0], a[2,0], a[0,1], a[1,1], ...`. Moving one step along axis 0 (rows) advances 1 element = 8 bytes. Moving one step along axis 1 (columns) advances the number of rows = 3 elements = 24 bytes.

Strides for F-order `(3, 5)` float64: `(8, 3×8)` = `(8, 24)`.

- A) Incorrect — `(40, 8)` are the C-order strides for shape `(3, 5)`: axis 0 stride = 5×8 = 40, axis 1 stride = 8. Fortran-order reverses which axis has the smallest stride.
- B) Correct — F-order strides `(8, 24)`: axis 0 (rows) varies fastest with stride 8, axis 1 (columns) has stride 24.
- C) Incorrect — if both strides were 8, adjacent elements along both axes would overlap in memory, which is impossible for a 2D array (each element has a unique address).
- D) Incorrect — strides are in bytes, not element counts. The shape dimensions `(5, 3)` are irrelevant to the stride formula; and strides are never equal to dimension sizes except by coincidence.

---

## Q20 — Cache Efficiency of np.sum vs Python Loop

> **Week reference:** Week 3

For a large C-contiguous 2D float64 array `a`, which is faster: `np.sum(a)` or a Python double for-loop summing all elements?

- A) The Python loop, because it avoids NumPy overhead and accesses elements directly
- B) `np.sum(a)` by a large margin, because it uses vectorized C code with SIMD and cache-friendly sequential access
- C) Both are equally fast for large arrays because memory bandwidth is the bottleneck for both
- D) The Python loop with `j` in the outer loop and `i` in the inner loop, because column-major access is faster

**Answer: B**

`np.sum(a)` iterates over the flat contiguous buffer in C with SIMD (AVX/SSE) instructions, processing 4–8 float64 values per clock cycle. A Python double for-loop has: (1) Python interpreter overhead per iteration (~50–100 ns), (2) boxing/unboxing of each float, and (3) no SIMD vectorization. For a 1000×1000 array, the Python loop runs ~1000× slower.

- A) Incorrect — Python loops have massive per-iteration overhead (interpreter bytecode dispatch, object boxing). NumPy's C-level loop with SIMD is orders of magnitude faster regardless of "overhead."
- B) Correct — `np.sum` uses contiguous sequential access (stride 8) with SIMD instructions, achieving near-peak memory bandwidth efficiency.
- C) Incorrect — memory bandwidth is the bottleneck only when the computation is trivially fast relative to data movement. Python loop overhead dominates long before memory bandwidth; the loop is CPU-bound in the interpreter, not memory-bound.
- D) Incorrect — Python loop with `j` outer, `i` inner on a C-order array accesses columns (stride = num_cols × 8 bytes) in the innermost loop — cache-unfriendly. Even if cache-friendly, it would still be ~1000× slower than `np.sum` due to Python overhead.

---

## Q21 — Strides After Slicing

> **Week reference:** Week 3

```python
import numpy as np
a = np.zeros((10, 20), dtype=np.float64)  # strides: (160, 8)
b = a[::2, :]   # every other row
```

What are `b.strides`?

- A) `(80, 8)` — row stride halved because every other row is selected
- B) `(160, 8)` — row stride unchanged because slicing returns a view
- C) `(320, 8)` — row stride doubled because every other row is skipped
- D) `(160, 4)` — column stride halved because the column count is unchanged

**Answer: C**

Slicing with `[::2]` along axis 0 returns a view where consecutive "rows" of `b` are two original rows apart. In the original array, two rows are 160 bytes apart. Skipping every other row means `b[0]` → `a[0]`, `b[1]` → `a[2]`, `b[2]` → `a[4]` — each step jumps 2×160 = 320 bytes.

Shape of `b`: `(5, 20)`. Strides of `b`: `(320, 8)`.

- A) Incorrect — a stride of 80 would mean rows are closer together than in the original, which is impossible when skipping rows. Slicing cannot reduce stride below the original element spacing.
- B) Incorrect — stride 160 would mean adjacent rows of `b` are 160 bytes apart, implying no rows were skipped. But `b[1]` corresponds to `a[2]`, which is 2×160 = 320 bytes from `a[0]`.
- C) Correct — `::2` on axis 0 doubles the row stride from 160 to 320 bytes. No data is copied; this is a view with modified stride metadata.
- D) Incorrect — the column stride (axis 1) is always determined by the element size (8 bytes for float64) and does not change when rows are sliced. Shape along axis 1 remains 20 columns.

---

## Q22 — C-Order vs F-Order for Matrix Multiply

> **Week reference:** Week 3

You are multiplying two square float64 matrices `A` (C-order) and `B` (F-order) using a naive triple loop with `k` as the innermost variable:

```python
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i,j] += A[i,k] * B[k,j]
```

For `A` in C-order and `B` in F-order, what is the cache behavior of the innermost `k` loop?

- A) Both `A[i,k]` and `B[k,j]` are accessed with stride 8 bytes — fully cache-efficient
- B) `A[i,k]` is row-wise (stride 8), but `B[k,j]` is also stride 8 because F-order stores columns contiguously
- C) `A[i,k]` is row-wise (stride 8), but `B[k,j]` accesses a column of B which in F-order is contiguous (stride 8)
- D) Both are cache-unfriendly because mixed C/F order always causes cache thrashing

**Answer: C**

For C-order `A` of shape `(N, N)`: strides = `(N×8, 8)`. The innermost `k` loop varies the column index of A: `A[i, 0], A[i, 1], ..., A[i, N-1]` — stride 8 bytes (row-wise, cache-friendly).

For F-order `B` of shape `(N, N)`: strides = `(8, N×8)`. The innermost `k` loop varies the row index of B: `B[0, j], B[1, j], ..., B[N-1, j]` — stride 8 bytes (column access in F-order is contiguous, cache-friendly).

This is actually the one case where mixed C/F order produces good cache behavior for a naive `ijk` loop.

- A) Incorrect in reasoning — the statement is correct that both are stride 8, but saying "fully cache-efficient" without explaining why B is stride 8 (F-order column contiguity) misses the key insight.
- B) Incorrect phrasing — this describes the same result as C but imprecisely. "Row-wise" for B is wrong; B[k,j] with varying k accesses a column of B, not a row.
- C) Correct — `A[i,k]` varying k is a row of C-order A (stride 8). `B[k,j]` varying k is a column of F-order B, which in F-order is physically contiguous (stride 8). Both are cache-friendly.
- D) Incorrect — mixed order does not always cause thrashing. Here the access pattern happens to align perfectly with each array's layout: C-order A accessed row-wise, F-order B accessed column-wise.

---
