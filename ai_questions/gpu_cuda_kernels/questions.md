# GPU / CUDA Kernels — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Best Block Shape for Coalescing (Row-Major)](#q1-best-block-shape-for-coalescing-row-major)
- [Q2 — Why 1×256 Is Best for Row-Major Arrays](#q2-why-1256-is-best-for-row-major-arrays)
- [Q3 — Worst Block Shape for Row-Major Coalescing](#q3-worst-block-shape-for-row-major-coalescing)
- [Q4 — Grid Size Calculation (1D)](#q4-grid-size-calculation-1d)
- [Q5 — 2D Grid Dimension Calculation](#q5-2d-grid-dimension-calculation)
- [Q6 — Maximum Threads Per Block and Warp Multiples](#q6-maximum-threads-per-block-and-warp-multiples)
- [Q7 — Optimal Array Layout for CUDA Convolution (CHW)](#q7-optimal-array-layout-for-cuda-convolution-chw)
- [Q8 — CPU vs CUDA Layout Are Opposite](#q8-cpu-vs-cuda-layout-are-opposite)
- [Q9 — Bounds Check Requirement](#q9-bounds-check-requirement)
- [Q10 — `@cuda.jit(device=True)` Decorator](#q10-cudajitdevicetrue-decorator)
- [Q11 — `cuda.grid(1)` Formula](#q11-cudagrid1-formula)
- [Q12 — Warp Size and SIMT Execution](#q12-warp-size-and-simt-execution)
- [Q13 — Worst Block Shape for Row-Major Read (Performance Trap)](#q13-worst-block-shape-for-row-major-read-performance-trap)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q14 — Reading nsys Profiler Output](#q14-reading-nsys-profiler-output)
- [Q15 — Numba Automatic Transfer Count (Naive)](#q15-numba-automatic-transfer-count-naive)
- [Q16 — Optimised Transfer Count (Pre-allocated Device Arrays)](#q16-optimised-transfer-count-pre-allocated-device-arrays)
- [Q17 — GPU vs CPU Total Time Calculation](#q17-gpu-vs-cpu-total-time-calculation)
- [Q18 — `cuda.syncthreads()` in Parallel Reduction](#q18-cudasyncthreads-in-parallel-reduction)
- [Q19 — Static vs Dynamic Scheduling for GPU Jobs](#q19-static-vs-dynamic-scheduling-for-gpu-jobs)
- [Q20 — GPU Amortisation Over Many Iterations](#q20-gpu-amortisation-over-many-iterations)
- [Q21 — Warp Divergence with if/else](#q21-warp-divergence-with-ifelse)
- [Q22 — 3D Array Coalescing: Which Block Shape?](#q22-3d-array-coalescing-which-block-shape)
- [Q23 — GPU Queue and Resource Flags (BSUB)](#q23-gpu-queue-and-resource-flags-bsub)
- [Q24 — `cuda.to_device` vs `cuda.device_array_like` Transfers](#q24-cudato_device-vs-cudadevice_array_like-transfers)
- [Q25 — Transfer Count with Mixed NumPy and Device Arrays](#q25-transfer-count-with-mixed-numpy-and-device-arrays)

---

> Topics: Thread blocks, warp coalescing, grid dimensions, memory access patterns, CPU vs CUDA layout rules.
> Exam frequency: **Every exam** — 4+ questions per exam.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions (Q1–Q13)](#q1--best-block-shape-for-coalescing-row-major)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice (Q14–Q25)](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — Best Block Shape for Coalescing (Row-Major)

> **Week reference:** Week 9

**Mental Model:** With `row, col = cuda.grid(2)`, row=x-dim and col=y-dim. Adjacent threads (Thread ID = threadIdx.x + threadIdx.y*blockDim.x) differ by 1 in threadIdx.x → **row varies, not col**. For coalesced access of `A[row, col]` (row-major), you need col to vary → need blockDim.x=1 so threadIdx.x is stuck at 0 and threadIdx.y (col) cycles instead. Block (1, N) achieves this. Trap: thinking col naturally varies in warps — it only does when blockDim.x=1.

A CUDA kernel uses `row, col = cuda.grid(2)` to index a row-major 2D array `A[row, col]`. With `row = x-dim`, adjacent threads differ by 1 in `row`. Which block shape makes `col` vary across warp threads instead, achieving coalesced access?

- A) (256, 1) — 256 rows, 1 column
- B) (16, 16) — 16 rows, 16 columns
- C) (1, 256) — 1 row, 256 columns
- D) (32, 32) — 32 rows, 32 columns

**Answer: C**

- A) Incorrect — with shape (256, 1), all 256 threads share the same `col` but differ in `row`. In row-major order, adjacent rows are `row_width` elements apart, so consecutive threads access memory with a stride of `row_width` → worst possible strided access. Parallelism exists but coalescing is zero.
- B) Incorrect — with shape (16, 16), one warp spans threads 0–15 (all in row 0, cols 0–15) and threads 16–31 (all in row 1, cols 0–15). Row 0 and row 1 are `row_width` apart in memory, so the warp issues two separate cache-line fetches instead of one → partial coalescing.
- C) Correct — with shape (1, 256), all 256 threads share the same `row` and have `col` values 0, 1, 2, ..., 255. In row-major order these are literally adjacent bytes/elements: `A[row, 0], A[row, 1], ..., A[row, 255]` → fully coalesced single transaction per warp.
- D) Incorrect — 32×32 = 1024 threads is valid and warp threads 0–31 (same `threadIdx.y`, consecutive `threadIdx.x`) do access sequential columns, so coalescing is actually good; (1, 256) is still the idiomatic choice for purely column-parallel work and maximally explicit about access pattern.

---

## Q2 — Why 1×256 Is Best for Row-Major Arrays

> **Week reference:** Week 9

**Mental Model:** Coalescing requires a warp's 32 threads to map to 32 consecutive addresses. With `row, col = cuda.grid(2)`, **row is the fast-varying index** (x-dim, threadIdx.x increments first). For `A[row, col]` in row-major memory, col (last axis) is what's adjacent. So you need to override the default: use block (1, 256) so blockDim.x=1 → row is locked → col varies via threadIdx.y → 32 consecutive col addresses → coalesced.

For a row-major 2D array accessed as `A[row, col]`, why does a block shape of `(1, 256)` give better memory coalescing than `(16, 16)`?

- A) Because 256 is a larger number of threads than 16×16=256, so there is more parallelism
- B) Because all 256 threads in a warp share the same row, so their col indices are consecutive → sequential memory addresses
- C) Because 1×256 uses fewer registers per thread, reducing register pressure
- D) Because row-major arrays store data column-by-column, so varying col accesses the same cache line

**Answer: B**

- A) Incorrect — 1×256 and 16×16 both have exactly 256 threads total; parallelism (total work) is identical. The difference is purely in how those threads map to memory addresses.
- B) Correct — with block shape (1, 256), every thread in a warp has the same `row` and a unique, consecutive `col` value. This means they access `A[row, col], A[row, col+1], ..., A[row, col+31]` which are contiguous in row-major layout → the hardware merges all 32 accesses into one memory transaction.
- C) Incorrect — register usage is determined by kernel logic (number of variables, loop depth), not by the block shape chosen at launch time.
- D) Incorrect — row-major arrays store data row-by-row (all elements of row 0, then row 1, etc.); varying `col` does access contiguous memory, which is exactly why (1, 256) is good — but the premise of the distractor is backwards.

---

## Q3 — Worst Block Shape for Row-Major Coalescing

> **Week reference:** Week 9

**Mental Model:** The worst case is when a warp's 32 threads all access the same column but different rows — each element is a full `row_width` stride apart in memory, forcing 32 separate cache-line fetches.

A kernel accesses a row-major array `x[row, col]` where `row, col = cuda.grid(2)`. Which block configuration gives the **worst** memory access performance?

- A) (1, 256)
- B) (16, 16)
- C) (256, 1)
- D) (32, 8)

**Answer: C**

- A) Incorrect — (1, 256) gives the best coalescing: all warp threads differ only in `col`, accessing `x[row, 0], x[row, 1], ..., x[row, 31]` — 32 consecutive addresses, one transaction.
- B) Incorrect — (16, 16) gives medium performance. Threads 0–15 access row 0 cols 0–15, threads 16–31 access row 1 cols 0–15. Two separate memory fetches are needed, but each fetch is still a contiguous 16-element burst within a row.
- C) Correct — with (256, 1), warp threads 0–31 all share `col = 0` but differ in `row`. They access `x[0, col], x[1, col], ..., x[31, col]`. For a 1000-column array each address is 1000 elements (4000 bytes) apart — 32 fully separate cache-line fetches, the absolute worst case.
- D) Incorrect — (32, 8) is moderate. Within a warp, 8 consecutive threads share the same `row` for their 8 `col` values; the warp spans 4 different rows but each row contributes a contiguous 8-element burst, giving partial coalescing.

---

## Q4 — Grid Size Calculation (1D)

> **Week reference:** Week 9

**Mental Model:** Always round UP when converting elements to blocks — `(N + tpb - 1) // tpb`. Rounding down leaves the last partial block's elements unprocessed; rounding up creates a few harmless extra threads that get filtered out by the bounds check.

You have an array of size N=500 and want to process it with a CUDA kernel using 32 threads per block. How many blocks do you need?

- A) 15
- B) 15.625
- C) 16
- D) 32

**Answer: C**

- A) Incorrect — 15 blocks × 32 threads = 480 threads, covering only indices 0–479. Elements 480–499 (20 elements) would never be processed — silent data loss. Always round up to avoid missing the last partial tile.
- B) Incorrect — fractional blocks are physically impossible; the CUDA runtime requires an integer block count. The fractional result 15.625 is the intermediate value; you must apply integer ceiling to get 16.
- C) Correct — using the formula `bpg = (N + tpb - 1) // tpb`: `(500 + 31) // 32 = 531 // 32 = 16`. This launches 16 × 32 = 512 threads; the extra 12 threads (indices 500–511) are suppressed by the `if i < n` bounds check in the kernel.
- D) Incorrect — 32 blocks × 32 threads = 1024 threads for 500 elements; this wastes resources (over 2× the necessary threads) and is a sign of a miscalculation, not a valid ceiling.

---

## Q5 — 2D Grid Dimension Calculation

> **Week reference:** Week 9

**Mental Model:** Apply the ceiling formula `(N + tpb - 1) // tpb` independently to each dimension. Both row and column dimensions use the same formula — a common mistake is applying it only to one axis.

You want to process a 200×200 output array with a 2D CUDA kernel using a block size of (16, 16). How many thread blocks are launched in total?

- A) 144 (12×12)
- B) 156 (12×13)
- C) 169 (13×13)
- D) 196 (14×14)

**Answer: C**

- A) Incorrect — `ceil(200/16) = ceil(12.5) = 13`, not 12. Using 12 blocks per dimension gives 12×16 = 192 threads, leaving 8 rows and 8 columns unprocessed — roughly 3000 output elements would be skipped.
- B) Incorrect — both dimensions require the same calculation since the array is square (200×200) and the block is square (16×16). Both produce `ceil(200/16) = 13`, so the result must be symmetric: 13×13, not 12×13.
- C) Correct — `(200 + 15) // 16 = 215 // 16 = 13` in both the x (columns) and y (rows) direction → 13 × 13 = 169 total blocks. The grid launches 208×208 = 43,264 threads total; the extra 208×8 + 200×8 edge threads are filtered by bounds checks.
- D) Incorrect — 14 blocks per dimension = 14×16 = 224 threads per axis, overshooting by 24 rows/cols. While harmless (bounds checks catch them), it wastes ~7% extra thread launches unnecessarily; 13 is the exact ceiling.

---

## Q6 — Maximum Threads Per Block and Warp Multiples

> **Week reference:** Week 9

**Mental Model:** The GPU hardware limit is 1024 threads per block. Warp size is 32 — if your block size isn't a multiple of 32, the last warp is only partially filled, wasting GPU lanes.

Which of the following statements about CUDA thread block sizing is correct?

- A) The maximum threads per block is 256; always use exactly 256
- B) The maximum threads per block is 1024; choose a multiple of 32 for best warp efficiency
- C) The maximum threads per block is 512; warp size is 16
- D) Thread block size should always equal the warp size of 32

**Answer: B**

- A) Incorrect — the hardware maximum is 1024 threads per block, not 256. Using 256 is a valid and common choice (8 warps per block), but it is not the limit. Claiming "always use exactly 256" is also wrong — the optimal size depends on register and shared-memory usage per thread.
- B) Correct — CUDA GPUs (all architectures from Kepler through Hopper) support up to 1024 threads per block. The warp size is 32; choosing block sizes that are multiples of 32 (e.g., 128, 256, 512) ensures every warp is fully populated, maximising occupancy and avoiding idle SIMT lanes.
- C) Incorrect — both numbers are wrong: maximum threads per block is 1024 (not 512) and warp size is 32 (not 16). These incorrect values would lead to half-filled warps and artificially constrained block counts.
- D) Incorrect — using only 32 threads (one warp) per block is technically valid but extremely inefficient. It limits the occupancy to one warp per SM, leaving most of the SM's compute resources idle. Typical practical choices are 128–512 threads per block.

---

## Q7 — Optimal Array Layout for CUDA Convolution (CHW)

> **Week reference:** Week 9

**Mental Model:** For coalescing, the axis that varies fastest across warp threads must be the innermost (last) axis in memory. In a CUDA convolution, threads vary in the spatial `col` (width) axis — so width must be the last dimension: channels × height × **width** = CHW.

A CUDA convolution kernel has threads that vary in `col` (spatial width axis) within a warp. The input tensor has shape `(channels, height, width)` (CHW). Why is CHW preferred over HWC for this kernel?

- A) CHW uses less memory than HWC for the same data
- B) CHW places the spatial axes (height, width) as the last two dimensions, so threads varying in `col` access consecutive memory addresses
- C) CHW is preferred because the CPU also uses CHW, making data transfer simpler
- D) CHW has fewer cache conflicts because channels are stored in separate memory pages

**Answer: B**

- A) Incorrect — CHW and HWC are two orderings of the same three axes; they store exactly the same data and use identical amounts of memory. Layout is purely about access pattern, not storage size.
- B) Correct — in CHW layout, for a fixed `(channel, row)`, all `width` elements are stored consecutively: `data[c, row, 0], data[c, row, 1], ..., data[c, row, W-1]`. Warp threads differing in `col` therefore access adjacent addresses → one coalesced transaction. In HWC, `data[row, col, 0..C-1]` interleaves channels, so threads varying in `col` jump by `C` elements each → strided, uncoalesced.
- C) Incorrect — this is backwards. CPU code (PyTorch with `channels_last`, most image libraries) typically prefers HWC (channels last / NHWC) because the CPU's inner loop often iterates over channels. CPU and CUDA prefer **opposite** layouts.
- D) Incorrect — cache conflicts (in shared memory) depend on access patterns and bank indexing, not on which dimension holds channels. Storing channels in separate pages has no direct effect on L2 cache conflicts.

---

## Q8 — CPU vs CUDA Layout Are Opposite

> **Week reference:** Week 9

**Mental Model:** The rule of thumb: whichever axis your innermost loop (or warp thread) iterates over must be the last array dimension. CPU: inner loop over channels → HWC. GPU warp: threads vary in spatial col → CHW. They are always opposite.

When optimising memory layout for image processing, which statement correctly describes the difference between CPU and CUDA preferences?

- A) Both CPU and CUDA prefer HWC (height × width × channels) layout for best cache performance
- B) Both CPU and CUDA prefer CHW (channels × height × width) layout
- C) CPU prefers HWC (channels last) because its inner loop iterates over channels; CUDA prefers CHW (channels first) because threads vary in spatial position
- D) CPU prefers CHW; CUDA prefers HWC because GPU memory is column-major

**Answer: C**

- A) Incorrect — CPU and CUDA have opposite optimal layouts. If they both preferred HWC, there would be no layout conversion needed when moving data between host and device — but in practice, this conversion is often necessary precisely because they differ.
- B) Incorrect — CPU benefits from HWC (channels last), not CHW. In a CPU pixel-processing loop, the inner loop typically iterates over the 3 or 4 color channels at a fixed spatial location `(i, j)`, so HWC layout `data[i, j, k]` keeps those channels contiguous in memory.
- C) Correct — on the CPU, the canonical inner loop is `for k in channels: process(data[i, j, k])`, making HWC cache-friendly. On a CUDA GPU, warp threads differ in the spatial column index, so CHW layout `data[k, i, j]` places the width axis last, making those thread accesses contiguous → coalesced.
- D) Incorrect — GPU memory is not column-major; CUDA global memory is row-major (C order), same as CPU numpy arrays. The real reason CUDA prefers CHW is warp-level spatial coalescing, not a fundamental memory-order difference between CPU and GPU.

---

## Q9 — Bounds Check Requirement

> **Week reference:** Week 9

**Mental Model:** Grid size always rounds up → some threads in the last block map to indices ≥ N. Without `if i < n`, those threads execute `out[i] = ...` on unallocated memory — silent data corruption or a segfault. The bounds check is mandatory, not optional.

A 1D CUDA kernel is launched with N=100 elements and 32 threads per block. Why is the following bounds check necessary inside the kernel?

```python
i = cuda.grid(1)
if i < n:
    out[i] = process(inp[i])
```

- A) It is not necessary; CUDA automatically ignores out-of-bounds threads
- B) It prevents threads from writing to out-of-bounds memory because the grid size is rounded up and some threads map to indices ≥ N
- C) It is needed because cuda.grid(1) can return negative values
- D) It ensures only the first warp executes, improving performance

**Answer: B**

- A) Incorrect — CUDA does NOT automatically skip out-of-bounds threads. Every thread in every block executes the kernel body unless explicitly guarded. Threads with `i ≥ n` would access `inp[100], inp[101], ..., inp[127]` and write to `out[100..127]` — undefined behaviour, potential memory corruption.
- B) Correct — with N=100 and tpb=32, the formula `ceil(100/32) = 4` launches 4 blocks = 128 threads. Threads with global index 100–127 (the last 28 threads of block 3) have no valid data to process. The `if i < n` guard ensures they do nothing, safely exiting without touching memory.
- C) Incorrect — `cuda.grid(1)` computes `blockIdx.x * blockDim.x + threadIdx.x`, all of which are non-negative integers. The result is always ≥ 0; negative values are not possible.
- D) Incorrect — the bounds check is a per-thread conditional evaluated independently by every thread. It does not restrict execution to one warp or affect which warps are scheduled; it merely causes out-of-bounds threads to skip the body.

---

## Q10 — `@cuda.jit(device=True)` Decorator

> **Week reference:** Week 9

**Mental Model:** Think of device functions as GPU-side helper functions. Just like a C `__device__` function, they are compiled for the GPU and can only be called from other GPU code — calling them from Python (host) will raise an error.

What does the `@cuda.jit(device=True)` decorator do in Numba CUDA?

- A) It compiles the function to run on the CPU but allows it to be called from a CUDA kernel
- B) It marks the function as a device function: callable from GPU kernels but NOT from host (Python) code
- C) It enables the function to launch child kernels (dynamic parallelism)
- D) It makes the function execute on the device once per block instead of once per thread

**Answer: B**

- A) Incorrect — `device=True` compiles the function for the GPU device, not the CPU. The opposite would be a regular Python/Numba CPU function. Device functions are inlined into the calling GPU kernel at the PTX level.
- B) Correct — a device function is compiled into GPU machine code and can only be invoked from within a `@cuda.jit` kernel or another device function. Calling it directly from Python host code raises a `TypeError` because there is no CPU entry point — it is purely a GPU subroutine.
- C) Incorrect — dynamic parallelism (launching a kernel from within a kernel using `cuda.jit` with `device_launch_parameters`) is a separate, advanced feature requiring specific GPU architecture support. It has nothing to do with `device=True`.
- D) Incorrect — device functions execute once per thread call, just like any subroutine. They have no special per-block execution semantics; if 256 threads each call the device function, it runs 256 times in parallel.

---

## Q11 — `cuda.grid(1)` Formula

> **Week reference:** Week 9

**Mental Model:** Each block holds `blockDim.x` threads. Block 0 owns indices 0..(blockDim-1), block 1 owns blockDim..(2·blockDim-1), etc. Global index = block offset + local offset = `blockIdx.x * blockDim.x + threadIdx.x`.

Which expression correctly computes the global thread index `i` in a 1D CUDA kernel, equivalent to `cuda.grid(1)`?

- A) `cuda.blockIdx.x + cuda.blockDim.x + cuda.threadIdx.x`
- B) `cuda.threadIdx.x * cuda.blockDim.x + cuda.blockIdx.x`
- C) `cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x`
- D) `cuda.blockDim.x * cuda.gridDim.x + cuda.threadIdx.x`

**Answer: C**

- A) Incorrect — using addition instead of multiplication between `blockIdx.x` and `blockDim.x` produces wrong results. For example, block 2 with 32 threads/block: `2 + 32 + 0 = 34` but the correct first index is `2 × 32 = 64`. Multiple blocks would produce overlapping and incorrect indices.
- B) Incorrect — `threadIdx.x` and `blockIdx.x` are swapped. This formula computes `threadIdx.x * blockDim.x + blockIdx.x`, which gives non-unique values: thread 1 in block 0 and thread 0 in block 1 would both yield index 1 (with blockDim=4: `1×4+0=4` vs `0×4+1=1` — actually different, but logically the axes are inverted and the global mapping is wrong).
- C) Correct — `blockIdx.x * blockDim.x + threadIdx.x` is the standard formula. With blockDim=32: block 0 gives indices 0–31 (`0×32 + 0..31`), block 1 gives 32–63 (`1×32 + 0..31`), block 2 gives 64–95, etc. Every thread gets a unique consecutive global index.
- D) Incorrect — `gridDim.x` is the total number of blocks in the grid (a constant for all threads), not the index of the current block. This formula computes the same large constant offset for every thread plus `threadIdx.x`, producing 32 distinct indices at one large offset — not a useful global index.

---

## Q12 — Warp Size and SIMT Execution

> **Week reference:** Week 9

**Mental Model:** A warp = 32 threads that move in lockstep through instructions (SIMT). If-else divergence within a warp forces sequential execution of both branches — the performance trap is O(branches) instead of O(1) for divergent code.

What is the CUDA warp size, and what does it mean for kernel execution?

- A) Warp size = 64; threads 0–63 execute the same instruction simultaneously on an SM
- B) Warp size = 32; threads 0–31 execute the same instruction simultaneously (SIMT); divergent branches serialise
- C) Warp size = 32; each thread in a warp executes independently on separate cores with no synchronisation
- D) Warp size = 16; this is why block sizes should be multiples of 16

**Answer: B**

- A) Incorrect — the warp size on all NVIDIA GPU architectures (Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Hopper) is 32, not 64. AMD GPUs use a wavefront of 64, which may be the source of this confusion, but for NVIDIA/CUDA the answer is always 32.
- B) Correct — a warp consists of 32 threads that are dispatched together to the SM's SIMT execution units. They execute the same instruction at the same time (Single Instruction Multiple Threads). If an `if/else` causes threads to diverge, the SM serialises both branches: all threads execute the `if` path (inactive threads masked), then all execute the `else` path — effectively halving throughput in the worst case.
- C) Incorrect — threads within a warp are NOT independent; they execute in lockstep. The key implication is branch divergence: if threads in the same warp take different code paths, both paths are serialised rather than truly running in parallel. Independence is only true at the warp level, not the thread level.
- D) Incorrect — warp size is 32, not 16. The practical implication is that block sizes should be multiples of **32** (e.g., 64, 128, 256, 512) to avoid launching partially-filled warps where some SIMT lanes are idle.

---

## Q13 — Worst Block Shape for Row-Major Read (Performance Trap)

> **Week reference:** Week 9

**Mental Model:** "More threads = more parallelism" is true, but coalescing is a separate concern. (256, 1) gives 256-way parallelism AND 256-way serialised memory — you get the work done, just very slowly because every warp issues 32 separate cache-line fetches instead of 1.

A kernel reads `x[row, col]` where `row, col = cuda.grid(2)` and `x` is stored in row-major (C) order. A colleague suggests using block shape `(256, 1)` because "256 threads per block maximises parallelism". What is wrong with this reasoning?

- A) Nothing — (256, 1) maximises parallelism AND gives the best memory coalescing for row-major arrays
- B) Block size 256 exceeds the maximum threads per block of 128
- C) With (256, 1) all threads in a warp have different row values and the same col value, so they access `x[0, col], x[1, col], ..., x[255, col]` — strided access across rows (stride = row_length), which is the worst possible pattern for coalescing
- D) (256, 1) is only bad because it does not use a multiple of the warp size 32

**Answer: C**

- A) Incorrect — (256, 1) does NOT give good coalescing. Parallelism (number of active threads) and coalescing (whether those threads access contiguous memory) are completely separate concerns and must both be optimised. 256-thread parallelism with 32 separate cache-line fetches per warp is vastly slower than 256-thread parallelism with 1 coalesced fetch per warp.
- B) Incorrect — the hardware maximum is 1024 threads per block, so 256 is well within limits. Claiming 128 as the maximum is incorrect for any modern NVIDIA GPU.
- C) Correct — with block shape (256, 1), `threadIdx.x` runs 0–255 and `threadIdx.y = 0` always. `cuda.grid(2)` assigns `row = blockIdx.x * 256 + threadIdx.x` and `col = blockIdx.y * 1 + 0`. Warp threads 0–31 differ in `row` but share `col`. For a 1000-column array, `x[row, col]` has stride 1000 elements = 4000 bytes between consecutive warp threads → 32 fully separate cache-line fetches → ~32× slower than coalesced (1, 256) access.
- D) Incorrect — 256 is a multiple of 32 (256 = 8 × 32), so warp alignment is perfectly fine and all warps are fully populated. The real problem is entirely the strided memory access pattern caused by varying `row` indices within a warp, which has nothing to do with warp-size multiples.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets patterns from all three exams not in Set 1: nsys profiling, transfer counting, syncthreads, warp divergence, scheduling, GPU vs CPU timing, 3D coalescing, BSUB GPU flags.

---

## Q14 — Reading nsys Profiler Output

> **Week reference:** Week 9

**Mental Model:** Sum HtoD + DtoH + kernel time for total GPU-related time. Divide each by the total. HtoD usually dominates for large inputs; DtoH is often negligible (output is small).

An nsys profile produces:

| Report | Component | Total Time (ns) |
|--------|-----------|-----------------|
| gpumemtimesum | HtoD | 48,000,000 |
| gpumemtimesum | DtoH | 500,000 |
| gpukernsum | compute_kernel | 24,000,000 |

Which component dominates and approximately what fraction does it represent?

- A) The kernel — ~33%
- B) HtoD transfers — ~66%
- C) DtoH transfers — ~0.7%
- D) HtoD transfers — ~33%

**Answer: B**

- A) Incorrect — kernel = 24 ms out of (48 + 0.5 + 24) = 72.5 ms total → 24/72.5 ≈ 33%; significant but not dominant
- B) Correct — HtoD = 48 ms / 72.5 ms ≈ 66%; moving input data to the GPU is the bottleneck
- C) Incorrect — DtoH = 0.5 ms is indeed ~0.7% of total, making it negligible; it is the smallest, not the largest, component
- D) Incorrect — 33% is the kernel's fraction, not HtoD's

---

## Q15 — Numba Automatic Transfer Count (Naive)

> **Week reference:** Week 9

**Mental Model:** Numba auto-transfers EVERY NumPy argument both HtoD (before) and DtoH (after) for each kernel call. Device arrays (`cuda.to_device`, `cuda.device_array_like`) bypass automatic transfer entirely.

A kernel takes 2 NumPy arrays as arguments and is called 40 times in a loop. How many automatic HtoD and DtoH transfers does Numba perform in total?

- A) 40 HtoD + 40 DtoH (one set per call, not per array)
- B) 80 HtoD + 80 DtoH (2 arrays × 40 calls, both directions)
- C) 40 HtoD + 0 DtoH (only inputs are transferred)
- D) 2 HtoD + 2 DtoH (Numba caches arrays across calls)

**Answer: B**

- A) Incorrect — Numba transfers each array separately; 2 arrays × 40 calls = 80 in each direction
- B) Correct — 2 NumPy args × 40 calls = 80 HtoD before each call + 80 DtoH after each call = 160 total; Numba cannot know which arrays are pure inputs
- C) Incorrect — Numba transfers all NumPy arguments DtoH after every call regardless of whether they are inputs or outputs
- D) Incorrect — Numba re-transfers NumPy arrays fresh on every call; there is no cross-call caching

---

## Q16 — Optimised Transfer Count (Pre-allocated Device Arrays)

> **Week reference:** Week 9

**Mental Model:** `cuda.to_device(arr)` = 1 HtoD. `cuda.device_array_like(arr)` = 0 transfers (GPU-only allocation, no data). Passing device arrays to a kernel = 0 automatic transfers. `d_arr.copy_to_host()` = 1 DtoH. Pre-allocate once, reuse every call.

Same 40-call scenario: both inputs are pre-transferred once with `cuda.to_device()` before the loop; the output is a `cuda.device_array_like()` copied to host once after the loop. How many total transfers?

- A) 2 HtoD + 1 DtoH (3 total)
- B) 80 HtoD + 80 DtoH (same as naive)
- C) 40 HtoD + 1 DtoH (inputs re-transferred each call)
- D) 0 HtoD + 40 DtoH (device arrays need no HtoD)

**Answer: A**

- A) Correct — `cuda.to_device()` × 2 = 2 HtoD (once before loop); 40 kernel calls with device-array args = 0 automatic transfers; `copy_to_host()` × 1 = 1 DtoH (once after loop); total = 3 transfers
- B) Incorrect — passing device arrays to the kernel suppresses automatic transfer; Numba only auto-transfers NumPy arrays
- C) Incorrect — `cuda.to_device()` copies once to a persistent device array; subsequent calls passing that device array do not re-transfer
- D) Incorrect — the initial `cuda.to_device()` does require 2 HtoD; only the per-call overhead disappears

---

## Q17 — GPU vs CPU Total Time Calculation

> **Week reference:** Week 9

**Mental Model:** GPU wall-clock time = HtoD + kernel + DtoH. Compare this total to CPU time. The kernel alone being fast does not mean GPU wins if transfers are large.

A workload takes 12 s on CPU. GPU pipeline: HtoD = 2.0 s, kernel = 1.5 s, DtoH = 0.5 s. How much faster is the GPU?

- A) 8× faster — only the kernel counts (1.5 s vs 12 s)
- B) 3× faster — total GPU = 4.0 s vs CPU 12 s
- C) GPU is slower — total GPU = 14.0 s
- D) 6× faster — HtoD and DtoH cancel out

**Answer: B**

- A) Incorrect — HtoD and DtoH are real pipeline latencies; ignoring them gives an overly optimistic result that won't match wall-clock timing
- B) Correct — GPU total = 2.0 + 1.5 + 0.5 = 4.0 s; speedup = 12.0 / 4.0 = 3×
- C) Incorrect — 2.0 + 1.5 + 0.5 = 4.0 s, not 14.0 s
- D) Incorrect — HtoD and DtoH add to total time; they do not cancel

---

## Q18 — `cuda.syncthreads()` in Parallel Reduction

> **Week reference:** Week 9

**Mental Model:** In a block reduction each step writes partial sums, then reads neighbours' results. `syncthreads()` is a barrier: ALL threads in the block must arrive before any may proceed. Without it, a fast warp can start step N+1 before a slow warp finishes step N → race condition → wrong output.

In a CUDA block reduction, `cuda.syncthreads()` is called at the end of each reduction step. What happens if it is removed?

- A) Nothing — threads within a warp execute in lockstep so no synchronisation is needed
- B) The kernel runs faster because barrier overhead is eliminated
- C) Race condition: a warp may read another warp's stale partial result from the previous step, producing incorrect output
- D) The kernel will not compile without `syncthreads()`

**Answer: C**

- A) Incorrect — lockstep only applies within a single warp (32 threads); a block has many warps that can be at different steps without a barrier
- B) Incorrect — removing the barrier reduces overhead but corrupts results; correctness takes priority
- C) Correct — without `syncthreads()`, warp 0 can advance to step s=2 while warp 1 is still writing step s=1 partial sums → warp 0 reads stale data → wrong final result
- D) Incorrect — `syncthreads()` is a runtime call; the kernel compiles fine without it

---

## Q19 — Static vs Dynamic Scheduling for GPU Jobs

> **Week reference:** Week 11 / Week 13

**Mental Model:** Static scheduling assigns a fixed chunk upfront. If work per item has high variance, some GPUs finish early and sit idle. Dynamic scheduling assigns work on demand. For GPU kernels: high runtime variance → use dynamic to prevent idle GPUs.

Two CUDA kernels profiled on 100 inputs: `kernel_A` (mean=50 ms, std=45 ms) and `kernel_B` (mean=50 ms, std=0.2 ms). You have 4 GPUs. Which needs dynamic scheduling?

- A) `kernel_A` — high variance means some inputs take much longer; static leaves GPUs idle
- B) `kernel_B` — low variance allows precise runtime prediction; use dynamic for precision
- C) Both equally — variance doesn't affect GPU scheduling
- D) Neither — GPU kernels must always use static scheduling

**Answer: A**

- A) Correct — with std=45 ms, some inputs could take ~140 ms while others take ~5 ms; static assignment (each GPU gets 25 inputs) means one GPU may still be running after the other three finish; dynamic eliminates this imbalance
- B) Incorrect — `kernel_B`'s low variance makes static scheduling efficient: all GPUs finish at nearly the same time
- C) Incorrect — variance is exactly what determines the static vs dynamic choice
- D) Incorrect — job-array-based dynamic scheduling is standard practice for variable-runtime GPU workloads on LSF

---

## Q20 — GPU Amortisation Over Many Iterations

> **Week reference:** Week 9

**Mental Model:** Extract per-iteration GPU cost from two benchmarks: slope = (time2 − time1)/(n2 − n1). Fixed overhead = total − n × per_iteration. At large n, fixed overhead is negligible and the per-iteration comparison decides the winner.

CPU: 0.2 s/iteration. GPU benchmark: 5 iterations = 1.5 s total, 10 iterations = 2.0 s total. At 1 million iterations, which is faster?

- A) CPU — per-iteration GPU cost (0.3 s) is worse than CPU (0.2 s)
- B) GPU — per-iteration GPU cost is 0.1 s, half the CPU cost; overhead is amortised
- C) Tie — both scale linearly from the benchmark
- D) CPU — GPU fixed overhead of 1.0 s always adds cost regardless of iteration count

**Answer: B**

- A) Incorrect — per-iteration cost = (2.0 − 1.5)/(10 − 5) = 0.1 s, not 0.3 s; GPU is 2× faster per iteration
- B) Correct — GPU overhead = 1.5 − 5×0.1 = 1.0 s (fixed); at 1M iterations: GPU ≈ 0.1M + 1.0 ≈ 100,001 s vs CPU = 0.2M = 200,000 s; GPU wins by ~2×
- C) Incorrect — the GPU has a 1.0 s fixed offset; it does not scale from zero
- D) Incorrect — 1.0 s fixed overhead contributes ~0.000001 s/iteration at 1M iterations; it is effectively negligible

---

## Q21 — Warp Divergence with if/else

> **Week reference:** Week 9

**Mental Model:** Warp divergence = threads in the SAME warp take different branches. The SM executes both branches sequentially with masking. Throughput for those instructions is reduced by the number of distinct paths (2 paths → ~2× slower).

A kernel contains: `if cuda.threadIdx.x % 2 == 0: data[i] *= 2; else: data[i] += 1`. With a 32-thread warp, what is the performance impact?

- A) No impact — the condition is resolved at compile time
- B) ~2× slowdown for this branch — SM serialises both paths, halving warp throughput
- C) 32× slowdown — each thread executes both branches
- D) No impact — if/else is always free on CUDA

**Answer: B**

- A) Incorrect — `threadIdx.x % 2` is a per-thread runtime value; it cannot be resolved at compile time
- B) Correct — threads 0, 2, ..., 30 take the `*=2` path; threads 1, 3, ..., 31 take the `+=1` path; the SM executes one pass per path with the other 16 masked → effective throughput halved for these instructions
- C) Incorrect — each thread executes only its own branch; the SM runs at most one pass per distinct path (here 2), not 32
- D) Incorrect — intra-warp divergence is a well-known CUDA performance hazard; it always costs extra SM passes

---

## Q22 — 3D Array Coalescing: Which Block Shape?

> **Week reference:** Week 9

**Mental Model:** Row-major `vol[I, J, K]`: stride of i = J×K, stride of j = K, stride of k = 1. For coalesced access, adjacent warp threads must differ in k (the last, fastest axis). With `i, j, k = cuda.grid(3)`, set blockDim.x=1, blockDim.y=1, blockDim.z=32 so 32 threads differ in k.

A 3D kernel accesses `vol[i, j, k]` where vol has shape `(D, H, W)` in row-major order, `i, j, k = cuda.grid(3)`. Which block shape gives the best coalescing?

- A) `(32, 1, 1)` — 32 threads in i
- B) `(1, 32, 1)` — 32 threads in j
- C) `(1, 1, 32)` — 32 threads in k
- D) `(8, 4, 1)` — spread across i and j

**Answer: C**

- A) Incorrect — 32 threads varying in i → stride = H×W elements between adjacent warp threads → worst coalescing
- B) Incorrect — 32 threads varying in j → stride = W → strided access, not coalesced
- C) Correct — blockDim=(1,1,32): 32 threads differ in k (z-dim, last axis, stride=1) → consecutive addresses → fully coalesced
- D) Incorrect — i and j are both slow axes; spreading across them produces strided accesses in two dimensions

---

## Q23 — GPU Queue and Resource Flags (BSUB)

> **Week reference:** Week 1

**Mental Model:** Two changes always needed for GPU jobs: (1) switch queue to a GPU queue (`gpuv100`, `gpua100`), (2) add `-gpu "num=1:mode=exclusive_process"`. Without both, the job lands on a CPU-only node or fails to claim a GPU device.

A job script has `#BSUB -q hpc`. Which two changes are required to run it on a GPU?

- A) Add `#BSUB -n 1` and extend wall time `-W`
- B) Change `-q hpc` to `-q gpuv100` and add `#BSUB -gpu "num=1:mode=exclusive_process"`
- C) Add `#BSUB -R "rusage[gpu=1]"` and keep queue as `hpc`
- D) Change queue to `-q gpu` and add `#BSUB -R "span[hosts=1]"`

**Answer: B**

- A) Incorrect — changing core count or wall time does not request a GPU node; queue and GPU resource flag are what matter
- B) Correct — `gpuv100` (or `gpua100`, `hpcintgpu`) selects a GPU node; `-gpu "num=1:mode=exclusive_process"` allocates one GPU in exclusive mode so no other job shares it
- C) Incorrect — `rusage[gpu=1]` is not valid syntax; the correct flag is `-gpu "num=N:mode=..."` and the queue must also be a GPU queue
- D) Incorrect — `gpu` alone is not a valid DTU HPC queue name; `span[hosts=1]` is for multi-core CPU jobs to stay on one node, not for GPU allocation

---

## Q24 — `cuda.to_device` vs `cuda.device_array_like` Transfers

> **Week reference:** Week 9

**Mental Model:** `cuda.to_device(arr)` copies host → device (1 HtoD). `cuda.device_array_like(arr)` allocates device memory with matching shape/dtype but transfers NO data (0 transfers). Use `device_array_like` for output buffers written by the kernel.

How many HtoD transfers does this code produce?

```python
x = np.random.rand(1000)
d_x = cuda.to_device(x)
d_out = cuda.device_array_like(x)
my_kernel[bpg, tpb](d_x, d_out)
```

- A) 0 — both calls allocate without copying
- B) 1 — only `cuda.to_device(x)` transfers data; `device_array_like` just allocates
- C) 2 — one for `d_x` and one for `d_out`
- D) 3 — both arrays plus the kernel launch itself

**Answer: B**

- A) Incorrect — `cuda.to_device(x)` explicitly copies `x` from host to device; that is one HtoD
- B) Correct — `cuda.to_device(x)` = 1 HtoD; `cuda.device_array_like(x)` = 0 (allocates uninitialised GPU memory, no host data copied); kernel receives device pointers → no implicit transfer
- C) Incorrect — `cuda.device_array_like` allocates empty GPU memory; there is no host data to copy for it
- D) Incorrect — kernel launches do not inherently transfer data; only NumPy arguments to `@cuda.jit` kernels and explicit `to_device`/`copy_to_host` calls cause transfers

---

## Q25 — Transfer Count with Mixed NumPy and Device Arrays

> **Week reference:** Week 9

**Mental Model:** In a loop, count: explicit `cuda.to_device()` calls = HtoD per call; NumPy arguments passed to `@cuda.jit` = auto HtoD+DtoH each call; explicit `copy_to_host()` = DtoH per call. Device arrays passed to kernels = 0 transfers.

```python
d_weights = cuda.to_device(weights)   # once before loop
for inp in inputs:                     # 10 NumPy arrays
    d_out = cuda.device_array_like(inp)
    kernel[bpg, tpb](inp, d_weights, d_out)
    results.append(d_out.copy_to_host())
```

How many total HtoD transfers occur?

- A) 10 HtoD (only `inp` per iteration, forget pre-load)
- B) 20 HtoD (`inp` + `d_weights` re-transferred each iteration)
- C) 11 HtoD (10 auto for `inp` + 1 explicit pre-load of `d_weights`)
- D) 30 HtoD (all three kernel arguments each iteration)

**Answer: C**

- A) Incorrect — forgets the initial `cuda.to_device(weights)` = 1 HtoD before the loop
- B) Incorrect — `d_weights` is already a device array; passing it to the kernel does not trigger a new HtoD; only NumPy arguments auto-transfer
- C) Correct — `cuda.to_device(weights)` = 1 HtoD (before loop); each iteration: `inp` (NumPy) → 1 auto HtoD; `d_weights`, `d_out` (device arrays) → 0; total HtoD = 1 + 10 = 11. DtoH = 10 (one `copy_to_host()` per iteration).
- D) Incorrect — `d_weights` and `d_out` are device arrays; they generate 0 automatic HtoD when passed to a kernel

---
