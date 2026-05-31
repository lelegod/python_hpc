# GPU / CUDA Kernels — MCQ Practice

> Topics: Thread blocks, warp coalescing, grid dimensions, memory access patterns, CPU vs CUDA layout rules.
> Exam frequency: **Every exam** — 4+ questions per exam.

---

## Q1 — Best Block Shape for Coalescing (Row-Major)

> **Week reference:** Week 9

**Mental Model:** A warp reads 32 consecutive elements in one transaction only if those elements are at consecutive memory addresses. With `cuda.grid(2)`, adjacent threads differ by 1 in `col` (the last index) — so you want ALL warp threads varying in `col`, not `row`.

A CUDA kernel uses `row, col = cuda.grid(2)` to index a row-major 2D array `A[row, col]`. Adjacent threads within a warp differ by 1 in `col`. Which block shape gives the **best** memory coalescing?

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

**Mental Model:** Coalescing is about whether a warp's 32 threads map to 32 consecutive addresses. In `cuda.grid(2)`, `col` is the fast-varying index — identical to the last array axis — so maximise the number of threads varying in `col` within a warp.

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
