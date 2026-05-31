# GPU / CUDA Kernels — MCQ Practice

> Topics: Thread blocks, warp coalescing, grid dimensions, memory access patterns, CPU vs CUDA layout rules.
> Exam frequency: **Every exam** — 4+ questions per exam.

---

## Q1 — Best Block Shape for Coalescing (Row-Major)

> **Week reference:** Week 9

A CUDA kernel uses `row, col = cuda.grid(2)` to index a row-major 2D array `A[row, col]`. Adjacent threads within a warp differ by 1 in `col`. Which block shape gives the **best** memory coalescing?

- A) (256, 1) — 256 rows, 1 column
- B) (16, 16) — 16 rows, 16 columns
- C) (1, 256) — 1 row, 256 columns
- D) (32, 32) — 32 rows, 32 columns

**Answer: C**

- A) Incorrect — 256 threads differ in row, same col → strided access across rows (worst coalescing)
- B) Incorrect — threads vary in both row and col → partial coalescing, medium performance
- C) Correct — all 256 threads share the same row, differ only in col → access consecutive memory addresses (fully coalesced)
- D) Incorrect — 32×32 = 1024 threads is valid and warp threads 0–31 (same threadIdx.y, consecutive threadIdx.x) do access sequential columns, so coalescing is actually good; (1,256) is still the idiomatic choice for purely column-parallel work and maximally explicit about access pattern.

---

## Q2 — Why 1×256 Is Best for Row-Major Arrays

> **Week reference:** Week 9

For a row-major 2D array accessed as `A[row, col]`, why does a block shape of `(1, 256)` give better memory coalescing than `(16, 16)`?

- A) Because 256 is a larger number of threads than 16×16=256, so there is more parallelism
- B) Because all 256 threads in a warp share the same row, so their col indices are consecutive → sequential memory addresses
- C) Because 1×256 uses fewer registers per thread, reducing register pressure
- D) Because row-major arrays store data column-by-column, so varying col accesses the same cache line

**Answer: B**

- A) Incorrect — 1×256 and 16×16 both have 256 threads total; parallelism is the same
- B) Correct — with block shape (1, 256), all threads in a warp have the same row and sequential col values, meaning they access A[row, col], A[row, col+1], ... which are contiguous in row-major layout
- C) Incorrect — register usage is determined by kernel logic, not block shape
- D) Incorrect — row-major arrays store data row-by-row, not column-by-column; varying col does access contiguous memory, but that is the reason (1, 256) is good, not a misstatement about layout

---

## Q3 — Worst Block Shape for Row-Major Coalescing

> **Week reference:** Week 9

A kernel accesses a row-major array `x[row, col]` where `row, col = cuda.grid(2)`. Which block configuration gives the **worst** memory access performance?

- A) (1, 256)
- B) (16, 16)
- C) (256, 1)
- D) (32, 8)

**Answer: C**

- A) Incorrect — (1, 256) gives the best coalescing: threads differ only in col → consecutive memory
- B) Incorrect — (16, 16) gives medium performance; 16 consecutive threads per row still have some locality
- C) Correct — all 256 threads have the same col value but differ in row → each thread accesses A[row, col], A[row+1, col], ... which are separated by the full row stride (worst strided access)
- D) Incorrect — (32, 8) is medium; 8 consecutive threads share a row for their 8 col values, giving moderate coalescing

---

## Q4 — Grid Size Calculation (1D)

> **Week reference:** Week 9

You have an array of size N=500 and want to process it with a CUDA kernel using 32 threads per block. How many blocks do you need?

- A) 15
- B) 15.625
- C) 16
- D) 32

**Answer: C**

- A) Incorrect — 15 blocks × 32 threads = 480 threads, which only covers elements 0–479; elements 480–499 would be missed
- B) Incorrect — fractional block counts are not possible; you must round up to a whole integer
- C) Correct — `ceil(500 / 32) = ceil(15.625) = 16`; the formula is `bpg = (N + tpb - 1) // tpb = (500 + 31) // 32 = 531 // 32 = 16`
- D) Incorrect — 32 blocks would be unnecessarily many (32 × 32 = 1024 threads for 500 elements)

---

## Q5 — 2D Grid Dimension Calculation

> **Week reference:** Week 9

You want to process a 200×200 output array with a 2D CUDA kernel using a block size of (16, 16). How many thread blocks are launched in total?

- A) 144 (12×12)
- B) 156 (12×13)
- C) 169 (13×13)
- D) 196 (14×14)

**Answer: C**

- A) Incorrect — `ceil(200/16) = ceil(12.5) = 13`, not 12; using 12 would leave the last 8 rows/cols unprocessed
- B) Incorrect — both dimensions require the same ceiling calculation: ceil(200/16) = 13 in each dimension
- C) Correct — `ceil(200/16) = 13` in both x and y directions → 13 × 13 = 169 total blocks
- D) Incorrect — 14 blocks per dimension would be overly conservative; 13 is sufficient

---

## Q6 — Maximum Threads Per Block and Warp Multiples

> **Week reference:** Week 9

Which of the following statements about CUDA thread block sizing is correct?

- A) The maximum threads per block is 256; always use exactly 256
- B) The maximum threads per block is 1024; choose a multiple of 32 for best warp efficiency
- C) The maximum threads per block is 512; warp size is 16
- D) Thread block size should always equal the warp size of 32

**Answer: B**

- A) Incorrect — the maximum is 1024, not 256; using only 256 is valid but not the maximum
- B) Correct — CUDA GPUs support up to 1024 threads per block, and the warp size is 32; choosing multiples of 32 (e.g., 128, 256, 512) avoids partially-filled warps and maximises occupancy
- C) Incorrect — maximum is 1024 and warp size is 32, not 512 and 16
- D) Incorrect — using only 32 threads per block severely limits parallelism; it is valid but very inefficient

---

## Q7 — Optimal Array Layout for CUDA Convolution (CHW)

> **Week reference:** Week 9

A CUDA convolution kernel has threads that vary in `col` (spatial width axis) within a warp. The input tensor has shape `(channels, height, width)` (CHW). Why is CHW preferred over HWC for this kernel?

- A) CHW uses less memory than HWC for the same data
- B) CHW places the spatial axes (height, width) as the last two dimensions, so threads varying in `col` access consecutive memory addresses
- C) CHW is preferred because the CPU also uses CHW, making data transfer simpler
- D) CHW has fewer cache conflicts because channels are stored in separate memory pages

**Answer: B**

- A) Incorrect — CHW and HWC store exactly the same amount of data; layout does not change memory size
- B) Correct — in CHW, data for a fixed (channel, row) is stored consecutively across the width axis; threads differing in col therefore access A[c, row, col], A[c, row, col+1], ... which are sequential in memory → coalesced access
- C) Incorrect — CPU typically prefers HWC (channels last), not CHW; they use OPPOSITE layouts
- D) Incorrect — cache conflicts depend on access patterns and stride, not on which dimension holds channels

---

## Q8 — CPU vs CUDA Layout Are Opposite

> **Week reference:** Week 9

When optimising memory layout for image processing, which statement correctly describes the difference between CPU and CUDA preferences?

- A) Both CPU and CUDA prefer HWC (height × width × channels) layout for best cache performance
- B) Both CPU and CUDA prefer CHW (channels × height × width) layout
- C) CPU prefers HWC (channels last) because its inner loop iterates over channels; CUDA prefers CHW (channels first) because threads vary in spatial position
- D) CPU prefers CHW; CUDA prefers HWC because GPU memory is column-major

**Answer: C**

- A) Incorrect — CPU and CUDA have opposite optimal layouts; they do not both prefer HWC
- B) Incorrect — CPU benefits from HWC (channels last), not CHW
- C) Correct — on the CPU, the inner loop often iterates over channels k for a fixed (i, j), so HWC (i, j, k) is cache-friendly; on a CUDA GPU, threads in a warp differ in the spatial column, so CHW (k, i, j) makes spatial axes innermost and is cache-friendly
- D) Incorrect — neither GPU memory nor the reasoning about column-major applies here; the real reason CUDA prefers CHW is warp-level coalescing along the spatial axis

---

## Q9 — Bounds Check Requirement

> **Week reference:** Week 9

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

- A) Incorrect — CUDA does NOT automatically skip out-of-bounds threads; without a guard, threads with i ≥ n would write to arbitrary memory, causing undefined behaviour
- B) Correct — with N=100 and tpb=32, we launch ceil(100/32)=4 blocks = 128 threads; threads 100–127 have i ≥ n and must be skipped via the bounds check
- C) Incorrect — cuda.grid(1) returns a non-negative integer equal to the global thread index
- D) Incorrect — the bounds check does not restrict execution to one warp; all threads evaluate the condition independently

---

## Q10 — `@cuda.jit(device=True)` Decorator

> **Week reference:** Week 9

What does the `@cuda.jit(device=True)` decorator do in Numba CUDA?

- A) It compiles the function to run on the CPU but allows it to be called from a CUDA kernel
- B) It marks the function as a device function: callable from GPU kernels but NOT from host (Python) code
- C) It enables the function to launch child kernels (dynamic parallelism)
- D) It makes the function execute on the device once per block instead of once per thread

**Answer: B**

- A) Incorrect — `device=True` means the function runs on the GPU device, not the CPU
- B) Correct — a device function is compiled for the GPU and can only be called from within a CUDA kernel (or another device function); calling it directly from Python host code raises an error
- C) Incorrect — dynamic parallelism (launching kernels from within kernels) is a different feature and is not what `device=True` means
- D) Incorrect — device functions are called per-thread just like regular kernel code; they have no special per-block execution semantics

---

## Q11 — `cuda.grid(1)` Formula

> **Week reference:** Week 9

Which expression correctly computes the global thread index `i` in a 1D CUDA kernel, equivalent to `cuda.grid(1)`?

- A) `cuda.blockIdx.x + cuda.blockDim.x + cuda.threadIdx.x`
- B) `cuda.threadIdx.x * cuda.blockDim.x + cuda.blockIdx.x`
- C) `cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x`
- D) `cuda.blockDim.x * cuda.gridDim.x + cuda.threadIdx.x`

**Answer: C**

- A) Incorrect — addition instead of multiplication gives incorrect indexing; block 2 and block 3 would give overlapping indices
- B) Incorrect — threadIdx and blockIdx are swapped; this formula does not produce unique global indices
- C) Correct — the global thread index is `blockIdx.x * blockDim.x + threadIdx.x`; block 0 contributes threads 0..blockDim-1, block 1 contributes blockDim..2*blockDim-1, etc.
- D) Incorrect — gridDim.x is the total number of blocks, not the block index; this formula does not produce per-thread indices

---

## Q12 — Warp Size and SIMT Execution

> **Week reference:** Week 9

What is the CUDA warp size, and what does it mean for kernel execution?

- A) Warp size = 64; threads 0–63 execute the same instruction simultaneously on an SM
- B) Warp size = 32; threads 0–31 execute the same instruction simultaneously (SIMT); divergent branches serialise
- C) Warp size = 32; each thread in a warp executes independently on separate cores with no synchronisation
- D) Warp size = 16; this is why block sizes should be multiples of 16

**Answer: B**

- A) Incorrect — the warp size on all NVIDIA GPUs (Kepler through Hopper) is 32, not 64
- B) Correct — a warp consists of 32 threads that execute the same instruction in lockstep (SIMT = Single Instruction, Multiple Threads); if threads diverge on a branch, both paths are serialised
- C) Incorrect — threads within a warp are NOT independent; they execute in lockstep and branch divergence causes serialisation
- D) Incorrect — warp size is 32, not 16; block sizes should therefore be multiples of 32

---

## Q13 — Worst Block Shape for Row-Major Read (Performance Trap)

> **Week reference:** Week 9

A kernel reads `x[row, col]` where `row, col = cuda.grid(2)` and `x` is stored in row-major (C) order. A colleague suggests using block shape `(256, 1)` because "256 threads per block maximises parallelism". What is wrong with this reasoning?

- A) Nothing — (256, 1) maximises parallelism AND gives the best memory coalescing for row-major arrays
- B) Block size 256 exceeds the maximum threads per block of 128
- C) With (256, 1) all threads in a warp have different row values and the same col value, so they access `x[0, col], x[1, col], ..., x[255, col]` — strided access across rows (stride = row_length), which is the worst possible pattern for coalescing
- D) (256, 1) is only bad because it does not use a multiple of the warp size 32

**Answer: C**

- A) Incorrect — (256, 1) does NOT give good coalescing; parallelism and coalescing are separate concerns and must both be optimised
- B) Incorrect — the maximum threads per block is 1024, so 256 is well within limits
- C) Correct — with block shape (256, 1), threads in a warp differ in the row dimension (stride = number of columns in the row) while sharing the same col; for a 1000-column array, consecutive threads access memory 1000 elements apart → severely non-coalesced (strided) access
- D) Incorrect — 256 is a multiple of 32 (256 = 8 × 32), so warp alignment is fine; the real problem is the strided memory access pattern caused by varying row indices

---
