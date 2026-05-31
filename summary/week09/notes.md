# Week 9 — Numba & GPU Computing (CUDA)

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md) · [CUDA Grid Visual](cuda_grid_visual.md)

## Contents

- [Overview](#overview)
- [Theory & Concepts](#theory-concepts)
  - [Why GPU?](#why-gpu)
  - [GPU Architecture](#gpu-architecture)
  - [Numba](#numba)
  - [CUDA Programming Model](#cuda-programming-model)
- [Mathematical Content](#mathematical-content)
- [Key Code Examples](#key-code-examples)
  - [Numba CPU JIT](#numba-cpu-jit)
  - [Minimal CUDA Kernel (1D)](#minimal-cuda-kernel-1d)
  - [CUDA Kernel (2D grid for matrix operations)](#cuda-kernel-2d-grid-for-matrix-operations)
  - [CUDA Vector Addition with Timing](#cuda-vector-addition-with-timing)
  - [Host-Device Memory Transfer Pattern](#host-device-memory-transfer-pattern)
  - [CUDA Matrix Multiplication Kernel](#cuda-matrix-multiplication-kernel)
- [Numba CPU JIT](#numba-cpu-jit)
- [Exercise Highlights](#exercise-highlights)
  - [Exercise 1: CPU Matrix Multiplication](#exercise-1-cpu-matrix-multiplication)
  - [Exercise 2: CUDA Vector Addition](#exercise-2-cuda-vector-addition)
  - [Exercise 3: CUDA Matrix Multiplication](#exercise-3-cuda-matrix-multiplication)
- [Key Takeaways](#key-takeaways)

---

## Overview

Week 9 introduces GPU computing using NVIDIA CUDA, accessed from Python via Numba's JIT compiler. The lecture covers why GPUs are useful, how they differ from CPUs architecturally, how to write GPU kernels using `@cuda.jit`, the thread/block/grid programming model, and the critical cost of memory transfers between CPU (host) and GPU (device). Exercises benchmark JIT-compiled CPU matrix multiplication for cache efficiency, then build and profile CUDA kernels for vector addition and matrix multiplication, isolating compute time from transfer overhead.

---

## Theory & Concepts

### Why GPU?

GPUs deliver far more raw floating-point throughput and memory bandwidth than CPUs, at the cost of architectural flexibility:

| Property | CPU (Intel Xeon Gold, 12 cores) | GPU (5120 CUDA cores) |
|---|---|---|
| Peak throughput | 1.7 TFLOPs | 14.0 TFLOPs |
| Memory bandwidth | 128 GB/s | 900 GB/s |
| Core character | Few, fast, flexible, independent | Many, slow, simple, interdependent |
| Design philosophy | Minimise latency | Maximise throughput |

GPUs are best for massively data-parallel workloads where the same operation is applied to many independent elements simultaneously. GPUs originated for graphics (per-pixel operations) but are now used as General Purpose GPUs (GPGPU) for scientific computing, ML, simulation, etc.

The CPU/GPU analogy from the lecture: a CPU is like a sports car (low latency, fast single task), a GPU is like a bus (high throughput, many passengers per trip).

---

### GPU Architecture

A GPU is a **co-processor**: it works independently and simultaneously with the CPU, but has its own separate memory. Data must be explicitly transferred between host (CPU) memory and device (GPU) memory over the PCI bus. The CPU always takes the controller role — it initiates transfers and schedules kernel launches.

**Memory hierarchy inside the GPU:**
- Global DRAM: large but slow (~100 ns R/W latency)
- L2 cache: shared across all Streaming Multiprocessors
- L1 cache / shared memory: per-SM, fast, shared by all threads in a block

**Streaming Multiprocessors (SMs):**
- The GPU contains many SMs. Each SM roughly equates to a multicore CPU.
- Each SM contains many CUDA cores (streaming processors) and its own L1 cache.
- Thread blocks are scheduled onto SMs. All threads in a block share the same L1 cache, enabling efficient intra-block data sharing.

**Warp execution (SIMT — Single Instruction, Multiple Threads):**
- Threads within a block are grouped into **warps** of 32 threads.
- All threads in a warp execute the same instruction simultaneously.
- Divergent branches (different threads taking different paths) cause serialisation and hurt performance.

**Latency vs throughput:**
- CPU R/W: 90 ns, multiply: 2 ns. For 100-element operation: serial total ~18,200 ns.
- GPU R/W: 100 ns, multiply: 40 ns per single thread — slower individually. But with 100 threads in parallel: total ~240 ns (~76x faster throughput).

---

### Numba

Numba is an open-source **JIT (Just In Time) compiler** that translates a subset of Python and NumPy code into fast native machine code at runtime.

**Normal compiled languages (C++):** Write → Compile → Run (compilation happens before runtime).

**JIT:** Write → Run → Compile to machine code on first call, cache result for subsequent calls.

Key properties:
- First call is slow (compilation overhead is roughly constant, independent of problem size). Always warm up before timing.
- Subsequent calls run at near-native speed.
- `@jit(nopython=True)` / `@njit`: CPU JIT compilation, no Python objects allowed (fastest mode).
- `@cuda.jit`: GPU kernel compilation.
- `@cuda.jit(device=True)`: helper function callable only from within a GPU kernel (not a kernel itself).

**CPU benchmark (dot product, 100,000-element vectors):**
- Pure Python loop: 21.5 ms
- `@jit(nopython=True)` first call: 57.9 ms (includes compilation)
- `@jit(nopython=True)` subsequent calls: 0.1 ms (~200x speedup over Python)
- `np.dot`: 0.06 ms (NumPy still wins for this simple case via optimised BLAS)

**CPU benchmark (dot product, 10,000,000-element vectors):**
- Pure Python: 2000 ms
- JIT first call: 59.6 ms (compilation constant regardless of size)
- JIT subsequent: 16.8 ms
- `np.dot`: 6.3 ms

JIT is most beneficial when NumPy cannot vectorise the operation (e.g., complex loops with data dependencies), or when writing custom GPU kernels.

**JIT ecosystem (GPU-focused):** PyTorch, JAX, TensorFlow, OpenAI Triton, Taichi Lang — all use JIT compilation for GPU code under the hood. Numba exposes raw CUDA kernels directly.

---

### CUDA Programming Model

A CUDA **kernel** is a function:
- Defined on the **host** (CPU), decorated with `@cuda.jit`
- Launched by the **host**, executes on the **device** (GPU)
- Has **no return value** — results are written to an output argument
- Should **not** call other kernels
- Can call `@cuda.jit(device=True)` helper functions

**Thread hierarchy:**

```
Grid
  └── Blocks (thread blocks)
        └── Threads
              └── (Warps of 32 threads — hardware grouping)
```

- The programmer specifies: **threads per block** (tpb) and **blocks per grid** (bpg).
- Maximum threads per block: **1024**.
- Typical tpb values: 128, 256, 512 (must be multiples of warp size 32 for efficiency).
- Thread blocks map to Streaming Multiprocessors; threads map to CUDA cores.

**Thread index access inside a kernel:**

| Variable | Meaning |
|---|---|
| `cuda.threadIdx.x` | Index of this thread within its block |
| `cuda.blockIdx.x` | Index of this block within the grid |
| `cuda.blockDim.x` | Number of threads per block |
| `cuda.gridDim.x` | Number of blocks in the grid |
| `cuda.grid(1)` | Shorthand: global 1D thread index |
| `cuda.grid(2)` | Shorthand: global 2D thread indices (i, j) |

Mental model: **one thread per output element**. The grid must cover all output elements.

**Grid size calculation** (always round up to cover all elements):
```python
def get_bpg(n, tpb):
    return (n + (tpb - 1)) // tpb
```
Because the grid may have more threads than elements, **always bounds-check** inside the kernel.

**Memory transfers** between host and device are explicit and expensive (PCI bus):
- `cuda.to_device(arr)` — copies NumPy array from host to device
- `device_arr.copy_to_host()` — copies result from device to host
- `cuda.device_array(shape)` — allocates array on device without copying
- `cuda.pinned(arr)` / `cuda.pinned_array(shape)` — allocates pinned (page-locked) host memory, which enables faster DMA transfers to/from GPU

**cuda.synchronize()** must be called after kernel launch when timing, because kernel launches are asynchronous — the CPU continues while the GPU runs.

---

## Mathematical Content

**Grid dimension formula:**
```
blocks_per_grid = ceil(N / threads_per_block)
                = (N + threads_per_block - 1) // threads_per_block
```

**Matrix multiplication FLOPs:** N x N matrix multiply requires N^3 FLOPs.

**Memory transfer dominance (vector addition, N=1,000,000):**
- Baseline (host arrays, implicit transfers): 5.99 ms
- Pinned host memory (faster DMA): 3.51 ms
- Pre-transferred device arrays (no transfer in timing): 0.042 ms
- Conclusion: ~98.3% of total time was spent on memory transfer, only ~1.2% on actual computation.

**Memory transfer dominance (matrix multiply, 2048x2048):**
- Device memory (no transfer): 42.06 ms (2.4x faster than NumPy's 101.83 ms)
- Pinned memory: 50.61 ms
- Host memory (with transfers): 63 ms
- Memory transfer accounts for ~17% of runtime (compared to ~98% for simple vector addition — more work per thread means better compute-to-transfer ratio).

**Thread block shape and cache efficiency (matrix multiply):**
- 16x16 blocks (256 threads): 42.06 ms (baseline)
- 256x1 blocks: 435.59 ms (slow — reads along column of A, many cache misses)
- 1x256 blocks: 29.23 ms (fast — reads along row of B, cache efficient)
- Moral: compute grid layout affects memory access patterns and must be chosen for cache efficiency.

---

## Key Code Examples

### Numba CPU JIT

```python
from numba import jit
import numpy as np

@jit(nopython=True)
def matmul_jit(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C

# Warm up (trigger compilation)
matmul_jit(A, B)
# Now time subsequent calls
```

Cache-efficient loop reorder (ikj instead of ijk): the innermost loop iterates over `k`, accessing `B[k,j]` — stepping down a column, which is not cache-friendly for row-major storage. Reordering to `i,k,j` makes the innermost loop access `B` along a row, giving ~6x speedup for 200x200 matrices.

### Minimal CUDA Kernel (1D)

```python
from numba import cuda
import numpy as np

@cuda.jit
def double_kernel(x, y):
    i = cuda.grid(1)          # global thread index
    if i < len(x):            # bounds check required
        y[i] = 2 * x[i]

# Helper device function (called from kernel, not from host)
@cuda.jit(device=True)
def mul(a, b):
    return a * b

# Launch
tpb = 512
bpg = (len(x) + tpb - 1) // tpb
double_kernel[bpg, tpb](x, y)
```

### CUDA Kernel (2D grid for matrix operations)

```python
from numba import cuda

@cuda.jit
def double_kernel_2d(x, y):
    i, j = cuda.grid(2)
    if i < x.shape[0] and j < x.shape[1]:
        y[i, j] = 2 * x[i, j]

tpb = (16, 16)
bpg = (x.shape[0] // tpb[0], x.shape[1] // tpb[1])
double_kernel_2d[bpg, tpb](x, y)
```

### CUDA Vector Addition with Timing

```python
from numba import cuda
import numpy as np
from time import perf_counter as time

@cuda.jit
def add_kernel(x, y, out):
    i = cuda.grid(1)
    if i < x.shape[0]:
        out[i] = x[i] + y[i]

n = 1_000_000
x = np.random.rand(n).astype(np.float32)
y = np.random.rand(n).astype(np.float32)
out = np.empty_like(x)

tpb = 256
bpg = (n + tpb - 1) // tpb

# Warm up
add_kernel[bpg, tpb](x, y, out)

# Time with host arrays (includes implicit transfers)
rep = 200
t = time()
for _ in range(rep):
    add_kernel[bpg, tpb](x, y, out)
cuda.synchronize()
print((time() - t) / rep * 1000, 'ms')
```

### Host-Device Memory Transfer Pattern

```python
from numba import cuda

# Host → Device
d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.device_array(len(x))

# Run kernel (no transfer overhead in timing)
add_kernel[bpg, tpb](d_x, d_y, d_out)
cuda.synchronize()

# Device → Host (only when needed)
result = d_out.copy_to_host()

# Pinned memory (faster DMA)
with cuda.pinned(x, y, out):
    add_kernel[bpg, tpb](x, y, out)
    cuda.synchronize()
```

### CUDA Matrix Multiplication Kernel

```python
from numba import cuda, float32

@cuda.jit
def matmul_kernel(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = float32(0.)          # local accumulator avoids repeated global mem access
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

# 2048x2048 matrices with 16x16 thread blocks
tpb = (16, 16)
bpg = ((A.shape[0] + tpb[0] - 1) // tpb[0],
       (A.shape[1] + tpb[1] - 1) // tpb[1])

d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.device_array((A.shape[0], B.shape[1]))

matmul_kernel[bpg, tpb](d_A, d_B, d_C)
cuda.synchronize()
```

---

## Numba CPU JIT

Summary of CPU JIT usage:

- `@jit(nopython=True)` or `@njit`: compiles function to native machine code. No Python objects inside the function. Supports NumPy arrays, scalars, loops, conditionals.
- `parallel=True` + `prange`: enables automatic parallelisation over CPU cores (from previous weeks).
- Compilation is triggered on **first call** with a specific argument type signature. Subsequent calls with the same types reuse the compiled version.
- The compilation overhead is roughly constant (not proportional to array size), so it only matters for small/fast functions.
- JIT shines for custom loops that NumPy cannot vectorise natively. For standard linear algebra, NumPy's BLAS calls are typically faster.

**Cache efficiency with JIT (matmul loop reordering):**

Original `ijk` order: the `k` loop accesses `B[k,j]` — column-wise in row-major storage, causing cache misses. Reordering to `ikj` makes the innermost loop access `C[i,j] += A[i,k] * B[k,j]` where `B` is accessed row-wise. Speedup: ~6x for 200x200 matrices.

---

## Exercise Highlights

### Exercise 1: CPU Matrix Multiplication

1. Add `@jit(nopython=True)` decorator to a triple-loop matmul.
2. Benchmark original vs JIT for 100x100 matrices. Expected speedup: ~470x (468 ms Python vs 0.99 ms JIT on Xeon Gold 6142).
3. Reorder loops from `ijk` to `ikj` for cache efficiency (Autolab submission).
4. Benchmark cache-optimised version: ~6x additional speedup for 200x200 (8.46 ms original loop order vs 1.39 ms optimised).
5. (Optional) Plot MFLOP/s vs matrix size in KB, mark cache level boundaries on the plot. Performance drops are visible at L1/L2/L3 capacity limits.

### Exercise 2: CUDA Vector Addition

1. Write `add_kernel` that computes `out[i] = x[i] + y[i]` element-wise (Autolab).
2. Benchmark with N=1,000,000 float32 arrays, 256 threads/block. Measured: ~5.99 ms with host arrays.
3. Run as batch job using queue `c02613` (Autolab).
4. Use pinned memory (`cuda.pinned`): run time drops to 3.51 ms (faster DMA).
5. Pre-transfer arrays to device (`cuda.to_device`): run time drops to 0.042 ms. Conclusion: 98.3% of time was memory transfer, only 1.2% was computation.

### Exercise 3: CUDA Matrix Multiplication

1. Write `matmul_kernel(A, B, C)` using `cuda.grid(2)` for 2D thread indexing, thread `(i,j)` computes `C[i,j]`.
2. Benchmark 2048x2048 matrices with 16x16 thread blocks: 63 ms GPU vs 101.83 ms NumPy (1.6x faster).
3. With pinned memory: 50.61 ms (2x faster than NumPy). With device memory: 42.06 ms (2.4x faster).
4. Compare 16x16, 256x1, and 1x256 thread blocks (all 256 threads/block):
   - 16x16: 42.06 ms
   - 256x1: 435.59 ms (column-wise A access, cache unfriendly)
   - 1x256: 29.23 ms (row-wise B access, cache friendly)

---

## Key Takeaways

- GPUs have thousands of simple cores optimised for throughput; CPUs have few fast cores optimised for latency. Neither is universally better — workload character matters.
- Numba's `@cuda.jit` lets you write CUDA kernels directly in Python. The kernel specifies what each thread does; the grid/block structure specifies how many threads run and how they're organised.
- The standard pattern: one thread per output element, grid size = ceil(N / threads_per_block).
- Max threads per block is 1024. Always use multiples of 32 (warp size). Typical choice: 128, 256, or 512.
- Always bounds-check inside kernels when grid size may exceed array size.
- GPU kernels have no return value — write outputs to a pre-allocated output argument.
- First call to any JIT-compiled function (CPU or GPU) includes compilation overhead. Always warm up before timing.
- Memory transfer between CPU and GPU is expensive and often dominates total runtime for simple kernels. Design programs to minimise transfers: transfer once, compute many times.
- Pinned (page-locked) host memory speeds up DMA transfers between CPU and GPU.
- Thread block shape affects cache access patterns on the GPU — must consider grid layout alongside code logic.
- The CPU (host) controls the GPU (device): it initiates memory transfers, launches kernels, and synchronises (`cuda.synchronize()`) before reading results.
- For the simple vector addition kernel, ~98% of wall time is memory transfer. For matrix multiply (more compute per element), only ~17% is transfer — better arithmetic intensity makes GPU use more worthwhile.
