# Week 10 — CuPy and GPU Profiling

## Overview

Week 10 covers three interconnected topics: GPU parallel reductions, GPU profiling with NVIDIA Nsight Systems (nsys), and CuPy (CUDA NumPy). The week revisits the reduction pattern from earlier weeks but now implements it directly in CUDA via Numba, progressively optimising the kernel using shared memory and strided indexing. GPU profiling with nsys is introduced as the tool to measure where time is actually spent on the GPU — exposing memory transfer costs, kernel launch overhead, and kernel execution time. CuPy is presented as a drop-in NumPy replacement that runs on the GPU with almost no code changes, making it the fastest path to GPU acceleration for array-heavy code.

---

## Theory & Concepts

### 1. GPU Reductions

A **reduction** collapses an array to a single value (e.g. sum). The parallel strategy is a tree-structured doubling:

- Step `s = 1`: pairs of neighbours are added, result stored in-place.
- Step `s = 2`: pairs of pairs are added, etc.
- After `log2(N)` steps, the final result is at index 0.

**CPU approach (from week 5):** each level calls `apply_async` on a process pool, then waits (synchronises). The synchronisation is explicit between levels.

**GPU approach:** each level is one kernel launch. CUDA kernels launched from the same stream execute sequentially by default, so inter-level synchronisation is automatic.

**Problem 1 — Kernel launch overhead:** a naive approach launches O(log N) kernels, one per level. Each launch has a fixed overhead (~microseconds). For large arrays this is acceptable; for very small arrays it dominates.

**Solution:** Do the full tree reduction *inside a single thread block* using `cuda.syncthreads()` for intra-block synchronisation. One kernel call reduces one block completely, outputting one partial sum per block. The outer Python loop then only iterates over the number of *blocks*, not the number of levels — O(N/TPB) iterations instead of O(log N).

**Block reduce kernel design:**
- Each kernel handles one thread block.
- The tree reduction is done inside the block using `cuda.syncthreads()`.
- Thread 0 writes the block's result to the output array.
- The outer Python `while bpg > 1` loop calls the kernel again on the partial sums until one block remains.

**Warning:** all threads in a block must reach `cuda.syncthreads()`, otherwise the kernel may hang on older GPUs.

### 2. Shared Memory

Shared memory is a manually-managed fast memory local to each thread block. Key properties:

- Speed comparable to L1 cache — much faster than global GPU memory.
- Manually allocated and controlled by the programmer.
- Shared among all threads in the same block; not visible across blocks.
- Very limited: < 100 KB per SM (similar to L1 cache).
- Best used when threads access the same data **repeatedly** and/or in **irregular (strided) patterns** that prevent the L1 cache from helping.

**When to use shared memory:** when threads in a block repeatedly read the same region of global memory, load it once into shared memory and operate there.

**Shared memory for reductions:**
1. Each thread loads one element from global memory into `sdata[tid]` (pad with 0.0 if out of bounds).
2. Synchronise: `cuda.syncthreads()`.
3. Do the tree reduction entirely within `sdata`.
4. Thread 0 writes `sdata[0]` to the output array.

This removes repeated global memory accesses during the reduction loop, which previously re-read `data[i]` and `data[i+s]` on every iteration.

**Motivational example — matrix transpose:** when computing `y[i,j] = x[j,i]`, either reading x or writing y will be non-coalesced (strided) regardless of thread layout. Solution: load a tile of x into shared memory row-wise (coalesced read), synchronise, then write columns of shared memory to y row-wise (coalesced write). Shared memory absorbs the transpose with no cache penalty.

### 3. Warp Divergence

A **warp** is a group of 32 threads that execute the same instruction simultaneously (SIMD). If threads in a warp take different branches, the warp must serialise — this is **warp divergence**.

The condition `if tid % (2 * s) == 0` causes divergence: as `s` grows, fewer and fewer threads satisfy the condition, but all threads in the warp still pay the cost of the idle path.

**Fix — strided index:** replace the masking condition with a computed strided index:
```python
idx = 2 * s * tid
if idx < cuda.blockDim.x:
    sdata[idx] += sdata[idx + s]
```
This assigns active threads to the lower portion of the warp, keeping more warps fully active and reducing divergence. Result: ~2.3x speedup on the kernel time.

### 4. GPU Memory Hierarchy (recap)

```
Global GPU memory  (large, slow, visible to all blocks)
    L2 cache       (shared across SMs)
        L1 / Shared memory  (per SM, very fast, manually managed for shared mem)
            Registers        (per thread, fastest)
```

### 5. GPU Profiling with NVIDIA nsys

`nsys` (NVIDIA Nsight Systems) is a command-line GPU profiler that records a trace of all CUDA API calls, kernel executions, and memory transfers.

**Workflow:**
```bash
# Record profile
nsys profile -o <output_name> python script.py [args]

# View statistics
nsys stats <output_name>.nsys-rep
```

**Key sections in `nsys stats` output:**

| Section | What it shows |
|---|---|
| OS Runtime Summary (`osrtsum`) | System calls — mostly ignorable |
| CUDA API Summary (`cudaapisum`) | Time spent in CUDA API: memcpy, malloc, kernel launch calls from the CPU side |
| CUDA GPU Kernel Summary (`gpukernsum`) | Actual GPU kernel execution times, grid/block dimensions, kernel names |
| GPU MemOps Summary by Time (`gpumemtimesum`) | Time spent on HtoD and DtoH memory copies |
| GPU MemOps Summary by Size (`gpumemsizesum`) | Volume of data transferred |

**Reading the output:**
- `Time (%)`: fraction of that section's total time.
- `Total Time (ns)`: absolute time.
- `Instances` / `Num Calls`: how many times the operation was called.
- `GridXYZ` / `BlockXYZ`: kernel launch dimensions (blocks per grid, threads per block).
- D = Device (GPU), H = Host (CPU).

**Key insight from profiling reductions:** without manual data transfer, each kernel call triggers implicit HtoD and DtoH copies. The `x[:n] = out[:n]` line (copying partial sums back) generates extra DtoH transfers. Manually transferring data to the GPU once at the start and keeping it there eliminates wasted transfers.

### 6. CuPy

CuPy is a NumPy-compatible array library that runs on the GPU. The API mirrors NumPy almost exactly — replace `import numpy as np` with `import cupy as cp` and most array operations run on the GPU automatically.

**Why CuPy is fast:** each CuPy operation dispatches a pre-compiled CUDA kernel. Operations on large arrays saturate GPU bandwidth with no kernel-writing required.

**Key difference from NumPy loops:** in NumPy, a Python-level loop over rows (single loop version) does not cause major overhead because NumPy operations are themselves batched. In CuPy, a Python-level loop issues many small kernel launches — the `cuLaunchKernel` overhead can exceed the actual compute time. The no-loop (fully vectorised) version avoids this and is dramatically faster.

**Performance observed in solutions:**
- NumPy single-loop haversine: ~10.42 s
- CuPy single-loop: ~3.29 s (3x speedup with almost no code change)
- CuPy no-loop: ~1.00 s (10x speedup over NumPy)

**Profile insight for CuPy:** the single-loop CuPy version shows that `cuLaunchKernel` calls consume more time than all kernel executions combined. This confirms that for CuPy, fewer, larger operations are always better.

---

## Mathematical Formulas

**Parallel reduction step count:**

Number of reduction levels = ceil(log2(N))

**Grid calculation (ceiling division):**
```
blocks_per_grid = (n + tpb - 1) // tpb
```

**Strided index for warp-divergence-free reduction:**
```
idx = 2 * s * tid
```
Active threads: tid = 0, 1, ..., (blockDim.x / (2*s)) - 1 — all in the lower portion of the warp.

**Kernel launch overhead dominance condition:**

For CuPy loops: if `cuLaunchKernel` total time > sum of kernel execution times, the loop is the bottleneck, not compute.

**Speedup achieved through optimisations (solutions, n = 400M):**

| Version | Kernel time |
|---|---|
| Baseline (global mem, masking) | ~204 ms |
| + Manual data transfer | similar kernel time, less transfer |
| + Shared memory | ~189 ms |
| + Strided index (no warp divergence) | ~82 ms (~2.3x vs shared mem) |
| + All optimisations (loop unrolling, warp reduce) | ~1.85 ms (~4.1x vs strided) |

Fully optimised kernel is roughly **9.4x faster** than the baseline.

---

## Key Code Examples

### reduce.py — Block Reduce Kernel (from file)

```python
from numba import cuda
import numpy as np
import sys

TPB = 64  # Threads per block

@cuda.jit
def reduce_kernel(data, out, n):
    # Get the 1D grid and block indices
    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    # Do reduction for threadblock
    s = 1
    while s < cuda.blockDim.x:
        if tid % (2 * s) == 0 and i + s < n:
            data[i] += data[i + s]
        s *= 2
        cuda.syncthreads()  # Ensure block is synchronized

    # Write result for this block to global memory
    if tid == 0:
        out[cuda.blockIdx.x] = data[i]

def get_grid(n, tpb):
    return (n + (tpb - 1)) // tpb  # Blocks per grid

def reduce(x):
    n = len(x)
    bpg = get_grid(n, TPB)
    out = cuda.device_array(bpg, dtype=x.dtype)
    while bpg > 1:
        reduce_kernel[bpg, TPB](x, out, n)
        n = bpg
        bpg = get_grid(n, TPB)
        x[:n] = out[:n]
    reduce_kernel[bpg, TPB](x, out, n)
    return out
```

### Shared Memory Kernel (from lecture slides, Exercise 4 solution)

```python
@cuda.jit
def reduce_kernel(data, out, n):
    # Declare shared memory — one slot per thread in the block
    sdata = cuda.shared.array(TPB, dtype=data.dtype)

    # Get the 1D grid and block indices
    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    # Each thread loads one element from global to shared memory
    sdata[tid] = data[i] if i < n else 0.0
    cuda.syncthreads()  # Ensure all threads have loaded data

    # Do reduction in shared memory
    s = 1
    while s < cuda.blockDim.x:
        if tid % (2 * s) == 0:
            sdata[tid] += sdata[tid + s]
        s *= 2
        cuda.syncthreads()  # Ensure block is synchronized

    # Write result for this block to global memory
    if tid == 0:
        out[cuda.blockIdx.x] = sdata[0]
```

### Strided Index Kernel — eliminates warp divergence (Exercise 5 solution)

```python
@cuda.jit
def reduce_kernel(data, out, n):
    sdata = cuda.shared.array(TPB, dtype=data.dtype)

    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    sdata[tid] = data[i] if i < n else 0.0
    cuda.syncthreads()

    s = 1
    while s < cuda.blockDim.x:
        idx = 2 * s * tid          # strided index — active threads are contiguous
        if idx < cuda.blockDim.x:
            sdata[idx] += sdata[idx + s]
        s *= 2
        cuda.syncthreads()

    if tid == 0:
        out[cuda.blockIdx.x] = sdata[0]
```

### Fully Optimised Kernel with Warp Unrolling (Exercise 7 solution)

```python
TPB = 126  # Threads per block

@cuda.jit(device=True)
def warp_reduce(sdata, tid):
    sdata[tid] += sdata[tid + 32]
    sdata[tid] += sdata[tid + 16]
    sdata[tid] += sdata[tid + 8]
    sdata[tid] += sdata[tid + 4]
    sdata[tid] += sdata[tid + 2]
    sdata[tid] += sdata[tid + 1]

@cuda.jit
def reduce_kernel(data, out, n):
    sdata = cuda.shared.array(shape=TPB, dtype=data.dtype)

    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * cuda.blockDim.x * 2 + cuda.threadIdx.x
    grid_size = cuda.blockDim.x * 2 * cuda.gridDim.x

    # Each thread loads multiple elements (grid-stride loop)
    sdata[tid] = 0.0
    while i < n:
        sdata[tid] += data[i]
        sdata[tid] += data[i + cuda.blockDim.x] \
                      if i + cuda.blockDim.x < n else 0.0
        i += grid_size
    cuda.syncthreads()

    # Unrolled tree reduction
    if TPB >= 512:
        if tid < 256: sdata[tid] += sdata[tid + 256]
        cuda.syncthreads()
    if TPB >= 256:
        if tid < 128: sdata[tid] += sdata[tid + 128]
        cuda.syncthreads()
    if TPB >= 128:
        if tid < 64: sdata[tid] += sdata[tid + 64]
        cuda.syncthreads()

    if tid < 32:
        warp_reduce(sdata, tid)

    if tid == 0:
        out[cuda.blockIdx.x] = sdata[0]
```

### nsys Profiling Commands

```bash
# Profile a script
nsys profile -o my_profile python script.py 4000000

# View all statistics
nsys stats my_profile.nsys-rep
```

### CuPy Drop-in Replacement

```python
import cupy as cp

# Load data directly onto GPU
data = cp.loadtxt(fname, delimiter=',', skiprows=1, usecols=(1, 2))

# Use exactly like NumPy — runs on GPU
def distance_matrix_noloop(p1, p2):
    p1 = cp.radians(p1)
    p2 = cp.radians(p2)
    dsin2 = cp.sin(0.5 * (p1[:, None, :] - p2[None, :, :])) ** 2
    cosprod = cp.cos(p1[:, None, 0]) * cp.cos(p2[None, :, 0])
    D = 2 * cp.arcsin(cp.sqrt(dsin2[:, :, 0] + cosprod * dsin2[:, :, 1]))
    D *= 6371
    return D
```

---

## Exercise Highlights

### Exercise 1 — GPU Reductions (7 sub-tasks)

**1.1 (Autolab):** Write a Python program that accepts `n` as a command-line argument, generates `n` random float32 numbers, and uses the block-reduce kernel to compute their sum. Use `np.random.rand` and `.astype('float32')`.

**1.2 — Profile without manual transfer:** Run `nsys profile -o prof python reduce.py 4000000` and examine stats. Observation: most time is in memory transfers (HtoD + DtoH), not computation. Multiple implicit transfers occur because `x[:n] = out[:n]` triggers DtoH+HtoD on every outer loop iteration.

**1.3 — Manual data transfer:** Transfer data to GPU once using `cuda.to_device(x)` before the loop and retrieve at the end. Result: only 1 HtoD and 1 DtoH transfer. Memory transfer time drops dramatically; kernel time is unchanged.

**1.4 — Add shared memory:** Load each block's data into `sdata` before reduction. Bounds-pad with `0.0`. Do reduction in `sdata`, write `sdata[0]` to output. Slight speed improvement; main benefit is enabling the next optimisation.

**1.5 — Strided index:** Replace `if tid % (2*s) == 0` with `idx = 2*s*tid; if idx < blockDim`. This reduces warp divergence by keeping active threads in contiguous warp lanes. Kernel time drops ~2.3x.

**1.6 — Profile optimised kernel:** The primary kernel now runs ~2.3x faster. Memory transfer patterns are unchanged.

**1.7 (Optional) — Further optimisations:** Implement loop unrolling and warp-level reduction (no syncthreads needed within a warp). Combine with loading 2 elements per thread. Achieves ~4.1x further speedup over strided-index version.

### Exercise 2 — CuPy (5 sub-tasks)

**2.1 (Autolab):** Convert both `distance_matrix_oneloop` and `distance_matrix_noloop` from NumPy to CuPy. Replace `import numpy as np` with `import cupy as cp` and update the data loading to use `cp.loadtxt`.

**2.2 — Time single-loop CuPy:** Using `locations_5000.csv`. NumPy baseline: ~10.42 s. CuPy single-loop: ~3.29 s. Over 3x speedup with almost no code change.

**2.3 — Profile single-loop CuPy:** The CUDA API Summary shows `cuLaunchKernel` consuming more time than all actual kernel executions combined. This is the cost of 5000 Python-loop iterations each launching GPU kernels. The GPU MemOps section shows 13545 small DtoD copies — internal CuPy buffer management overhead.

**2.4 — Time no-loop CuPy:** ~1.00 s. Over 3x faster than single-loop CuPy, and ~10x faster than NumPy.

**2.5 — Profile no-loop CuPy:** Far fewer kernel launches. The dominant costs shift to the compute kernels themselves (sin, arcsin, multiply, etc.) and a single large memory allocation. The `cuLaunchKernel` overhead is negligible. This confirms the CuPy rule: fewer, larger operations always outperform many small operations.

---

## Key Takeaways

1. **GPU reductions require care:** unlike CPU reductions where each level synchronises via `pool.wait()`, GPU reductions use `cuda.syncthreads()` for intra-block sync; inter-block sync is handled by launching separate kernels.

2. **Shared memory is not always faster — but it enables further optimisations:** moving reduction from global memory to shared memory gives a modest speedup, but it removes the bounds check in the reduction loop (shared memory is always block-sized) and sets up warp-level unrolling.

3. **Warp divergence is a real bottleneck:** the `tid % (2*s) == 0` condition causes 50–75% of threads in active warps to be idle. Replacing it with a strided index gives ~2.3x kernel speedup.

4. **Memory transfer is often the bottleneck, not compute:** the nsys profiler consistently shows that HtoD and DtoH transfers dominate for small-to-medium workloads. Manually transferring data to the GPU once and keeping it there is the single most impactful optimisation for data-transfer-bound code.

5. **CuPy is the fastest path to GPU acceleration for array code:** replacing `numpy` with `cupy` requires almost no changes and gives 3–10x speedups on array-heavy workloads. It is the right tool when you do not need to write custom kernels.

6. **For CuPy, avoid Python loops:** each loop iteration launches separate GPU kernels. The `cuLaunchKernel` overhead can exceed all compute time combined. Always prefer vectorised (no-loop) operations over Python-level loops in CuPy code.

7. **nsys is the essential GPU profiling tool:** it reveals kernel times, memory transfer volumes/times, and launch counts simultaneously. Without it, GPU optimisation is guesswork.

8. **The full optimisation chain for reductions achieves ~9.4x speedup:** naive block-reduce → shared memory → strided index → loop unrolling + warp reduce. Each step targets a specific GPU bottleneck.
