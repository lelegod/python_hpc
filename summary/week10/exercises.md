# Week 10 Exercises — GPU Computing with CuPy + Parallel Reductions

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Section 1: GPU Reductions](#section-1-gpu-reductions)
- [Exercise 1.1 `[AUTOLAB]`](#exercise-11-autolab)
- [Exercise 1.2 `[PRACTICE]`](#exercise-12-practice)
- [Exercise 1.3 `[PRACTICE]`](#exercise-13-practice)
- [Exercise 1.4 `[PRACTICE]`](#exercise-14-practice)
- [Exercise 1.5 `[PRACTICE]`](#exercise-15-practice)
- [Exercise 1.6 `[PRACTICE]`](#exercise-16-practice)
- [Exercise 1.7 `[PRACTICE]` (Optional)](#exercise-17-practice-optional)
- [Section 2: CuPy](#section-2-cupy)
- [Exercise 2.1 `[AUTOLAB]`](#exercise-21-autolab)
- [Exercise 2.2 `[PRACTICE]`](#exercise-22-practice)
- [Exercise 2.3 `[PRACTICE]`](#exercise-23-practice)
- [Exercise 2.4 `[PRACTICE]`](#exercise-24-practice)
- [Exercise 2.5 `[PRACTICE]`](#exercise-25-practice)

---

---

## Section 1: GPU Reductions

This section is based on the [parallel reductions in CUDA presentation](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) by Mark Harris.

The following kernel implementation is used as the starting point throughout this section:

```python
from numba import cuda

TPB = 128  # Threads per block

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

---

## Exercise 1.1 `[AUTOLAB]`

Make a Python program that takes a number *n* as a command line argument. The program must then generate *n* random float32 numbers and use the above code to compute their sum. Use `np.random.rand` to generate the random numbers and convert them to float32 with `np.astype`. Finally, print the sum (and only the sum).

> **Solution:**
>
> The student file `reduce.py` implements this. It wires together `np.random.rand`, `.astype('float32')`, the `reduce` helper, and prints the single result value.
>
> ```python
> from numba import cuda
> import numpy as np
> import sys
>
> TPB = 64  # Threads per block
>
> @cuda.jit
> def reduce_kernel(data, out, n):
>     # Get the 1D grid and block indices
>     tid = cuda.threadIdx.x
>     i = cuda.grid(1)
>
>     # Do reduction for threadblock
>     s = 1
>     while s < cuda.blockDim.x:
>         if tid % (2 * s) == 0 and i + s < n:
>             data[i] += data[i + s]
>         s *= 2
>         cuda.syncthreads()  # Ensure block is synchronized
>
>     # Write result for this block to global memory
>     if tid == 0:
>         out[cuda.blockIdx.x] = data[i]
>
> def get_grid(n, tpb):
>     return (n + (tpb - 1)) // tpb  # Blocks per grid
>
> def reduce(x):
>     n = len(x)
>     bpg = get_grid(n, TPB)
>     out = cuda.device_array(bpg, dtype=x.dtype)
>     while bpg > 1:
>         reduce_kernel[bpg, TPB](x, out, n)
>         n = bpg
>         bpg = get_grid(n, TPB)
>         x[:n] = out[:n]
>     reduce_kernel[bpg, TPB](x, out, n)
>     return out
>
> def generate(n):
>     return np.random.rand(n).astype('float32')
>
> if __name__ == '__main__':
>     n = int(sys.argv[1])
>     print(reduce(generate(n))[0])
> ```

---

## Exercise 1.2 `[PRACTICE]`

For now, do *not* manually transfer the data to the GPU — let Numba do it when the kernel is called. Run your program for *n* = 4,000,000 under the nsys profiler and view the statistics. Where is time spent?

Hint: run `nsys profile -o <prof_data_file> python program.py <args>`.
To view statistics: `nsys stats <prof_data_file>.nsys-rep`.

> **Solution:**
>
> The profiler shows that the overwhelming majority of wall time is in **memory transfers**, not in the GPU kernels themselves. Specifically:
>
> - The GPU kernel summary shows the first (largest) kernel call dominates kernel execution time (91.8% of kernel time).
> - The memory transfer summary shows multiple HtoD (Host to Device) and DtoH (Device to Host) transfers occurring for every kernel call. This is because Numba implicitly copies the array to the GPU and back for each call — including the `x[:n] = out[:n]` line which triggers a DtoH transfer inside the loop.
>
> Example profiler output (abridged):
>
> ```
> [6/8] Executing 'gpukernsum' stats report
> Time (%)  Total Time (ns)  Instances  Avg (ns)  ...  Name
>     91.8           204735          1  204735.0  ...  reduce_kernel ...
>      3.2             7040          1    7040.0  ...  reduce_kernel ...
>
> [7/8] Executing 'gpumemtimesum' stats report
> Time (%)  Total Time (ns)  Count  Avg (ns)       Operation
>     56.4         13064802      4  3266200.5  [CUDA memcpy HtoD]
>     43.6         10102040      8  1262755.0  [CUDA memcpy DtoH]
>
> [8/8] Executing 'gpumemsizesum' stats report
> Total (MB)  Count  ...  Operation
>     64.126      8  ...  [CUDA memcpy DtoH]
>     64.000      4  ...  [CUDA memcpy HtoD]
> ```
>
> The large amount of DtoH traffic (64 MB) reveals that Numba is unnecessarily copying device arrays back to the host repeatedly inside the reduction loop.

---

## Exercise 1.3 `[PRACTICE]`

Modify your program such that you manually transfer the data to the GPU to avoid wasted transfers. Re-run your program under the nsys profiler. What has changed?

> **Solution:**
>
> Before calling `reduce`, explicitly move the data to the GPU with `cuda.to_device(x)` and only retrieve the final result at the end. This eliminates the repeated implicit HtoD/DtoH round-trips.
>
> After manual transfer, the profiler shows:
> - Only **one HtoD transfer** (16 MB — the full input array, once) and **one DtoH transfer** (4 bytes — the single scalar result).
> - Total memory transfer time drops from ~23 ms to ~3.3 ms.
> - Kernel execution times are essentially unchanged.
>
> Example profiler output (abridged):
>
> ```
> ** GPU MemOps Summary (by Time) (gpumemtimesum):
> Time (%)  Total Time (ns)  Count  Avg (ns)       Operation
>     99.9          3347336      1  3347336.0  [CUDA memcpy HtoD]
>      0.1             1792      1     1792.0  [CUDA memcpy DtoH]
>
> ** GPU MemOps Summary (by Size) (gpumemsizesum):
> Total (MB)  Count  ...  Operation
>     16.000      1  ...  [CUDA memcpy HtoD]
>      0.000      1  ...  [CUDA memcpy DtoH]
> ```

---

## Exercise 1.4 `[PRACTICE]`

Modify the kernel such that each thread block loads its part of the data into a shared memory array. Then, do the reduction in the shared array instead of in the global `data` array. Note: this will not have a large effect on speed now, but will make the next optimization simpler (less book keeping) and faster.

> **Solution:**
>
> Declare a shared array with `cuda.shared.array(TPB, dtype=data.dtype)`, load one element per thread into it, synchronize, then perform the reduction entirely in shared memory. Write only the block result back to global memory.
>
> ```python
> @cuda.jit
> def reduce_kernel(data, out, n):
>     # Declare shared memory
>     sdata = cuda.shared.array(TPB, dtype=data.dtype)
>
>     # Get the 1D grid and block indices
>     tid = cuda.threadIdx.x
>     i = cuda.grid(1)
>
>     # Each thread loads one element from global to shared memory
>     sdata[tid] = data[i] if i < n else 0.0
>     cuda.syncthreads()  # Ensure all threads have loaded data
>
>     # Do reduction for threadblock
>     s = 1
>     while s < cuda.blockDim.x:
>         if tid % (2 * s) == 0:
>             sdata[tid] += sdata[tid + s]
>         s *= 2
>         cuda.syncthreads()  # Ensure block is synchronized
>
>     # Write result for this block to global memory
>     if tid == 0:
>         out[cuda.blockIdx.x] = sdata[0]
> ```
>
> Key points:
> - `cuda.shared.array` is allocated per-block and is much faster to access than global memory.
> - Indexing now uses `tid` (thread index within block) instead of `i` (global thread index).
> - `cuda.syncthreads()` after loading is mandatory to prevent data races before the reduction starts.
> - The speed improvement is small at this stage — the main benefit is the setup for the next optimization (warp divergence fix).

---

## Exercise 1.5 `[PRACTICE]`

A major reason the current kernel is slow is the `if tid % (s * 2) == 0` condition. This causes a lot of warp divergence which slows the kernel down. Instead of "masking out" threads, change the kernel to use a strided index as shown on [slide 11 of the Mark Harris presentation](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf#page=11).

> **Solution:**
>
> Replace the modulo mask with a strided index `idx = 2 * s * tid`. Threads with a low `tid` do all the work, keeping active threads contiguous within a warp and eliminating divergence.
>
> ```python
> @cuda.jit
> def reduce_kernel(data, out, n):
>     # Declare shared memory
>     sdata = cuda.shared.array(TPB, dtype=data.dtype)
>
>     # Get the 1D grid and block indices
>     tid = cuda.threadIdx.x
>     i = cuda.grid(1)
>
>     # Each thread loads one element from global to shared memory
>     sdata[tid] = data[i] if i < n else 0.0
>     cuda.syncthreads()  # Ensure all threads have loaded data
>
>     # Do reduction for threadblock
>     s = 1
>     while s < cuda.blockDim.x:
>         idx = 2 * s * tid
>         if idx < cuda.blockDim.x:
>             sdata[idx] += sdata[idx + s]
>         s *= 2
>         cuda.syncthreads()  # Ensure block is synchronized
>
>     # Write result for this block to global memory
>     if tid == 0:
>         out[cuda.blockIdx.x] = sdata[0]
> ```
>
> The key change: `idx = 2 * s * tid` means that in each round, only the first half of threads (those with small `tid`) are active. Threads within the same warp now tend to make the same branch decision, reducing warp divergence significantly.

---

## Exercise 1.6 `[PRACTICE]`

Profile the optimized kernel (from Exercise 1.5). What has changed?

> **Solution:**
>
> The strided-index kernel is substantially faster. The largest kernel call drops from ~188,670 ns (shared memory version) to ~81,696 ns — roughly a **2.3x speedup**.
>
> Example profiler output (abridged):
>
> ```
> ** CUDA GPU Kernel Summary (gpukernsum):
> Time (%)  Total Time (ns)  Instances  Avg (ns)  ...  Name
>     78.6            81696          1   81696.0  ...  reduce_kernel ...
>      4.8             4991          1    4991.0  ...  reduce_kernel ...
>      3.5             3680          1    3680.0  ...  reduce_kernel ...
>      3.5             3648          1    3648.0  ...  reduce_kernel ...
>
> ** GPU MemOps Summary (by Time) (gpumemtimesum):
> Time (%)  Total Time (ns)  Count  Avg (ns)       Operation
>     99.9          3279721      1  3279721.0  [CUDA memcpy HtoD]
>      0.1             1792      1     1792.0  [CUDA memcpy DtoH]
> ```
>
> Memory transfers are unchanged (only one HtoD, one DtoH — same as after Exercise 1.3). The gain is entirely from reduced warp divergence inside the kernel.

---

## Exercise 1.7 `[PRACTICE]` (Optional)

These exercises were only the first few optimizations presented in the above reference. There is still plenty of performance to be squeezed out! Implement some (or all) of the remaining optimizations. The code in the presentation is in C++, but everything translates neatly to Numba functions.

Note: modern CUDA versions have also introduced helper functions that can make reductions even faster. See [this NVIDIA technical blog](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/).

Hint: you may need to apply the reduction to a larger set of numbers to really see speed benefits.

> **Solution:**
>
> Applying all optimizations from the Mark Harris presentation (loop unrolling, multiple elements per thread, warp-level reduction without `syncthreads`) yields the following fully optimized kernel:
>
> ```python
> TPB = 126  # Threads per block
>
> @cuda.jit(device=True)
> def warp_reduce(sdata, tid):
>     sdata[tid] += sdata[tid + 32]
>     sdata[tid] += sdata[tid + 16]
>     sdata[tid] += sdata[tid + 8]
>     sdata[tid] += sdata[tid + 4]
>     sdata[tid] += sdata[tid + 2]
>     sdata[tid] += sdata[tid + 1]
>
> @cuda.jit
> def reduce_kernel(data, out, n):
>     # Declare shared array
>     sdata = cuda.shared.array(shape=TPB, dtype=data.dtype)
>
>     # Each thread loads many elements from global to shared memory
>     tid = cuda.threadIdx.x
>     i = cuda.blockIdx.x * cuda.blockDim.x * 2 + cuda.threadIdx.x
>     grid_size = cuda.blockDim.x * 2 * cuda.gridDim.x
>
>     sdata[tid] = 0.0
>     while i < n:
>         sdata[tid] += data[i]
>         sdata[tid] += data[i + cuda.blockDim.x] \
>                     if i + cuda.blockDim.x < n else 0.0
>         i += grid_size
>     cuda.syncthreads()
>
>     # Do reduction in shared memory
>     if TPB >= 512:
>         if tid < 256:
>             sdata[tid] += sdata[tid + 256]
>         cuda.syncthreads()
>     if TPB >= 256:
>         if tid < 128:
>             sdata[tid] += sdata[tid + 128]
>         cuda.syncthreads()
>     if TPB >= 128:
>         if tid < 64:
>             sdata[tid] += sdata[tid + 64]
>         cuda.syncthreads()
>
>     if tid < 32:
>         warp_reduce(sdata, tid)
>
>     # Write result for this block to global memory
>     if tid == 0:
>         out[cuda.blockIdx.x] = sdata[0]
>
> def get_grid(n, tpb):
>     return (n + (tpb - 1)) // tpb  # Blocks per grid
>
> def reduce(x):
>     n = len(x)
>     bpg = get_grid(n, TPB)
>     bpg = (bpg + 1) // (2 * 2048)  # Determined by playing with values
>     out = cuda.device_array(bpg, dtype=x.dtype)
>     while bpg > 1:
>         reduce_kernel[bpg, TPB](x, out, n)
>         n = bpg
>         bpg = get_grid(n, TPB)
>         bpg = (bpg + 1) // 2
>         x[:n] = out[:n]
>     reduce_kernel[bpg, TPB](x, out, n)
>     return out
> ```
>
> On 400,000,000 random numbers: the strided-index version takes 7.61 ms; the fully optimized version takes 1.85 ms — a **4.1x additional speedup**, and roughly **9.4x faster** than the original naive kernel. Almost an order of magnitude total improvement.

---

## Section 2: CuPy

This section revisits the haversine distance matrix code from Week 4 and accelerates it with CuPy. The original NumPy implementations were:

```python
def distance_matrix_oneloop(p1, p2):
    p1 = np.radians(p1)
    p2 = np.radians(p2)

    D = np.empty((len(p1), len(p2)))
    for i in range(len(p1)):
        dsin2 = np.sin(0.5 * (p1[i] - p2)) ** 2
        cosprod = np.cos(p1[i, 0]) * np.cos(p2[:, 0])
        a = dsin2[:, 0] + cosprod * dsin2[:, 1]
        row = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        D[i, :] = row

    D *= 6371  # Earth radius in km
    return D

def distance_matrix_noloop(p1, p2):
    p1 = np.radians(p1)
    p2 = np.radians(p2)
    dsin2 = np.sin(0.5 * (p1[:, None, :] - p2[None, :, :])) ** 2
    cosprod = np.cos(p1[:, None, 0]) * np.cos(p2[None, :, 0])
    D = 2 * np.arcsin(np.sqrt(dsin2[:, :, 0] + cosprod * dsin2[:, :, 1]))
    D *= 6371  # Earth radius in km
    return D
```

Location data on the cluster: `/dtu/projects/02613_2025/data/locations/`

---

## Exercise 2.1 `[AUTOLAB]`

Convert both implementations such that they use CuPy operations instead of NumPy. Also modify your Python program from week 4 that used the distance matrix functions to use CuPy.

> **Solution:**
>
> Replace all `np.` calls with `cp.` (CuPy). Use `cp.loadtxt` to load data directly onto the GPU and `cp.triu_indices` for the stats calculation. The student file `distance_matrix_cupy.py` contains the complete implementation:
>
> ```python
> import sys
> import cupy as cp
>
>
> def distance_matrix_oneloop(p1, p2):
>     p1 = cp.radians(p1)
>     p2 = cp.radians(p2)
>
>     D = cp.empty((len(p1), len(p2)))
>     for i in range(len(p1)):
>         dsin2 = cp.sin(0.5 * (p1[i] - p2)) ** 2
>         cosprod = cp.cos(p1[i, 0]) * cp.cos(p2[:, 0])
>         a = dsin2[:, 0] + cosprod * dsin2[:, 1]
>         D[i, :] = 2 * cp.arctan2(cp.sqrt(a), cp.sqrt(1 - a))
>
>     D *= 6371
>     return D
>
>
> def distance_matrix_noloop(p1, p2):
>     p1 = cp.radians(p1)
>     p2 = cp.radians(p2)
>
>     dsin2 = cp.sin(0.5 * (p1[:, None, :] - p2[None, :, :])) ** 2
>     cosprod = cp.cos(p1[:, None, 0]) * cp.cos(p2[None, :, 0])
>     a = dsin2[:, :, 0] + cosprod * dsin2[:, :, 1]
>     D = 2 * cp.arctan2(cp.sqrt(a), cp.sqrt(1 - a))
>
>     D *= 6371
>     return D
>
>
> def load_points(fname):
>     data = cp.loadtxt(fname, delimiter=',', skiprows=1, usecols=(1, 2))
>     return data
>
>
> def distance_stats(D):
>     assert D.shape[0] == D.shape[1], 'D must be square'
>     idx = cp.triu_indices(D.shape[0], k=1)
>     distances = D[idx]
>     return {
>         'mean': float(distances.mean()),
>         'std': float(distances.std()),
>         'max': float(distances.max()),
>         'min': float(distances.min()),
>     }
>
>
> fname = sys.argv[1]
> points = load_points(fname)
> D = distance_matrix_noloop(points, points)
> stats = distance_stats(D)
> print(stats)
> ```

---

## Exercise 2.2 `[PRACTICE]`

Measure the run time for your CuPy program using the single loop version. Use the subset with 5000 rows `locations_5000.csv` file. How does the run time compare to the CPU version from week 4? You may need to run the programs a few times to get stable times. For simplicity, you may simply measure the run time with the `time` command: `time python points.py <path to data>`

> **Solution:**
>
> Run with: `time python distance_matrix_cupy.py /dtu/projects/02613_2025/data/locations/locations_5000.csv`
>
> Reference results (hardware-dependent):
> - NumPy CPU (fastest single-loop implementation from Week 4): **10.42 s**
> - CuPy single-loop version: **3.29 s**
>
> This is a **3x+ speedup** over the best CPU version, achieved with almost no code changes — just swapping `np` for `cp`.

---

## Exercise 2.3 `[PRACTICE]`

Re-run the program under the nsys profiler. Where does the program spend its time? Be sure to check all sections (except the OS Runtime Summary) to get a full overview.

> **Solution:**
>
> Profile with: `nsys profile -o cupy_oneloop python distance_matrix_cupy.py <data_path>`
> Then: `nsys stats cupy_oneloop.nsys-rep`
>
> Key findings from the profiler:
>
> 1. **CUDA API Summary**: `cuLaunchKernel` alone accounts for 65% of total API time — more than the sum of all actual kernel execution times. There are 135,485 kernel launches for the 5000-row single-loop version.
>
> 2. **GPU Kernel Summary**: Many small, fast kernels (e.g., `cupy_sin`, `cupy_multiply`, `cupy_arcsin`) are each called 13,545 times (once per loop iteration × number of operations). Each individual call is only ~3,000 ns, but the sheer count dominates.
>
> 3. **GPU MemOps Summary**: 1,467 MB of Device-to-Device (DtoD) copies are performed — CuPy creates and copies temporary intermediate arrays for each element-wise operation inside the loop.
>
> **Lesson**: For CuPy, repeatedly launching many small operations (as in the single-loop version) is expensive. The kernel launch overhead and DtoD memory copies outweigh the compute gains. This is different from NumPy on CPU, where the single-loop approach performs well.

---

## Exercise 2.4 `[PRACTICE]`

Measure the run time for your CuPy program using the no loop version. What is the new run time?

> **Solution:**
>
> Switch from `distance_matrix_oneloop` to `distance_matrix_noloop` and re-run:
> `time python distance_matrix_cupy.py /dtu/projects/02613_2025/data/locations/locations_5000.csv`
>
> Reference results:
> - CuPy single-loop version: 3.29 s
> - CuPy no-loop version: **1.00 s**
>
> Another **3x+ speedup** over the single-loop CuPy version, and an **order of magnitude faster** than the original NumPy CPU version. By issuing one large array operation instead of thousands of small ones, CuPy can fully saturate the GPU.

---

## Exercise 2.5 `[PRACTICE]`

Re-run the program under the nsys profiler. Where does the program now spend its time? What has changed compared to the single loop version? Be sure to check all sections to get a full overview.

> **Solution:**
>
> Key changes compared to the single-loop profile:
>
> 1. **CUDA API Summary**: `cuLaunchKernel` drops to just 0.2% of API time (45 calls vs. 135,485). The dominant cost is now `cuModuleLoadData` (JIT compilation of CuPy kernels, a one-time cost) and `cudaMalloc`.
>
> 2. **GPU Kernel Summary**: Each CuPy operation (`sin`, `multiply`, `subtract`, etc.) now runs only **1-2 times** instead of thousands. Each instance is much larger (e.g., `cupy_sin` takes 7.6 ms for one call vs. 3,000 ns × 13,545 calls).
>
> 3. **GPU MemOps Summary**: The `[CUDA memcpy DtoD]` row is **completely gone**. There are no longer any device-to-device copies — all computation happens in large contiguous array operations.
>
> **Moral**: When using CuPy, large vectorized array operations are critical to GPU performance. Each operation launch has overhead; avoid loops that issue many small GPU operations. This is the GPU equivalent of the NumPy vectorization lesson from Week 4, but the effect is even more pronounced on a GPU.
