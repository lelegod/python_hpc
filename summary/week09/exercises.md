# Week 9 Exercises — Numba (JIT) + GPU/CUDA Computing

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md) · [CUDA Grid Visual](cuda_grid_visual.md)

## Contents

- [Exercise 1: CPU Matrix Multiplication](#exercise-1-cpu-matrix-multiplication)
  - [1.1 `[PRACTICE]`](#11-practice)
  - [1.2 `[PRACTICE]`](#12-practice)
  - [1.3 `[AUTOLAB]`](#13-autolab)
  - [1.4 `[PRACTICE]`](#14-practice)
  - [1.5 (Optional) `[PRACTICE]`](#15-optional-practice)
- [Exercise 2: CUDA Vector Addition](#exercise-2-cuda-vector-addition)
  - [2.1 `[AUTOLAB]`](#21-autolab)
  - [2.2 `[PRACTICE]`](#22-practice)
  - [2.3 `[AUTOLAB]`](#23-autolab)
  - [2.4 `[PRACTICE]`](#24-practice)
  - [2.5 `[PRACTICE]`](#25-practice)
- [Exercise 3: CUDA Matrix Multiplication](#exercise-3-cuda-matrix-multiplication)
  - [3.1 `[PRACTICE]`](#31-practice)
  - [3.2 `[PRACTICE]`](#32-practice)
  - [3.3 `[PRACTICE]`](#33-practice)
  - [3.4 `[PRACTICE]`](#34-practice)

---

---

## Exercise 1: CPU Matrix Multiplication

Before using the GPU, we will warm up with a CPU exercise to get used to Numba and JIT compiled code. In this exercise, we will explore implementations of matrix multiplication, checking the performance of each implementation and how it is related with how the CPU caches. Consider the following Python implementation of matrix multiplication:

```python
import numpy as np

def matmul(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C
```

### 1.1 `[PRACTICE]`

Using the `@jit(nopython=True)` decorator, add Numba JIT compilation to the above function.

> **Solution:**
>
> Add the decorator from `numba` above the function definition:
>
> ```python
> import numpy as np
> from numba import jit
>
> @jit(nopython=True)
> def matmul_jit(A, B):
>     C = np.zeros((A.shape[0], B.shape[1]))
>     for i in range(A.shape[0]):
>         for j in range(B.shape[1]):
>             for k in range(A.shape[1]):
>                 C[i, j] += A[i, k] * B[k, j]
>     return C
> ```

---

### 1.2 `[PRACTICE]`

Measure the wall time of the original and JIT compiled version for two 100×100 matrices. How much faster is the JIT compiled version? Remember to run the JIT compiled version once before timing so it can compile.

> **Solution:**
>
> ```python
> from time import perf_counter as time
>
> # Original
> t = time()
> for _ in range(10):
>     matmul(A, B)
> print((time() - t) / 10 * 1000, 'ms')
>
> # JIT compiled
> matmul_jit(A, B)  # Run once to compile
> t = time()
> for _ in range(1000):  # Need more iterations for stable time
>     matmul_jit(A, B)
> print((time() - t) / 1000 * 1000, 'ms')
> ```
>
> Using an Intel Xeon Gold 6142 CPU, the original function takes ~468.78 ms and the JIT compiled version takes ~0.99 ms — a speed-up of over 470x.

---

### 1.3 `[AUTOLAB]`

Assuming `A`, `B` and `C` are stored row-wise, the innermost loop (over `k`) is not accessing `B` in a cache efficient manner. Make a new version of `matmul` where you re-order the loops such that access is cache efficient.

> **Solution:**
>
> The fix is to swap the `j` and `k` loops so the innermost loop iterates over `j` (the column index of `B`), giving contiguous row-access of `B` in row-major order. The loop order becomes `i → k → j`:
>
> ```python
> import numpy as np
> from numba import jit
>
> @jit(nopython=True)
> def matmul(A, B):
>     C = np.zeros((A.shape[0], B.shape[1]))
>     for i in range(A.shape[0]):
>         for k in range(A.shape[1]):      # swap k and j vs original ijk
>             for j in range(B.shape[1]):
>                 C[i, j] += A[i, k] * B[k, j]
>     return C
> ```

---

### 1.4 `[PRACTICE]`

Measure the performance of your optimized version for N×N matrices, where N is at least 200. What do you observe?

> **Solution:**
>
> Using an Intel Xeon Gold 6142 CPU with 200×200 matrices:
> - Original loop order (`i→j→k`): ~8.46 ms
> - Optimized loop order (`i→k→j`): ~1.39 ms
>
> Roughly a 6x speed-up from optimizing for cache efficiency. The innermost loop now accesses consecutive memory locations in both `B` (row-wise) and `C` (row-wise), dramatically reducing cache misses.

---

### 1.5 (Optional) `[PRACTICE]`

Measure the performance of your optimized version in MFLOP/s for N×N matrices where N goes from 20 to 2000. Plot the performance against the size of the matrix in kilobytes. Run the measurement as a batch job where you specify a CPU model so your results are repeatable. Mark the sizes of each cache level on your plot. What do you observe?

Hint: the matrix multiplication uses N³ FLOPs.
Hint: recall how you measured performance from the exercises in week 3.

> **Solution:**
>
> Job script (`submit.sh`):
>
> ```bash
> #!/bin/bash
> #BSUB -q hpc
> #BSUB -J mm_cpu
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -R "rusage[mem=8GB]"
> #BSUB -R "select[model == XeonGold6226R]"
> #BSUB -W 00:20
> #BSUB -o batch_output/matmul_cpu_%J.out
> #BSUB -e batch_output/matmul_cpu_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> lscpu
> python -u timematmulcpu.py
> ```
>
> Timing script:
>
> ```python
> from time import perf_counter as time
> import numpy as np
> from numba import jit
>
> @jit(nopython=True)
> def matmul_jit(A, B):
>     C = np.empty((A.shape[0], B.shape[1]))
>     for i in range(A.shape[0]):
>         for j in range(B.shape[1]):
>             for k in range(A.shape[1]):
>                 C[i, j] += A[i, k] * B[k, j]
>     return C
>
> @jit(nopython=True)
> def matmul_jit_optim(A, B):
>     # Optimized loop order (i->k->j)
>     ...
>
> ns = np.logspace(1, 3 + np.log10(2), 15).astype(int)
> rep = 10
>
> A = np.random.rand(ns[0], ns[0])
> B = np.random.rand(ns[0], ns[0])
>
> # Run once to make sure it's jitted
> matmul_jit(A, B)
> matmul_jit_optim(A, B)
>
> t1, t2 = [], []
> for n in ns:
>     A = np.random.rand(n, n)
>     B = np.random.rand(n, n)
>
>     t = time()
>     for i in range(rep):
>         matmul_jit(A, B)
>     t1.append((time() - t) / rep)
>
>     t = time()
>     for j in range(rep):
>         matmul_jit_optim(A, B)
>     t2.append((time() - t) / rep)
>
> print('ns = [' + ', '.join(map(str, ns)) + ']')
> print('t1 = [' + ', '.join(map(str, t1)) + ']')
> print('t2 = [' + ', '.join(map(str, t2)) + ']')
> ```
>
> The optimized loop order is significantly faster. Performance dips are visible when the matrices no longer fit in each cache level (L1, L2, L3).

---

## Exercise 2: CUDA Vector Addition

### 2.1 `[AUTOLAB]`

Make a CUDA kernel `add_kernel` that takes two vectors x and y as input and returns a new vector a where aᵢ = xᵢ + yᵢ, i.e., a is the element-wise sum of x and y.

> **Solution:**
>
> Student file: `/Users/kyleelyk/Documents/DTU/SEM2/python_hpc/week9/add_kernel.py`
>
> ```python
> import numpy as np
> from numba import cuda
> from time import perf_counter
>
> @cuda.jit
> def add_kernel(x, y, out):
>     i = cuda.grid(1)
>     if i < x.shape[0]:
>         out[i] = x[i] + y[i]
>
> def main():
>     n = 1_000_000
>     x = np.random.rand(n).astype(np.float32)
>     y = np.random.rand(n).astype(x.dtype)
>     out = np.empty_like(x)
>
>     threadsperblock = 256
>     blockspergrid = (n + threadsperblock - 1) // threadsperblock
>
>     # Warmup — ensures kernel is JIT compiled before timing
>     add_kernel[blockspergrid, threadsperblock](x, y, out)
>     cuda.synchronize()
>
>     rep = 200
>     t = perf_counter()
>     for _ in range(rep):
>         add_kernel[blockspergrid, threadsperblock](x, y, out)
>     cuda.synchronize()
>     print((perf_counter() - t) / rep * 1000, 'ms')
>
> if __name__ == '__main__':
>     main()
> ```
>
> Key points:
> - `@cuda.jit` marks the function as a CUDA kernel.
> - `cuda.grid(1)` returns the global thread index in a 1D grid.
> - The bounds check `if i < x.shape[0]` is essential because the grid may have more threads than array elements.
> - `threadsperblock = 256` and `blockspergrid = (n + threadsperblock - 1) // threadsperblock` ensures all elements are covered.

---

### 2.2 `[PRACTICE]`

Write a Python program that measures the run time of your kernel with two random vectors. Each input vector should be a NumPy array of length 1,000,000.
Hint: Run the kernel once before timing so it has been JIT compiled.

> **Solution:**
>
> ```python
> from time import perf_counter as time
> import numpy as np
> from numba import cuda
>
> @cuda.jit
> def add_kernel(x, y, out):
>     # Omitted — see 2.1
>     ...
>
> n = 1_000_000
> x = np.random.rand(n).astype(np.float32)
> y = np.random.rand(n).astype(x.dtype)
> out = np.empty_like(x)
>
> threadsperblock = 256
> blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
>
> add_kernel[blockspergrid, threadsperblock](x, y, out)  # Ensure compiled
>
> rep = 200
> t = time()
> for _ in range(rep):
>     add_kernel[blockspergrid, threadsperblock](x, y, out)
> cuda.synchronize()
> print((time() - t) / rep * 1000, 'ms')
> ```
>
> Note: `cuda.synchronize()` is called after the loop (not inside) to wait for all GPU work to complete before stopping the timer.

---

### 2.3 `[AUTOLAB]`

Run your timing program as a batch job so results are repeatable. Use the queue `c02613`.

> **Solution:**
>
> Student file: `/Users/kyleelyk/Documents/DTU/SEM2/python_hpc/week9/gpu_job.sh`
>
> ```bash
> #!/bin/bash
> #BSUB -q c02613
> #BSUB -J add_kernel
> #BSUB -n 4
> #BSUB -R "span[hosts=1]"
> #BSUB -R "rusage[mem=4GB]"
> #BSUB -gpu "num=1:mode=exclusive_process"
> #BSUB -W 00:10
> #BSUB -o add_kernel_%J.out
> #BSUB -e add_kernel_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> python add_kernel.py
> ```
>
> Key directives:
> - `-q c02613` — GPU queue for the course.
> - `-gpu "num=1:mode=exclusive_process"` — request 1 GPU in exclusive mode (no GPU sharing between jobs).
> - `-n 4` — 4 CPU cores (needed to feed the GPU).
>
> The average run time on the cluster is approximately 5.99 ms.

---

### 2.4 `[PRACTICE]`

Repeat your time measurements where you ensure the arrays reside in pinned memory before passing them to your CUDA kernel. What do you observe and why?
Hint: To get pinned memory, use `numba.cuda.pinned_array`, `numba.cuda.pinned_array_like` or `numba.cuda.pinned`.

> **Solution:**
>
> Wrap the timing code in the `cuda.pinned` context manager so that `x`, `y`, and `out` are pinned (page-locked) in CPU memory:
>
> ```python
> with cuda.pinned(x, y, out):
>     add_kernel[blockspergrid, threadsperblock](x, y, out)  # Ensure compiled
>
>     rep = 200
>     t = time()
>     for _ in range(rep):
>         add_kernel[blockspergrid, threadsperblock](x, y, out)
>     cuda.synchronize()
>     print((time() - t) / rep * 1000, 'ms')
> ```
>
> Pinned (page-locked) memory enables Direct Memory Access (DMA) transfers between CPU and GPU, bypassing OS paging. This speeds up host-to-device and device-to-host transfers significantly. Run time decreases to approximately 3.51 ms.
>
> Important: do not repeatedly pin memory inside the timing loop — pin it once before timing starts, otherwise you measure the pinning overhead rather than the kernel overhead.

---

### 2.5 `[PRACTICE]`

Repeat your time measurements where you ensure the arrays reside in GPU memory before passing them to your CUDA kernel. What do you observe and why? Based on your results, how much time is spent on data transfer vs. computation?

> **Solution:**
>
> Transfer the arrays to GPU memory once before timing, then time only the kernel launches:
>
> ```python
> # Transfer to GPU
> d_x = cuda.to_device(x)
> d_y = cuda.to_device(y)
> d_out = cuda.device_array(len(x))
>
> add_kernel[blockspergrid, threadsperblock](d_x, d_y, d_out)  # Ensure compiled
>
> rep = 200
> t = time()
> for _ in range(rep):
>     add_kernel[blockspergrid, threadsperblock](d_x, d_y, d_out)
> cuda.synchronize()
> print((time() - t) / rep * 1000, 'ms')
> ```
>
> With arrays already on the GPU, the run time drops to approximately 0.042 ms. Comparing:
> - GPU memory (pure compute): ~0.042 ms
> - Pinned CPU memory: ~3.51 ms
>
> This means only ~1.2% of the pinned-memory run time is actual computation — the remaining ~98.8% is memory transfer. Moral: for compute-light kernels like vector addition, memory transfer dominates. Minimizing transfers (keep data on GPU between kernel calls) is critical.

---

## Exercise 3: CUDA Matrix Multiplication

### 3.1 `[PRACTICE]`

Make a CUDA kernel `matmul_kernel(A, B, C)` that multiplies two matrices together. Let `A` and `B` be the input matrices and `C` be the output matrix you write to. You may assume the dimensions of `A` and `B` match so they can be multiplied. Structure your kernel such that you use a 2D compute grid where thread (`i`, `j`) computes the value of `C[i, j]`.
Hint: use `cuda.grid(2)` to get 2D grid position of the current thread.

> **Solution:**
>
> ```python
> from numba import cuda, float32
>
> @cuda.jit
> def matmul_kernel(A, B, C):
>     i, j = cuda.grid(2)
>     if i < C.shape[0] and j < C.shape[1]:
>         tmp = float32(0.)  # Avoids repeated memory access to C
>         for k in range(A.shape[1]):
>             tmp += A[i, k] * B[k, j]
>         C[i, j] = tmp
> ```
>
> Key points:
> - `cuda.grid(2)` returns the `(i, j)` global 2D thread position.
> - Bounds check required: the 2D grid may extend beyond the matrix dimensions.
> - Accumulating into a local `tmp` variable avoids repeated global memory writes to `C[i, j]` during the inner loop.

---

### 3.2 `[PRACTICE]`

Measure the run time of your kernel for `A` and `B` being 2048×2048 matrices and use 16×16 threads per block. Compare the run time with NumPy (e.g., by timing `A @ B`). What do you observe?

> **Solution:**
>
> ```python
> d_A = cuda.to_device(A)
> d_B = cuda.to_device(B)
> d_C = cuda.device_array((A.shape[0], B.shape[1]))
>
> matmul_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C)  # Warmup/compile
>
> rep = 20
> t = time()
> for i in range(rep):
>     matmul_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C)
> cuda.synchronize()
> print((time() - t) / rep * 1000, 'ms')
> ```
>
> Note: `threadsperblock` and `blockspergrid` are now 2-tuples since we define a 2D grid, e.g.:
> ```python
> threadsperblock = (16, 16)
> blockspergrid = (
>     (A.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
>     (B.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
> )
> ```
>
> Results (2048×2048, 16×16 threads/block, data already on GPU):
> - CUDA `matmul_kernel`: ~63 ms
> - NumPy `A @ B`: ~101.83 ms
>
> The GPU implementation is already ~1.6x faster than NumPy, even with a naive kernel and no shared memory tiling.

---

### 3.3 `[PRACTICE]`

Repeat your measurements where the matrices reside in pinned and device memory. What fraction of time is spent on memory transfers to and from the GPU with and without pinned memory?

> **Solution:**
>
> Pinned memory (wrap timing in `cuda.pinned`):
>
> ```python
> with cuda.pinned(A, B, C):
>     matmul_kernel[blockspergrid, threadsperblock](A, B, C)  # Warmup
>
>     rep = 20
>     t = time()
>     for i in range(rep):
>         matmul_kernel[blockspergrid, threadsperblock](A, B, C)
>     cuda.synchronize()
>     print((time() - t) / rep * 1000, 'ms')
> ```
>
> Device memory:
>
> ```python
> d_A = cuda.to_device(A)
> d_B = cuda.to_device(B)
> d_C = cuda.device_array((A.shape[0], B.shape[1]))
>
> matmul_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C)  # Warmup
>
> rep = 20
> t = time()
> for i in range(rep):
>     matmul_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C)
> cuda.synchronize()
> print((time() - t) / rep * 1000, 'ms')
> ```
>
> Results:
> - Default (pageable) CPU memory: ~63 ms
> - Pinned CPU memory: ~50.61 ms (~2x faster than NumPy)
> - Device (GPU) memory: ~42.06 ms (~2.4x faster than NumPy)
>
> Fraction of time on memory transfers:
> - Without pinned memory: ~33% transfer, ~67% computation
> - With pinned memory: ~17% transfer, ~83% computation
>
> The matrix multiplication kernel is much more compute-bound than vector addition, so the transfer fraction is much smaller (~17–33% vs ~99% for vector addition).

---

### 3.4 `[PRACTICE]`

Repeat your measurements using 256×1 and 1×256 threads per block. Let the matrices be in GPU memory before passing them to the kernel. What do you observe and why? Run your measurements as a batch job so results are repeatable.

> **Solution:**
>
> Note that 16×16 = 256, so the total number of threads per block is unchanged across all three configurations — only the shape differs.
>
> Results (2048×2048, data in GPU memory):
> - 16×16 threads/block: ~42.06 ms
> - 256×1 threads/block: ~435.59 ms (much slower)
> - 1×256 threads/block: ~29.23 ms (faster)
>
> Explanation — consider the inner loop of `matmul_kernel`:
> ```python
> for k in range(A.shape[1]):
>     tmp += A[i, k] * B[k, j]
> ```
> - **256×1 configuration**: each block covers 256 rows of `A` with a fixed column `j`. Reading `A[i, k]` iterates along a single column (non-contiguous in row-major memory), causing many cache misses — hence the slow runtime.
> - **1×256 configuration**: each block covers 256 columns of `B` with a fixed row `i`. Reading `B[k, j]` iterates along a row (contiguous in row-major memory), which is cache-efficient — hence the faster runtime.
>
> Moral: with CUDA, cache efficiency depends not just on the code logic but also on the layout of the compute grid. Always ensure data access patterns align with the grid structure.
