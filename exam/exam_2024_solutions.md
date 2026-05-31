# 02613 Python HPC — Exam 2024 Full Solutions
**Date:** 28 May 2024 | **Duration:** 4 hours | **Format:** Mixed (open-ended + MCQ) | **Pages:** 14

---

**Context:** All questions are set in a scenario where you are a performance consultant sent by DTU to help a company called Synergistic Logistics Operational Technologies Hub. They are doing many different kinds of data processing, all of which are currently running too slow.

**Global assumptions (stated on exam cover):**
- All arrays are NumPy arrays and stored row-wise (C-order).
- `np` refers to the NumPy Python module.

---

## Question 1 — Fixing a BSUB Job Script

One of the engineers wants to move some processing to batch jobs. They show you a job script they are working on. They are having issues with the resource specifications. The job script is:

```bash
#!/bin/bash
#BSUB -J proc
#BSUB -q hpc
#BSUB -W 00:05
#BSUB -R "rusage[mem=4GB]"
#BSUB -n 1
#BSUB -o proc_%J.out
#BSUB -e proc_%J.err

python process.py allthe.data
```

The program usually runs best with 8 cores. It uses 16 GB of memory and takes at most 15 minutes to complete.

**How would you change the job script above to accommodate this? Use only the resources you need.**

**Full Solution:**

The corrected script is:

```bash
#!/bin/bash
#BSUB -J proc
#BSUB -q hpc
#BSUB -W 00:15
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 8
#BSUB -o proc_%J.out
#BSUB -e proc_%J.err

python process.py allthe.data
```

**Line-by-line explanation of every change:**

1. `#BSUB -W 00:15` — Changed from `00:05` (5 minutes) to `00:15` (15 minutes). The program takes at most 15 minutes, so the wall-clock limit must be at least 15 minutes. Setting it to exactly 15 minutes uses only what you need.

2. `#BSUB -n 8` — Changed from `1` to `8`. The program runs best with 8 cores, so we request exactly 8 CPU slots. Requesting only 1 would mean the program cannot use multi-processing effectively.

3. `#BSUB -R "rusage[mem=2GB]"` — Changed from `4GB` to `2GB`. The `rusage[mem=...]` directive specifies memory **per core/slot**. Since we are requesting 8 cores and the total memory needed is 16 GB, the per-core memory is 16 GB / 8 = 2 GB. Setting `4GB` per core would request 32 GB total, which wastes cluster resources.

4. `#BSUB -R "span[hosts=1]"` — Added this new line. When requesting multiple cores (`-n 8`), LSF may spread them across multiple physical nodes by default. For a shared-memory multiprocessing program, all cores must be on the same node (so they share the same RAM). `span[hosts=1]` enforces this constraint.

**Key concept tested:** Correct LSF/BSUB resource specification — wall time, number of cores, per-core memory (not total memory), and the `span[hosts=1]` requirement for shared-memory parallelism.

---

## Question 2 — Parallel Fraction from a Speed-Up Plot (MCQ)

Another engineer is considering using multi-processing. A previous engineer created a speed-up plot. The plot shows speed-up vs. number of processes. The curve rises steeply at first, then flattens out and saturates at approximately 5.

**Given the above speed-up plot, what is the parallel fraction of the program?**

**Options:**
- A) 0.2
- B) 0.5
- C) 0.8
- D) 0.9

**Correct Answer: C) 0.8**

**Why C is correct:** Amdahl's Law states that the theoretical maximum speed-up with infinite processors is:

```
S_max = 1 / (1 - F)
```

where F is the parallel fraction. The plot clearly saturates (levels off) at a speed-up of approximately 5. Setting S_max = 5:

```
5 = 1 / (1 - F)
1 - F = 1/5 = 0.2
F = 0.8
```

Therefore the parallel fraction is 0.8.

**Why the others are wrong:**
- A) 0.2 — This would give S_max = 1/(1-0.2) = 1/0.8 = 1.25. The plot clearly shows speedups well above 1.25, so 0.2 cannot be the parallel fraction.
- B) 0.5 — This would give S_max = 1/(1-0.5) = 2. The plot saturates at ~5, not at 2.
- D) 0.9 — This would give S_max = 1/(1-0.9) = 10. The plot saturates well below 10 (at ~5), so 0.9 is too high.

---

## Question 3 — Should They Pursue Parallelization? (Open-ended)

Management wants at least a speed-up of 4 before they consider parallelization to be worth it. Currently, the most powerful machine at the company has 8 cores.

**Given your estimated parallel fraction (F = 0.8), should they pursue parallelization? Explain your reasoning.**

**Full Solution:**

Using Amdahl's Law with F = 0.8 and n = 8 cores:

```
S(8) = 1 / ((1 - F) + F/n)
     = 1 / ((1 - 0.8) + 0.8/8)
     = 1 / (0.2 + 0.1)
     = 1 / 0.3
     = 10/3
     ≈ 3.33
```

Since S(8) ≈ 3.33 < 4, **no, they should not pursue parallelization**.

With their most powerful machine (8 cores), the maximum achievable speed-up is approximately 3.33, which falls short of the required threshold of 4. Even with infinite cores, the maximum possible speed-up would be S_max = 5 (from Q2), so no feasible number of cores can reach the target of 4 — but crucially, their actual hardware ceiling of 8 cores keeps them below 4.

**Key concept tested:** Applying Amdahl's Law to a real decision-making scenario; understanding that the parallel fraction sets a hard ceiling on achievable speed-up and that hardware limits matter.

---

## Question 4 — Parallelization Approach for `process_number` (MCQ)

Another engineer is trying to compute the value of the following function for each value of an array `numbers = [1, 2, 3, ..., 100 000 000]`:

```python
def process_number(n):
    s = 0
    for i in range(n):
        s += i
    return s
```

They want to use parallelization to spread the work out to several workers.

**What Python parallelization approach should they use for this function?**

**Options:**
- A) Multi-threading
- B) Multi-processing
- C) Does not matter

**Correct Answer: B) Multi-processing**

**Why B is correct:** The `process_number` function is pure Python CPU-bound computation (a Python `for` loop doing arithmetic). Python has the Global Interpreter Lock (GIL), which prevents multiple threads from executing Python bytecode simultaneously. Multi-threading in Python does NOT provide parallelism for CPU-bound pure-Python code — threads take turns holding the GIL. Multi-processing creates separate processes, each with its own GIL and Python interpreter, so they can run truly in parallel on multiple CPU cores.

**Why the others are wrong:**
- A) Multi-threading — The GIL prevents true parallel execution of CPU-bound Python code. Multiple threads would execute serially (taking turns), providing no speed-up for this pure-Python loop.
- C) Does not matter — It does matter. Multi-threading would not help here due to the GIL; only multi-processing provides genuine parallelism.

---

## Question 5 — Static vs. Dynamic Scheduling for `process_number` (Open-ended)

**Should they use static scheduling or dynamic scheduling for this task? Explain your answer.**

**Full Solution:**

They should use **dynamic scheduling**.

The reason is that the time it takes to process each number `n` varies significantly across the input array. `process_number(n)` runs a loop of `range(n)` iterations, so its execution time is proportional to `n`. The array contains values from 1 to 100,000,000, meaning `process_number(1)` is nearly instantaneous while `process_number(100_000_000)` takes a very long time.

With **static scheduling**, each worker is assigned a fixed chunk of the input array upfront. If the chunks are assigned in order (e.g., worker 1 gets small numbers, worker 2 gets large numbers), the workload is severely imbalanced — the worker handling large numbers will take far longer to finish than others, leaving most cores idle while one struggles. Even with round-robin assignment, static scheduling cannot perfectly balance this highly variable workload.

With **dynamic scheduling**, work items are handed out one at a time (or in small chunks) as workers become free. This ensures that no worker sits idle while another is overloaded, leading to much better load balancing for this non-uniform task.

**Key concept tested:** Understanding that dynamic scheduling is appropriate when task durations are heterogeneous (unequal), which is the case here because execution time scales with `n`.

---

## Question 6 — Why `abssum` Cannot Be Used in a Parallel Reduction (Open-ended)

An engineer wants to compute the absolute value of a sum, i.e., |x_1 + x_2 + ... + x_n|. They want to use a parallel reduction tree and have written the following function for each partial result:

```python
def abssum(x, y):
        return abs(x + y)  # Compute |x + y|
```

However, they are having trouble getting correct answers consistently.

**Explain why this function cannot be used with the parallel reduction tree framework. What should the engineer do instead?**

**Full Solution:**

The `abssum` function **cannot be used** in a parallel reduction tree because it is **not associative**.

A parallel reduction tree requires the combining function to be associative, i.e., `f(f(a, b), c) == f(a, f(b, c))` must hold for all inputs. This is because the tree applies the function in different groupings depending on how the parallel tree is structured, and correctness requires that the grouping does not affect the result.

For `abssum`, associativity fails. Concrete counterexample:

```
abssum(abssum(1, 2), -3) = abssum(|1+2|, -3) = abssum(3, -3) = |3 + (-3)| = |0| = 0

abssum(1, abssum(2, -3)) = abssum(1, |2+(-3)|) = abssum(1, |-1|) = abssum(1, 1) = |1+1| = 2

0 ≠ 2
```

The absolute value "wrapper" breaks associativity because it discards sign information at intermediate steps, causing different groupings to produce different results.

**What they should do instead:** Perform a normal parallel sum-reduction (which is associative) to compute x_1 + x_2 + ... + x_n, and then take the absolute value of the final result at the end:

```python
def sum_reduce(x, y):
    return x + y  # Simple addition — associative

total = parallel_reduce(sum_reduce, values)
result = abs(total)  # Take abs only once, at the end
```

**Key concept tested:** The associativity requirement for parallel reduction trees; understanding why applying a non-associative transformation inside the reduction produces incorrect results.

---

## Question 7 — NumPy Broadcasting to Subtract a Mean Image (MCQ)

An engineer is working with a collection of RGB images stored as an N x H x W x 3 array named `images` where N is the number of images, H is the image height, W is the image width, and the last axis represents the RGB channels. They want to subtract a mean image `mim` stored as an H x W array from each color channel in every image in `images`.

**Which of the following NumPy operations will achieve this?**

**Options:**
- A) `images - mim`
- B) `images - mim[:, :, None]`
- C) `images - mim[None]`

**Correct Answer: B) `images - mim[:, :, None]`**

**Why B is correct:** `images` has shape (N, H, W, 3). We want to subtract a value for each (H, W) pixel position from all N images and all 3 channels. `mim` has shape (H, W). We need to broadcast `mim` across the N and channel dimensions.

`mim[:, :, None]` reshapes `mim` from (H, W) to (H, W, 1). NumPy broadcasting then expands this to (N, H, W, 3) by prepending a new axis for N and expanding the trailing 1 to 3. This correctly subtracts the same mean pixel value from each channel at each spatial position in each image.

**Why the others are wrong:**
- A) `images - mim` — NumPy tries to broadcast (N, H, W, 3) against (H, W). Broadcasting aligns shapes from the right: the last two dimensions of `images` are (W, 3) and `mim` has shape (H, W). These are not compatible (W != H in general, and there is no channel axis in `mim`). This will raise a shape error or produce wrong results.
- C) `images - mim[None]` — `mim[None]` reshapes `mim` from (H, W) to (1, H, W). Broadcasting against (N, H, W, 3) aligns as: the last dimension of `mim[None]` is W and the last dimension of `images` is 3. Unless W == 3, this will fail or produce wrong results. Even if W == 3, the semantics would be wrong.

---

## Question 8 — Re-ordering Loops for Cache Efficiency (Open-ended)

An engineer is using Numba to speed up processing of the `images` array (shape N x H x W x 3). Their function is:

```python
@jit(nopython=True)
def process(images):
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            for k in range(images.shape[2]):
                for l in range(images.shape[3]):
                    x = images[i, j, k, l]
                    ...
```

They tried reversing the loop order but it did not help. You inspect the strides of the `images` array and get `(600, 40, 8, 200)`.

**Based on this information, how would you re-order the loops in the `process` function? Explain your answer.**

**Full Solution:**

For cache-efficient memory access, the innermost loop should iterate over the axis with the **shortest (smallest) stride**, because the shortest stride means consecutive iterations access memory locations that are closest together — maximally exploiting spatial locality and cache line reuse.

The strides are:
- Axis 0 (i, images N-axis): stride 600
- Axis 1 (j, images H-axis): stride 40
- Axis 2 (k, images W-axis): stride 8
- Axis 3 (l, images channel-axis): stride 200

Ranking from shortest to longest stride:
1. Axis 2 (k): stride 8 — shortest, should be the **innermost** loop
2. Axis 1 (j): stride 40
3. Axis 0 (i): stride 600
4. Axis 3 (l): stride 200

Ordering from innermost to outermost: **k, j, l, i**

The re-ordered function:

```python
@jit(nopython=True)
def process(images):
    for i in range(images.shape[0]):
        for l in range(images.shape[3]):
            for j in range(images.shape[1]):
                for k in range(images.shape[2]):
                    x = images[i, j, k, l]
                    ...
```

**Note:** Simply "reversing" the original loop order gives (l, k, j, i) with innermost being l (stride 200), which is not optimal. The correct approach is to sort by stride value, not simply reverse.

**Key concept tested:** Stride-based loop ordering to maximise cache efficiency; understanding that the shortest stride axis should be innermost to minimise cache misses.

---

## Question 9 — Reading a Profiler Output: Number of Samples (MCQ)

The statistics department has a Python program. The profiler output (for a small data subset) is:

```
238 function calls (237 primitive calls) in 44.126 seconds

Ordered by: cumulative time

ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   2/1    0.000    0.000   44.126   44.126 {built-in method builtins.exec}
     1    0.000    0.000   44.126   44.126 process.py:1(<module>)
     1    0.000    0.000   20.009   20.009 ourlib.py:3(load_params)
     1    0.000    0.000   15.013   15.013 ourlib.py:8(prepare_model)
    10    0.000    0.000    5.055    0.505 ourlib.py:13(process_sample)
     1    0.000    0.000    3.007    3.007 ourlib.py:18(save)
     1    0.000    0.000    1.001    1.001 ourlib.py:26(load_data)
     1    0.000    0.000    0.042    0.042 <frozen importlib._bootstrap>:97...
     1    0.000    0.000    0.042    0.042 <frozen importlib._bootstrap>:94...
     1    0.000    0.000    0.041    0.041 <frozen importlib._bootstrap>:66...
     ...
```

**How many samples were in the data subset?**

**Options:**
- A) 2
- B) 5
- C) 10

**Correct Answer: C) 10**

**Why C is correct:** The profiler shows that `process_sample` (ourlib.py:13) was called **10 times** (`ncalls = 10`). Looking at the program code, `process_sample` is called once per sample inside the loop `for s in samples`. Therefore, the data subset contained exactly 10 samples.

**Why the others are wrong:**
- A) 2 — The only entry with ncalls=2 is the built-in `exec` call (shown as `2/1` which means 2 total calls, 1 primitive). This is an interpreter artifact, not related to sample count.
- B) 5 — No function appears 5 times. The per-call time for `process_sample` is 0.505 seconds, and 10 * 0.505 = 5.055, but the number of calls is 10, not 5.

---

## Question 10 — Which Function to Optimize for Normal Workloads (Open-ended)

**On what function would you focus your optimization efforts with respect to normal workloads? Explain your answer.**

**Full Solution:**

Focus optimization efforts on **`process_sample`**.

**Reasoning from the profiler data:**

For the 10-sample subset, `process_sample` took a total cumulative time of 5.055 seconds, with a per-call time of 0.505 seconds per sample.

The normal workload is **at least 1000 samples** at a time. If we scale `process_sample` linearly (which is safe to assume since it is called independently for each sample):

```
Expected time for 1000 samples = 1000 * 0.505 = 505 seconds
```

Compare this to the other functions:
- `load_params`: 20.009 s — called once, does not scale with sample count
- `prepare_model`: 15.013 s — called once, does not scale with sample count
- `save`: 3.007 s — called once (may scale slightly, but unclear)
- `load_data`: 1.001 s — called once

At 1000 samples, `process_sample` alone would dominate the runtime at ~505 seconds, which is far larger than all the one-time setup costs combined (~39 seconds). Therefore, optimizing `process_sample` will have by far the greatest impact on real workload performance.

Note: `load_data` and `save` may also scale with more samples, but this is impossible to determine from the given profiler output, whereas `process_sample` is definitively the bottleneck.

**Key concept tested:** Correctly interpreting `cProfile` output; distinguishing fixed-cost one-time calls from per-sample costs; projecting profiler results to realistic workload sizes (Amdahl intuition applied to profiling).

---

## Question 11 — Cache Efficiency: CPU Parallel Version of `conv_channels` (Open-ended)

The machine learning department is processing multi-spectral images where each image is of size h x w and each pixel stores c channels. They need a function that convolves each pixel with a length-c kernel. Their parallel CPU function in pseudo-code is:

```python
def conv_channels(image, kernel, out, h, w, c):
    # Process each pixel in parallel
    parallel for i in range(h):
        parallel for j in range(w):
            o = 0.0
            # Loop over channels for current pixel
            for k in range(c):
                o += image[???] * kernel[k]
            out[i, j] = o
```

The two storage options being considered are:
- `image` has shape c x h x w (channels-first)
- `image` has shape h x w x c (channels-last)

**Given the above pseudo-code, how should the `image` array be stored to maximize cache efficiency? Explain your answer.**

**Full Solution:**

The `image` array should be stored with **channels as the last dimension**, i.e., shape **h x w x c**.

**Reasoning:**

The innermost loop in this function iterates over `k` (the channel index). For cache efficiency, consecutive iterations of the innermost loop should access memory locations that are as close together as possible. Since arrays are stored row-wise (C-order), the last axis has the shortest stride (stride = size of one element).

- If `image` has shape **h x w x c** (channels-last): `image[i, j, k]` accesses elements along the last axis as `k` increments. Consecutive values of `k` are contiguous in memory. Each iteration of the inner `k`-loop steps by one element — maximally cache-friendly.

- If `image` has shape **c x h x w** (channels-first): `image[k, i, j]` accesses elements along the first axis as `k` increments. The stride along the first axis is h*w elements. Consecutive values of `k` are separated by h*w elements in memory — cache-unfriendly, producing many cache misses.

Since each thread (handling one pixel at (i,j)) runs the inner `k`-loop, aligning the channel axis with the shortest stride ensures maximum cache re-use.

**Key concept tested:** Connecting loop structure to memory layout; understanding that the innermost loop variable should index the axis with the shortest stride (last axis for row-major storage).

---

## Question 12 — Cache Efficiency: CUDA Kernel Version of `conv_channels` (Open-ended)

To further increase performance, they explore GPU processing with CUDA. The CUDA kernel is:

```python
@cuda.jit
def conv_channels_kernel(image, kernel, out, h, w, c):
    i, j = cuda.grid(2)
    if i < h and j < w:
        o = 0.0
        for k in range(c):
            o += image[???] * kernel[k]
        out[i, j] = o
```

The kernel is called with thread blocks of size 32 x 32.

**Given the above implementation, how should the `image` array be stored to maximize cache efficiency? Explain your answer.**

**Full Solution:**

The `image` array should be stored with **channels as the first dimension**, i.e., shape **c x h x w**.

**Reasoning:**

The GPU memory access pattern is fundamentally different from the CPU case (Q11). The key considerations are:

1. **Coalesced global memory access (warp-level):** In CUDA, threads in the same warp execute the same instruction simultaneously. For global memory reads to be coalesced (i.e., served in as few memory transactions as possible), threads in a warp should access consecutive memory addresses in the same instruction. A 32x32 thread block has 32 warps; the threads in each warp differ in their `j` index (column). When all threads in a warp execute `image[k, i, j]` (channels-first layout), they access `image[k, i, j], image[k, i, j+1], ..., image[k, i, j+31]` — consecutive memory addresses along the `w` axis. This is perfectly coalesced.

2. **If channels-last (h x w x c):** All threads in a warp execute `image[i, j, k]`. For a fixed `i` and `k`, threads access `image[i, j, k], image[i, j+1, k], ..., image[i, j+31, k]`. These are separated by `c` elements in memory (stride c), not contiguous. This produces uncoalesced (strided) memory accesses, which is very inefficient on a GPU.

3. **Shared cache across threads:** Threads in the same thread block share the L1 cache/shared memory. When all threads in a block process their inner `k`-loop simultaneously, they each access a different channel slice `image[k, i, j]`. With channels-first layout, all the (i,j) values within the block for a given `k` are stored together, allowing threads in the block to collectively load a contiguous region of memory.

**Summary:** On GPU (unlike CPU), the spatial axes (h, w) must be the innermost (last) dimensions so that adjacent threads in a warp access adjacent memory addresses. This means channels must be the first dimension.

**Key concept tested:** CUDA memory coalescing; understanding that warp-level parallel access patterns (not sequential loop iteration) determine GPU cache efficiency; contrast with CPU cache-line logic.

---

## Question 13 — GPU Transfer Speed from nsys Profiler (MCQ)

The nsys profiler output for the GPU implementation is:

```
** CUDA GPU Kernel Summary (gpukernsum):

Time (%)  Total Time (sec)  Instances  Avg (sec) ... Name
--------  ----------------  ---------  --------- ... -------------------------
   100.0            0.5000          1     0.5000 ... cudapy::__main__::conv_chan...

** GPU MemOps Summary (by Time) (gpumemtimesum):

Time (%)  Total Time (sec)  Count  Avg (sec)  Med (sec) ...  Operation
--------  ----------------  -----  ---------  --------- ...  ------------------
    83.3            2.5000      2     1.2500     1.2500 ...  [CUDA memcpy HtoD]
    16.7            0.5000      1     0.5000     0.5000 ...  [CUDA memcpy DtoH]

** GPU MemOps Summary (by Size) (gpumemsizesum):

Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB) ...  Operation
----------  -----  --------  --------  --------  -------- ...  ------------------
 25000.000      2  12500.000 12500.000     0.000 25000.000 ...  [CUDA memcpy HtoD]
  1000.000      1  1000.000  1000.000  1000.000  1000.000 ...  [CUDA memcpy DtoH]
```

**Given the above profiler output, what is the estimated transfer speed from CPU memory to GPU memory?**

**Options:**
- A) Around 2 GB/s
- B) Around 10 GB/s
- C) Around 12.5 GB/s

**Correct Answer: B) Around 10 GB/s**

**Why B is correct:** The HtoD (Host to Device, i.e., CPU to GPU) transfer statistics are:
- Total size transferred: 25,000 MB = 25 GB (but note: Count = 2, with one transfer being 0 MB — so the actual data transfer is ~25 GB total, or one meaningful transfer of ~25 GB)
- Total time: 2.5 seconds

Transfer speed = Total size / Total time = 25,000 MB / 2.5 s = 10,000 MB/s = approximately 10 GB/s.

**Why the others are wrong:**
- A) Around 2 GB/s — This would imply 25 GB transferred in ~12.5 seconds, but the profiler shows it took 2.5 seconds, not 12.5.
- C) Around 12.5 GB/s — This is the average size per transfer (12,500 MB per transfer), not the transfer speed. Confusing data size with data rate.

---

## Question 14 — GPU vs. CPU Speed Comparison (Open-ended)

An engineer informs you that the current pipeline using the parallel CPU version `conv_channels` can compute the results for the same inputs in 7 seconds.

**Keeping the rest of the pipeline the same, how much faster would it be to use the CUDA version `conv_channels_kernel` compared to the parallel CPU version `conv_channels`?**

**Full Solution:**

From the profiler output, the total time for the GPU pipeline (for the same computation) consists of:

| Component | Time (s) |
|---|---|
| HtoD memory transfer (CPU to GPU) | 2.5 |
| GPU kernel execution (`conv_channels_kernel`) | 0.5 |
| DtoH memory transfer (GPU to CPU) | 0.5 |
| **Total GPU pipeline time** | **3.5** |

The CPU version takes 7 seconds.

Speed-up of GPU version over CPU version:

```
Speed-up = CPU time / GPU total time = 7.0 / 3.5 = 2x
```

**The GPU version is 2x faster than the parallel CPU version.**

**Important note:** The GPU kernel itself (0.5 s) is 14x faster than the CPU computation (7 s), but when the memory transfer overhead is included, the actual system-level speed-up drops to only 2x. This illustrates that memory transfer bottlenecks are a critical consideration in GPU computing — the PCIe bandwidth between CPU and GPU can dominate the total time for data-intensive workloads.

**Key concept tested:** Including memory transfer time in GPU performance analysis; understanding that "kernel time" alone does not represent the real-world speed-up; Amdahl's Law applied to GPU pipelines.

---

## Question 15 — Recoding DataFrame Columns to Reduce Memory (Open-ended)

The business analytics department has a large DataFrame of factory productivity data. Summary:

| col. name | dtype  | #unique  | min        | max        | example    | size    |
|-----------|--------|----------|------------|------------|------------|---------|
| date      | object | 70 079   | 1829-09-01 | 2024-05-28 | 2024-05-28 | 7.98 GB |
| location  | object | 8        | N/A        | N/A        | Lyngby     | 7.64 GB |
| mach_id   | int64  | 5 731    | -1         | 5 730      | 1 234      | 1.07 GB |
| units     | int64  | 43 923   | 932        | 68 837     | 1 234      | 1.07 GB |

**Briefly explain how and why/why not you would recode each column of the data frame in order to reduce the memory footprint.**

**Full Solution:**

**`date` column (object, 7.98 GB):**
Recode it. It is currently stored as Python string objects (`object` dtype), which have very high memory overhead (each string is a separate heap-allocated Python object). It should be recoded to a `datetime64` data type, which stores dates as a compact 64-bit integer. This eliminates the string object overhead. Alternatively, encoding as an integer (e.g., number of days since the minimum date, using `int32`) would also be acceptable and very compact. The saving would be dramatic — from ~8 GB to roughly 0.5 GB or less.

**`location` column (object, 7.64 GB):**
Recode it. It is stored as string objects but has only **8 unique values**. This is an ideal candidate for a **categorical** dtype in pandas. A categorical column stores only 8 unique strings once, and then uses a small integer code per row to look up the string. With 8 categories, a `uint8` index (0–255) is sufficient — 1 byte per row instead of a large Python string object. This would reduce this column from ~7.64 GB to roughly 0.13 GB (the size of a `uint8` array of the same number of rows as `mach_id`'s ~1.07 GB / 8 bytes * 1 byte). Encoding directly as a `uint8` integer is also acceptable.

**`mach_id` column (int64, 1.07 GB):**
Recode it. The values range from -1 to 5730, which fits within an `int16` range (int16 can hold -32 768 to 32 767). Recoding from `int64` (8 bytes per value) to `int16` (2 bytes per value) reduces memory by 4x, from ~1.07 GB to ~0.27 GB.

**`units` column (int64, 1.07 GB):**
Recode it. The values range from 932 to 68,837. This does not fit in `int16` (max 32,767) but fits in `int32` (max ~2.1 billion) or `uint32` (max ~4.3 billion). Recoding from `int64` (8 bytes) to `int32` (4 bytes) reduces memory by 2x, from ~1.07 GB to ~0.53 GB. Note: `uint32` would also work since the minimum value is 932 (positive).

**Key concept tested:** Identifying wasteful data types (`object` for strings/dates, oversized integer types) and selecting the smallest appropriate dtype based on the range of values and cardinality of the data.

---

## Question 16 — Speeding Up Date-Based Row Extraction (Open-ended)

The most common operation on the DataFrame is to extract the rows corresponding to a certain day and then compute some summary statistics. They typically conduct many such operations every time the DataFrame is used.

**Briefly explain how they may speed up such operations.**

**Full Solution:**

They should **set the `date` column as the DataFrame index and sort it**.

In pandas, setting a column as the index (using `df.set_index('date')`) and then sorting the index (using `df.sort_index()`) enables very efficient row selection by date. Specifically:

- **Sorted index lookups** use binary search (O(log n)) rather than a full linear scan (O(n)) through all rows.
- `df.loc['2024-05-28']` on a sorted DatetimeIndex is extremely fast.
- Since they perform many such operations every time the DataFrame is loaded, the one-time overhead of building and sorting the index is amortized over many queries and is well worth it.

Without a sorted index, every date-based query requires pandas to scan all rows to find matching dates — O(n) per query. With a sorted index, each query is O(log n), which is a massive improvement for large DataFrames.

**Key concept tested:** Pandas indexing for query performance; understanding the trade-off between one-time index-building cost and many-times query speed-up.

---

## Question 17 — Maximum Rows per Chunk for Power Measurement DataFrame (MCQ)

The business analytics department also works with power measurement DataFrames. A summary of a typical DataFrame (already memory-optimized):

| col. name | dtype   | #unique   | min           | max           | size    |
|-----------|---------|-----------|---------------|---------------|---------|
| sensor_id | uint32  | 11 073    | 8 031         | 124 282       | 1.91 GB |
| timestamp | uint64  | 2 505 600 | 1 706 742 000 | 1 716 847 200 | 3.83 GB |
| power     | float64 | 1 825 832 | 0.000         | 48.217        | 3.83 GB |

The hardware available for processing has only 200 MB of available memory for holding the data.

**What is the maximum amount of rows that can fit in each data frame chunk?**

**Options:**
- A) Around 10 000
- B) Around 10 000 000
- C) Around 10 000 000 000

**Correct Answer: B) Around 10 000 000**

**Why B is correct:** First calculate the bytes per row. Each row contains:
- `sensor_id`: uint32 = 4 bytes
- `timestamp`: uint64 = 8 bytes
- `power`: float64 = 8 bytes
- **Total per row: 20 bytes**

Maximum rows = Available memory / bytes per row:
```
200 MB = 200 * 1024 * 1024 bytes = 209,715,200 bytes

Rows = 209,715,200 / 20 = 10,485,760 ≈ 10,000,000
```

So approximately 10 million rows fit in 200 MB.

We can verify by using the total DataFrame size. Total size = 1.91 + 3.83 + 3.83 = 9.57 GB. From the timestamp column (uint64, 3.83 GB), the number of rows = 3.83 GB / 8 bytes = 3.83 * 1024^3 / 8 ≈ 514 million rows. Chunk fraction = 200 MB / 9,570 MB ≈ 2%, so chunk size ≈ 514M * 0.02 ≈ 10M rows.

**Why the others are wrong:**
- A) Around 10 000 — This would imply each row uses 200 MB / 10,000 = 20 KB per row, which is absurd for three numeric fields totalling 20 bytes.
- C) Around 10 000 000 000 (10 billion) — This would require 200 billion bytes (200 GB) of RAM, far exceeding the 200 MB limit.

---

## Question 18 — Job Dependency for Post-Processing Job (Open-ended)

The business analytics team uses a job array to process data from several months. The job array script:

```bash
#!/bin/bash
#BSUB -J power[1-12]
#BSUB -q hpc
#BSUB -W 02:30
#BSUB -R "rusage[mem=250MB]"
#BSUB -n 1
#BSUB -o power_%I_%J.out
#BSUB -e power_%I_%J.err

python poweranalysis.py $LSB_JOBINDEX data/power_measurements
```

They have another job that needs to run once all jobs in the above job array have finished processing. It does not matter if each individual job finished successfully or not — as long as it finished.

**What must they add to their job script in order to achieve this?**

**Full Solution:**

In the dependent job's script, they must add the following BSUB directive:

```bash
#BSUB -w "ended(power)"
```

**Explanation:**

- `#BSUB -w` specifies a job dependency condition — the job will wait in the queue until the condition is met before it is allowed to start.
- `ended(power)` means: wait until all jobs with the job name `power` (i.e., the entire `power[1-12]` job array) have reached an `ENDED` state.
- In LSF, a job has `ENDED` state when it has finished regardless of whether it exited successfully (`DONE`) or with an error (`EXIT`). This is exactly the requirement — it should run once all array jobs have finished, regardless of success or failure.
- If the condition were `done(power)`, the dependent job would only start if all array jobs finished *successfully*, which is not what they want.

**Key concept tested:** LSF job dependencies using `#BSUB -w`; the distinction between `ended()` (finished regardless of outcome) and `done()` (finished successfully only).

---

## Question 19 — Parallelizing a Dynamical System Simulation (Open-ended)

The numerical simulations department is simulating a dynamical system given a range of initial values:

```python
@jit(nopython=True, nogil=True)  # nogil=True --> Release GIL in this function
def simulate_single(x0, n, step):
    x = x0
    for i in range(n):
        x = x + step * np.cos(x) / (np.sin(5*x) + 2.5)
    return x


def simulate(n, x0s, step):
    m = len(x0s)
    out = []
    for j in range(m):
        r = simulate_single(x0s[j], n, step)
        out.append(r)
    return out
```

`simulate_single` performs `n` steps of simulation from a single initial value `x0`. `simulate` calls `simulate_single` with `m` different initial values.

**How and where would you apply parallelization in the above code? To what extent? What is your expected speed-up? How does your advice depend on `m` and `n`? You may ignore overhead in your analysis.**

**Full Solution:**

**Where to apply parallelization:** Parallelization should be applied to the outer `for j in range(m)` loop inside `simulate`, not inside `simulate_single`.

**Why not inside `simulate_single`:** The inner loop `for i in range(n)` in `simulate_single` has a **sequential dependency** — each iteration computes `x` from the previous value of `x`. There is no way to parallelize this loop because step `i+1` depends on the result of step `i`. This is a strict serial dependency chain.

**Why the outer loop in `simulate` is parallelizable:** Each call `simulate_single(x0s[j], n, step)` is completely independent of all other calls. The result for initial value `x0s[j]` does not depend on the result for any other `x0s[k]`. These m independent simulations are embarrassingly parallel.

**How to apply it — multi-threading:** The key insight is that `simulate_single` is decorated with `@jit(nopython=True, nogil=True)`. The `nogil=True` flag means that when this Numba-compiled function runs, it **releases the Python GIL**. This makes it safe and effective to use **multi-threading** (rather than multi-processing) — Python threads calling this function will run truly in parallel because the GIL is released for the duration of the Numba computation.

Using Python's `concurrent.futures.ThreadPoolExecutor` or `threading` module with up to `m` threads:

```python
from concurrent.futures import ThreadPoolExecutor

def simulate_parallel(n, x0s, step, num_threads):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(simulate_single, x0, n, step) for x0 in x0s]
        return [f.result() for f in futures]
```

**Expected speed-up:** With `t` threads (ignoring overhead), the speed-up is approximately `min(t, m)`. With `m` threads, all simulations run in parallel and the speed-up is `m` (we finish `m` times faster). In practice, speed-up is limited by the number of physical CPU cores.

**Dependence on `m` and `n`:**
- **`m` (number of initial values):** Larger `m` means more independent tasks, so parallelization is more beneficial. If `m` is small (e.g., m=1), there is nothing to parallelize. The maximum useful number of threads is `m`.
- **`n` (number of simulation steps per initial value):** `n` does not affect *whether* we can parallelize or *how* we apply it. It affects the granularity of each task — larger `n` means each task takes longer, which reduces the relative overhead of thread creation and coordination. For very small `n`, overhead might matter (though we are told to ignore it). The speed-up formula `min(t, m)` does not depend on `n`.

**Key concept tested:** Identifying the parallelizable level of a nested computation; understanding serial dependencies that prevent inner-loop parallelism; knowing that `nogil=True` in Numba enables effective Python multi-threading for CPU-bound Numba code; distinguishing multi-threading from multi-processing and knowing when each is appropriate.

---

*End of solutions document. Questions 1–19 covered in full.*
