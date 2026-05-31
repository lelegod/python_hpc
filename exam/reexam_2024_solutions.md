# 02613 Python HPC — Re-exam 2024 Full Solutions
**Duration:** 4 hours | **Format:** Mixed (open-ended + MCQ)

**Date:** 21.08.2024

**General notes:**
- All arrays are NumPy arrays and stored row-wise.
- `np` refers to the NumPy Python module.

---

## Context — The Scenario

You are a performance consultant sent by DTU to help a company called Strategic Networked Advanced Intelligence Logistics (SNAIL). They are doing many different kinds of data processing, all of which are currently running too slow.

---

## Question 1 — LSF Job Script Resource Requests

One of the engineers shows you the following job script:

```bash
#!/bin/bash
#BSUB -J classify
#BSUB -q hpc
#BSUB -W 02:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -n 8
#BSUG -R "span[hosts=1]"
#BSUB -o proc_%J.out
#BSUB -e proc_%J.err

python classify.py data.csv
```

What resources does this job script request? Mention maximum wall-time, memory usage and cpu cores.

**Mental Model:** This is a "read and interpret BSUB directives" question, with a hidden typo trap. The key insight is that `rusage[mem=...]` is memory *per core*, not total — so you must multiply by `n`. The thought process is: (1) wall-time is read directly from `-W`, (2) cores is read directly from `-n`, (3) memory = per-core value × n cores. The trap is (a) stating memory as "4 GB" without noting it is per core and computing the total, and (b) not noticing the typo `#BSUG` which invalidates the `span[hosts=1]` line so it has no effect.

**Full Solution:**

The job script requests the following resources:

- **Maximum wall-time:** 2 hours (`#BSUB -W 02:00`)
- **CPU cores:** 8 cores (`#BSUB -n 8`)
- **Memory:** 4 GB of memory **per core**, which is a total of **32 GB** (`#BSUB -R "rusage[mem=4GB]"` — the `rusage[mem=...]` directive specifies memory per slot/core, so 4 GB × 8 cores = 32 GB total)

Note: The `#BSUG -R "span[hosts=1]"` line on line 7 has a typo (`#BSUG` instead of `#BSUB`), so the span constraint is not actually applied — but the question asks only about wall-time, memory, and cores.

**Key concept tested:** Reading and interpreting LSF/BSUB job script directives, and understanding that `rusage[mem=...]` is memory per core (not total).

---

## Question 2 — Modifying a Job Script for GPU Execution

To improve performance, the engineers have restructured `classify.py` to run their algorithm on a GPU.

How would you change the job script above such that the program is run on a node with a GPU?

**Mental Model:** This is a "GPU job submission on DTU HPC" question. There are exactly two required changes: (1) switch to a GPU-capable queue, (2) add the `-gpu` resource directive. The thought process is: CPU jobs use `-q hpc`; GPU jobs need a queue that has GPU nodes (`gpuv100`, `gpua100`, `hpcintogpu`), plus an explicit GPU count request via `#BSUB -gpu`. The trap is only changing the queue without adding the `-gpu` directive — on DTU HPC you need both, or the scheduler won't allocate a GPU to your job even if the queue has GPU nodes.

**Full Solution:**

Two changes are needed:

1. **Change the queue** from `hpc` to a GPU-enabled queue. Valid options on DTU HPC are:
   - `gpuv100`
   - `gpua100`
   - `hpcintogpu`

   So change: `#BSUB -q hpc` to e.g. `#BSUB -q gpuv100`

2. **Add a GPU resource request** directive:
   ```bash
   #BSUB -gpu "num=1:mode=exclusive_process"
   ```
   This requests 1 GPU in exclusive process mode (ensuring the GPU is not shared with other jobs).

The modified relevant lines would look like:
```bash
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
```

**Key concept tested:** How to request GPU resources in LSF batch scripts on DTU HPC — both queue selection and the `-gpu` directive.

---

## Question 3 — Amdahl's Law: Time on 1 Processor

Another engineer is working on a different program using CPU parallel processing. The program takes 10 minutes to run on 4 processors. They have estimated the parallel fraction as F = 0.8.

How many minutes would the program take to run on 1 processor?

**Mental Model:** This is an "Amdahl's Law in reverse" question. The key insight is that you first compute the speedup S(4) from F and p=4, then use T(1) = T(4) × S(4) to back out the serial time. The thought process is: (1) write S(p) = 1/((1-F) + F/p), (2) plug in F=0.8, p=4, compute S(4) = 1/(0.2 + 0.2) = 2.5, (3) T(1) = T(4) × S(4) = 10 × 2.5 = 25 min. The trap is trying to directly compute T(1) = T(4) / F or T(1) = T(4) × (1-F) without using the full Amdahl formula — these give wrong answers.

**Full Solution:**

Using Amdahl's Law, the speedup on p processors is:

$$S(p) = \frac{1}{(1 - F) + \frac{F}{p}}$$

For p = 4 and F = 0.8:

$$S(4) = \frac{1}{(1 - 0.8) + \frac{0.8}{4}} = \frac{1}{0.2 + 0.2} = \frac{1}{0.4} = 2.5$$

The speedup is 2.5, meaning the 4-processor version is 2.5x faster than the 1-processor version.

Since the 4-processor time is 10 minutes:

$$T(1) = T(4) \times S(4) = 10 \times 2.5 = \mathbf{25 \text{ minutes}}$$

The program would take **25 minutes** on 1 processor.

**Key concept tested:** Applying Amdahl's Law in reverse — using the measured parallel time and speedup formula to back-calculate the serial time.

---

## Question 4 — Amdahl's Law: Reduced Serial Time

An engineer now claims they can reduce the run time of the non-parallelizable part by 3 minutes.

Given the reduced serial run time, what will the new run time on 4 processors be?

**Mental Model:** This is a "only the serial portion changes" question. The key insight is to split T(1) into its serial and parallel components, modify only the serial part, then re-apply the parallel execution formula T(p) = T_serial + T_parallel/p. The thought process is: (1) from Q3, T(1) = 25 min, F = 0.8, so T_serial = 0.2 × 25 = 5 min and T_parallel = 20 min, (2) reduce serial by 3: new T_serial = 2 min, (3) T_parallel unchanged = 20 min, (4) T_new(4) = 2 + 20/4 = 7 min. The trap is re-applying the Amdahl speedup formula from Q3 with the new serial time, which requires recomputing F and is more complex than directly using T = T_serial + T_parallel/p.

**Full Solution:**

From Question 3, the original total time on 1 processor is 25 minutes. With F = 0.8:
- Parallel portion: 0.8 × 25 = 20 minutes
- Serial (non-parallelizable) portion: 0.2 × 25 = 5 minutes

When running on 4 processors, the time is structured as:

$$T(4) = T_{\text{serial}} + \frac{T_{\text{parallel}}}{4} = 5 + \frac{20}{4} = 5 + 5 = 10 \text{ minutes} \checkmark$$

(This matches the given 10-minute runtime — consistency check.)

Now the serial part is reduced by 3 minutes: new T_serial = 5 - 3 = 2 minutes.

The parallel part T_parallel = 20 minutes does not change (only the serial part was optimized).

New runtime on 4 processors:

$$T_{\text{new}}(4) = (T_{\text{serial}} - 3) + \frac{T_{\text{parallel}}}{4} = (5 - 3) + \frac{20}{4} = 2 + 5 = \mathbf{7 \text{ minutes}}$$

The new run time on 4 processors is **7 minutes**.

**Key concept tested:** Understanding that only the serial fraction is affected by the optimization; directly applying the parallel execution time formula T = T_serial + T_parallel/p.

---

## Question 5 — Zarr Block Shape for Column Access

An engineer has a matrix of size 1000 × 100000 with data type `float64`. It is stored as a Zarr array. Their code computes the sum of a given set of columns:

```python
def process(fname, columns):
    x = zarr.open(fname, mode='r')
    s = 0
    for i in columns:
        s += x[:, i].sum()
    return s
```

Only considering the above application, which of the following block shapes for the Zarr array will achieve the best performance?

**Options:**
- a) 10 × 10000
- b) 100 × 1000
- c) 1000 × 100

**Mental Model:** This is a "Zarr block shape vs access pattern" question. The key insight is that Zarr reads whole blocks from disk when any element of a block is accessed, so you want one column access to touch as few blocks as possible. The thought process is: (1) the access pattern is x[:, i] — entire column of 1000 rows, (2) for each block shape, count how many blocks contain all 1000 rows for a single column: you need (1000 rows / block_rows) blocks per column, (3) 1000×100 → 1000/1000 = 1 block per column, 100×1000 → 1000/100 = 10 blocks, 10×10000 → 1000/10 = 100 blocks. The trap is choosing the wide block (10×10000) thinking "it spans more columns so it covers more data" — wide blocks mean more blocks are needed to span the full column height.

**Correct Answer: c) 1000 × 100**

**Why c) is correct:** Zarr reads data in whole blocks from disk whenever any element of a block is accessed. The code accesses entire columns: `x[:, i]` reads all 1000 rows for a single column index `i`. With block shape 1000 × 100, each block spans all 1000 rows and 100 columns. A single column access `x[:, i]` touches exactly **one block** (since the block covers the full column height of 1000 rows). This minimizes the number of block reads from disk.

**Why the others are wrong:**
- a) 10 × 10000: Each block covers only 10 rows but 10000 columns. To read an entire column of 1000 rows, you need 1000/10 = **100 block reads**. The width is irrelevant — you still need all 1000 rows, requiring 100 partial blocks. Very inefficient.
- b) 100 × 1000: Each block covers 100 rows and 1000 columns. To read an entire column of 1000 rows, you need 1000/100 = **10 block reads**. Better than (a) but still 10x worse than (c). A student who sees "100 is closer to 1000 than 10 is" but fails to compute the block reads ends up here.

---

## Question 6 — Memory Required for One Zarr Block

The matrix is 1000 × 100000 with data type `float64`. Using the best block shape from Question 5 (1000 × 100):

How much memory will it take to load a single block into memory?

**Mental Model:** This is a "array size in bytes" calculation. The thought process is exactly: elements = rows × cols = 1000 × 100 = 100,000; bytes = elements × bytes_per_element = 100,000 × 8 = 800,000 bytes = 800 KB. There is no trap here — it is a direct calculation. The only mistake would be forgetting that float64 = 8 bytes (not 4 bytes like float32).

**Full Solution:**

Block shape: 1000 × 100 = 100,000 elements.

Data type `float64` = 8 bytes per element.

Memory per block = 100,000 × 8 = **800,000 bytes = 800 KB ≈ 0.8 MB**.

**Key concept tested:** Computing memory size from array dimensions and element byte size (float64 = 8 bytes).

---

## Question 7 — Line Profiler: Number of Steps

The simulation department has run the `simulate` function through a line profiler:

```
Line #    Hits        Time  Per Hit   % Time  Line Contents
==============================================================
     3                                        @profile
     4                                        def simulate(x, y, n_steps):
     5         1       1.0      1.0       0.0    z = 0.0
     6         1       1.0      1.0       0.0    n = len(x)
     7     10001    2000.0      0.2      13.3    for i in range(n_steps):
     8     10000    5000.0      0.5      33.3        a = x[i] * x[i] + 4
     9     10000    5000.0      0.5      33.3        b = y[n - i - 1] / x[i]
    10     10000    3000.0      0.3      20.0        z = z + a / b
    11         1       0.3      0.3       0.0    return z
```

How many steps was the simulation run for, i.e., what was the value of `n_steps`?

**Mental Model:** This is a "loop body hits = number of iterations" question. The key insight is that the loop body lines are executed once per iteration, so their hit count equals n_steps. The thought process is: (1) look at lines 8, 9, 10 (inside the loop body) — each has Hits = 10000, (2) each runs once per iteration, so n_steps = 10000. The loop header (line 7) has 10001 hits because the condition `i < n_steps` is evaluated n_steps+1 times (the final check that terminates the loop). The trap is reading 10001 from the loop header instead of 10000 from the body, or misidentifying which line is the "loop counter" vs "loop body".

**Full Solution:**

The loop body lines (8, 9, 10) were each hit **10000 times**. The loop header (line 7) was hit **10001 times** — once for each iteration plus once for the final check that terminates the loop.

Therefore `n_steps = **10000**`.

**Key concept tested:** Reading line profiler output — the loop body hit count equals the number of iterations, which equals `n_steps`.

---

## Question 8 — FLOP/s Calculation from Profiler Output

The time unit is microseconds (μs) and 1 s = 10^6 μs. Assume `x` and `y` have type `float32`.

How many FLOP/s (floating point operations per second) does the above function achieve?

**Options:**
- a) 2.67 × 10^6 FLOP/s
- b) 3.33 × 10^6 FLOP/s
- c) 4.67 × 10^6 FLOP/s

**Mental Model:** This is a "count FLOPs carefully then divide by time" question. The key insight is to count only *floating-point* operations (not integer index arithmetic), then use the total time from the loop body lines only. The thought process is: (1) line 8: multiply + add = 2 FLOPs, (2) line 9: divide = 1 FLOP (the `n - i - 1` is integer arithmetic, not floating-point), (3) line 10: divide + add = 2 FLOPs, (4) total = 5 FLOPs/iteration × 10000 iterations = 50,000 FLOPs, (5) total time = (5000 + 5000 + 3000) μs = 15,000 μs = 0.015 s, (6) FLOP/s = 50,000 / 0.015 = 3.33 × 10^6. The trap is counting `n - i - 1` as a FLOP (it's integer) giving 7 FLOPs/iteration and landing on answer c.

**Correct Answer: b) 3.33 × 10^6 FLOP/s**

**Why b) is correct:**

Count the floating point operations per loop iteration (only operations inside the loop body count):

- **Line 8:** `a = x[i] * x[i] + 4` — 1 multiply + 1 add = **2 FLOPs**
- **Line 9:** `b = y[n - i - 1] / x[i]` — 1 divide = **1 FLOP** (integer arithmetic `n - i - 1` does not count; it operates on integer indices, not floating-point values)
- **Line 10:** `z = z + a / b` — 1 divide + 1 add = **2 FLOPs**

Total FLOPs per iteration = 2 + 1 + 2 = **5 FLOPs**

Total FLOPs for 10000 iterations = 5 × 10000 = **50,000 FLOPs**

Total time (from profiler, lines 8 + 9 + 10):
- Line 8: 5000 μs
- Line 9: 5000 μs
- Line 10: 3000 μs
- Total = 15,000 μs = 15 × 10^-3 s = 0.015 s

FLOP/s = 50,000 / 0.015 = (50 × 10^3) / (15 × 10^-3) = (10/3) × 10^6 = **3.33 × 10^6 FLOP/s**

**Why the others are wrong:**
- a) 2.67 × 10^6 FLOP/s: This results from undercounting FLOPs as 4 per iteration instead of 5 (e.g., missing one of the operations on line 10), giving 40,000 FLOPs / 0.015 s = 2.67 × 10^6. A student who counts divide-then-add as 1 operation instead of 2 falls here.
- c) 4.67 × 10^6 FLOP/s: This results from overcounting FLOPs as 7 per iteration (e.g., counting `n - i - 1` as floating-point operations, adding 2 extra integer subtractions), giving 70,000 / 0.015 = 4.67 × 10^6. This is the most common mistake — including integer index arithmetic in the FLOP count.

---

## Question 9 — NumPy Vectorization of simulate

Which of the following NumPy expressions will compute the same value as the `simulate` function in the above profiler output?

**Options:**
- a) `(x * x + 4) / (y[::-1] / x)`
- b) `np.sum((x * x + 4) / (y / x))`
- c) `np.sum((x * x + 4) / (y[::-1] / x))`

**Mental Model:** This is a "trace the loop, vectorize each operation, check two details" question. The two details to get right are: (1) does the code reverse `y`? Yes — `y[n - i - 1]` accesses y in reverse order as i goes from 0 to n-1, (2) is the result a scalar sum or an array? The function accumulates z with `z += a/b`, so the result is a scalar — requiring `np.sum()`. The thought process: trace each line to its vectorized form, then check both traps. The first trap is forgetting `np.sum()` (choosing a). The second trap is forgetting `[::-1]` (choosing b).

**Correct Answer: c) `np.sum((x * x + 4) / (y[::-1] / x))`**

**Why c) is correct:**

Tracing the loop logic:
- `a = x[i] * x[i] + 4` — vectorized as `x * x + 4`
- `b = y[n - i - 1] / x[i]` — `y[n - i - 1]` accesses `y` in reverse order (as i increases from 0 to n-1, the index n-i-1 decreases from n-1 to 0), so vectorized as `y[::-1] / x`
- `z = z + a / b` — accumulates sum of `a / b` over all iterations, so the total is `np.sum((x * x + 4) / (y[::-1] / x))`

**Why the others are wrong:**
- a) `(x * x + 4) / (y[::-1] / x)`: Correctly reverses y but is missing the `np.sum(...)`. This returns an array of per-iteration `a/b` values rather than their accumulated sum `z`. The `simulate` function returns a single scalar; this option returns an array.
- b) `np.sum((x * x + 4) / (y / x))`: Has the `np.sum(...)` correctly but fails to reverse `y`. The loop uses `y[n - i - 1]`, not `y[i]`. Using plain `y` computes a completely different sum because each term pairs `x[i]` with the wrong element of `y`.

---

## Question 10 — NumPy Broadcasting Shape

Two NumPy arrays `a` and `b`:
- Shape of `a`: 100 × 1 × 6 × 3
- Shape of `b`: 100 × 1 × 3

Given the operation `c = a + b`, what is the shape of the output array `c`?

**Options:**
- a) 100 × 100 × 6 × 3
- b) 100 × 1 × 6 × 3
- c) Won't broadcast.

**Mental Model:** This is a "broadcasting with left-padding" question. The key insight is that NumPy left-pads the shorter shape with 1s to match the rank of the longer shape before applying the broadcasting rules. The thought process is: (1) a is 4D, b is 3D — left-pad b to 4D: 1 × 100 × 1 × 3, (2) compare dimension by dimension: (100 vs 1) → 100, (1 vs 100) → 100, (6 vs 1) → 6, (3 vs 3) → 3, (3) output = 100 × 100 × 6 × 3. The trap is option b (100 × 1 × 6 × 3), which forgets to apply the left-padding step and treats b's first dimension (100) as aligning with a's second dimension (1), leaving it as 1 instead of broadcasting to 100.

**Correct Answer: a) 100 × 100 × 6 × 3**

**Why a) is correct:**

Apply NumPy broadcasting rules step by step:

Step 1 — Align shapes from the right:
```
a:  100,   1,  6,  3
b:       100,  1,  3
```

Step 2 — Left-pad the shorter shape with 1s to make equal length:
```
a:  100,   1,  6,  3
b:    1, 100,  1,  3
```

Step 3 — For each dimension, the output size is the maximum of the two sizes (provided one of them is 1 or they are equal):
```
dim 1: max(100, 1)   = 100
dim 2: max(1, 100)   = 100
dim 3: max(6, 1)     = 6
dim 4: max(3, 3)     = 3
```

Output shape: **100 × 100 × 6 × 3**

All dimensions are compatible (each pair is either equal or one is 1), so broadcasting succeeds.

**Why the others are wrong:**
- b) 100 × 1 × 6 × 3: Incorrect — this forgets to left-pad `b`. A student who skips the padding step and assumes b's size-100 dimension aligns with a's second dimension (size 1) and then "broadcasts to 1" makes this error. But correctly padded, b's second dimension is 100, and a's second dimension is 1, so the result is 100, not 1.
- c) Won't broadcast: Incorrect — all dimension pairs after padding are compatible (every pair is equal or one is 1). There is no conflicting pair of non-1 unequal sizes, so broadcasting succeeds cleanly.

---

## Question 11 — CUDA Thread Block Configuration for Best Performance

The CUDA kernel `average3x3` computes a 3×3 average around every pixel of an n×n image and adds it to an output array `y`:

```python
@cuda.jit
def average3x3(x, y, n):
    row, col = cuda.grid(2)
    s = 0.0
    if 1 <= row < n - 1 and 1 <= col < n - 1:
        for j in range(-1, 2):
            for i in range(-1, 2):
                s += x[row + i, col + j]
        y[row, col] += s / 9  # Add to y
```

Which of the following thread block configurations will have the **best** performance, i.e., run the **fastest**?

**Options:**
- a) 16 × 16
- b) 256 × 1
- c) 1 × 256

**Mental Model:** This is a "CUDA warp coalescing — pick the block shape that makes col vary in the warp" question. Key facts: with `row, col = cuda.grid(2)`, row=x-dim and col=y-dim. Adjacent threads (consecutive IDs) differ by 1 in threadIdx.x → row varies, NOT col. For coalesced access of `x[row, col]`, you need col to vary → col only varies when threadIdx.x is exhausted → need blockDim.x=1 → block (1, 256). The thought process: Thread ID = threadIdx.x + threadIdx.y * blockDim.x. For warp threads 0–31: with (1, 256): threadIdx.x=0 always, threadIdx.y=0..31 → col varies ✅. With (256, 1): threadIdx.x=0..31, threadIdx.y=0 → row varies ❌. The trap is choosing (256, 1) by thinking "more threads in first dimension = better" without tracing which index actually varies in the warp.

**Correct Answer: c) 1 × 256**

**Why c) is correct:**

At each iteration of the inner loops, every thread in a **warp** accesses an element of `x` at the same time. For good memory performance (coalesced access), these simultaneous accesses should be to **sequential (contiguous) memory locations**. Since arrays are stored row-wise in memory, sequential elements in a row are contiguous.

With block shape 1 × 256: threads are laid out with 1 row and 256 columns. Threads in the same warp differ in their `col` index while sharing the same `row` index. When they access `x[row + i, col + j]`, with fixed `i` and `j`, they access elements that are consecutive in the same row — which are contiguous in memory. This gives **coalesced memory access**.

**Why the others are wrong:**
- a) 16 × 16: Threads in a warp span multiple rows and columns. Memory accesses are not fully coalesced — threads in the same warp access elements from different rows, which are not adjacent in memory. Performance is worse than (c) because each warp load spans multiple rows instead of a clean contiguous stripe.
- b) 256 × 1: Threads are laid out with 256 rows and 1 column. Threads in the same warp differ in their `row` index while sharing the same `col` index. When they access `x[row + i, col + j]`, they access elements in different rows — these are separated by n elements in memory (stride = n). This is **strided access**, which is the worst possible access pattern for a GPU and results in the lowest performance. A student who picks this by thinking "more threads = better" without considering memory layout falls here.

---

## Question 12 — CUDA Block Configuration Reasoning (Open-ended)

Explain your reasoning for the answer to Question 11.

**Mental Model:** This is the same reasoning as Q11 but written out explicitly. The key insight to state clearly is: coalesced access means threads in a warp access consecutive addresses simultaneously; row-major storage makes consecutive addresses go along columns; therefore the warp must vary in the column dimension (1 × 256). The trap is giving a vague answer like "256 × 1 is bad" without specifically explaining the stride vs. contiguous memory argument.

**Full Solution:**

At each iteration of the loops, every thread in a warp (or thread block) accesses an element of `x` at the same time. For good performance, the threads should access **sequential elements** in order to be cache-efficient (coalesced memory access).

Since NumPy arrays (and CUDA device arrays) are stored **row-wise** (C-order), sequential elements are those in the same row at adjacent column indices.

With configuration 1 × 256, all threads in a warp have the same row index and consecutive column indices. When accessing `x[row + i, col + j]`, they access elements at the same row offset but consecutive column positions — these are contiguous in memory. This achieves **coalesced access**.

With 256 × 1, all threads have consecutive row indices but the same column index. Accessing `x[row + i, col + j]` means each thread reads from a different row — these memory locations are n elements apart. This is **strided access** and is highly cache-inefficient.

With 16 × 16, partial coalescing occurs but not as good as 1 × 256 since threads in a warp still span multiple rows.

**Key concept tested:** CUDA memory coalescing — threads in a warp should access contiguous memory locations to maximize memory bandwidth utilization.

---

## Question 13 — GPU Memory Transfers in sumavg (Current Implementation)

The engineers want to apply the `average3x3` kernel to several images and accumulate the results. `x_all` is a 100 × 2000 × 2000 NumPy array in host memory. Their code:

```python
def sumavg(x_all, threadsperblock, blockspergrid):
    y = np.zeros(x_all.shape[1:], dtype=x_all.dtype)
    for x in x_all:
        average3x3[blockspergrid, threadsperblock](x, y, len(x))
    return y
```

How many host-to-device (HtoD) and device-to-host (DtoH) transfers will `sumavg` perform if called with `x_all`?

**Mental Model:** This is a "Numba auto-transfer behavior" question. The key insight is that Numba transfers ALL NumPy args HtoD before the kernel AND DtoH after — regardless of whether the array is input or output. It does not know which arrays you intend to read back. Per iteration: 1 HtoD for x, 1 HtoD for y, kernel runs, 1 DtoH for x, 1 DtoH for y → 2 HtoD + 2 DtoH per iteration → 200 HtoD + 200 DtoH total. The trap is thinking Numba only brings back y (output) and not x (input) — it transfers everything both ways.

**Full Solution:**

Both `x` and `y` are NumPy arrays residing in **host memory**. When Numba's `@cuda.jit` kernel is called with host arrays, Numba automatically transfers each array to the GPU before the kernel and transfers it back after.

In each iteration of the loop:
- `x` (a 2000 × 2000 image slice) is transferred **HtoD** (1 transfer) and **DtoH** (1 transfer) — Numba transfers ALL NumPy args back after the kernel, even input-only arrays it didn't modify
- `y` (the 2000 × 2000 accumulator) is transferred **HtoD** (1 transfer) and **DtoH** (1 transfer) — sent to GPU so the kernel can add to it, then retrieved so the next iteration can send the updated version

With 100 images (m = 100):
- **HtoD transfers:** 100 (for `x`) + 100 (for `y`) = **200 HtoD transfers**
- **DtoH transfers:** 100 (for `x`) + 100 (for `y`) = **200 DtoH transfers**

Total: **200 HtoD and 200 DtoH transfers** = 400 total.

**Key concept tested:** Understanding that Numba automatically transfers host arrays to/from GPU on every kernel invocation when they are not explicitly allocated on the device.

---

## Question 14 — Optimal GPU Transfer Count for sumavg

How many transfers would be needed in an optimal implementation of `sumavg`?

**Mental Model:** This is a "keep data on the GPU between iterations" question. The key insight is: if `y` is pre-allocated *on the GPU* using `cuda.device_array`, it never needs to cross the PCIe bus between iterations — only transferred back once at the very end. The thought process is: (1) x images must still come in one at a time: 100 HtoD transfers, (2) y stays on GPU the whole loop: 0 intermediate DtoH transfers, only 1 at the end, (3) total = 100 HtoD + 1 DtoH = 101. The trap is thinking that all 100 x images could also be sent once (they can't — the question states `x_all` is in host memory and only one image is processed at a time).

**Full Solution:**

The optimal implementation avoids redundant transfers by keeping data on the GPU as long as possible:

- **`y`** can be **allocated directly on the GPU** (e.g., using `cuda.device_array`) and stays on the GPU throughout all 100 iterations. It only needs to be transferred back to host **once** after the loop completes.
  - Cost: **1 DtoH transfer**

- **Each image `x`** (a single 2000 × 2000 slice) must still be transferred to the GPU one at a time because only one image fits in GPU memory at a time. After the kernel runs on each image, the image data does not need to come back.
  - Cost: **100 HtoD transfers**

**Total optimal transfers: 100 HtoD + 1 DtoH = 101 transfers**

Compared to the current 400 transfers (200 HtoD + 200 DtoH), this is a substantial improvement.

**Key concept tested:** GPU memory management — pre-allocating output arrays on the device and minimizing round-trips by only transferring results back once at the end.

---

## Question 15 — Job Array Dependencies and EXIT State

An engineer has two job scripts:

**bjob_prepare.sh** (job array):
```bash
#!/bin/bash
#BSUB -J prepare[1-10]
#BSUB -q hpc
#BSUB -W 2:00
#BSUB -n 1
#BSUB -R "rusage[mem=512MB]"
#BSUB -o batch_output/prepare_%I_%J.out
#BSUB -e batch_output/prepare_%I_%J.err

python prepare.py ${LSB_JOBINDEX}
```

**bjob_compute.sh** (final job):
```bash
#!/bin/bash
#BSUB -J compute
#BSUB -q hpc
#BSUB -W 5:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=512MB]"
#BSUB -w done(prepare)
#BSUB -o batch_output/compute_%J.out
#BSUB -e batch_output/compute_%J.err

python compute.py
```

After a few hours, `bjobs -A` shows:
```
$ bjobs -A
JOBID     ARRAY_SPEC  OWNER    NJOBS PEND DONE RUN EXIT SSUSP USUSP PSUSP
22185215  prepare[    user123     10    0    4   5    1      0     0     0
```

When will the `compute` job start?

**Options:**
- a) As soon as possible
- b) When the remaining jobs have finished
- c) Never

**Mental Model:** This is a "done vs ended and the effect of a failed job" question. The key insight is that `done()` requires all jobs to exit successfully, and a job already in EXIT state has permanently failed — it will never change to DONE. The thought process is: (1) the dependency is `done(prepare)`, (2) `done()` requires ALL array jobs to exit with success, (3) the bjobs output shows EXIT=1 — one job has already failed, (4) a failed job cannot un-fail, so `done(prepare)` will never be satisfied, (5) the compute job will never start. The trap is choosing b ("when the remaining jobs finish"), reasoning that once all currently running jobs complete, the condition will be met — forgetting that the already-failed job permanently prevents `done()` from being satisfied.

**Correct Answer: c) Never**

**Why c) is correct:** The `compute` job has the dependency condition `#BSUB -w done(prepare)`. The `done()` condition is only satisfied when **all jobs** in the `prepare` array finish with **success** (EXIT code 0). The `bjobs -A` output shows 1 job in the `EXIT` state — meaning one job has already exited with a **failure**. Since a failed job will never re-run and the `done()` condition requires all jobs to complete successfully, this condition can **never** be fulfilled. The `compute` job will never start.

**Why the others are wrong:**
- a) As soon as possible: Incorrect — there is an active dependency condition `#BSUB -w done(prepare)` preventing immediate start. Additionally, 5 jobs are still running. There is no scenario under which the compute job starts immediately.
- b) When the remaining jobs have finished: This is the most tempting wrong answer. A student who correctly identifies that "done(prepare) means all jobs finish" but forgets that a job already in EXIT state can never become DONE chooses this. When the 5 running jobs finish, we will have 4 DONE + 1 EXIT + 5 new DONE = 9 DONE + 1 EXIT. `done()` still requires all 10 to be DONE — the 1 permanent EXIT failure prevents this forever.

---

## Question 16 — Parallel Row vs Column Sum: Cache Efficiency

An engineer wants to sum all numbers in a large n×n matrix `x` using parallel processing. They have two implementations:

```python
# Parallel row sum                    # Parallel column sum
def summatrix(x, n_cores):            def summatrix(x, n_cores):
    with ThreadPool(n_cores) as p:        with ThreadPool(n_cores) as p:
        # Sum rows in parallel                # Sum columns in parallel
        rowsums = p.map(                     colsums = p.map(
            lambda a: a.sum(), x                 lambda a: a.sum(), x.T
        )                                    )
    return sum(rowsums)               return sum(colsums)
```

Assume n is large enough that a row/column cannot fit in the CPU cache.

Which of the two parallel implementations would you expect to be faster? Explain your reasoning.

**Mental Model:** This is a "row-major storage and sequential vs strided access" question. The key insight is that iterating over a row accesses contiguous memory (sequential), while iterating over a column accesses elements n positions apart (strided). The thought process is: (1) NumPy arrays are C-order (row-major), so row elements are contiguous, (2) summing a row = sequential scan = cache-friendly, (3) summing a column = stride-n access = one useful element per cache line = very cache-unfriendly, (4) row sum wins. The trap is thinking `x.T` just changes the view so performance is equivalent — but transposing changes the access pattern from strided to sequential or vice versa, and since n is large (so columns don't fit in cache), the penalty is severe.

**Full Solution:**

The **parallel row sum** implementation is expected to be faster.

**Reasoning:**

NumPy arrays are stored in **row-major (C) order**, meaning the elements of each row are contiguous in memory. When a thread sums a row (`a.sum()` where `a` is a row), it reads memory sequentially — this is **cache-friendly** access with high spatial locality. The CPU cache works efficiently because each cache line loaded contains elements that will all be used.

For the **parallel column sum**, each thread sums a column (because `x.T` transposes the matrix, so iterating over `x.T` gives columns of the original `x`). Elements of a column are **not** contiguous in memory — they are n elements apart (one full row width). This is **strided memory access**, which is cache-inefficient: each cache line loaded contains only one useful element out of n elements. Since n is large enough that a column cannot fit in cache, this leads to many cache misses and dramatically reduced memory bandwidth utilization.

Therefore, the row sum implementation benefits from sequential/coalesced memory access and will be significantly faster.

**Key concept tested:** Row-major storage order and its impact on cache efficiency — sequential access patterns are orders of magnitude faster than strided access for large arrays.

---

## Question 17 — Multi-threading vs Multi-processing for NumPy

This implementation uses multi-threading (via `ThreadPool`). The engineer worries whether it would be better to use multi-processing instead.

Is it appropriate to use multi-threading instead of multi-processing in this case? Explain why/why not.

**Mental Model:** This is a "NumPy releases the GIL" question. The key insight is that the GIL only blocks parallel Python bytecode execution — NumPy operations run in C and explicitly release the GIL. The thought process is: (1) the concern is the GIL preventing parallel CPU work, (2) but the computation here is `a.sum()` which is a NumPy C-level operation, (3) NumPy releases the GIL during its internal C computations, (4) so threads genuinely run in parallel. The trap is answering "No, use multiprocessing because of the GIL" — correct in general for pure Python CPU-bound code, but wrong here because NumPy is not pure Python.

**Full Solution:**

**Yes, it is appropriate to use multi-threading in this case.**

**Reasoning:**

The standard concern with Python multi-threading is the **Global Interpreter Lock (GIL)** — Python's GIL prevents multiple Python threads from executing Python bytecode simultaneously, which limits CPU-bound tasks when using pure Python code.

However, **NumPy releases the GIL** during its internal C-level computations. Since each thread in this implementation is running `a.sum()` — which is a NumPy operation — the GIL is released for the duration of the actual computation. This means the threads can run **truly in parallel** on multiple CPU cores.

Since each thread is only executing a NumPy function (not Python interpreter code), multi-threading achieves genuine parallelism here without the GIL bottleneck.

Multi-processing would also work but has higher overhead due to process creation and the need to copy/share memory between processes. For NumPy workloads, multi-threading is the preferred approach.

**Key concept tested:** The GIL and NumPy's GIL-releasing behavior — understanding when multi-threading is safe and effective for parallel Python programs.

---

## Question 18 — Parallel Reduction vs Parallel Row Sum Speedup

Another engineer suggests reshaping `x` to a flat array of length n^2 and performing a **parallel reduction** instead. Assume unlimited computational resources.

How much faster would a parallel reduction be compared to the approach from Question 16? Explain your reasoning and provide your answer as a function of n.

**Mental Model:** This is a "O(n) vs O(log n) parallel time complexity" question. The key insight is to separately analyze the time complexity of each approach under the assumption of unlimited processors. The thought process is: (1) parallel row sum: each row takes O(n) time to sum (sequentially within each row), then the serial final sum of n row-sums takes O(n) time — total O(n), (2) parallel reduction on n^2 elements: each step halves the problem, takes O(log(n^2)) = O(2 log n) steps, (3) speedup = O(n) / O(log n) = n/log2(n). The trap is thinking the parallel row sum takes O(1) time (forgetting that within each row, the sum is still sequential), or using log base 10 instead of log base 2 for the reduction.

**Full Solution:**

**Time for the parallel row sum approach (from Q16):**

The parallel row sum has two phases:
1. **Parallel phase:** Each of the n rows is summed in parallel. Each row has n elements, so each row sum takes time proportional to n (serial scan within a row). With unlimited cores, all rows run simultaneously. Time = n.
2. **Serial phase:** The n row sums are summed serially: `sum(rowsums)`. This takes time proportional to n.

Total time for parallel row sum = n + n = **2n**.

**Time for parallel reduction on n^2 elements:**

A parallel reduction on an array of length n^2 with unlimited processors takes O(log(n^2)) time, since at each step we halve the number of elements:

$$T_{\text{reduction}} = \log_2(n^2) = 2\log_2(n)$$

**Speedup of parallel reduction over parallel row sum:**

$$\text{Speedup} = \frac{T_{\text{row sum}}}{T_{\text{reduction}}} = \frac{2n}{2\log_2(n)} = \frac{n}{\log_2(n)}$$

The parallel reduction is **n / log2(n) times faster** than the parallel row sum approach.

For large n, this is a substantial improvement — for example, at n = 1024: speedup = 1024/10 ≈ 102x.

**Key concept tested:** Time complexity of parallel reduction (O(log n) steps with unlimited processors) vs two-phase parallel sum (O(n) dominated by serial summation phase); speedup analysis as a function of problem size.

---

*End of Re-exam 2024 Solutions*
