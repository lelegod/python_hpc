# 02613 Python HPC — Exam 2024 Full Solutions

> **Exam files:** [2024 Exam](exam_2024_solutions.md) · [2024 Re-exam](reexam_2024_solutions.md) · [F25 Exam](exam_F25_solutions.md) · · **Root:** [STUDY_GUIDE](../STUDY_GUIDE.md) · [Exam Review](../exam_review.md)

## Contents

- [Question 1 — Fixing a BSUB Job Script](#question-1-fixing-a-bsub-job-script)
- [Question 2 — Parallel Fraction from a Speed-Up Plot (MCQ)](#question-2-parallel-fraction-from-a-speed-up-plot-mcq)
- [Question 3 — Should They Pursue Parallelization? (Open-ended)](#question-3-should-they-pursue-parallelization-open-ended)
- [Question 4 — Parallelization Approach for `process_number` (MCQ)](#question-4-parallelization-approach-for-process_number-mcq)
- [Question 5 — Static vs. Dynamic Scheduling for `process_number` (Open-ended)](#question-5-static-vs-dynamic-scheduling-for-process_number-open-ended)
- [Question 6 — Why `abssum` Cannot Be Used in a Parallel Reduction (Open-ended)](#question-6-why-abssum-cannot-be-used-in-a-parallel-reduction-open-ended)
- [Question 7 — NumPy Broadcasting to Subtract a Mean Image (MCQ)](#question-7-numpy-broadcasting-to-subtract-a-mean-image-mcq)
- [Question 8 — Re-ordering Loops for Cache Efficiency (Open-ended)](#question-8-re-ordering-loops-for-cache-efficiency-open-ended)
- [Question 9 — Reading a Profiler Output: Number of Samples (MCQ)](#question-9-reading-a-profiler-output-number-of-samples-mcq)
- [Question 10 — Which Function to Optimize for Normal Workloads (Open-ended)](#question-10-which-function-to-optimize-for-normal-workloads-open-ended)
- [Question 11 — Cache Efficiency: CPU Parallel Version of `conv_channels` (Open-ended)](#question-11-cache-efficiency-cpu-parallel-version-of-conv_channels-open-ended)
- [Question 12 — Cache Efficiency: CUDA Kernel Version of `conv_channels` (Open-ended)](#question-12-cache-efficiency-cuda-kernel-version-of-conv_channels-open-ended)
- [Question 13 — GPU Transfer Speed from nsys Profiler (MCQ)](#question-13-gpu-transfer-speed-from-nsys-profiler-mcq)
- [Question 14 — GPU vs. CPU Speed Comparison (Open-ended)](#question-14-gpu-vs-cpu-speed-comparison-open-ended)
- [Question 15 — Recoding DataFrame Columns to Reduce Memory (Open-ended)](#question-15-recoding-dataframe-columns-to-reduce-memory-open-ended)
- [Question 16 — Speeding Up Date-Based Row Extraction (Open-ended)](#question-16-speeding-up-date-based-row-extraction-open-ended)
- [Question 17 — Maximum Rows per Chunk for Power Measurement DataFrame (MCQ)](#question-17-maximum-rows-per-chunk-for-power-measurement-dataframe-mcq)
- [Question 18 — Job Dependency for Post-Processing Job (Open-ended)](#question-18-job-dependency-for-post-processing-job-open-ended)
- [Question 19 — Parallelizing a Dynamical System Simulation (Open-ended)](#question-19-parallelizing-a-dynamical-system-simulation-open-ended)

---
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

**Mental Model:** This is a "read the spec and translate to BSUB directives" question. The key insight is that `rusage[mem=...]` specifies memory *per core*, not total memory — this is the single most common trap. The thought process is: (1) wall-time = what the problem states directly, (2) cores = what the problem states directly, (3) memory = total / cores because `rusage` is per-slot, (4) ask "will multiple cores be on the same node?" — yes for shared-memory programs, so add `span[hosts=1]`. The trap is writing `16GB` for memory without dividing by the number of cores, which wastes cluster resources and may cause the job to fail or be queued indefinitely.

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

**Mental Model:** This is a "Amdahl's Law inversion from a saturation value" question. The key insight is that a speedup curve that flattens and saturates at a maximum value S_max directly gives you the parallel fraction through the formula F = 1 - 1/S_max. The thought process is: (1) read the saturation value from the plot (~5), (2) plug into S_max = 1/(1-F) and solve for F, giving F = 1 - 1/5 = 0.8. The trap is picking option A (0.2) by confusing the serial fraction (1-F = 0.2) with the parallel fraction (F = 0.8) — a sign-flip error.

**Correct Answer: C) 0.8**

**Why C is correct:** Amdahl's Law states that the theoretical maximum speed-up with infinite processors is:

```
S_max = 1 / (1 - F)
```

where F is the parallel fraction. The plot clearly saturates (levels off) at a speed-up of approximately 5. Setting S_max = 5:

```
5 = 1 / (1 - F)
1 - F = 1/5 = 0.2
F = 1 - 0.2 = 0.8
```

Therefore the parallel fraction is 0.8.

**Why the others are wrong:**
- A) 0.2 — This is the *serial* fraction (1 - F), not the parallel fraction. Confusing 1-F with F is the classic trap. Plugging F=0.2 back: S_max = 1/(1-0.2) = 1.25, but the plot clearly shows speedups well above 1.25.
- B) 0.5 — This would give S_max = 1/(1-0.5) = 2. The plot saturates at ~5, not 2. A student who misreads the saturation level as 2 might land here.
- D) 0.9 — This would give S_max = 1/(1-0.9) = 10. The plot saturates well below 10 (at ~5), so 0.9 is too high. A student who overestimates the saturation level, or who forgets to read the plot carefully, ends up here.

---

## Question 3 — Should They Pursue Parallelization? (Open-ended)

Management wants at least a speed-up of 4 before they consider parallelization to be worth it. Currently, the most powerful machine at the company has 8 cores.

**Given your estimated parallel fraction (F = 0.8), should they pursue parallelization? Explain your reasoning.**

**Mental Model:** This is a "apply Amdahl's Law to a real hardware constraint and threshold" question. The key insight is that you must evaluate S(p) at the *actual* hardware limit p=8, not at the theoretical infinite-core maximum. The thought process is: (1) write the full Amdahl formula S(n) = 1/((1-F) + F/n), (2) plug in F=0.8 and n=8, (3) compute the result, (4) compare to the threshold of 4. The trap is comparing the threshold to S_max=5 (infinite cores) and incorrectly concluding "yes, since 5 > 4" — but the question explicitly constrains hardware to 8 cores.

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

**Mental Model:** This is a "GIL identification" question. The key insight is to ask: is this code pure Python CPU-bound work, or does it release the GIL? The thought process is: (1) look at the code — it is a pure Python `for` loop doing integer arithmetic, (2) pure Python loops hold the GIL continuously, (3) threads cannot run this code simultaneously because the GIL serializes them, (4) therefore processes (which each have their own GIL) are required. The trap is choosing "Does not matter" by thinking parallelism is parallelism — it does matter because threads literally cannot run in parallel here.

**Correct Answer: B) Multi-processing**

**Why B is correct:** The `process_number` function is pure Python CPU-bound computation (a Python `for` loop doing arithmetic). Python has the Global Interpreter Lock (GIL), which prevents multiple threads from executing Python bytecode simultaneously. Multi-threading in Python does NOT provide parallelism for CPU-bound pure-Python code — threads take turns holding the GIL one at a time. Multi-processing creates separate processes, each with its own GIL and Python interpreter, so they can run truly in parallel on multiple CPU cores.

**Why the others are wrong:**
- A) Multi-threading — The GIL prevents true parallel execution of CPU-bound Python code. Choosing threads here means each thread must acquire the GIL before executing any bytecode instruction; only one thread runs at a time, giving no speed-up over serial code and actually adding overhead from context switching.
- C) Does not matter — A student might reason "either way creates workers so either way is fine." This is wrong because multi-threading is actively harmful here: it adds scheduling overhead while delivering zero parallel speedup due to the GIL.

---

## Question 5 — Static vs. Dynamic Scheduling for `process_number` (Open-ended)

**Should they use static scheduling or dynamic scheduling for this task? Explain your answer.**

**Mental Model:** This is a "workload balance" question. The key insight is to ask: is the execution time of each task uniform or variable? The thought process is: (1) `process_number(n)` runs exactly `n` iterations, so its time is proportional to `n`, (2) the input array goes from 1 to 100,000,000, so task times vary by a factor of 100 million, (3) static scheduling pre-assigns fixed chunks of inputs to workers — with uniform chunks by index, one worker gets the large numbers and works vastly longer than others, (4) dynamic scheduling hands out tasks on demand so no worker idles. The trap is defaulting to static because "it's simpler" without considering that extreme workload imbalance makes static scheduling catastrophically inefficient here.

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

**Mental Model:** This is an "associativity requirement for reduction trees" question. The key insight is that a reduction tree applies the combining function in different orderings depending on how the tree is structured — if the function is not associative, different tree shapes give different answers. The thought process is: (1) recall the associativity requirement: f(f(a,b),c) must equal f(a,f(b,c)), (2) try a concrete counterexample with a sign-cancelling case, (3) show the two groupings give different numbers. The trap is thinking "abs(x+y) looks like it gives the same result as abs(x) + abs(y)" or thinking the function is fine because addition itself is associative — the absolute value wrapper breaks it.

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

**Mental Model:** This is a "NumPy broadcasting alignment" question. The key insight is that broadcasting aligns shapes from the *right*, and you need to manually add axes to `mim` until it has the right rank and the dimensions align correctly. The thought process is: (1) `images` is (N, H, W, 3), (2) `mim` is (H, W) — 2D, (3) we want mim to broadcast across N and the channel axis, (4) so we need mim to become (H, W, 1) to match the last 3 dimensions of images as (H, W, 3). The trap is `mim[None]` which prepends a dimension giving (1, H, W) — this aligns the W axis against the 3-channel axis, which is wrong.

**Correct Answer: B) `images - mim[:, :, None]`**

**Why B is correct:** `images` has shape (N, H, W, 3). We want to subtract a value for each (H, W) pixel position from all N images and all 3 channels. `mim` has shape (H, W). We need to broadcast `mim` across the N and channel dimensions.

`mim[:, :, None]` reshapes `mim` from (H, W) to (H, W, 1). NumPy broadcasting then aligns shapes from the right:
```
images:          (N, H, W, 3)
mim[:,:,None]:   (   H, W, 1)   → broadcasts to (N, H, W, 3)
```
The trailing 1 expands to 3 (channels), and N is prepended automatically. This correctly subtracts the same mean pixel value from each channel at each spatial position in each image.

**Why the others are wrong:**
- A) `images - mim` — Broadcasting aligns from the right: `images` has last two dims (W, 3) and `mim` has dims (H, W). The rightmost dims are 3 vs W — unless H happens to equal 3, this raises a shape mismatch error. Even if sizes happened to match by coincidence, the semantic alignment would be wrong. A student who thinks "subtract an array" naively without checking alignment falls into this trap.
- C) `images - mim[None]` — `mim[None]` gives shape (1, H, W). Broadcasting against (N, H, W, 3): the rightmost dim of `mim[None]` is W but the rightmost dim of `images` is 3. This misaligns the channel axis with the width axis, producing wrong results or an error. A student who correctly thinks "add a leading dimension" but inserts it at the wrong end lands here.

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

**Mental Model:** This is a "stride-based loop ordering" question. The key insight is that the innermost loop should index the axis with the *smallest* stride, because smaller stride means consecutive iterations touch closer memory addresses, maximizing cache line reuse. The thought process is: (1) list the stride for each axis, (2) sort axes by stride ascending, (3) the smallest-stride axis goes innermost, the largest goes outermost. The trap — which the problem explicitly mentions — is "reversing" the loop order (giving innermost = l with stride 200) without actually sorting by stride value. Reversing is not the same as stride-sorting.

**Full Solution:**

For cache-efficient memory access, the innermost loop should iterate over the axis with the **shortest (smallest) stride**, because the shortest stride means consecutive iterations access memory locations that are closest together — maximally exploiting spatial locality and cache line reuse.

The strides are:
- Axis 0 (i, N-axis): stride 600
- Axis 1 (j, H-axis): stride 40
- Axis 2 (k, W-axis): stride 8
- Axis 3 (l, channel-axis): stride 200

Ranking from shortest to longest stride (ascending):
1. Axis 2 (k): stride 8 — shortest, should be the **innermost** loop
2. Axis 1 (j): stride 40
3. Axis 3 (l): stride 200
4. Axis 0 (i): stride 600 — longest, should be the **outermost** loop

Sorting all four by stride ascending: 8, 40, 200, 600 → axes k, j, l, i. From innermost to outermost: **k → j → l → i**.

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

**Mental Model:** This is a "read ncalls to infer loop count" question. The key insight is that a function called once per sample will have ncalls equal to the number of samples. The thought process is: (1) identify which function corresponds to per-sample processing — `process_sample` is named for it, (2) read its ncalls = 10, (3) conclude there were 10 samples. The traps are: option A (2) from misreading the `2/1` entry for `exec` (that is an interpreter artifact), and option B (5) from confusing the per-call time (0.505 s) or dividing total time (5.055) by per-call (0.505) and misidentifying 5 as the answer — but 5.055/0.505 = 10.01, confirming ncalls=10.

**Correct Answer: C) 10**

**Why C is correct:** The profiler shows that `process_sample` (ourlib.py:13) was called **10 times** (`ncalls = 10`). In the program, `process_sample` is called once per sample inside the loop `for s in samples`. Therefore, the data subset contained exactly 10 samples.

**Why the others are wrong:**
- A) 2 — The only entry with a call count near 2 is the built-in `exec` call (shown as `2/1`, meaning 2 total calls including recursive calls, 1 primitive call). This is a Python interpreter artifact for module execution, not related to sample count. A student who scans for any "2" in the ncalls column and stops there picks this incorrectly.
- B) 5 — There is no function called 5 times. A student might mistakenly divide the total cumtime (5.055 s) by the per-call time (0.505 s) and think the answer is 10, then round down, or misread 0.505 as "5 calls" due to the decimal — but the ncalls column clearly shows 10.

---

## Question 10 — Which Function to Optimize for Normal Workloads (Open-ended)

**On what function would you focus your optimization efforts with respect to normal workloads? Explain your answer.**

**Mental Model:** This is a "profiler output extrapolation to realistic workload" question. The key insight is that one-time setup functions (load_params, prepare_model) are fixed costs that don't scale, while per-sample functions scale linearly with sample count. The thought process is: (1) separate functions into "called once" vs "called per sample", (2) project the per-sample cost to the stated normal workload size (1000+ samples), (3) compare projected costs. The trap is optimizing `load_params` (20 seconds) because it looks like the biggest cost in the profiler output — but that was for only 10 samples; at 1000 samples, `process_sample` dominates by orders of magnitude.

**Full Solution:**

Focus optimization efforts on **`process_sample`**.

**Reasoning from the profiler data:**

For the 10-sample subset, `process_sample` took a total cumulative time of 5.055 seconds, with a per-call time of 0.505 seconds per sample.

The normal workload is **at least 1000 samples** at a time. If we scale `process_sample` linearly (which is safe to assume since it is called independently for each sample):

```
Expected time for 1000 samples = 1000 × 0.505 = 505 seconds
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

**Mental Model:** This is a "match the innermost loop variable to the last (shortest-stride) axis" question. The key insight is that C-order (row-major) arrays have their last axis contiguous in memory, so you want the innermost loop to index the last axis. The thought process is: (1) identify the innermost loop — it is `k` (channel index), (2) for sequential cache access, k should index the axis that is contiguous in memory, (3) in C-order, the last axis is contiguous, (4) so channel axis should be last: shape h x w x c. The trap is channels-first (c x h x w), which seems "natural" for image processing frameworks but puts the loop's variable (k) on the first axis with a very large stride.

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

**Mental Model:** This is a "GPU coalescing flips the optimal layout" question — the opposite answer from Q11. The key insight is that on GPU, performance is determined by *warp-level* simultaneous access. Adjacent threads in a warp differ by 1 in threadIdx.x (x-dim, the FIRST return value of `cuda.grid(2)`). For coalesced access of `image[k, i, j]`, the warp-varying index must map to the **last axis** (stride-1). The DTU lecture shows the fix is to use `j, i = cuda.grid(2)` (swapped) so that j=x-dim varies in the warp, then j indexes W (last axis) → coalesced. The answer (channels-first, c×h×w) is correct because with the swapped convention j indexes W. The trap is applying Q11's CPU reasoning (channels-last) to the GPU case.

**Full Solution:**

The `image` array should be stored with **channels as the first dimension**, i.e., shape **c x h x w**.

**Reasoning:**

The GPU memory access pattern is fundamentally different from the CPU case (Q11). The key considerations are:

1. **Coalesced global memory access (warp-level):** In CUDA, threads in the same warp execute the same instruction simultaneously. For global memory reads to be coalesced (i.e., served in as few memory transactions as possible), threads in a warp should access consecutive memory addresses in the same instruction. A 32x32 thread block has 32 warps; the threads in each warp differ in their `j` index (column). When all threads in a warp execute `image[k, i, j]` (channels-first layout), they access `image[k, i, j], image[k, i, j+1], ..., image[k, i, j+31]` — consecutive memory addresses along the `w` axis. This is perfectly coalesced.

2. **If channels-last (h x w x c):** All threads in a warp execute `image[i, j, k]`. For a fixed `i` and `k`, threads access `image[i, j, k], image[i, j+1, k], ..., image[i, j+31, k]`. These are separated by `c` elements in memory (stride c), not contiguous. This produces uncoalesced (strided) memory accesses, which is very inefficient on a GPU.

3. **Shared cache across threads:** Threads in the same thread block share the L1 cache/shared memory. When all threads in a block process their inner `k`-loop simultaneously, they each access a different channel slice `image[k, i, j]`. With channels-first layout, all the (i,j) values within the block for a given `k` are stored together, allowing threads in the block to collectively load a contiguous region of memory.

**Summary:** On GPU (unlike CPU), the spatial axes (h, w) must be the innermost (last) dimensions so that adjacent threads in a warp access adjacent memory addresses. This means channels must be the first dimension.

**Key concept tested:** CUDA memory coalescing; understanding that warp-level parallel access patterns (not sequential loop iteration) determine GPU cache efficiency; contrast with CPU cache-line logic.

> **⚠️ Accuracy note:** The DTU lecture (Week 9 slides 51–52) explicitly shows that `i, j = cuda.grid(2)` with tpb=(32,32) gives a **"Column-wise layout"** (i varies in warp → row varies → bad). The lecture's FIX is to use `j, i = cuda.grid(2)` with the grid also swapped: `bpg = (x.shape[1]//tpb[0], x.shape[0]//tpb[1])`. The solution above assumes the swapped (j,i) convention without stating it, which is inconsistent with the kernel as written. The correct interpretation: the exam Q12 solution is only valid if you also swap the grid setup. The principle is correct — for `image[k, i, j]` with shape (c,h,w) to be coalesced, j (the last/W index) must be the warp-varying dimension, which requires `j, i = cuda.grid(2)` not `i, j = cuda.grid(2)`. **For the exam: answer channels-first (c, h, w) and explain "the column index j indexes the W dimension (last axis, stride 1)".**

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

**Mental Model:** This is a "bandwidth = size / time" calculation from two separate tables in the profiler. The key insight is to use total size from the size table and total time from the time table — not to confuse the "average size per transfer" (12,500 MB) with the transfer speed. The thought process is: (1) find HtoD total size from the size table: 25,000 MB, (2) find HtoD total time from the time table: 2.5 s, (3) compute speed = 25,000 MB / 2.5 s = 10,000 MB/s ≈ 10 GB/s. The trap is picking option C (12.5 GB/s) by taking the average size per transfer (12,500 MB) and mistaking it for bandwidth.

**Correct Answer: B) Around 10 GB/s**

**Why B is correct:** The HtoD (Host to Device, i.e., CPU to GPU) transfer statistics are:
- Total size transferred: 25,000 MB (from the size table, HtoD row)
- Total time: 2.5 seconds (from the time table, HtoD row)

Transfer speed = Total size / Total time = 25,000 MB / 2.5 s = 10,000 MB/s = approximately 10 GB/s.

**Why the others are wrong:**
- A) Around 2 GB/s — This would imply 25 GB transferred in ~12.5 seconds. The profiler shows it took 2.5 seconds, not 12.5. A student who divides time by size (inverted formula) or misreads the time column lands here.
- C) Around 12.5 GB/s — This is the *average size per individual transfer* (12,500 MB per transfer), not the transfer speed. A student who reads the "Avg (MB)" column from the size table and labels it as bandwidth makes this mistake — confusing data size per transaction with data rate.

---

## Question 14 — GPU vs. CPU Speed Comparison (Open-ended)

An engineer informs you that the current pipeline using the parallel CPU version `conv_channels` can compute the results for the same inputs in 7 seconds.

**Keeping the rest of the pipeline the same, how much faster would it be to use the CUDA version `conv_channels_kernel` compared to the parallel CPU version `conv_channels`?**

**Mental Model:** This is a "total GPU time includes transfers, not just kernel time" question. The key insight is that you must add HtoD transfer time + kernel time + DtoH transfer time to get the real GPU pipeline time before computing speedup. The thought process is: (1) read kernel time from the profiler: 0.5 s, (2) read HtoD time: 2.5 s, (3) read DtoH time: 0.5 s, (4) total GPU time = 3.5 s, (5) speedup = 7.0 / 3.5 = 2x. The trap is computing kernel-only speedup: 7.0 / 0.5 = 14x, ignoring that the 2.5 s transfer time completely dominates the pipeline and reduces real-world speedup to just 2x.

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

**Mental Model:** This is a "choose the smallest correct dtype for each column" question. The key insight is to check three things for each column: (1) is it stored as `object` (which is a pointer to a heap-allocated Python object and is always wasteful)? If yes, recode. (2) For integers, does the value range fit in a smaller int type? (3) For strings with very few unique values, can it become categorical? The thought process column by column: dates as object are strings — use datetime64; location with 8 unique values is perfect for categorical; mach_id range -1 to 5730 fits in int16 (max 32767); units range 932 to 68837 does not fit in int16 but fits in int32. The trap for `location` is recoding to a general string type rather than categorical, missing the huge compression opportunity.

**Full Solution:**

**`date` column (object, 7.98 GB):**
Recode it. It is currently stored as Python string objects (`object` dtype), which have very high memory overhead (each string is a separate heap-allocated Python object). It should be recoded to a `datetime64` data type, which stores dates as a compact 64-bit integer. This eliminates the string object overhead. Alternatively, encoding as an integer (e.g., number of days since the minimum date, using `int32`) would also be acceptable and very compact. The saving would be dramatic — from ~8 GB to roughly 0.5 GB or less.

**`location` column (object, 7.64 GB):**
Recode it. It is stored as string objects but has only **8 unique values**. This is an ideal candidate for a **categorical** dtype in pandas. A categorical column stores only 8 unique strings once, and then uses a small integer code per row to look up the string. With 8 categories, a `uint8` index (0–255) is sufficient — 1 byte per row instead of a large Python string object. This would reduce this column from ~7.64 GB to roughly 0.13 GB. Encoding directly as a `uint8` integer is also acceptable. A student who suggests "convert to string but smaller" without using categorical misses the key optimization here.

**`mach_id` column (int64, 1.07 GB):**
Recode it. The values range from -1 to 5730, which fits within an `int16` range (int16 can hold -32 768 to 32 767). Recoding from `int64` (8 bytes per value) to `int16` (2 bytes per value) reduces memory by 4x, from ~1.07 GB to ~0.27 GB. The trap is choosing `uint16` (0 to 65535) — but the minimum value is -1 (negative!), so an unsigned type would not work.

**`units` column (int64, 1.07 GB):**
Recode it. The values range from 932 to 68,837. This does not fit in `int16` (max 32,767) but fits in `int32` (max ~2.1 billion) or `uint32` (max ~4.3 billion). Recoding from `int64` (8 bytes) to `int32` (4 bytes) reduces memory by 2x, from ~1.07 GB to ~0.53 GB. The trap is trying `int16` for this column without checking: 68,837 > 32,767, so int16 would overflow and corrupt the data.

**Key concept tested:** Identifying wasteful data types (`object` for strings/dates, oversized integer types) and selecting the smallest appropriate dtype based on the range of values and cardinality of the data.

---

## Question 16 — Speeding Up Date-Based Row Extraction (Open-ended)

The most common operation on the DataFrame is to extract the rows corresponding to a certain day and then compute some summary statistics. They typically conduct many such operations every time the DataFrame is used.

**Briefly explain how they may speed up such operations.**

**Mental Model:** This is a "Pandas indexing for repeated lookup" question. The key insight is that a sorted index enables binary search (O(log n)) instead of linear scan (O(n)) for every query, and the one-time cost of building/sorting the index is amortized over the many queries that follow. The thought process is: (1) the operation is "find all rows where date = X" — this is a lookup, (2) without an index, pandas scans every row each time: O(n), (3) with a sorted DatetimeIndex, pandas uses binary search: O(log n), (4) one-time index build cost is worth it because "many operations every time" the DataFrame is used. The trap is suggesting re-sorting the DataFrame or loading only the needed rows from disk — valid in some contexts, but the standard Pandas answer is: set and sort the index.

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

**Mental Model:** This is a "bytes-per-row calculation" question. The key insight is to sum the byte sizes of all dtypes per row, then divide the available memory by that per-row size. The thought process is: (1) uint32 = 4 bytes, uint64 = 8 bytes, float64 = 8 bytes, total = 20 bytes/row, (2) available = 200 MB = 200 × 10^6 bytes, (3) rows = 200×10^6 / 20 = 10×10^6 = 10 million. The trap is option A (10,000) — which implies 20 KB per row, absurd for three numeric fields — arrived at by misplacing a power of ten when converting MB to bytes.

**Correct Answer: B) Around 10 000 000**

**Why B is correct:** First calculate the bytes per row. Each row contains:
- `sensor_id`: uint32 = 4 bytes
- `timestamp`: uint64 = 8 bytes
- `power`: float64 = 8 bytes
- **Total per row: 20 bytes**

Maximum rows = Available memory / bytes per row:
```
200 MB = 200 × 1024 × 1024 bytes ≈ 209,715,200 bytes

Rows = 209,715,200 / 20 = 10,485,760 ≈ 10,000,000
```

So approximately 10 million rows fit in 200 MB.

**Why the others are wrong:**
- A) Around 10 000 — This would imply each row uses 200 MB / 10,000 = 20 KB per row, which is absurd for three numeric fields totalling 20 bytes. A student who miscounts zeros when converting MB to bytes (using 200 bytes instead of 200 million bytes) arrives here.
- C) Around 10 000 000 000 (10 billion) — This would require 200 billion bytes (200 GB) of RAM, far exceeding the 200 MB limit. A student who confuses the answer order of magnitude, or divides MB by bytes without unit conversion, lands here.

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

**Mental Model:** This is a "ended vs done job dependency" question. The key insight is that LSF has two relevant dependency conditions: `done()` which requires all jobs to exit *successfully*, and `ended()` which triggers when all jobs have *finished regardless of success or failure*. The thought process is: (1) the requirement is "finished regardless of outcome", (2) this maps exactly to `ended()`, not `done()`, (3) the syntax is `#BSUB -w "ended(power)"` referencing the job array by its name. The trap — which many students fall into — is writing `done(power)` because it seems like the natural "jobs are done" phrasing, but `done()` would block forever if any job fails.

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

**Mental Model:** This is a "identify parallelizable loop vs serial-dependency loop, then choose threading vs multiprocessing" question. The thought process has three parts: (1) can the inner loop be parallelized? No — x at step i depends on x at step i-1, strict serial chain. (2) Can the outer loop be parallelized? Yes — each simulate_single(x0s[j]) is completely independent. (3) Should we use threads or processes? Check for `nogil=True` — this Numba decorator releases the GIL during execution, so threads CAN run in parallel despite being Python threads. The trap is (a) trying to parallelize the inner loop and (b) defaulting to multiprocessing when threading with nogil=True is both sufficient and lower-overhead.

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
