# 02613 Python HPC — Exam F25 Full Solutions

> **Exam files:** [2024 Exam](exam_2024_solutions.md) · [2024 Re-exam](reexam_2024_solutions.md) · [F25 Exam](exam_F25_solutions.md) · · **Root:** [STUDY_GUIDE](../STUDY_GUIDE.md) · [Exam Review](../exam_review.md)

## Contents

- [Question 1 — LSF Memory Requests (rusage per core)](#question-1-lsf-memory-requests-rusage-per-core)
- [Question 2 — float16 Precision and Rounding](#question-2-float16-precision-and-rounding)
- [Question 3 — Set Intersection as a Parallel Reduction Operator](#question-3-set-intersection-as-a-parallel-reduction-operator)
- [Question 4 — NumPy reshape and Row-Major Indexing](#question-4-numpy-reshape-and-row-major-indexing)
- [Question 5 — NumPy Broadcasting for Image Normalization](#question-5-numpy-broadcasting-for-image-normalization)
- [Question 6 — cProfile: Identifying the Bottleneck (cumtime)](#question-6-cprofile-identifying-the-bottleneck-cumtime)
- [Question 7 — Identifying Parallelizable Loops (Data Dependencies)](#question-7-identifying-parallelizable-loops-data-dependencies)
- [Question 8 — Amdahl's Law: Inferring F from Measured Speedup](#question-8-amdahls-law-inferring-f-from-measured-speedup)
- [Question 9 — The `time` Command: Wall Time vs CPU Time in Parallel Programs](#question-9-the-time-command-wall-time-vs-cpu-time-in-parallel-programs)
- [Question 10 — GPU Kernel Scheduling: Static vs Dynamic](#question-10-gpu-kernel-scheduling-static-vs-dynamic)
- [Question 11 — line_profiler: Extrapolating Runtime to Larger Dataset](#question-11-line_profiler-extrapolating-runtime-to-larger-dataset)
- [Question 12 — CUDA Memory Coalescing: Worst Case for ray_step Traversal](#question-12-cuda-memory-coalescing-worst-case-for-ray_step-traversal)
- [Question 13 — CUDA Thread Block Count for Output Image](#question-13-cuda-thread-block-count-for-output-image)
- [Question 14 — nsys Profiler: Identifying Dominant Cost (HtoD vs Kernel vs DtoH)](#question-14-nsys-profiler-identifying-dominant-cost-htod-vs-kernel-vs-dtoh)
- [Question 15 — Numba CUDA with NumPy Arrays: Memory Transfer Count](#question-15-numba-cuda-with-numpy-arrays-memory-transfer-count)
- [Question 16 — Cache Behavior: Random Access Pattern](#question-16-cache-behavior-random-access-pattern)
- [Question 17 — Pandas DataFrame Reduction: Choosing the Right Dtype](#question-17-pandas-dataframe-reduction-choosing-the-right-dtype)
- [Question 18 — Chunked Processing: Maximum Chunk Size for RAM Constraint](#question-18-chunked-processing-maximum-chunk-size-for-ram-constraint)
- [Question 19 — numpy.memmap: Actual Memory Footprint](#question-19-numpymemmap-actual-memory-footprint)
- [Question 20 — Zarr Chunk Size for Row-Sum Operation](#question-20-zarr-chunk-size-for-row-sum-operation)
- [Question 21 — N-body Gravity: Which Optimization Claim is NOT Correct](#question-21-n-body-gravity-which-optimization-claim-is-not-correct)
- [Question 22 — Simulation Parallelism: Choosing the Right Strategy](#question-22-simulation-parallelism-choosing-the-right-strategy)
- [Question 23 — GPU vs CPU: Benchmarking Pitfall with Transfer Overhead](#question-23-gpu-vs-cpu-benchmarking-pitfall-with-transfer-overhead)
- [Question 24 — Amdahl's Law and the Unknown Sequential Fraction](#question-24-amdahls-law-and-the-unknown-sequential-fraction)
- [Quick Reference: All Answers](#quick-reference-all-answers)

---
**Format:** MCQ only (24 questions, 4 options each) | **Duration:** 4 hours
> This exam is the same format as the upcoming exam. Study this carefully.

---

## Question 1 — LSF Memory Requests (rusage per core)

Consider the following job script:

```bash
#!/bin/bash
#BSUB -J simulate
#BSUB -q hpc
#BSUB -W 10:00
#BSUB -R "rusage[mem=???GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o sim_%J.out
#BSUB -e sim_%J.err

python simulate.py initconds.npy
```

For the simulate.py script to run, it will need at least 100GB of memory. To achieve this, what must we insert instead of "???" in the line `#BSUB -R "rusage[mem=???GB]"`?

**Options:**
- A) 25GB
- B) 100GB
- C) 10GB
- D) 50GB

**Mental Model:** This is a rusage memory calculation. The instant you see `rusage[mem=???]` paired with a core count (`-n 4`), your job is to divide total memory by cores — rusage is always a per-core value in LSF, not a total. The classic trap is writing the total (100 GB) directly, which causes LSF to reserve 4x that amount. Always ask: "Is this total or per core?" — it is always per core.

**Correct Answer: A)**

**Why A is correct:** The `#BSUB -n 4` directive requests 4 cores. In LSF, the `rusage[mem=X]` value specifies memory **per core**, not total. The total memory needed is 100 GB, so each core must be allocated 100 GB ÷ 4 = **25 GB per core**. LSF then reserves 4 × 25 = 100 GB in total for the job. Write 25, not 100.

**Why the others are wrong:**
- B) 100GB — This is the per-total-memory-needed value, not the per-core value. LSF would reserve 4 × 100 = 400 GB, wasting 4× the needed resources and potentially causing the job to wait much longer in queue or be rejected by the scheduler.
- C) 10GB — This gives only 4 × 10 = 40 GB total, which is less than the 100 GB the script needs. The job would start successfully but crash mid-run with an out-of-memory error.
- D) 50GB — This gives 4 × 50 = 200 GB total. The job will run (no crash), but it over-requests by 2×. Other users on the cluster cannot use the wasted 100 GB while your job holds it.

---

## Question 2 — float16 Precision and Rounding

Recall that 16-bit floating point numbers, as returned by NumPy's "finfo", have a resolution of 0.001, a minimum value of -6.55040e04 and a maximum value of 6.55040e04. Given this, what value will the following Python code print:

```python
import numpy as np
a = np.array(10000, dtype='float16')
b = np.array(1, dtype='float16')
print(a + b)
```

**Options:**
- A) 10001
- B) 10000
- C) 9999
- D) 10000.5

**Mental Model:** This is a float16 precision-at-scale problem. The key insight is that "resolution = 0.001" means the relative precision, not absolute. At magnitude M, the absolute gap between adjacent representable values is approximately M × resolution. So at M = 10000, the gap is 10000 × 0.001 = 10. Any addition smaller than the gap (here, adding 1) is simply swallowed by rounding. The trap is thinking float16's resolution of 0.001 means it can represent differences as small as 0.001 — it cannot at large magnitudes.

**Correct Answer: B)**

**Why B is correct:** float16 has a relative resolution of 0.001. At a magnitude of 10000, the smallest representable difference is approximately 10000 × 0.001 = **10**. This means consecutive float16 values near 10000 are spaced roughly 10 apart (e.g., ...9990, 10000, 10010...). Adding 1 to 10000 produces 10001 in exact arithmetic, but 10001 is not representable in float16 — the nearest representable value is 10000 itself. IEEE round-to-nearest-even rounds 10001 back to **10000**. The addition is a no-op.

**Why the others are wrong:**
- A) 10001 — To get this result, float16 would need to represent a difference of 1 at magnitude 10000. That requires a resolution of 0.0001, but float16's resolution is only 0.001 (10× coarser). The significand simply does not have enough bits.
- C) 9999 — IEEE floating point rounds to the nearest representable value, which is 10000 (distance 1) not 9999 (distance 2). Rounding down to 9999 would mean the result is further from the true value, violating round-to-nearest semantics.
- D) 10000.5 — This would require float16 to represent a value halfway between two grid points near 10000. But with a grid spacing of ~10, the representable values jump from 10000 to ~10010 — there is no 10000.5 in float16's representable set.

---

## Question 3 — Set Intersection as a Parallel Reduction Operator

Can the operation of set intersection (∩) be used in a parallel reduction framework? Recall that the intersection of two sets is a new set containing all elements present in *both* input sets. For example, {1, 2, 3, 4} ∩ {2, 3, 5, 6} = {2, 3}.

**Options:**
- A) Yes
- B) No, ∩ is *not* associative (but it *is* commutative)
- C) No, ∩ is *not* commutative (but it *is* associative)
- D) No, ∩ is *neither* associative nor commutative

**Mental Model:** Parallel reduction requires exactly two properties: commutativity (order of operands doesn't matter) and associativity (grouping of operations doesn't matter). For any set-membership-based operation, ask: "Does the result depend on which set is on the left vs right?" (commutativity) and "Does the result depend on which pair I combine first?" (associativity). For intersection, membership in the result means "in all sets" — neither ordering nor grouping changes that.

**Correct Answer: A)**

**Why A is correct:** A parallel reduction requires the operation to be both **commutative** and **associative**.

- **Commutative:** A ∩ B = B ∩ A. An element x is in A ∩ B if and only if x ∈ A and x ∈ B. Swapping the operands tests the same two conditions — order is irrelevant.
- **Associative:** (A ∩ B) ∩ C = A ∩ (B ∩ C). An element x is in the left-hand side iff x ∈ A, x ∈ B, and x ∈ C. The same is true for the right-hand side. The grouping of parentheses does not change the membership test; it still reduces to "is x in every set?"

Both properties hold, so set intersection can be used in a parallel reduction tree where any two subsets can be intersected in any order.

**Why the others are wrong:**
- B) Claiming ∩ is not associative is incorrect. As shown above, (A ∩ B) ∩ C = A ∩ (B ∩ C) always holds. The confusion may arise from thinking the intermediate result (A ∩ B) "loses" information, but the final result is the same regardless of intermediate grouping.
- C) Claiming ∩ is not commutative is incorrect. The membership condition "x ∈ A and x ∈ B" is symmetric — it makes no distinction between which set is named first. A ∩ B and B ∩ A always yield the same set.
- D) This is doubly wrong. Both properties hold. This answer would only be correct for an operation like set difference (A \ B ≠ B \ A, and (A \ B) \ C ≠ A \ (B \ C)), not intersection.

---

## Question 4 — NumPy reshape and Row-Major Indexing

Given 2D row-wise array, a:

| 1  | 5  | 43 | 51 | 32 |
|----|----|----|----|----|
| 73 | 2  | 4  | 67 | 37 |
| 9  | 3  | 54 | 8  | 22 |

What is the value of `a.reshape(-1)[8]`?

**Options:**
- A) 67
- B) 51
- C) 32
- D) 8

**Mental Model:** When you see `reshape(-1)`, immediately write out the full flat array row by row (left to right, top to bottom — that's C/row-major order). Then count from index 0. The trap is 1-based counting or forgetting that NumPy is 0-indexed. A quick way: index 8 means "the 9th element." Count across the rows: row 0 has indices 0-4, row 1 has indices 5-9. Index 8 is the 4th element of row 1 (indices 5,6,7,8 = 73,2,4,67).

**Correct Answer: A)**

**Why A is correct:** `a.reshape(-1)` flattens the array to 1D in row-major (C) order — elements are read row by row, left to right. The full flat array is:

```
Index: 0   1   2    3   4   5   6  7   8   9  10 11  12 13  14
Value: 1   5   43  51  32  73   2  4  67  37   9  3  54   8  22
```

Row 0 occupies indices 0–4, row 1 occupies indices 5–9. Index 8 = the 4th element of row 1 = **67**.

**Why the others are wrong:**
- B) 51 — 51 is at index 3 (row 0, column 3). This could come from using 1-based indexing (counting 51 as the "8th" element starting from 1) or misremembering which row starts at which index.
- C) 32 — 32 is at index 4 (row 0, last element). This is 4 positions short of the correct answer; a common error is forgetting that row 1 starts at index 5, not index 4.
- D) 8 — The value 8 is at index 13 in the flattened array (row 2, column 3). This might come from incorrectly looking at the column index (column 3 contains 51, 67, 8) and picking the third value, rather than computing the linear index.

---

## Question 5 — NumPy Broadcasting for Image Normalization

Assume you have an array of RGB images stored as an N x H x W x 3 array named "images". For each image, you have a mean RGB pixel stored as an N x 3 array "mean_pixels". You need to subtract each mean pixel from all pixels in the corresponding image in "images". Which of the following lines of Python code achieves this?

**Options:**
- A) `images - mean_pixels[:, None, None]`
- B) `images - mean_pixels[None, None]`
- C) `images - mean_pixels[None, :, None]`
- D) `images - mean_pixels`

**Mental Model:** NumPy broadcasting aligns shapes from the right. The target shape is (N, H, W, 3). You need mean_pixels (N, 3) to broadcast to that shape, which requires inserting two size-1 dimensions between N and 3. Think of it as: "I have N and 3 already correct; I need to insert placeholders for H and W." `[:, None, None]` means "keep axis 0 (N), insert axis, insert axis" → (N, 1, 1, 3). NumPy then broadcasts 1→H and 1→W automatically. Draw the shapes and align them right-to-left to verify.

**Correct Answer: A)**

**Why A is correct:** `images` has shape (N, H, W, 3). `mean_pixels` has shape (N, 3). To broadcast correctly, we need `mean_pixels` to become shape **(N, 1, 1, 3)** so that NumPy will expand the H and W dimensions to match `images`. `mean_pixels[:, None, None]` keeps the first axis (N), then inserts two new axes, yielding (N, 1, 1, 3). Broadcasting then expands: (N, 1, 1, 3) → (N, H, W, 3). Each image's mean is correctly subtracted from all H×W pixels in that image only.

**Why the others are wrong:**
- B) `mean_pixels[None, None]` — `None` at the front inserts axes before the existing axes, giving shape (1, 1, N, 3). Aligning from the right against (N, H, W, 3): axis-0=1, axis-1=1, axis-2=N, axis-3=3. The N axis is in position 2 (the W position), not position 0. This misaligns images with the wrong means and will either error (if N ≠ W) or silently produce wrong results (if N == W by coincidence).
- C) `mean_pixels[None, :, None]` — This gives shape (1, N, 1, 3). The N axis is now in position 1 (the H position). This aligns each mean with the wrong spatial axis, associating image index N with height position H rather than the image index position.
- D) `images - mean_pixels` — mean_pixels has shape (N, 3). NumPy aligns from the right: (N, 3) vs (N, H, W, 3). The trailing 3 matches. But the remaining (N,) vs (N, H, W) cannot broadcast unless N == H or N == W (and even then it would be semantically wrong). This raises a broadcast error in general.

---

## Question 6 — cProfile: Identifying the Bottleneck (cumtime)

Consider the following output of the function level profiler cProfile after running a script that renders a movie of a simulated scene:

```
1428 function calls (1427 primitive calls) in 17.101 seconds

Ordered by: cumulative time

ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   2/1    0.000    0.000   17.101   17.101 {built-in method builtins.exec}
     1    0.003    0.003   17.101   17.101 dorender.py:1(<module>)
   201    0.001    0.000    8.841    0.044 render.py:13(render_scene)
   200    0.001    0.000    4.845    0.024 render.py:8(advance_scene)
     1    0.000    0.000    2.405    2.405 render.py:18(save_to_mp4)
     1    0.000    0.000    1.005    1.005 render.py:3(load_scene)
     1    0.000    0.000    0.002    0.002 <frozen importlib._bootstr...
     1    0.000    0.000    0.002    0.002 <frozen importlib._bootstr...
     1    0.000    0.000    0.002    0.002 <frozen importlib._bootstr...
```

What function takes the most time overall?

**Options:**
- A) render_scene
- B) advance_scene
- C) save_to_mp4
- D) load_scene

**Mental Model:** cProfile gives you two time columns: `tottime` (time spent in the function itself, excluding callees) and `cumtime` (total time including all called sub-functions). "Overall time" means cumtime — use tottime only when you want to know which function is itself slow, ignoring its children. The table is already sorted by cumtime (descending), so the answer is simply the first non-boilerplate row: render_scene at 8.841s.

**Correct Answer: A)**

**Why A is correct:** The **cumtime** (cumulative time) column measures the total time spent inside a function and all functions it calls — this is the correct metric for "time overall." The output is sorted by cumtime (descending). `render_scene` has cumtime = **8.841 seconds**, the highest of all user-defined functions. It was called 201 times (once per frame for 200 rendered frames plus the initial state), accumulating 8.841 s total. This dwarfs all other functions.

**Why the others are wrong:**
- B) advance_scene — cumtime = 4.845 seconds. This is the second most expensive function but is less than 55% of render_scene's total time. advance_scene handles physics/time-stepping; render_scene handles the more expensive pixel rendering.
- C) save_to_mp4 — cumtime = 2.405 seconds. Called only once (writing the final video file). Accounts for only about 14% of total run time, much less than render_scene's 52%.
- D) load_scene — cumtime = 1.005 seconds. The cheapest user-defined function, called once at startup. Represents about 6% of total run time.

---

## Question 7 — Identifying Parallelizable Loops (Data Dependencies)

The script which generated the profiling output from the previous question is shown below:

```python
import sys
import render as R

scene_fname = sys.argv[1]
movie_fname = sys.argv[2]
scene = R.load_scene(scene_fname)

dt = 0.01  # Time step
n_steps = 200

all_scenes = [scene]
for i in range(n_steps):
    scene = R.advance_scene(scene, dt)
    all_scenes.append(scene)

all_frames = []
for scene in all_scenes:
    frame = R.render_scene(scene)
    all_frames.append(frame)

R.save_to_mp4(movie_fname, all_frames)
```

To speed-up the run time, we want to apply parallelization. Of the below options, what parts of the script are good candidates for parallelization?

**Options:**
- A) The first for-loop which calls R.advance_scene
- B) The second for-loop which calls R.render_scene
- C) Both for-loops
- D) None of the for-loops

**Mental Model:** When evaluating a loop for parallelism, ask one question: "Does iteration i+1 depend on the result of iteration i?" If yes, the loop is sequential by definition (each iteration cannot start until the previous one finishes). If no, the iterations are independent and the loop is embarrassingly parallel. In the first loop, `scene` is updated each iteration and passed into the next — a textbook sequential dependency. In the second loop, each call only reads from `all_scenes[i]` and writes to `all_frames[i]` independently.

**Correct Answer: B)**

**Why B is correct:** In the second for-loop, each call to `R.render_scene(scene)` takes an element from `all_scenes` and produces an independent frame. There is no dependency between iterations — rendering frame i does not read from or write to the result of rendering frame j. The output `all_frames[i]` depends only on `all_scenes[i]`. This is an embarrassingly parallel workload (a pure map operation), making it an ideal candidate for `multiprocessing.Pool.map` or similar.

**Why the others are wrong:**
- A) The first for-loop has an explicit **sequential data dependency**: `scene = R.advance_scene(scene, dt)` means the output of iteration i becomes the input to iteration i+1. You cannot compute step 5 without first computing steps 1–4. There is no way to parallelize this without changing the algorithm fundamentally.
- C) Both loops cannot be parallelized because the first loop has the dependency described above. Saying "both" incorrectly implies the first loop can be parallelized.
- D) Claiming neither loop can be parallelized incorrectly ignores the second loop. The second loop is a textbook embarrassingly parallel pattern — all inputs are pre-computed and available, and all outputs are independent.

---

## Question 8 — Amdahl's Law: Inferring F from Measured Speedup

After adding parallelization to the Python program, an engineer measured the speed-up using 3 cores to be 2.5. Given this, what is the expected theoretical maximum speed-up?

**Options:**
- A) Around 15x
- B) Around 12x
- C) Around 10x
- D) Around 7x

**Mental Model:** This is a two-step Amdahl's Law problem. Step 1: use the measured speedup at p=3 to solve for F (the parallel fraction). Step 2: compute the theoretical maximum (p → ∞) as 1/(1-F). The formula to solve for F from measured speedup S at p cores is: F = p(1 - 1/S) / (p - 1). Plug in numbers, then compute 1/(1-F). The trap is guessing F without solving the equation, or forgetting the maximum speedup formula.

**Correct Answer: C)**

**Why C is correct:** We use Amdahl's Law. The speedup with p processors is:

```
S(p) = 1 / (1 - F + F/p)
```

where F is the parallel fraction. We know S(3) = 2.5. Solving for F:

```
1/S(p) = 1 - F + F/p
1/2.5 = 1 - F + F/3
0.4 = 1 - F + F/3
0.4 - 1 = -F + F/3
-0.6 = F(-1 + 1/3)
-0.6 = F(-2/3)
F = 0.6 × (3/2) = 0.9
```

The theoretical maximum speedup (p → ∞) is:

```
S_max = 1 / (1 - F) = 1 / (1 - 0.9) = 1 / 0.1 = 10
```

**Why the others are wrong:**
- A) 15x — This would imply F ≈ 0.933. Verification: S(3) = 1/(0.067 + 0.933/3) = 1/(0.067 + 0.311) = 1/0.378 ≈ 2.65. This does not match the observed 2.5, so 15x is inconsistent with the data.
- B) 12x — Implies F ≈ 0.917. Verification: S(3) = 1/(0.083 + 0.917/3) = 1/(0.083 + 0.306) = 1/0.389 ≈ 2.57. Close but not 2.5; this F is slightly too high.
- D) 7x — Implies F ≈ 0.857. Verification: S(3) = 1/(0.143 + 0.857/3) = 1/(0.143 + 0.286) = 1/0.429 ≈ 2.33. This is below 2.5, meaning this F is too small. The measured 2.5 speedup implies a larger parallel fraction than 0.857.

---

## Question 9 — The `time` Command: Wall Time vs CPU Time in Parallel Programs

Consider the following output from using the "time" command to time a Python program:

```
real    0m12.03s
user    0m12.00s
sys     0m0.034s
```

The program was run single threaded. Assume the program can be perfectly parallelized. We now run it on two cores and again use the "time" command to time it. What is the expected output?

**Options:**
- A)
```
real    0m6.03s
user    0m6.00s
sys     0m0.034s
```
- B)
```
real    0m6.03s
user    0m12.00s
sys     0m0.034s
```
- C)
```
real    0m12.03s
user    0m6.00s
sys     0m0.034s
```
- D)
```
real    0m12.03s
user    0m12.00s
sys     0m0.034s
```

**Mental Model:** The three `time` fields measure completely different things. `real` = wall clock (how long you waited). `user` = total CPU time used by user-space code, summed across all cores. `sys` = total CPU time used by kernel code, summed across all cores. Parallelization on 2 cores cuts the wall clock in half (real halves). But each core still does its share of CPU work — 2 cores × 6 seconds each = 12 CPU-seconds total (user stays the same). Never confuse "wall time halves" with "CPU time halves."

**Correct Answer: B)**

**Why B is correct:** The `time` command reports three distinct values:
- **real** (wall-clock time): the actual elapsed time from start to finish. With perfect parallelization on 2 cores, the work is split evenly and each core runs for ~6 seconds. Wall time halves: **~6 seconds**.
- **user** (CPU time in user space): the **sum** of CPU time consumed across all cores. With 2 cores each running for ~6 seconds, user time = 2 × 6 = **~12 seconds** (unchanged from single-threaded).
- **sys**: kernel CPU time, also summed across cores. It remains **~0.034 seconds** (essentially unchanged; kernel overhead is minimal).

Option B correctly shows real halved while user and sys remain constant.

**Why the others are wrong:**
- A) Shows both real and user halved to ~6 seconds. User time is not halved because it is the *aggregate* CPU time across all cores. If anything, user time can slightly *increase* due to parallelization overhead (lock contention, context switching). It never halves.
- C) Shows real unchanged at 12 seconds but user halved to 6 seconds. This is backwards: parallelization reduces wall clock time (real decreases), and the same total CPU work is still done (user stays constant or slightly increases). Real does not stay at 12.
- D) Shows no change at all in any metric. This is what you would see if the parallelization code was written but the program still ran on only one core (e.g., due to the GIL in CPython threading). The premise states perfect parallelization, so real must decrease.

---

## Question 10 — GPU Kernel Scheduling: Static vs Dynamic

Consider the following kernel summaries from two different Python programs profiled with nsys:

```
** CUDA GPU Kernel Summary (gpukernsum):

 Time (%)  Total Time (ms)  ... Avg (ms)  ... StdDev (ms)  ... Name
 --------  ---------------  ... --------  ... -----------  ... ----------
  100.0         500.0000    ...  20.0000  ...    40.0000   ... kernel1...
```

```
** CUDA GPU Kernel Summary (gpukernsum):

 Time (%)  Total Time (ms)  ... Avg (ms)  ... StdDev (ms)  ... Name
 --------  ---------------  ... --------  ... -----------  ... ----------
  100.0         500.0000    ...  20.0000  ...      0.0500  ... kernel2...
```

The programs were run on a computer with 4 GPUs meaning 4 kernels can run simultaneously. In both programs, work is divided to the GPUs based on static scheduling, i.e., tasks are divided equally up front, the kernels are all launched on the GPUs and we then wait for them to finish. Based on the profiling output, should another scheduling strategy be employed?

**Options:**
- A) No, static scheduling is optimal for both cases.
- B) Yes, the program with kernel1 should consider dynamic scheduling. The other program with kernel2 should stick with static scheduling.
- C) Yes, the program with kernel2 should consider dynamic scheduling. The other program with kernel1 should stick with static scheduling.
- D) Yes, both programs should use dynamic scheduling.

**Mental Model:** The key signal for static vs dynamic scheduling is the **standard deviation** of task runtimes relative to the average. High StdDev means tasks have wildly different runtimes — static scheduling will assign equal counts of tasks to each GPU, but some GPUs will be unlucky and get all the slow tasks. Dynamic scheduling (a work queue) eliminates this imbalance. Near-zero StdDev means all tasks take the same time — static scheduling distributes work perfectly, and dynamic scheduling only adds overhead. Look at StdDev/Avg (coefficient of variation): kernel1 = 40/20 = 200% (high variance → dynamic); kernel2 = 0.05/20 = 0.25% (uniform → static fine).

**Correct Answer: B)**

**Why B is correct:** The key signal is **StdDev (standard deviation)** of kernel execution time relative to the average.

- **kernel1**: StdDev = 40.0 ms with Avg = 20.0 ms. Coefficient of variation = 200% — individual kernel invocations vary wildly in runtime (from nearly 0 ms to potentially ~100+ ms). With static scheduling and 4 GPUs, some GPUs receive a batch containing many slow-running kernels while others get mostly fast kernels. The fast GPUs finish their batch and sit idle while the overloaded GPU finishes. Dynamic scheduling assigns work on-demand from a shared queue, so whenever a GPU finishes, it immediately picks up the next task. This eliminates idle time and minimizes total wall time.
- **kernel2**: StdDev = 0.05 ms with Avg = 20.0 ms. Coefficient of variation = 0.25% — all kernel invocations take essentially the same time. Static scheduling distributes work equally upfront, and since each task takes the same time, all 4 GPUs finish at nearly the same time. No idle time occurs. Switching to dynamic scheduling would only add queue management overhead with zero benefit.

**Why the others are wrong:**
- A) Static is not optimal for kernel1 due to the enormous runtime variance. With 200% coefficient of variation, the fastest kernels may be 10× faster than the slowest, causing severe load imbalance and wasted GPU cycles.
- C) This reverses the correct assignment. kernel2 is the uniform one (static is ideal). kernel1 is the variable one (needs dynamic). Choosing dynamic for kernel2 adds overhead for no gain, and keeping static for kernel1 leaves the load imbalance problem unresolved.
- D) kernel2 genuinely does not need dynamic scheduling. Its near-zero variance means static scheduling achieves near-perfect load balance, and adding a dynamic task queue introduces synchronization overhead that could actually slow things down slightly.

---

## Question 11 — line_profiler: Extrapolating Runtime to Larger Dataset

Consider the following output from the "kernprof" line profiler, which has profiled a function called "process":

```
Timer unit: 1e-06 s

Total time: 3.27424 s
File: process.py
Function: process at line 97

Line #   Hits        Time  Per Hit   % Time  Line Contents
===========================================================
    97                                        @profile
    98                                        def process(data, params):
    99       1  2005064.0    2e+06     61.2       conds = prep_conds(params)
   100       1       4.0      4.0      0.0       result = []
   101    1001     740.0      0.7      0.0       for x in data:
   102    1000  1266748.0   1266.7     38.7           y = process_single(x, conds)
   103    1000    1685.0      1.7      0.1           result.append(y)
   104
   105       1       1.0      1.0      0.0       return result

  3.27 seconds - process.py:97 - process
```

The function processes data from the "data" argument according to a set of parameters in the "params" argument. For profiling, the "process" function was called with a small dataset. Normally, it would be called with a dataset of 10,000 entries. What is the estimated run time of "process" for a normal workload?

**Options:**
- A) Around 14.7 seconds
- B) Around 32.7 seconds
- C) Around 3.5 hours
- D) Around 12.7 seconds

**Mental Model:** This is a profiler extrapolation problem. The critical skill is separating **data-dependent** costs (scale with dataset size) from **data-independent** costs (constant regardless of size). The for-loop ran 1000 times → dataset size is 1000. For 10,000 elements, the loop runs 10× more. But `prep_conds` was called once and depends on `params`, not `data` — it stays constant. The trap (option B = 32.7s) is naively multiplying the whole runtime by 10 without separating these components.

**Correct Answer: A)**

**Why A is correct:** Step-by-step calculation:

1. The for-loop (lines 101–103) ran **1001 hits on line 101** (for-loop header runs once per element plus the final check), meaning **1000 elements** in the dataset. So the profiling run used a dataset of size 1000.
2. For a 10,000-element dataset, the loop runs **10× as many times**.
3. Loop total time (small dataset) = 740 + 1,266,748 + 1,685 = **1,269,173 µs = 1.269 seconds**.
4. Loop time for 10,000 elements = 1.269 × 10 = **12.69 seconds**.
5. `prep_conds` (line 99) takes **2,005,064 µs = 2.005 seconds**. It is called once and depends only on `params`, not `data`. Its runtime does **not** scale with dataset size — it stays at ~2.005 seconds regardless of whether we process 1,000 or 10,000 items.
6. Total estimated time = 12.69 + 2.005 ≈ **14.7 seconds**.

**Why the others are wrong:**
- B) 32.7 seconds — This results from multiplying the entire 3.27-second run by 10. This ignores the fact that `prep_conds` does not scale with data size. Multiplying it by 10 adds a phantom 20 extra seconds that will not actually occur.
- C) 3.5 hours — This is wildly too large. One path to this error: misreading the timer unit (1e-06 s = microseconds) and treating the raw numbers as seconds rather than microseconds, then scaling up. Or using a completely wrong scaling factor.
- D) 12.7 seconds — This correctly scales the loop portion (12.69 s) but forgets to add the constant `prep_conds` cost of ~2.005 seconds. It accounts for the data-dependent part but drops the data-independent part.

---

## Question 12 — CUDA Memory Coalescing: Worst Case for ray_step Traversal

Consider the following CUDA kernel function which renders a depth map of a volume "vol", i.e., each pixel in the output "dmap" is the distance to the nearest solid object in the volume. We define any point in "vol" with a value under 0.5 to be empty space and over 0.5 to be solid. The rendering is done by starting at the location of the output pixel and then stepping through the volume along the direction "ray_step" until we hit something solid.

```python
@cuda.jit
def render_depthmap(dmap, vol, u_step, v_step, ray_step, max_steps):
    j, i = cuda.grid(2)
    stepx, stepy, stepz = ray_step
    if i < dmap.shape[0] and j < dmap.shape[1]:
        # Get location of pixel in dmap
        x = i * u_step[0] + j * v_step[0]
        y = i * u_step[1] + j * v_step[1]
        z = i * u_step[2] + j * v_step[2]
        # Count steps until we hit something
        for s in range(max_steps):
            vi, vj, vk = int(x), int(y), int(z)
            if 0 <= vi < vol.shape[0] and \
               0 <= vj < vol.shape[1] and \
               0 <= vk < vol.shape[2]:
                if vol[vi, vj, vk] >= 0.5:
                    break  # We hit something!
            x += stepx
            y += stepy
            z += stepz
        dmap[i, j] = s
```

Assume we call the kernel with thread blocks of size 16 x 16. Assume also that dmap and vol are stored row-wise and that max_steps is 500. Which of the following values for u_step, v_step and ray_step arguments do we expect to have the **worst** performance with respect to memory efficiency?

**Options:**
- A) u_step: [1.0, 0.0, 0.0], v_step: [0.0, 1.0, 0.0], ray_step: [0.0, 0.0, 1.0]
- B) u_step: [0.0, 1.0, 0.0], v_step: [0.0, 0.0, 1.0], ray_step: [1.0, 0.0, 0.0]
- C) u_step: [1.0, 0.0, 0.0], v_step: [0.0, 0.0, 1.0], ray_step: [0.0, 1.0, 0.0]
- D) All inputs should have the roughly same performance.

**Mental Model:** GPU memory coalescing means threads in the same warp should access adjacent (contiguous) memory addresses in the same instruction. In a 2D thread block, adjacent threads differ by 1 in `j` (the second grid index). For `vol[vi, vj, vk]` with row-major storage, adjacent memory addresses differ by 1 in `vk` (the last/innermost axis has stride 1). So for coalesced access, adjacent-j threads must map to adjacent-vk values. Look at v_step: if v_step = [0, 1, 0], adjacent j threads differ in vj (axis 1), not vk (axis 2). Axis 1 stride = vol.shape[2], so they access memory locations far apart — worst case.

**Correct Answer: A)**

**Why A is correct:** For GPU memory coalescing, threads with adjacent IDs must access adjacent memory locations simultaneously. In a 16×16 thread block with 2D indexing `j, i = cuda.grid(2)`, threads with adjacent `j` values are consecutive within a warp.

The starting position of each thread in `vol` is determined by: x = i*u_step[0] + j*v_step[0], y = i*u_step[1] + j*v_step[1], z = i*u_step[2] + j*v_step[2]. Adjacent threads (j vs j+1) differ by v_step in (x, y, z), which maps to (vi, vj, vk). `vol` is row-major with shape (dim0, dim1, dim2), so the memory stride for each axis is: axis 0 stride = dim1×dim2, axis 1 stride = dim2, axis 2 stride = 1.

In **option A**: v_step = [0.0, 1.0, 0.0] → adjacent j-threads differ in vj (axis 1) by 1, with a memory stride of dim2 (= 500). Adjacent threads access memory locations 500 elements apart — **non-coalesced, very slow**. Every warp access requires 16 separate cache-line fetches instead of 1.

In **options B and C**: v_step = [0.0, 0.0, 1.0] → adjacent j-threads differ in vk (axis 2, stride 1). Adjacent threads access consecutive memory locations — **coalesced, fast**. The entire warp's access can be served by a single cache-line fetch.

**Why the others are wrong:**
- B) v_step = [0.0, 0.0, 1.0] gives adjacent threads adjacent vk values (stride 1). This is perfectly coalesced access. Good performance, not worst.
- C) Similarly, v_step = [0.0, 0.0, 1.0] is the same pattern as B for the memory access. Coalesced, good performance.
- D) The access pattern is entirely determined by v_step, which maps adjacent threads to different axes of `vol`. The axis 1 vs axis 2 choice is the difference between stride=500 (non-coalesced) and stride=1 (coalesced) — a 500× difference in memory bandwidth efficiency. These options are far from equivalent.

---

## Question 13 — CUDA Thread Block Count for Output Image

Consider again the same CUDA kernel function as in the previous questions which renders a depth map of a volume.

Given a volume of size 500 x 500 x 500, output image of size 200 x 200 and thread blocks of size 16 x 16, how many thread blocks are needed?

**Options:**
- A) 13 x 13
- B) 32 x 32 x 500
- C) 32 x 32
- D) 3 x 3

**Mental Model:** The grid size covers the **output** (dmap), not the input (vol). The volume size is a red herring — it is used inside the kernel during ray marching, but it has nothing to do with how many threads you launch. Each thread handles one output pixel. Formula: blocks per dimension = ceil(output_size / block_size) = ceil(200 / 16) = 13. That's it. The trap is using 500 (the volume size) and computing ceil(500/16) = 32.

**Correct Answer: A)**

**Why A is correct:** In CUDA, the grid is sized to cover all output elements — here, every thread writes exactly one pixel of the output `dmap` (200×200). The kernel reads from `vol` but the grid dimensions have nothing to do with `vol`'s size.

Calculation:
- Output image: 200 × 200 pixels
- Thread block: 16 × 16 = 256 threads per block
- Blocks needed per dimension: ceil(200 / 16) = ceil(12.5) = **13**
- Grid: **13 × 13** thread blocks

The 13×13 grid launches 13×13×16×16 = 43,264 threads, which covers all 40,000 output pixels (with 3,264 threads that execute but are guarded by the `if i < dmap.shape[0] and j < dmap.shape[1]` check).

**Why the others are wrong:**
- B) 32 × 32 × 500 — This incorrectly brings the volume's third dimension (500) into the grid shape. The kernel is 2D (one thread per 2D output pixel). There is no iteration over the third volume dimension in the grid — that is handled by the `for s in range(max_steps)` loop inside each thread. This is the most common mistake on this type of question.
- C) 32 × 32 — This correctly applies the formula ceil(size/16) but uses the wrong size: 500 instead of 200. ceil(500/16) = 32, which would be correct for a 500×500 output image, not a 200×200 one.
- D) 3 × 3 — This would come from ceil(200/64) ≈ 3 or ceil(200/80) ≈ 3. Possibly confusing the block size (16×16) with a different value, or dividing by the total block size (256) in each dimension rather than the per-dimension size (16).

---

## Question 14 — nsys Profiler: Identifying Dominant Cost (HtoD vs Kernel vs DtoH)

Consider the following output from the nsys profiler after running the "render_depthmap" kernel (some parts of the profiler output omitted):

```
** CUDA GPU Kernel Summary (gpukernsum):

 Time (%)  Total Time (ms)  Instances  Avg (ms)  ...           Name
 --------  ---------------  ---------  --------  ...  ---------------------...
  100.0          13.8403          1    13.8403   ...  cudapy::__main__::rend...

** GPU MemOps Summary (by Time) (gpumemtimesum):

 Time (%)  Total Time (ms)  Count  Avg (ms)  ...  Operation
 --------  ---------------  -----  --------  ---  -----------------
   99.4          26.7968      4    6.6992    ...  [CUDA memcpy HtoD]
    0.6           0.1608      4    0.0402    ...  [CUDA memcpy DtoH]

** GPU MemOps Summary (by Size) (gpumemsizesum):

 Total (MB)  Count  Avg (MB)  Med (MB)  ...  Operation
 ----------  -----  --------  --------  ---  -----------------
    125.000     4    31.250    0.000    ...  [CUDA memcpy HtoD]
      2.000     4     0.500    0.000    ...  [CUDA memcpy DtoH]
```

Which of the following parts takes the most time?

**Options:**
- A) Running the kernel
- B) Host to device transfers (HtoD, CPU to GPU)
- C) Device to host transfers (DtoH, GPU to CPU)
- D) All parts take roughly the same amount of time

**Mental Model:** Read the "Total Time (ms)" column from each section of the nsys output and compare. This question is testing whether you know to look across all three sections — gpukernsum (kernel), gpumemtimesum HtoD (CPU→GPU), and gpumemtimesum DtoH (GPU→CPU) — not just the kernel section. The sizes explain the times: 125 MB going to GPU (vol + dmap as input) takes more time than the 13 ms of computation; only 2 MB comes back (just dmap), so DtoH is negligible.

**Correct Answer: B)**

**Why B is correct:** Reading the "Total Time (ms)" column from each section:
- Kernel execution (gpukernsum): **13.84 ms**
- HtoD transfers — CPU to GPU (gpumemtimesum): **26.80 ms**
- DtoH transfers — GPU to CPU (gpumemtimesum): **0.16 ms**

HtoD at **26.8 ms is the dominant cost**, roughly double the kernel execution time. The size section explains why: 125 MB of data is sent to the GPU (the 500×500×500 float32 volume = 125 MB, plus the small dmap array), while only 2 MB (the 200×200 output dmap) is copied back. Transferring large input data dominates the total runtime.

**Why the others are wrong:**
- A) The kernel takes 13.84 ms — significant, but less than half of the HtoD transfer time. The bottleneck is getting data to the GPU, not the computation itself.
- C) DtoH (GPU to CPU) is only 0.16 ms, accounting for 0.6% of total memory operation time. Only 2 MB is transferred back (the output `dmap` at 200×200 = 40,000 bytes), so this is trivially fast and essentially negligible.
- D) The three components differ dramatically: HtoD is 168× more time than DtoH, and 1.9× more than the kernel. They are far from equal.

---

## Question 15 — Numba CUDA with NumPy Arrays: Memory Transfer Count

Consider the following Python program which defines a CUDA kernel and then calls it with NumPy arrays:

```python
from numba import cuda
import numpy as np

@cuda.jit
def square(y, x):
    i = cuda.grid(1)
    y[i] = x[i] * x[i]

x = np.zeros(1024**3, dtype='uint8')
y = np.empty_like(x)
square[1024*1024, 1024](y, x)
```

How many host to device (HtoD) and device to host (DtoH) memory transfers will this code perform? How many are necessary in an optimal implementation?

**Options:**
- A) Will do 2 HtoD and 2 DtoH transfers, but only 1 HtoD and 1 DtoH are necessary.
- B) Will do 1 HtoD and 1 DtoH transfers, and 1 HtoD and 1 DtoH are necessary.
- C) Will do 2 HtoD and 2 DtoH transfers, and 2 HtoD and 2 DtoH are necessary.
- D) Will do 2 HtoD and 1 DtoH transfers, and 1 HtoD and 2 DtoH are necessary.

**Mental Model:** When Numba's @cuda.jit is called with plain NumPy arrays, it conservatively copies all arguments both to the GPU (before) and back to the host (after), regardless of whether they are inputs or outputs. This is Numba's "automatic" behavior. The optimal implementation uses `cuda.to_device()` and `cuda.device_array()` to control transfers explicitly: only send `x` to GPU (HtoD), only bring `y` back (DtoH). The question is testing whether you know about Numba's implicit transfer behavior vs the explicit optimal version.

**Correct Answer: A)**

**Why A is correct:** When Numba's `@cuda.jit` kernel is called with plain **NumPy arrays**, Numba automatically handles transfers using a "copy-on-call" approach:
1. **Before the kernel runs**: Both `x` and `y` are copied from host to GPU — **2 HtoD transfers** (one for `x` = 1 GB, one for `y` = 1 GB). Numba conservatively copies all arguments even if some are write-only outputs.
2. **After the kernel runs**: Both `x` and `y` are copied from GPU back to host — **2 DtoH transfers**. Numba conservatively copies all arguments even if some are read-only inputs.

An **optimal implementation** using explicit device arrays would:
- `cuda.to_device(x)` → **1 HtoD** (send only the input)
- `cuda.device_array_like(y)` → 0 transfers (allocate directly on GPU, no need to send `y`'s empty initial state to GPU)
- Copy only `y` back: **1 DtoH** (retrieve only the output)

Result: only **1 HtoD + 1 DtoH** are necessary.

**Why the others are wrong:**
- B) Claims only 1 HtoD and 1 DtoH actually happen. This is incorrect — Numba's automatic behavior with NumPy arrays transfers all arguments in both directions. If you want only 1+1, you must use explicit `cuda.to_device()` and `cuda.device_array()` calls.
- C) Claims 2+2 transfers are both what happens and what is necessary. While 2+2 correctly describes what happens with NumPy arrays, it is not what is *necessary*. An optimal implementation only requires 1+1.
- D) Claims 2 HtoD and 1 DtoH occur, and 1 HtoD and 2 DtoH are necessary. Both parts are wrong. Numba performs 2+2, and the optimal is 1+1 (not 1+2). `y` does not need to be sent to the GPU (it is a write-only output), and `x` does not need to come back (it is a read-only input).

---

## Question 16 — Cache Behavior: Random Access Pattern

You find that one of your programs have low performance. You inspect the code, and you find that you access a large data structure in a random fashion. What is the likely cause of the low performance?

**Options:**
- A) A slower part of the memory hierarchy is used.
- B) The caches are too small.
- C) There are too few cache levels.
- D) There is not enough information to find the root cause.

**Mental Model:** Random access of a large data structure destroys cache locality. The key insight: caches work by exploiting temporal locality (reusing recently accessed data) and spatial locality (pre-fetching adjacent data). Random access provides neither — every access is to a new, unpredictable location that is almost certainly not in cache. This forces every access to go all the way down the memory hierarchy to DRAM. The cause is not "caches are too small" or "too few levels" — even infinite cache sizes wouldn't help if you never reuse data. The cause is that you're forced to use a *slower layer* (DRAM) of the hierarchy.

**Correct Answer: A)**

**Why A is correct:** With purely random access patterns over a large data structure, virtually every memory access results in a **cache miss** — the data is not in L1, L2, or L3 cache because it was never accessed recently (no temporal locality) and adjacent elements are not needed next (no spatial locality). Every miss must be served from **main memory (DRAM)**, which is roughly 10–100× slower than L3 cache and 200–1000× slower than L1 cache. The program is running slowly because the CPU is constantly waiting on DRAM instead of fast cache. This is precisely "using a slower part of the memory hierarchy."

**Why the others are wrong:**
- B) "The caches are too small" misdiagnoses the root cause. Even with caches 1000× larger, random access would still cause misses — because the problem is that the same data is never accessed twice (no temporal locality), not that the working set barely exceeds the current cache size. Larger caches cannot fix an access pattern with no locality. The root cause is the *pattern*, not the *size*.
- C) "Too few cache levels" is the same fundamental mistake as B. Adding more cache levels (L4, L5, etc.) still cannot help when no data element is reused. More cache hierarchy only helps if there is some locality to exploit at a slower tier.
- D) There is more than enough information. Random access over a large data structure is a textbook, well-understood performance anti-pattern with a definitive cause: cache-miss-dominated latency forcing use of DRAM. This is not an ambiguous situation.

---

## Question 17 — Pandas DataFrame Reduction: Choosing the Right Dtype

Consider the following summary of a pandas DataFrame:

| Col. name   | dtype  | #unique | min | max  | example           | size   |
|-------------|--------|---------|-----|------|-------------------|--------|
| filename    | object | 487 M   | N/A | N/A  | packinglist.docx  | 21.3 GB|
| version     | int64  | 43      | 0   | 42   | 1                 | 2.1 GB |
| sizediff_kb | int64  | 4366    | -12 | 4353 | 203               | 2.1 GB |

We wish to reduce the size of the DataFrame. How might we best reduce the size of the "version" column?

**Options:**
- A) Convert to a datetime data type
- B) Convert to a smaller floating point type
- C) Convert to a smaller integer type
- D) Cannot be reduced

**Mental Model:** When you see an integer column with a small range (0–42), immediately ask: "What is the smallest integer type that can hold this range?" The dtype hierarchy for integers is int64 (8B) → int32 (4B) → int16 (2B) → int8 (1B, range -128 to 127) → uint8 (1B, range 0 to 255). Since 0–42 fits in uint8 (one byte), the size reduction is 8× — from 2.1 GB to ~262 MB. Never use float for integer data (risk of rounding errors). Never use datetime for non-time data.

**Correct Answer: C)**

**Why C is correct:** The `version` column holds integer values with min = 0 and max = 42. It is currently stored as **int64 = 8 bytes per value**. We can safely downcast to a much smaller integer type:
- **uint8** (unsigned 8-bit integer): stores values 0 to 255. Since all version values are in [0, 42], this is a perfect fit. Size = **1 byte per value**.
- **int8** (signed 8-bit integer): stores values -128 to 127. Also covers [0, 42] with room to spare.

Size reduction: 8 bytes → 1 byte = **8× reduction**. The 2.1 GB column shrinks to approximately 262 MB. This is a lossless conversion — no precision is lost because int8/uint8 can exactly represent every value in [0, 42].

**Why the others are wrong:**
- A) Converting to datetime makes no semantic sense — `version` is a version number (0, 1, 2, ... 42), not a timestamp. Datetime would not reduce the column's size (datetime64 in pandas is still 8 bytes) and would make the data semantically meaningless.
- B) Converting to a floating point type introduces a semantic mismatch: `version` is integer-valued, and float representations can introduce rounding errors in downstream operations such as equality comparisons (`version == 5`), groupby, and merges. Even if float16 (2 bytes) saves space, using float for inherently integer data is a code correctness risk. Integer types are always preferred for integer data.
- D) The column can absolutely be reduced. The range [0, 42] is representable in just 6 bits; uint8 (1 byte) provides an 8× lossless compression from int64.

---

## Question 18 — Chunked Processing: Maximum Chunk Size for RAM Constraint

Consider the following summary of another pandas DataFrame:

| Col. name  | dtype  | #unique | min      | max      | example   | size   |
|------------|--------|---------|----------|----------|-----------|--------|
| fname_hash | int64  | 487 M   | 17542445 | 91284623 | 41625192  | 2.1 GB |
| version    | int64  | 612 M   | 0        | 42       | 1         | 2.1 GB |
| sizediff_kb| int64  | 4366    | -12      | 4353     | 203       | 2.1 GB |

We wish to run a processing script on this DataFrame on a computer system with only 24 MB RAM. We will perform the processing in chunks of N rows at a time to reduce memory needs. Assume no memory reductions are made on the DataFrame. Of the following alternatives, what is the maximum chunk size we may use?

**Options:**
- A) 800000
- B) 2400000
- C) 1100000
- D) 500000

**Mental Model:** This is a straightforward bytes-per-row calculation. All 3 columns are int64 (8 bytes each). Bytes per row = 3 × 8 = 24 bytes. Maximum rows = available RAM ÷ bytes_per_row = 24 MB ÷ 24 bytes/row. Working in SI: 24,000,000 ÷ 24 = 1,000,000 rows max. Among the choices, find the largest option ≤ 1,000,000. The trap (option D = 500,000) is valid but not the *maximum* — the question explicitly asks for the maximum. Option A (800,000) is the largest choice that doesn't exceed 1,000,000.

**Correct Answer: A)**

**Why A is correct:** Each column uses int64 dtype = **8 bytes per value**. There are 3 columns, so each row occupies 3 × 8 = **24 bytes**.

Maximum rows fitting in 24 MB:
- Using SI (1 MB = 1,000,000 bytes): 24,000,000 bytes ÷ 24 bytes/row = **1,000,000 rows maximum**
- Using binary (1 MB = 1,048,576 bytes): 25,165,824 bytes ÷ 24 bytes/row = **1,048,576 rows maximum**

Among the four choices, we need the largest value that is ≤ 1,000,000:
- 800,000 × 24 = 19,200,000 bytes = 19.2 MB ✓ (fits)
- 1,100,000 × 24 = 26,400,000 bytes = 26.4 MB ✗ (exceeds 24 MB)
- 2,400,000 × 24 = 57,600,000 bytes = 57.6 MB ✗ (way over)
- 500,000 × 24 = 12,000,000 bytes = 12 MB ✓ (fits, but not the maximum)

The largest valid option is **800,000**.

**Why the others are wrong:**
- B) 2,400,000 rows × 24 bytes = 57,600,000 bytes ≈ **57.6 MB**. This is 2.4× the available 24 MB. Loading this chunk would immediately exhaust all memory and likely crash the process with an OOM error.
- C) 1,100,000 rows × 24 bytes = 26,400,000 bytes ≈ **26.4 MB**. This exceeds 24 MB by 10% regardless of whether you use 1000-based or 1024-based MB definitions. Just slightly over the limit, but still over.
- D) 500,000 rows × 24 bytes = 12,000,000 bytes = **12 MB**. This fits in 24 MB and is a valid chunk size, but the question asks for the **maximum** chunk size. Using 500,000 when 800,000 is allowed would process fewer rows per chunk and require more iterations than necessary.

---

## Question 19 — numpy.memmap: Actual Memory Footprint

Consider the below Python program:

```python
import numpy as np

x = np.memmap('bigarray_u8.raw', mode='r',
              dtype='uint8', shape=10_000_000_000)
y = np.array(x[::100_000])  # Copy data to y
print(y.mean())
```

Disregarding the Python interpreter, what is the maximum memory requirement of the above Python program?

**Options:**
- A) Around 10 GB
- B) Around 100 MB
- C) Around 10 KB
- D) Around 100 KB

**Mental Model:** `np.memmap` does NOT load the file into RAM — it creates a virtual mapping. The OS only loads pages when you actually access them. The memory footprint is determined by what you materialize into a real array. `x[::100_000]` takes every 100,000th element from a 10-billion element array: 10,000,000,000 ÷ 100,000 = 100,000 elements. Each uint8 = 1 byte. So `y` = 100,000 bytes = 100 KB. The trap (option A = 10 GB) assumes the entire memmap is loaded. The trap (option B = 100 MB) is off by 1000× from the correct answer.

**Correct Answer: D)**

**Why D is correct:** `np.memmap` creates a **memory-mapped view** of the file. The 10 billion bytes of the file are NOT loaded into RAM — they are mapped into virtual address space, and the OS only fetches pages on demand (lazy loading). The `np.memmap` object itself uses negligible memory.

`x[::100_000]` is a stride-100,000 slice of the 10,000,000,000-element array:
- Number of elements = 10,000,000,000 ÷ 100,000 = **100,000 elements**
- Each element is uint8 = **1 byte**
- Total bytes = 100,000 × 1 = **100,000 bytes = 100 KB**

The `np.array(...)` call materializes these 100,000 elements into a regular in-memory NumPy array `y`. This is the only data truly resident in RAM. The maximum memory requirement is therefore approximately **100 KB**.

**Why the others are wrong:**
- A) 10 GB — This would be the memory usage if the entire 10-billion-element file were loaded into RAM. But `np.memmap` explicitly avoids this; only the 100,000 elements accessed by the strided slice are actually loaded by the OS.
- B) 100 MB — This is 1000× too large. 100 MB = 100,000,000 bytes, which would correspond to 100,000,000 elements of uint8 or 100,000 elements of float64 (8 bytes × 100,000). The actual array `y` has 100,000 uint8 elements = 100,000 bytes = 100 KB, not 100 MB.
- C) 10 KB — This is 10× too small. 10 KB = 10,000 bytes = 10,000 uint8 elements. The stride calculation gives 10,000,000,000 ÷ 100,000 = 100,000 elements, not 10,000. Off by a factor of 10 (perhaps computing 10B/1,000,000 instead of 10B/100,000).

---

## Question 20 — Zarr Chunk Size for Row-Sum Operation

Consider the following Python function:

```python
def process(a):
    s = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        s[i] = np.sum(a[i])
    return s
```

Assume we call the "process" function with a Zarr 1024 x 1024 array of data type float64. What chunk size will result in the best performance?

**Options:**
- A) (1, 1024)
- B) (1024, 1)
- C) (32, 32)
- D) Chunk size will not affect performance

**Mental Model:** Zarr stores data in chunks and must load a full chunk before you can read any data from it. Each `a[i]` access loads a complete row. The question is: how many chunks must be loaded to satisfy one `a[i]` row access? Choose the chunk shape that makes one row = exactly one chunk. A (1, 1024) chunk is a single complete row — one chunk load per `a[i]`. A (1024, 1) chunk is a single column — accessing one row requires loading 1024 separate column chunks. Align chunk boundaries with your access pattern.

**Correct Answer: A)**

**Why A is correct:** The function iterates over rows (`a[i]`) and computes the sum of each row. Each access `a[i]` requests all 1024 columns of row i as a 1D slice.

Zarr stores data in chunks and must load entire chunks to serve any read. The goal is to minimize chunk loads per `a[i]` access:

| Chunk size | Chunks per row access | Total chunk loads for 1024 rows |
|---|---|---|
| **(1, 1024)** | Each chunk = one full row. `a[i]` = **1 chunk load**. | 1024 × 1 = **1,024 loads** |
| **(1024, 1)** | Each chunk = one column. `a[i]` spans 1024 columns = **1024 chunk loads**. | 1024 × 1024 = **1,048,576 loads** |
| **(32, 32)** | Each row spans 1024/32 = 32 column chunks. **32 chunk loads per row**. | 1024 × 32 = **32,768 loads** |

Chunk (1, 1024) requires 1000× fewer chunk loads than (1024, 1) and 32× fewer than (32, 32). Since each chunk load carries decompression and I/O overhead, minimizing chunk loads directly minimizes runtime.

**Why the others are wrong:**
- B) (1024, 1) is the worst possible chunk shape for this access pattern. Every `a[i]` row access must load 1024 separate single-column chunks. Over 1024 rows, that is over 1 million chunk loads — a catastrophic mismatch between access pattern (row-by-row) and storage layout (column chunks). Performance would be extremely poor.
- C) (32, 32) is better than B but still 32× worse than A for this workload. Each row access spans 32 chunks (since 1024 columns ÷ 32 columns/chunk = 32 chunks per row). While manageable, it is far from optimal.
- D) Chunk size absolutely affects performance with Zarr. Each chunk load triggers decompression (Zarr uses compression by default) and I/O operations, both of which have significant per-chunk overhead. The difference between 1,024 chunk loads and 1,048,576 loads is enormous.

---

## Question 21 — N-body Gravity: Which Optimization Claim is NOT Correct

A program simulates the motion of N objects in a solar system. A function calculates the gravitational force between all pairs of objects. The following numpy arrays are used:

- pos, shape (N, 3) where 3 is the number of spatial dimensions. Contains the positions of all objects for all timesteps.
- forces, shape (N, N, 3). Contains the force vector between all pairs of objects.
- mass, shape (N,). Contains the mass of all objects.

The Python code is shown below:

```python
def calc_forces(pos, mass):
    N = pos.shape[0]
    G = 6.6743e-11  # Gravitational constant
    forces = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            p0 = pos[i], p1 = pos[j]
            m0 = mass[i], m1 = mass[j]
            r = np.sqrt(np.sum((p1 - p0) * (p1 - p0)))
            forces[i,j] = G * m0 * m1 / (r * r)
    return forces
```

The code is correct but slow. Which of the following statements is **not** correct?

**Options:**
- A) Using the numba @jit decorator on the function is expected to improve performance
- B) The loops can be easily parallelized
- C) The processing time can be improved using only numpy
- D) The order of the loops can be switched with no impact

**Mental Model:** This question asks for the FALSE statement — the one optimization claim that is wrong. Three of the four options describe valid optimizations; one makes a false claim about loop ordering. The key is cache locality: `forces` is stored row-major (C order). The original loop (outer=i, inner=j) increments j in the inner loop, accessing `forces[i, 0], forces[i, 1], ..., forces[i, N-1]` — sequential memory access (stride 1). Swapping loops (outer=j, inner=i) accesses `forces[0, j], forces[1, j], ..., forces[N-1, j]` — strided memory access (stride N). These have different cache behavior, so the claim "no impact" is false.

**Correct Answer: D)**

**Why D is correct:** The statement "the order of the loops can be switched with no impact" is **false**. In the original code, the inner loop increments `j` while `i` is fixed, so `forces[i, j]` accesses consecutive memory locations (row-major order, stride = 1 element). This is **cache-friendly sequential access** — each iteration reads the next element in memory.

If the loops are swapped (outer = j, inner = i), the inner loop accesses `forces[i, j]` where i increments and j is fixed. In row-major storage, `forces[0, j], forces[1, j], ...` are **N elements apart in memory** (stride = N). This **strided access pattern causes cache thrashing** — each access likely evicts the previously loaded cache line. For large N, this can cause 10–100× performance degradation. The loop order absolutely has an impact.

**Why the others are wrong (i.e., these statements ARE correct):**
- A) Numba `@jit` compiles Python for-loops to native machine code via LLVM. Python loops with per-element operations like this typically run 50–200× faster under Numba compared to the Python interpreter. This is one of the most effective optimizations for loop-heavy numerical code.
- B) The two nested loops compute `forces[i, j]` for each pair (i, j). There is no dependency between any two pairs — `forces[2, 3]` does not depend on `forces[1, 5]`. All N² iterations are independent, making this trivially parallelizable via `multiprocessing` or `numba.prange`.
- C) The nested loops could be replaced by NumPy broadcasting: `diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]` computes all N×N pairwise position differences simultaneously, and subsequent operations can also be vectorized. This eliminates Python loop overhead by delegating to NumPy's optimized C/Fortran backend.

---

## Question 22 — Simulation Parallelism: Choosing the Right Strategy

A sports ball manufacturer performs computational fluid dynamics simulations of a ball's motion through the air for 10 newly proposed designs. For each design, a large range of parameters are simulated. The simulation state is stored in a Python dictionary called params. In the following code, all_params contains a Python list with parameters for all simulations:

```python
def simulate_ball(params):
    while params.height > 0:
        params = simulate_one_time_step(params)
    save_results(params)

def simulate(all_params):
    for params in all_params:
        simulate_ball(params)
```

The function simulate_one_time_step is written in Python by another developer and may not be modified. Profiling on a small number of samples reveals run times of the simulate_ball function between a few minutes and a few hours. Given this information, which of the following approaches is expected to improve performance the most?

**Options:**
- A) Use multithreading with dynamic scheduling to call simulate_one_time_step in parallel to simulate more time steps in parallel
- B) Use multiprocessing with dynamic scheduling to run several calls to simulate_ball in parallel
- C) Use multithreading with static scheduling to run several calls to simulate_ball in parallel
- D) Use a GPU to run several calls to simulate_ball in parallel

**Mental Model:** This question has three independent constraints that each eliminate some options. Constraint 1: sequential dependency within a single simulation (each time step feeds the next) — eliminates parallelizing time steps (rules out A). Constraint 2: Python GIL blocks CPU-parallel threads for CPU-bound Python code — eliminates multithreading for performance (rules out A and C). Constraint 3: runtimes vary from minutes to hours — requires dynamic scheduling to prevent load imbalance (rules out C and D). The only option that satisfies all three constraints is B.

**Correct Answer: B)**

**Why B is correct:** Three independent constraints narrow the options to exactly one:

1. **Sequential dependency within a single simulation**: The while-loop in `simulate_ball` runs `params = simulate_one_time_step(params)` repeatedly. Each iteration depends on the previous `params` state. Time steps cannot be parallelized — step t+1 cannot start until step t completes. This eliminates any strategy that tries to parallelize within a single simulation (rules out A).

2. **Python GIL prevents CPU-parallel multithreading**: `simulate_one_time_step` is pure Python and cannot be modified. Python's Global Interpreter Lock (GIL) ensures only one thread executes Python bytecode at a time. CPU-bound multithreading provides essentially zero speedup for Python code. This eliminates all multithreading options for CPU-bound work (rules out A and C).

3. **Variable runtimes require dynamic scheduling**: Simulations take "between a few minutes and a few hours." With static scheduling (upfront equal division), a worker assigned to all long-running simulations will run for hours while workers with short simulations sit idle. Dynamic scheduling (work queue) assigns a new simulation to each worker as soon as it finishes its current one, keeping all cores fully utilized. This rules out static scheduling (rules out C).

**Multiprocessing** (separate processes) bypasses the GIL. **Dynamic scheduling** (e.g., `multiprocessing.Pool` with `imap_unordered`) handles variable runtimes. **B is the only option satisfying all three constraints.**

**Why the others are wrong:**
- A) Multithreading within a single simulation cannot parallelize time steps because of the sequential dependency (each step modifies and returns the new state). Even if the GIL were lifted, the fundamental algorithmic dependency prevents this. Double failure.
- C) Multithreading across multiple simulations is blocked by the GIL for Python code — no CPU-bound parallelism. Additionally, static scheduling with runtimes varying minutes-to-hours creates severe load imbalance: the worker unlucky enough to get all the multi-hour simulations would run until completion while others sit idle after finishing their short simulations.
- D) GPUs excel at data-parallel operations (same operation on thousands of independent data elements simultaneously). This simulation involves: a Python while-loop with variable termination, a Python dictionary, sequential state updates, and complex control flow. None of these patterns map to GPU execution. The simulations are not data-parallel at the GPU kernel level.

---

## Question 23 — GPU vs CPU: Benchmarking Pitfall with Transfer Overhead

A colleague asks for your advice on speeding up a function. Your colleague is not allowed to show you the code but presents the following information. The function takes two arguments: a large numpy array of approximately 10 GB and the number of iterations to run. In each iteration, each array element is updated with no dependencies on other array elements and with a fixed number of floating-point operations. The function returns the updated array after the specified number of iterations. Performance benchmarking has been done using 5 and 10 iterations.

A heavily optimized CPU implementation was found to be too slow taking 0.5 seconds for 5 iterations and 1 second for 10 iterations. A simple GPU implementation was found to take 0.85 seconds for 5 iterations and 1.1 seconds for 10 iterations including transferring the input array to device memory and the output array back to host memory. Since benchmarking showed no improvement, it has been decided to not use the GPU implementation. Your colleague now asks for your advice. What do you say?

**Options:**
- A) The described problem is not suited for GPU implementation
- B) If the CPU implementation is already optimized, no further performance improvements can be expected
- C) The simple GPU implementation should be optimized for them to expect any performance improvement
- D) The current GPU implementation already has better performance, and they will see that by running more iterations

**Mental Model:** When comparing CPU vs GPU with memory transfer overhead, separate the timing into fixed costs (constant overhead regardless of iterations) and variable costs (scale with iterations). Fit a line to both: GPU = fixed_overhead + per_iter_cost × n. Use the two data points (5 iterations, 10 iterations) to compute: per_iter_cost = (1.1 - 0.85) / (10 - 5) = 0.05 s/iter. CPU per_iter_cost = (1.0 - 0.5) / (10 - 5) = 0.1 s/iter. GPU is already 2× faster per iteration — the overhead amortizes quickly. Find the break-even point and advise using more iterations in the benchmark.

**Correct Answer: D)**

**Why D is correct:** We decompose the timing models using the two data points for each implementation:

**GPU timing:**
- 5 iterations = 0.85 s, 10 iterations = 1.10 s
- Per-iteration cost: (1.10 − 0.85) / (10 − 5) = 0.25 / 5 = **0.05 s/iteration**
- Fixed overhead (transfer cost): 0.85 − (5 × 0.05) = 0.85 − 0.25 = **0.60 s** (constant, independent of iterations)

**CPU timing:**
- 5 iterations = 0.50 s, 10 iterations = 1.00 s
- Per-iteration cost: (1.00 − 0.50) / (10 − 5) = 0.50 / 5 = **0.10 s/iteration**
- No fixed overhead

**Break-even point:** GPU total = CPU total → 0.60 + 0.05n = 0.10n → 0.60 = 0.05n → **n = 12 iterations**

For any n > 12 iterations, the GPU is faster. The benchmarking used only 5 and 10 iterations — both below the break-even point of 12. The conclusion to not use the GPU was a **benchmarking error**: the test iterations were too few to amortize the one-time transfer overhead. In production with large iteration counts (e.g., 100 or 1000), GPU is approximately **2× faster per iteration** (0.05 vs 0.10 s/iter), yielding a clear win once the constant 0.60 s transfer cost is negligible.

**Why the others are wrong:**
- A) The problem is an ideal GPU candidate: per-element updates with no inter-element dependencies is the textbook data-parallel embarrassingly parallel workload. The per-iteration speedup (GPU = 0.05 s/iter vs CPU = 0.10 s/iter) confirms the GPU is already computing faster. "Not suited for GPU" is factually wrong.
- B) "No further improvement possible" is false. The GPU per-iteration cost is already 2× lower than the optimized CPU. The benchmark was run with too few iterations to see past the transfer overhead. Any benchmark shorter than the break-even point will misleadingly show the GPU as slower, even though it is faster per unit of work.
- C) The **simple** GPU implementation already achieves 0.05 s/iter, which is half the CPU's 0.10 s/iter. No additional GPU optimization is needed to outperform the CPU at realistic iteration counts. The problem is not optimization of the GPU code — it is the misleading short-iteration benchmark.

---

## Question 24 — Amdahl's Law and the Unknown Sequential Fraction

Assume you have a program that runs in thirty-two (32) seconds on one processor. Your manager wants the program to run faster than eight (8) seconds. You profile the program and find one function that takes up sixteen (16) seconds of the execution time and can easily be made parallel. What should you tell your manager?

**Options:**
- A) The program cannot run faster than sixteen (16) seconds.
- B) The program can possibly be made to run faster than eight (8) seconds, but further analysis of the program is needed.
- C) The program can be made to run faster than eight (8) seconds by rewriting the sequential part of the program.
- D) The program cannot run faster than (16 + 16/p) seconds where p is the number of processors.

**Mental Model:** This is Amdahl's Law with **incomplete information**. You know: total = 32 s, one parallelizable function = 16 s, remaining = 16 s (unknown composition). If you parallelize the known 16-second function perfectly (reduce to 0), you still have 16 seconds left — not meeting the 8-second goal. But you don't know what those other 16 seconds contain. They might also be parallelizable (then the goal is achievable) or they might be truly sequential (then it's not). The honest answer is: possibly achievable, but you need to analyze the remaining 16 seconds. Watch out for options that assert certainty when the information is incomplete.

**Correct Answer: B)**

**Why B is correct:** We know:
- Total runtime: 32 seconds
- One identified parallelizable function: **16 seconds**
- Remaining time: 32 − 16 = **16 seconds** (composition unknown — could be sequential, could contain more parallelizable code)

**Best-case analysis**: If we perfectly parallelize the known 16-second function (infinite cores → 0 seconds), the program takes at minimum the remaining 16 seconds. This does not meet the 8-second target. However, we have **no information** about what the other 16 seconds contains. It might include:
- Other parallelizable functions (which could reduce the total below 8 seconds)
- Code amenable to algorithmic optimization
- I/O operations that overlap with computation

Since the remaining 16 seconds is unanalyzed, the intellectually honest answer is: the 8-second target **might** be achievable, but **further profiling of the remaining 16 seconds is required** to know whether there are more optimization opportunities.

**Why the others are wrong:**
- A) "Cannot run faster than 16 seconds" implicitly assumes the remaining 16 seconds is entirely sequential and irreducible. But we have no data justifying this assumption. If the other 16 seconds contains parallelizable functions (not yet identified), the actual achievable minimum could be well below 16 seconds. This option overclaims certainty.
- C) "Can be made to run faster than 8 seconds by rewriting the sequential part" is overconfident in the other direction. We do not know that rewriting the remaining 16 seconds will be sufficient or even possible. Without profiling the other 16 seconds, we cannot assert this. This option also overclaims certainty.
- D) "Cannot run faster than (16 + 16/p) seconds" models the program as having a fixed 16-second sequential floor plus a 16-second perfectly-parallelizable part, and nothing else. While this formula correctly captures Amdahl's bound for the *known* parallel portion, it incorrectly asserts the remaining 16 seconds is an irreducible sequential constant. We have no such data. As with option A, this assumes facts not in evidence.

---

## Quick Reference: All Answers

| Q  | Answer | Topic |
|----|--------|-------|
| 1  | A      | LSF rusage mem = total / n cores |
| 2  | B      | float16 precision at large magnitudes |
| 3  | A      | Set intersection is both commutative and associative |
| 4  | A      | reshape(-1) row-major flattening, index 8 = 67 |
| 5  | A      | Broadcasting: mean_pixels[:, None, None] → (N,1,1,3) |
| 6  | A      | cProfile cumtime: render_scene = 8.841s |
| 7  | B      | Only 2nd loop (render_scene) is parallelizable |
| 8  | C      | Amdahl: S(3)=2.5 → F=0.9 → S_max=10 |
| 9  | B      | real halves, user stays same (CPU time summed) |
| 10 | B      | kernel1 (high StdDev) → dynamic; kernel2 → static |
| 11 | A      | prep_conds fixed + loop * 10 = 14.7s |
| 12 | A      | v_step=[0,1,0] → non-coalesced axis-1 access |
| 13 | A      | ceil(200/16) = 13 → 13 x 13 blocks |
| 14 | B      | HtoD = 26.8ms > kernel 13.8ms > DtoH 0.16ms |
| 15 | A      | Numba auto-transfers both arrays: 2 HtoD + 2 DtoH |
| 16 | A      | Random access → slower memory hierarchy used |
| 17 | C      | version (0-42) fits in uint8/int8 |
| 18 | A      | 24MB / 24 bytes/row = 1,000,000 rows; max valid = 800,000 |
| 19 | D      | 10B / 100K stride = 100K elements * 1 byte = 100 KB |
| 20 | A      | Chunk (1,1024) = 1 chunk load per row |
| 21 | D      | Loop swap DOES impact performance (cache locality) |
| 22 | B      | Multiprocessing + dynamic scheduling |
| 23 | D      | GPU 0.05s/iter < CPU 0.1s/iter; transfer amortized |
| 24 | B      | Unknown other 16s → need further analysis |
