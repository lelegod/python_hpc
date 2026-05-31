# 02613 Python HPC — Exam F25 Full Solutions
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

**Correct Answer: A)**

**Why A is correct:** The `#BSUB -n 4` directive requests 4 cores. In LSF, the `rusage[mem=X]` value specifies memory **per core**, not total. The total memory needed is 100 GB, so each core must be allocated 100 GB / 4 = **25 GB**. LSF will then reserve 4 x 25 = 100 GB in total for the job.

**Why the others are wrong:**
- B) 100GB — This would request 100 GB per core, giving a total of 400 GB. You would be over-requesting by 4x, which wastes cluster resources and may cause the job to wait longer in the queue or be rejected.
- C) 10GB — This would give only 4 x 10 = 40 GB total, which is less than the required 100 GB. The script would likely crash with an out-of-memory error.
- D) 50GB — This would give 4 x 50 = 200 GB total. Again over-requesting by 2x; the job will run but wastefully holds resources other users could use.

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

**Correct Answer: B)**

**Why B is correct:** float16 has a resolution (relative precision) of 0.001. At the magnitude of 10000, the smallest representable difference is approximately 10000 * 0.001 = 10. This means the gap between consecutive representable float16 values near 10000 is about 10. The value 10001 would require representing 1.0001e4, but the significand does not have enough bits to encode that extra digit. Therefore, 10000 + 1 rounds back to 10000.

**Why the others are wrong:**
- A) 10001 — This would require float16 to represent a difference of 1 at magnitude 10000. The resolution is too coarse; 1 is below the representable granularity at this scale, so the result cannot be 10001.
- C) 9999 — Rounding down to 9999 would require an error of -2, which is not how IEEE rounding works. Round-to-nearest-even rounds to the closest representable value (10000), not below it.
- D) 10000.5 — float16 cannot represent 10000.5 either, for the same reason it cannot represent 10001. Any fractional addition at this magnitude is lost to rounding.

---

## Question 3 — Set Intersection as a Parallel Reduction Operator

Can the operation of set intersection (∩) be used in a parallel reduction framework? Recall that the intersection of two sets is a new set containing all elements present in *both* input sets. For example, {1, 2, 3, 4} ∩ {2, 3, 5, 6} = {2, 3}.

**Options:**
- A) Yes
- B) No, ∩ is *not* associative (but it *is* commutative)
- C) No, ∩ is *not* commutative (but it *is* associative)
- D) No, ∩ is *neither* associative nor commutative

**Correct Answer: A)**

**Why A is correct:** A parallel reduction requires the operation to be both **commutative** and **associative**.

- **Commutative:** A ∩ B = B ∩ A. If x is in A and B, it is in both A ∩ B and B ∩ A. Order does not matter.
- **Associative:** (A ∩ B) ∩ C = A ∩ (B ∩ C). An element x is in (A ∩ B) ∩ C if and only if x is in all three sets A, B, and C. The grouping of the intersections does not affect the result.

Since both properties hold, set intersection can be used in a parallel reduction tree.

**Why the others are wrong:**
- B) Claiming ∩ is not associative is wrong. As shown above, (A ∩ B) ∩ C = A ∩ (B ∩ C) always holds because membership in the result depends only on being in all sets, not on the order of evaluation.
- C) Claiming ∩ is not commutative is wrong. An element is in A ∩ B if and only if it is in both A and B — the symmetric condition means swapping A and B makes no difference.
- D) Both properties do hold, so this is doubly wrong. ∩ is both commutative and associative, making it perfectly suited for parallel reduction.

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

**Correct Answer: A)**

**Why A is correct:** `a.reshape(-1)` flattens the array to 1D. Because the array is stored row-major (C order), elements are laid out row by row: [1, 5, 43, 51, 32, 73, 2, 4, 67, 37, 9, 3, 54, 8, 22]. Index 8 (0-based) is **67**.

Index mapping: 0=1, 1=5, 2=43, 3=51, 4=32, 5=73, 6=2, 7=4, **8=67**, 9=37, 10=9, 11=3, 12=54, 13=8, 14=22.

**Why the others are wrong:**
- B) 51 — 51 is at index 3 in the flattened array (row 0, column 3). Confusing 1-based indexing with 0-based, or misreading the reshape result.
- C) 32 — 32 is at index 4 (last element of row 0). Off by 4 from the correct position.
- D) 8 — 8 is at index 13 in the flattened array (row 2, column 3). This would be the result if one mistakenly looked at column indices rather than the flattened linear index.

---

## Question 5 — NumPy Broadcasting for Image Normalization

Assume you have an array of RGB images stored as an N x H x W x 3 array named "images". For each image, you have a mean RGB pixel stored as an N x 3 array "mean_pixels". You need to subtract each mean pixel from all pixels in the corresponding image in "images". Which of the following lines of Python code achieves this?

**Options:**
- A) `images - mean_pixels[:, None, None]`
- B) `images - mean_pixels[None, None]`
- C) `images - mean_pixels[None, :, None]`
- D) `images - mean_pixels`

**Correct Answer: A)**

**Why A is correct:** `images` has shape (N, H, W, 3). `mean_pixels` has shape (N, 3). To broadcast correctly, we need `mean_pixels` to have shape (N, 1, 1, 3) so that NumPy will expand the H and W dimensions automatically. `mean_pixels[:, None, None]` inserts two new axes after the first dimension, giving shape **(N, 1, 1, 3)**, which broadcasts to (N, H, W, 3).

**Why the others are wrong:**
- B) `mean_pixels[None, None]` — This inserts two axes at the front, giving shape (1, 1, N, 3). This does not align with the (N, H, W, 3) shape and will either raise a broadcast error or produce incorrect results because N and 3 would be in the wrong positions.
- C) `mean_pixels[None, :, None]` — This gives shape (1, N, 1, 3). The N axis is now in position 1 (the H position), not position 0. This misaligns the per-image means with the wrong images.
- D) `images - mean_pixels` — mean_pixels has shape (N, 3), which cannot broadcast against (N, H, W, 3) directly. NumPy would attempt to align shapes from the right: (N, 3) vs (..., W, 3). Unless N happens to equal W, this raises a broadcast error. Even if N == W it would be semantically wrong.

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

**Correct Answer: A)**

**Why A is correct:** The **cumtime** (cumulative time) column is the correct metric for "time overall" — it includes time spent inside the function and all functions it calls. `render_scene` has cumtime = **8.841 seconds**, the highest of all listed user functions. It was called 201 times (once for the initial scene plus 200 rendered frames).

**Why the others are wrong:**
- B) advance_scene — cumtime = 4.845 seconds. This is the second most expensive function but less than half the time of render_scene. advance_scene handles physics simulation, render_scene handles the more expensive visual rendering.
- C) save_to_mp4 — cumtime = 2.405 seconds. Called only once (writing the final video file). Much less than render_scene's total.
- D) load_scene — cumtime = 1.005 seconds. The cheapest of the four named functions; called once at startup.

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

**Correct Answer: B)**

**Why B is correct:** In the second for-loop, each call to `R.render_scene(scene)` takes a scene object from `all_scenes` and produces an independent frame. There is no dependency between iterations — rendering frame i does not affect rendering frame j. This is an embarrassingly parallel workload.

**Why the others are wrong:**
- A) The first for-loop has a **sequential data dependency**: each iteration produces a new `scene` that becomes the input to the next iteration. `scene = R.advance_scene(scene, dt)` means iteration i+1 cannot start until iteration i completes. This is fundamentally serial.
- C) Both loops cannot be parallelized because the first loop has the dependency described above. Only the second is a valid candidate.
- D) Claiming neither can be parallelized ignores the obvious independence of the rendering loop. The second loop is a textbook example of an embarrassingly parallel pattern (map operation over independent inputs).

---

## Question 8 — Amdahl's Law: Inferring F from Measured Speedup

After adding parallelization to the Python program, an engineer measured the speed-up using 3 cores to be 2.5. Given this, what is the expected theoretical maximum speed-up?

**Options:**
- A) Around 15x
- B) Around 12x
- C) Around 10x
- D) Around 7x

**Correct Answer: C)**

**Why C is correct:** We use Amdahl's Law. The speedup with p processors is:

S(p) = 1 / (1 - F + F/p)

where F is the parallel fraction. We know S(3) = 2.5. Solving for F:

1/S(p) = 1 - F + F/p

Rearranging: F - F/p = 1 - 1/S(p), so F(p-1)/p = 1 - 1/S(p)

F = p(1 - 1/S(p)) / (p - 1) = 3 * (1 - 1/2.5) / (3 - 1) = 3 * (1 - 0.4) / 2 = 3 * 0.6 / 2 = **0.9**

The theoretical maximum speedup (p → infinity) is:

S_max = 1 / (1 - F) = 1 / (1 - 0.9) = 1 / 0.1 = **10**

**Why the others are wrong:**
- A) 15x — This would imply F ≈ 0.933. Plugging back in: S(3) = 1/(0.067 + 0.933/3) = 1/(0.067 + 0.311) = 1/0.378 ≈ 2.65, which does not match the observed 2.5.
- B) 12x — Implies F ≈ 0.917. S(3) = 1/(0.083 + 0.917/3) = 1/(0.083 + 0.306) = 1/0.389 ≈ 2.57, which does not match 2.5.
- D) 7x — Implies F ≈ 0.857. S(3) = 1/(0.143 + 0.857/3) = 1/(0.143 + 0.286) = 1/0.429 ≈ 2.33, which is less than 2.5. This F is too low.

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

**Correct Answer: B)**

**Why B is correct:** The `time` command reports three distinct values:
- **real** (wall-clock time): the actual elapsed time from start to finish. With perfect parallelization on 2 cores, this halves to ~6 seconds.
- **user** (CPU time in user space): the total CPU time consumed across all cores, summed. With 2 cores each doing ~6 seconds of work, user time stays at ~12 seconds (or may even increase slightly due to overhead).
- **sys**: kernel CPU time, also summed across cores, essentially unchanged.

So real goes down, but user and sys remain constant (or increase). Option B correctly shows real halved while user stays at 12 seconds.

**Why the others are wrong:**
- A) Shows both real and user halved to ~6 seconds. This is wrong because user time is the **sum** of CPU time across all cores. If 2 cores each run for 6 seconds, user = 12 seconds, not 6. This confusion mistakes wall time for per-core CPU time.
- C) Shows real unchanged at 12 seconds but user halved to 6 seconds. This is backwards — parallelization reduces wall time (real), not necessarily CPU time. A program running the same work on 2 cores uses roughly the same total CPU time but in half the wall time.
- D) Shows no change at all. This would be the result if parallelization had zero effect — i.e., the program is still running on 1 core. This contradicts the premise of perfect parallelization.

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

**Correct Answer: B)**

**Why B is correct:** The key signal is **StdDev (standard deviation)** of kernel execution time.

- **kernel1**: StdDev = 40.0 ms with Avg = 20.0 ms. This is an extremely high coefficient of variation (200%). Individual kernel invocations vary wildly in runtime. With static scheduling, some GPUs will receive a batch of slow kernels and others a batch of fast kernels. The fast GPUs finish and sit idle while waiting for the overloaded slow GPU. Dynamic scheduling assigns work on-demand, keeping all GPUs busy and minimizing idle time.
- **kernel2**: StdDev = 0.05 ms with Avg = 20.0 ms. This is extremely uniform runtime. All kernel instances take roughly the same time, so dividing work equally up front (static scheduling) distributes load perfectly. Dynamic scheduling would only add queue management overhead with no benefit.

**Why the others are wrong:**
- A) Static is not optimal for kernel1 because of the enormous runtime variance. Load imbalance will cause significant GPU idle time.
- C) This reversal is wrong. kernel2 is the uniform one — static is fine. kernel1 is the variable one — it needs dynamic scheduling.
- D) kernel2 does not need dynamic scheduling. Its near-zero variance means static scheduling achieves perfect load balance, and adding a dynamic task queue introduces unnecessary overhead.

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

**Correct Answer: A)**

**Why A is correct:** Step-by-step calculation:

1. The for-loop (lines 101-103) ran 1000 times, so the small dataset had **1000 elements**.
2. For a 10,000-element dataset, the loop runs **10x** as many times.
3. Loop total time (small dataset) = 740 + 1,266,748 + 1,685 = **1,269,173 microseconds = 1.269 seconds**.
4. Loop time for 10,000 elements = 1.269 * 10 = **12.69 seconds**.
5. `prep_conds` (line 99) takes **2,005,064 microseconds = 2.005 seconds**. Crucially, it depends only on `params`, not `data`, so its runtime does not scale with dataset size. It remains ~2.005 seconds.
6. Total estimated time = 12.69 + 2.005 = **~14.7 seconds**.

**Why the others are wrong:**
- B) 32.7 seconds — This would result from multiplying the entire 3.27-second profiling run by 10. This is wrong because `prep_conds` does not scale with data size; only the loop portion scales.
- C) 3.5 hours — This is wildly too large. It could arise from incorrectly assuming all parts scale with data and multiplying by a much larger factor, or confusing the timer unit (microseconds vs seconds).
- D) 12.7 seconds — This accounts only for the scaled loop time but forgets to add the constant `prep_conds` time of ~2 seconds.

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

**Correct Answer: A)**

**Why A is correct:** For memory coalescing on GPU, threads in the same warp (adjacent thread IDs) should access adjacent memory addresses simultaneously. In a 16x16 thread block, threads with adjacent `j` values (i.e., adjacent within the same row) are adjacent in a warp.

The memory access pattern for `vol[vi, vj, vk]` is determined by how `ray_step` moves through `vol` at each iteration. Since `vol` is row-major, the last axis (vk, axis 2) has stride 1 — adjacent vk values are adjacent in memory. For coalesced access, we want adjacent threads to differ in vk.

In option A: v_step = [0.0, 1.0, 0.0]. Adjacent threads (differing by 1 in j) differ in `vj` (axis 1), not in `vk` (axis 2). Since axis 1 has a larger stride than axis 2 in row-major layout, adjacent threads access memory locations that are far apart. This is **non-coalesced** and worst for memory efficiency.

In options B and C: v_step has [0, 0, 1] — adjacent threads differ in `vk` (axis 2, stride 1). These access contiguous memory locations, giving good coalescing.

**Why the others are wrong:**
- B) v_step = [0.0, 0.0, 1.0] means adjacent threads (differing in j) differ in vk (the last axis, stride 1). This gives coalesced access — adjacent threads read adjacent memory.
- C) Similarly, v_step = [0.0, 0.0, 1.0] gives the same coalesced pattern as B for the initial access. Better than A.
- D) The inputs do not have the same performance. The v_step direction determines whether adjacent threads access adjacent memory (coalesced, fast) or strided memory (non-coalesced, slow). Option A is significantly worse.

---

## Question 13 — CUDA Thread Block Count for Output Image

Consider again the same CUDA kernel function as in the previous questions which renders a depth map of a volume.

Given a volume of size 500 x 500 x 500, output image of size 200 x 200 and thread blocks of size 16 x 16, how many thread blocks are needed?

**Options:**
- A) 13 x 13
- B) 32 x 32 x 500
- C) 32 x 32
- D) 3 x 3

**Correct Answer: A)**

**Why A is correct:** Each thread writes exactly one element of the output `dmap`. The output image is 200 x 200. The thread block size is 16 x 16. We need enough blocks to cover all 200 x 200 output pixels:

- Blocks needed per dimension = ceil(200 / 16) = ceil(12.5) = **13**
- Total thread blocks = **13 x 13**

The volume size (500 x 500 x 500) is irrelevant for determining the grid size — it only matters for the ray-marching inner loop. The kernel reads from `vol` but writes to `dmap`, and the grid is sized to cover `dmap`.

**Why the others are wrong:**
- B) 32 x 32 x 500 — This incorrectly uses the volume dimension (500) as a third grid dimension, and 32 = ceil(500/16) applies the volume size rather than the output image size. The kernel is 2D (one thread per output pixel) and has no reason to involve the 500 dimension in the grid shape.
- C) 32 x 32 — This would be correct for a 500 x 500 output image (ceil(500/16) = 32). But the output image is 200 x 200, not 500 x 500.
- D) 3 x 3 — This applies ceil(200/64) ≈ 3 rather than ceil(200/16) = 13. Using the wrong block size calculation (perhaps confusing 16x16 = 256 threads with 64x64 blocks).

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

**Correct Answer: B)**

**Why B is correct:** Reading the "Total Time (ms)" column from each section:
- Kernel execution: **13.84 ms**
- HtoD transfers (CPU to GPU): **26.80 ms**
- DtoH transfers (GPU to CPU): **0.16 ms**

HtoD at 26.8 ms is the largest, accounting for nearly double the kernel execution time. The 125 MB of data being copied to the GPU (the large `vol` array plus `dmap`) takes more time than the actual computation.

**Why the others are wrong:**
- A) The kernel takes 13.84 ms — significant but less than half of the HtoD transfer time. Simply running the computation is not the bottleneck here.
- C) DtoH (GPU to CPU) is only 0.16 ms — negligible. Only 2 MB is transferred back (just the output `dmap`), so this is trivially fast.
- D) The three components differ by roughly 2x (kernel vs HtoD) and 170x (DtoH vs HtoD). They are far from equal.

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

**Correct Answer: A)**

**Why A is correct:** When Numba's `@cuda.jit` kernel is called with plain NumPy arrays, Numba automatically:
1. Copies **both** arguments to the GPU before the kernel runs: **2 HtoD transfers** (one for `x`, one for `y`).
2. Copies **both** arguments back to host memory after the kernel: **2 DtoH transfers**.

However, semantically `x` is a read-only input — it never needs to go back to the host. And `y` is a write-only output — it starts empty and does not need to be sent to the GPU before the kernel. An optimal implementation using `cuda.to_device(x)` and `cuda.device_array_like(y)` would require only:
- **1 HtoD** (transfer `x` to GPU)
- **1 DtoH** (transfer `y` back to host)

**Why the others are wrong:**
- B) Claims only 1 HtoD and 1 DtoH actually happen. This ignores Numba's automatic transfer behavior with NumPy arrays. Numba conservatively transfers all arguments in both directions when passed as NumPy arrays.
- C) Claims 2+2 transfers are both what happens and what is necessary. While the 2+2 describes what happens, 2+2 is not necessary. An optimized version only needs 1+1.
- D) Claims 2 HtoD and 1 DtoH occur, and 1 HtoD and 2 DtoH are necessary. Both parts are wrong: Numba performs 2+2, and the optimal is 1+1 (not 1+2).

---

## Question 16 — Cache Behavior: Random Access Pattern

You find that one of your programs have low performance. You inspect the code, and you find that you access a large data structure in a random fashion. What is the likely cause of the low performance?

**Options:**
- A) A slower part of the memory hierarchy is used.
- B) The caches are too small.
- C) There are too few cache levels.
- D) There is not enough information to find the root cause.

**Correct Answer: A)**

**Why A is correct:** With purely random access patterns across a large data structure, the data being accessed is unlikely to be in any cache level. Every access results in a **cache miss** that must be served from main memory (DRAM), which is 10-100x slower than L3 cache. The root cause is that the working set accessed by the random pattern exceeds cache capacity, forcing the CPU to use the slower DRAM level of the memory hierarchy.

**Why the others are wrong:**
- B) "The caches are too small" is not correct as a root cause analysis. With random access, even arbitrarily large caches would not help — you would still get cache misses because you never reuse the same data element. The problem is the access *pattern*, not the cache *size*. Larger caches cannot fix random access.
- C) "Too few cache levels" is similarly wrong. Adding more cache levels (L4, L5...) would not fix the fundamental problem that random access produces no temporal locality. Each access still fetches a different cache line that is never reused.
- D) There is sufficient information: random access of a large data structure is a well-known cause of cache miss-dominated performance. This is not an ambiguous situation requiring more information.

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

**Correct Answer: C)**

**Why C is correct:** The `version` column holds integers with min=0 and max=42. It is currently stored as int64 (8 bytes per value). The values fit comfortably in:
- **uint8**: stores 0 to 255 — covers 0 to 42 perfectly (1 byte per value, 8x reduction)
- **int8**: stores -128 to 127 — also covers 0 to 42 (1 byte per value, 8x reduction)

Converting to uint8 or int8 would reduce the 2.1 GB column to about 262 MB, an 8x saving.

**Why the others are wrong:**
- A) Converting to datetime makes no semantic sense. `version` is a version number (0-42), not a timestamp. datetime would add no benefit and would be semantically incorrect. You cannot store integer version numbers as dates without distorting meaning.
- B) Converting to a floating point type (e.g., float16 or float32) would be inappropriate for integer data. Integer-to-float conversion can introduce rounding errors in downstream operations (comparisons, joins, groupby). Since the data is inherently integer, an integer type is always preferred.
- D) The column can definitely be reduced. Changing from int64 to int8 or uint8 is a direct, lossless reduction since all values fit within the smaller range.

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

**Correct Answer: A)**

**Why A is correct:** Each column uses int64 dtype = **8 bytes per value**. There are 3 columns, so each row uses 8 * 3 = **24 bytes**.

Maximum rows that fit in 24 MB:
- Using 1 MB = 1,000,000 bytes: 24 MB = 24,000,000 bytes. Rows = 24,000,000 / 24 = **1,000,000**
- Using 1 MB = 1,024 * 1,024 bytes = 1,048,576 bytes: 24 MB = 25,165,824 bytes. Rows = 25,165,824 / 24 = **1,048,576**

The only answer choice that is ≤ 1,000,000 is **800,000**. All other options (1,100,000; 2,400,000) exceed the memory limit. Option D (500,000) is valid but not the maximum.

Since the question asks for the **maximum** chunk size that still fits, 800,000 is the largest valid option. (1,100,000 would require 26.4 MB which exceeds 24 MB.)

**Why the others are wrong:**
- B) 2,400,000 rows * 24 bytes = 57,600,000 bytes = ~57.6 MB. This is 2.4x the available 24 MB. The system would run out of memory.
- C) 1,100,000 rows * 24 bytes = 26,400,000 bytes = ~26.4 MB. This slightly exceeds the 24 MB limit regardless of whether you use 1000-based or 1024-based MB.
- D) 500,000 rows * 24 bytes = 12,000,000 bytes = 12 MB. This fits, but it is not the maximum. The question asks for the maximum allowable chunk size, so this is a valid but suboptimal answer — 800,000 is larger and still fits.

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

**Correct Answer: D)**

**Why D is correct:** `np.memmap` creates a memory-mapped view of the file — the file's 10 billion bytes are **not loaded into RAM**. The OS maps the file into virtual address space and only loads pages on demand.

`x[::100_000]` is a strided slice that takes every 100,000th element of the 10,000,000,000-element array:
- Number of elements = 10,000,000,000 / 100,000 = **100,000 elements**
- Each element is uint8 = 1 byte
- Total memory = 100,000 bytes = **100 KB**

This is the only data actually loaded into RAM. The `np.array(...)` call materializes these 100,000 elements into a regular in-memory array.

**Why the others are wrong:**
- A) 10 GB — This would be the size if the entire memmap file were loaded into RAM. But memmap does not load the whole file; only the accessed pages are loaded, and only the 100,000 elements extracted into `y` are truly in RAM.
- B) 100 MB — This is 100x too large. This would correspond to 100,000,000 elements of uint8, or 100,000 elements of float64 (8 bytes each). The actual array is 100,000 elements of uint8 = 100 KB.
- C) 10 KB — This is 10x too small. 10 KB = 10,000 bytes = 10,000 elements. The slice extracts 100,000 elements, not 10,000.

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

**Correct Answer: A)**

**Why A is correct:** The function iterates over rows (`a[i]`) and computes the sum of each row. Each iteration `a[i]` accesses all 1024 columns of row i as a 1D slice.

Zarr loads data in chunks. The goal is to minimize the number of chunk loads per `a[i]` access:
- **Chunk size (1, 1024)**: Each chunk is one complete row. Accessing `a[i]` loads exactly **1 chunk**. Over 1024 iterations: 1024 chunk loads total.
- **Chunk size (1024, 1)**: Each chunk is one complete column. Accessing `a[i]` (a row) requires reading across all 1024 column chunks: **1024 chunk loads per row**. Over 1024 iterations: 1024 * 1024 = 1,048,576 chunk loads total.
- **Chunk size (32, 32)**: Each row access requires loading 1024/32 = 32 column chunks. Over 1024 iterations: 1024 * 32 = 32,768 chunk loads total.

Option A requires the minimum chunk loads (1024 total) and therefore best performance.

**Why the others are wrong:**
- B) (1024, 1) is the worst possible choice. Every row access requires loading 1024 separate column-chunks, resulting in over a million chunk loads. This destroys performance due to I/O overhead per chunk load.
- C) (32, 32) is better than B but still 32x worse than A. Each row access spans 32 chunks (since 1024 columns / 32 columns-per-chunk = 32 chunks per row).
- D) Chunk size absolutely affects performance with Zarr. Chunk loading has overhead (decompression, disk seeks), and minimizing the number of chunks loaded per operation is critical.

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

**Correct Answer: D)**

**Why D is correct:** The statement "the order of the loops can be switched with no impact" is **false** and is therefore the answer. The `forces` array is stored row-major (C order). In the original code, the inner loop increments `j`, which means `forces[i, j]` accesses consecutive elements in memory as j increases — **excellent spatial locality** (cache-friendly, sequential access). If the loops are swapped (outer = j, inner = i), then `forces[j, i]` is accessed, which steps by N elements in memory with each increment of i — **poor spatial locality** (strided access, cache-unfriendly). The loop order change meaningfully impacts performance.

**Why the others are wrong (i.e., these statements ARE correct):**
- A) Numba @jit will JIT-compile the Python for-loops to native machine code. Python loops are extremely slow due to interpreter overhead; Numba can provide 10-100x speedup for loop-heavy numerical code.
- B) The two loops have no dependencies between iterations (forces[i,j] does not depend on forces[i',j'] for any other i', j'). Both loops can be parallelized trivially using multiprocessing or `numba.prange`.
- C) The nested loops could be replaced by NumPy broadcasting operations (computing all pairwise distances at once using pos[:, np.newaxis, :] - pos[np.newaxis, :, :]), which offloads the computation to NumPy's optimized C backend and eliminates Python loop overhead.

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

**Correct Answer: B)**

**Why B is correct:** Three constraints narrow the options:

1. **Sequential dependency within a single ball simulation**: Each call to `simulate_one_time_step` depends on the output of the previous one (the updated `params`). Therefore, we cannot parallelize time steps within a single ball simulation (rules out A).

2. **Python GIL prevents multithreading speedup for CPU-bound Python code**: `simulate_one_time_step` is Python code that may not be modified. Python's Global Interpreter Lock (GIL) prevents true parallel CPU execution in threads. Multithreading gives no speedup for pure Python computation (rules out A and C).

3. **Variable runtime requires dynamic scheduling**: Ball simulations take anywhere from minutes to hours. With static scheduling (equal upfront division), some workers finish quickly and sit idle while others work on long simulations. Dynamic scheduling (work queue) assigns new simulations to free workers, keeping all cores busy (rules out C and D for different reasons).

Multiprocessing bypasses the GIL (separate processes), and dynamic scheduling handles the variable runtimes. **B is the best choice.**

**Why the others are wrong:**
- A) Multithreading within a single simulation cannot parallelize time steps because they are sequentially dependent. Additionally, the GIL prevents Python threads from running Python code in parallel.
- C) Multithreading of multiple ball simulations is blocked by the GIL — no CPU speedup for Python code. Additionally, static scheduling would cause load imbalance given the highly variable runtimes.
- D) GPUs excel at data-parallel workloads (thousands of identical simple operations). A sequential simulation involving Python dictionary manipulation and complex control flow does not map well to GPU execution. The variable-length while-loop per simulation and Python data structures are not GPU-friendly.

---

## Question 23 — GPU vs CPU: Benchmarking Pitfall with Transfer Overhead

A colleague asks for your advice on speeding up a function. Your colleague is not allowed to show you the code but presents the following information. The function takes two arguments: a large numpy array of approximately 10 GB and the number of iterations to run. In each iteration, each array element is updated with no dependencies on other array elements and with a fixed number of floating-point operations. The function returns the updated array after the specified number of iterations. Performance benchmarking has been done using 5 and 10 iterations.

A heavily optimized CPU implementation was found to be too slow taking 0.5 seconds for 5 iterations and 1 second for 10 iterations. A simple GPU implementation was found to take 0.85 seconds for 5 iterations and 1.1 seconds for 10 iterations including transferring the input array to device memory and the output array back to host memory. Since benchmarking showed no improvement, it has been decided to not use the GPU implementation. Your colleague now asks for your advice. What do you say?

**Options:**
- A) The described problem is not suited for GPU implementation
- B) If the CPU implementation is already optimized, no further performance improvements can be expected
- C) The simple GPU implementation should be optimized for them to expect any performance improvement
- D) The current GPU implementation already has better performance, and they will see that by running more iterations

**Correct Answer: D)**

**Why D is correct:** We can decompose the GPU timing into fixed overhead (memory transfer) and per-iteration computation:

- GPU: 5 iterations = 0.85s, 10 iterations = 1.1s
- Difference: 5 extra iterations cost 1.1 - 0.85 = **0.25 seconds** → **0.05 seconds per iteration**
- Fixed transfer overhead = 0.85 - (5 * 0.05) = 0.85 - 0.25 = **0.60 seconds** (constant regardless of iterations)

- CPU: 5 iterations = 0.5s, 10 iterations = 1.0s → **0.1 seconds per iteration** (no fixed overhead)

At n iterations, total time:
- CPU: 0.1n seconds
- GPU: 0.6 + 0.05n seconds

Break-even: 0.1n = 0.6 + 0.05n → 0.05n = 0.6 → n = **12 iterations**

For n > 12, the GPU is faster. In real use with millions of iterations, the GPU's 0.05 s/iter vs CPU's 0.10 s/iter gives a **2x speedup** (minus the negligible constant 0.6s overhead). The benchmarking was done with too few iterations to see past the transfer overhead.

**Why the others are wrong:**
- A) The problem is perfectly suited for GPU: per-element updates with no dependencies is the textbook GPU-friendly pattern (embarrassingly data-parallel). The GPU's 0.05 s/iter vs CPU's 0.10 s/iter confirms the GPU is computing faster per iteration.
- B) "No further improvement possible" is false. Other hardware (GPUs) can vastly outperform even optimized CPU code for data-parallel workloads. The benchmark shows the GPU computes each iteration in half the time of the CPU once you account for the constant transfer cost.
- C) The simple GPU implementation already has **better per-iteration performance** than the optimized CPU. No further GPU optimization is needed to beat the CPU at large iteration counts — the math shows this at n > 12 iterations.

---

## Question 24 — Amdahl's Law and the Unknown Sequential Fraction

Assume you have a program that runs in thirty-two (32) seconds on one processor. Your manager wants the program to run faster than eight (8) seconds. You profile the program and find one function that takes up sixteen (16) seconds of the execution time and can easily be made parallel. What should you tell your manager?

**Options:**
- A) The program cannot run faster than sixteen (16) seconds.
- B) The program can possibly be made to run faster than eight (8) seconds, but further analysis of the program is needed.
- C) The program can be made to run faster than eight (8) seconds by rewriting the sequential part of the program.
- D) The program cannot run faster than (16 + 16/p) seconds where p is the number of processors.

**Correct Answer: B)**

**Why B is correct:** We know:
- Total runtime: 32 seconds
- One parallelizable function: 16 seconds
- Remaining time: 32 - 16 = **16 seconds** (unknown composition — could be sequential, could contain more parallelizable parts)

If we perfectly parallelize the known 16-second function (sending p → infinity), it contributes 0 seconds. The remaining 16 seconds would give a total of **16 seconds** — not yet at the 8-second goal.

However, the key word is "easily be made parallel" for one function. We have **no information** about whether the remaining 16 seconds contains other parallelizable opportunities. If the other 16 seconds can also be parallelized or optimized, the 8-second target might be achievable. Since we don't know, the honest answer is: possibly, but further profiling of the other 16 seconds is needed.

**Why the others are wrong:**
- A) "Cannot run faster than 16 seconds" assumes the remaining 16 seconds is entirely sequential and irreducible. But we have no such information. The other 16 seconds is unexplored and might contain parallelizable or optimizable code.
- C) "Can be made to run faster than 8 seconds by rewriting the sequential part" is overconfident. Rewriting sequential code might help, but this claim cannot be made without knowing what the other 16 seconds actually does. There is no guarantee that rewriting will be sufficient.
- D) "Cannot run faster than (16 + 16/p) seconds" assumes the entire known 16-second function is parallelizable and nothing else is, and that it achieves perfect linear speedup. While this formula correctly models the Amdahl bound for the known parallel part, it incorrectly asserts the remaining 16 seconds is a fixed sequential floor when we have no data on those 16 seconds.

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
