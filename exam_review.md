# 02613 Python HPC — Exam Review

> **Root files:** [STUDY_GUIDE](STUDY_GUIDE.md) · [Exam Review](exam_review.md) · [Cheat Sheet](master_cheat_sheet.md) · [Tips & Pitfalls](tips_and_tricks.md) · [README](README.md)

---

## Contents

- [Exam Format](#exam-format)
- [Topic Distribution by Exam](#topic-distribution-by-exam)
- [Question Type Analysis](#question-type-analysis)
  - [1. Read and interpret a job script](#1-read-and-interpret-a-job-script)
  - [2. Amdahl's law calculations](#2-amdahls-law-calculations)
  - [3. GIL and parallelisation strategy](#3-gil-and-parallelisation-strategy)
  - [4. Parallel reduction associativity](#4-parallel-reduction-associativity)
  - [5. NumPy broadcasting](#5-numpy-broadcasting)
  - [6. Cache efficiency and loop ordering](#6-cache-efficiency-and-loop-ordering)
  - [7. Reading profiler output](#7-reading-profiler-output)
  - [8. GPU kernel thread blocks and coalescing](#8-gpu-kernel-thread-blocks-and-coalescing)
  - [9. Pandas memory optimisation](#9-pandas-memory-optimisation)
  - [10. Chunked processing and memory budgeting](#10-chunked-processing-and-memory-budgeting)
- [High-Frequency Topics](#high-frequency-topics)
- [Per-Exam Breakdown](#per-exam-breakdown)
  - [2024 Exam](#2024-exam-28052024-14-pages-19-questions)
  - [2024 Re-exam](#2024-re-exam-21082024-10-pages-18-questions)
  - [F25 Exam](#f25-exam-2025-18-pages-24-questions--all-multiple-choice)
- [Common Traps and Mistakes](#common-traps-and-mistakes)
- [What the Exam Rewards](#what-the-exam-rewards)
- [Exam Preparation Strategy](#exam-preparation-strategy)
  - [Week-by-week priority](#week-by-week-priority)
  - [Exam-day tactics](#exam-day-tactics)
  - [What is never tested](#what-is-never-tested-based-on-these-three-exams)

---

## Exam Format

> **THIS YEAR'S EXAM IS MCQ ONLY.** Study accordingly — the F25 exam is the best model to practice with.

**Duration:** 4 hours

**Aids allowed:** Written works of reference (textbooks, notes, printed materials). No digital aids — bring physical notes/cheat sheets.

**Format:** Entirely multiple-choice. ~24 questions, each with 4 options (A/B/C/D). No partial credit for reasoning — just pick the letter. Strategy: eliminate wrong answers, watch for distractors that are almost-right.

**Historical formats (for context only, NOT this year's format):**
- **2024 Exam and 2024 Re-exam:** Free-response, 18-19 questions. Required showing calculations and justifying answers.
- **F25 Exam:** MCQ only, 24 questions — same format as this year's exam. **Use F25 as your primary practice exam.**

**Default conventions (stated on every exam):**
- All arrays are NumPy arrays stored row-wise (C order).
- `np` refers to the NumPy module.

**Key instruction:** For open-ended questions, unexplained answers receive substantially reduced credit. Always show your reasoning, state your assumptions, and motivate your conclusions — even when the calculation is simple.

---

## Topic Distribution by Exam

| Topic | 2024 Exam | 2024 Re-exam | F25 Exam |
|---|---|---|---|
| LSF/BSUB job scripts (reading, writing, modifying) | Q1, Q18 | Q1, Q2 | Q1 |
| Amdahl's law / parallel fraction / speed-up | Q2, Q3 | Q3, Q4 | Q8, Q24 |
| GIL / multi-threading vs multi-processing | Q4, Q19 | Q17 | Q22 |
| Static vs dynamic scheduling | Q5 | — | Q10, Q22 |
| Parallel reduction (associativity, commutativity) | Q6 | Q18 | Q3 |
| NumPy broadcasting | Q7 | Q9, Q10 | Q5 |
| Cache efficiency / memory layout / loop ordering | Q8, Q11, Q12 | Q16 | Q12, Q16, Q20, Q21 |
| Profiling (cProfile / line profiler) — reading output | Q9, Q10 | Q7, Q8 | Q6, Q11 |
| Profiling — scaling estimates for normal workload | Q10 | Q7 | Q11 |
| CUDA / GPU kernels (thread blocks, warps, grid) | Q11, Q12, Q13, Q14 | Q11, Q12, Q13, Q14 | Q12, Q13, Q14, Q15 |
| GPU memory transfers (HtoD / DtoH, nsys profiler) | Q13, Q14 | Q13, Q14 | Q14, Q15 |
| Pandas DataFrame memory reduction / dtype recoding | Q15 | — | Q17, Q18 |
| Pandas indexing / query acceleration | Q16 | — | — |
| Chunked processing / memory budgeting | Q17 | Q5, Q6 | Q18, Q19, Q20 |
| Job arrays and job dependencies (BSUB -w) | Q18 | Q15 | — |
| Numba @jit / nogil | Q19 | — | Q21 |
| Zarr arrays / block shapes | — | Q5, Q6 | Q20 |
| Floating point precision / dtype arithmetic | — | — | Q2 |
| NumPy reshape / memory order | — | — | Q4 |
| FLOP/s calculation from profiler output | — | Q8 | — |
| time command / wall time vs CPU time | — | — | Q9 |
| np.memmap | — | — | Q19 |
| Gravitational simulation / loop optimisation | — | — | Q21 |
| Simulate_ball / parallelisation strategy selection | — | — | Q22 |
| GPU amortisation over many iterations | — | — | Q23 |

---

## Question Type Analysis

### 1. Read and interpret a job script
You are given a `#BSUB` script and asked what it requests, what is wrong, or how to fix it. You must know all common flags: `-n` (cores), `-W` (wall time), `-R "rusage[mem=XGB]"` (memory per core, so total = mem × cores), `-R "span[hosts=1]"` (keep all cores on one node), `-R "select[gpu]"` / `-gpu` / GPU queue names, `-w done(jobname)` / `-w ended(jobname)` (job dependencies), `-J name[1-N]` (job array), `$LSB_JOBINDEX` (array index).

*Example (2024 Q1):* Script requests 1 core, 4 GB, 5 min. Program needs 8 cores, 16 GB, 15 min. Fix: `-n 8`, `-W 00:15`, `-R "rusage[mem=2GB]"` (because 16 GB / 8 cores = 2 GB per core), add `-R "span[hosts=1]"`.

### 2. Amdahl's law calculations
Given a speed-up plot, measured speed-up, or observed run times, derive the parallel fraction F and use it to answer a follow-up question.

Formula: S(p) = 1 / (1 - F + F/p). Rearranging for F: F = p(1 - 1/S(p)) / (p - 1).

The theoretical maximum speed-up as p → ∞ is 1/(1-F).

*Example (2024 Q2/Q3):* Speed-up plot saturates at 5. Max speed-up = 1/(1-F) = 5, so F = 0.8. With 8 cores: S(8) = 1/(0.2 + 0.8/8) = 1/(0.2 + 0.1) = 10/3 ≈ 3.33 < 4. Do not pursue parallelisation.

*Example (F25 Q8):* Measured S(3) = 2.5. Solve for F: F = 3(1 - 1/2.5)/(3-1) = 3(0.4)/2 = 0.9. Max speed-up = 1/(1-0.9) = 10.

*Example (Re-exam Q3):* Given F=0.8, S(4)=2.5, time on 4 procs = 10 min → time on 1 proc = 25 min.

### 3. GIL and parallelisation strategy
Decide whether to use multi-threading or multi-processing, and whether the strategy is viable given the code context.

Rules:
- Pure Python (loops, no NumPy releasing GIL) → GIL blocks threads → use **multi-processing**.
- NumPy operations, Numba with `nogil=True`, or other GIL-releasing code → **multi-threading** works.
- `simulate_single` with `@jit(nogil=True)` releases GIL → threading viable (2024 Q19).
- Independent tasks with variable run time → **dynamic scheduling** (e.g., process_number(n) where n varies hugely).
- Pure Python with high variance → multi-processing + dynamic.

### 4. Parallel reduction associativity
For a function to be usable in a parallel reduction tree, it must be **associative** (and ideally commutative, though commutativity alone is not sufficient).

*Example (2024 Q6):* `abssum(x,y) = abs(x+y)`. Test: ||(1+2)+(-3)|| = 0 ≠ ||1+(2+(-3))|| = 2. Not associative → cannot use in parallel reduction. Solution: do a normal parallel sum, then take abs at the end.

*Example (F25 Q3):* Set intersection is both associative and commutative → can use in parallel reduction.

### 5. NumPy broadcasting
Given two arrays with shapes, determine the output shape of an elementwise operation.

Broadcasting rules:
1. Right-align shapes.
2. Left-pad shorter shape with 1s.
3. Each dimension must match, or one of them must be 1.
4. Output shape: take the max of each dimension.

*Example (Re-exam Q10):* a has shape 100×1×6×3, b has shape 100×1×3. Align: a=(100,1,6,3), b=(1,100,1,3). Result: (100,100,6,3).

*Example (F25 Q5):* images is N×H×W×3, mean_pixels is N×3. To subtract per-image mean from all pixels: need mean_pixels[:, None, None] i.e. shape N×1×1×3. Code: `images - mean_pixels[:, None, None]`.

### 6. Cache efficiency and loop ordering
Given an array's strides, determine the optimal loop order. The axis with the **smallest stride** (i.e., the last axis in a row-wise array) should be the **innermost** loop. For CPUs, this maximises spatial locality. For CUDA, threads in a warp access sequential elements along the last axis (coalesced access).

*Example (2024 Q8):* images array strides (600, 40, 8, 200). Smallest stride is axis 2 (stride 8). Inner-most to outer-most: k, j, l, i.

*Example (2024 Q11 vs Q12):* CPU context — channels (k, inner sequential loop) should be last dimension. CUDA context — threads in a block all access k at same time; to coalesce, channels should be the first dimension so spatial axes (i, j) are last.

### 7. Reading profiler output
**cProfile** (function-level): columns are `ncalls`, `tottime` (time in function only), `percall`, `cumtime` (total including sub-calls). To find the bottleneck for a *scaled* workload, multiply `percall` by the expected call count.

**kernprof** (line-level): columns are `Hits`, `Time`, `Per Hit`, `% Time`, `Line Contents`. Use `Hits` to determine how many times the loop ran.

**nsys** (GPU profiler): `gpukernsum` shows kernel execution time. `gpumemtimesum` shows HtoD and DtoH transfer time. `gpumemsizesum` shows transfer sizes. Transfer speed = size / time.

*Example (2024 Q9):* `process_sample` called 10 times with percall=0.505s. For 1000 samples: 1000 × 0.505 = 505s dominates.

*Example (Re-exam Q7):* Lines in for-loop hit 10000 times → n_steps = 10000.

*Example (Re-exam Q8):* Total FLOPs = 5 per iteration × 10000 iterations = 50000. Total time = 15 ms. FLOP/s = 50000 / 0.015 = 3.33 × 10^6.

### 8. GPU kernel thread blocks and coalescing
- Each thread handles one output element. Grid dimensions = ceil(output_dim / block_dim).
- Threads in the same warp (consecutive thread IDs) should access consecutive memory addresses.
- In `cuda.grid(2)` the convention is `(row, col)` where **row = x-dim (first return)** and col = y-dim. Thread ID = threadIdx.x + threadIdx.y × blockDim.x, so adjacent threads differ by 1 in **row** (x-dim), *not* col. The DTU lecture writes `j, i = cuda.grid(2)` to name x as j (the column-direction), which varies across the warp. For coalesced `A[row, col]` access, col (last axis) must be the varying dimension → requires blockDim.x = 1.
- For a 1×256 block (blockDim.x=1, blockDim.y=256): threadIdx.x=0 always → col varies via threadIdx.y → best coalescing for row-major arrays.
- For a 16×16 block: row (x-dim) varies 0–15 per warp half → partial coalescing.
- For a 256×1 block (blockDim.x=256, blockDim.y=1): row varies 0–31 across the warp, col never changes → worst coalescing.

*Example (Re-exam Q11/Q12):* CUDA kernel with `row, col = cuda.grid(2)`, reads `x[row + i, col + j]`. Threads adjacent in warp differ by 1 in row (x-dim). Best performance with 1×256 blocks (blockDim.x=1 → row locked → col varies → all threads share the same row, iterate along columns — sequential access → coalesced).

*Example (F25 Q13):* dmap is 200×200 output, block size 16×16. Blocks needed = ceil(200/16) × ceil(200/16) = 13×13.

### 9. Pandas memory optimisation
Given a DataFrame summary (col name, dtype, #unique, min, max, size), decide how to reduce memory:
- High #unique object column (strings like dates) → convert to `datetime`.
- Low #unique object column (e.g., 8 unique locations) → convert to `category` (uint8 index).
- int64 with small range → downcast to int16/int32/uint8 as appropriate (check min/max fit in target dtype range).
- float64 with acceptable precision loss → float32 (but never convert integer semantics to float).

*Example (2024 Q15):* date (70079 unique, object) → datetime. location (8 unique, object) → category/uint8. mach_id (range -1 to 5730) → int16. units (range 932 to 68837) → uint32 or int32.

### 10. Chunked processing and memory budgeting
Given dtype sizes and available memory, compute maximum chunk size.

bytes per row = sum of (bytes per dtype) for each column.
max rows = available_memory_bytes / bytes_per_row.

*Example (2024 Q17):* Three columns: uint32 (4B), uint64 (8B), float64 (8B) = 20 bytes/row. Memory = 200 MB = 200×10^6 bytes. Max rows = 200×10^6 / 20 = 10^7 ≈ 10 million.

*Example (F25 Q18):* Three int64 columns = 24 bytes/row. 24 MB = 24×10^6 bytes. Max rows = 10^6 = 1 million (option A: 800000 also valid given 1 MB = 1024^2 bytes interpretation).

---

## High-Frequency Topics

These topics appear in **all three exams** or are tested multiple times within a single exam. Master these first.

**1. Amdahl's Law** — Every exam. Know the formula forward and backward. Practice: given S(p), find F; given F and p, find S; find max speed-up; determine if a target is achievable. The re-exam also has a more advanced variant: computing single-processor time from multi-processor time and F, then changing the serial portion.

**2. LSF/BSUB job scripts** — Every exam. Know all flags. Especially: memory is per core (total / n cores = value to put in rusage), GPU queues, job dependencies (`-w ended(name)` vs `-w done(name)`), job arrays, `$LSB_JOBINDEX`.

**3. Cache efficiency and memory layout** — Every exam. Know: row-wise storage means last dimension has smallest stride. Inner loop should iterate over the last dimension. For CUDA, coalescing requires adjacent warp threads (varying in x-dim = row with `row,col=cuda.grid(2)`, or j with lecture's `j,i=cuda.grid(2)`) to access adjacent memory (last array dimension).

**4. GPU memory transfers (CUDA/nsys)** — Every exam. Know: Numba automatically transfers NumPy arrays HtoD before a kernel call and DtoH after. Optimal code pre-allocates on GPU to avoid redundant transfers. Calculate transfer speed from nsys output (size / time). Distinguish kernel time from transfer time.

**5. NumPy broadcasting** — 2024 exam and re-exam both test this. The right-align and pad-with-1s rule. Know how to insert axes with `None` / `np.newaxis` to make broadcasting work as intended.

**6. Profiling output interpretation** — Every exam. cProfile cumtime column for overall bottleneck. Line profiler Hits column to determine input size. Scaling to normal workload: multiply percall × expected_ncalls. nsys for GPU timing breakdown.

**7. Multi-threading vs multi-processing / GIL** — 2024 exam and re-exam. Know: GIL blocks pure Python threads. NumPy and Numba (with nogil=True) release the GIL. Distinguish CPU-bound (needs multi-processing or GIL-free threads) from I/O-bound (threading fine).

**8. Static vs dynamic scheduling** — 2024 exam and F25. Use dynamic when task duration varies significantly (unequal work). Use static when tasks are roughly equal (lower overhead). High stddev in runtime → dynamic. Low stddev → static.

**9. Pandas dtype optimisation and chunking** — 2024 exam and F25. Given a DataFrame summary, choose the right dtype reduction. Compute chunk sizes from memory budget.

**10. Parallel reduction requirements** — 2024 exam and F25. The operation must be associative (and commutative for unordered reduction). Test with a counterexample when unsure.

---

## Per-Exam Breakdown

### 2024 Exam (28.05.2024, 14 pages, 19 questions)

The exam uses a narrative framing: you are a performance consultant visiting different departments of a company. Each department presents a problem. Questions 1–19 follow this story arc.

---

**Q1 — LSF job script correction**
- Topic: BSUB flags, resource specification
- What's asked: Fix a script that requests wrong resources for a program needing 8 cores, 16 GB, 15 min.
- Key insight: Memory in `rusage[mem=XGB]` is per core. 16 GB / 8 cores = 2 GB per core. Add `span[hosts=1]` to keep cores on one node. Change `-n 1` to `-n 8`, `-W 00:05` to `-W 00:15`.
- Answer: `-n 8`, `-W 00:15`, `-R "rusage[mem=2GB]"`, `-R "span[hosts=1]"`

**Q2 — Parallel fraction from speed-up plot (MC)**
- Topic: Amdahl's law
- What's asked: Given a speed-up plot that saturates at 5, pick the parallel fraction.
- Key insight: Max speed-up = 1/(1-F). If saturates at 5 then F = 0.8.
- Answer: (c) 0.8

**Q3 — Should they parallelise? (open-ended)**
- Topic: Amdahl's law applied to a decision
- What's asked: With F=0.8 and max 8 cores, does parallelisation reach speed-up 4?
- Key insight: S(8) = 1/(0.2 + 0.1) = 10/3 ≈ 3.33 < 4. No.
- Answer: No. Show the Amdahl calculation explicitly.

**Q4 — Threading vs multiprocessing (MC)**
- Topic: GIL, Python parallelism
- What's asked: `process_number(n)` is a pure Python loop. Which approach?
- Key insight: Pure Python loops are GIL-bound → threading is blocked → use multiprocessing.
- Answer: (b) Multi-processing

**Q5 — Static vs dynamic scheduling (open-ended)**
- Topic: Scheduling
- What's asked: For `process_number(n)` called on `numbers = [1, 2, ..., 100_000_000]`, static or dynamic?
- Key insight: `process_number(n)` runs a loop of length n. Larger n = much longer. Task times vary enormously → dynamic scheduling to avoid load imbalance.
- Answer: Dynamic. Explain the variable workload.

**Q6 — Parallel reduction validity (open-ended)**
- Topic: Reduction, associativity
- What's asked: Why can `abssum(x,y) = abs(x+y)` not be used in parallel reduction tree?
- Key insight: Not associative. Counterexample: |(1+2)+(-3)| = 0 ≠ |1+(2+(-3))| = 2.
- Answer: Not associative. Instead: do normal parallel sum, then take abs at end.

**Q7 — NumPy broadcasting (MC)**
- Topic: Broadcasting, array shapes
- What's asked: images is N×H×W×3, mim is H×W. Subtract mim from each color channel of each image.
- Key insight: Need to broadcast mim from (H,W) to (N,H,W,3). `mim[:, :, None]` has shape (H,W,1) which broadcasts to (N,H,W,3) when combined. Answer is `images - mim[:, :, None]`.
- Answer: (b) `images - mim[:, :, None]`

**Q8 — Loop reordering from strides (open-ended)**
- Topic: Cache efficiency, memory layout
- What's asked: `images` strides are (600, 40, 8, 200). Reorder loops in `process(images)` for cache efficiency.
- Key insight: Smallest stride = axis 2 (stride 8, index k). Innermost loop should be axis 2. From inner to outer: k (stride 8), j (stride 40), l (stride 200), i (stride 600). Wait — axis 3 has stride 200 which is larger than axis 2's stride 8. So ordered by increasing stride: axis 2 (8) < axis 1 (40) < axis 3 (200) < axis 0 (600). Inner to outer: k, j, l, i.
- Answer: Loop order innermost to outermost: k, j, l, i. Explain that shortest stride = innermost = most cache-friendly.

**Q9 — Profiler: number of samples (MC)**
- Topic: Reading cProfile output
- What's asked: `process_sample` called 10 times. How many samples were in the subset?
- Key insight: ncalls for process_sample is 10. Each sample → 1 call → 10 samples.
- Answer: (c) 10

**Q10 — Profiler: where to optimise (open-ended)**
- Topic: Profiling, scaling analysis
- What's asked: For normal workloads (1000 samples), which function to optimise?
- Key insight: `process_sample` has percall = 0.505s and ncalls scales with samples. At 1000 samples: 1000 × 0.505 = 505s. `prepare_model` (15s) and `save` (3s) and `load_data` (1s) do not scale with sample count. So `process_sample` dominates.
- Answer: Focus on `process_sample`. Show the scaling calculation.

**Q11 — CPU cache efficiency for array layout (open-ended)**
- Topic: Cache, parallelism, inner loop axis
- What's asked: `conv_channels` parallel CPU function with inner loop over k (channels). Should image be c×h×w or h×w×c?
- Key insight: Inner sequential loop is k (channels). For cache efficiency, sequential loop should iterate over the last (shortest stride) axis. So channels should be the last dimension: shape h×w×c.
- Answer: h×w×c (channels last). Explain: last axis = smallest stride = sequential memory access = fewest cache misses.

**Q12 — CUDA cache efficiency for array layout (open-ended)**
- Topic: CUDA coalescing, warp behaviour
- What's asked: Same convolution now as CUDA kernel with 32×32 thread blocks. How should image be stored?
- Key insight: In CUDA, threads in a warp (varying along j dimension of the 2D grid) all execute simultaneously. They all access the same k (channel) at a given moment. For coalesced access, values of each channel for different pixels should be contiguous in memory. That means spatial axes (h, w) should be last. So channels first: c×h×w.
- Answer: c×h×w (channels first). Explain warp coalescing: adjacent threads in warp share same k, need adjacent pixels in memory.

**Q13 — GPU transfer speed from nsys (MC)**
- Topic: nsys profiler, memory transfer bandwidth
- What's asked: HtoD transfers: 2.5s total, 25000 MB total size. Transfer speed?
- Key insight: 25000 MB / 2.5s = 10000 MB/s = 10 GB/s.
- Answer: (b) Around 10 GB/s

**Q14 — GPU vs CPU speed comparison (open-ended)**
- Topic: GPU pipeline performance, total time calculation
- What's asked: GPU pipeline total time vs CPU time (7s)?
- Key insight: GPU total = kernel time + HtoD + DtoH = 0.5 + 2.5 + 0.5 = 3.5s. CPU = 7s. GPU is 2× faster.
- Answer: 2× faster. Show the calculation: 0.5 + 2.5 + 0.5 = 3.5s vs 7s.

**Q15 — DataFrame dtype reduction (open-ended)**
- Topic: Pandas memory optimisation
- What's asked: Recode 4 columns (date/object, location/object, mach_id/int64, units/int64).
- Key insight: date → datetime (strings waste space). location (8 unique) → category with uint8 index. mach_id (range -1 to 5730) → int16 (fits in ±32767). units (range 932 to 68837) → uint32 or int32.
- Answer: Explain each column's reasoning. Get full credit by stating the target dtype and why it fits.

**Q16 — Pandas indexing for speed (open-ended)**
- Topic: Pandas query optimisation
- What's asked: Common operation is extract rows by date then compute statistics. Currently slow. How to speed it up?
- Key insight: Set the date column as the DataFrame index and sort it. This converts row-by-date lookups from O(n) scan to O(log n) binary search. Worth the upfront cost since operations are repeated many times.
- Answer: Set date as sorted index. Explain the speed benefit for repeated lookups.

**Q17 — Maximum chunk size calculation (MC)**
- Topic: Chunked processing, memory budget
- What's asked: DataFrame with 3 columns (uint32, uint64, float64). Memory = 200 MB. Max rows per chunk?
- Key insight: Bytes per row = 4 + 8 + 8 = 20. 200 MB = 200×10^6 bytes. Rows = 200×10^6 / 20 = 10^7 = 10 million.
- Answer: (b) Around 10 000 000

**Q18 — Job dependency for job array completion (open-ended)**
- Topic: LSF job dependencies
- What's asked: A follow-up job must run once all jobs in array `power[1-12]` have finished (regardless of success/failure).
- Key insight: `-w done(power)` waits for successful completion. `-w ended(power)` waits for termination regardless of exit status.
- Answer: Add `#BSUB -w ended(power)` to the dependent job's script.

**Q19 — Parallelisation strategy for simulation (open-ended)**
- Topic: Parallelisation, GIL, expected speed-up
- What's asked: `simulate_single` uses Numba with `nogil=True`. `simulate` calls it m times (independent). How/where to parallelise?
- Key insight: Cannot parallelise inside `simulate_single` (each step depends on previous). Can parallelise over the m calls in `simulate`. Since `simulate_single` releases the GIL, use **multi-threading** (lighter weight than multiprocessing). Speed-up up to m times (one thread per initial value). Speed-up does not depend on n.
- Answer: Parallelise the outer loop in `simulate` over j. Use threading (GIL released). Speed-up ≈ min(m, num_cores). Explain why `simulate_single` cannot be parallelised.

---

### 2024 Re-exam (21.08.2024, 10 pages, 18 questions)

Same narrative framing (performance consultant). More condensed — 10 pages for 18 questions, some questions build on each other numerically.

---

**Q1 — Read job script resources (open-ended)**
- Topic: BSUB flags
- What's asked: Script has `-W 02:00`, `-R "rusage[mem=4GB]"`, `-n 8`. What resources are requested?
- Key insight: 2 hours wall time. 8 cores. 4 GB per core = 32 GB total memory.
- Answer: 2h wall time, 8 cores, 32 GB total memory. State the per-core vs total distinction explicitly.

**Q2 — Modify job script for GPU (open-ended)**
- Topic: LSF GPU configuration
- What's asked: Change script so it runs on a GPU node.
- Key insight: Change queue `-q hpc` to a GPU queue: `gpuv100`, `gpua100`, or `hpcintgpu`. Add `-gpu "num=1:mode=exclusive_process"`.
- Answer: Change queue name. Add GPU resource flag.

**Q3 — Single-processor time from multi-processor (open-ended)**
- Topic: Amdahl's law (inverse)
- What's asked: F=0.8, runs on 4 procs in 10 min. How long on 1 proc?
- Key insight: S(4) = 1/(0.2 + 0.8/4) = 1/(0.2 + 0.2) = 2.5. Time on 1 proc = 10 × 2.5 = 25 min.
- Answer: 25 minutes. Show the Amdahl calculation.

**Q4 — Effect of reducing serial part (open-ended)**
- Topic: Amdahl's law, serial time manipulation
- What's asked: If non-parallelisable part reduced by 3 min, new time on 4 procs?
- Key insight: T_serial + T_parallel/4 = 10. Reducing T_serial by 3 gives 10 - 3 = 7 min. (The parallel part on 4 procs does not change.)
- Answer: 7 minutes. Explain that only T_serial changed.

**Q5 — Zarr block shape for column access (MC)**
- Topic: Zarr chunked arrays, I/O performance
- What's asked: Code reads entire columns of a 1000×100000 float64 matrix. Best block shape?
- Key insight: The code accesses `x[:, i]` — entire columns. Zarr reads blocks from disk. A block that contains an entire column minimises block reads per column access. Shape 1000×100 means each block covers all 1000 rows for 100 columns → one block per column access → best.
- Answer: (c) 1000×100. Explain: each column access loads exactly one block.

**Q6 — Zarr block memory size (open-ended)**
- Topic: Memory calculation
- What's asked: How much memory does one block of shape 1000×100 of float64 take?
- Key insight: 100000 elements × 8 bytes = 800000 bytes ≈ 800 KB.
- Answer: 800 KB. Show: 1000 × 100 × 8 bytes.

**Q7 — Line profiler: n_steps value (open-ended)**
- Topic: Line profiler reading
- What's asked: Loop lines hit 10000 times. What is n_steps?
- Key insight: Lines inside the loop are hit once per iteration. 10000 hits = 10000 iterations = n_steps = 10000.
- Answer: 10000. State the reasoning from Hits column.

**Q8 — FLOP/s from line profiler (MC)**
- Topic: Performance analysis, floating point operations
- What's asked: Function performs 5 FLOPs per iteration, 10000 iterations. Total time 15 ms. FLOP/s?
- Key insight: Total FLOPs = 5 × 10000 = 50000. Time = 15ms = 0.015s. FLOP/s = 50000/0.015 = 3.33×10^6.
- Answer: (b) 3.33×10^6 FLOP/s

**Q9 — NumPy expression equivalent to simulate function (MC)**
- Topic: NumPy vectorisation
- What's asked: `simulate` computes sum of (x[i]*x[i]+4) / (y[n-i-1]/x[i]) over i. Which NumPy expression?
- Key insight: y is accessed in reverse order: y[n-1], y[n-2], ..., y[0], i.e., `y[::-1]`. The full expression is `np.sum((x*x + 4) / (y[::-1] / x))`.
- Answer: (c) `np.sum((x * x + 4) / (y[::-1] / x))`

**Q10 — Broadcasting shape result (MC)**
- Topic: NumPy broadcasting
- What's asked: a is 100×1×6×3, b is 100×1×3. What is shape of a + b?
- Key insight: Right-align: a=(100,1,6,3), b=(100,1,3). Pad b: b=(1,100,1,3). Broadcast: (100,100,6,3).
- Answer: (a) 100×100×6×3

**Q11 — Best CUDA thread block configuration (MC)**
- Topic: CUDA thread blocks, warp coalescing
- What's asked: `average3x3` kernel: `row, col = cuda.grid(2)`. Accesses `x[row+i, col+j]`. Best block: 16×16, 256×1, or 1×256?
- Key insight: Threads in a warp differ by 1 in the first grid dimension (row, x-dim). For coalesced `x[row, col]` access, col (last axis) must vary — not row. Block (1, 256) achieves this: blockDim.x=1 → threadIdx.x=0 always → row is locked → col varies via threadIdx.y → all threads share the same row with different cols → adjacent elements along last axis → coalesced.
- Answer: (c) 1×256

**Q12 — Explain reasoning for Q11 (open-ended)**
- Topic: CUDA warp coalescing explanation
- What's asked: Explain why 1×256 is best.
- Key insight: With 1×256 (blockDim.x=1), threadIdx.x=0 always → row is fixed per warp → col varies. All threads share the same row and differ in col, so their accesses to x[row+i, col+j] walk along a row (sequential in memory) → coalesced. For 16×16: row varies 0–15 in the warp (same col for first 16 threads) → different rows → stride = array width → non-coalesced. For 256×1: row varies 0–255, col=fixed → worst strided access.
- Answer: With 1×256, warp threads share same row, access sequential columns → coalesced. For 16×16 or 256×1, threads have different rows → strided (non-sequential) memory access → worse.

**Q13 — HtoD / DtoH transfer counts (open-ended)**
- Topic: Numba GPU memory management
- What's asked: `sumavg` iterates over 100 images, calling a CUDA kernel each time with NumPy arrays. How many transfers?
- Key insight: Numba automatically transfers both x (input) and y (output) HtoD before each call and DtoH after. 100 iterations × 2 HtoD + 100 iterations × 2 DtoH = 200 HtoD + 200 DtoH.
- Answer: 200 HtoD, 200 DtoH.

**Q14 — Optimal transfer count (open-ended)**
- Topic: GPU memory optimisation
- What's asked: Minimum transfers for `sumavg`?
- Key insight: x_all can be transferred one image at a time: 100 HtoD. y can be allocated on GPU and only transferred back once: 1 DtoH. Total: 101 transfers.
- Answer: 100 HtoD (one per image), 1 DtoH (final result). Total 101.

**Q15 — Job dependency with failed jobs (MC)**
- Topic: LSF `-w done` vs `-w ended`
- What's asked: `compute` has `-w done(prepare)`. bjobs shows 1 job in EXIT state. When does compute start?
- Key insight: `-w done(prepare)` requires ALL jobs in the array to complete SUCCESSFULLY. One job is in EXIT (failure) state. The condition will never be met. Compute will never start.
- Answer: (c) Never. Explain: `done` requires all jobs successful; one has exited with failure.

**Q16 — Parallel row sum vs column sum (open-ended)**
- Topic: Cache efficiency with parallelism
- What's asked: Parallel row sum vs parallel column sum on a row-wise matrix. Which is faster?
- Key insight: Parallel row sum: each thread calls `a.sum()` on a row of x. Each row is contiguous in memory → sequential access → cache-efficient. Parallel column sum: each thread sums a column of x.T (which is a row of x.T, i.e., a column of x). Columns are non-contiguous in row-wise storage → strided access → many cache misses.
- Answer: Parallel row sum is faster. Explain row-wise storage and strided column access.

**Q17 — Multi-threading appropriateness (open-ended)**
- Topic: GIL, NumPy, threading
- What's asked: The parallel row/column sum uses NumPy's `.sum()`. Is multi-threading appropriate?
- Key insight: NumPy releases the GIL during computation. Therefore threading is fine and avoids process spawn overhead.
- Answer: Yes, appropriate. NumPy releases GIL, so multiple threads can truly run in parallel.

**Q18 — Parallel reduction speed-up vs parallel row sum (open-ended)**
- Topic: Reduction tree complexity analysis
- What's asked: Approach from Q16 uses time 2n (n for row sums in parallel, n for summing row sums serially). Parallel reduction on n^2 elements uses time log2(n^2) = 2·log2(n). Speed-up?
- Key insight: Speed-up = 2n / (2·log2(n)) = n / log2(n).
- Answer: Speed-up is n / log2(n). Show the complexity analysis: Q16 approach takes O(n) with unlimited cores; reduction takes O(log n^2) = O(log n).

---

### F25 Exam (2025, 18 pages, 24 questions — all multiple choice)

The F25 exam is entirely multiple choice with 4 options each. The answer key is embedded in the same document (printed in red). No narrative framing — each question is self-contained.

---

**Q1 — LSF memory per core calculation (MC)**
- Topic: BSUB rusage memory semantics
- What's asked: Script uses 4 cores and program needs 100 GB total. What value for `rusage[mem=???GB]`?
- Key insight: rusage memory is per core. 100 GB / 4 cores = 25 GB.
- Answer: A) 25GB

**Q2 — float16 precision / arithmetic (MC)**
- Topic: Floating point representation, precision limits
- What's asked: float16 resolution 0.001, max 6.55×10^4. What does 10000 + 1 print?
- Key insight: float16 resolution is relative. At value 10000 = 1e4, the smallest representable increment is 10000 × 0.001 = 10, meaning 10001 cannot be represented. Rounds to 10000.
- Answer: B) 10000

**Q3 — Set intersection in parallel reduction (MC)**
- Topic: Reduction requirements (associativity, commutativity)
- What's asked: Can set intersection be used in parallel reduction?
- Key insight: Set intersection is both commutative (A∩B = B∩A) and associative ((A∩B)∩C = A∩(B∩C)) → yes.
- Answer: A) Yes

**Q4 — NumPy reshape and indexing (MC)**
- Topic: Row-wise storage, reshape
- What's asked: 3×5 row-wise array a. What is `a.reshape(-1)[8]`?
- Key insight: reshape(-1) flattens row-wise to [1,5,43,51,32,73,2,4,67,37,9,3,54,8,22]. Index 8 is 67.
- Answer: A) 67

**Q5 — NumPy broadcasting for image subtraction (MC)**
- Topic: Broadcasting with `None` indexing
- What's asked: images N×H×W×3, mean_pixels N×3. Subtract per-image mean from all pixels.
- Key insight: mean_pixels[:, None, None] has shape N×1×1×3 which broadcasts to N×H×W×3.
- Answer: A) `images - mean_pixels[:, None, None]`

**Q6 — cProfile: which function takes most overall time (MC)**
- Topic: Reading cProfile output
- What's asked: From profiler output, which function has highest cumtime?
- Key insight: cumtime for render_scene = 8.841s, advance_scene = 4.845s, save_to_mp4 = 2.405s, load_scene = 1.005s. Highest is render_scene.
- Answer: A) render_scene

**Q7 — Parallelisable loop identification (MC)**
- Topic: Data dependencies, parallelism candidates
- What's asked: Script has two loops: loop 1 updates `scene` (each iteration depends on previous), loop 2 calls `render_scene(scene)` for independent scenes. Which can be parallelised?
- Key insight: Loop 1: scene = R.advance_scene(scene, dt) — sequential dependency. Loop 2: frame = R.render_scene(scene) — each iteration independent.
- Answer: B) The second for-loop (render_scene)

**Q8 — Amdahl's law: solve for F and max speed-up (MC)**
- Topic: Amdahl's law
- What's asked: Measured S(3) = 2.5. What is theoretical max speed-up?
- Key insight: F = 3(1 - 1/2.5)/(3-1) = 3(0.4)/2 = 0.9. Max = 1/(1-0.9) = 10.
- Answer: C) Around 10x

**Q9 — `time` command: wall time vs CPU time in parallel (MC)**
- Topic: Measuring parallelism with `time`
- What's asked: Single-threaded: real=12s, user=12s, sys=0.034s. Run on 2 cores. Expected output?
- Key insight: Wall time (real) halves: ~6s. CPU time (user) stays same or increases (summed over cores): still ~12s. sys stays same.
- Answer: B) real=6s, user=12s, sys=0.034s

**Q10 — Static vs dynamic scheduling from kernel stddev (MC)**
- Topic: Scheduling strategy, kernel runtime variance
- What's asked: Two kernels on 4 GPUs. kernel1 has stddev=40ms (high), kernel2 has stddev=0.05ms (low). Which needs dynamic scheduling?
- Key insight: High stddev → variable runtime → static scheduling leads to imbalance → use dynamic. Low stddev → uniform → static fine.
- Answer: B) kernel1 should use dynamic scheduling; kernel2 stick with static.

**Q11 — Line profiler: scale to normal workload (MC)**
- Topic: Line profiler, scaling estimate
- What's asked: Profiled with 1000 entries. Normal workload = 10000 entries. `prep_conds` takes 2s (fixed). Loop takes (740+1266748+1685) μs × 1000 = 1.269s. At 10000: 12.69s. Total = 12.69 + 2.005 = 14.70s.
- Answer: A) Around 14.7 seconds

**Q12 — CUDA kernel: worst memory access pattern (MC)**
- Topic: CUDA memory coalescing, ray marching
- What's asked: `render_depthmap` kernel reads vol[vi,vj,vk] where steps change with ray_step. Which u_step/v_step/ray_step gives worst memory efficiency?
- Key insight: vol is row-wise, last axis (vk) has smallest stride. Adjacent warp threads differ in the x-dim (column direction of the output, j with lecture convention). For coalesced access, adjacent threads should read adjacent vk. This happens when v_step changes vk (i.e., v_step=[0,0,1]). Worst case: v_step and u_step don't vary vk → steps go along the first two (high-stride) axes.
- Answer: A) u_step [1,0,0], v_step [0,1,0], ray_step [0,0,1]

**Q13 — Number of thread blocks for output image (MC)**
- Topic: CUDA grid dimensions
- What's asked: Output dmap 200×200, thread blocks 16×16. How many blocks needed?
- Key insight: ceil(200/16) = ceil(12.5) = 13 in each dimension. 13×13 blocks.
- Answer: A) 13×13

**Q14 — nsys profiler: which part takes most time (MC)**
- Topic: GPU profiling, nsys output
- What's asked: Kernel=13.8ms, HtoD=26.8ms total, DtoH=0.16ms total. Which takes most time?
- Key insight: HtoD transfers (26.8ms) > kernel (13.8ms) > DtoH (0.16ms).
- Answer: B) Host to device transfers

**Q15 — Numba CUDA array transfers with NumPy inputs (MC)**
- Topic: Numba automatic memory transfers
- What's asked: `square[grid, block](y, x)` called with NumPy arrays. How many HtoD and DtoH? What is optimal?
- Key insight: Numba transfers both x and y HtoD and both DtoH: 2 HtoD + 2 DtoH. Optimally: only x needs HtoD (input), only y needs DtoH (output): 1 HtoD + 1 DtoH.
- Answer: A) Does 2 HtoD + 2 DtoH, but only 1 HtoD + 1 DtoH necessary.

**Q16 — Random access and memory hierarchy (MC)**
- Topic: Cache and memory hierarchy
- What's asked: Program accesses a large data structure randomly. What causes low performance?
- Key insight: Random access means cache lines loaded for one element are not reused for the next access. The data is too large to fit in cache, so every access goes to main memory (DRAM) — a slower level of the memory hierarchy.
- Answer: A) A slower part of the memory hierarchy is used.

**Q17 — DataFrame version column dtype reduction (MC)**
- Topic: Pandas dtype optimisation
- What's asked: `version` column is int64 with min=0, max=42. Best reduction?
- Key insight: Values 0–42 fit in uint8 (0–255) or int8 (-128–127). It is numeric (not datetime). Integer, so not float (rounding errors). Smaller integer type is correct.
- Answer: C) Convert to a smaller integer type

**Q18 — Maximum chunk size for 24 MB RAM (MC)**
- Topic: Chunked processing
- What's asked: DataFrame with 3 int64 columns. 24 MB RAM. Max chunk size?
- Key insight: 3 columns × 8 bytes = 24 bytes/row. 24 MB = 24×10^6 bytes. Rows = 10^6 = 1,000,000. With 1 MB = 1024^2: 24 × 1024^2 / 24 = 1,048,576. Closest option is A) 800000 (valid conservative answer).
- Answer: A) 800000

**Q19 — np.memmap memory footprint (MC)**
- Topic: Memory mapping, strided access
- What's asked: `np.memmap` of 10^10 uint8 elements. `x[::100_000]` copies to y. Memory requirement?
- Key insight: memmap itself uses no RAM (just maps the file). `y = np.array(x[::100_000])` creates 10^10 / 10^5 = 10^5 elements of uint8 = 100,000 bytes ≈ 100 KB.
- Answer: D) Around 100 KB

**Q20 — Zarr chunk shape for row-sum access pattern (MC)**
- Topic: Zarr performance, access patterns
- What's asked: `process(a)` computes `s[i] = np.sum(a[i])` for each row. Best Zarr chunk shape for 1024×1024 float64 array?
- Key insight: Each iteration accesses an entire row. A chunk of shape (1, 1024) = one full row per chunk minimises block loads per iteration (1 block per row). Shape (1024, 1) requires 1024 blocks per row.
- Answer: A) (1, 1024)

**Q21 — N-body force calculation: which statement is NOT correct (MC)**
- Topic: Numba, NumPy, loop ordering, cache
- What's asked: Which of (A) Numba @jit helps, (B) loops can be parallelised, (C) NumPy can improve, (D) loop order swap has no impact — is NOT correct?
- Key insight: forces array is row-wise. Outer loop is over i (rows), inner over j. Row-major order means row access is sequential → cache-efficient. Swapping to outer j / inner i means column access → strided, cache-inefficient. So D is NOT correct.
- Answer: D) The order of the loops can be switched with no impact.

**Q22 — Simulate_ball parallelisation strategy (MC)**
- Topic: Parallelisation strategy, GIL, dynamic scheduling, variable run times
- What's asked: `simulate_one_time_step` is sequential Python. Simulations take variable time (mins to hours). Best approach?
- Key insight: Cannot parallelise within simulate_ball (sequential). Can parallelise across different simulate_ball calls. Variable time → dynamic scheduling. Python code → GIL issue → multi-processing (not threading). GPU not suited (Python simulation code). 
- Answer: B) Multiprocessing with dynamic scheduling.

**Q23 — GPU vs CPU: benchmark over few vs many iterations (MC)**
- Topic: GPU amortisation of transfer overhead
- What's asked: CPU: 0.1s/iter. GPU: 0.25s/iter + 0.6s constant overhead. Benchmark with few iterations showed GPU slower. Advice?
- Key insight: GPU has fixed overhead (0.6s transfers) amortised over iterations. For millions of iterations: CPU total ≈ 0.1 × millions; GPU total ≈ 0.25 × millions + 0.6 ≈ 0.25 × millions (overhead negligible). Wait — per-iteration GPU is 0.25s vs CPU 0.1s. GPU is still slower per iteration. However the question states GPU = 0.85s for 5 iter, 1.1s for 10 iter. Per-iter GPU cost = (1.1-0.85)/(10-5) = 0.05s. Overhead = 0.85 - 5×0.05 = 0.6s. CPU = 0.1s/iter. GPU per-iter = 0.05s/iter → GPU is 2× faster per iteration once overhead is accounted for. At millions of iterations, GPU wins.
- Answer: D) The current GPU implementation already has better performance, and they will see that by running more iterations.

**Q24 — Amdahl's law with unknown serial fraction (MC)**
- Topic: Amdahl's law, profiling-based parallelisation limits
- What's asked: Program = 32s total. One function = 16s and is parallelisable. Can we reach 8s?
- Key insight: The remaining 16s is unknown — could be partly parallelisable. Best case: parallelize the known 16s completely (→0) plus the unknown 16s completely (→0) = 0s theoretical. But we only know the 16s parallelisable part. We cannot say if 8s target is reachable without knowing if the other 16s can also be parallelised.
- Answer: B) The program can possibly be made to run faster than 8 seconds, but further analysis is needed.

---

## Common Traps and Mistakes

**1. rusage memory is per core, not total.**
Every exam tests this. Students who write 16 GB instead of 2 GB per core when 8 cores are requested get this wrong. Always divide total memory by number of cores.

**2. `-w done(name)` vs `-w ended(name)`.**
`done` requires successful completion of ALL jobs. If even one job fails (EXIT state), the dependent job will never start. Use `ended` when you want to run regardless of individual job success. The re-exam Q15 directly exploits this: one job has exited, so `done` condition is permanently false.

**3. Confusing max speed-up (1/(1-F)) with speed-up at p processors.**
Max speed-up as p→∞ is 1/(1-F). Speed-up at specific p is 1/(1-F + F/p). Many students use the wrong formula. The F25 Q8 and Q24 test the distinction.

**4. Not showing reasoning on open-ended questions.**
The exam explicitly states unexplained answers get substantially reduced credit. A correct answer without a formula derivation or counterexample will lose points.

**5. Associativity test for parallel reduction.**
Students check commutativity but forget associativity. Commutativity is not sufficient. `abs(x+y)` is commutative (abs(x+y) = abs(y+x)) but not associative — and that is why it fails. Always test with a concrete counterexample.

**6. Cache ordering: inner loop should be smallest stride, not largest.**
Intuition sometimes tells students to put the longest-running dimension innermost. The correct rule is smallest stride innermost (most cache-friendly sequential access). With strides like (600, 40, 8, 200), the smallest is 8 (axis 2), not the last axis (axis 3 has stride 200).

**7. CPU cache vs GPU cache in layout questions.**
2024 Q11 and Q12 are back-to-back but have opposite correct answers for array layout. CPU: sequential inner loop → last dimension (channels last). CUDA: threads in warp access same channel simultaneously → channels first. Students who memorise one rule and apply it to both get Q12 wrong.

**8. Numba automatic transfers — direction and count.**
Numba transfers every NumPy argument HtoD before a kernel and DtoH after, regardless of whether it is input or output. The `square[grid, block](y, x)` call transfers BOTH x and y in both directions: 2 HtoD + 2 DtoH. Only 1 HtoD + 1 DtoH is optimal. Students often undercount.

**9. float16 precision — relative, not absolute.**
The resolution 0.001 is a relative resolution (similar to epsilon). At a value of 10000, the absolute precision is 10000 × 0.001 = 10, not 0.001. 10000 + 1 cannot be distinguished from 10000.

**10. `time` command: wall time vs CPU time.**
real = wall clock time (goes down with parallelism). user + sys = CPU time (stays constant or increases with parallelism since CPU time is summed over cores). Students confuse these and predict user time halving.

**11. Zarr blocks for column vs row access.**
When code accesses entire columns (`x[:, i]`), the optimal block shape spans all rows but few columns (e.g., 1000×100 for a 1000×100000 matrix). Students often pick the "squarish" or default block shape.

**12. `done` dependency requires ALL jobs to succeed.**
Related to trap 2 but more subtle: the re-exam shows `bjobs -A` where 4 jobs are running, 5 are done, and 1 has exited. Students might think the dependency waits for running jobs to finish. The critical insight is the EXIT state permanently blocks a `done` condition.

---

## What the Exam Rewards

**Precise formulas with numbers substituted in.** For Amdahl's law questions, writing out S(p) = 1/(1-F+F/p) with values substituted and showing the arithmetic gets full credit. A correct answer without the formula gets reduced credit.

**Reasoning through concrete examples.** For reduction questions, providing a specific counterexample (e.g., ||(1+2)+(-3)|| = 0 ≠ 2) is the expected answer format. Abstract statements without examples are insufficiently convincing.

**Understanding trade-offs.** Questions like "static vs dynamic scheduling" and "multi-threading vs multi-processing" reward answers that acknowledge both options and explain why one is preferred given the specific constraints (variable task time, GIL releasing, etc.).

**Correct identification of the bottleneck for the actual workload.** In profiling questions, you must scale from the profiling subset to the normal workload. Just identifying the largest cumtime in the profiler output is insufficient — you need to project it to production scale (e.g., ×100 samples).

**Careful reading of profiler column semantics.** The exam tests whether you know the difference between `tottime` (own time only) and `cumtime` (inclusive of sub-calls), and between `percall` for ncalls vs primitive calls.

**Correctly counting GPU transfers, not just direction.** Both the count (how many times) and the direction (HtoD vs DtoH) matter. The optimal count also matters (e.g., 100+1 vs 200+200).

**Practical engineering judgment.** Questions like Q24 (F25) reward saying "we don't know enough" when that is the honest answer, rather than confidently applying Amdahl to only part of the program.

---

## Exam Preparation Strategy

### Week-by-week priority

**Priority 1: Amdahl's law (all three exams)**
Drill: given S(p) find F; given F find S(p) for arbitrary p; find max speed-up; determine if a target speed-up is achievable with finite cores; inverse: given time on p cores and F, find time on 1 core; partial speedup: reducing only the serial portion.

Practice the formula in all directions. Write it out 20 times from memory. Know the max speed-up formula 1/(1-F) without thinking.

**Priority 2: LSF/BSUB job scripts (all three exams)**
Memorise every common flag. Build a mental "cheat sheet":
- `-n X`: X cores
- `-W HH:MM`: wall time
- `-R "rusage[mem=XGB]"`: X GB per core (total = X × n)
- `-R "span[hosts=1]"`: all cores on one node
- `-q gpuv100` / `-gpu "num=1:mode=exclusive_process"`: GPU node
- `-J name[1-N]` + `$LSB_JOBINDEX`: job array
- `-w done(name)`: wait for successful completion
- `-w ended(name)`: wait for any termination

**Priority 3: Cache / memory layout (all three exams)**
Two rules to drill:
- CPU: loop over last dimension (smallest stride) in innermost loop.
- CUDA: threads in warp vary in x-dim (= j with lecture's `j,i=cuda.grid(2)`, = row with `row,col=cuda.grid(2)`). The varying index must be the **last** array dimension for coalesced access. Lecture convention: `j, i = cuda.grid(2)` → `A[i, j]` (j last) → coalesced. Exam convention: `row, col = cuda.grid(2)` → `A[col, row]` (row last) → coalesced. Do NOT use `A[row, col]` with the exam convention — row varies (first index) → strided.

Practice: given strides, sort axes by stride and order loops accordingly.

**Priority 4: GPU memory transfers (all three exams)**
Practice: for a kernel called with k NumPy arrays, how many automatic HtoD and DtoH transfers? Answer: k HtoD + k DtoH. What is optimal? Inputs: 1 HtoD each. Outputs: 1 DtoH each. Input-only arrays: no DtoH. Output-only arrays: no HtoD (just allocate on device). Calculate from nsys: total size / total time = bandwidth.

**Priority 5: Profiler output (all three exams)**
Practice reading both cProfile and line profiler output:
- cProfile: cumtime for overall cost, ncalls for call count, percall for cost per call.
- line profiler: Hits tells you how many iterations.
- Scaling: percall × expected_ncalls gives projected cost at normal workload.
- FLOP/s: count FLOPs per loop iteration × iterations / total time.

**Priority 6: Broadcasting (2024 + re-exam)**
Practice the right-align and pad-with-1s algorithm on at least 10 shape pairs. Particularly: when does a reshape to add a `None` axis make broadcasting work for image-mean subtraction scenarios? This comes up in the same form across multiple exams.

**Priority 7: Scheduling and GIL decisions**
For any piece of code:
1. Is it CPU-bound or I/O-bound?
2. Does it release the GIL (NumPy, Numba nogil=True)?
3. Are task times uniform or variable?

Answer gives: threading vs multiprocessing, static vs dynamic.

**Priority 8: Pandas dtype optimisation**
Practice on DataFrame summary tables: for each column, identify target dtype by checking min/max against dtype ranges (int8: -128 to 127, uint8: 0 to 255, int16: -32768 to 32767, int32: ±2.1×10^9, float32: ~7 decimal digits). Know: object strings → datetime or category. Low cardinality → category.

**Priority 9: Chunked processing arithmetic**
Practice: given N columns of dtype with dtype_size bytes, compute bytes per row, then max rows = available_bytes / bytes_per_row.

**Priority 10: Parallel reduction correctness**
Know: must be associative. Test with: does (a op b) op c == a op (b op c) for a concrete counterexample? Commutative is a bonus but not required. Know the correct alternative when the naive operation fails (e.g., do regular sum then abs).

### Exam-day tactics

- For multiple-choice questions in the free-response exams: still write 1-2 sentences of reasoning. You can lose credit on MC questions if you mark without justification on a non-MC exam.
- For F25-style pure MC: eliminate wrong answers first, use your formula sheet.
- On open-ended questions: if you are not sure, state your assumption, compute the answer under that assumption, and note the assumption explicitly. Partial credit is awarded.
- Amdahl calculations: always present the formula, substitute, and evaluate. Never just write the number.
- GPU questions: draw a quick mental picture of the thread block grid to count transfers or identify which threads are in the same warp.
- Memory layout questions: write out the strides explicitly before concluding which dimension is innermost.

### What is never tested (based on these three exams)

- Writing a complete parallelised program from scratch.
- Specific Python syntax beyond simple snippets.
- Detailed CUDA shared memory optimisation (only coalescing is tested).
- MPI.
- Specific Zarr/pandas API calls beyond the access patterns shown.
