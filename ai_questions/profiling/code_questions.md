# Profiling — Code-Based MCQ Practice

> Format: Each question shows cProfile or line_profiler output to interpret.
> Exam frequency: **Every exam**.

---

## Q1 — cProfile: Bottleneck at Scale

A pipeline is profiled with 10 samples. The output below is from that run. If the pipeline is deployed with **1000 samples**, which function is the bottleneck?

```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     1    0.001    0.001   15.532   15.532 main (main.py:1)
    10    0.002    0.000   15.524    1.552 process_batch (pipeline.py:12)
    10    0.010    0.001   14.982    1.498 run_model (model.py:5)
    10    9.850    0.985    9.850    0.985 compute_features (features.py:8)
     1    2.100    2.100    2.100    2.100 load_data (io.py:3)
```

- A) `load_data` — it has the highest `percall` among fixed-cost functions
- B) `run_model` — it has the highest `cumtime` per call
- C) `compute_features` — it scales with samples and dominates at 1000 samples
- D) `process_batch` — it wraps everything and has the highest `cumtime`

**Answer: C**

- A) Incorrect — `load_data` is called only once (ncalls=1), so its cost stays fixed at 2.1s regardless of sample count
- B) Incorrect — `run_model`'s own tottime is only 0.010s; its cumtime is inflated by calling `compute_features`
- C) Correct — `compute_features` has percall=0.985s and ncalls scales with samples; at 1000 samples: 0.985 × 1000 = 985s
- D) Incorrect — `process_batch` is a wrapper whose cumtime comes from calling sub-functions, not its own work

---

## Q2 — cProfile: tottime vs cumtime

The cProfile output below is from a model inference run:

```
ncalls  tottime  cumtime  percall(cum)  filename:lineno(function)
    10    0.010   14.982         1.498  run_model (model.py:5)
    10    9.850    9.850         0.985  compute_features (features.py:8)
    10    4.900    4.900         0.490  matrix_multiply (linalg.py:22)
    10    0.220    0.220         0.022  normalise (preprocess.py:7)
```

What does `tottime = 0.010` for `run_model` tell you?

- A) `run_model` spent 0.010s in total across all 10 calls, including time in sub-functions
- B) `run_model`'s own code (excluding time inside callees) took 0.010s across all 10 calls
- C) Each call to `run_model` took 0.010s end-to-end
- D) `run_model` was the fastest function in the call graph

**Answer: B**

- A) Incorrect — that description matches cumtime, not tottime; tottime explicitly excludes time in callees
- B) Correct — tottime counts only the lines executing inside run_model itself; its 14.982s cumtime comes from callees like compute_features and matrix_multiply
- C) Incorrect — 0.010s is the total across all 10 calls; per-call own-code time would be 0.001s, and end-to-end per call is cumtime/ncalls = 1.498s
- D) Incorrect — tottime comparison is not a valid measure of "fastest"; normalise has higher tottime (0.220s) but run_model's own logic is minimal

---

## Q3 — line_profiler: Reading Hit Counts

The following line_profiler output is from profiling a numerical simulation:

```
Timer unit: 1e-06 s

Total time: 0.040161 s
File: simulate.py
Function: run_sim at line 3

Line #   Hits      Time   Per Hit   % Time  Line Contents
     3      1      2.0       2.0      0.0   def run_sim(n_steps):
     4      1      1.0       1.0      0.0       x = 0.0
     5   8001    160.0       0.0      0.4   for i in range(n_steps):
     6   8000  39840.0       5.0     99.6       x += heavy_compute(i)
     7      1      0.2       0.2      0.0   return x
```

What is the value of `n_steps`?

- A) 8001
- B) 8000
- C) 160
- D) Cannot be determined from this output

**Answer: B**

- A) Incorrect — 8001 is the hit count for the for-statement line, which includes one extra hit for the loop-exit check after the last iteration
- B) Correct — the loop body (line 6) executes exactly once per iteration and was hit 8000 times, so n_steps = 8000
- C) Incorrect — 160 is the total Time (µs) for line 5, not a hit count
- D) Incorrect — n_steps is directly readable from the loop body hit count (8000)

---

## Q4 — line_profiler: FLOP/s Calculation

The line_profiler output below is for a dot-product-like kernel (timer unit: 1 µs):

```
Timer unit: 1e-06 s

Line #   Hits      Time   Per Hit   % Time  Line Contents
     6      1      1.0       1.0      0.0       result = 0.0
     7  10000      8.0       0.0      0.1   for i in range(n):
     8  10000  15000.0       1.5     99.9       result += a[i]*b[i] + c[i]
```

How many FLOP/s does this kernel achieve? (Assume 3 floating-point operations per iteration: one multiply, one add, one add.)

- A) 2 × 10^6 FLOP/s
- B) 3 × 10^6 FLOP/s
- C) 2 × 10^9 FLOP/s
- D) 15 × 10^6 FLOP/s

**Answer: A**

- A) Correct — total FLOPs = 3 × 10000 = 30,000; total time = 15000 µs = 0.015s; FLOP/s = 30,000 / 0.015 = 2 × 10^6
- B) Incorrect — 3 × 10^6 would require 0.01s total time, but line 8 takes 0.015s
- C) Incorrect — 2 × 10^9 (2 GFLOP/s) would require only 15 µs total, not 15000 µs; this confuses µs with ns
- D) Incorrect — 15 × 10^6 would result from dividing FLOPs by time in µs rather than seconds (30000 / 0.015 ≠ 15e6)

---

## Q5 — cProfile: Two Bottlenecks at Scale

A model pipeline is profiled with **500 samples**. Which **two** functions should be optimised first for a production run of **5000 samples**?

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
     5    2.000    0.400   10.000    2.000  prepare_model (setup.py:3)
   500    0.001    0.000    0.020    0.000  validate (check.py:9)
   500    0.400    0.001   14.000    0.028  run_inference (model.py:11)
   500    8.500    0.017    8.500    0.017  preprocess_sample (prep.py:4)
     1    0.500    0.500    0.500    0.500  load_weights (io.py:2)
```

- A) `prepare_model` and `load_weights`
- B) `preprocess_sample` and `run_inference`
- C) `validate` and `preprocess_sample`
- D) `run_inference` and `load_weights`

**Answer: B**

- A) Incorrect — both have ncalls that do not scale linearly with samples (5 and 1 respectively), so they remain small at 5000 samples
- B) Correct — both scale with ncalls=500; at 5000 samples: preprocess_sample ≈ 85s (0.017 × 5000) and run_inference ≈ 140s (0.028 cumtime × 5000)
- C) Incorrect — validate has negligible percall (~0.00004s) and will remain trivial even at 5000 samples
- D) Incorrect — load_weights is called once (fixed cost at 0.5s) and will not grow with sample count

---

## Q6 — nsys: HtoD Memory Bandwidth

An Nsight Systems profile of a GPU kernel shows the following memory transfer summary:

```
[gpumemsizesum]
Type     Total (MB)
HtoD     2500.00
DtoH        5.00

[gpumemtimesum]
Type     Total Time (ms)
HtoD     250.000
DtoH       0.500
```

What is the host-to-device (HtoD) memory bandwidth achieved?

- A) 1 GB/s
- B) 10 GB/s
- C) 100 GB/s
- D) 500 MB/s

**Answer: B**

- A) Incorrect — 1 GB/s would require 2500s for 2500 MB, not 0.25s
- B) Correct — 2500 MB / 0.25s = 10,000 MB/s = 10 GB/s, consistent with realistic PCIe 3.0 ×16 achieved bandwidth
- C) Incorrect — 100 GB/s would require only 25ms for 2500 MB, far exceeding PCIe 3.0 limits (~16 GB/s peak)
- D) Incorrect — 500 MB/s would require 5s for 2500 MB, not 250ms

---

## Q7 — cProfile: Recursive Call Notation

A cProfile run produces the following output for a recursive function:

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
   2/1    0.200    0.100    5.000    5.000  recursive_search (search.py:8)
```

What does the `2/1` in the `ncalls` column mean?

- A) The function was called 2 times and returned 1 result
- B) The function made 2 total calls, with 1 primitive (non-recursive) call and 1 recursive sub-call
- C) The function failed twice before succeeding once
- D) The function has 2 entry points and 1 exit point

**Answer: B**

- A) Incorrect — ncalls has nothing to do with return values; the format is total_calls/primitive_calls
- B) Correct — in cProfile, total/primitive means 2 total invocations occurred but only 1 was a top-level (non-recursive) entry; the right-hand percall is cumtime/primitive_calls
- C) Incorrect — cProfile does not track failures or exceptions in the ncalls notation
- D) Incorrect — Python functions have one entry point; this notation is specific to cProfile's recursion tracking

---

## Q8 — cProfile: Projecting Runtime from a Subset

A developer profiles a pipeline on **50 samples** to estimate production cost. The output is:

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
     1    0.050    0.050    0.050    0.050  main (run.py:1)
    50    0.000    0.000    0.050    0.001  load_item (loader.py:3)
    50    0.000    0.000   24.500    0.490  process_item (pipeline.py:7)
     1    3.600    3.600    3.600    3.600  initialize (setup.py:2)
```

What is the estimated wall-clock time for **5000 samples** (round to nearest second)?

- A) ~245s
- B) ~2450s
- C) ~2454s
- D) ~4900s

**Answer: C**

- A) Incorrect — 245s only accounts for process_item scaled by 500× instead of 100×, missing the correct 100× scale-up from 50 to 5000 samples
- B) Incorrect — 2450s is process_item alone (0.490 × 5000); it ignores the fixed initialize cost of 3.6s and load_item cost
- C) Correct — initialize (fixed) = 3.6s + load_item (0.001 × 5000 = 5s) + process_item (0.490 × 5000 = 2450s) ≈ 2459s, closest to 2454s
- D) Incorrect — 4900s would imply process_item percall of ~0.98s, double what the profile shows

---

## Q9 — line_profiler: Which Line to Optimise First

The following line_profiler output is for a data processing function (timer unit: 1 µs):

```
Timer unit: 1e-06 s

Total time: 0.175000 s
File: process.py
Function: run_pipeline at line 1

Line #   Hits      Time    Per Hit   % Time  Line Contents
     3      1   50000.0   50000.0     28.6   data = load_csv(path)
     4      1     500.0     500.0      0.3   data = data.dropna()
     5   1000   25000.0      25.0     14.3   row = parse_row(data[i])
     6   1000  100000.0     100.0     57.1   result = heavy_model(row)
     7      1      10.0      10.0      0.0   return results
```

Which line should be optimised first, and why?

- A) Line 3 (`load_csv`) — it has the highest per-hit time at 50,000 µs
- B) Line 5 (`parse_row`) — it is called 1000 times and is a loop bottleneck
- C) Line 6 (`heavy_model`) — it accounts for 57.1% of total time and is called in a loop
- D) Line 4 (`dropna`) — it is called once but could be eliminated entirely

**Answer: C**

- A) Incorrect — load_csv has high per-hit time but is a fixed one-time cost (28.6%); optimising line 6 first yields more total speedup per Amdahl's law
- B) Incorrect — parse_row accounts for only 14.3% of runtime; heavy_model is the dominant cost and should be attacked first
- C) Correct — heavy_model consumes 57.1% of total time across 1000 loop iterations and scales with data size, making it the highest-leverage optimisation target
- D) Incorrect — dropna takes only 0.3% of runtime; eliminating it would have negligible impact on total wall time

---

## Q10 — cProfile: Sorting by tottime vs cumtime

A developer runs:

```bash
python -m cProfile -s cumulative script.py | head -20
```

And gets this output (sorted by `cumtime`):

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
     1    0.001    0.001   45.210   45.210  main (script.py:1)
     1    0.002    0.002   45.209   45.209  run_experiment (exp.py:3)
   100    0.050    0.001   45.000    0.450  simulate (sim.py:8)
   100    0.120    0.001   44.850    0.449  core_loop (sim.py:22)
   100   44.700    0.447   44.700    0.447  integrate (math.py:5)
   100    0.030    0.000    0.030    0.000  log_result (log.py:11)
```

The developer wants to find the function that spends the most time in **its own code** (not waiting for callees). Which column should they look at, and which function wins?

- A) Sort by `cumtime`; answer is `main`
- B) Sort by `tottime`; answer is `integrate`
- C) Sort by `percall (cum)`; answer is `simulate`
- D) Sort by `ncalls`; answer is `simulate`, `core_loop`, or `integrate` (tie)

**Answer: B**

- A) Incorrect — cumtime includes all sub-call time, so main and run_experiment appear highest only because they transitively call everything; their own code takes 0.001s and 0.002s respectively
- B) Correct — tottime measures only own-code execution; integrate has tottime=44.700s (virtually all runtime) and calls no significant sub-functions, so its cumtime equals its tottime
- C) Incorrect — percall (cum) is cumtime per primitive call and still includes callee time; simulate shows 0.450s but its own tottime is only 0.050s
- D) Incorrect — ncalls identifies how many times a function was called, not how much of its own time it consumed

---
