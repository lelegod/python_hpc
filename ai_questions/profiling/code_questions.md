# Profiling — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — cProfile: Bottleneck at Scale](#q1-cprofile-bottleneck-at-scale)
- [Q2 — cProfile: tottime vs cumtime](#q2-cprofile-tottime-vs-cumtime)
- [Q3 — line_profiler: Reading Hit Counts](#q3-line_profiler-reading-hit-counts)
- [Q4 — line_profiler: FLOP/s Calculation](#q4-line_profiler-flops-calculation)
- [Q5 — cProfile: Two Bottlenecks at Scale](#q5-cprofile-two-bottlenecks-at-scale)
- [Q6 — nsys: HtoD Memory Bandwidth](#q6-nsys-htod-memory-bandwidth)
- [Q7 — cProfile: Recursive Call Notation](#q7-cprofile-recursive-call-notation)
- [Q8 — cProfile: Projecting Runtime from a Subset](#q8-cprofile-projecting-runtime-from-a-subset)
- [Q9 — line_profiler: Which Line to Optimise First](#q9-line_profiler-which-line-to-optimise-first)
- [Q10 — cProfile: Sorting by tottime vs cumtime](#q10-cprofile-sorting-by-tottime-vs-cumtime)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q11 — cProfile: Identifying Own-Code Bottleneck](#q11-cprofile-identifying-own-code-bottleneck)
- [Q12 — line_profiler: Inferring N from Hits](#q12-line_profiler-inferring-n-from-hits)
- [Q13 — cProfile: percall Arithmetic Verification](#q13-cprofile-percall-arithmetic-verification)
- [Q14 — nsys: Kernel vs Transfer Time Ratio](#q14-nsys-kernel-vs-transfer-time-ratio)
- [Q15 — cProfile: Projecting a Scaling Function](#q15-cprofile-projecting-a-scaling-function)
- [Q16 — line_profiler: FLOP/s from Output](#q16-line_profiler-flops-from-output)
- [Q17 — cProfile: ncalls and Fixed vs Variable Cost](#q17-cprofile-ncalls-and-fixed-vs-variable-cost)
- [Q18 — nsys: Diagnosing PCIe Bottleneck](#q18-nsys-diagnosing-pcie-bottleneck)
- [Q19 — line_profiler: Hits Column Disambiguation](#q19-line_profiler-hits-column-disambiguation)
- [Q20 — cProfile: Sorting Strategy for Optimisation](#q20-cprofile-sorting-strategy-for-optimisation)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q21 — timeit: What Does the Return Value Mean?](#q21--timeit-what-does-the-return-value-mean)
- [Q22 — perf_counter: Spot the Bug](#q22--perf_counter-spot-the-bug)
- [Q23 — timeit: Number Parameter Effect](#q23--timeit-number-parameter-effect)
- [Q24 — memory_profiler: Reading the Increment Column](#q24--memory_profiler-reading-the-increment-column)
- [Q25 — memory_profiler: Peak Memory from Output](#q25--memory_profiler-peak-memory-from-output)
- [Q26 — sys.getsizeof: List vs Contents](#q26--sysgetsizeof-list-vs-contents)
- [Q27 — pstats: Programmatic Sort and Print](#q27--pstats-programmatic-sort-and-print)
- [Q28 — cProfile.run() vs -m cProfile](#q28--cprofilerun-vs--m-cprofile)
- [Q29 — dis: Comparing Two Expressions](#q29--dis-comparing-two-expressions)
- [Q30 — timeit: GC Disabled Effect on Result](#q30--timeit-gc-disabled-effect-on-result)

---

> Format: Each question shows cProfile or line_profiler output to interpret.
> Exam frequency: **Every exam**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--cprofile-bottleneck-at-scale)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

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
- C) ~2459s
- D) ~4900s

**Answer: C**

- A) Incorrect — 245s only accounts for process_item scaled by 500× instead of 100×, missing the correct 100× scale-up from 50 to 5000 samples
- B) Incorrect — 2450s is process_item alone (0.490 × 5000); it ignores the fixed initialize cost of 3.6s and load_item cost
- C) Correct — initialize (fixed) = 3.6s + load_item (0.001 × 5000 = 5s) + process_item (0.490 × 5000 = 2450s) = 3.6 + 5.0 + 2450.0 = 2458.6 ≈ 2459s
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

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets cProfile cumtime vs tottime, line profiler Hits interpretation, percall scaling, nsys GPU profiler output, and FLOP/s calculations

---

## Q11 — cProfile: Identifying Own-Code Bottleneck

> **Week reference:** Week 2

Given the following cProfile output, which function's **own code** is responsible for the most execution time?

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
     1    0.002    0.002   88.000   88.000  main (app.py:1)
     1    0.001    0.001   87.990   87.990  run_pipeline (pipe.py:4)
   500    0.050    0.000   87.000    0.174  process_record (proc.py:9)
   500   85.000    0.170   85.000    0.170  compress_data (codec.py:3)
   500    1.900    0.004    1.900    0.004  write_output (io.py:7)
```

- A) `main` — it has the highest cumtime at 88.000 s
- B) `run_pipeline` — it orchestrates all other functions
- C) `compress_data` — it has the highest tottime at 85.000 s
- D) `process_record` — it is called 500 times and has high cumtime per call

**Answer: C**

- A) Incorrect — `main` has tottime=0.002s; its cumtime of 88s is entirely from callees. Cumtime is the wrong column for isolating own-code cost.
- B) Incorrect — `run_pipeline` has tottime=0.001s; it is a thin wrapper that delegates all work. High cumtime with tiny tottime is the signature of a pure orchestrator.
- C) Correct — `compress_data` has tottime=85.000s and cumtime=85.000s (they are equal, meaning it calls no expensive sub-functions). It is doing 85s of real computation in its own code.
- D) Incorrect — `process_record` has tottime=0.050s across 500 calls; it is largely a pass-through to `compress_data` and `write_output`.

---

## Q12 — line_profiler: Inferring N from Hits

> **Week reference:** Week 2

A `line_profiler` report for a batch processing function is shown below (timer unit: 1 µs):

```
Timer unit: 1e-06 s

Total time: 0.620500 s
File: batch.py
Function: process_batch at line 1

Line #   Hits       Time   Per Hit   % Time  Line Contents
     3      1    1200.0    1200.0      0.2   config = load_config()
     4   2501    5000.0       2.0      0.8   for idx in range(batch_size):
     5   2500  614000.0     245.6     99.0       result[idx] = heavy_fn(idx)
     6      1     300.0     300.0      0.0   return result
```

What is the value of `batch_size`?

- A) 2501
- B) 2500
- C) 614000
- D) Cannot be determined; the output does not contain enough information

**Answer: B**

- A) Incorrect — 2501 is the Hits count for line 4 (the loop header), which gets one extra execution when Python confirms the iterator is exhausted after the final iteration.
- B) Correct — the loop body (line 5) executes exactly once per iteration: Hits = 2500 = batch_size. The loop header always shows n+1 hits for a range(n) loop.
- C) Incorrect — 614000 is the total time in microseconds for line 5, not a hit count.
- D) Incorrect — batch_size is directly readable from the loop body Hits column (2500).

---

## Q13 — cProfile: percall Arithmetic Verification

> **Week reference:** Week 2

A cProfile output row reads:

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
    40    3.200    0.080   12.000    0.300  batch_transform (etl.py:15)
```

A student claims the `tottime percall` should be 0.085 s. Is the student correct?

- A) Yes — percall = cumtime / ncalls = 12.000 / 40 = 0.300, so tottime percall must also be recalculated as cumtime / ncalls
- B) No — percall = tottime / ncalls = 3.200 / 40 = 0.080 s, which matches the output exactly
- C) Yes — percall = (cumtime − tottime) / ncalls = (12.000 − 3.200) / 40 = 0.220 s, not 0.080 s
- D) No — percall is always rounded to the nearest 0.1 s, giving 0.1 s

**Answer: B**

- A) Incorrect — the first percall column = tottime / ncalls, not cumtime / ncalls. The student has mixed up the two percall columns. cumtime / ncalls = 0.300 s, which appears in the second percall column.
- B) Correct — tottime percall = 3.200 / 40 = 0.080 s exactly. The student's claim of 0.085 is wrong; there is no rounding to 0.085 from 0.080.
- C) Incorrect — (cumtime − tottime) / ncalls = callee time per call = 0.220 s, which is meaningful but not what the first percall column reports.
- D) Incorrect — cProfile does not round to the nearest 0.1 s; it reports floating-point values with millisecond or better resolution.

---

## Q14 — nsys: Kernel vs Transfer Time Ratio

> **Week reference:** Week 9

An nsys profile of a GPU workload shows:

```
[gpukernsum]
Kernel                         Total Time (ms)
matmul_kernel                      450.000
relu_kernel                         30.000

[gpumemtimesum]
Type    Total Time (ms)
HtoD            20.000
DtoH             5.000
```

What percentage of total GPU activity (kernels + transfers) is spent on memory transfers?

- A) ~4.9% — transfers are a small fraction; the workload is compute-bound
- B) ~50% — transfers and kernels are roughly balanced
- C) ~20% — transfers dominate the workload
- D) ~96% — almost all time is in kernel execution

**Answer: A**

- A) Correct — total kernel time = 450 + 30 = 480 ms; total transfer time = 20 + 5 = 25 ms; total = 505 ms; transfer fraction = 25/505 ≈ 4.95%. The application is compute-bound, making kernel optimisation (arithmetic intensity, occupancy) the correct focus.
- B) Incorrect — 50% would require transfers ≈ 480 ms, but they total only 25 ms.
- C) Incorrect — 20% would require transfers ≈ 120 ms. The actual 25 ms is far lower.
- D) Incorrect — 96% applies to kernel time (480/505 ≈ 95.0%), not transfer time. The question asks about the transfer fraction.

---

## Q15 — cProfile: Projecting a Scaling Function

> **Week reference:** Week 2

A cProfile run on **200 documents** produces:

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
   200    4.000    0.020   16.000    0.080  parse_document (parser.py:5)
     1    0.800    0.800    0.800    0.800  build_index (index.py:3)
```

The production system processes **10,000 documents**. What is the estimated total wall time?

- A) ~800 s (parse_document dominates: 0.020 × 10,000)
- B) ~800.8 s (parse_document: 200 s + build_index: 0.8 s is wrong scale)
- C) ~800.8 s (parse_document: 0.080 × 10,000 = 800 s + build_index: 0.8 s)
- D) ~200 s (parse_document: 0.020 × 10,000 = 200 s + build_index: 0.8 s ≈ 201 s)

**Answer: C**

- A) Incorrect — this uses tottime percall (0.020 s) which only counts parse_document's own code. Using cumtime percall (0.080 s) gives the correct total end-to-end cost per document including all sub-calls.
- B) Incorrect — the scale factor applied to parse_document's tottime (0.020 × 10,000 = 200 s) uses the wrong percall column and underestimates by 4×.
- C) Correct — parse_document cumtime percall = 0.080 s; at 10,000 calls: 0.080 × 10,000 = 800 s. build_index is called once (fixed): 0.800 s. Total ≈ 800.8 s.
- D) Incorrect — 200 s is the tottime projection using 0.020 s/call, which misses the ~600 s spent in parse_document's callees.

---

## Q16 — line_profiler: FLOP/s from Output

> **Week reference:** Week 2

The following line_profiler output is for a vector operation (timer unit: 1 µs). Each iteration performs 4 floating-point operations (2 multiplies + 2 adds):

```
Timer unit: 1e-06 s

Line #   Hits       Time   Per Hit   % Time  Line Contents
     8   5001     200.0       0.0      0.1   for i in range(n):
     9   5000  200000.0      40.0     99.9       y[i] = a*x[i]**2 + b*x[i] + c
```

What is the FLOP/s throughput of this loop?

- A) 100,000 FLOP/s (= 4 × 5000 / 200 µs — time in µs not seconds)
- B) 100,000,000 FLOP/s = 100 MFLOP/s (= 4 × 5000 / 0.200 s)
- C) 25,000,000 FLOP/s = 25 MFLOP/s (= 5000 / 0.200 s — missing FLOPs per iter)
- D) 4,000,000 FLOP/s = 4 MFLOP/s (= 4 / 0.000040 s — wrong divisor)

**Answer: B**

- A) Incorrect — this correctly computes (4 × 5000) / total_time but uses 200 µs in the denominator instead of converting to seconds. 200,000 µs = 0.200 s. The correct result is 20,000 FLOPs / 0.200 s = 100,000,000 FLOP/s.
- B) Correct — total FLOPs = 4 ops/iter × 5000 iters = 20,000 FLOPs; total time = 200,000 µs = 0.200 s; FLOP/s = 20,000 / 0.200 = 100,000,000 = 100 MFLOP/s.
- C) Incorrect — this divides iterations by time rather than FLOPs by time, missing the 4× multiplier for operations per iteration.
- D) Incorrect — this divides FLOPs per iteration (4) by the Per Hit time (40 µs = 0.000040 s), giving a per-iteration rate rather than the total throughput.

---

## Q17 — cProfile: ncalls and Fixed vs Variable Cost

> **Week reference:** Week 2

A cProfile output for a script processing N files shows:

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
     1    0.100    0.100   60.000   60.000  orchestrate (main.py:1)
    50    0.500    0.010   59.400    1.188  analyse_file (analyse.py:8)
     1   10.000   10.000   10.000   10.000  generate_report (report.py:3)
    50    8.000    0.160    8.000    0.160  read_file (io.py:5)
```

If N is doubled to 100 files, which functions' runtimes are expected to approximately double?

- A) `orchestrate` and `generate_report` only
- B) `analyse_file` and `read_file` only
- C) All four functions — everything scales with N
- D) None — cProfile percall values are constant so tottime does not change

**Answer: B**

- A) Incorrect — `orchestrate` has ncalls=1 (fixed, will stay at ~0.1s own code); `generate_report` also has ncalls=1 and will remain ~10s regardless of file count.
- B) Correct — `analyse_file` (ncalls=50) and `read_file` (ncalls=50) scale linearly with N. At N=100 files, both ncalls double and tottime approximately doubles: read_file ~16s, analyse_file tottime ~1s.
- C) Incorrect — fixed-ncalls functions (`generate_report`, `orchestrate`) do not scale with N; only per-file functions grow.
- D) Incorrect — percall being constant means tottime = percall × ncalls scales with ncalls. As N doubles, per-file ncalls doubles, so tottime doubles for those functions.

---

## Q18 — nsys: Diagnosing PCIe Bottleneck

> **Week reference:** Week 9

An nsys report shows:

```
gpukernsum:    0.020 s total
gpumemtimesum: 2.500 s total
```

A colleague suggests "just add more CUDA threads per block to speed this up." Is this advice correct?

- A) Yes — more threads per block improves GPU occupancy and reduces both kernel and transfer time
- B) No — the bottleneck is memory transfer (2.5 s >> 0.02 s); adding threads per block only affects kernel execution time, not PCIe bandwidth
- C) Yes — higher occupancy hides the latency of memory transfers by overlapping them with computation
- D) No — the only fix is to reduce the kernel's arithmetic operations

**Answer: B**

- A) Incorrect — threads per block affects compute occupancy and kernel throughput, but has no effect on PCIe transfer speed. With only 0.020 s of kernel time, improving kernel performance would change total time from 2.52 s to at most 2.50 s — negligible.
- B) Correct — the 2.5 s transfer time (PCIe bottleneck) dwarfs the 0.020 s kernel time. The fix must target data movement: reduce total bytes transferred, use pinned memory, batch transfers, or keep data resident on GPU across multiple kernel calls.
- C) Incorrect — CUDA streams can overlap transfers with computation, but the transfers themselves (2.5 s) must still complete. This technique helps amortize latency but does not reduce bandwidth consumption.
- D) Incorrect — reducing kernel arithmetic is irrelevant when the kernel is only 0.020 s; the 2.5 s transfer is the bottleneck.

---

## Q19 — line_profiler: Hits Column Disambiguation

> **Week reference:** Week 2

A student profiles a function that calls a helper inside a nested loop:

```
Line #   Hits       Time   Per Hit   % Time  Line Contents
    10      5     500.0     100.0      0.1   for batch in batches:
    11      4   10000.0    2500.0      2.0       setup_batch(batch)
    12      5     200.0      40.0      0.0       for item in batch:
    13     20  490000.0   24500.0     97.9           process(item)
```

How many items are in each batch on average?

- A) 20 (total items processed)
- B) 4 (items per batch = total items / number of batches = 20 / 5)
- C) 5 (number of batches)
- D) Cannot be determined — line 11 Hits (4) differs from line 10 Hits (5), which indicates missing data

**Answer: B**

- A) Incorrect — 20 is the total Hits for line 13, which counts all item-processing across all batches, not per-batch count.
- B) Correct — outer loop Hits = 5 batches (loop header 5+1=6 is not shown here, so 5 = actual iterations); inner loop body Hits = 20 total items across 5 batches → 20 / 5 = 4 items per batch on average. (Note: line 11's Hits=4 indicates one batch was skipped by an early-continue or the last iteration, but items per batch is still 20/5=4.)
- C) Incorrect — 5 is the number of batches (Hits on line 10), not items per batch.
- D) Incorrect — the discrepancy between line 10 (Hits=5) and line 11 (Hits=4) suggests line 11 was skipped once (e.g., an empty batch or a conditional), but item count per batch is still determinable from total items (20) / total batches (5).

---

## Q20 — cProfile: Sorting Strategy for Optimisation

> **Week reference:** Week 2

A developer has a cProfile report with 50 functions. She wants to find the function that, if optimised, would give the largest reduction in total wall-clock time. Which sort key and column should she use FIRST?

- A) Sort by `ncalls` descending — most-called function has highest impact
- B) Sort by `tottime` descending — find which function's own code takes the most time
- C) Sort by `cumtime` descending — find which function's call tree costs the most end-to-end
- D) Sort by `percall (tottime)` descending — costliest function per invocation matters most

**Answer: C**

- A) Incorrect — call frequency alone does not determine impact. A function called 10,000 times with 0.001 ms per call contributes 10 s total; a function called once with 30 s cumtime is a bigger target.
- B) Incorrect — tottime is useful for finding where CPU instructions live, but a wrapper with low tottime and high cumtime hides the real opportunity. You could miss an entire slow call tree.
- C) Correct — sort by cumtime descending to identify the highest-cost call trees first. The top entries show where wall-clock time is actually going. Then drill into those entries with tottime to distinguish wrappers from actual work.
- D) Incorrect — percall (tottime) is useful when comparing functions called different numbers of times, but it ignores total impact. A function with 0.1 s/call called 1000 times (100 s total) is a better target than one with 1 s/call called once.

---

## Set 3 — Extended Practice

- [Q21 — timeit: What Does the Return Value Mean?](#q21--timeit-what-does-the-return-value-mean)
- [Q22 — perf_counter: Spot the Bug](#q22--perf_counter-spot-the-bug)
- [Q23 — timeit: Number Parameter Effect](#q23--timeit-number-parameter-effect)
- [Q24 — memory_profiler: Reading the Increment Column](#q24--memory_profiler-reading-the-increment-column)
- [Q25 — memory_profiler: Peak Memory from Output](#q25--memory_profiler-peak-memory-from-output)
- [Q26 — sys.getsizeof: List vs Contents](#q26--sysgetsizeof-list-vs-contents)
- [Q27 — pstats: Programmatic Sort and Print](#q27--pstats-programmatic-sort-and-print)
- [Q28 — cProfile.run() vs -m cProfile](#q28--cprofilerun-vs--m-cprofile)
- [Q29 — dis: Comparing Two Expressions](#q29--dis-comparing-two-expressions)
- [Q30 — timeit: GC Disabled Effect on Result](#q30--timeit-gc-disabled-effect-on-result)

---

## Q21 — timeit: What Does the Return Value Mean?

> **Week reference:** Week 2

What does the following code print?

```python
import timeit

result = timeit.timeit("x = [i**2 for i in range(1000)]", number=500)
print(result)
```

- A) The time in seconds for a single execution of the list comprehension
- B) The total time in seconds for 500 executions of the list comprehension
- C) The average time in seconds per execution (total / 500)
- D) The time in milliseconds for 500 executions

**Answer: B**

- A) Incorrect — `timeit.timeit()` with `number=500` runs the statement 500 times and returns the TOTAL elapsed time for all 500 runs. To get per-execution time, divide the result by `number`.
- B) Correct — `timeit.timeit(stmt, number=N)` returns the total wall-clock time in seconds for N executions of `stmt`. To get the average per-call time, you compute `result / 500` yourself. The return value is always the raw total, not a per-iteration average.
- C) Incorrect — `timeit` does not divide by `number`; the division is left to the caller. The IPython `%timeit` magic does display the per-loop time, but the raw `timeit.timeit()` function returns the total.
- D) Incorrect — `timeit.timeit()` always returns time in seconds (a float), never milliseconds. If `result` is 0.85, that means 850 ms total, not 850 ms per call.

---

## Q22 — perf_counter: Spot the Bug

> **Week reference:** Week 2

What does the following code print, and is there a bug?

```python
from time import perf_counter
import numpy as np

A = np.random.rand(500, 500)
p = 11

start = perf_counter()
np.save("saved.npy", np.linalg.matrix_power(A, p + 1))
end = perf_counter()

print(f"{start - end}")
```

- A) Prints a positive float — the elapsed time in seconds; no bug
- B) Prints a negative float — the expression `start - end` is backwards; the correct expression is `end - start`
- C) Prints `0.0` because `perf_counter` is not precise enough for NumPy operations
- D) Raises a `TypeError` because `perf_counter` timestamps cannot be subtracted

**Answer: B**

- A) Incorrect — `start` is recorded before the operation and `end` after, so `start < end`. The expression `start - end` produces a negative value, not a positive elapsed time.
- B) Correct — this is a real bug present in `week2/numpy5.py`. The correct idiom is `end - start` (later minus earlier), which gives a positive elapsed time. `start - end` gives a negative number with the same magnitude. This is an exam-style trap: the code runs without error but silently reports negative time.
- C) Incorrect — `perf_counter()` has nanosecond resolution and is more than sufficient for NumPy operations. A 500×500 matrix power computation typically takes tens to hundreds of milliseconds, well within perf_counter's measurement range.
- D) Incorrect — `perf_counter()` returns a plain Python `float`. Floating-point subtraction is always valid. No exception is raised; the expression simply yields a negative result.

---

## Q23 — timeit: Number Parameter Effect

> **Week reference:** Week 2

Consider the following two `timeit` calls:

```python
import timeit

t1 = timeit.timeit("sum(range(10000))", number=1)
t2 = timeit.timeit("sum(range(10000))", number=10000)
```

Which statement is TRUE?

- A) `t1` and `t2` will be approximately equal because the per-call time is constant
- B) `t2 / 10000` is a more reliable estimate of the per-call time than `t1`, because averaging over many repetitions reduces the effect of OS scheduling noise
- C) `t2` is always exactly 10000 × `t1` because Python execution is perfectly deterministic
- D) `t1` is the preferred measurement because running only once avoids the garbage collector overhead that accumulates over 10000 runs

**Answer: B**

- A) Incorrect — `t1` measures a single execution (susceptible to OS jitter); `t2` measures 10,000 executions. `t2` is approximately 10,000 × `t1`, not approximately equal to `t1`.
- B) Correct — `t2 / 10000` averages out transient noise from OS scheduler preemptions, cache cold-starts, and branch predictor misses that affect any individual run. A single measurement (`t1`) can vary by ±20–50% for fast operations; the mean of 10,000 runs is far more stable. This is precisely the rationale behind `timeit`'s `number` parameter.
- C) Incorrect — Python execution is not perfectly deterministic; OS scheduling, branch prediction, and CPU frequency scaling introduce variation. `t2` will be close to 10,000 × `t1` but not exactly equal. The variance of `t2` is much lower than 10,000 × `t1`'s variance, which is the point.
- D) Incorrect — `timeit` disables the garbage collector by default (regardless of `number`), so GC accumulation is not a factor in the comparison. Additionally, single-shot timing is less reliable, not more reliable, for microbenchmarks.

---

## Q24 — memory_profiler: Reading the Increment Column

> **Week reference:** Week 2

The following `memory_profiler` output is from a function that builds a large list (timer unit: MiB):

```
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     3     52.3 MiB     52.3 MiB           1   def build_data(n):
     4     52.3 MiB      0.0 MiB           1       result = []
     5    814.6 MiB    762.3 MiB           1       result = [i * 1.5 for i in range(n)]
     6    814.6 MiB      0.0 MiB           1       return result
```

How much memory did the list comprehension on line 5 allocate?

- A) 814.6 MiB — the current RSS after line 5 executes
- B) 762.3 MiB — the Increment column shows how much was newly allocated on that line
- C) 52.3 MiB — the baseline memory before the function ran
- D) 0.0 MiB — line 6 shows no increment, so the allocation must be on line 6

**Answer: B**

- A) Incorrect — 814.6 MiB is the total process RSS (resident set size) after line 5 completes, which includes all memory allocated before the function was called (52.3 MiB) plus the new allocation. The RSS is cumulative, not the allocation for this line alone.
- B) Correct — the Increment column records the change in RSS relative to the previous line. Line 5's Increment of 762.3 MiB means the list comprehension caused approximately 762 MiB of new memory to be allocated. This is the correct column to read for per-line allocation cost.
- C) Incorrect — 52.3 MiB is the process baseline before `build_data` ran. It includes the Python interpreter and any previously imported modules. It is not related to line 5's allocation.
- D) Incorrect — line 6 (`return result`) has Increment = 0.0 MiB because returning a reference to an existing object does not allocate new memory. The allocation happened on line 5 where the list comprehension created all the float objects and the list structure.

---

## Q25 — memory_profiler: Peak Memory from Output

> **Week reference:** Week 2

Consider this `memory_profiler` output for a pipeline function:

```
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     1     48.0 MiB     48.0 MiB           1   def pipeline():
     2    548.0 MiB    500.0 MiB           1       data = load_dataset()
     3    798.0 MiB    250.0 MiB           1       features = extract(data)
     4    548.0 MiB   -250.0 MiB           1       del data
     5    798.0 MiB    250.0 MiB           1       model = train(features)
     6    548.0 MiB   -250.0 MiB           1       del features
     7    548.0 MiB      0.0 MiB           1       return model
```

What is the peak memory usage of this function, and on which line does it occur?

- A) 548.0 MiB — line 2, when the dataset is first loaded
- B) 798.0 MiB — lines 3 and 5, where both intermediate objects are allocated before the previous one is deleted
- C) 1596.0 MiB — line 5, when `data`, `features`, and `model` all coexist in memory
- D) 250.0 MiB — the largest single Increment value

**Answer: B**

- A) Incorrect — line 2 shows 548.0 MiB RSS, but line 3 reaches 798.0 MiB. The peak is not at the first large allocation.
- B) Correct — the peak RSS is 798.0 MiB, reached on both line 3 (data + features coexist) and line 5 (features + model coexist). The `del data` on line 4 frees 250 MiB, bringing it back to 548 MiB before `train` allocates another 250 MiB. The peak is 798 MiB, not the sum of all allocations.
- C) Incorrect — 1596.0 MiB would only occur if `data`, `features`, and `model` all coexisted simultaneously. But `del data` on line 4 removes `data` before `model` is created on line 5. The memory profiler output confirms the RSS never exceeds 798 MiB.
- D) Incorrect — 250.0 MiB is the size of the largest single allocation (on line 3 and line 5 individually), but the peak RSS includes the cumulative effect of all prior allocations still in memory. Peak = base + all live allocations = 48 + 500 + 250 = 798 MiB.

---

## Q26 — sys.getsizeof: List vs Contents

> **Week reference:** Week 2

What is the output of the following code? (Assume CPython 3.12, 64-bit)

```python
import sys

data = [1, 2, 3]
print(sys.getsizeof(data))
print(sys.getsizeof(data) == sys.getsizeof(data[0]) * 3)
```

- A) Prints `88` then `True` — the list size is exactly 3 × the size of one integer
- B) Prints a number around 88 (list overhead + 3 pointers) then `False` — list size includes header bytes and 8-byte pointers, not the integer objects themselves
- C) Prints `84` then `True` — each integer is 28 bytes and 3 × 28 = 84
- D) Prints `24` then `False` — Python stores only the 3 integer values (8 bytes each) without a header

**Answer: B**

- A) Incorrect — the list size is NOT simply 3 × the integer size. `sys.getsizeof(1)` is approximately 28 bytes (Python int header), but the list stores 8-byte pointers (references) to those objects, not the objects inline. The list structure also has its own header (~56 bytes). The list size and the integer size measure different things.
- B) Correct — `sys.getsizeof([1, 2, 3])` returns approximately 88 bytes on CPython 3.12 (56-byte list header + 3 × 8-byte pointers = 80 bytes, plus over-allocation buffer). `sys.getsizeof(1)` is approximately 28 bytes. Since 88 ≠ 28 × 3 = 84, the second print is `False`. This demonstrates that `getsizeof` on a list does NOT measure the memory occupied by the elements.
- C) Incorrect — 84 = 3 × 28 would mean the list size equals the sum of its integer sizes, which ignores the list header and the fact that lists store pointers, not embedded objects.
- D) Incorrect — Python objects always have a header (type pointer, reference count, value). A small Python integer takes ~28 bytes, not 8. The 8-byte figure would apply to a NumPy int64 array element, not a Python int.

---

## Q27 — pstats: Programmatic Sort and Print

> **Week reference:** Week 2

What does the following code do, and what is the correct interpretation of its output?

```python
import cProfile
import pstats
import io

pr = cProfile.Profile()
pr.enable()
result = sum(i**2 for i in range(100_000))
pr.disable()

stream = io.StringIO()
stats = pstats.Stats(pr, stream=stream)
stats.sort_stats('tottime')
stats.print_stats(5)
print(stream.getvalue())
```

- A) Profiles the `sum()` call only; `sort_stats('tottime')` sorts by total wall-clock time; prints 5 lines of output
- B) Profiles the generator expression and `sum()` call; `sort_stats('tottime')` sorts by own-code time excluding callees; `print_stats(5)` prints the top 5 functions by that metric
- C) Raises a `ValueError` because `pstats.Stats` cannot accept a `cProfile.Profile` object — it only accepts filenames
- D) Profiles only the code inside `cProfile.Profile()`, not the code between `enable()` and `disable()`

**Answer: B**

- A) Incorrect — cProfile profiles everything between `pr.enable()` and `pr.disable()`, which includes the entire generator expression and `sum()` call, not just `sum()`. Also, `sort_stats('tottime')` sorts by own-code time, not total wall-clock time (that would be `'cumulative'`).
- B) Correct — `cProfile.Profile` used as a context-manager-equivalent (enable/disable) records all function calls between the two calls. `sort_stats('tottime')` ranks functions by time spent in their own code (excluding callees). `print_stats(5)` limits output to the top 5 most expensive functions by that sort key. The `io.StringIO` stream captures the output as a string instead of printing to stdout.
- C) Incorrect — `pstats.Stats` can accept either a filename string (path to a `.prof` file) OR a `cProfile.Profile` object directly. Passing a `Profile` instance is a common in-memory workflow when you do not want to save a file.
- D) Incorrect — `pr.enable()` starts recording; `pr.disable()` stops it. All Python function calls between these two lines are captured. The `cProfile.Profile()` constructor itself does not start profiling.

---

## Q28 — cProfile.run() vs -m cProfile

> **Week reference:** Week 2

What is the difference between the following two approaches to profiling the same script?

```python
# Approach A — in script.py
import cProfile
cProfile.run('main()', 'output.prof')
```

```bash
# Approach B — shell command
python -m cProfile -o output.prof script.py
```

- A) Approach A profiles only the `main()` function and its callees; Approach B profiles the entire script including module-level import and initialisation code
- B) Approach A saves to a binary `.prof` file; Approach B can only print to stdout and cannot save to a file
- C) Both approaches are identical in every respect; the `-o` flag and `cProfile.run()` second argument produce the same binary output
- D) Approach B disables the garbage collector; Approach A does not

**Answer: A**

- A) Correct — `cProfile.run('main()')` starts profiling when `main()` is called and stops when it returns. Module-level code (imports, global variable definitions) that executes before `main()` is called is NOT profiled. Approach B (`python -m cProfile script.py`) profiles the entire execution from the first bytecode instruction, including all imports and top-level statements. For scripts with expensive import-time initialisation, this distinction matters.
- B) Incorrect — Approach B with `-o output.prof` does save to a binary file in the same format as `cProfile.run(stmt, 'output.prof')`. Without `-o`, Approach B prints to stdout. The `-o` flag is explicitly for saving the binary profile.
- C) Incorrect — as explained in A, the scope of profiling differs. Approach A misses module-level setup cost; Approach B captures the whole execution. They produce different `.prof` files when module-level cost is non-trivial.
- D) Incorrect — neither approach modifies garbage collector behaviour. GC is only disabled by `timeit.timeit()`, not by cProfile. Both cProfile approaches run with the normal GC active.

---

## Q29 — dis: Comparing Two Expressions

> **Week reference:** Week 2

A developer runs `dis.dis` on two equivalent functions:

```python
import dis

def f1(x): return x * x
def f2(x): return x ** 2

dis.dis(f1)
# LOAD_FAST   x
# LOAD_FAST   x
# BINARY_OP   *
# RETURN_VALUE

dis.dis(f2)
# LOAD_FAST   x
# LOAD_CONST  2
# BINARY_OP   **
# RETURN_VALUE
```

Both produce the same mathematical result. What does this bytecode comparison reveal about their performance?

- A) `f1` and `f2` are identical in performance; bytecode opcodes always execute in the same time
- B) `f1` (`x * x`) may be faster than `f2` (`x ** 2`) because `BINARY_OP **` (power) involves a more general computation path (potentially calling `__pow__` with type dispatch) whereas `BINARY_OP *` (multiply) is a simpler operation for numeric types
- C) `f2` is always faster because `x ** 2` is a CPython-optimised special case that avoids the double `LOAD_FAST` in `f1`
- D) The bytecode difference is irrelevant; Python JIT-compiles both to equivalent machine code at runtime

**Answer: B**

- A) Incorrect — different opcodes have different execution costs. `BINARY_OP *` for numeric types dispatches to a highly optimised multiplication routine; `BINARY_OP **` must handle the general case of exponentiation, including non-integer exponents, large integers, and `__pow__` protocol dispatch, making it slower for the simple `x**2` case.
- B) Correct — `x * x` compiles to two `LOAD_FAST` operations followed by a multiply, which is a direct C-level multiplication. `x ** 2` uses the power operator, which in CPython invokes a more general code path that checks whether the exponent is a small positive integer (as an optimisation) but still carries more overhead than a plain multiply. Benchmarking typically shows `x * x` is faster than `x ** 2` for floats and small integers.
- C) Incorrect — CPython does have a special-case optimisation for `x ** 2` in some versions, but this optimisation still produces equivalent or slightly slower code than `x * x` because the power path has more overhead. Additionally, `x ** 2` with a literal `2` does use `LOAD_CONST` (one fewer `LOAD_FAST`) but the power dispatch overhead dominates.
- D) Incorrect — CPython is a bytecode interpreter, not a JIT compiler (in standard CPython 3.x). Each bytecode instruction is interpreted separately at runtime. There is no JIT compilation in CPython that would equate these two paths. Pypy or Numba would JIT-compile them, but standard CPython does not.

---

## Q30 — timeit: GC Disabled Effect on Result

> **Week reference:** Week 2

Consider the following two timing approaches:

```python
import timeit
import gc

# Approach A — default timeit (GC disabled during timing)
t_a = timeit.timeit(
    "result = [Node() for _ in range(10000)]",
    setup="from __main__ import Node",
    number=1000
)

# Approach B — timeit with GC enabled
t_b = timeit.timeit(
    "result = [Node() for _ in range(10000)]",
    setup="import gc; gc.enable(); from __main__ import Node",
    number=1000
)
```

Assuming `Node` is a class that creates cyclic references, which statement is TRUE?

- A) `t_a` and `t_b` will be identical because GC state has no effect on timing
- B) `t_a` will be smaller than `t_b` because Approach A disables the GC, removing periodic GC pause overhead that Approach B incurs; however `t_a` underestimates real-world cost when GC is active
- C) `t_b` will be smaller than `t_a` because enabling GC allows memory to be freed sooner, reducing allocation pressure
- D) `t_a` will be larger than `t_b` because disabling GC causes memory to accumulate, increasing allocation time

**Answer: B**

- A) Incorrect — GC state directly affects timing when objects with cyclic references are being created. Periodic GC collection pauses (which happen when generation thresholds are exceeded) can add milliseconds to the total time. Disabling GC eliminates these pauses, making `t_a` smaller.
- B) Correct — `timeit` disables GC by default to produce more reproducible results. Without GC, `Node` objects accumulate in memory (since cyclic references prevent reference-counting from freeing them), but no collection pauses occur. Approach B allows GC to run, which periodically collects the cyclic garbage but introduces collection pauses. `t_a < t_b` for cyclic-reference-heavy code. The tradeoff is that `t_a` underestimates real production cost where GC runs normally.
- C) Incorrect — while GC does free cyclic garbage sooner, the collection pause itself takes time. For code that creates many cyclic objects, GC pauses increase total time, not decrease it. Allocation pressure from accumulating unreachable objects would be an issue if memory is exhausted, but the GC overhead of collection is what dominates timing.
- D) Incorrect — memory accumulation from disabled GC does not increase allocation time significantly unless the system runs out of memory. The effect of GC being disabled is to remove collection pauses, which decreases (not increases) measured time. The direction of the effect in B is correct.

---
