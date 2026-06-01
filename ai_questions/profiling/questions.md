# Profiling — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — cProfile cumtime vs tottime](#q1-cprofile-cumtime-vs-tottime)
- [Q2 — Identifying the Bottleneck in cProfile](#q2-identifying-the-bottleneck-in-cprofile)
- [Q3 — Scaling ncalls to Production](#q3-scaling-ncalls-to-production)
- [Q4 — Fixed-Cost Functions Do Not Scale](#q4-fixed-cost-functions-do-not-scale)
- [Q5 — Line Profiler: What Does the Hits Column Mean?](#q5-line-profiler-what-does-the-hits-column-mean)
- [Q6 — Inferring Loop Count from Hits](#q6-inferring-loop-count-from-hits)
- [Q7 — FLOP/s Calculation](#q7-flops-calculation)
- [Q8 — Projecting Production Runtime from Profile](#q8-projecting-production-runtime-from-profile)
- [Q9 — nsys Report Sections](#q9-nsys-report-sections)
- [Q10 — GPU Memory Bandwidth from nsys](#q10-gpu-memory-bandwidth-from-nsys)
- [Q11 — Choosing the Right Profiler](#q11-choosing-the-right-profiler)
- [Q12 — Reading tottime for Own-Code Time](#q12-reading-tottime-for-own-code-time)
- [Q13 — Comparing Two Functions at Scale](#q13-comparing-two-functions-at-scale)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q14 — percall Formula](#q14-percall-formula)
- [Q15 — Large cumtime, Small tottime Diagnosis](#q15-large-cumtime-small-tottime-diagnosis)
- [Q16 — Line Profiler: Loop Header vs Loop Body Hits](#q16-line-profiler-loop-header-vs-loop-body-hits)
- [Q17 — FLOP/s and Hardware Utilisation](#q17-flops-and-hardware-utilisation)
- [Q18 — nsys gpukernsum vs gpumemtimesum Bottleneck Diagnosis](#q18-nsys-gpukernsum-vs-gpumemtimesum-bottleneck-diagnosis)
- [Q19 — Interpreting nsys Bandwidth Against Hardware Spec](#q19-interpreting-nsys-bandwidth-against-hardware-spec)
- [Q20 — cProfile percall Consistency Check](#q20-cprofile-percall-consistency-check)
- [Q21 — Scaling with Non-Linear Growth](#q21-scaling-with-non-linear-growth)
- [Q22 — line_profiler % Time Interpretation](#q22-line_profiler-time-interpretation)
- [Q23 — nsys Kernel Time Granularity](#q23-nsys-kernel-time-granularity)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q24 — timeit GC Behaviour](#q24--timeit-gc-behaviour)
- [Q25 — timeit vs perf_counter for Microbenchmarks](#q25--timeit-vs-perf_counter-for-microbenchmarks)
- [Q26 — %timeit Auto-Repeat in IPython](#q26--timeit-auto-repeat-in-ipython)
- [Q27 — memory_profiler Granularity](#q27--memory_profiler-granularity)
- [Q28 — memory_profiler vs cProfile Decorator Clash](#q28--memory_profiler-vs-cprofile-decorator-clash)
- [Q29 — sys.getsizeof Shallow Measurement](#q29--sysgetsizeof-shallow-measurement)
- [Q30 — pstats: Programmatic cProfile Analysis](#q30--pstats-programmatic-cprofile-analysis)
- [Q31 — cProfile Overhead on Short Functions](#q31--cprofile-overhead-on-short-functions)
- [Q32 — dis Module: What It Reveals](#q32--dis-module-what-it-reveals)
- [Q33 — Choosing tottime Sort for Own-Code Bottleneck Hunt](#q33--choosing-tottime-sort-for-own-code-bottleneck-hunt)

---

> Topics: cProfile columns, line profiler Hits, scaling to production workload, FLOP/s, nsys.
> Exam frequency: **Every exam**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--cprofile-cumtime-vs-tottime)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — cProfile cumtime vs tottime

What is the difference between `cumtime` and `tottime` in cProfile output?

- A) `cumtime` is time per call; `tottime` is total time across all calls
- B) `cumtime` includes time spent in sub-calls (callees); `tottime` is time spent only in the function's own code
- C) `cumtime` is the cumulative number of calls; `tottime` is total wall-clock time
- D) `cumtime` and `tottime` are identical when a function makes no recursive calls

**Answer: B**

> **Week reference:** Week 2
> **Mental Model:** Think of `tottime` as "blame only this function" and `cumtime` as "blame this entire call tree." The trap is using `tottime` to find bottlenecks — a tiny `tottime` can hide massive downstream cost.

- A) Incorrect — `percall` is time per call; `cumtime` and `tottime` are both totals across all calls. Confusing `percall` with `cumtime` is a classic mis-read of the cProfile header.
- B) Correct — `cumtime` = own time + all callee time, so it captures the full cost of entering a function. `tottime` = own code only, excluding callees — useful for isolating which function is personally doing expensive work.
- C) Incorrect — Both columns measure time in seconds, not call counts. Call counts are in the `ncalls` column.
- D) Incorrect — They are equal only when a function makes absolutely NO sub-calls; recursion is a special case where both can match, but the general claim is false.

---

## Q2 — Identifying the Bottleneck in cProfile

Given the following cProfile output:

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
     1    0.002    0.002   42.500   42.500  main()
     1    0.001    0.001   40.300   40.300  compute()
  5000    0.050    0.000    2.100    0.000  helper()
     1    2.050    2.050    2.050    2.050  load_data()
```

Which column should you use to identify the function that represents the overall bottleneck?

- A) `tottime`, because it measures only the function's own code
- B) `ncalls`, because a frequently called function is always the bottleneck
- C) `cumtime`, because it captures total time including all sub-calls
- D) `percall`, because it shows average cost and scales linearly

**Answer: C**

> **Week reference:** Week 2
> **Mental Model:** Sort by `cumtime` descending to find bottlenecks — it reveals where wall-clock time is actually going regardless of call depth. `tottime` tells you who is doing the work; `cumtime` tells you what is costing the most.

- A) Incorrect — `tottime` misses time spent in callees. Here `compute()` shows tottime=0.001 s but cumtime=40.3 s, meaning it calls something very expensive. Sorting by `tottime` would bury `compute()` near the bottom.
- B) Incorrect — Frequency alone does not determine the bottleneck. `helper()` is called 5000 times but its cumtime of 2.1 s is far less than `compute()`'s 40.3 s from a single call.
- C) Correct — `cumtime` accounts for all work done within a call path. `compute()` with cumtime=40.3 s is the real bottleneck even though its tottime is only 0.001 s — all the cost lives inside its callees.
- D) Incorrect — `percall` is useful for per-invocation cost comparison, but does not capture total program impact. A function called once with percall=0.001 s clearly matters less than one called once with percall=40.3 s.

---

## Q3 — Scaling ncalls to Production

A cProfile run on 10 samples shows:

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
    10    5.050    0.505    5.050    0.505  process_sample()
```

If the production workload processes 1000 samples, what is the projected total time for `process_sample`?

- A) 5.05 s (same as profiled — sample size doesn't matter)
- B) 50.5 s (10× scale-up from 10 to 100 samples)
- C) 505 s (1000 × 0.505 s per call)
- D) 0.505 s (percall stays constant regardless of workload)

**Answer: C**

> **Week reference:** Week 2
> **Mental Model:** Profile on a small subset, then project with projected = percall × production_ncalls. The trap is confusing "percall is constant" with "total time is constant" — percall is constant, but total grows linearly.

- A) Incorrect — The workload scales with ncalls; running on 10 samples tells you the cost per call, not the cost at scale. Treating the profile result as the answer ignores the 100× difference in workload.
- B) Incorrect — This would be correct for 100 samples (10× more than the 10 profiled), not 1000. The scale factor is 1000 / 10 = 100×, not 10×.
- C) Correct — projected = percall × production_ncalls = 0.505 s × 1000 = 505 s. The percall value (0.505 s) is stable from the profile run, so it is the reliable input to the projection.
- D) Incorrect — `percall` is the per-call cost and does remain constant, but total runtime = percall × ncalls. With 1000 calls, total = 505 s, not 0.505 s.

---

## Q4 — Fixed-Cost Functions Do Not Scale

A cProfile run on 100 samples shows:

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
     1    1.200    1.200    1.200    1.200  load_data()
   100    0.030    0.000    3.000    0.030  process_sample()
```

At a production workload of 1000 samples, approximately how long will `load_data()` take?

- A) 12 s (scales linearly with sample count: 1.2 × 10)
- B) 1200 s (1000 × 1.2 s)
- C) ~1.2 s (fixed cost — called once regardless of sample count)
- D) 0.0012 s (amortized across 1000 samples)

**Answer: C**

> **Week reference:** Week 2
> **Mental Model:** Check ncalls first. ncalls=1 means fixed cost — it does not scale with workload size. Only functions whose ncalls grows with workload size need projection; functions called once stay flat.

- A) Incorrect — `load_data` has ncalls=1 in the profile and ncalls=1 in production; there is no linear scaling. The 10× factor would only apply if the function were called 10 times more.
- B) Incorrect — This assumes ncalls grows to 1000 at production scale, but the function loads data once regardless of how many samples are then processed. ncalls stays at 1.
- C) Correct — ncalls=1 means this is a one-time fixed cost. The projection is trivially 1.200 s × 1 = 1.2 s at any sample count. In contrast, `process_sample` (ncalls=100 → ncalls=1000) would project to 0.03 s × 1000 = 30 s.
- D) Incorrect — Amortizing describes how you might think about the cost per sample (0.0012 s each), but the function's actual runtime is still ~1.2 s total. Amortization is an accounting concept, not a runtime reduction.

---

## Q5 — Line Profiler: What Does the Hits Column Mean?

In `line_profiler` output, what does the `Hits` column represent for a given line of code?

- A) The number of CPU cache hits when that line was executed
- B) The number of times that line was executed during the profiled run
- C) The total memory allocated on that line
- D) The number of function calls made from that line

**Answer: B**

> **Week reference:** Week 2
> **Mental Model:** Hits = execution count for that line. For a loop body, Hits equals the number of loop iterations. For the loop header, Hits = iterations + 1 (the final boundary check). This lets you infer loop parameters directly from the profiler output.

- A) Incorrect — `Hits` has nothing to do with CPU cache behavior. Cache hit rates are hardware-level metrics measured by tools like `perf` or hardware performance counters, not `line_profiler`.
- B) Correct — `Hits` counts how many times the profiler observed execution of that specific line during the profiled run. A line inside a loop with 5000 iterations will show Hits=5000.
- C) Incorrect — Memory allocation is not tracked by `line_profiler`. Use `memory_profiler` (with `@profile` decorator and `mprof run`) to track per-line memory.
- D) Incorrect — Function calls made from a line are not counted by `Hits`. To count sub-calls, use cProfile which tracks `ncalls` per function.

---

## Q6 — Inferring Loop Count from Hits

A `line_profiler` report for a simulation function shows:

```
Line #   Hits    Time   Per Hit   % Time  Line Contents
     5   5001    20.0      0.0      0.2   for i in range(n_steps):
     6   5000  9980.0      2.0     99.8       result += compute(i)
```

What is the value of `n_steps` used in this profiled run?

- A) 1 (the loop header ran once)
- B) 9990 (the total time in microseconds)
- C) 5000 (the loop body line executed 5000 times)
- D) Cannot be determined from `Hits` alone

**Answer: C**

> **Week reference:** Week 2
> **Mental Model:** Read `n_steps` from the loop body Hits, not the loop header. The header always gets one extra hit for the final exhaustion check (Hits = n + 1), but the body executes exactly n times. Always use the body line as the ground truth for iteration count.

- A) Incorrect — Line 5 (the loop header) shows Hits=5001, not 1. The header is executed once per iteration to check the condition, plus once more when the iterator is exhausted. The loop ran many times, not once.
- B) Incorrect — 9990 is time in microseconds (line 5 time + line 6 time ≈ 20 + 9980), not a loop count. Time and Hits are completely separate columns.
- C) Correct — The loop body (line 6) executed 5000 times, so `n_steps = 5000`. The header's extra hit (5001 vs 5000) is from Python checking `range(n_steps)` one final time after the last iteration to confirm exhaustion.
- D) Incorrect — `Hits` on the loop body directly and unambiguously gives the number of iterations. This is one of the primary use cases for `line_profiler`.

---

## Q7 — FLOP/s Calculation

A numerical function performs 8 floating-point operations per iteration, runs for 10,000 iterations, and completes in 0.04 seconds. What is its throughput in FLOP/s?

- A) 80,000 FLOP/s (8 × 10,000 / 0.04 is wrong — divide by Hits not time)
- B) 2,000,000 FLOP/s (= 8 × 10,000 / 0.04)
- C) 250,000 FLOP/s (= 10,000 / 0.04)
- D) 200 FLOP/s (= 8 / 0.04)

**Answer: B**

> **Week reference:** Week 2
> **Mental Model:** FLOP/s = (FLOPs_per_iter × Hits) / total_time_s. Get total FLOPs first (ops × iterations), then divide by wall-clock seconds. Compare to peak hardware FLOP/s to judge efficiency.

- A) Incorrect — This is the correct numerical result (2,000,000) but the label says 80,000, and the parenthetical explanation is wrong about how to compute it. Dividing by time in the denominator is exactly correct.
- B) Correct — FLOP/s = (FLOPs_per_iter × n_iters) / total_time = (8 × 10,000) / 0.04 = 80,000 / 0.04 = 2,000,000 FLOP/s = 2 MFLOPs. To interpret this, compare against the CPU's theoretical peak (e.g., 50+ GFLOP/s for a modern core), revealing headroom for optimization.
- C) Incorrect — This omits the 8 FLOPs per iteration, computing only iterations/time = 250,000 iter/s, which is a rate of iterations, not floating-point operations.
- D) Incorrect — This ignores the number of iterations entirely, computing only FLOPs_per_iter / time, which has no physical meaning.

---

## Q8 — Projecting Production Runtime from Profile

A cProfile run on 100 samples shows:

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
   100    2.000    0.020    2.000    0.020  process_chunk()
```

The production workload processes 5000 samples. What is the projected total time for `process_chunk()`?

- A) 2 s (the profile result is representative as-is)
- B) 20 s (100 × 0.02 × 10 = wrong factor)
- C) 100 s (5000 × 0.02 s per call)
- D) 0.02 s (percall is constant so total is irrelevant)

**Answer: C**

> **Week reference:** Week 2
> **Mental Model:** Scale factor = production_ncalls / profile_ncalls = 5000 / 100 = 50×. Apply it: 2.000 s × 50 = 100 s. Equivalently use percall directly: 0.02 s × 5000 = 100 s. Both routes give the same answer.

- A) Incorrect — The profile was run on 100 samples; production has 50× more. The 2 s result is for 100 calls only. Using it as-is would underestimate production cost by 50×.
- B) Incorrect — This would be the answer for 1000 samples (10× scale: 2 s × 10 = 20 s), not 5000 samples. The scale factor is 5000 / 100 = 50, not 10.
- C) Correct — projected = percall × production_ncalls = 0.02 s × 5000 = 100 s. Equivalently: profiled_total × (production_ncalls / profile_ncalls) = 2 s × 50 = 100 s.
- D) Incorrect — percall being constant is exactly why total time scales: total = percall × ncalls. If percall = 0.02 s and ncalls = 5000, total = 100 s. "Constant percall" means linear scaling, not constant total.

---

## Q9 — nsys Report Sections

When analyzing a GPU program with NVIDIA Nsight Systems (`nsys`), which report sections correspond to GPU kernel execution time and host-to-device / device-to-host memory transfer times respectively?

- A) `gpukernsum` for kernel time; `cpuexecsum` for memory transfer time
- B) `gpukernsum` for kernel time; `gpumemtimesum` for memory transfer time
- C) `gpumemtimesum` for kernel time; `gpumemsizesum` for memory transfer time
- D) `cudaapisum` for kernel time; `gpukernsum` for memory transfer time

**Answer: B**

> **Week reference:** Week 9
> **Mental Model:** `gpukernsum` = what the GPU computed; `gpumemtimesum` = how long data spent moving across the PCIe bus. If transfer time >> kernel time, the bottleneck is memory bandwidth, not compute.

- A) Incorrect — `cpuexecsum` reports CPU-side execution times (host code), not GPU memory transfers. It is useful for diagnosing host-side overhead, not PCIe transfer bottlenecks.
- B) Correct — `gpukernsum` summarizes GPU kernel execution times (how long each CUDA kernel ran on device); `gpumemtimesum` summarizes HtoD and DtoH transfer durations. These are the two key sections for GPU performance diagnosis.
- C) Incorrect — `gpumemtimesum` is for transfer time, not kernel time. `gpumemsizesum` reports transfer sizes in bytes, which combined with `gpumemtimesum` lets you compute effective bandwidth.
- D) Incorrect — `cudaapisum` tracks CUDA API call durations on the CPU side (e.g., `cudaMalloc`, `cudaMemcpy` launch overhead). `gpukernsum` is kernel time, not transfer time.

---

## Q10 — GPU Memory Bandwidth from nsys

An nsys report shows:

```
gpumemsizesum: 2500 MB total transferred
gpumemtimesum: 0.25 s total transfer time
```

What is the effective GPU memory bandwidth?

- A) 625 MB/s (2500 / 4 — dividing by wrong value)
- B) 6250 MB/s (2500 × 2.5)
- C) 10,000 MB/s = 10 GB/s (2500 / 0.25)
- D) 2500 MB/s (size equals bandwidth when time is 1 s)

**Answer: C**

> **Week reference:** Week 9
> **Mental Model:** bandwidth = size / time, same as any throughput calculation. Compare the result against the GPU's theoretical peak bandwidth (e.g., ~900 GB/s for an H100) to judge PCIe vs on-chip transfer efficiency.

- A) Incorrect — Dividing by 4 has no physical meaning here. 625 MB/s would be unusually slow even for PCIe Gen 3. The correct denominator is the transfer time in seconds.
- B) Incorrect — Multiplying size by time gives units of MB·s, not MB/s. Bandwidth is always size divided by time, not size multiplied by time.
- C) Correct — bandwidth = total_size / total_time = 2500 MB / 0.25 s = 10,000 MB/s = 10 GB/s. This is a reasonable PCIe Gen 4 ×16 bandwidth (~64 GB/s peak bidirectional), suggesting reasonable but not maximal utilization.
- D) Incorrect — 2500 MB/s would only be correct if transfer time were exactly 1 s. Since time = 0.25 s, the transfer was 4× faster than 2500 MB/s.

---

## Q11 — Choosing the Right Profiler

You want to identify which specific lines within a slow function are responsible for most of the execution time. Which tool and command should you use?

- A) `python -m cProfile script.py` — cProfile automatically provides line-level timing
- B) `kernprof -l -v script.py` with `@profile` decorator on the function — this is `line_profiler`
- C) `nsys profile python script.py` — Nsight Systems provides line-level Python timing
- D) `time python script.py` — wall-clock time reveals which lines are slow

**Answer: B**

> **Week reference:** Week 2
> **Mental Model:** Tool hierarchy: `time` → whole-program; `cProfile` → function-level; `line_profiler` → line-level. Each zoom level requires a different tool. Go from coarse to fine: use `cProfile` first to find the slow function, then `line_profiler` to pinpoint the slow lines within it.

- A) Incorrect — cProfile provides function-level granularity only. It tells you which function is slow, but not which line inside that function is the culprit.
- B) Correct — `kernprof -l -v` runs `line_profiler`; decorating a function with `@profile` enables per-line timing for that function. The `-l` flag saves `.lprof` binary output; `-v` prints the result immediately. Each line shows Hits, Time, Per Hit, and % Time.
- C) Incorrect — `nsys` profiles GPU activity and CUDA API calls. It has no awareness of Python source lines and is the wrong tool for CPU-side Python profiling.
- D) Incorrect — `time` measures total wall-clock time for the whole program with no per-function or per-line breakdown. It is the starting point for noticing a problem, not diagnosing it.

---

## Q12 — Reading tottime for Own-Code Time

A cProfile report shows:

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
     1    5.000    5.000   45.000   45.000  main()
     1    0.500    0.500   40.000   40.000  compute()
```

How much time does `main()` spend executing its OWN code (not inside `compute()` or other callees)?

- A) 45 s (that is main's total runtime)
- B) 40 s (the time delegated to compute())
- C) 5 s (tottime = own code only, excluding callees)
- D) 0.5 s (same as compute's tottime)

**Answer: C**

> **Week reference:** Week 2
> **Mental Model:** cumtime − tottime = time spent inside callees. For main(): 45 − 5 = 40 s in callees (matching compute's cumtime). `tottime` isolates the function's own instructions; use it to judge whether the function itself is the problem or just a wrapper around expensive callees.

- A) Incorrect — 45 s is `cumtime`, which includes all callee time. Calling it "main's runtime" is true in the wall-clock sense, but it does not represent main's own code. You cannot optimize main's own code based on cumtime alone.
- B) Incorrect — 40 s is the time spent inside `compute()` (its cumtime), which is subtracted from main's cumtime to get main's own time. It represents callees, not main's own instructions.
- C) Correct — `tottime` reports time spent in the function's own instructions, excluding any callees. main's 5 s of tottime means 5 s is spent in main's own lines; the other 40 s is in callees like compute().
- D) Incorrect — 0.5 s is `compute()`'s own code time (compute's tottime), unrelated to `main()`'s tottime. The two functions have independent tottime values.

---

## Q13 — Comparing Two Functions at Scale

A cProfile profile shows two functions:

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
     1    0.010    0.010   30.000   30.000  transform_dataset()
  1000    0.400    0.000    0.500    0.001  validate_record()
```

At production scale: `transform_dataset` is still called once; `validate_record` is called 50,000 times. Which function is the bigger bottleneck at production scale?

- A) `validate_record` — it is called far more often so it always dominates
- B) `transform_dataset` — its cumtime (30 s) still exceeds validate_record's projected time (50,000 × 0.0005 = 25 s)
- C) Both are equal — cumtime of 30 s each
- D) `validate_record` — tottime (0.4 s) is greater than transform_dataset's tottime (0.01 s)

**Answer: B**

> **Week reference:** Week 2
> **Mental Model:** Always compare projected total times, not raw ncalls or tottime. Project each function: fixed-ncalls functions stay flat; scaling functions use percall × production_ncalls. Then compare the two projected totals head-to-head.

- A) Incorrect — Higher call count doesn't automatically make a function the bottleneck; you must compute projected total time. validate_record at 50,000 calls × 0.001 s/call = 50 s... wait, let's be precise: percall = 0.0005 s (cumtime 0.5 s / 1000 calls). Projection = 0.0005 × 50,000 = 25 s. That is still less than transform_dataset's 30 s.
- B) Correct — `transform_dataset` stays at ~30 s (ncalls=1, fixed). `validate_record` projects to 50,000 × (0.500 s / 1000) = 50,000 × 0.0005 s/call = 25 s. 30 s > 25 s, so `transform_dataset` is still the larger bottleneck at this scale, though both deserve attention.
- C) Incorrect — 25 s ≠ 30 s; they differ by 5 s. Equal projected time would require validate_record to be called 60,000 times (30 s / 0.0005 s/call), not 50,000.
- D) Incorrect — Comparing `tottime` values ignores callee time entirely and misrepresents actual impact. transform_dataset's tottime of 0.01 s is tiny, but its cumtime of 30 s shows it delegates massively. Always use cumtime for bottleneck comparison.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets cProfile cumtime vs tottime, line profiler Hits interpretation, percall scaling, nsys GPU profiler output, and FLOP/s calculations

---

## Q14 — percall Formula

> **Week reference:** Week 2

In cProfile output, what is the formula for the first `percall` column (the one adjacent to `tottime`)?

- A) percall = cumtime / ncalls
- B) percall = tottime / ncalls
- C) percall = tottime × ncalls
- D) percall = cumtime − tottime

**Answer: B**

`percall` in the `tottime` column = tottime / ncalls. It represents the average time the function's own code takes per invocation, not including callees. The second `percall` column (next to cumtime) = cumtime / ncalls, which includes callees. A) is the cumtime-based percall. C) and D) are not meaningful cProfile metrics.

---

## Q15 — Large cumtime, Small tottime Diagnosis

> **Week reference:** Week 2

A function `pipeline()` shows `tottime = 0.003 s` and `cumtime = 85.000 s` in cProfile. What is the correct interpretation?

- A) `pipeline()` has a bug causing an infinite loop in its own code
- B) `pipeline()` itself does almost no work; almost all 85 s is spent inside functions it calls
- C) `pipeline()` was called 85 / 0.003 ≈ 28,333 times
- D) `pipeline()` and its callees together execute 85 s of CPU time per call

**Answer: B**

cumtime − tottime = 84.997 s is spent inside callees of `pipeline()`. The function is a thin orchestrator: it does 0.003 s of its own work then delegates everything else. A) is wrong — an infinite loop in own code would increase tottime, not cumtime. C) misreads the columns — ncalls is a separate column. D) is close but "per call" is wrong; 85 s is the total across all calls, not per call (unless ncalls = 1, which we don't know).

---

## Q16 — Line Profiler: Loop Header vs Loop Body Hits

> **Week reference:** Week 2

A `line_profiler` report shows:

```
Line #   Hits    Time  Per Hit  % Time  Line Contents
    10    201    40.0      0.2      0.1  for i in range(n):
    11    200  39960.0    199.8     99.9      process(data[i])
```

What is the value of `n`, and why does line 10 show 201 hits instead of 200?

- A) n = 201; line 11 is called once before the loop starts
- B) n = 200; line 10 gets one extra hit when Python checks the iterator exhaustion condition after the last iteration
- C) n = 200; line 10 executes n+1 times due to Python's off-by-one in range()
- D) n = 201; cProfile counts the function definition line as one extra hit

**Answer: B**

The loop body (line 11) executes exactly n = 200 times — one per iteration. The loop header (line 10) is evaluated n+1 = 201 times: once per iteration to get the next value, plus one final time when the iterator raises `StopIteration` and the loop exits. A) is wrong — line 11 doesn't run before the loop. C) range() has no off-by-one; the extra hit is the exhaustion check. D) confuses line_profiler with cProfile.

---

## Q17 — FLOP/s and Hardware Utilisation

> **Week reference:** Week 2

A matrix multiply of shape 512 × 512 × 512 completes in 0.002 s. Approximately how many GFLOP/s does this achieve, and is it likely compute-bound on a modern CPU (assume ~200 GFLOP/s peak)?

- A) ~134 GFLOP/s — yes, near peak
- B) ~0.134 GFLOP/s — no, far below peak; likely memory-bound
- C) ~134 GFLOP/s — no, far below peak; likely memory-bound
- D) ~268 GFLOP/s — exceeds hardware peak; result is impossible

**Answer: C**

FLOPs = 2 × 512 × 512 × 512 = 2 × 134,217,728 ≈ 2.68 × 10^8. Time = 0.002 s. FLOP/s = 2.68×10^8 / 0.002 = 1.34×10^11 = 134 GFLOP/s. This is 67% of the 200 GFLOP/s peak — respectable but still below peak, so it is not fully compute-bound; memory bandwidth or overhead still limits it. A) and C) share the same number but differ on "near peak" — 134/200 = 67% is not "near peak." D) is wrong: 134 GFLOP/s < 200 GFLOP/s.

---

## Q18 — nsys gpukernsum vs gpumemtimesum Bottleneck Diagnosis

> **Week reference:** Week 9

An nsys report for a GPU application shows:

```
gpukernsum:    total kernel time = 0.050 s
gpumemtimesum: total transfer time = 1.200 s
```

What is the primary bottleneck, and what strategy addresses it?

- A) Compute-bound — optimise kernel arithmetic intensity
- B) Memory transfer-bound — reduce host-device data movement or use pinned memory / overlap transfers
- C) Memory transfer-bound — increase the number of CUDA threads per block
- D) Latency-bound — use CUDA streams to parallelize kernel launches

**Answer: B**

Transfer time (1.2 s) is 24× longer than kernel time (0.05 s), so the bottleneck is PCIe data movement, not GPU computation. The fix is to reduce how much data crosses the bus (e.g., keep data on-device between kernels, reduce precision, use zero-copy when appropriate) or hide latency by overlapping transfers with computation using CUDA streams. A) is wrong — compute is not the bottleneck. C) increasing threads per block changes compute occupancy, not transfer speed. D) streams help overlap but don't reduce total transfer volume, which is the root issue here.

---

## Q19 — Interpreting nsys Bandwidth Against Hardware Spec

> **Week reference:** Week 9

A GPU node has a PCIe 4.0 ×16 link with a theoretical peak bidirectional bandwidth of ~64 GB/s. An nsys report shows 4000 MB transferred HtoD in 0.5 s. What is the achieved bandwidth, and what does this suggest?

- A) 8 GB/s — ~12.5% of peak; transfers are likely serialized or small, leaving significant room for improvement
- B) 8 GB/s — ~100% of peak; the application is fully bandwidth-saturated
- C) 80 GB/s — exceeds peak bandwidth; the nsys measurement is incorrect
- D) 0.5 GB/s — far below peak; the kernel is the real bottleneck

**Answer: A**

Bandwidth = 4000 MB / 0.5 s = 8000 MB/s = 8 GB/s. Peak is 64 GB/s (unidirectional), so utilisation ≈ 12.5%. This is low, suggesting transfers are fragmented (many small copies), not using pinned/page-locked memory, or running synchronously when they could be overlapped. B) is wrong — 8 GB/s is far below 64 GB/s. C) is wrong — 8 GB/s does not exceed peak. D) miscomputes: 4000 MB / 0.5 s = 8 GB/s, not 0.5 GB/s.

---

## Q20 — cProfile percall Consistency Check

> **Week reference:** Week 2

A cProfile report shows:

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
   200    6.000    0.030   10.000    0.050  analyse()
```

A colleague claims: "the second `percall` column shows 0.050 s, so each call to `analyse()` spends 0.050 s in callee functions." Is this correct?

- A) Yes — the second percall is always the time spent in callees per call
- B) No — the second percall is cumtime / ncalls = 10.000 / 200 = 0.050 s, which is the total (own + callee) time per call; callee time per call = cumtime/ncalls − tottime/ncalls = 0.050 − 0.030 = 0.020 s
- C) No — the second percall equals tottime / ncalls and the first percall equals cumtime / ncalls
- D) Yes — percall always equals cumtime / ncalls for both columns

**Answer: B**

The second percall = cumtime / ncalls = 0.050 s is the end-to-end time per call including callees. To isolate callee time per call: 0.050 − 0.030 = 0.020 s per call in callees. A) mistakes "time in callees per call" for the cumtime percall value itself. C) reverses the column definitions. D) is wrong — the first percall = tottime / ncalls = 0.030 s, not cumtime / ncalls.

---

## Q21 — Scaling with Non-Linear Growth

> **Week reference:** Week 2

A profiled run processes N = 1,000 items and shows `sort_data()` with `tottime = 2.0 s` and `ncalls = 1`. The sort algorithm is O(N log N). At N = 10,000 items, what is the best estimate of `sort_data()` runtime?

- A) 20 s (linear scale: 10× items → 10× time)
- B) ~23.3 s (O(N log N) scaling: 10 × log(10,000)/log(1,000) ≈ 10 × 1.333 ≈ 13.33 → 2.0 × 13.33/6)
- C) ~26.7 s (O(N log N): ratio = (10,000 × log₁₀(10,000)) / (1,000 × log₁₀(1,000)) = 4/3 × 10 × 2.0 s)
- D) 200 s (quadratic scale: 10× items → 100× time is wrong for O(N log N))

**Answer: C**

For O(N log N): scaled_time = base_time × (N₂ log N₂) / (N₁ log N₁). Using log₁₀: (10,000 × 4) / (1,000 × 3) = 40,000 / 3,000 = 13.33. Estimated time = 2.0 × 13.33 = 26.67 s ≈ 26.7 s. A) assumes O(N) which is wrong for a sort. B) makes an arithmetic error in the ratio. D) assumes O(N²) which would be a naive sort, not a standard library sort.

---

## Q22 — line_profiler % Time Interpretation

> **Week reference:** Week 2

A `line_profiler` report shows:

```
Line #   Hits      Time   Per Hit   % Time  Line Contents
    15   1000   80000.0      80.0     40.0   intermediate = transform(x)
    16   1000  120000.0     120.0     60.0   result = model_predict(intermediate)
```

A developer decides to optimise `transform()` because it has a lower `Per Hit` value and therefore "must be faster." Is this reasoning valid?

- A) Yes — Per Hit directly indicates which function is more efficient per call
- B) No — `% Time` shows `model_predict` consumes 60% of total time; optimising it delivers more absolute speedup per Amdahl's law
- C) Yes — a lower Per Hit means the function is called more efficiently and should be optimised to keep it that way
- D) No — only `ncalls` (Hits) should guide optimisation decisions

**Answer: B**

`% Time` is the correct guide for optimisation priority: `model_predict` takes 60% of runtime vs 40% for `transform`. By Amdahl's law, halving `model_predict` time saves 30% total, while halving `transform` saves only 20%. Per Hit (80 µs vs 120 µs) is irrelevant here — both lines are called the same number of times (Hits = 1000), so total time = Per Hit × Hits directly. C) is wrong reasoning — "efficient" functions still need optimisation if they dominate total time. D) is wrong — Hits alone can't guide optimisation.

---

## Q23 — nsys Kernel Time Granularity

> **Week reference:** Week 9

Which of the following statements about `gpukernsum` in an nsys profile is TRUE?

- A) `gpukernsum` includes both kernel execution time AND HtoD/DtoH memory transfer time
- B) `gpukernsum` reports the CPU time spent launching kernels via CUDA API calls
- C) `gpukernsum` shows only GPU kernel execution time, excluding memory transfers which appear in `gpumemtimesum`
- D) `gpukernsum` and `gpumemtimesum` always sum to the total GPU wall-clock time

**Answer: C**

`gpukernsum` reports the aggregate execution time of CUDA kernels running on the GPU device — pure compute time. Memory transfers (HtoD/DtoH via `cudaMemcpy` or unified memory migrations) are reported separately in `gpumemtimesum`. A) is the most common misconception. B) describes `cudaapisum`, which tracks CPU-side API call overhead. D) is false — there can also be synchronization gaps, idle time, and other GPU activity not captured in either summary table.

---

## Set 3 — Extended Practice

- [Q24 — timeit GC Behaviour](#q24--timeit-gc-behaviour)
- [Q25 — timeit vs perf_counter for Microbenchmarks](#q25--timeit-vs-perf_counter-for-microbenchmarks)
- [Q26 — %timeit Auto-Repeat in IPython](#q26--timeit-auto-repeat-in-ipython)
- [Q27 — memory_profiler Granularity](#q27--memory_profiler-granularity)
- [Q28 — memory_profiler vs cProfile Decorator Clash](#q28--memory_profiler-vs-cprofile-decorator-clash)
- [Q29 — sys.getsizeof Shallow Measurement](#q29--sysgetsizeof-shallow-measurement)
- [Q30 — pstats: Programmatic cProfile Analysis](#q30--pstats-programmatic-cprofile-analysis)
- [Q31 — cProfile Overhead on Short Functions](#q31--cprofile-overhead-on-short-functions)
- [Q32 — dis Module: What It Reveals](#q32--dis-module-what-it-reveals)
- [Q33 — Choosing tottime Sort for Own-Code Bottleneck Hunt](#q33--choosing-tottime-sort-for-own-code-bottleneck-hunt)

---

## Q24 — timeit GC Behaviour

> **Week reference:** Week 2

**Mental Model:** `timeit.timeit()` deliberately disables Python's garbage collector during the timed run. This removes GC pauses from the measurement, making results more reproducible — but it also means the benchmark does not reflect real-world performance when GC overhead is significant.

Which statement about `timeit.timeit()` and the garbage collector is correct?

- A) `timeit.timeit()` enables the garbage collector to simulate realistic production conditions
- B) `timeit.timeit()` disables the garbage collector by default, which can make short-lived object benchmarks appear faster than in production
- C) `timeit.timeit()` disables the garbage collector permanently for the rest of the Python session
- D) The garbage collector has no effect on `timeit` results because Python uses reference counting, not GC

**Answer: B**

- A) Incorrect — the default behaviour is the opposite: GC is disabled. Enabling GC during timing would require passing `setup="import gc; gc.enable()"` explicitly or calling `timeit.Timer(...).timeit()` with a custom setup.
- B) Correct — `timeit` disables GC via `gc.disable()` before each timed loop and restores it after. This prevents GC pauses from inflating measurements, but benchmarks that allocate many short-lived objects will look faster than they are in production where GC runs normally.
- C) Incorrect — `timeit` restores the original GC state after the timed run. The disable is scoped to the measurement, not the interpreter session.
- D) Incorrect — CPython does use reference counting as its primary mechanism, but it also has a cyclic garbage collector that handles reference cycles. The cyclic GC can trigger pauses for code that creates many inter-referencing objects; this is precisely what `timeit` disables.

---

## Q25 — timeit vs perf_counter for Microbenchmarks

> **Week reference:** Week 2

**Mental Model:** `timeit.timeit()` runs the statement many times (controlled by `number`) and returns the total time, averaging out OS scheduling noise. `time.perf_counter()` measures a single execution, which is susceptible to interrupts and context switches. For microbenchmarks (sub-millisecond code), `timeit` is the right tool; `perf_counter` is for coarse-grained wall-clock measurement of larger work.

A developer writes `t = perf_counter(); f(); elapsed = perf_counter() - t` and finds the result varies by ±30% between runs. Which is the BEST explanation?

- A) `perf_counter` has lower resolution than `timeit` and cannot measure sub-millisecond operations
- B) A single-shot `perf_counter` measurement is susceptible to OS scheduling jitter; `timeit` averages many repetitions to reduce this noise
- C) `f()` has non-deterministic behaviour; repeating it would not help
- D) `perf_counter` returns CPU time, not wall-clock time, so it is inappropriate for timing Python functions

**Answer: B**

- A) Incorrect — `time.perf_counter()` has nanosecond resolution on modern systems (it is the highest-resolution clock available in Python). Low resolution is not the issue.
- B) Correct — a single measurement captures whatever OS scheduling event happened to occur during that one run. `timeit` runs the code `number` times (defaulting to enough repetitions so the total takes ≥0.2 s) and divides, effectively averaging out transient OS-level interruptions and cache-warming effects.
- C) Incorrect — the problem is not the function's behaviour; it is that a single measurement does not average out external noise. Most functions are deterministic; the variance comes from the measurement environment.
- D) Incorrect — `time.perf_counter()` returns wall-clock time (the highest-precision monotonic clock). `time.process_time()` returns CPU time. This is a common confusion but is not the issue described.

---

## Q26 — %timeit Auto-Repeat in IPython

> **Week reference:** Week 2

**Mental Model:** `%timeit` in IPython automatically chooses the number of loops and repeat count so that the total timing takes a few seconds, then reports the best (minimum) of the repeats. It reports the best run rather than mean to minimize the impact of outliers caused by OS interrupts. This is different from `timeit.timeit()` which returns the total time and requires manual loop count selection.

What does `%timeit` in IPython report when it says `"1.23 ms ± 45 µs per loop (mean ± std dev of 7 runs, 1000 loops each)"`?

- A) The function was called once; 1.23 ms is the wall-clock time for that single call
- B) The function was called 7 times total; 1.23 ms is the average across all 7 calls
- C) The function was called 7,000 times total (7 runs × 1,000 loops); 1.23 ms is the mean per-loop time across those 7 run-totals, and ±45 µs is the standard deviation between runs
- D) `%timeit` always runs exactly 100,000 iterations regardless of the function's runtime

**Answer: C**

- A) Incorrect — the output explicitly states "7 runs, 1000 loops each". The function was called 7,000 times total, not once.
- B) Incorrect — 7 runs × 1,000 loops = 7,000 calls. The 1.23 ms is the average time per single loop iteration, not per run. One "run" of 1000 loops took approximately 1.23 s total.
- C) Correct — `%timeit` executes `number` loops per run and `repeat` runs. Each run measures `number` executions and divides to get per-loop time. It reports the mean and std dev across the `repeat` runs. Choosing `number` automatically ensures each run takes long enough to be meaningful (typically ≥200 ms).
- D) Incorrect — `%timeit` adaptively selects `number` based on how long the code takes. For fast code it might use 1,000,000 loops; for slow code it might use just 1 loop per run. The 100,000 figure is not fixed.

---

## Q27 — memory_profiler Granularity

> **Week reference:** Week 2

**Mental Model:** `memory_profiler` (used via `@profile` decorator and `mprof run` or `python -m memory_profiler`) provides **line-by-line memory usage** expressed in MiB, not function-level aggregates. Each line shows current RSS (Mem usage) and the delta (Increment) since the previous line. This is in contrast to cProfile which is function-level, and line_profiler which is time-based line-level.

Which statement correctly describes `memory_profiler`'s output granularity?

- A) `memory_profiler` reports total heap allocation per function, similar to how cProfile reports tottime per function
- B) `memory_profiler` reports memory usage and increment per source line for decorated functions, expressed in MiB
- C) `memory_profiler` reports peak GPU memory usage per kernel launch, in MB
- D) `memory_profiler` and `line_profiler` produce identical output — only the metric column differs (memory vs time)

**Answer: B**

- A) Incorrect — `memory_profiler` does not provide per-function aggregates. It shows the RSS at each line, so you see memory go up when an allocation occurs and stay the same (or decrease after a GC) at other lines. Function-level memory cost must be inferred from the first and last line of the function.
- B) Correct — `memory_profiler` outputs a table with columns: Line #, Mem usage (current process RSS in MiB), Increment (change from previous line in MiB), Occurrences (how many times the line ran), and Line Contents. Each decorated function gets its own table.
- C) Incorrect — `memory_profiler` profiles CPU-side (host) memory only. GPU memory is tracked by NVIDIA tools such as `nsys` or `nvidia-smi`. `memory_profiler` has no GPU awareness.
- D) Incorrect — while both tools produce line-level tables, the columns differ fundamentally. `line_profiler` shows Hits, Time, Per Hit, and % Time. `memory_profiler` shows Mem usage, Increment, and Occurrences. They are complementary tools, not equivalent ones.

---

## Q28 — memory_profiler vs cProfile Decorator Clash

> **Week reference:** Week 2

**Mental Model:** Both `memory_profiler` and `line_profiler` use a `@profile` decorator, but they are imported from different packages and cannot be applied to the same function simultaneously without careful management. When profiling with `kernprof`, the `@profile` decorator is injected into the global namespace; when profiling with `memory_profiler`, it is injected differently. Mixing them causes a `NameError` or incorrect results.

A developer wants to profile both the memory usage AND the line-level execution time of a function simultaneously. They write:

```python
from line_profiler import profile as lp
from memory_profiler import profile as mp

@lp
@mp
def my_func():
    ...
```

What is the most likely outcome when running this under `kernprof -l -v`?

- A) Both profilers work correctly; stacking decorators is the supported way to profile with multiple tools simultaneously
- B) Only `line_profiler` captures data because it wraps `mp`; `memory_profiler` sees a wrapped callable, not the original function, and may report incorrect line numbers or no data
- C) A `SyntaxError` is raised immediately because two `@profile` decorators cannot be stacked in Python
- D) `memory_profiler` takes precedence and silently disables `line_profiler`

**Answer: B**

- A) Incorrect — stacking profiling decorators is not officially supported and leads to instrumentation interference. The inner decorator wraps the function; the outer decorator wraps the wrapper. Line number mapping in the inner tool may break because it sees a `wrapper` object rather than the original function's `__code__`.
- B) Correct — `line_profiler` wraps the already-wrapped function returned by `@mp`. The `line_profiler` records timing for the `mp`-wrapped callable, which is correct. However, `memory_profiler` instruments the original function's bytecode for line tracking; when that function is further wrapped by `line_profiler`, the bytecode-level tracking may be disrupted or show incorrect line references. In practice, the tools should be run in separate profiling sessions, not stacked.
- C) Incorrect — Python allows stacking any number of decorators; there is no syntax restriction. The issue is semantic, not syntactic.
- D) Incorrect — decorators are applied bottom-up (inner first); `@lp` is outermost and is applied last, so `line_profiler` wraps `memory_profiler`'s output, not the reverse. Neither "takes precedence" in a meaningful sense.

---

## Q29 — sys.getsizeof Shallow Measurement

> **Week reference:** Week 2

**Mental Model:** `sys.getsizeof(obj)` returns the **shallow** size of the object in bytes — the size of the object header and its immediate fields, but NOT the size of any objects it references. For containers like lists, it counts the list's internal array of pointers but not the pointed-to objects. To get total (deep) size, you must recursively call `getsizeof` on all referenced objects.

What does `sys.getsizeof([1, 2, 3, 4, 5])` return?

- A) The total memory occupied by the list and all five integer objects it contains
- B) The size of the list object's internal structure (header + pointer array for 5 elements), excluding the memory occupied by the integer objects themselves
- C) Always 8 bytes, since Python stores all objects as 64-bit pointers
- D) 40 bytes, since each Python integer occupies 8 bytes and there are 5 of them

**Answer: B**

- A) Incorrect — `getsizeof` is explicitly shallow. It counts only the bytes the list object itself uses (not its contents). To get the deep size, you would use a recursive approach or a library like `pympler.asizeof`.
- B) Correct — for a list of 5 elements, `getsizeof` returns approximately 120 bytes on CPython 3.x (56 bytes for the list object header + 8 bytes × 5 for the pointer array + over-allocation padding). The five integer objects (each ~28 bytes) are not counted.
- C) Incorrect — `getsizeof` does not return a fixed value. It varies by type: an empty list is ~56 bytes, an empty dict ~248 bytes, a small integer ~28 bytes. The 8-byte pointer size is an implementation detail, not what `getsizeof` reports.
- D) Incorrect — 40 bytes would be 8 bytes × 5 integers, which conflates the C-level pointer size with the full Python integer object size, and also ignores the list header overhead. CPython integers are ~28 bytes each (not 8), and `getsizeof` on the list does not count them at all.

---

## Q30 — pstats: Programmatic cProfile Analysis

> **Week reference:** Week 2

**Mental Model:** `pstats.Stats` loads a `.prof` binary file produced by cProfile and lets you sort, filter, and print the profiling data programmatically. The two most important sort keys are `'cumulative'` (for finding call-tree bottlenecks) and `'tottime'` (for finding functions with expensive own-code). `stats.print_stats(N)` limits output to the top N functions.

Which `pstats` code correctly loads a profile file and prints the top 10 functions sorted by cumulative time?

- A) `pstats.Stats('out.prof').sort_stats('tottime').print_stats(10)`
- B) `pstats.Stats('out.prof').sort_stats('cumulative').print_stats(10)`
- C) `pstats.Stats('out.prof').print_stats('cumulative', 10)`
- D) `pstats.Stats('out.prof', sort='cumulative').print_stats(10)`

**Answer: B**

- A) Incorrect — `sort_stats('tottime')` sorts by own-code time, not cumulative time. This would show the functions whose own instructions are most expensive, not the highest-cost call trees. Use this when you want to identify "who is doing the work," not "what path costs the most."
- B) Correct — `sort_stats('cumulative')` sets the sort key to cumulative time; `print_stats(10)` limits output to the top 10 rows. This is the standard idiom for bottleneck hunting via the call tree. The equivalent shell command is `python -m cProfile -s cumulative script.py`.
- C) Incorrect — `print_stats` does not accept a sort key as its first argument. The sort key must be set separately via `sort_stats()`. `print_stats(N)` only accepts an integer (limit) or a regex string (filter by function name), not a sort key name.
- D) Incorrect — `pstats.Stats()` does not accept a `sort=` keyword argument. The constructor accepts filenames (strings) or `cProfile.Profile` objects to merge. Sorting is done via the `sort_stats()` method chained afterwards.

---

## Q31 — cProfile Overhead on Short Functions

> **Week reference:** Week 2

**Mental Model:** cProfile is a **deterministic profiler** — it hooks into every function call and return using Python's `sys.setprofile` mechanism. Each hook invocation adds a small but fixed overhead (~1–5 µs per call). For functions called millions of times with sub-microsecond bodies, the profiler overhead can exceed the actual function cost, making the measured times unreliable and the optimisation ranking misleading.

A developer profiles a tight inner loop that calls a 0.5 µs helper function 10 million times. The cProfile output shows the helper with `tottime = 15 s` but the un-profiled run completes in 5 s. What is the best explanation?

- A) The developer made an arithmetic error; cProfile cannot add more than 10% overhead
- B) cProfile's deterministic per-call hook adds fixed overhead (≈1–5 µs) per function call; at 10 million calls the profiler overhead (≈10–50 s) dominates the actual 5 s runtime, making the measurement unreliable
- C) The helper function has a memory leak that only manifests under the profiler's memory tracking
- D) cProfile disables CPU caches during profiling, causing the 3× slowdown

**Answer: B**

- A) Incorrect — cProfile can add substantial overhead for call-intensive code. There is no 10% cap; the overhead is proportional to ncalls, not to the function's own runtime. For a 0.5 µs function called 10M times, even 1 µs of profiler overhead per call doubles the measured time.
- B) Correct — cProfile intercepts every `call` and `return` event. Each interception carries a constant overhead of roughly 1–5 µs (depending on platform). At 10 million calls: 10M × 1 µs = 10 s overhead, dwarfing the real 5 s runtime. The profiled result (15 s) is the real time plus profiler overhead. For such code, a statistical profiler (e.g., `py-spy`) or `timeit` is more appropriate.
- C) Incorrect — cProfile does not track memory at all. It has no memory profiling capability. Memory leaks would not be exposed or caused by cProfile.
- D) Incorrect — cProfile does not interact with CPU caches. The overhead is purely software-level Python interpreter hooks, not hardware cache invalidation.

---

## Q32 — dis Module: What It Reveals

> **Week reference:** Week 2

**Mental Model:** The `dis` module disassembles Python bytecode and shows the sequence of CPython virtual machine instructions for a function or code object. It is useful for understanding why two semantically equivalent Python expressions have different performance — e.g., why `x * x` can be faster than `x ** 2` (the `**` operator may invoke `pow()` with an extra `BINARY_OP` dispatch vs a simpler `BINARY_OP` for `*`).

What does `import dis; dis.dis(f)` reveal about function `f`?

- A) The x86 machine-code instructions that the CPU executes when `f` is called
- B) The CPython bytecode instructions (opcodes) for `f`, showing exactly which virtual machine operations execute and in what order
- C) The memory addresses of all objects referenced by `f` at the time of the call
- D) The call graph of all functions that `f` calls, equivalent to a single-level cProfile output

**Answer: B**

- A) Incorrect — `dis` disassembles Python bytecode, not native machine code. CPython compiles Python source to bytecode (.pyc), and `dis` shows that intermediate representation. To see machine code, you would need a JIT compiler (Numba, PyPy) and its own disassembly tools.
- B) Correct — `dis.dis(f)` prints the bytecode of `f` as a sequence of opcodes (e.g., `LOAD_FAST`, `BINARY_OP`, `CALL`, `RETURN_VALUE`) with their arguments and source-line annotations. This lets you compare the bytecode cost of two equivalent Python expressions to understand micro-performance differences.
- C) Incorrect — `dis` operates on the function's code object (static bytecode), not on runtime objects. It knows nothing about which specific objects are in memory during execution; that is the domain of `gc.get_objects()`, `id()`, or a memory profiler.
- D) Incorrect — `dis` does not trace function calls or produce call graphs. It is a static bytecode disassembler, not a dynamic profiler. Viewing call graphs is the role of cProfile with `pstats` or visualisation tools like `snakeviz`.

---

## Q33 — Choosing tottime Sort for Own-Code Bottleneck Hunt

> **Week reference:** Week 2

**Mental Model:** The workflow for profiling is: (1) sort by `cumtime` descending to find the expensive call trees, (2) identify the deepest leaf functions in those trees, (3) switch to `tottime` descending to confirm which leaf is actually doing the most raw work in its own code. Skipping to `tottime` first risks missing a wrapper chain; starting with `cumtime` first risks misidentifying orchestrators as bottlenecks.

After running `python -m cProfile -s cumtime script.py`, a developer sees that `pipeline()` tops the list with `cumtime=120s` but `tottime=0.001s`. She drills in and eventually finds `matrix_ops()` with `tottime=95s` and `cumtime=95s`. She now wants to confirm there are no other expensive own-code functions she may have missed. What should she do next?

- A) Re-run with `python -m cProfile -s tottime script.py` to see all functions ranked by own-code cost, confirming whether `matrix_ops` is the unique top entry
- B) Re-run with `python -m cProfile -s ncalls script.py` to find functions called most frequently
- C) Switch to `line_profiler` immediately, since cProfile has already confirmed the bottleneck
- D) There is nothing more to check; `cumtime` sort always exposes all bottlenecks

**Answer: A**

- A) Correct — sorting by `tottime` produces a definitive ranking of functions by their own-code cost, independently of call-tree depth. This surfaces any other functions that may be doing significant work but appear low in the `cumtime` list (because they are called from a non-top-level path). It is the correct next step to confirm `matrix_ops` is the unique bottleneck.
- B) Incorrect — `ncalls` sorts by call frequency, not cost. A function called 10 million times with 0.0001 s tottime each contributes 1000 s and would appear low in a `tottime` sort unless that sort is used. Sorting by `ncalls` does not reveal cost; it only reveals call frequency.
- C) Incorrect — while `line_profiler` is the natural next step after identifying the bottleneck function, the developer has not yet confirmed that no other expensive function exists at the same depth. Switching to `line_profiler` before completing the `tottime` survey risks missing a second bottleneck of comparable cost.
- D) Incorrect — `cumtime` sort reveals which call trees are expensive, but deep leaf functions with high `tottime` can appear anywhere in the sorted list depending on their callers' cumtime. For example, a leaf called directly from `main()` may have `cumtime = tottime = 90s` but appear 5 rows down because `main`'s cumtime is 120s. Sorting by `cumtime` alone does not guarantee the leaf's `tottime` is immediately visible.

---
