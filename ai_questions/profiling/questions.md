# Profiling — MCQ Practice

> Topics: cProfile columns, line profiler Hits, scaling to production workload, FLOP/s, nsys.
> Exam frequency: **Every exam**.

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
