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

- A) Incorrect — `percall` is time per call; `cumtime` and `tottime` are totals across all calls.
- B) Correct — `cumtime` = own time + all callee time; `tottime` = own code only, excluding callees.
- C) Incorrect — Both columns measure time, not call counts.
- D) Incorrect — They are equal only when a function makes NO sub-calls at all; recursion is a special case.

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

- A) Incorrect — `tottime` misses time spent in callees; `compute()` shows only 0.001s tottime but 40.3s cumtime.
- B) Incorrect — Frequency alone doesn't determine bottleneck; a function called once can dominate total time.
- C) Correct — `cumtime` accounts for all work done within a call path, making it the right metric for bottleneck identification.
- D) Incorrect — `percall` is useful for per-invocation cost but doesn't capture total program impact.

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

- A) Incorrect — The workload scales with ncalls; 1000 samples means 1000 calls.
- B) Incorrect — This would be correct for 100 samples, not 1000.
- C) Correct — projected = percall × production_ncalls = 0.505 × 1000 = 505 s.
- D) Incorrect — `percall` is per-call cost; total time still scales with the number of calls.

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

- A) Incorrect — `load_data` has ncalls=1 and does not scale with sample count.
- B) Incorrect — This assumes ncalls grows to 1000, but the function is called once.
- C) Correct — ncalls=1 means this is a one-time fixed cost independent of production workload size.
- D) Incorrect — Amortizing the cost doesn't change actual runtime; the function still takes ~1.2 s total.

---

## Q5 — Line Profiler: What Does the Hits Column Mean?

In `line_profiler` output, what does the `Hits` column represent for a given line of code?

- A) The number of CPU cache hits when that line was executed
- B) The number of times that line was executed during the profiled run
- C) The total memory allocated on that line
- D) The number of function calls made from that line

**Answer: B**

- A) Incorrect — `Hits` has nothing to do with CPU cache behavior.
- B) Correct — `Hits` counts how many times the profiler observed execution of that specific line.
- C) Incorrect — Memory allocation is not tracked by `line_profiler`; use `memory_profiler` for that.
- D) Incorrect — Function calls made from a line are not what `Hits` counts.

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

- A) Incorrect — Line 5 shows Hits=5001 (n_steps iterations + 1 for the final StopIteration check); the loop body Hits (5000) is the reliable indicator of the number of iterations.
- B) Incorrect — 9990 is time in microseconds, not a loop count.
- C) Correct — The loop body (line 6) executed 5000 times, so `n_steps = 5000`.
- D) Incorrect — `Hits` on the loop body directly gives the number of iterations.

---

## Q7 — FLOP/s Calculation

A numerical function performs 8 floating-point operations per iteration, runs for 10,000 iterations, and completes in 0.04 seconds. What is its throughput in FLOP/s?

- A) 80,000 FLOP/s (8 × 10,000 / 0.04 is wrong — divide by Hits not time)
- B) 2,000,000 FLOP/s (= 8 × 10,000 / 0.04)
- C) 250,000 FLOP/s (= 10,000 / 0.04)
- D) 200 FLOP/s (= 8 / 0.04)

**Answer: B**

- A) Incorrect — This is the correct numerical result but the parenthetical explanation is wrong; dividing by time is correct.
- B) Correct — FLOP/s = (FLOPs_per_iter × n_iters) / total_time = (8 × 10,000) / 0.04 = 2,000,000 FLOP/s = 2×10^6.
- C) Incorrect — This omits the 8 FLOPs per iteration.
- D) Incorrect — This ignores the number of iterations entirely.

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

- A) Incorrect — The profile was run on 100 samples; production has 50× more, so runtime scales accordingly.
- B) Incorrect — This would be the answer for 1000 samples (10× scale), not 5000 (50×).
- C) Correct — projected = percall × production_ncalls = 0.02 × 5000 = 100 s.
- D) Incorrect — percall being constant means total time does scale proportionally with ncalls.

---

## Q9 — nsys Report Sections

When analyzing a GPU program with NVIDIA Nsight Systems (`nsys`), which report sections correspond to GPU kernel execution time and host-to-device / device-to-host memory transfer times respectively?

- A) `gpukernsum` for kernel time; `cpuexecsum` for memory transfer time
- B) `gpukernsum` for kernel time; `gpumemtimesum` for memory transfer time
- C) `gpumemtimesum` for kernel time; `gpumemsizesum` for memory transfer time
- D) `cudaapisum` for kernel time; `gpukernsum` for memory transfer time

**Answer: B**

- A) Incorrect — `cpuexecsum` reports CPU-side API call times, not GPU memory transfers.
- B) Correct — `gpukernsum` = GPU kernel execution times; `gpumemtimesum` = HtoD and DtoH transfer durations.
- C) Incorrect — `gpumemtimesum` is for transfer time, not kernel time; `gpumemsizesum` is transfer sizes.
- D) Incorrect — `cudaapisum` tracks CUDA API calls on the CPU side; `gpukernsum` is kernel time, not transfer time.

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

- A) Incorrect — Dividing by 4 has no physical meaning here.
- B) Incorrect — Multiplying instead of dividing gives wrong units.
- C) Correct — bandwidth = total_size / total_time = 2500 MB / 0.25 s = 10,000 MB/s = 10 GB/s.
- D) Incorrect — 2500 MB/s would only be correct if transfer time were exactly 1 s.

---

## Q11 — Choosing the Right Profiler

You want to identify which specific lines within a slow function are responsible for most of the execution time. Which tool and command should you use?

- A) `python -m cProfile script.py` — cProfile automatically provides line-level timing
- B) `kernprof -l -v script.py` with `@profile` decorator on the function — this is `line_profiler`
- C) `nsys profile python script.py` — Nsight Systems provides line-level Python timing
- D) `time python script.py` — wall-clock time reveals which lines are slow

**Answer: B**

- A) Incorrect — cProfile provides function-level granularity only, not line-by-line breakdowns.
- B) Correct — `kernprof -l -v` runs `line_profiler`; decorating a function with `@profile` enables per-line timing for that function.
- C) Incorrect — `nsys` profiles GPU activity and CUDA API calls, not Python source line timing.
- D) Incorrect — `time` measures total wall-clock time for the whole program, with no per-line information.

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

- A) Incorrect — 45 s is `cumtime`, which includes all callee time.
- B) Incorrect — 40 s is the time spent inside `compute()`, not in `main()`'s own code.
- C) Correct — `tottime` reports time spent in the function's own instructions, excluding any callees; 5 s is `main()`'s own contribution.
- D) Incorrect — 0.5 s is `compute()`'s own code time, unrelated to `main()`'s tottime.

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

- A) Incorrect — Higher call count doesn't automatically make a function the bottleneck; you must compute projected total time.
- B) Correct — `transform_dataset` stays at ~30 s; `validate_record` projects to 50,000 × 0.0005 s/call = 25 s. 30 s > 25 s, so `transform_dataset` is still the larger bottleneck.
- C) Incorrect — 25 s ≠ 30 s; they are not equal at production scale.
- D) Incorrect — Comparing `tottime` values ignores callee time and misrepresents actual impact.

---
