# Amdahl's Law — Code-Based MCQ Practice

> Format: Each question includes code, output, or a data table to interpret.
> Exam frequency: **Every exam** — highest priority topic.

---

## Q1 — Derive F from Timing Output

> **Week reference:** Week 5

You run a program on 1 core and then on 4 cores and observe the following:

```
Serial run:   T(1) = 80s
Parallel run: T(4) = 32s
```

Using the formula derived from Amdahl's Law:

```
F = p * (1 - 1/S) / (p - 1)
```

where S = T(1) / T(p) and p = number of cores. What is the parallel fraction F?

- A) F = 0.60
- B) F = 0.75
- C) F = 0.80
- D) F = 0.90

**Answer: C**

- A) Incorrect — This would imply S ≈ 2.0, but S = 80/32 = 2.5 here.
- B) Incorrect — This would result from a different speedup value (S ≈ 2.17).
- C) Correct — S = 80/32 = 2.5; F = 4×(1 - 1/2.5)/(4-1) = 4×0.6/3 = 0.80.
- D) Incorrect — F = 0.9 would give S = 1/(0.1 + 0.9/4) = 1/0.325 ≈ 3.08, not 2.5.

---

## Q2 — Speedup Table and S_max

> **Week reference:** Week 5

A student measures the speedup of their program at various core counts:

```
Cores | Speedup
------|---------
1     | 1.0
2     | 1.8
4     | 3.0
8     | 4.5
16    | 5.6
```

The speedup appears to be approaching a plateau near S_max ≈ 6. Using Amdahl's Law (S_max = 1 / (1 - F)), what is the parallel fraction F implied by S_max = 6?

- A) F = 0.75
- B) F = 0.80
- C) F ≈ 0.833
- D) F = 0.90

**Answer: C**

- A) Incorrect — F = 0.75 gives S_max = 1/(1-0.75) = 4.0, not 6.
- B) Incorrect — F = 0.80 gives S_max = 1/(1-0.80) = 5.0, not 6.
- C) Correct — S_max = 1/(1-F) = 6 → 1-F = 1/6 → F = 5/6 ≈ 0.833.
- D) Incorrect — F = 0.90 gives S_max = 1/(1-0.90) = 10.0, not 6.

---

## Q3 — Reading Speedup from perf_counter Code

> **Week reference:** Week 5

The following code times a serial and a parallel run, then prints speedup:

```python
from time import perf_counter

t1 = perf_counter(); run_serial(); t2 = perf_counter()
t3 = perf_counter(); run_parallel(8); t4 = perf_counter()
print(f"Speedup: {(t2-t1)/(t4-t3):.2f}")
```

Output:

```
Speedup: 3.20
```

Using the formula F = p × (1 - 1/S) / (p - 1) with p = 8 and S = 3.20, what is F?

- A) F ≈ 0.71
- B) F ≈ 0.786
- C) F ≈ 0.84
- D) F ≈ 0.90

**Answer: B**

- A) Incorrect — This underestimates; check arithmetic: 1 - 1/3.2 = 1 - 0.3125 = 0.6875.
- B) Correct — F = 8 × (1 - 1/3.2) / (8 - 1) = 8 × 0.6875 / 7 = 5.5 / 7 ≈ 0.786.
- C) Incorrect — This would require a higher speedup than 3.20.
- D) Incorrect — F = 0.9, p = 8 gives S = 1/(0.1 + 0.9/8) = 1/0.2125 ≈ 4.71, not 3.20.

---

## Q4 — Wall-Clock vs User Time for Speedup

> **Week reference:** Week 5

A student wants to measure parallel speedup. They run the following on an 8-core machine using `/usr/bin/time`:

```
Serial run:
  real    0m40.000s
  user    0m39.800s

Parallel run (8 processes):
  real    0m8.000s
  user    0m58.000s
```

The student computes speedup as `user_serial / user_parallel = 39.8 / 58.0 ≈ 0.69` and concludes the parallel code is slower. What is wrong?

- A) The student should divide `real` times, not `user` times.
- B) The student should use `sys` time instead of `user` time.
- C) The student should sum all per-process `user` times and then divide.
- D) The speedup calculation formula is wrong; it should be `user_parallel / user_serial`.

**Answer: A**

- A) Correct — Amdahl speedup is based on wall-clock (real) time: S = 40/8 = 5.0. `user` time sums CPU time across all processes, so 8 processes × ~7.25s = ~58s; dividing user times gives nonsense for speedup.
- B) Incorrect — `sys` time measures kernel calls, not computation; it is not the right metric either.
- C) Incorrect — Summing per-process user time gives total CPU work, not elapsed wall time, which is irrelevant for speedup.
- D) Incorrect — Inverting the ratio would give values > 1 here, but it is still the wrong measure since it uses user time.

---

## Q5 — Predicting T(p) from F

> **Week reference:** Week 5

A program has T(1) = 100s with a parallel fraction F = 0.9. The following function predicts the parallel runtime:

```python
def predicted_time(T1, F, p):
    return (1 - F) * T1 + F * T1 / p

print(predicted_time(100, 0.9, 10))
```

What does this code print?

- A) 10.0
- B) 19.0
- C) 9.0
- D) 91.0

**Answer: B**

- A) Incorrect — 10.0 would be the result if the entire program were parallelisable (F=1).
- B) Correct — Serial part: (1-0.9)×100 = 10s. Parallel part: 0.9×100/10 = 9s. Total = 10 + 9 = 19s.
- C) Incorrect — 9.0 is only the parallelisable portion divided by cores, missing the serial part.
- D) Incorrect — 91.0 would result from computing F×T1 + (1-F)×T1/p, which confuses serial and parallel fractions.

---

## Q6 — Estimating S_max from Multiple Timing Points

> **Week reference:** Week 5

You collect these wall-clock measurements for a fixed workload:

```
p=2:  T(2) = 60s
p=4:  T(4) = 40s
p=8:  T(8) = 30s
```

You know T(1) = 100s. Using the pair (p=2, T=60s), you estimate F, then compute S_max = 1/(1-F). Which answer best approximates S_max?

- A) S_max ≈ 3.0
- B) S_max ≈ 5.0
- C) S_max ≈ 6.7
- D) S_max ≈ 10.0

**Answer: B**

- A) Incorrect — S_max = 3 implies F ≈ 0.667; check: S(2) = 1/(0.333+0.333) = 1.5, giving T(2)=67s, not 60s.
- B) Correct — S(2) = 100/60 = 1.667; F = 2×(1-1/1.667)/(2-1) = 2×0.4/1 = 0.80; S_max = 1/(1-0.8) = 5.0.
- C) Incorrect — F ≈ 0.85 gives S(2) ≈ 1.73, T(2) ≈ 57.7s — close but not matching 60s.
- D) Incorrect — F = 0.9 gives S(2) = 1/(0.1+0.45) = 1.82, T(2) ≈ 55s, not 60s.

---

## Q7 — Two-Phase Program on 4 Cores

> **Week reference:** Week 5

A program has two sequential phases:

- **Phase 1 (serial):** 20 seconds — cannot be parallelised
- **Phase 2 (parallel):** 80 seconds on 1 core — fully parallelisable

The following code models the runtime on p cores:

```python
T1_serial   = 20   # phase 1, fixed
T1_parallel = 80   # phase 2, single-core time
p = 4

T_total = T1_serial + T1_parallel / p
S = (T1_serial + T1_parallel) / T_total
print(f"T({p}) = {T_total}s,  S = {S:.2f}")
```

What does this code print?

- A) T(4) = 30s,  S = 3.33
- B) T(4) = 40s,  S = 2.50
- C) T(4) = 25s,  S = 4.00
- D) T(4) = 45s,  S = 2.22

**Answer: B**

- A) Incorrect — T(4) = 30 would require T1_parallel/p = 10, so p=8 not p=4.
- B) Correct — T(4) = 20 + 80/4 = 20 + 20 = 40s; S = 100/40 = 2.50.
- C) Incorrect — T(4) = 25 would mean T1_parallel/p = 5, impossible for p=4 and 80s parallel phase.
- D) Incorrect — T(4) = 45 would require the parallel phase to take 25s, implying p ≈ 3.2.

---

## Q8 — Does This Timing Code Correctly Measure Speedup?

> **Week reference:** Week 5

A colleague measures parallel speedup with the following snippet:

```python
from multiprocessing import Pool
from time import perf_counter

# Serial baseline
t0 = perf_counter()
result_serial = compute_serial(data)
t1 = perf_counter()
T_serial = t1 - t0

# Parallel run
t2 = perf_counter()
with Pool(processes=8) as pool:
    result_parallel = pool.map(compute_chunk, chunks)
# pool.__exit__ calls pool.terminate() then pool.join()
t3 = perf_counter()
T_parallel = t3 - t2

print(f"Speedup: {T_serial / T_parallel:.2f}")
```

Does this code correctly measure wall-clock speedup?

- A) No — the timer should stop inside the `with` block, before the pool closes.
- B) No — `perf_counter` measures CPU time, not wall-clock time.
- C) Yes — `perf_counter` measures wall-clock time, and the timer correctly wraps all parallel work including joining.
- D) No — the pool startup overhead is included in T_parallel, making the speedup appear lower than it really is.

**Answer: C**

- A) Incorrect — Stopping before pool exit would miss the join/cleanup; you need workers to finish before recording end time.
- B) Incorrect — `perf_counter` returns wall-clock (elapsed) time, not CPU time. `process_time()` measures CPU time.
- C) Correct — `perf_counter` is wall-clock; wrapping the entire `with Pool(...)` block (which joins on exit) correctly captures total parallel elapsed time.
- D) Incorrect — While pool startup overhead is included, this is intentional and standard practice; it gives a realistic speedup measure. The code is methodologically correct.

---

## Q9 — Spotting an Impossible Speedup Claim

> **Week reference:** Week 5

A colleague reports that their program achieves a speedup of S(16) = 12 with a measured parallel fraction F = 0.9. You check their claim using Amdahl's Law:

```python
F = 0.9
p = 16
S_amdahl = 1 / ((1 - F) + F / p)
print(f"Theoretical S({p}) = {S_amdahl:.2f}")
```

What does the code print, and what does this reveal about the colleague's claim?

- A) Prints 9.00 — the claim of 12 is impossible; S_max with F=0.9 is only 10.
- B) Prints 6.40 — the claim of 12 is impossible; Amdahl's Law gives at most ~6.4 for F=0.9, p=16.
- C) Prints 12.00 — the claim is consistent with Amdahl's Law.
- D) Prints 14.40 — the claim of 12 is actually conservative; the true speedup should be higher.

**Answer: B**

- A) Incorrect — S_max = 1/(1-0.9) = 10 is the absolute ceiling (infinite cores), but at p=16 the value is lower.
- B) Correct — S(16) = 1/(0.1 + 0.9/16) = 1/(0.1 + 0.05625) = 1/0.15625 = 6.40. A claim of 12 violates Amdahl's Law for F=0.9.
- C) Incorrect — The formula gives 6.40, not 12.00.
- D) Incorrect — Amdahl's Law gives an upper bound; no value above S_max = 10 (and certainly not above S(16) = 6.4) is achievable without super-linear effects.

---

## Q10 — Computing Parallel Efficiency

> **Week reference:** Week 5

The following code computes parallel efficiency from timing measurements:

```python
T1 = 100   # serial runtime (seconds)
Tp = 15    # parallel runtime on p cores (seconds)
p  = 8

S = T1 / Tp
E = S / p
print(f"Efficiency: {E:.2f}")
```

What does this code print?

- A) Efficiency: 0.75
- B) Efficiency: 0.83
- C) Efficiency: 1.00
- D) Efficiency: 6.67

**Answer: B**

- A) Incorrect — E = 0.75 would require S = 6.0 (i.e., Tp = 100/6 ≈ 16.67s), not Tp = 15s.
- B) Correct — S = 100/15 ≈ 6.667; E = 6.667/8 ≈ 0.833, which rounds to 0.83.
- C) Incorrect — E = 1.0 (perfect efficiency) would require S = p = 8, meaning Tp = 100/8 = 12.5s.
- D) Incorrect — 6.67 is the speedup S, not the efficiency; efficiency divides S by p.

---
