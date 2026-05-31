# Amdahl's Law — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Derive F from Timing Output](#q1-derive-f-from-timing-output)
- [Q2 — Speedup Table and S_max](#q2-speedup-table-and-s_max)
- [Q3 — Reading Speedup from perf_counter Code](#q3-reading-speedup-from-perf_counter-code)
- [Q4 — Wall-Clock vs User Time for Speedup](#q4-wall-clock-vs-user-time-for-speedup)
- [Q5 — Predicting T(p) from F](#q5-predicting-tp-from-f)
- [Q6 — Estimating S_max from Multiple Timing Points](#q6-estimating-s_max-from-multiple-timing-points)
- [Q7 — Two-Phase Program on 4 Cores](#q7-two-phase-program-on-4-cores)
- [Q8 — Does This Timing Code Correctly Measure Speedup?](#q8-does-this-timing-code-correctly-measure-speedup)
- [Q9 — Spotting an Impossible Speedup Claim](#q9-spotting-an-impossible-speedup-claim)
- [Q10 — Computing Parallel Efficiency](#q10-computing-parallel-efficiency)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q11 — Reading F from a Pool.map Timing Loop](#q11-reading-f-from-a-poolmap-timing-loop)
- [Q12 — What Does This Amdahl Sweep Print?](#q12-what-does-this-amdahl-sweep-print)
- [Q13 — Reverse-Solve F from a Pool Benchmark](#q13-reverse-solve-f-from-a-pool-benchmark)
- [Q14 — Minimum p from a Bisection Search](#q14-minimum-p-from-a-bisection-search)
- [Q15 — Does Efficiency Decrease in This Loop?](#q15-does-efficiency-decrease-in-this-loop)
- [Q16 — Spotting the Bug: Wrong Speedup Formula](#q16-spotting-the-bug-wrong-speedup-formula)
- [Q17 — Gustafson Scaled Speedup in Code](#q17-gustafson-scaled-speedup-in-code)
- [Q18 — Predicting T(p) and Checking Against Observed](#q18-predicting-tp-and-checking-against-observed)
- [Q19 — Reading S_max from a Loop That Converges](#q19-reading-s_max-from-a-loop-that-converges)
- [Q20 — Full Amdahl Pipeline: Compute F, S_max, E, T(p)](#q20-full-amdahl-pipeline-compute-f-s_max-e-tp)

---

> Format: Each question includes code, output, or a data table to interpret.
> Exam frequency: **Every exam** — highest priority topic.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--derive-f-from-timing-output)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

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

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets reverse-solving for F and p, efficiency calculations, and multi-program comparisons at scale

---

## Q11 — Reading F from a Pool.map Timing Loop

> **Week reference:** Week 5

A student benchmarks their program across several pool sizes and records wall-clock times:

```python
from multiprocessing import Pool
from time import perf_counter

T1 = 200  # measured serial baseline (seconds)

results = {}
for p in [1, 2, 4, 8]:
    t0 = perf_counter()
    with Pool(processes=p) as pool:
        pool.map(work, chunks)
    results[p] = perf_counter() - t0

print(f"S(4) = {T1 / results[4]:.2f}")
print(f"S(8) = {T1 / results[8]:.2f}")
```

The output is:

```
S(4) = 2.50
S(8) = 3.33
```

Using S(4) = 2.5 and p = 4, what parallel fraction F does Amdahl's Law imply?

- A) F = 0.67
- B) F = 0.80
- C) F = 0.90
- D) F = 0.95

**Answer: B**

- A) Incorrect — F = 0.67 gives S(4) = 1/(0.33 + 0.67/4) = 1/(0.33+0.167) = 1/0.5 = 2.0, not 2.5. This F underestimates the observed speedup.
- B) Correct — F = p(1 - 1/S)/(p-1) = 4(1 - 1/2.5)/3 = 4 × 0.6/3 = 2.4/3 = 0.80. Verify: S(4) = 1/(0.2 + 0.2) = 2.5 ✓.
- C) Incorrect — F = 0.9 gives S(4) = 1/(0.1 + 0.225) = 1/0.325 ≈ 3.08, not 2.5. A higher F would produce a higher speedup than observed.
- D) Incorrect — F = 0.95 gives S(4) = 1/(0.05 + 0.2375) = 1/0.2875 ≈ 3.48, far above the observed 2.5.

---

## Q12 — What Does This Amdahl Sweep Print?

> **Week reference:** Week 5

```python
def amdahl_speedup(F, p):
    return 1 / ((1 - F) + F / p)

F = 0.75
for p in [1, 2, 4, 8]:
    print(f"p={p}  S={amdahl_speedup(F, p):.2f}")
```

Which output is correct?

- A)
```
p=1  S=1.00
p=2  S=1.50
p=4  S=2.00
p=8  S=2.91
```
- B)
```
p=1  S=1.00
p=2  S=1.60
p=4  S=2.29
p=8  S=2.91
```
- C)
```
p=1  S=1.00
p=2  S=1.60
p=4  S=2.29
p=8  S=3.20
```
- D)
```
p=1  S=0.75
p=2  S=1.50
p=4  S=3.00
p=8  S=6.00
```

**Answer: B**

- A) Incorrect — p=2: 1/(0.25+0.375)=1/0.625=1.60, not 1.50. The value 1.50 would require a larger serial fraction than 25%.
- B) Correct — p=1: 1.00; p=2: 1/(0.25+0.375)=1/0.625=1.60; p=4: 1/(0.25+0.1875)=1/0.4375≈2.286→2.29; p=8: 1/(0.25+0.09375)=1/0.34375≈2.909→2.91.
- C) Incorrect — p=8: 1/0.34375≈2.91, not 3.20. S=3.20 for p=8 with F=0.75 would require a denominator of 1/3.20≈0.3125, but (1-0.75)+0.75/8=0.34375.
- D) Incorrect — p=1 should always give S=1.0 by definition (serial baseline). Option D shows S=0.75 at p=1, which confuses F with S(1).

---

## Q13 — Reverse-Solve F from a Pool Benchmark

> **Week reference:** Week 5

A student collects these wall-clock times using `multiprocessing.Pool`:

```python
# T(1) = 90s (serial baseline)
# T(3) = 45s (measured with Pool(3))

T1, Tp, p = 90, 45, 3
S = T1 / Tp
F = p * (1 - 1/S) / (p - 1)
print(f"S = {S:.2f},  F = {F:.2f}")
```

What does the code print?

- A) S = 2.00,  F = 0.75
- B) S = 2.00,  F = 1.00
- C) S = 0.50,  F = 0.75
- D) S = 2.00,  F = 0.67

**Answer: A**

- A) Correct — S = 90/45 = 2.0; F = 3(1 - 1/2)/(3-1) = 3 × 0.5 / 2 = 1.5/2 = 0.75. Verify: S(3) = 1/(0.25 + 0.75/3) = 1/(0.25+0.25) = 2.0 ✓.
- B) Incorrect — F = 1.00 would mean the program is entirely parallel with no serial fraction at all. S(3) = 3.0 with F=1. But S = 2.0 < 3.0, so F < 1.
- C) Incorrect — S = T1/Tp = 90/45 = 2.0, not 0.50. S < 1 would indicate slowdown; inverting numerator and denominator (Tp/T1) gives 0.5, a common mistake.
- D) Incorrect — F = 0.67 gives S(3) = 1/(0.33 + 0.22) = 1/0.556 ≈ 1.80, not 2.0. F = 0.67 underestimates the observed speedup; the program is more parallel than 67%.

---

## Q14 — Minimum p from a Bisection Search

> **Week reference:** Week 5

A student finds the minimum cores needed for a target speedup using a loop:

```python
F = 0.9
target = 5.0

p = 1
while True:
    S = 1 / ((1 - F) + F / p)
    if S >= target:
        break
    p += 1

print(f"Minimum p = {p},  S = {S:.2f}")
```

What does this code print?

- A) Minimum p = 5,  S = 5.00
- B) Minimum p = 9,  S = 5.00
- C) Minimum p = 10,  S = 5.26
- D) Minimum p = 45,  S = 5.00

**Answer: B**

- A) Incorrect — S(5) = 1/(0.1 + 0.9/5) = 1/(0.1+0.18) = 1/0.28 ≈ 3.57 < 5.0. p = 5 does not meet the target.
- B) Correct — Solve: 5 = 1/(0.1 + 0.9/p) → 0.9/p = 0.2 - 0.1 = 0.1 → p = 9. S(9) = 1/(0.1 + 0.1) = 1/0.2 = 5.0 exactly. The loop exits at p = 9.
- C) Incorrect — p = 10 gives S(10) = 1/(0.1 + 0.09) = 1/0.19 ≈ 5.26, which exceeds the target. But p = 9 already meets S = 5.0 exactly, so the loop breaks at 9, not 10.
- D) Incorrect — p = 45 would be needed for a much higher target speedup close to S_max = 10. For S = 5.0 with F = 0.9, the exact solution is p = 9.

---

## Q15 — Does Efficiency Decrease in This Loop?

> **Week reference:** Week 5

```python
F = 0.85

for p in [1, 2, 4, 8, 16, 32]:
    S = 1 / ((1 - F) + F / p)
    E = S / p
    print(f"p={p:2d}  E={E:.3f}")
```

Which statement about the printed values is correct?

- A) E stays approximately constant at 0.85 because F is fixed.
- B) E decreases monotonically from 1.000 at p=1 toward 0 as p grows.
- C) E increases toward 1.0 as p grows because more cores improve efficiency.
- D) E first increases then decreases, peaking around p = 8.

**Answer: B**

- A) Incorrect — 0.85 is the parallel fraction F, not the efficiency. E = S(p)/p falls strictly as p increases because speedup grows sub-linearly.
- B) Correct — At p=1: S=1, E=1.000. At p=2: S=1/(0.15+0.425)≈1.739, E≈0.870. At p=4: S≈2.759, E≈0.690. At p=8: S≈4.923, E≈0.615. At p=16: E≈0.308. At p=32: even lower. E decreases monotonically toward 0 as p → ∞ due to the serial bottleneck.
- C) Incorrect — Efficiency can never exceed 1 and never increases with more cores in the Amdahl model. The serial fraction wastes a growing share of each core's capacity as p increases.
- D) Incorrect — E is strictly decreasing in the Amdahl model; it has no peak or trough. Super-linear speedup (where E > 1 is possible briefly due to cache effects) is outside the scope of Amdahl's Law.

---

## Q16 — Spotting the Bug: Wrong Speedup Formula

> **Week reference:** Week 5

A student writes a function to compute parallel efficiency but makes a subtle error:

```python
def efficiency(T1, Tp, p):
    S = Tp / T1        # BUG: should be T1/Tp
    E = S / p
    return E

print(efficiency(100, 20, 8))
```

What does this code print, and why is it wrong?

- A) Prints 0.025 — the bug inverts speedup; the correct answer should be 0.625.
- B) Prints 0.025 — the bug inverts speedup; the correct answer should be 1.60.
- C) Prints 1.600 — the bug divides by p instead of multiplying.
- D) Prints 0.625 — the code is correct as written.

**Answer: A**

- A) Correct — S = Tp/T1 = 20/100 = 0.20 (inverted). E = 0.20/8 = 0.025. The correct formula is S = T1/Tp = 100/20 = 5.0; E = 5.0/8 = 0.625.
- B) Incorrect — The code prints 0.025 (correct identification of the bug), but the correct efficiency is 0.625, not 1.60. 1.60 would imply super-linear speedup (S > p), which is not the case here.
- C) Incorrect — The code prints 0.025, not 1.600. 1.600 would result from computing Tp/T1 without dividing by p, i.e., just S_wrong = 20/100 × something; 20/100 = 0.2, not 1.6.
- D) Incorrect — The formula S = Tp/T1 divides parallel time by serial time. Since T_parallel < T_serial for a speedup, this gives S < 1, representing a slowdown rather than a speedup.

---

## Q17 — Gustafson Scaled Speedup in Code

> **Week reference:** Week 5

```python
def gustafson(alpha, p):
    return p - alpha * (p - 1)

def amdahl(F, p):
    return 1 / ((1 - F) + F / p)

alpha = 0.05   # serial fraction for Gustafson
F     = 1 - alpha  # = 0.95

p = 20
print(f"Gustafson: {gustafson(alpha, p):.2f}")
print(f"Amdahl:    {amdahl(F, p):.2f}")
```

What does this code print?

- A) Gustafson: 20.00 / Amdahl: 20.00
- B) Gustafson: 19.05 / Amdahl: 10.26
- C) Gustafson: 19.95 / Amdahl: 20.00
- D) Gustafson: 19.05 / Amdahl: 20.00

**Answer: B**

- A) Incorrect — Gustafson: 20 - 0.05×19 = 20 - 0.95 = 19.05, not 20. Amdahl: 1/(0.05 + 0.95/20) = 1/0.0975 ≈ 10.26, not 20.
- B) Correct — Gustafson: 20 - 0.05×19 = 19.05. Amdahl: 1/(0.05 + 0.0475) = 1/0.0975 ≈ 10.26. Gustafson gives a much higher scaled speedup because it assumes the workload grows with p.
- C) Incorrect — Gustafson gives 19.05, not 19.95. 19.95 would result from alpha = 0.005 (0.5% serial), not 0.05. Amdahl is also not 20 at finite p=20; S_max = 20 requires p→∞.
- D) Incorrect — Gustafson is correctly 19.05, but Amdahl at p=20 with F=0.95 is 10.26, not 20. Amdahl only reaches S_max=20 as p→∞.

---

## Q18 — Predicting T(p) and Checking Against Observed

> **Week reference:** Week 5

```python
T1 = 160    # serial runtime (seconds)
F  = 0.875  # parallel fraction
p  = 8

T_predicted = T1 * ((1 - F) + F / p)
T_observed  = 25  # seconds

print(f"Predicted T({p}) = {T_predicted:.1f}s")
print(f"Observed  T({p}) = {T_observed}s")
print(f"Match: {abs(T_predicted - T_observed) < 1}")
```

What does this code print?

- A) Predicted T(8) = 37.5s / Observed T(8) = 25s / Match: False
- B) Predicted T(8) = 25.0s / Observed T(8) = 25s / Match: True
- C) Predicted T(8) = 20.0s / Observed T(8) = 25s / Match: False
- D) Predicted T(8) = 37.5s / Observed T(8) = 25s / Match: True

**Answer: A**

- A) Correct — T_predicted = 160 × ((1-0.875) + 0.875/8) = 160 × (0.125 + 0.109375) = 160 × 0.234375 = 37.5s. Since |37.5 - 25| = 12.5 > 1, Match: False.
- B) Incorrect — T_predicted = 25.0 would require the denominator to be 25/160 = 0.15625: (1-F)+F/8 = 0.15625 → F = 0.9. The code uses F = 0.875, not 0.9.
- C) Incorrect — T_predicted = 20.0 would require denominator = 20/160 = 0.125: (1-F)+F/8 = 0.125 → F = 1.0 (fully parallel). F = 0.875 ≠ 1.0.
- D) Incorrect — T_predicted is correctly 37.5 but Match would be False, not True. |37.5 - 25| = 12.5, which is not less than 1.

---

## Q19 — Reading S_max from a Loop That Converges

> **Week reference:** Week 5

```python
F = 0.9

prev = 0
for p in [10, 100, 1000, 10000, 100000]:
    S = 1 / ((1 - F) + F / p)
    print(f"p={p:6d}  S={S:.4f}  delta={S-prev:.4f}")
    prev = S
```

The first few lines of output are:

```
p=    10  S=5.2632  delta=5.2632
p=   100  S=9.0909  delta=3.8277
p=  1000  S=9.9010  delta=0.8100
p= 10000  S=9.9900  delta=0.0890
p=100000  S=9.9990  delta=0.0090
```

What does this output reveal about S_max, and what is the dominant factor at large p?

- A) S_max = 10; the delta per decade shrinks by ~10× each step, showing S approaches 1/(1-F) asymptotically.
- B) S_max = 9; the series converges at S = 9 because F = 0.9 directly equals S_max.
- C) S_max = ∞; the output shows S will keep growing without bound as p → ∞.
- D) S_max = 10; the delta per decade stays constant, showing linear convergence.

**Answer: A**

- A) Correct — S_max = 1/(1-0.9) = 10. Each decade of p reduces the delta by roughly 10× (3.83→0.81→0.089→0.009), showing that S(p) converges to 10 asymptotically. At large p, F/p → 0 and the denominator is dominated by (1-F) = 0.1.
- B) Incorrect — S_max = 9 would require 1/(1-F) = 9, meaning 1-F = 1/9 ≈ 0.111, so F ≈ 0.889. For F = 0.9, the ceiling is exactly 10.
- C) Incorrect — The deltas are clearly shrinking: 5.26, 3.83, 0.81, 0.089, 0.009. The series converges; it does not grow without bound.
- D) Incorrect — The delta per decade is not constant; it shrinks by a factor of approximately 10 each decade (geometric convergence, not linear). This is because F/p contributes a term that falls by 10× when p increases by 10×.

---

## Q20 — Full Amdahl Pipeline: Compute F, S_max, E, T(p)

> **Week reference:** Week 5

```python
T1 = 100   # serial runtime (seconds)
T4 = 40    # parallel runtime on 4 cores
p  = 4

S = T1 / T4
F = p * (1 - 1/S) / (p - 1)
S_max = 1 / (1 - F)
E = S / p
T8 = T1 * ((1 - F) + F / 8)

print(f"S(4)  = {S:.2f}")
print(f"F     = {F:.2f}")
print(f"S_max = {S_max:.2f}")
print(f"E(4)  = {E:.3f}")
print(f"T(8)  = {T8:.1f}s")
```

What does this code print?

- A) S(4)=2.50 / F=0.80 / S_max=5.00 / E(4)=0.625 / T(8)=30.0s
- B) S(4)=2.50 / F=0.80 / S_max=5.00 / E(4)=0.625 / T(8)=20.0s
- C) S(4)=2.50 / F=0.80 / S_max=5.00 / E(4)=0.313 / T(8)=30.0s
- D) S(4)=2.50 / F=0.75 / S_max=4.00 / E(4)=0.625 / T(8)=30.0s

**Answer: A**

- A) Correct — S=100/40=2.5; F=4(1-0.4)/3=4×0.6/3=0.80; S_max=1/0.2=5.00; E=2.5/4=0.625; T(8)=100×(0.2+0.1)=100×0.3=30.0s.
- B) Incorrect — T(8) = 20s would imply S(8) = 5.0 = S_max, which requires p→∞. With F=0.8, S(8)=1/0.3=3.33, giving T(8)=100/3.33≈30.0s, not 20s.
- C) Incorrect — E(4) = 0.313 = 0.625/2 results from dividing S by p twice (i.e., computing S/p² instead of S/p). E = S/p = 2.5/4 = 0.625.
- D) Incorrect — F=0.75 would give S(4)=1/(0.25+0.1875)=1/0.4375≈2.29, not 2.5. The formula F=p(1-1/S)/(p-1) with S=2.5 and p=4 gives exactly 0.80.
