# Amdahl's Law — MCQ Practice

> Topics: S(p) formula, solving for F, maximum speedup, efficiency, serial time calculations.
> Exam frequency: **Every exam** — highest priority topic.

---

## Q1 — Compute Speedup F=0.9 p=8
> **Week reference:** Week 5

A program has a parallel fraction F = 0.9. Using Amdahl's Law, what is the speedup S(8) when running on 8 cores?

- A) 5.0
- B) 7.2
- C) 4.71
- D) 3.33

**Answer: C**

- A) Incorrect — S(8) = 5.0 would require F ≈ 0.933; it ignores the 0.1 serial fraction correctly.
- B) Incorrect — 7.2 ≈ 0.9 × 8, which mistakenly multiplies F by p instead of applying Amdahl's formula.
- C) Correct — S(8) = 1 / ((1 - 0.9) + 0.9/8) = 1 / (0.1 + 0.1125) = 1 / 0.2125 ≈ 4.71.
- D) Incorrect — 3.33 = 1 / 0.3 corresponds to F = 0.8 with p = 8, not F = 0.9.

---

## Q2 — Solve for F Then S_max
> **Week reference:** Week 5

Measurements show that a program achieves S(3) = 2.5 on 3 cores. Using F = p(1 - 1/S(p)) / (p - 1), what is the parallel fraction F, and what is the theoretical maximum speedup S_max?

- A) F = 0.8, S_max = 5
- B) F = 0.9, S_max = 10
- C) F = 0.75, S_max = 4
- D) F = 0.6, S_max = 2.5

**Answer: B**

- A) Incorrect — F = 3(1 - 1/2.5) / 2 = 3(0.6)/2 = 0.9, not 0.8; S_max = 1/0.2 = 10, not 5.
- B) Correct — F = 3(1 - 0.4)/2 = 0.9; S_max = 1/(1 - 0.9) = 10.
- C) Incorrect — F = 0.75 would give S(3) = 1/(0.25 + 0.25) = 2.0, not 2.5.
- D) Incorrect — F = 0.6 would give S(3) = 1/(0.4 + 0.2) ≈ 1.67, not 2.5.

---

## Q3 — Derive F From Saturation Plot
> **Week reference:** Week 5

A speedup-vs-cores plot shows that speedup levels off and never exceeds 4, regardless of how many cores are added. What is the parallel fraction F of this program?

- A) F = 0.80
- B) F = 0.75
- C) F = 0.25
- D) F = 0.90

**Answer: B**

- A) Incorrect — F = 0.80 gives S_max = 1/(1 - 0.80) = 5, not 4.
- B) Correct — S_max = 4 means 1/(1 - F) = 4, so 1 - F = 0.25, thus F = 0.75.
- C) Incorrect — F = 0.25 would give S_max = 1/0.75 ≈ 1.33, far below 4.
- D) Incorrect — F = 0.90 gives S_max = 10, not 4.

---

## Q4 — Achievability of Target Speedup
> **Week reference:** Week 5

A program has parallel fraction F = 0.8. A developer wants to achieve a speedup of at least 4× using 8 cores. Is this achievable?

- A) Yes — S(8) = 4.0 exactly.
- B) No — S(8) ≈ 3.33, which falls short of 4.
- C) Yes — S(8) ≈ 4.71 with F = 0.8.
- D) No — S_max = 4.0, so S(8) cannot reach 4.

**Answer: B**

- A) Incorrect — S(8) = 1/(0.2 + 0.1) = 3.33, not 4.0.
- B) Correct — S(8) = 1/((1 - 0.8) + 0.8/8) = 1/(0.2 + 0.1) = 1/0.3 ≈ 3.33, below the target of 4.
- C) Incorrect — S(8) ≈ 4.71 corresponds to F = 0.9, not F = 0.8.
- D) Incorrect — S_max = 1/(1 - 0.8) = 5, so 4× is theoretically possible with infinite cores but not with p = 8.

---

## Q5 — Recover T(1) From T(4)
> **Week reference:** Week 5

A parallel run on 4 cores takes T(4) = 10 minutes, and the program has parallel fraction F = 0.8. Using T(1) = T(p) × S(p), what is the single-core runtime T(1)?

- A) 20 minutes
- B) 25 minutes
- C) 40 minutes
- D) 12.5 minutes

**Answer: B**

- A) Incorrect — 20 min would imply S(4) = 2.0, which requires F = 0.667, not 0.8.
- B) Correct — S(4) = 1/((1 - 0.8) + 0.8/4) = 1/(0.2 + 0.2) = 2.5; T(1) = 10 × 2.5 = 25 minutes.
- C) Incorrect — 40 min would imply S(4) = 4, which would require F ≈ 1.0 (fully parallel).
- D) Incorrect — 12.5 min would imply S(4) = 0.8, meaning the parallel version is slower than serial.

---

## Q6 — Compute T(8) From T(1)
> **Week reference:** Week 5

A program with F = 0.75 runs in T(1) = 100 seconds on one core. How long does it take on 8 cores?

- A) 75 seconds
- B) 12.5 seconds
- C) 34.3 seconds
- D) 43.5 seconds

**Answer: C**

- A) Incorrect — 75s would imply S(8) = 1.33, drastically underestimating the speedup from parallelism.
- B) Incorrect — 12.5s assumes linear (perfect) speedup of 8×, which is impossible with a 25% serial portion.
- C) Correct — S(8) = 1/(0.25 + 0.75/8) = 1/(0.25 + 0.09375) = 1/0.34375 ≈ 2.909; T(8) = 100 / 2.909 ≈ 34.3s.
- D) Incorrect — 43.5s corresponds approximately to S(8) ≈ 2.3, which does not match F = 0.75.

---

## Q7 — Effect of Halving the Serial Fraction
> **Week reference:** Week 5

A program runs on 16 cores with F = 0.8 (serial fraction = 20%). An optimization cuts the serial portion in half to 10%, making the new serial fraction 0.1 and the new parallel fraction F' = 0.9. What is the new speedup S'(16) compared to the original S(16)?

- A) S(16) ≈ 4.0, S'(16) ≈ 6.4 — halving the serial fraction nearly doubles the speedup.
- B) S(16) ≈ 4.0, S'(16) ≈ 4.0 — halving the serial code has no effect on speedup.
- C) S(16) ≈ 4.0, S'(16) ≈ 16 — the speedup becomes linear with perfect optimization.
- D) S(16) ≈ 4.0, S'(16) ≈ 8.0 — halving the serial fraction exactly doubles the speedup.

**Answer: A**

- A) Correct — S(16) = 1/(0.2 + 0.05) = 4.0 (original); actually: S(16) = 1/(0.2+0.8/16)=1/0.25=4.0 for F=0.8, and S'(16)=1/(0.1+0.9/16)=1/(0.1+0.05625)=1/0.15625≈6.4 for F=0.9.
- B) Incorrect — reducing serial code directly increases speedup because the serial bottleneck is smaller.
- C) Incorrect — linear speedup requires F = 1.0 (no serial portion at all), not merely halving it.
- D) Incorrect — the relationship between serial fraction and speedup is not linear; halving serial fraction does not exactly double speedup.

---

## Q8 — Compute Efficiency E(p)
> **Week reference:** Week 5

A program with parallel fraction F = 0.9 runs on 10 cores. What is the parallel efficiency E(10) = S(10) / p?

- A) 0.90
- B) 0.53
- C) 0.10
- D) 1.00

**Answer: B**

- A) Incorrect — 0.90 confuses F with efficiency; efficiency is always less than F for p > 1.
- B) Correct — S(10) = 1/(0.1 + 0.9/10) = 1/(0.1 + 0.09) = 1/0.19 ≈ 5.26; E(10) = 5.26/10 ≈ 0.526.
- C) Incorrect — 0.10 is the serial fraction (1 - F), not the efficiency.
- D) Incorrect — E(10) = 1.0 would require perfect linear speedup S(10) = 10, impossible with a serial fraction.

---

## Q9 — Maximum Speedup With 5% Serial Fraction
> **Week reference:** Week 5

A program has a serial (non-parallelizable) fraction of 5%. What is the maximum possible speedup achievable, regardless of how many cores are used?

- A) 95
- B) 20
- C) 5
- D) Unlimited

**Answer: B**

- A) Incorrect — 95 confuses the parallel fraction (F = 0.95 = 95%) with the maximum speedup.
- B) Correct — S_max = 1/(1 - F) = 1/(1 - 0.95) = 1/0.05 = 20.
- C) Incorrect — 5 corresponds to S_max if F = 0.8 (serial fraction = 20%), not 5%.
- D) Incorrect — Amdahl's Law proves speedup is strictly bounded by 1/(1 - F) for any F < 1.

---

## Q10 — Wall-Clock vs CPU Time in Speedup
> **Week reference:** Week 5

Which formula correctly defines parallel speedup S(p) for a program run on p cores?

- A) S(p) = (total CPU time on p cores) / (CPU time on 1 core)
- B) S(p) = (wall-clock time on 1 core) / (wall-clock time on p cores)
- C) S(p) = p × (wall-clock time on p cores) / (wall-clock time on 1 core)
- D) S(p) = (wall-clock time on p cores) / (wall-clock time on 1 core)

**Answer: B**

- A) Incorrect — using total CPU time (sum across all cores) inflates the numerator by up to a factor of p, giving misleading results.
- B) Correct — speedup is T(1)/T(p) where T refers to wall-clock (elapsed) time, measuring actual time saved.
- C) Incorrect — this formula divides rather than inverting the ratio, and multiplies by p without justification.
- D) Incorrect — this is the inverse of the correct formula; it produces a value less than 1, representing slowdown not speedup.

---

## Q11 — Asymptotic Behavior of S(p)
> **Week reference:** Week 5

A program has parallel fraction F = 0.9. Which statement is correct about S(p) as p → ∞?

- A) S(p) → ∞ because more cores always give more speedup.
- B) S(p) → 10 because S_max = 1/(1 - F) = 1/0.1 = 10.
- C) S(p) → 9 because the parallel fraction is 90%.
- D) S(p) → 10 only when p = 10; after that it decreases.

**Answer: B**

- A) Incorrect — Amdahl's Law proves speedup is bounded; the serial fraction creates a hard ceiling.
- B) Correct — as p → ∞, F/p → 0, so S(p) → 1/(1 - F) = 1/0.1 = 10; S(p) approaches but never exceeds 10.
- C) Incorrect — S_max = 9 would require 1/(1 - F) = 9, i.e., F ≈ 0.889, not F = 0.9.
- D) Incorrect — S(p) is monotonically increasing with p, approaching S_max asymptotically; it never decreases.

---

## Q12 — Compare Two Programs at p=20
> **Week reference:** Week 5

Program A has parallel fraction F_A = 0.9 and Program B has parallel fraction F_B = 0.95. On 20 cores, which program achieves higher speedup, and approximately by how much?

- A) Program A: S_A ≈ 6.9, Program B: S_B ≈ 10.3 — Program B is faster.
- B) Program A: S_A ≈ 9.0, Program B: S_B ≈ 9.5 — almost equal because both F values are close to 1.
- C) Program A: S_A ≈ 6.9, Program B: S_B ≈ 6.9 — the same, since both benefit from 20 cores equally.
- D) Program A: S_A ≈ 10, Program B: S_B ≈ 20 — each equals its respective S_max.

**Answer: A**

- A) Correct — S_A(20) = 1/(0.1 + 0.9/20) = 1/0.145 ≈ 6.9; S_B(20) = 1/(0.05 + 0.95/20) = 1/0.0975 ≈ 10.3; the smaller serial fraction of B makes a large difference.
- B) Incorrect — these values ignore the formula; the serial fraction strongly differentiates performance even when F values seem close.
- C) Incorrect — the two programs have different serial fractions and therefore different Amdahl bounds; they cannot have identical speedup.
- D) Incorrect — S(20) approaches but never equals S_max; S_A(20) ≈ 6.9 << S_max_A = 10, and S_B(20) ≈ 10.3 << S_max_B = 20.

---
