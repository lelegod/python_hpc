# Amdahl's Law — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Compute Speedup F=0.9 p=8](#q1-compute-speedup-f09-p8)
- [Q2 — Solve for F Then S_max](#q2-solve-for-f-then-s_max)
- [Q3 — Derive F From Saturation Plot](#q3-derive-f-from-saturation-plot)
- [Q4 — Achievability of Target Speedup](#q4-achievability-of-target-speedup)
- [Q5 — Recover T(1) From T(4)](#q5-recover-t1-from-t4)
- [Q6 — Compute T(8) From T(1)](#q6-compute-t8-from-t1)
- [Q7 — Effect of Halving the Serial Fraction](#q7-effect-of-halving-the-serial-fraction)
- [Q8 — Compute Efficiency E(p)](#q8-compute-efficiency-ep)
- [Q9 — Maximum Speedup With 5% Serial Fraction](#q9-maximum-speedup-with-5-serial-fraction)
- [Q10 — Wall-Clock vs CPU Time in Speedup](#q10-wall-clock-vs-cpu-time-in-speedup)
- [Q11 — Asymptotic Behavior of S(p)](#q11-asymptotic-behavior-of-sp)
- [Q12 — Compare Two Programs at p=20](#q12-compare-two-programs-at-p20)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q13 — Reverse-Solve for F from Raw Times](#q13-reverse-solve-for-f-from-raw-times)
- [Q14 — Minimum Cores for Target Speedup](#q14-minimum-cores-for-target-speedup)
- [Q15 — Efficiency Comparison Across Core Counts](#q15-efficiency-comparison-across-core-counts)
- [Q16 — Gustafson's Law vs Amdahl's Law](#q16-gustafsons-law-vs-amdahls-law)
- [Q17 — Two Programs, Large Core Count](#q17-two-programs-large-core-count)
- [Q18 — Finding T(1) From T(p) and S_max](#q18-finding-t1-from-tp-and-s_max)
- [Q19 — Amdahl at Extreme Core Counts](#q19-amdahl-at-extreme-core-counts)
- [Q20 — Serial Fraction Bottleneck at Scale](#q20-serial-fraction-bottleneck-at-scale)
- [Q21 — Comparing Efficiency Before and After Optimization](#q21-comparing-efficiency-before-and-after-optimization)
- [Q22 — Amdahl Limit When Serial Work is Fixed Overhead](#q22-amdahl-limit-when-serial-work-is-fixed-overhead)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q23 — Optimize the Serial Part vs Add Cores](#q23--optimize-the-serial-part-vs-add-cores)
- [Q24 — Two-Point Speedup Curve to Pin Down F](#q24--two-point-speedup-curve-to-pin-down-f)
- [Q25 — Weak Scaling vs Strong Scaling](#q25--weak-scaling-vs-strong-scaling)
- [Q26 — Parallel Fraction from a Plot Saturation Value](#q26--parallel-fraction-from-a-plot-saturation-value)
- [Q27 — S(p) vs T(p) Confusion Trap](#q27--sp-vs-tp-confusion-trap)
- [Q28 — Serial Fraction Derived from a Two-Phase Program](#q28--serial-fraction-derived-from-a-two-phase-program)
- [Q29 — Gustafson Scaled Speedup Exceeds Amdahl S_max](#q29--gustafson-scaled-speedup-exceeds-amdahl-s_max)
- [Q30 — Hardware Ceiling vs Theoretical S_max](#q30--hardware-ceiling-vs-theoretical-s_max)
- [Q31 — Efficiency as p Approaches Infinity](#q31--efficiency-as-p-approaches-infinity)
- [Q32 — Parallel Fraction Cannot Exceed 1](#q32--parallel-fraction-cannot-exceed-1)

---

> Topics: S(p) formula, solving for F, maximum speedup, efficiency, serial time calculations.
> Exam frequency: **Every exam** — highest priority topic.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--compute-speedup-f09-p8)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — Compute Speedup F=0.9 p=8
> **Week reference:** Week 5

**Mental Model:** Direct formula application — plug F and p into S(p) = 1/((1-F) + F/p) and compute the denominator carefully — the trap is multiplying F×p instead of dividing F/p.

A program has a parallel fraction F = 0.9. Using Amdahl's Law, what is the speedup S(8) when running on 8 cores?

- A) 5.0
- B) 7.2
- C) 4.71
- D) 3.33

**Answer: C**

- A) Incorrect — S = 5.0 would require a denominator of 0.2, which means (1-F) + F/8 = 0.2. With F=0.9 the denominator is 0.1 + 0.1125 = 0.2125, not 0.2. F ≈ 0.933 would be needed to get S=5.
- B) Incorrect — 7.2 = 0.9 × 8 is what you get if you mistakenly multiply F by p instead of dividing. That formula has no basis in Amdahl's Law; it ignores the serial bottleneck entirely.
- C) Correct — S(8) = 1 / ((1 - 0.9) + 0.9/8) = 1 / (0.1 + 0.1125) = 1 / 0.2125 ≈ 4.71.
- D) Incorrect — 3.33 = 1/0.3, which means the denominator is 0.3. With F=0.9 and p=8: (1-0.9) + 0.9/8 = 0.2125 ≠ 0.3. A denominator of 0.3 corresponds to F=0.8 with p=8: 0.2 + 0.1 = 0.3.

---

## Q2 — Solve for F Then S_max
> **Week reference:** Week 5

**Mental Model:** Back-solving from a measured speedup — use F = p(1 - 1/S(p))/(p-1) to find F, then S_max = 1/(1-F) — the trap is computing F from incorrect algebra or confusing S(p) with S_max.

Measurements show that a program achieves S(3) = 2.5 on 3 cores. Using F = p(1 - 1/S(p)) / (p - 1), what is the parallel fraction F, and what is the theoretical maximum speedup S_max?

- A) F = 0.8, S_max = 5
- B) F = 0.9, S_max = 10
- C) F = 0.75, S_max = 4
- D) F = 0.6, S_max = 2.5

**Answer: B**

- A) Incorrect — plugging in: F = 3(1 - 1/2.5)/(3-1) = 3(1 - 0.4)/2 = 3(0.6)/2 = 1.8/2 = 0.9, not 0.8. S_max = 1/(1-0.9) = 10, not 5.
- B) Correct — F = 3(1 - 0.4)/2 = 0.9; S_max = 1/(1 - 0.9) = 1/0.1 = 10. Verify: S(3) = 1/(0.1 + 0.9/3) = 1/(0.1+0.3) = 1/0.4 = 2.5 ✓.
- C) Incorrect — if F=0.75 then S(3) = 1/(0.25 + 0.75/3) = 1/(0.25 + 0.25) = 1/0.5 = 2.0, not 2.5. F=0.75 underpredicts the observed speedup.
- D) Incorrect — if F=0.6 then S(3) = 1/(0.4 + 0.6/3) = 1/(0.4 + 0.2) = 1/0.6 ≈ 1.67, not 2.5. F=0.6 gives a much lower speedup than measured.

---

## Q3 — Derive F From Saturation Plot
> **Week reference:** Week 5

**Mental Model:** Reading S_max from a plot and inverting — S_max = 1/(1-F) means (1-F) = 1/S_max, so F = 1 - 1/S_max — the trap is plugging S_max directly in as F.

A speedup-vs-cores plot shows that speedup levels off and never exceeds 4, regardless of how many cores are added. What is the parallel fraction F of this program?

- A) F = 0.80
- B) F = 0.75
- C) F = 0.25
- D) F = 0.90

**Answer: B**

- A) Incorrect — if F=0.80 then S_max = 1/(1-0.80) = 1/0.20 = 5. But the plot saturates at 4, not 5. So F=0.80 gives the wrong saturation value.
- B) Correct — S_max = 4 means 1/(1-F) = 4, so 1-F = 1/4 = 0.25, thus F = 1 - 0.25 = 0.75. Check: 1/(1-0.75) = 1/0.25 = 4 ✓.
- C) Incorrect — F=0.25 means only 25% of the program is parallel. S_max = 1/(1-0.25) = 1/0.75 ≈ 1.33, far below the observed saturation of 4.
- D) Incorrect — if F=0.90 then S_max = 1/(1-0.90) = 1/0.10 = 10. That would produce a saturation ceiling at 10, not 4.

---

## Q4 — Achievability of Target Speedup
> **Week reference:** Week 5

**Mental Model:** Comparing S(p) to S_max to judge feasibility — compute both S(p) and S_max and compare against the target — the trap is assuming S_max is achievable or confusing F with S(p).

A program has parallel fraction F = 0.8. A developer wants to achieve a speedup of at least 4× using 8 cores. Is this achievable?

- A) Yes — S(8) = 4.0 exactly.
- B) No — S(8) ≈ 3.33, which falls short of 4.
- C) Yes — S(8) ≈ 4.71 with F = 0.8.
- D) No — S_max = 4.0, so S(8) cannot reach 4.

**Answer: B**

- A) Incorrect — S(8) = 1/((1-0.8) + 0.8/8) = 1/(0.2 + 0.1) = 1/0.3 = 3.33, not 4.0. To reach S(8)=4 you would need a denominator of 0.25, requiring a higher F.
- B) Correct — S(8) = 1/((1-0.8) + 0.8/8) = 1/(0.2 + 0.1) = 1/0.3 ≈ 3.33. This is below 4, so the target is not met with 8 cores and F=0.8.
- C) Incorrect — S(8) ≈ 4.71 requires F=0.9, not F=0.8. Check: 1/(0.1 + 0.9/8) = 1/0.2125 ≈ 4.71. The question specifies F=0.8.
- D) Incorrect — S_max = 1/(1-0.8) = 1/0.2 = 5, not 4. So 4× is theoretically possible with enough cores, just not achievable with only p=8.

---

## Q5 — Recover T(1) From T(4)
> **Week reference:** Week 5

**Mental Model:** Reconstructing serial time — T(1) = T(p) × S(p), so first compute S(p) using Amdahl, then multiply — the trap is confusing speedup with efficiency or using T(p) alone.

A parallel run on 4 cores takes T(4) = 10 minutes, and the program has parallel fraction F = 0.8. Using T(1) = T(p) × S(p), what is the single-core runtime T(1)?

- A) 20 minutes
- B) 25 minutes
- C) 40 minutes
- D) 12.5 minutes

**Answer: B**

- A) Incorrect — 20 min would imply S(4) = 20/10 = 2.0. But if S(4)=2.0 then from Amdahl: 2 = 1/((1-F)+F/4), giving (1-F)+F/4 = 0.5, which solves to F = 0.667. The question says F=0.8, so this is inconsistent.
- B) Correct — S(4) = 1/((1-0.8) + 0.8/4) = 1/(0.2 + 0.2) = 1/0.4 = 2.5; T(1) = T(4) × S(4) = 10 × 2.5 = 25 minutes.
- C) Incorrect — 40 min would imply S(4) = 40/10 = 4.0. For S(4)=4 the denominator must be 0.25, meaning (1-F)+F/4=0.25, which requires F≈1.0 (essentially fully parallel). F=0.8 cannot yield S(4)=4.
- D) Incorrect — 12.5 min would imply S(4) = 12.5/10 = 1.25, meaning the 4-core version is only 25% faster than serial. S(4) < 1 would mean slowdown; 1.25 is far below what Amdahl predicts for F=0.8.

---

## Q6 — Compute T(8) From T(1)
> **Week reference:** Week 5

**Mental Model:** Forward prediction — compute S(p) then divide T(1) — the trap is using p itself as the speedup (assumes perfect parallelism) or forgetting that the serial fraction sets a floor on runtime.

A program with F = 0.75 runs in T(1) = 100 seconds on one core. How long does it take on 8 cores?

- A) 75 seconds
- B) 12.5 seconds
- C) 34.3 seconds
- D) 43.5 seconds

**Answer: C**

- A) Incorrect — 75s implies S(8) = 100/75 ≈ 1.33. That would mean 8 cores give almost no speedup, which corresponds to F ≈ 0. With F=0.75 the speedup is much higher.
- B) Incorrect — 12.5s assumes S(8) = 8 (perfect linear speedup). Perfect speedup requires F=1.0 (no serial portion). With 25% serial, the serial portion alone takes 25s, so T(8) cannot be below 25s.
- C) Correct — S(8) = 1/((1-0.75) + 0.75/8) = 1/(0.25 + 0.09375) = 1/0.34375 ≈ 2.909; T(8) = 100 / 2.909 ≈ 34.3s.
- D) Incorrect — 43.5s implies S(8) = 100/43.5 ≈ 2.3. That corresponds approximately to a lower F (around 0.57). For F=0.75 with 8 cores the speedup is ~2.9, giving a shorter runtime than 43.5s.

---

## Q7 — Effect of Halving the Serial Fraction
> **Week reference:** Week 5

**Mental Model:** Sensitivity of speedup to serial fraction — small reductions in the serial fraction yield disproportionately large speedup gains at high core counts — the trap is assuming the relationship is linear (halving serial → doubling speedup).

A program runs on 16 cores with F = 0.8 (serial fraction = 20%). An optimization cuts the serial portion in half to 10%, making the new serial fraction 0.1 and the new parallel fraction F' = 0.9. What is the new speedup S'(16) compared to the original S(16)?

- A) S(16) ≈ 4.0, S'(16) ≈ 6.4 — halving the serial fraction nearly doubles the speedup.
- B) S(16) ≈ 4.0, S'(16) ≈ 4.0 — halving the serial code has no effect on speedup.
- C) S(16) ≈ 4.0, S'(16) ≈ 16 — the speedup becomes linear with perfect optimization.
- D) S(16) ≈ 4.0, S'(16) ≈ 8.0 — halving the serial fraction exactly doubles the speedup.

**Answer: A**

- A) Correct — S(16) = 1/((1-0.8) + 0.8/16) = 1/(0.2 + 0.05) = 1/0.25 = 4.0; S'(16) = 1/((1-0.9) + 0.9/16) = 1/(0.1 + 0.05625) = 1/0.15625 ≈ 6.4. Halving the serial fraction cuts the bottleneck denominator from 0.25 to 0.15625, a 37.5% reduction that yields a 60% speedup gain.
- B) Incorrect — the serial fraction is the entire bottleneck; reducing it directly reduces the denominator of Amdahl's formula and always increases speedup. It cannot have zero effect.
- C) Incorrect — linear speedup of S(16)=16 requires F=1.0 (zero serial fraction). Halving the serial fraction from 20% to 10% still leaves a 10% serial bottleneck, so S_max = 1/0.1 = 10, far below 16.
- D) Incorrect — the relationship is nonlinear. Halving the serial fraction from 20% to 10% changes the denominator from 0.25 to 0.15625, a ratio of 1.6 (not 2.0). Exact doubling would require halving the entire denominator, not just its serial term.

---

## Q8 — Compute Efficiency E(p)
> **Week reference:** Week 5

**Mental Model:** Efficiency = speedup per core — E(p) = S(p)/p, always between 0 and 1, always less than F — the trap is equating efficiency with F or assuming E=1 for any finite p.

A program with parallel fraction F = 0.9 runs on 10 cores. What is the parallel efficiency E(10) = S(10) / p?

- A) 0.90
- B) 0.53
- C) 0.10
- D) 1.00

**Answer: B**

- A) Incorrect — 0.90 is the parallel fraction F, not the efficiency. Efficiency is always lower than F for p > 1 because cores are never perfectly utilized; the serial fraction causes idle time.
- B) Correct — S(10) = 1/((1-0.9) + 0.9/10) = 1/(0.1 + 0.09) = 1/0.19 ≈ 5.26; E(10) = S(10)/p = 5.26/10 ≈ 0.526.
- C) Incorrect — 0.10 is the serial fraction (1-F), not the efficiency. The serial fraction is the fraction of work that cannot be parallelized, a very different quantity.
- D) Incorrect — E = 1.0 would require perfect linear speedup S(10) = 10, which is only possible if F = 1.0. With a 10% serial fraction the efficiency is always below 1.

---

## Q9 — Maximum Speedup With 5% Serial Fraction
> **Week reference:** Week 5

**Mental Model:** Amdahl's ceiling — S_max = 1/(1-F), computed directly from the serial fraction — the trap is confusing the parallel fraction percentage with S_max (e.g., thinking 95% → S_max=95).

A program has a serial (non-parallelizable) fraction of 5%. What is the maximum possible speedup achievable, regardless of how many cores are used?

- A) 95
- B) 20
- C) 5
- D) Unlimited

**Answer: B**

- A) Incorrect — 95 conflates the parallel fraction percentage with the maximum speedup. If 95% is parallel then F=0.95, and S_max = 1/(1-0.95) = 1/0.05 = 20. The 95 is the numerator concept, not the answer.
- B) Correct — serial fraction = 5% means F = 0.95. S_max = 1/(1-F) = 1/(1-0.95) = 1/0.05 = 20. No matter how many cores are added, speedup cannot exceed 20.
- C) Incorrect — S_max = 5 would require 1/(1-F) = 5, meaning 1-F = 0.2, so F = 0.8 (20% serial). That applies to a program that is 20% serial, not 5% serial.
- D) Incorrect — Amdahl's Law rigorously proves speedup is bounded by 1/(1-F) for any F < 1. As long as any serial portion exists (even 0.001%), the ceiling is finite. Only F=1.0 (fully parallel) gives unlimited speedup.

---

## Q10 — Wall-Clock vs CPU Time in Speedup
> **Week reference:** Week 5

**Mental Model:** Speedup measures time saved, not compute consumed — it is always wall-clock time on 1 core divided by wall-clock time on p cores — the trap is using total CPU time (which scales with p and inflates the ratio).

Which formula correctly defines parallel speedup S(p) for a program run on p cores?

- A) S(p) = (total CPU time on p cores) / (CPU time on 1 core)
- B) S(p) = (wall-clock time on 1 core) / (wall-clock time on p cores)
- C) S(p) = p × (wall-clock time on p cores) / (wall-clock time on 1 core)
- D) S(p) = (wall-clock time on p cores) / (wall-clock time on 1 core)

**Answer: B**

- A) Incorrect — total CPU time on p cores is the sum of time across all cores. If parallelism is perfect this equals p × T(p) ≈ T(1), giving a ratio near 1 regardless of actual speedup. Using CPU time masks the real elapsed-time improvement.
- B) Correct — speedup is S(p) = T(1)/T(p) where T is wall-clock (elapsed) time. This measures the actual reduction in time the user experiences waiting for the result.
- C) Incorrect — this formula multiplies by p in the wrong direction and inverts the ratio. It would produce values greater than p for reasonable speedups, which is physically impossible.
- D) Incorrect — this is T(p)/T(1), the inverse of the correct formula. For a faster parallel run T(p) < T(1), so this ratio is less than 1 — it measures slowdown, not speedup.

---

## Q11 — Asymptotic Behavior of S(p)
> **Week reference:** Week 5

**Mental Model:** Taking the limit as p → ∞ — F/p → 0, leaving only the serial term (1-F) in the denominator — the trap is thinking more cores always help indefinitely or confusing S_max with a value that requires a specific p.

A program has parallel fraction F = 0.9. Which statement is correct about S(p) as p → ∞?

- A) S(p) → ∞ because more cores always give more speedup.
- B) S(p) → 10 because S_max = 1/(1 - F) = 1/0.1 = 10.
- C) S(p) → 9 because the parallel fraction is 90%.
- D) S(p) → 10 only when p = 10; after that it decreases.

**Answer: B**

- A) Incorrect — Amdahl's Law proves the opposite. As p→∞, F/p→0, so S(p)→1/(1-F). The serial fraction (1-F) creates a hard ceiling; adding infinite cores cannot eliminate the serial bottleneck.
- B) Correct — as p→∞: S(p) = 1/((1-F) + F/p) → 1/(1-F) = 1/0.1 = 10. The curve approaches 10 asymptotically, getting closer with each additional core but never crossing it.
- C) Incorrect — S_max = 9 would require 1/(1-F) = 9, meaning 1-F = 1/9 ≈ 0.111, so F ≈ 0.889. For F=0.9 the limit is 10, not 9.
- D) Incorrect — S(p) is strictly monotonically increasing with p; it never decreases. More cores always improve speedup (or at worst leave it unchanged), approaching S_max asymptotically from below.

---

## Q12 — Compare Two Programs at p=20
> **Week reference:** Week 5

**Mental Model:** Small differences in F translate to large speedup differences at high core counts — compute both S(p) values independently and compare — the trap is assuming near-equal F values produce near-equal speedups at large p.

Program A has parallel fraction F_A = 0.9 and Program B has parallel fraction F_B = 0.95. On 20 cores, which program achieves higher speedup, and approximately by how much?

- A) Program A: S_A ≈ 6.9, Program B: S_B ≈ 10.3 — Program B is faster.
- B) Program A: S_A ≈ 9.0, Program B: S_B ≈ 9.5 — almost equal because both F values are close to 1.
- C) Program A: S_A ≈ 6.9, Program B: S_B ≈ 6.9 — the same, since both benefit from 20 cores equally.
- D) Program A: S_A ≈ 10, Program B: S_B ≈ 20 — each equals its respective S_max.

**Answer: A**

- A) Correct — S_A(20) = 1/((1-0.9) + 0.9/20) = 1/(0.1 + 0.045) = 1/0.145 ≈ 6.9; S_B(20) = 1/((1-0.95) + 0.95/20) = 1/(0.05 + 0.0475) = 1/0.0975 ≈ 10.3. Halving the serial fraction from 10% to 5% delivers ~50% more speedup at p=20.
- B) Incorrect — these values ignore the formula entirely. The serial fractions differ by a factor of 2 (10% vs 5%), and at p=20 this produces a 50% difference in speedup (6.9 vs 10.3), not the tiny gap suggested here.
- C) Incorrect — identical speedup would require identical F values. F_A=0.9 and F_B=0.95 have different serial fractions, so their Amdahl formulas give different denominators and different speedups at any p > 1.
- D) Incorrect — S(p) approaches but never reaches S_max with finite p. S_A(20)≈6.9 is well below S_max_A=10, and S_B(20)≈10.3 is well below S_max_B=20. Reaching S_max requires p→∞.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets reverse-solving for F and p, efficiency calculations, and multi-program comparisons at scale

---

## Q13 — Reverse-Solve for F from Raw Times

> **Week reference:** Week 5

A program runs for T(1) = 120 seconds on one core. On 6 cores it runs for T(6) = 30 seconds. Using Amdahl's Law rearranged to solve for F, what is the parallel fraction?

- A) F = 0.80
- B) F = 0.90
- C) F ≈ 0.933
- D) F ≈ 0.967

**Answer: B**

- A) Incorrect — F = 0.80 gives S(6) = 1/(0.2 + 0.8/6) = 1/0.333 ≈ 3.0, meaning T(6) = 120/3 = 40s, not 30s. F = 0.80 produces too little speedup.
- B) Correct — S = 120/30 = 4. Using F = p(1 - 1/S)/(p-1): F = 6(1 - 0.25)/5 = 6 × 0.75 / 5 = 4.5/5 = 0.90. Verify: S(6) = 1/(0.1 + 0.15) = 1/0.25 = 4.0 ✓.
- C) Incorrect — F = 0.933 gives S(6) = 1/(0.067 + 0.933/6) = 1/(0.067 + 0.155) = 1/0.222 ≈ 4.5, meaning T(6) ≈ 26.7s — too fast for the observed 30s.
- D) Incorrect — F = 0.967 gives S(6) ≈ 1/(0.033 + 0.161) = 1/0.194 ≈ 5.15, meaning T(6) ≈ 23.3s — again too fast. A higher F implies higher speedup and shorter T(6).

---

## Q14 — Minimum Cores for Target Speedup

> **Week reference:** Week 5

A program has parallel fraction F = 0.8. A developer needs at least S = 3× speedup. What is the minimum integer number of cores p that achieves this?

- A) p = 4
- B) p = 6
- C) p = 8
- D) p = 10

**Answer: B**

- A) Incorrect — S(4) = 1/(0.2 + 0.8/4) = 1/0.4 = 2.5 < 3. Four cores fall short of the target.
- B) Correct — Solve: 3 = 1/(0.2 + 0.8/p) → 0.8/p = 1/3 - 0.2 = 2/15 → p = 0.8 × 15/2 = 6. S(6) = 1/(0.2 + 0.1333) = 1/0.3333 = 3.0 exactly. Six is the minimum.
- C) Incorrect — S(8) = 1/(0.2 + 0.1) = 1/0.3 ≈ 3.33 also exceeds the target, but 6 already meets it — 8 is not the minimum.
- D) Incorrect — S(10) = 1/(0.2 + 0.08) = 1/0.28 ≈ 3.57 exceeds the target by even more but wastes four extra cores relative to the minimum of 6.

---

## Q15 — Efficiency Comparison Across Core Counts

> **Week reference:** Week 5

A program with F = 0.85 is run on 4 cores and again on 16 cores. Which of the following correctly describes how efficiency E(p) = S(p)/p changes?

- A) E(4) ≈ 0.69, E(16) ≈ 0.31 — efficiency decreases as more cores are added.
- B) E(4) ≈ 0.69, E(16) ≈ 0.69 — efficiency stays constant as long as F is fixed.
- C) E(4) ≈ 0.85, E(16) ≈ 0.85 — efficiency equals the parallel fraction F.
- D) E(4) ≈ 0.69, E(16) ≈ 0.85 — efficiency increases toward F as more cores are added.

**Answer: A**

- A) Correct — S(4) = 1/(0.15 + 0.2125) = 1/0.3625 ≈ 2.759; E(4) = 2.759/4 ≈ 0.69. S(16) = 1/(0.15 + 0.053125) = 1/0.203125 ≈ 4.923; E(16) = 4.923/16 ≈ 0.31. Efficiency strictly falls as p rises.
- B) Incorrect — Efficiency is not constant with fixed F. Speedup grows sub-linearly because the serial fraction bottleneck is unchanged, so S(p)/p falls as p increases.
- C) Incorrect — Efficiency equals F only at p = 1. For any p > 1, the serial fraction causes idle time on each core, pulling E below F.
- D) Incorrect — The trend is reversed. Efficiency decreases toward zero as p → ∞, not toward F. The serial fraction wastes a proportionally larger share of total core-time at high core counts.

---

## Q16 — Gustafson's Law vs Amdahl's Law

> **Week reference:** Week 5

A program has a serial fraction α = 0.1 (10% serial). Using Gustafson's Law on p = 16 cores, what is the scaled speedup S_G(16)?

- A) S_G = 10 — same as Amdahl's S_max with F = 0.9.
- B) S_G = 14.5 — computed as p - α(p - 1).
- C) S_G = 6.4 — same as Amdahl's S(16) with F = 0.9.
- D) S_G = 16 — because 90% of the work is parallelisable.

**Answer: B**

- A) Incorrect — 10 is Amdahl's S_max = 1/(1-0.9) = 1/0.1, which fixes problem size. Gustafson's Law grows the problem size with p, so it predicts a higher scaled speedup.
- B) Correct — S_G(16) = p - α(p-1) = 16 - 0.1 × 15 = 16 - 1.5 = 14.5. Gustafson allows speedup to grow nearly linearly with p when the workload scales with core count.
- C) Incorrect — 6.4 is Amdahl's fixed-size formula result: S(16) = 1/(0.1 + 0.9/16) = 1/0.15625 = 6.4. Gustafson's scaled speedup of 14.5 is much higher.
- D) Incorrect — Perfect speedup of 16 would require α = 0. With 10% serial work, Gustafson gives 14.5, not 16.

---

## Q17 — Two Programs, Large Core Count

> **Week reference:** Week 5

Program X has F_X = 0.92 and Program Y has F_Y = 0.96. As the number of cores p grows very large, what is the ratio S_max_Y / S_max_X?

- A) Ratio ≈ 1.04 — the programs have nearly equal maximum speedup.
- B) Ratio = 3.0 — Program Y's maximum speedup is three times that of Program X.
- C) Ratio = 2.0 — halving the serial fraction doubles the maximum speedup.
- D) Ratio ≈ 1.5 — Program Y is 50% faster at large core counts.

**Answer: C**

- A) Incorrect — 0.96/0.92 ≈ 1.04 compares the F values directly. S_max depends on the serial fraction (1-F), not F itself: (1-0.92) = 0.08 and (1-0.96) = 0.04, a 2:1 ratio, not 1.04:1.
- B) Incorrect — Ratio = 3.0 would require (1-F_X)/(1-F_Y) = 3. Here 0.08/0.04 = 2, not 3. Three-fold difference would need, e.g., serial fractions of 0.12 and 0.04.
- C) Correct — S_max_X = 1/0.08 = 12.5; S_max_Y = 1/0.04 = 25. Ratio = 25/12.5 = 2.0. Halving the serial fraction from 8% to 4% exactly doubles S_max because S_max = 1/(1-F) is inversely proportional to the serial fraction.
- D) Incorrect — Ratio = 1.5 would require S_max_Y = 1.5 × 12.5 = 18.75, meaning 1-F_Y = 1/18.75 ≈ 0.053. The actual serial fraction is 0.04, not 0.053.

---

## Q18 — Finding T(1) From T(p) and S_max

> **Week reference:** Week 5

A program's speedup curve saturates at S_max = 5. On 4 cores it takes T(4) = 20 seconds. What is the single-core runtime T(1)?

- A) T(1) = 40 seconds
- B) T(1) = 50 seconds
- C) T(1) = 80 seconds
- D) T(1) = 100 seconds

**Answer: B**

- A) Incorrect — T(1) = 40s implies S(4) = 40/20 = 2.0. With F = 0.8 the formula gives S(4) = 1/(0.2+0.2) = 2.5 ≠ 2.0. These numbers are inconsistent.
- B) Correct — S_max = 5 → F = 0.8. S(4) = 1/(0.2 + 0.8/4) = 1/0.4 = 2.5. T(1) = T(4) × S(4) = 20 × 2.5 = 50s.
- C) Incorrect — T(1) = 80s implies S(4) = 4.0. For S(4) = 4 with p = 4 the denominator must be 0.25; solving (1-F) + F/4 = 0.25 gives F = 1.0. But F = 0.8, so 80s is impossible.
- D) Incorrect — T(1) = 100s implies S(4) = 5.0 = S_max. S_max is only reached as p → ∞; with finite p = 4, S(4) is always strictly less than S_max. S(4) = 2.5 < 5 here.

---

## Q19 — Amdahl at Extreme Core Counts

> **Week reference:** Week 5

A program with F = 0.95 is run on 20 cores and then on 10000 cores. Which of the following is closest to the ratio S(10000) / S(20)?

- A) About 1.05 — the two speedups are nearly equal.
- B) About 1.9 — S(10000) is roughly double S(20).
- C) About 500 — speedup scales almost linearly with core count at high F.
- D) Exactly 20 — S(10000) equals S_max.

**Answer: B**

- A) Incorrect — The two speedups differ by nearly a factor of 2. S(20) ≈ 10.3 and S(10000) ≈ 20.0; going from 20 to 10000 cores still closes most of the gap to S_max = 20.
- B) Correct — S(20) = 1/(0.05 + 0.0475) = 1/0.0975 ≈ 10.26; S(10000) = 1/(0.05 + 0.000095) ≈ 19.96. Ratio ≈ 19.96/10.26 ≈ 1.94, i.e., roughly double. The remaining headroom between S(20) and S_max = 20 is almost fully exploited by 10000 cores.
- C) Incorrect — Linear scaling requires F = 1.0. With S_max = 20 (serial fraction 5%), even infinite cores yield only 20×. A ratio of 500 is impossible.
- D) Incorrect — S_max = 1/(1-0.95) = 20 is the limit as p → ∞. At p = 10000, S ≈ 19.96, very close but not exactly equal to 20. Exact equality requires infinite cores.

---

## Q20 — Serial Fraction Bottleneck at Scale

> **Week reference:** Week 5

A data-science pipeline with F = 0.98 runs on a 128-core node. A colleague claims that switching to a 256-core node would approximately double the speedup. Is the colleague correct?

- A) Yes — doubling cores always doubles speedup when F > 0.5.
- B) No — at p = 128 the program is already near S_max, so doubling cores gives minimal gain.
- C) No — S(256) is actually lower than S(128) because communication overhead grows with core count.
- D) Yes — S(256) ≈ 2 × S(128) because the program is 98% parallel.

**Answer: B**

- A) Incorrect — Doubling cores doubles speedup only when the program is far below S_max (efficiency near 1). With S_max = 50 and S(128) ≈ 36.2 the program is already at ~72% of its ceiling; doubling gives only ~16% gain.
- B) Correct — S_max = 1/0.02 = 50. S(128) = 1/(0.02 + 0.007656) ≈ 36.2; S(256) = 1/(0.02 + 0.003828) ≈ 41.9. That is only a 16% improvement, far from doubling.
- C) Incorrect — Pure Amdahl's Law predicts S(256) > S(128) (more cores never hurt). The real issue is diminishing returns, not a slowdown. Communication overhead is a real concern but is outside the Amdahl model.
- D) Incorrect — A high F does not guarantee proportional scaling once you approach S_max. At 72% of the ceiling, adding cores buys diminishing returns regardless of how large F is.

---

## Q21 — Comparing Efficiency Before and After Optimization

> **Week reference:** Week 5

Before optimization a program has F = 0.75. After optimization the serial fraction is halved from 25% to 12.5%, giving F = 0.875. Both are run on 8 cores. Which answer correctly shows both efficiencies?

- A) E_before ≈ 0.36, E_after ≈ 0.53 — efficiency increases by about 17 percentage points.
- B) E_before ≈ 0.36, E_after ≈ 0.36 — efficiency is unchanged because p is the same.
- C) E_before ≈ 0.75, E_after ≈ 0.875 — efficiency equals the parallel fraction F.
- D) E_before ≈ 0.36, E_after ≈ 0.59 — efficiency increases by about 23 percentage points.

**Answer: A**

- A) Correct — S_before(8) = 1/(0.25 + 0.75/8) = 1/0.34375 ≈ 2.909; E_before = 2.909/8 ≈ 0.364. S_after(8) = 1/(0.125 + 0.875/8) = 1/0.234375 ≈ 4.267; E_after = 4.267/8 ≈ 0.533. Efficiency increases by ~17 percentage points.
- B) Incorrect — Efficiency depends on both F and p. Reducing the serial fraction changes S(p) and therefore E(p) even when p is held fixed.
- C) Incorrect — E = F only at p = 1. For any p > 1, the serial fraction creates idle time on each core, pulling E strictly below F.
- D) Incorrect — E_after ≈ 0.59 corresponds to F = 0.90 (serial fraction 10%), not F = 0.875 (serial fraction 12.5%). The optimization halves 25% to 12.5%, not to 10%.

---

## Q22 — Amdahl Limit When Serial Work is Fixed Overhead

> **Week reference:** Week 5

A program always spends exactly 4 seconds on serial setup regardless of problem size. The total single-core runtime is T(1) = 40 seconds. What is S_max, and what speedup S(8) does Amdahl's Law predict on 8 cores?

- A) S_max = 10, S(8) ≈ 4.71
- B) S_max = 10, S(8) = 8.0
- C) S_max = 40, S(8) ≈ 4.71
- D) S_max = 20, S(8) ≈ 4.71

**Answer: A**

- A) Correct — Serial fraction = 4/40 = 0.10, so F = 0.90. S_max = 1/(1-0.9) = 10. S(8) = 1/(0.1 + 0.9/8) = 1/(0.1+0.1125) = 1/0.2125 ≈ 4.71.
- B) Incorrect — S(8) = 8.0 requires perfect linear speedup (F = 1.0). With 4 seconds of mandatory serial work, the parallel part alone takes at least 4s on any number of cores, so T(8) ≥ 4s and S(8) ≤ 10. Linear speedup is impossible.
- C) Incorrect — S_max = 40 would require 1/(1-F) = 40, meaning 1-F = 0.025, so F = 0.975. That corresponds to only 1 second of serial work. The problem states 4 seconds (10% of 40s).
- D) Incorrect — S_max = 20 requires F = 0.95 (serial fraction 5%), which corresponds to 2 seconds of serial work. The setup is 4 seconds (10%), giving S_max = 10, not 20.

---

## Set 3 — Extended Practice

> Targets: optimize-serial-vs-add-cores decisions, two-phase programs, Gustafson scaled speedup, hardware ceiling vs S_max, S(p)/T(p) confusion, weak vs strong scaling, and edge cases not covered in Sets 1–2.

---

## Q23 — Optimize the Serial Part vs Add Cores

> **Week reference:** Week 5

**Mental Model:** When the serial fraction is the bottleneck, optimizing it always raises S_max and benefits every core count — whereas adding more cores to a fixed S_max delivers only diminishing returns beyond a certain point.

A program has a serial phase of 20 seconds and a parallel phase of 100 seconds (total: 120 seconds on one core), matching the DTU quiz scenario. The maximum speedup S_max = 6. A new optimization reduces the serial phase from 20 s to 5 s, giving a new total T'(1) = 105 s. After optimization, what is the new S_max?

- A) S_max' = 6 — the maximum speedup is unchanged because the parallel phase still takes 100 s.
- B) S_max' = 21 — because the new serial fraction is 5/105.
- C) S_max' = 20 — because the serial phase was reduced by a factor of 4.
- D) S_max' = 100 — because the parallel phase dominates the new total.

**Answer: B**

- A) Incorrect — S_max depends on the serial fraction, which changes when the serial phase is reduced. The old serial fraction was 20/120 = 1/6, giving S_max = 6. After optimization the serial fraction is 5/105 = 1/21, giving S_max = 21. Reducing the serial phase always increases S_max.
- B) Correct — New parallel fraction F' = 100/105 = 20/21. New serial fraction = 5/105 = 1/21. S_max' = 1/(1 - 20/21) = 1/(1/21) = 21. The optimization dramatically raises the ceiling from 6× to 21×.
- C) Incorrect — The original serial fraction was 1/6, giving S_max = 6. Multiplying S_max by 4 (=20/5) would give 24, not 21, and does not follow the formula. S_max is not proportional to the reduction factor of the serial phase in isolation.
- D) Incorrect — S_max = 100 would require the entire program to be parallelizable (F = 1.0). There is still 5 s of serial work, so the serial fraction is 5/105 ≠ 0, and S_max is finite at 21.

---

## Q24 — Two-Point Speedup Curve to Pin Down F

> **Week reference:** Week 5

**Mental Model:** Two distinct speedup measurements at different core counts are sufficient to uniquely determine F — solve the Amdahl equation at each point and check consistency, or solve the two simultaneous equations algebraically.

A developer measures S(2) = 1.6 and S(4) = 2.29 for the same program. Using only the measurement at p = 4 and the formula F = p(1 - 1/S)/(p - 1), what is the parallel fraction F?

- A) F = 0.60
- B) F = 0.75
- C) F = 0.80
- D) F = 0.90

**Answer: B**

- A) Incorrect — F = 0.60 gives S(4) = 1/(0.40 + 0.15) = 1/0.55 ≈ 1.82, not 2.29. This value of F produces too little speedup.
- B) Correct — F = 4 × (1 - 1/2.29) / (4 - 1) = 4 × (1 - 0.4367) / 3 = 4 × 0.5633 / 3 ≈ 2.253 / 3 ≈ 0.751 ≈ 0.75. Verify: S(4) = 1/(0.25 + 0.75/4) = 1/(0.25 + 0.1875) = 1/0.4375 ≈ 2.286 ≈ 2.29 ✓. Cross-check with S(2): 1/(0.25 + 0.375) = 1/0.625 = 1.60 ✓.
- C) Incorrect — F = 0.80 gives S(4) = 1/(0.20 + 0.20) = 1/0.40 = 2.50, not 2.29. F = 0.80 overestimates the observed speedup.
- D) Incorrect — F = 0.90 gives S(4) = 1/(0.10 + 0.225) = 1/0.325 ≈ 3.08, well above the observed 2.29. A higher F always produces higher speedup; 0.90 is too large.

---

## Q25 — Weak Scaling vs Strong Scaling

> **Week reference:** Week 5

**Mental Model:** Strong scaling fixes the problem size and adds cores — speedup is bounded by Amdahl's Law. Weak scaling grows the problem proportionally with the number of cores — the goal is constant runtime, and Gustafson's Law describes the scaled speedup.

Which statement best distinguishes weak scaling from strong scaling in the context of Amdahl's and Gustafson's Laws?

- A) Strong scaling is described by Gustafson's Law; weak scaling is described by Amdahl's Law.
- B) Strong scaling keeps problem size fixed and is bounded by Amdahl's Law; weak scaling grows the problem with core count and is better described by Gustafson's Law.
- C) Both laws apply equally to both scaling regimes; the difference is only in how efficiency is defined.
- D) Weak scaling is used only when F = 1.0; strong scaling applies for all values of F.

**Answer: B**

- A) Incorrect — the assignment is reversed. Amdahl's Law models strong scaling (fixed problem, more cores). Gustafson's Law models weak scaling (problem grows with cores, constant runtime per core).
- B) Correct — Strong scaling: fixed problem size, increase cores, speedup bounded by S_max = 1/(1-F) (Amdahl). Weak scaling: problem size grows proportionally with cores so each core always has the same workload; Gustafson's formula S_G(p) = p - α(p-1) captures the "scaled" speedup achievable and grows nearly linearly with p when α is small.
- C) Incorrect — the two laws model fundamentally different experimental designs. Amdahl assumes a fixed amount of work; Gustafson assumes that work scales with available resources. Efficiency is defined consistently in both (E = S/p), but the formulas and limits differ.
- D) Incorrect — weak scaling is not restricted to F = 1.0. Programs with any serial fraction can be run under a weak-scaling regime; the key is that the problem size increases to keep per-core work constant, not that the program is fully parallel.

---

## Q26 — Parallel Fraction from a Plot Saturation Value

> **Week reference:** Week 5

**Mental Model:** Reading S_max from a speedup-vs-cores plot and inverting Amdahl's formula — the most common exam trap is confusing the serial fraction (1 - F = 1/S_max) with the parallel fraction F, accidentally reporting 0.2 when the answer is 0.8.

A speed-up curve plotted from timing data for a production program saturates at approximately 5 as the number of cores increases. What is the parallel fraction F?

- A) F = 0.20
- B) F = 0.75
- C) F = 0.80
- D) F = 0.95

**Answer: C**

- A) Incorrect — F = 0.20 would give S_max = 1/(1 - 0.20) = 1/0.80 = 1.25. A saturation at 1.25 is nearly no speedup at all. This confuses the serial fraction (1 - F = 0.20) with the parallel fraction (F = 0.80).
- B) Incorrect — F = 0.75 gives S_max = 1/(1 - 0.75) = 4.0, not 5. A program saturating at 4 would have F = 0.75; the question states saturation at 5.
- C) Correct — S_max = 5 → 1/(1-F) = 5 → 1-F = 0.20 → F = 0.80. This is the standard inversion of Amdahl's formula and matches the DTU 2024 exam Q2 scenario exactly.
- D) Incorrect — F = 0.95 gives S_max = 1/(1 - 0.95) = 20, which is far above the observed saturation of 5. F = 0.95 would correspond to a much more parallel program.

---

## Q27 — S(p) vs T(p) Confusion Trap

> **Week reference:** Week 5

**Mental Model:** Speedup S(p) = T(1)/T(p) is always ≥ 1 and increases with p (approaching S_max). Runtime T(p) = T(1)/S(p) is always ≤ T(1) and decreases with p (approaching a floor of (1-F)×T(1)). A very common exam trap is to compute S(p) but report T(p), or vice versa.

A program with F = 0.9 runs for T(1) = 50 seconds on one core. Which answer correctly gives BOTH S(10) and T(10)?

- A) S(10) ≈ 5.26, T(10) ≈ 9.5 seconds
- B) S(10) ≈ 5.26, T(10) = 50 seconds
- C) S(10) = 10, T(10) = 5 seconds
- D) S(10) ≈ 9.5, T(10) ≈ 5.26 seconds

**Answer: A**

- A) Correct — S(10) = 1/((1-0.9) + 0.9/10) = 1/(0.1 + 0.09) = 1/0.19 ≈ 5.263. T(10) = T(1)/S(10) = 50/5.263 ≈ 9.5 s. Sanity check: the serial portion alone takes (1-F)×T(1) = 0.1×50 = 5 s, so T(10) must be above 5 s — 9.5 s is consistent.
- B) Incorrect — T(10) = 50 s would mean no speedup at all (S = 1). Using 10 cores with F = 0.9 clearly provides speedup above 1; the runtime must be well below 50 s.
- C) Incorrect — S(10) = 10 would require F = 1.0 (zero serial fraction). With a 10% serial portion, S_max = 10 is the ceiling only as p → ∞; at finite p = 10, S is strictly less than S_max. S(10) ≈ 5.26, not 10.
- D) Incorrect — this swaps the values of S(10) and T(10). The numeric value 9.5 is T(10) in seconds, not the speedup. The speedup ≈ 5.26 is a dimensionless ratio, not a time. Swapping them is the classic S(p) / T(p) confusion trap.

---

## Q28 — Serial Fraction Derived from a Two-Phase Program

> **Week reference:** Week 5

**Mental Model:** For a program split into a fixed serial phase and a fully parallel phase, the serial fraction is always the serial phase time divided by the total single-core time — not divided by the parallel phase time alone.

A program consists of two phases running on one core: a serial data-loading phase (20 s) that cannot be parallelized, and a parallel computation phase (80 s) that is fully parallelizable. What is the serial fraction used in Amdahl's Law, and what is S_max?

- A) Serial fraction = 20/80 = 0.25, S_max = 4
- B) Serial fraction = 20/100 = 0.20, S_max = 5
- C) Serial fraction = 80/100 = 0.80, S_max = 1.25
- D) Serial fraction = 20/100 = 0.20, S_max = 20

**Answer: B**

- A) Incorrect — the serial fraction is always expressed as a fraction of the total runtime, not of the parallel phase alone. 20/80 = 0.25 would mean 25% serial, giving S_max = 4, but this uses the wrong denominator.
- B) Correct — total single-core runtime T(1) = 20 + 80 = 100 s. Serial fraction = 20/100 = 0.20. Parallel fraction F = 0.80. S_max = 1/(1 - 0.80) = 1/0.20 = 5. This matches the course quiz scenario directly.
- C) Incorrect — 80/100 = 0.80 is the parallel fraction F, not the serial fraction. Plugging 0.80 in as the serial fraction into 1/(1-F) would give S_max = 1/0.20 = 5, but the formula would be misapplied (the parallel fraction cannot equal the serial fraction in Amdahl's formula).
- D) Incorrect — the serial fraction 0.20 is correct, but S_max = 1/0.20 = 5, not 20. S_max = 20 would require a serial fraction of 0.05 (5%). This error arises from multiplying the fraction by 100 before inverting.

---

## Q29 — Gustafson Scaled Speedup Exceeds Amdahl S_max

> **Week reference:** Week 5

**Mental Model:** Amdahl's S_max is a hard ceiling for a fixed workload. Gustafson's scaled speedup S_G(p) can exceed S_max because the workload grows with p — more cores process more data in the same time, so the "speedup" measures a different quantity (work done vs a hypothetical single-core run of the same large problem).

A program has serial fraction α = 0.1 (Gustafson notation). On p = 20 cores, Gustafson's scaled speedup S_G(20) = 19.05. What is Amdahl's S_max for the same program (using F = 1 - α = 0.9), and which is larger?

- A) S_max = 9; S_G = 19.05 > S_max because Gustafson's model is more optimistic.
- B) S_max = 10; S_G = 19.05 > S_max because Gustafson grows the problem with cores.
- C) S_max = 10; S_max > S_G because the Amdahl ceiling is always higher than any finite-p scaled speedup.
- D) S_max = 0.9; Gustafson and Amdahl give equivalent results when F = 1 - α.

**Answer: B**

- A) Incorrect — S_max = 1/(1 - 0.9) = 1/0.1 = 10, not 9. S_max = 9 would require a serial fraction of 1/9 ≈ 0.111, corresponding to F ≈ 0.889. The correct answer is S_max = 10.
- B) Correct — S_max (Amdahl, fixed problem) = 1/(1 - 0.9) = 10. S_G(20) = 20 - 0.1 × 19 = 19.05. S_G = 19.05 > S_max = 10 because Gustafson's model assumes the problem size scales with p: when you add cores you solve a proportionally larger problem in the same wall time. The comparison is not apples-to-apples; S_G measures scaled throughput, not speedup on a fixed problem.
- C) Incorrect — the comparison is reversed. Amdahl's S_max = 10 is lower than Gustafson's S_G(20) = 19.05. Amdahl sets a ceiling for fixed-size speedup; Gustafson's quantity can exceed that ceiling because it describes a different experiment (scaled problem size).
- D) Incorrect — S_max is not 0.9. F = 0.9 is the parallel fraction; S_max = 1/(1 - F) = 10. The two laws use the same parameters but answer different questions: Amdahl asks "how fast can you solve the same problem?" Gustafson asks "how much more work can you do in the same time?"

---

## Q30 — Hardware Ceiling vs Theoretical S_max

> **Week reference:** Week 5

**Mental Model:** S_max = 1/(1-F) is an asymptotic limit requiring infinite cores. When evaluating whether a speedup target is achievable, you must check S(p) at the actual maximum available core count — not compare the target against S_max. The target may be below S_max yet still unreachable with limited hardware.

A program has parallel fraction F = 0.8 and the best available hardware has 8 cores. Management needs a speedup of at least 4×. S_max = 5 — is the target achievable?

- A) Yes — the target (4×) is below S_max (5×), so enough cores will get there.
- B) No — S(8) ≈ 3.33 < 4, and the hardware ceiling is below the target even though S_max = 5.
- C) Yes — S(8) = 4.0 exactly with F = 0.8.
- D) No — S_max = 5, which means the speedup can never exceed 5, and 4 is within that ceiling but the hardware prevents it from being reached.

**Answer: B**

- A) Incorrect — the target being below S_max only guarantees it is theoretically achievable with a sufficient number of cores. It does not guarantee it is achievable with the specific hardware available (8 cores). Always evaluate S(p) at the actual core count.
- B) Correct — S(8) = 1/((1-0.8) + 0.8/8) = 1/(0.2 + 0.1) = 1/0.3 ≈ 3.33 < 4. The 8-core machine cannot deliver the required 4× speedup. To reach 4×, you would need to solve 4 = 1/(0.2 + 0.8/p) → p = 0.8/(0.25 - 0.2) = 16 cores. This is the exact DTU 2024 exam Q3 scenario.
- C) Incorrect — S(8) = 1/0.3 ≈ 3.33, not 4.0. For S(8) = 4.0 with p = 8 you would need 1/(0.2 + 0.8/8) = 1/0.3 ≠ 4; the denominator would need to be 0.25, requiring F ≈ 0.933.
- D) Incorrect — this answer is true but misleading: yes, the hardware prevents reaching 4× in practice, but the reasoning given is confused. S_max = 5 means the absolute ceiling is 5; 4 is within that ceiling and would be achievable with more cores (~16). The issue is that the company only has 8 cores, not that the physics forbids it.

---

## Q31 — Efficiency as p Approaches Infinity

> **Week reference:** Week 5

**Mental Model:** Efficiency E(p) = S(p)/p is a measure of how well cores are utilized. As p → ∞, S(p) → S_max (a finite constant), so E(p) → S_max/p → 0. Efficiency always falls toward zero as cores increase — it never approaches F or any other fixed fraction.

A program with parallel fraction F = 0.85 is run on successively more cores. Which statement about the long-run behavior of efficiency E(p) = S(p)/p is correct?

- A) E(p) → 0.85 as p → ∞, because efficiency is bounded below by the parallel fraction.
- B) E(p) → 0 as p → ∞, because S(p) approaches a finite constant while p grows without bound.
- C) E(p) → 1 as p → ∞, because the serial bottleneck becomes negligible relative to the total work.
- D) E(p) remains approximately constant at 1/(1-F) for large p.

**Answer: B**

- A) Incorrect — E(p) approaches 0, not F. F is the parallel fraction of work; efficiency measures utilization per core. As p → ∞, S(p) → S_max = 1/(1-F) ≈ 6.67 (for F = 0.85), so E(p) = S(p)/p → 6.67/∞ = 0. The parallel fraction F is irrelevant as a lower bound for E.
- B) Correct — S(p) → S_max = 1/(1-F) = 1/0.15 ≈ 6.67 as p → ∞. E(p) = S(p)/p → 6.67/p → 0. Efficiency is not bounded away from zero; it falls without limit as cores are added beyond the point of diminishing returns.
- C) Incorrect — E → 1 would mean perfect linear speedup (S = p), which requires F = 1.0. With a 15% serial fraction the serial portion always consumes a growing fraction of total core-time as p increases, driving E toward zero, not toward 1.
- D) Incorrect — 1/(1-F) is S_max, a speedup value greater than 1. Efficiency E(p) = S(p)/p is always strictly between 0 and 1 for any p > 1 and F < 1. Efficiency cannot equal or approach a quantity larger than 1.

---

## Q32 — Parallel Fraction Cannot Exceed 1

> **Week reference:** Week 5

**Mental Model:** F is a fraction of total runtime so it must satisfy 0 ≤ F ≤ 1. If back-solving yields F > 1, there is an error in the data or the formula — the most common cause is accidentally using T(p)/T(1) (slowdown ratio) instead of T(1)/T(p) (speedup).

A student measures T(1) = 50 s and T(4) = 60 s and applies the formula F = p(1 - 1/S)/(p - 1) with S = T(1)/T(p) = 50/60. They obtain a negative F value. What does this indicate?

- A) The serial fraction exceeds 100%, which is valid for programs with synchronization overhead.
- B) The parallel code is slower than the serial code (slowdown), violating the assumption of Amdahl's Law that adding cores helps.
- C) F is negative because the formula requires S > 1; a sub-unity speedup indicates measurement error that must be discarded.
- D) The student used the wrong formula; the correct one gives F between 0 and 1 for any S value.

**Answer: B**

- A) Incorrect — a parallel fraction cannot exceed 100% (F > 1 is not meaningful) and cannot be negative. Synchronization overhead is a real-world cost not modelled by basic Amdahl's Law; it is one reason observed speedup can be less than 1.
- B) Correct — S = T(1)/T(p) = 50/60 ≈ 0.833 < 1 means the 4-core run is slower than the 1-core run (slowdown). In this regime, 1 - 1/S = 1 - 1.2 = -0.2, giving F = 4 × (-0.2) / 3 ≈ -0.267. A negative F signals that the data violates the Amdahl assumption (more cores → no improvement). Real causes include excessive IPC overhead, memory bandwidth saturation, or process-spawn costs dominating a short workload.
- C) Incorrect — the formula is correct; it is defined for any S. For S < 1 it correctly returns a negative F, which is a diagnostic result indicating invalid (slowdown) data rather than a programming error in the formula.
- D) Incorrect — the formula is valid regardless of S. There is no alternative formula that produces F ∈ [0, 1] when S < 1; the data itself is outside the regime where Amdahl's Law applies.

---
