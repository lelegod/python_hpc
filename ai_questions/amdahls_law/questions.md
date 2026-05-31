# Amdahl's Law — MCQ Practice

> Topics: S(p) formula, solving for F, maximum speedup, efficiency, serial time calculations.
> Exam frequency: **Every exam** — highest priority topic.

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
