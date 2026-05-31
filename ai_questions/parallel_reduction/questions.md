# Parallel Reduction — MCQ Practice

> Topics: Associativity, commutativity, counterexamples, tree depth, valid/invalid operators.
> Exam frequency: **Every exam**.

---

## Q1 — Associativity Requirement

> **Week reference:** Week 6

Which property is **required** for an operator to be used in a parallel reduction that splits work into subproblems and combines the results?

- A) Commutativity: `a op b == b op a`
- B) Associativity: `(a op b) op c == a op (b op c)`
- C) Idempotency: `a op a == a`
- D) Distributivity over addition

**Answer: B**

- A) Incorrect — commutativity is also needed for unordered reductions, but associativity is the core requirement for splitting and recombining subproblems.
- B) Correct — associativity guarantees that grouping subproblems differently does not change the result, which is essential for parallel decomposition.
- C) Incorrect — idempotency (e.g. min on the same element) is not required in general for parallel reductions.
- D) Incorrect — distributivity over addition is unrelated to the parallel reduction correctness requirement.

---

## Q2 — Commutativity in Unordered Reductions

> **Week reference:** Week 6

In a parallel tree reduction, worker threads may combine partial results in any order. Which additional property (beyond associativity) is required for the reduction to remain correct regardless of combination order?

- A) Monotonicity
- B) Distributivity
- C) Commutativity
- D) Invertibility

**Answer: C**

- A) Incorrect — monotonicity (result only grows/shrinks) is not required for correctness of parallel reductions.
- B) Incorrect — distributivity over another operation is not needed for a parallel reduction.
- C) Correct — commutativity (`a op b == b op a`) ensures that combining operands in any order gives the same result, which is necessary when threads merge partial results without fixed ordering.
- D) Incorrect — invertibility (existence of an inverse element) is not required for reduction.

---

## Q3 — abssum Commutativity Test

> **Week reference:** Week 6

Consider `abssum(x, y) = abs(x + y)`. Is this operator **commutative**?

- A) No — `abs(x + y) != abs(y + x)` in general
- B) Yes — `abs(x + y) == abs(y + x)` always
- C) Only when both x and y are non-negative
- D) Only when x and y have the same sign

**Answer: B**

- A) Incorrect — addition is commutative, so `x + y == y + x`, meaning `abs(x + y) == abs(y + x)` always.
- B) Correct — since `x + y == y + x` for all real numbers, taking abs of both sides gives equality, so abssum is commutative.
- C) Incorrect — the equality holds for all real numbers, not just non-negative ones.
- D) Incorrect — sign does not affect commutativity here; the equality is universal.

---

## Q4 — abssum Associativity Counterexample

> **Week reference:** Week 6

Consider `abssum(x, y) = abs(x + y)`. Evaluate `abssum(abssum(1, 2), -3)` and `abssum(1, abssum(2, -3))`. What do you get?

- A) Both equal 0
- B) Both equal 2
- C) Left-associative gives 0; right-associative gives 2
- D) Left-associative gives 2; right-associative gives 0

**Answer: C**

- A) Incorrect — only the left-associative grouping gives 0.
- B) Incorrect — only the right-associative grouping gives 2.
- C) Correct — Left grouping: `abssum(abssum(1,2), -3) = abssum(3, -3) = abs(0) = 0`. Right grouping: `abssum(2,-3) = abs(-1) = 1`, then `abssum(1, 1) = abs(2) = 2`. So left = 0 ≠ right = 2 — not associative.
- D) Incorrect — the left grouping gives 0 and the right grouping gives 2, not the other way around.

---

## Q5 — Why abssum Fails Parallel Reduction

> **Week reference:** Week 6

`abssum(x, y) = abs(x + y)` cannot be safely used as a parallel reduction operator. What is the **correct reason**?

- A) abssum is not commutative
- B) abssum is not associative
- C) abssum is both non-commutative and non-associative
- D) abssum does not have an identity element

**Answer: B**

- A) Incorrect — abssum IS commutative (`abs(x+y) == abs(y+x)`); this is a common trap.
- B) Correct — abssum is NOT associative, as shown by the counterexample `|(1+2)+(-3)| = 0` but `|1+(2+(-3))| = 2`. Different groupings give different results.
- C) Incorrect — abssum is commutative; only non-associativity causes the failure.
- D) Incorrect — while the identity element question is secondary, the primary failure is lack of associativity.

---

## Q6 — Fixing abssum for Parallel Use

> **Week reference:** Week 6

You need to compute `abs(sum(array))` in parallel. Which approach is correct?

- A) Use abssum as the reduction operator directly in a parallel tree reduction
- B) Compute a parallel sum using the `+` operator, then take `abs` of the final scalar result
- C) Apply `abs` to each element first, then do a parallel sum
- D) abssum cannot be parallelised at all; it must be done serially

**Answer: B**

- A) Incorrect — using abssum directly fails because it is not associative; intermediate abs calls corrupt the result.
- B) Correct — `+` is both associative and commutative, so parallel sum is valid; applying `abs` once to the final scalar produces the correct `abs(sum(array))`.
- C) Incorrect — `sum(abs(x_i))` computes the sum of absolute values, which is a different (and generally larger) quantity than `abs(sum(x_i))`.
- D) Incorrect — the computation can be parallelised via the approach in B.

---

## Q7 — Valid Reduction Operators

> **Week reference:** Week 6

Which of the following operators is **valid** for use in a parallel reduction (associative AND commutative)?

- A) Matrix multiplication
- B) String concatenation
- C) `max(a, b)`
- D) Subtraction

**Answer: C**

- A) Incorrect — matrix multiplication is associative but NOT commutative in general (`AB ≠ BA`), so it fails for unordered parallel reductions.
- B) Incorrect — string concatenation is associative but NOT commutative (`"ab" != "ba"`), so it also fails for unordered reductions.
- C) Correct — `max` is both associative (`max(max(a,b),c) == max(a,max(b,c))`) and commutative (`max(a,b) == max(b,a)`), making it valid for parallel reduction.
- D) Incorrect — subtraction is neither associative (`(a-b)-c != a-(b-c)`) nor commutative (`a-b != b-a`).

---

## Q8 — Matrix Multiply in Parallel Reduction

> **Week reference:** Week 6

Why can matrix multiplication **not** be used as the operator in an unordered parallel tree reduction?

- A) It is not associative: `(AB)C != A(BC)` in general
- B) It is not commutative: `AB != BA` in general
- C) It is both non-associative and non-commutative
- D) It has no identity element

**Answer: B**

- A) Incorrect — matrix multiplication IS associative: `(AB)C == A(BC)` always holds; this is a common trap.
- B) Correct — matrix multiplication is NOT commutative, so combining partial products in arbitrary order (as a tree reduction does) can yield wrong results.
- C) Incorrect — MM is associative; only non-commutativity is the issue.
- D) Incorrect — the identity matrix I satisfies `AI == IA == A`, so an identity element exists.

---

## Q9 — Set Intersection as Reduction Operator

> **Week reference:** Week 6

Is set intersection (`∩`) a valid operator for parallel reduction?

- A) No — it is commutative but not associative
- B) No — it is associative but not commutative
- C) Yes — it is both associative and commutative
- D) No — it has no identity element, so a neutral initial value cannot be set

**Answer: C**

- A) Incorrect — set intersection is also associative: `(A∩B)∩C == A∩(B∩C)`.
- B) Incorrect — set intersection is also commutative: `A∩B == B∩A`.
- C) Correct — `A∩B == B∩A` (commutative) and `(A∩B)∩C == A∩(B∩C)` (associative), so set intersection satisfies both requirements for parallel reduction.
- D) Incorrect — the universal set U (or the union of all sets in context) acts as an identity element: `A∩U == A`.

---

## Q10 — Binary Tree Reduction Depth

> **Week reference:** Week 6

A parallel binary tree reduction is applied to an array of **N = 16** elements, with enough processors so every pair can be combined simultaneously. How many rounds (parallel steps) are needed to produce the final result?

- A) 8 rounds
- B) 16 rounds
- C) 4 rounds
- D) 2 rounds

**Answer: C**

- A) Incorrect — 8 rounds would correspond to N/2 sequential steps, not a tree reduction.
- B) Incorrect — 16 rounds is the serial cost; the tree reduction is faster.
- C) Correct — a binary tree on N=16 elements has depth `ceil(log2(16)) = log2(16) = 4` rounds.
- D) Incorrect — 2 rounds would only handle 4 elements (2^2), not 16.

---

## Q11 — Flat Two-Level Reduction: Optimal Chunks and Speedup

> **Week reference:** Week 6

In a flat (two-level) parallel reduction, N = 100 elements are divided into T equal chunks. Each chunk is reduced serially (cost N/T), then the T partial results are combined serially (cost T). Total time ≈ N/T + T. What is the **optimal** number of chunks T, and what is the resulting **speedup** over serial (cost N)?

- A) Optimal T = N/2 = 50; speedup = 2
- B) Optimal T = log2(N) ≈ 7; speedup ≈ N/log2(N) ≈ 14
- C) Optimal T = sqrt(N) = 10; speedup = sqrt(N)/2 = 5
- D) Optimal T = N = 100; speedup = N = 100

**Answer: C**

- A) Incorrect — T = N/2 gives time N/(N/2) + N/2 = 2 + 50 = 52, not optimal.
- B) Incorrect — T = log2(N) corresponds to a binary tree reduction depth, not a flat two-level scheme.
- C) Correct — minimising N/T + T by setting the derivative to zero gives T = sqrt(N) = 10. Time = 100/10 + 10 = 20. Speedup = N / (2*sqrt(N)) = sqrt(N)/2 = 5.
- D) Incorrect — T = N means each chunk has 1 element, time = 1 + 100 = 101, which is actually slower than serial.

---
