# Parallel Reduction — MCQ Practice

> Topics: Associativity, commutativity, counterexamples, tree depth, valid/invalid operators.
> Exam frequency: **Every exam**.

---

## Q1 — Associativity Requirement

> **Week reference:** Week 6

**Mental Model:** Parallel reduction splits the array into sub-groups and merges results — different groupings must produce the same answer. Trap: students confuse commutativity (order of operands) with associativity (order of grouping) and pick A.

Which property is **required** for an operator to be used in a parallel reduction that splits work into subproblems and combines the results?

- A) Commutativity: `a op b == b op a`
- B) Associativity: `(a op b) op c == a op (b op c)`
- C) Idempotency: `a op a == a`
- D) Distributivity over addition

**Answer: B**

- A) Incorrect — commutativity only matters when combining partial results in arbitrary order; it does not guarantee correctness of different groupings. You can have an associative but non-commutative operator (e.g. matrix multiply) that is still safe for ordered parallel reductions.
- B) Correct — associativity guarantees that regrouping sub-problems does not change the final result, which is exactly what parallel decomposition requires. Without it, Thread1 computing (a op b) then merging with Thread2's (c op d) may give a different answer than the serial left-to-right pass.
- C) Incorrect — idempotency (e.g. min on the same element) is not required; most useful operators like sum are not idempotent. It is a separate algebraic property with no role in reduction correctness.
- D) Incorrect — distributivity over addition is a relationship between two operators (e.g. multiplication distributes over addition); it is entirely unrelated to the single-operator parallel reduction correctness requirement.

---

## Q2 — Commutativity in Unordered Reductions

> **Week reference:** Week 6

**Mental Model:** In a tree reduction, which thread pairs with which is non-deterministic — so the operand order within each pair is also non-deterministic. Commutativity pins down that `a op b == b op a` regardless of pairing order.

In a parallel tree reduction, worker threads may combine partial results in any order. Which additional property (beyond associativity) is required for the reduction to remain correct regardless of combination order?

- A) Monotonicity
- B) Distributivity
- C) Commutativity
- D) Invertibility

**Answer: C**

- A) Incorrect — monotonicity (result only grows or shrinks) is a descriptive property of some operators but places no constraint on whether arbitrary combination order produces the correct result.
- B) Incorrect — distributivity describes how two different operators interact (e.g. `a*(b+c) == a*b + a*c`); it is irrelevant to combining partial results of a single-operator reduction.
- C) Correct — commutativity (`a op b == b op a`) ensures Thread1 computing `partial_A op partial_B` gives the same result as Thread2 computing `partial_B op partial_A`, which is necessary when merge order is non-deterministic.
- D) Incorrect — invertibility (existence of an inverse, e.g. subtraction for addition) is not required; many valid reduction operators like max and min have no inverse.

---

## Q3 — abssum Commutativity Test

> **Week reference:** Week 6

**Mental Model:** Break the operator into its components: `abssum(x,y) = abs(x+y)`. Addition is always commutative, so the outer `abs` of a commutative inner expression is also commutative. Don't confuse commutativity with associativity here.

Consider `abssum(x, y) = abs(x + y)`. Is this operator **commutative**?

- A) No — `abs(x + y) != abs(y + x)` in general
- B) Yes — `abs(x + y) == abs(y + x)` always
- C) Only when both x and y are non-negative
- D) Only when x and y have the same sign

**Answer: B**

- A) Incorrect — addition is commutative for all real numbers, so `x + y == y + x` always. Applying `abs` to equal quantities gives equal results; no counterexample exists.
- B) Correct — since `x + y == y + x` for all real numbers, it follows that `abs(x + y) == abs(y + x)` universally. The sign, magnitude, and relative values of x and y are irrelevant.
- C) Incorrect — the equality holds for all reals including negative and mixed-sign inputs; restricting to non-negatives is an unnecessary and incorrect qualification.
- D) Incorrect — same-sign is not a requirement; the commutative equality holds for e.g. x=3, y=−7 just as well as for same-sign pairs.

---

## Q4 — abssum Associativity Counterexample

> **Week reference:** Week 6

**Mental Model:** To disprove associativity, find one counterexample with mixed signs so intermediate abs calls cancel negatives differently. The numbers 1, 2, −3 are the canonical counterexample: they sum to 0 in one grouping but not the other.

Consider `abssum(x, y) = abs(x + y)`. Evaluate `abssum(abssum(1, 2), -3)` and `abssum(1, abssum(2, -3))`. What do you get?

- A) Both equal 0
- B) Both equal 2
- C) Left-associative gives 0; right-associative gives 2
- D) Left-associative gives 2; right-associative gives 0

**Answer: C**

- A) Incorrect — only the left-associative grouping reaches 0; the right-associative path passes through `abs(2+(−3)) = 1` before the final step.
- B) Incorrect — only the right-associative grouping gives 2; the left-associative path cancels to 0 at the first inner call.
- C) Correct — Left grouping: `abssum(1,2) = abs(3) = 3`, then `abssum(3,−3) = abs(0) = 0`. Right grouping: `abssum(2,−3) = abs(−1) = 1`, then `abssum(1,1) = abs(2) = 2`. Result: 0 ≠ 2, proving non-associativity.
- D) Incorrect — the values are the right ones but assigned to the wrong groupings; left gives 0, not 2.

---

## Q5 — Why abssum Fails Parallel Reduction

> **Week reference:** Week 6

**Mental Model:** The most common trap on this question is answering A (not commutative) — abssum IS commutative. The failure is purely non-associativity: the premature abs call inside the operator corrupts intermediate partial results.

`abssum(x, y) = abs(x + y)` cannot be safely used as a parallel reduction operator. What is the **correct reason**?

- A) abssum is not commutative
- B) abssum is not associative
- C) abssum is both non-commutative and non-associative
- D) abssum does not have an identity element

**Answer: B**

- A) Incorrect — abssum IS commutative: `abs(x+y) == abs(y+x)` always. Selecting this answer is the classic trap; commutativity is not the problem.
- B) Correct — abssum is NOT associative, as proven by `|(1+2)+(−3)| = 0 ≠ |1+(2+(−3))| = 2`. In a parallel reduction, different thread groupings produce different final answers, making the result non-deterministic and incorrect.
- C) Incorrect — abssum is commutative; only non-associativity causes the failure. Claiming both properties fail is factually wrong and would mislead any debugging effort.
- D) Incorrect — while 0 serves as an identity element for abssum (since `abs(x+0) = abs(x)` when x ≥ 0, though this breaks for negative x), the primary and decisive failure is non-associativity.

---

## Q6 — Fixing abssum for Parallel Use

> **Week reference:** Week 6

**Mental Model:** Decouple the non-associative `abs` from the associative `+`. Run parallel sum (valid because + is both associative and commutative), then apply `abs` exactly once to the scalar result. Never push the abs inside the reduction step.

You need to compute `abs(sum(array))` in parallel. Which approach is correct?

- A) Use abssum as the reduction operator directly in a parallel tree reduction
- B) Compute a parallel sum using the `+` operator, then take `abs` of the final scalar result
- C) Apply `abs` to each element first, then do a parallel sum
- D) abssum cannot be parallelised at all; it must be done serially

**Answer: B**

- A) Incorrect — using abssum directly fails because it is not associative; every intermediate abs call potentially zeroes out cancellations that should survive to the final step, giving a wrong answer.
- B) Correct — `+` is both associative and commutative, so parallel sum is valid and gives the exact same result as serial sum. Applying `abs` once to the scalar final result then correctly computes `abs(sum(array))`.
- C) Incorrect — `sum(abs(x_i))` is the L1 norm (sum of absolute values), which is a different and generally larger quantity than `abs(sum(x_i))`. For example, `abs(1 + (−1)) = 0` but `abs(1) + abs(−1) = 2`.
- D) Incorrect — the decomposition in B shows the computation is fully parallelisable; the serial constraint only arises if abssum is incorrectly used as the reduction operator itself.

---

## Q7 — Valid Reduction Operators

> **Week reference:** Week 6

**Mental Model:** Check both conditions: associative (grouping doesn't matter) AND commutative (order doesn't matter). `max` passes both trivially. Subtraction fails both, making it a good foil for understanding why both properties are needed.

Which of the following operators is **valid** for use in a parallel reduction (associative AND commutative)?

- A) Matrix multiplication
- B) String concatenation
- C) `max(a, b)`
- D) Subtraction

**Answer: C**

- A) Incorrect — matrix multiplication is associative (`(AB)C == A(BC)` always) but NOT commutative in general (`AB ≠ BA` for most non-square or non-diagonal matrices). An unordered parallel tree reduction could combine partial products in the wrong order, corrupting the result.
- B) Incorrect — string concatenation is associative (`("ab"+"cd")+"ef" == "ab"+("cd"+"ef")`) but NOT commutative (`"ab"+"cd" != "cd"+"ab"`). Order of partial results matters, so unordered merging would give wrong strings.
- C) Correct — `max` is both associative (`max(max(a,b),c) == max(a,max(b,c))` by definition of maximum) and commutative (`max(a,b) == max(b,a)`). It satisfies both requirements for a parallel reduction.
- D) Incorrect — subtraction is neither associative (`(5−3)−1 = 1` vs `5−(3−1) = 3`) nor commutative (`a−b ≠ b−a` in general). Both properties fail, making it doubly invalid.

---

## Q8 — Matrix Multiply in Parallel Reduction

> **Week reference:** Week 6

**Mental Model:** The key distinction: matrix multiplication IS associative (chain rule works) but IS NOT commutative. An unordered tree reduction only needs commutativity to be wrong — the parallel pairing could silently reorder factors.

Why can matrix multiplication **not** be used as the operator in an unordered parallel tree reduction?

- A) It is not associative: `(AB)C != A(BC)` in general
- B) It is not commutative: `AB != BA` in general
- C) It is both non-associative and non-commutative
- D) It has no identity element

**Answer: B**

- A) Incorrect — matrix multiplication IS associative: `(AB)C == A(BC)` always holds; this is one of the fundamental properties of matrix multiplication used in linear algebra. Selecting A is the main trap.
- B) Correct — matrix multiplication is NOT commutative in general. For example, `[[1,2],[0,1]] × [[1,0],[1,1]] ≠ [[1,0],[1,1]] × [[1,2],[0,1]]`. An unordered tree reduction may combine `partial_A × partial_B` instead of `partial_B × partial_A`, giving a wrong result.
- C) Incorrect — matrix multiplication is associative; claiming both properties fail is factually wrong and would mislead analysis of other matrix operations.
- D) Incorrect — the identity matrix I satisfies `AI == IA == A` for any conformable A, so a valid identity element exists.

---

## Q9 — Set Intersection as Reduction Operator

> **Week reference:** Week 6

**Mental Model:** Set intersection behaves like `min` on membership — both are symmetric and grouping-independent. Verify by thinking: does order of intersection matter? Does grouping matter? Both answers are no, confirming it's valid.

Is set intersection (`∩`) a valid operator for parallel reduction?

- A) No — it is commutative but not associative
- B) No — it is associative but not commutative
- C) Yes — it is both associative and commutative
- D) No — it has no identity element, so a neutral initial value cannot be set

**Answer: C**

- A) Incorrect — set intersection is also associative: `(A∩B)∩C == A∩(B∩C)` because an element is in the result iff it is in all three sets, regardless of how the intersection is parenthesised.
- B) Incorrect — set intersection is also commutative: `A∩B == B∩A` because an element is in both sets symmetrically; there is no notion of "first" or "second" set.
- C) Correct — both properties hold. `A∩B == B∩A` (commutative) and `(A∩B)∩C == A∩(B∩C)` (associative), satisfying the requirements for a valid parallel reduction operator.
- D) Incorrect — the universal set U (containing all elements under consideration) acts as the identity: `A∩U == A` for any set A. An identity element does exist.

---

## Q10 — Binary Tree Reduction Depth

> **Week reference:** Week 6

**Mental Model:** Each round halves the number of partial results. Starting from N, you need ceil(log2(N)) halvings to reach 1. For N=16 = 2^4, that is exactly 4 rounds. Speedup over serial: N rounds → log2(N) rounds, ratio = N/log2(N).

A parallel binary tree reduction is applied to an array of **N = 16** elements, with enough processors so every pair can be combined simultaneously. How many rounds (parallel steps) are needed to produce the final result?

- A) 8 rounds
- B) 16 rounds
- C) 4 rounds
- D) 2 rounds

**Answer: C**

- A) Incorrect — 8 rounds would correspond to N/2 = 8 sequential pair-wise operations in a flat scheme, not a tree. A tree cuts rounds by a further factor of log₂(N)/N each level.
- B) Incorrect — 16 rounds is the serial cost, where each element is processed one at a time. The tree reduction's whole purpose is to reduce this to log₂(N) rounds.
- C) Correct — a binary tree on N=16 elements has depth `ceil(log₂(16)) = log₂(16) = 4` rounds: 16→8→4→2→1 partial results, one combine operation per round.
- D) Incorrect — 2 rounds would only handle N=4 elements (2²=4); for N=16 you need 4 halvings.

---

## Q11 — Flat Two-Level Reduction: Optimal Chunks and Speedup

> **Week reference:** Week 6

**Mental Model:** Two competing costs: intra-chunk serial work (N/T, decreases with more chunks) and inter-chunk merge work (T, increases with more chunks). Optimal T minimises their sum; calculus gives T = sqrt(N). Speedup = N / (2*sqrt(N)) = sqrt(N)/2.

In a flat (two-level) parallel reduction, N = 100 elements are divided into T equal chunks. Each chunk is reduced serially (cost N/T), then the T partial results are combined serially (cost T). Total time ≈ N/T + T. What is the **optimal** number of chunks T, and what is the resulting **speedup** over serial (cost N)?

- A) Optimal T = N/2 = 50; speedup = 2
- B) Optimal T = log2(N) ≈ 7; speedup ≈ N/log2(N) ≈ 14
- C) Optimal T = sqrt(N) = 10; speedup = sqrt(N)/2 = 5
- D) Optimal T = N = 100; speedup = N = 100

**Answer: C**

- A) Incorrect — T = N/2 = 50 gives total time N/(N/2) + N/2 = 2 + 50 = 52, much worse than optimal. Speedup = 100/52 ≈ 1.9, not 2.
- B) Incorrect — T = log₂(N) ≈ 7 corresponds to the depth of a binary tree reduction, which is a different algorithm; in the flat two-level model, T=7 gives time ≈ 100/7 + 7 ≈ 21.3, close to but not optimal.
- C) Correct — minimising N/T + T: take derivative d/dT = −N/T² + 1 = 0, giving T = sqrt(N) = 10. Total time = 100/10 + 10 = 20. Speedup = N / (2·sqrt(N)) = sqrt(N)/2 = 10/2 = 5.
- D) Incorrect — T = N = 100 means each chunk has 1 element (no intra-chunk work), then all 100 partials merged serially: time = 1 + 100 = 101, which is actually slower than serial.

---
