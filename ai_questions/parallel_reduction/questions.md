# Parallel Reduction — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Associativity Requirement](#q1-associativity-requirement)
- [Q2 — Commutativity in Unordered Reductions](#q2-commutativity-in-unordered-reductions)
- [Q3 — abssum Commutativity Test](#q3-abssum-commutativity-test)
- [Q4 — abssum Associativity Counterexample](#q4-abssum-associativity-counterexample)
- [Q5 — Why abssum Fails Parallel Reduction](#q5-why-abssum-fails-parallel-reduction)
- [Q6 — Fixing abssum for Parallel Use](#q6-fixing-abssum-for-parallel-use)
- [Q7 — Valid Reduction Operators](#q7-valid-reduction-operators)
- [Q8 — Matrix Multiply in Parallel Reduction](#q8-matrix-multiply-in-parallel-reduction)
- [Q9 — Set Intersection as Reduction Operator](#q9-set-intersection-as-reduction-operator)
- [Q10 — Binary Tree Reduction Depth](#q10-binary-tree-reduction-depth)
- [Q11 — Flat Two-Level Reduction: Optimal Chunks and Speedup](#q11-flat-two-level-reduction-optimal-chunks-and-speedup)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q12 — Subtraction as Reduction Operator](#q12-subtraction-as-reduction-operator)
- [Q13 — Average via Parallel Reduction](#q13-average-via-parallel-reduction)
- [Q14 — Tree Reduction Depth for N=1024](#q14-tree-reduction-depth-for-n1024)
- [Q15 — Pool.map Return Value](#q15-poolmap-return-value)
- [Q16 — functools.reduce after Pool.map](#q16-functoolsreduce-after-poolmap)
- [Q17 — Warp Reduction Step Count](#q17-warp-reduction-step-count)
- [Q18 — cuda.syncthreads() Purpose in Shared Memory Reduction](#q18-cudasyncthreads-purpose-in-shared-memory-reduction)
- [Q19 — cuda.atomic.add Purpose](#q19-cudaatomicadd-purpose)
- [Q20 — XOR as Reduction Operator](#q20-xor-as-reduction-operator)
- [Q21 — Non-Power-of-Two Tree Depth](#q21-non-power-of-two-tree-depth)

---

> Topics: Associativity, commutativity, counterexamples, tree depth, valid/invalid operators.
> Exam frequency: **Every exam**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--associativity-requirement)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

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

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets commutativity/associativity requirements, tree reduction depth, warp-level reduction, and Pool.map + functools.reduce patterns

---

## Q12 — Subtraction as Reduction Operator

> **Week reference:** Week 6

A student proposes using subtraction as the operator in a parallel tree reduction to compute the "signed alternating sum" of an array. Which statement is correct?

- A) Subtraction is valid because it is commutative
- B) Subtraction is invalid because it is neither commutative nor associative
- C) Subtraction is valid because it is associative
- D) Subtraction is invalid only for floating-point arrays; it works for integers

**Answer: B**

Subtraction fails both requirements. It is not commutative: `a − b ≠ b − a` in general (e.g. 5 − 3 = 2, but 3 − 5 = −2). It is not associative: `(5 − 3) − 1 = 1` but `5 − (3 − 1) = 3`. Since a parallel tree reduction requires both properties for correct results regardless of thread pairing and merge order, subtraction is doubly disqualified. The integer vs. floating-point distinction is irrelevant; the algebraic failure applies to all number types.

---

## Q13 — Average via Parallel Reduction

> **Week reference:** Week 6

You want to compute the average of a large array in parallel. A colleague suggests using the average operator `avg(a, b) = (a + b) / 2` as the reduction operator in a tree reduction. Why is this wrong?

- A) avg is not commutative, so thread ordering affects the result
- B) avg is commutative and associative, so it is valid
- C) avg is not associative; different groupings give different results
- D) avg cannot be parallelised at all and must be computed serially

**Answer: C**

`avg(a, b) = (a + b) / 2` is commutative (`avg(a,b) = avg(b,a)`) but not associative. Counterexample: `avg(avg(1, 3), 5) = avg(2, 5) = 3.5`, but `avg(1, avg(3, 5)) = avg(1, 4) = 2.5`. Different tree groupings give different results. The correct approach is to use parallel sum (which is associative and commutative) and divide the scalar result by N once at the end.

---

## Q14 — Tree Reduction Depth for N=1024

> **Week reference:** Week 6

A parallel binary tree reduction is applied to an array of N = 1024 elements, with sufficient processors for all pairs to combine simultaneously each round. How many parallel rounds are required?

- A) 512 rounds
- B) 10 rounds
- C) 1024 rounds
- D) 5 rounds

**Answer: B**

For N = 1024 = 2^10, the tree depth is ceil(log₂(1024)) = 10 rounds. Each round halves the active partial results: 1024 → 512 → 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1. The serial cost would be 1023 operations; the parallel tree compresses this to 10 rounds. 512 rounds (A) is the number of operations in round 1. 5 rounds (D) corresponds to N = 32, not 1024.

---

## Q15 — Pool.map Return Value

> **Week reference:** Week 5

```python
from multiprocessing import Pool

def partial_sum(chunk):
    return sum(chunk)

chunks = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
with Pool(3) as p:
    result = p.map(partial_sum, chunks)
```

What does `result` contain after `Pool.map` completes?

- A) A single integer: the total sum of all elements
- B) A list of three integers: `[6, 15, 24]`
- C) A Pool object with lazy results
- D) A list of three lists, each unchanged from the input

**Answer: B**

`Pool.map` applies `partial_sum` to each chunk in parallel and returns a list of the individual return values in the same order as the input. Here `partial_sum([1,2,3])=6`, `partial_sum([4,5,6])=15`, `partial_sum([7,8,9])=24`, so `result = [6, 15, 24]`. To obtain the total sum a second serial step is needed: `sum(result)` or `functools.reduce(lambda a,b: a+b, result)`. A common exam trap is thinking `Pool.map` returns a single reduced scalar (answer A).

---

## Q16 — functools.reduce after Pool.map

> **Week reference:** Week 5

After `Pool.map` returns partial results from each worker, a student writes:

```python
from functools import reduce
final = reduce(lambda a, b: a + b, partial_results)
```

What role does `reduce` play here and is it parallel or serial?

- A) It distributes the combination work across all Pool workers — parallel
- B) It applies the lambda to each element independently — parallel
- C) It serially combines the partial results left-to-right into a single value — serial
- D) It is equivalent to `sum(partial_results)` but runs in a new process — parallel

**Answer: C**

`functools.reduce` is a purely serial fold: it applies the binary function left-to-right across the iterable, accumulating a single value. For `[6, 15, 24]` it computes `((6 + 15) + 24) = 45`. It does not spawn processes or threads. Its role in the two-phase reduction pattern is the serial "combine" step that merges the small list of partial results produced by the parallel `Pool.map` phase. Answer A is wrong because reduce has no knowledge of the Pool. Answer D is wrong: while `reduce(lambda a,b:a+b, xs)` and `sum(xs)` produce the same numerical result, `reduce` runs in the calling process, not a new one.

---

## Q17 — Warp Reduction Step Count

> **Week reference:** Week 9

A CUDA warp contains 32 threads performing a parallel reduction using shuffle-down instructions (`__shfl_down_sync`). The reduction proceeds with strides 16, 8, 4, 2, 1. How many shuffle steps are needed to produce the final result in thread 0?

- A) 32 steps
- B) 6 steps
- C) 5 steps
- D) 4 steps

**Answer: C**

A warp has 32 = 2^5 threads, so the tree depth is ceil(log₂(32)) = 5. Each shuffle-down step halves the active thread count: stride 16 (32→16 active), stride 8 (16→8), stride 4 (8→4), stride 2 (4→2), stride 1 (2→1). Five steps total, with no `__syncthreads()` needed since all threads in a warp execute synchronously. Answer B (6) would correspond to N=64; answer D (4) would correspond to N=16.

---

## Q18 — cuda.syncthreads() Purpose in Shared Memory Reduction

> **Week reference:** Week 9

In a CUDA shared-memory reduction kernel, `cuda.syncthreads()` is called after each stride. What happens if `cuda.syncthreads()` is omitted?

- A) The kernel runs faster because synchronisation overhead is removed; results are still correct
- B) Threads may read stale or partially-written shared memory values from the previous stride, producing incorrect results
- C) The reduction silently falls back to a serial loop
- D) Only the first block produces a correct result; all other blocks fail

**Answer: B**

Without `cuda.syncthreads()`, threads within a block are not guaranteed to have completed their writes to shared memory before other threads begin reading those locations in the next stride. This is a classic data race: thread `tid` may read `shared[tid + stride]` before the thread responsible for `tid + stride` has written its updated value. The result is non-deterministic and typically wrong. The synchronisation barrier does add overhead (A is wrong), but correctness cannot be traded for speed here. The fallback to serial (C) does not happen in CUDA; the hardware simply allows the race to proceed.

---

## Q19 — cuda.atomic.add Purpose

> **Week reference:** Week 9

At the end of a CUDA block-level reduction, thread 0 of each block executes:

```python
if tid == 0:
    cuda.atomic.add(result, 0, shared[0])
```

Why is `cuda.atomic.add` used instead of `result[0] += shared[0]`?

- A) `cuda.atomic.add` is faster than the += operator for GPU memory
- B) Multiple blocks finish at unpredictable times and may simultaneously write to `result[0]`; atomic add prevents lost updates
- C) `result[0] += shared[0]` would cause shared memory corruption
- D) The atomic operation is needed to trigger a final `cuda.syncthreads()` across all blocks

**Answer: B**

When multiple blocks each contribute a partial result to the same global output scalar, their thread-0 threads can execute simultaneously. Without atomicity, two threads reading `result[0]`, adding their partial sums, and writing back could overwrite each other — a classic read-modify-write race. `cuda.atomic.add` guarantees that each update is applied to the current value without interference. Answer A is wrong: atomic operations are slower than plain memory writes due to serialisation. Answer C is wrong: shared memory is per-block and `result` is global memory; the issue is global memory contention. Answer D is wrong: `cuda.syncthreads()` only synchronises threads within a block; there is no cross-block barrier in standard CUDA.

---

## Q20 — XOR as Reduction Operator

> **Week reference:** Week 6

Is the bitwise XOR operator (`^`) valid for use in a parallel reduction?

- A) No — XOR is commutative but not associative
- B) No — XOR is associative but not commutative
- C) Yes — XOR is both associative and commutative
- D) Only for arrays of even length

**Answer: C**

Bitwise XOR is both commutative (`a ^ b = b ^ a` by bit-level symmetry) and associative (`(a ^ b) ^ c = a ^ (b ^ c)` because each bit position is independent and XOR is a Boolean operation satisfying both properties). It therefore satisfies both requirements for a parallel reduction. A common use is checking parity or detecting duplicates: XOR-reduce an array; if a value appears an even number of times it cancels out. Array length (D) has no bearing on the algebraic properties.

---

## Q21 — Non-Power-of-Two Tree Depth

> **Week reference:** Week 6

A parallel binary tree reduction is applied to an array of N = 20 elements. How many parallel rounds are needed?

- A) 4 rounds
- B) 5 rounds
- C) 10 rounds
- D) 20 rounds

**Answer: B**

For a non-power-of-two N, the tree depth is ceil(log₂(N)). log₂(20) ≈ 4.32, so ceil(4.32) = 5 rounds. In practice, the first round pairs up elements; 20 elements give 10 pairs → 10 partials, then 5 → 3 (one element sits idle) → 2 → 1, taking 5 rounds total. Answer A (4 rounds) is wrong because 2^4 = 16 < 20; 4 rounds can only handle up to 16 elements exactly. Answer C (10) is N/2 — the number of operations in round 1, not the total rounds.

---
