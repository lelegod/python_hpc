# Parallel Reduction — Code-Based MCQ Practice

> Format: Each question shows a Python function or reduction implementation to analyse.
> Exam frequency: **Every exam**.

---

## Q1 — Associativity Counterexample

```python
def abssum(x, y):
    return abs(x + y)
```

Can `abssum` be used as the operator in a parallel reduction tree?

**A)** Yes — it produces the same result regardless of grouping  
**B)** No — it is not associative  
**C)** Yes — it is commutative, which is sufficient  
**D)** No — it is not commutative  

**Answer: B**

- A) Incorrect — grouping changes the result; `(1 op 2) op (-3) = 0` but `1 op (2 op (-3)) = 2`.
- B) Correct — `abssum(abssum(1,2), -3) = abs(0) = 0` ≠ `abssum(1, abssum(2,-3)) = abs(2) = 2`, so not associative.
- C) Incorrect — commutativity alone is not sufficient; associativity is required for tree reductions.
- D) Incorrect — `abs(x+y) = abs(y+x)`, so `abssum` is actually commutative; the real problem is non-associativity.

---

## Q2 — Max Operator

```python
def combine(x, y):
    return max(x, y)
```

Is `combine` valid for use as the operator in a parallel reduction?

**A)** No — `max` is commutative but not associative  
**B)** Yes — `max` is associative and commutative  
**C)** No — `max` requires a total ordering which Python does not guarantee  
**D)** Yes — but only for numeric types  

**Answer: B**

- A) Incorrect — `max(max(a,b), c) = max(a, max(b,c))` always equals the largest value, so it is associative.
- B) Correct — `max` is both associative and commutative, making it valid for any parallel reduction strategy.
- C) Incorrect — Python's built-in `max` works on any totally ordered type, and the claim is too strong; the property holds for numeric and comparable types used in practice.
- D) Incorrect — `max` works correctly for any comparable type (e.g. strings), not just numerics.

---

## Q3 — Matrix Multiplication

```python
import numpy as np

def op(A, B):
    return A @ B   # matrix multiply
```

Is `op` valid as a parallel reduction operator?

**A)** Yes — matrix multiplication is associative  
**B)** No — matrix multiplication is neither associative nor commutative  
**C)** Yes — matrix multiplication is both associative and commutative  
**D)** No — matrix multiplication is associative but NOT commutative, so it cannot be used in unordered reductions  

**Answer: D**

- A) Incorrect — while associativity holds, commutativity does not, which disqualifies unordered reductions.
- B) Incorrect — matrix multiplication is associative; `(A @ B) @ C = A @ (B @ C)` always holds.
- C) Incorrect — `A @ B ≠ B @ A` in general, so it is not commutative.
- D) Correct — associativity holds but commutativity fails, so unordered parallel reductions produce wrong results; ordered tree reduction is impractical for general use.

---

## Q4 — Binary Tree Reduction Rounds

```python
arr = list(range(32))
step = 1
while step < len(arr):
    for i in range(0, len(arr), 2 * step):
        if i + step < len(arr):
            arr[i] += arr[i + step]
    step *= 2
```

How many rounds (iterations of the outer `while` loop) does this tree reduction take to reduce `arr` to a single result?

**A)** 16  
**B)** 5  
**C)** 6  
**D)** 32  

**Answer: B**

- A) Incorrect — 16 would be the number of pairs in the first round, not the number of rounds.
- B) Correct — each round halves the active elements (32→16→8→4→2→1), giving ⌈log₂(32)⌉ = 5 rounds.
- C) Incorrect — 6 rounds would be needed for N=64; ⌈log₂(32)⌉ = 5, not 6.
- D) Incorrect — 32 is the input size, not the number of rounds; that would be a serial loop.

---

## Q5 — Optimal Chunk Count (Flat Reduction)

```python
N = 10000
T = ???   # number of parallel chunks

# Each of T workers sums N//T elements independently (parallel phase).
# Then a single serial sum combines the T partial results (serial phase).
# Total work proportional to: N/T  +  T
```

What value of `T` minimises the total time `N/T + T` for N = 10000?

**A)** T = 50  
**B)** T = 1000  
**C)** T = 100  
**D)** T = 200  

**Answer: C**

- A) Incorrect — T=50 gives total 10000/50 + 50 = 250, which is larger than the minimum of 200 at T=100.
- B) Incorrect — T=1000 gives 10000/1000 + 1000 = 1010, far larger than optimal.
- C) Correct — minimising `N/T + T` by setting the derivative to zero gives T = √N = √10000 = 100, with total cost 200.
- D) Incorrect — T=200 gives 10000/200 + 200 = 250, larger than 200 at the optimal T=100.

---

## Q6 — Custom Additive Operator

```python
def custom_op(x, y):
    return x + y + 1
```

Which of the following correctly characterises `custom_op`?

**A)** Associative but not commutative — valid only for ordered tree reductions  
**B)** Neither associative nor commutative — not valid for parallel reduction  
**C)** Both associative and commutative — valid for parallel reduction  
**D)** Commutative but not associative — not valid for parallel reduction  

**Answer: C**

- A) Incorrect — `custom_op` is also commutative since `x+y+1 = y+x+1`.
- B) Incorrect — `(a op b) op c = a+b+c+2 = a op (b op c)`, so it is associative.
- C) Correct — both `(a op b) op c = a+b+c+2` and `a op (b op c) = a+b+c+2` (associative), and `x op y = y op x` (commutative); identity element is -1.
- D) Incorrect — the operator is associative (same grouped result regardless of order), not merely commutative.

---

## Q7 — String Concatenation

```python
def string_concat(x, y):
    return x + y   # string concatenation
```

A student wants to use `string_concat` in a parallel reduction over a list of strings. Which statement is correct?

**A)** Valid for any parallel reduction — string concatenation is both associative and commutative  
**B)** Not valid — string concatenation is neither associative nor commutative  
**C)** Valid only for ordered (tree) reductions — it is associative but NOT commutative  
**D)** Valid only for unordered reductions — it is commutative but NOT associative  

**Answer: C**

- A) Incorrect — `"ab" + "cd" = "abcd"` ≠ `"cd" + "ab" = "cdab"`, so string concatenation is not commutative.
- B) Incorrect — `("a"+"b")+"c" = "abc" = "a"+("b"+"c")`, so it is associative.
- C) Correct — associativity holds (grouping preserves order) but commutativity fails, so only strictly ordered tree reductions produce correct output.
- D) Incorrect — string concatenation is associative, not merely commutative; unordered reductions would produce non-deterministic results.

---

## Q8 — ThreadPool Row-Sum Analysis

```python
from multiprocessing.pool import ThreadPool
import numpy as np

def row_sum(row):
    return np.sum(row)

matrix = np.random.rand(1000, 1000)
with ThreadPool(8) as pool:
    partial_sums = pool.map(row_sum, matrix)
total = sum(partial_sums)
```

Compared to a naive serial double-loop summing all N² = 1,000,000 elements, what is the approximate speedup of this two-phase parallel reduction?

**A)** Speedup ≈ 1 (no benefit due to Python GIL)  
**B)** Speedup ≈ 1000 (all 1000 rows run fully in parallel)  
**C)** Speedup ≈ 8 (limited by 8 threads in parallel phase)  
**D)** Speedup ≈ 4 (halved because NumPy releases GIL but overhead reduces gain)  

**Answer: C**

- A) Incorrect — `np.sum` releases the GIL, so threads run genuinely in parallel and real speedup is achieved.
- B) Incorrect — only 8 threads are available; 1000 rows are distributed across 8 workers, not run simultaneously.
- C) Correct — 8 threads each handle ~125 rows in parallel; serial final phase (1000 additions) is negligible, giving speedup ≈ 8×.
- D) Incorrect — there is no inherent halving; the GIL is released by NumPy so threads run at full concurrency up to the thread count.

---

## Key Facts Summary

| Property | Required For |
|----------|-------------|
| Associativity | All parallel reductions (tree and flat) |
| Commutativity | Unordered reductions (workers combine in any order) |

| Reduction Type | Depth / Time | Speedup |
|----------------|-------------|---------|
| Binary tree | ⌈log₂(N)⌉ rounds | N / log₂(N) |
| Flat (two-level) | Optimal T = √N | √N / 2 |

**Operators quick reference:**

| Operator | Associative | Commutative | Valid? |
|----------|------------|------------|--------|
| `+` (numbers) | Yes | Yes | Yes |
| `max` / `min` | Yes | Yes | Yes |
| `@` (matrix mul) | Yes | No | Ordered only |
| `+` (strings) | Yes | No | Ordered only |
| `abs(x+y)` | No | Yes | No |
| `x+y+1` | Yes | Yes | Yes |
