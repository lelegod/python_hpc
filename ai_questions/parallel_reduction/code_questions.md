# Parallel Reduction — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Associativity Counterexample](#q1-associativity-counterexample)
- [Q2 — Max Operator](#q2-max-operator)
- [Q3 — Matrix Multiplication](#q3-matrix-multiplication)
- [Q4 — Binary Tree Reduction Rounds](#q4-binary-tree-reduction-rounds)
- [Q5 — Optimal Chunk Count (Flat Reduction)](#q5-optimal-chunk-count-flat-reduction)
- [Q6 — Custom Additive Operator](#q6-custom-additive-operator)
- [Q7 — String Concatenation](#q7-string-concatenation)
- [Q8 — ThreadPool Row-Sum Analysis](#q8-threadpool-row-sum-analysis)
- [Key Facts Summary](#key-facts-summary)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q9 — Pool.map + functools.reduce Pattern](#q9-poolmap-functoolsreduce-pattern)
- [Q10 — Subtraction in a Tree Reduction](#q10-subtraction-in-a-tree-reduction)
- [Q11 — CUDA Shared Memory Reduction Kernel](#q11-cuda-shared-memory-reduction-kernel)
- [Q12 — Identifying the Serial Combine Step](#q12-identifying-the-serial-combine-step)
- [Q13 — XOR Reduction Correctness](#q13-xor-reduction-correctness)
- [Q14 — Warp Shuffle Reduction](#q14-warp-shuffle-reduction)
- [Q15 — Parallel Average: Wrong Approach](#q15-parallel-average-wrong-approach)
- [Q16 — Missing syncthreads Bug](#q16-missing-syncthreads-bug)
- [Q17 — Counting Reduction Steps for N=256](#q17-counting-reduction-steps-for-n256)
- [Q18 — Custom Reduction: Weighted Sum](#q18-custom-reduction-weighted-sum)

---

> Format: Each question shows a Python function or reduction implementation to analyse.
> Exam frequency: **Every exam**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--associativity-counterexample)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

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

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets commutativity/associativity requirements, tree reduction depth, warp-level reduction, and Pool.map + functools.reduce patterns

---

## Q9 — Pool.map + functools.reduce Pattern

> **Week reference:** Week 5

```python
from multiprocessing import Pool
from functools import reduce

def partial_sum(chunk):
    return sum(chunk)

data = list(range(100))
chunks = [data[i:i+25] for i in range(0, 100, 25)]

with Pool(4) as p:
    partials = p.map(partial_sum, chunks)

final = reduce(lambda a, b: a + b, partials)
```

What does `partials` contain immediately after `p.map` returns?

- A) A single integer: 4950 (the sum of 0 through 99)
- B) A list of four integers, one partial sum per chunk
- C) A Pool future object that resolves lazily
- D) A list of four lists, each being the original chunk unchanged

**Answer: B**

`Pool.map` applies `partial_sum` to each of the four chunks in parallel and returns a list of return values in input order. Each chunk has 25 elements, so `partials = [sum(0..24), sum(25..49), sum(50..74), sum(75..99)] = [300, 925, 1550, 2175]`. The serial `reduce` step then combines these into 4950. Answer A is the final result, not the intermediate `partials`. Answer C is wrong: `Pool.map` blocks until all workers finish and returns a concrete list.

---

## Q10 — Subtraction in a Tree Reduction

> **Week reference:** Week 6

```python
import functools

arr = [10, 3, 2, 1]
result = functools.reduce(lambda a, b: a - b, arr)
```

If this is naively "parallelised" by splitting into two chunks `[10, 3]` and `[2, 1]`, computing each chunk's reduce serially, then subtracting the results, what answer is produced, and is it correct?

- A) 4; correct — subtraction is associative so parallelisation is valid
- B) 4; incorrect — serial gives 4 but the parallel grouping also gives 4 by coincidence; this does not prove associativity
- C) Serial gives 4; parallel gives 6 — the results differ, demonstrating that subtraction is not associative
- D) Serial gives 4; parallel gives 8 — the results differ, demonstrating that subtraction is not associative

**Answer: C**

Serial: `((10 − 3) − 2) − 1 = 7 − 2 − 1 = 4`. Parallel chunk 1: `10 − 3 = 7`. Parallel chunk 2: `2 − 1 = 1`. Combining: `7 − 1 = 6`. Serial = 4, parallel = 6: they disagree, confirming that subtraction is not associative and cannot be safely parallelised with arbitrary groupings.

---

## Q11 — CUDA Shared Memory Reduction Kernel

> **Week reference:** Week 9

```python
from numba import cuda
import numpy as np

BLOCK_SIZE = 16

@cuda.jit
def reduce_kernel(arr, result):
    shared = cuda.shared.array(shape=BLOCK_SIZE, dtype=np.float64)
    tid = cuda.threadIdx.x
    shared[tid] = arr[cuda.grid(1)]
    cuda.syncthreads()
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        cuda.atomic.add(result, 0, shared[0])
```

How many iterations does the `while` loop execute per block (assuming BLOCK_SIZE = 16)?

- A) 16 iterations
- B) 8 iterations
- C) 4 iterations
- D) 5 iterations

**Answer: C**

Starting at stride = 16 // 2 = 8, the loop halves stride each iteration: 8 → 4 → 2 → 1 → 0 (exits). That is 4 iterations: stride = 8, 4, 2, 1. This equals ceil(log₂(16)) = 4. Answer A (16) is the block size, not the loop count. Answer B (8) is the first stride value, not the iteration count. Answer D (5) would apply to BLOCK_SIZE = 32.

---

## Q12 — Identifying the Serial Combine Step

> **Week reference:** Week 5

```python
from multiprocessing import Pool
from functools import reduce
import math

def partial_product(chunk):
    result = 1
    for x in chunk:
        result *= x
    return result

values = list(range(1, 13))   # [1, 2, ..., 12]
chunks = [values[i:i+3] for i in range(0, 12, 3)]  # 4 chunks of 3

with Pool(4) as p:
    partials = p.map(partial_product, chunks)

final = reduce(lambda a, b: a * b, partials)
```

Which line is the serial combine step, and what is the value of `final`?

- A) `p.map(...)` is the serial step; `final` = 479001600
- B) `reduce(...)` is the serial step; `final` = 479001600
- C) Both `p.map` and `reduce` are parallel; `final` = 479001600
- D) `reduce(...)` is the serial step; `final` = 40320

**Answer: B**

`p.map` is the parallel phase — four workers each compute a partial product simultaneously. `reduce` is the serial combine phase — it folds the four partial products left-to-right in the calling process: no parallelism. `final = 12! = 479001600` (the product of 1 through 12). Multiplication is associative and commutative, so splitting into chunks gives the correct result regardless of grouping.

---

## Q13 — XOR Reduction Correctness

> **Week reference:** Week 6

```python
import functools, operator
from multiprocessing import Pool

data = [3, 5, 6, 3, 5]   # 3 and 5 appear twice, 6 once

def chunk_xor(chunk):
    return functools.reduce(operator.xor, chunk)

chunks = [[3, 5], [6, 3, 5]]
with Pool(2) as p:
    partials = p.map(chunk_xor, chunks)

final = partials[0] ^ partials[1]
```

What is `final`, and why is the parallel result correct?

- A) final = 0; XOR is commutative but not associative, so the result is coincidentally correct
- B) final = 6; XOR is both associative and commutative, so the parallel result equals the serial result
- C) final = 3; XOR is not commutative so the parallel result is incorrect
- D) final = 6; but the result would differ for a different chunk split, because XOR is not associative

**Answer: B**

XOR is both associative and commutative, so any grouping and any merge order gives the same result. Serial: `3^5^6^3^5 = 6` (3 and 5 each appear twice and cancel to 0; 6 remains). Parallel chunk 1: `3^5 = 6`. Chunk 2: `6^3^5 = 0`. Final: `6^0 = 6`. Correct. Answer D is wrong: because XOR is associative, a different chunk split also gives 6.

---

## Q14 — Warp Shuffle Reduction

> **Week reference:** Week 9

```python
from numba import cuda
import numpy as np

@cuda.jit
def warp_reduce_kernel(arr, out):
    tid = cuda.threadIdx.x          # 0..31, one warp
    val = arr[tid]
    # Warp-level shuffle reduction
    offset = 16
    while offset > 0:
        val += cuda.shfl_down_sync(0xFFFFFFFF, val, offset)
        offset //= 2
    if tid == 0:
        out[0] = val
```

How many iterations does the shuffle loop execute, and which thread holds the final sum after the loop?

- A) 4 iterations; thread 16 holds the final sum
- B) 5 iterations; thread 0 holds the final sum
- C) 5 iterations; all 32 threads hold the final sum
- D) 32 iterations; thread 0 holds the final sum

**Answer: B**

Starting at offset = 16, the loop halves each iteration: 16 → 8 → 4 → 2 → 1 → 0 (exits). That is 5 iterations = ceil(log₂(32)). After the loop, thread 0 holds the sum of all 32 values because it accumulates contributions from threads at distances 16, 8, 4, 2, and 1 via the shuffle-down chain. Other threads accumulate partial sums only. Answer C is wrong: `shfl_down_sync` shifts values toward lower thread IDs; only thread 0 accumulates all contributions.

---

## Q15 — Parallel Average: Wrong Approach

> **Week reference:** Week 5

```python
from multiprocessing import Pool

def avg(a, b):
    return (a + b) / 2.0

def chunk_avg(chunk):
    return sum(chunk) / len(chunk)

data = [2.0, 4.0, 6.0, 8.0]
chunks = [[2.0, 4.0], [6.0, 8.0]]

with Pool(2) as p:
    partials = p.map(chunk_avg, chunks)

final_wrong = (partials[0] + partials[1]) / 2.0
final_correct = sum(data) / len(data)
```

What are `final_wrong` and `final_correct`, and why do they differ?

- A) Both equal 5.0; averaging partial averages is equivalent to averaging the full array when chunks are equal size
- B) `final_wrong` = 5.0, `final_correct` = 5.0; they are equal here but would differ for unequal-size chunks
- C) `final_wrong` = 5.0, `final_correct` = 5.0; they always agree because avg is commutative
- D) `final_wrong` = 5.5, `final_correct` = 5.0; averaging partial averages is wrong in general

**Answer: B**

Here the chunks happen to be equal size (2 elements each), so `final_wrong = (3.0 + 7.0) / 2 = 5.0 = final_correct`. However, this equality is coincidental: for unequal-size chunks (e.g. `[2.0, 4.0, 6.0]` and `[8.0]`), `final_wrong = (4.0 + 8.0) / 2 = 6.0` but `final_correct = 5.0`. The correct parallel approach is to reduce sum and count separately, then divide once: `total_sum / total_count`.

---

## Q16 — Missing syncthreads Bug

> **Week reference:** Week 9

```python
@cuda.jit
def buggy_reduce(arr, result):
    shared = cuda.shared.array(shape=32, dtype=np.float64)
    tid = cuda.threadIdx.x
    shared[tid] = arr[tid]
    # BUG: no cuda.syncthreads() here
    stride = 16
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        result[0] = shared[0]
```

The `cuda.syncthreads()` is missing after loading `arr` into `shared`. What is the likely consequence?

- A) No consequence — the GPU guarantees shared memory is visible before any thread proceeds
- B) Thread 0 may read uninitialized or stale shared memory values in the first stride step
- C) The kernel will raise a CUDA exception and terminate
- D) Only threads with tid < 16 are affected; threads with tid >= 16 compute correctly

**Answer: B**

Without a barrier after loading `shared[tid] = arr[tid]`, the first stride-16 iteration may execute before all threads have finished writing to shared memory. Thread 0 adds `shared[0] += shared[16]`, but `shared[16]` may not yet contain `arr[16]` — another thread may still be executing the load. This is a shared-memory data race yielding non-deterministic, typically incorrect results. CUDA does not raise an exception for this (C is wrong); it silently allows the race. The bug affects any thread that reads a location written by another thread, not just tid < 16 (D is wrong).

---

## Q17 — Counting Reduction Steps for N=256

> **Week reference:** Week 6

A parallel binary tree reduction processes N = 256 elements. Each round, all pairs combine simultaneously. How many rounds complete the reduction, and what is the parallel speedup over a serial reduction (which takes N−1 = 255 operations)?

- A) 8 rounds; speedup = 255/8 ≈ 31.9
- B) 128 rounds; speedup = 2
- C) 8 rounds; speedup = 8
- D) 16 rounds; speedup = 255/16 ≈ 15.9

**Answer: A**

N = 256 = 2^8, so tree depth = ceil(log₂(256)) = 8 rounds. Serial cost is N−1 = 255 operations. Parallel cost is 8 rounds. Speedup = 255/8 ≈ 31.9. The speedup is not simply equal to the number of rounds (C is wrong): the rounds are cheap because many operations run simultaneously, and the speedup ratio compares total serial operations to parallel rounds.

---

## Q18 — Custom Reduction: Weighted Sum

> **Week reference:** Week 5

```python
from multiprocessing import Pool
from functools import reduce

weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
values  = [10,  20,  30,  40,  50,  60,  70,  80]
pairs   = list(zip(values, weights))

def weighted_partial(chunk):
    return sum(v * w for v, w in chunk)

chunks = [pairs[i:i+2] for i in range(0, 8, 2)]

with Pool(4) as p:
    partials = p.map(weighted_partial, chunks)

total = reduce(lambda a, b: a + b, partials)
```

Is this parallel weighted sum correct, and what is `total`?

- A) Incorrect — weighted sum is not associative so parallelising it is invalid
- B) Correct — `total` = 180.0
- C) Correct — `total` = 360.0 (each weight applied twice due to chunking)
- D) Incorrect — `Pool.map` cannot handle tuples in chunks

**Answer: B**

`total = sum(v * w for v, w in pairs) = 0.5*(10+20+30+40+50+60+70+80) = 0.5*360 = 180.0`. The parallel approach is valid because the overall reduction is a sum of independent terms — addition is associative and commutative. Chunking into partial sums and then summing the partials gives the same result as the serial sum. Answer A is wrong: addition of weighted terms is associative and commutative, so parallelisation is perfectly valid. Answer C (360.0) would require weights of 1.0 instead of 0.5. Answer D is wrong: `Pool.map` handles any picklable iterable including lists of tuples.

---
