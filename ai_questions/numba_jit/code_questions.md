# Numba JIT — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Timing Bug (Warmup)](#q1-timing-bug-warmup)
- [Q2 — GIL and Threading (No `nogil`)](#q2-gil-and-threading-no-nogil)
- [Q3 — GIL Released (`nogil=True`)](#q3-gil-released-nogiltrue)
- [Q4 — Unsupported Type in `nopython` Mode](#q4-unsupported-type-in-nopython-mode)
- [Q5 — Interpreting a Speedup Result](#q5-interpreting-a-speedup-result)
- [Q6 — Applying `@njit` to Already-Vectorized NumPy](#q6-applying-njit-to-already-vectorized-numpy)
- [Q7 — `cache=True` Behaviour](#q7-cachetrue-behaviour)
- [Q8 — CUDA Kernel Called from `@jit` Function](#q8-cuda-kernel-called-from-jit-function)
- [Key Facts Summary](#key-facts-summary)

## Set 3 — Extended Practice

- [Q9 — prange on Inner Loop Only](#q9--prange-on-inner-loop-only)
- [Q10 — Object Mode Fallback (Plain @jit)](#q10--object-mode-fallback-plain-jit)
- [Q11 — numba.typed.List in @njit](#q11--numbatypedlist-in-njit)
- [Q12 — @numba.vectorize Type Signature](#q12--numbavectorize-type-signature)
- [Q13 — prange Reduction: Is it Safe?](#q13--prange-reduction-is-it-safe)
- [Q14 — Two Type Specializations, One Function](#q14--two-type-specializations-one-function)
- [Q15 — cuda.synchronize() Timing Trap](#q15--cudasynchronize-timing-trap)
- [Q16 — Loop-Carried Dependency with prange](#q16--loop-carried-dependency-with-prange)
- [Q17 — cache=True and __pycache__ Location](#q17--cachetrue-and-__pycache__-location)
- [Q18 — @numba.stencil Output](#q18--numbastencil-output)
- [Q19 — dict Literal in @njit](#q19--dict-literal-in-njit)
- [Q20 — Calling Undecorated Python Function from @njit](#q20--calling-undecorated-python-function-from-njit)
- [Q21 — parallel=True with range vs prange](#q21--paralleltrue-with-range-vs-prange)
- [Q22 — print() Inside @njit](#q22--print-inside-njit)
- [Q23 — Default Arguments in @njit](#q23--default-arguments-in-njit)
- [Q24 — fastmath=True: Purpose and Risk](#q24--fastmathtrue-purpose-and-risk)
- [Q25 — In-Place Array Modification Return Value](#q25--in-place-array-modification-return-value)

---

> Format: Each question shows Numba-decorated Python code to analyse.
> Exam frequency: **2024 exam + F25**.

---

## Q1 — Timing Bug (Warmup)

```python
from numba import jit
from time import perf_counter

@jit(nopython=True)
def dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

t = perf_counter()
result = dot(a, b)
print(f"Time: {perf_counter()-t:.4f}s")
```

**What is wrong with this benchmark?**

- A) `perf_counter()` is not accurate enough for Numba functions
- B) The first call to `dot(a, b)` includes JIT compilation time, making the measurement misleading
- C) `nopython=True` disables timing instrumentation
- D) Nothing is wrong; this is a valid benchmark

**Answer: B**

- A) Incorrect — `perf_counter` is a high-resolution wall-clock timer and is accurate enough for any Python function, including Numba-compiled ones; the issue has nothing to do with timer precision.
- B) Correct — The first call to a `@jit`-decorated function triggers JIT compilation. The measured time therefore includes compilation overhead (which can be seconds), not just the actual execution time of the loop. Fix: call `dot(a, b)` once as a warmup before starting the timer, then time the second (already-compiled) call.
- C) Incorrect — `nopython=True` only controls whether Numba falls back to object mode; it has no effect on timing instrumentation or `perf_counter`.
- D) Incorrect — Including compilation time in the first-call measurement is a classic Numba benchmarking mistake; the result is not a valid measure of steady-state runtime.

---

## Q2 — GIL and Threading (No `nogil`)

```python
from numba import jit
from multiprocessing.pool import ThreadPool

@jit(nopython=True)   # NOTE: no nogil=True
def simulate(n):
    x = 0.0
    for i in range(n):
        x += i
    return x

with ThreadPool(4) as pool:
    results = pool.map(simulate, [10**6]*4)
```

**Will the 4 threads run truly in parallel?**

- A) Yes — `nopython=True` automatically releases the GIL
- B) Yes — `ThreadPool` always bypasses the GIL for numerical code
- C) No — the GIL is still held during execution; threads will serialize
- D) No — `ThreadPool` requires `multiprocessing.Pool` for Numba functions

**Answer: C**

- A) Incorrect — `nopython=True` only means Numba compiles to native code without the Python interpreter; it does not automatically release the GIL. You must explicitly pass `nogil=True` to release it.
- B) Incorrect — `ThreadPool` is a Python threading construct and is fully subject to the GIL. It does not bypass the GIL for any code, numerical or otherwise.
- C) Correct — Without `nogil=True`, Numba acquires the GIL before entering the compiled function and holds it throughout. All 4 threads will execute, but only one at a time — they serialize on the GIL, giving no parallel speedup.
- D) Incorrect — `ThreadPool` from `multiprocessing.pool` works fine with Numba functions; the issue is GIL serialization, not an incompatibility between `ThreadPool` and Numba.

---

## Q3 — GIL Released (`nogil=True`)

```python
from numba import jit
from multiprocessing.pool import ThreadPool

@jit(nopython=True, nogil=True)   # nogil=True added
def simulate(n):
    x = 0.0
    for i in range(n):
        x += i
    return x

with ThreadPool(4) as pool:
    results = pool.map(simulate, [10**6]*4)
```

**Now will the 4 threads run truly in parallel?**

- A) No — `nogil=True` only affects GPU execution
- B) No — the GIL still applies regardless of Numba flags
- C) Yes — `nogil=True` releases the GIL; threads can overlap execution on multiple CPU cores
- D) Yes — but only if the input arrays are NumPy arrays, not plain integers

**Answer: C**

- A) Incorrect — `nogil=True` is a CPU-side flag; it has nothing to do with GPU execution. It instructs Numba to release the Python GIL when the compiled CPU function runs.
- B) Incorrect — The GIL applies to all Python threads by default; Numba's `nogil=True` is specifically designed to opt out of it for the duration of a compiled function call.
- C) Correct — `@jit(nopython=True, nogil=True)` compiles to native code and releases the GIL for the duration of each call. `ThreadPool` threads can then execute `simulate` concurrently on separate CPU cores with no serialization, achieving true parallel speedup.
- D) Incorrect — `nogil=True` works with any input type that Numba can type-infer, not only NumPy arrays. Plain Python integers like `10**6` are supported in nopython mode.

---

## Q4 — Unsupported Type in `nopython` Mode

```python
from numba import jit

@jit(nopython=True)
def process(arr):
    result = {}   # Python dict — not supported in nopython mode
    for i in range(len(arr)):
        result[i] = arr[i] ** 2
    return result
```

**Will this function compile and run successfully?**

- A) Yes — Numba silently converts the dict to a typed dictionary
- B) Yes — nopython mode supports all built-in Python types
- C) No — Python dicts are not supported in nopython mode; this raises a `TypingError`
- D) No — `nopython=True` forbids using `**` (power) on array elements

**Answer: C**

- A) Incorrect — Numba does not silently convert plain Python dicts. It attempts to type-infer the dict and fails immediately; there is no automatic upgrade to `numba.typed.Dict`.
- B) Incorrect — `nopython=True` mode supports a limited subset of Python: numeric types, NumPy arrays, tuples, and `numba.typed` containers. Generic Python dicts, lists, and arbitrary objects are not supported.
- C) Correct — Numba cannot infer a static type for a plain Python `dict` in nopython mode and raises a `numba.core.errors.TypingError` at compilation time. The fix is to use `numba.typed.Dict` with explicit key (`int64`) and value (`float64`) type declarations.
- D) Incorrect — The `**` power operator on array elements is fully supported in nopython mode; the error is caused entirely by the unsupported `dict`, not by the power operation.

---

## Q5 — Interpreting a Speedup Result

```python
import numpy as np
from numba import njit
from time import perf_counter

def slow_sum(arr):
    s = 0.0
    for x in arr:
        s += x
    return s

@njit
def fast_sum(arr):
    s = 0.0
    for i in range(len(arr)):
        s += arr[i]
    return s

arr = np.random.rand(10**6)
small_arr = arr[:100]

# Warmup
fast_sum(small_arr)

# Benchmark
t1 = perf_counter(); slow_sum(arr); t1 = perf_counter() - t1
t2 = perf_counter(); fast_sum(arr); t2 = perf_counter() - t2
print(f"Speedup: {t1/t2:.0f}x")
```

**The output is `Speedup: 143x`. Is this result expected?**

- A) No — Numba can only achieve 2-5x speedup over pure Python
- B) No — the benchmark is invalid because warmup was called on a different-sized array
- C) Yes — 100-200x speedup is typical for tight math loops comparing pure Python to @njit
- D) Yes — but only because the warmup call pre-loaded the data into CPU cache

**Answer: C**

- A) Incorrect — 2-5x speedup is typical when comparing NumPy vectorized code against optimised alternatives; comparing a raw Python `for` loop to `@njit` eliminates interpreter overhead entirely, producing far larger gains.
- B) Incorrect — Numba compiles based on argument *types*, not sizes. `small_arr` and `arr` are both `float64` 1D NumPy arrays, so the warmup call on `small_arr` compiles the correct specialization, which is then reused for `arr`. The benchmark is valid.
- C) Correct — Eliminating Python interpreter overhead (object creation, type checking, bytecode dispatch) per loop iteration is the source of the large speedup. 100-200x is entirely normal for tight arithmetic loops, and the warmup strategy is correct.
- D) Incorrect — While warmup data may reside in cache, the full array (8 MB for 10^6 float64 values) far exceeds typical L3 cache; the speedup is dominated by removing interpreter overhead, not cache effects from the warmup call.

---

## Q6 — Applying `@njit` to Already-Vectorized NumPy

```python
import numpy as np
from numba import njit

@njit
def vectorized_sum(arr):
    return np.sum(arr)   # Already a NumPy vectorized operation
```

**Will `@njit` significantly speed up this function compared to calling `np.sum(arr)` directly?**

- A) Yes — @njit always produces faster code than NumPy operations
- B) Yes — @njit enables SIMD auto-vectorization that NumPy cannot use
- C) No — `np.sum` is already implemented in compiled C; @njit adds compilation overhead with negligible benefit
- D) No — @njit cannot call NumPy functions; this raises a TypingError

**Answer: C**

- A) Incorrect — `@njit` excels at replacing slow Python loops, but for code that is already compiled C (like `np.sum`), there is no interpreter overhead left to remove. Wrapping it in Numba adds compilation cost without a meaningful execution benefit.
- B) Incorrect — NumPy already uses SIMD (SSE/AVX) internally in its C implementation of `np.sum`. Numba can also auto-vectorize, but it provides no advantage when the function is already a thin wrapper around a single pre-optimized C call.
- C) Correct — `np.sum` is implemented in highly optimized compiled C/Fortran. Wrapping it with `@njit` incurs JIT compilation overhead on the first call and gives negligible speedup thereafter; the performance bottleneck (the Python interpreter) was never present. `@njit` delivers large gains specifically when eliminating Python-level loops.
- D) Incorrect — Numba supports calling many NumPy functions (including `np.sum`) from nopython mode; this does not raise a TypingError. The NumPy API supported by Numba is documented and `np.sum` is included.

---

## Q7 — `cache=True` Behaviour

```python
from numba import jit

@jit(nopython=True, cache=True)
def heavy_compute(arr):
    result = 0.0
    for i in range(len(arr)):
        for j in range(len(arr)):
            result += arr[i] * arr[j]
    return result
```

**What does `cache=True` do?**

- A) Caches intermediate array results in RAM between function calls
- B) Saves the compiled native binary to disk so that subsequent script runs skip recompilation
- C) Enables memoization — if the same input is passed again, the cached output is returned
- D) Stores the function's bytecode in a Python `.pyc` file

**Answer: B**

- A) Incorrect — Caching intermediate array results in RAM between calls would be memoization (e.g., using `functools.lru_cache`). `cache=True` has nothing to do with storing computation results.
- B) Correct — `cache=True` instructs Numba to serialize the compiled native binary (`.nbi`/`.nbc` files) into a `__pycache__` directory. On the next run of the same script, Numba detects the cached binary, validates it (checking source hash and Python version), and loads it directly — skipping JIT compilation entirely. This removes startup latency for expensive-to-compile functions.
- C) Incorrect — Memoization returns a cached *result* when the same input is provided again. `cache=True` caches the compiled *code*, not the outputs. Calling `heavy_compute` with a different array still executes the full computation.
- D) Incorrect — Python `.pyc` files store Python bytecode, not compiled native machine code. Numba's `cache=True` writes architecture-specific native binaries, not Python bytecode.

---

## Q8 — CUDA Kernel Called from `@jit` Function

```python
from numba import jit, cuda

@cuda.jit
def gpu_kernel(arr):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] *= 2.0

@jit(nopython=True)
def cpu_caller(arr):
    gpu_kernel[32, 256](arr)   # BUG: launching CUDA kernel from nopython context
```

**Will `cpu_caller` successfully launch the GPU kernel?**

- A) Yes — Numba supports mixed CPU/GPU kernel launches from any compiled context
- B) Yes — `@jit(nopython=True)` can call `@cuda.jit` kernels if the array is on the device
- C) No — CUDA kernels can only be launched from host Python code, not from inside a `@jit`/nopython function
- D) No — the grid configuration `[32, 256]` is invalid; it should be `(32, 256)`

**Answer: C**

- A) Incorrect — Numba does not support mixed CPU/GPU kernel launches from within a compiled context. The CUDA runtime API is a host-side interface and cannot be invoked from inside a `nopython` JIT function.
- B) Incorrect — Even if the array is already on the device (a `cuda.device_array`), the kernel launch syntax `gpu_kernel[32, 256](arr)` is a host-side Python operation. It calls into the CUDA runtime, which is inaccessible from within a compiled nopython function.
- C) Correct — CUDA kernel launches invoke the CUDA runtime API, which is host-only. A `@jit(nopython=True)` function cannot call into the host runtime. The correct pattern is to keep `cpu_caller` as regular host Python (no `@jit`) or have it call a Python-level wrapper that performs the launch.
- D) Incorrect — The grid configuration `[32, 256]` using square brackets is the correct Numba CUDA launch syntax (blocks=32, threads=256). Parentheses are not used for the grid spec; this syntax is not the source of the error.

---

---

## Set 3 — Extended Practice

---

## Q9 — prange on Inner Loop Only

```python
from numba import njit, prange

@njit(parallel=True)
def row_max(A):
    result = np.empty(A.shape[0])
    for i in range(A.shape[0]):        # outer: serial range
        m = A[i, 0]
        for j in prange(A.shape[1]):   # inner: prange
            if A[i, j] > m:
                m = A[i, j]
        result[i] = m
    return result
```

**What is the most significant performance problem with this code?**

- A) `prange` is not allowed inside a non-parallel outer loop; this raises a compile error
- B) The inner `prange` launches a thread pool for every outer-loop iteration, adding per-row scheduling overhead that may exceed computation time for small arrays
- C) `prange` on the inner loop automatically parallelises the outer loop as well, causing a race on `result[i]`
- D) There is no problem; parallelising the inner loop is the optimal strategy here

**Answer: B**

- A) Incorrect — Numba does not raise an error for `prange` in an inner loop position. The code compiles and runs; the issue is performance, not correctness.
- B) Correct — the outer `range` iterates serially, invoking the parallel scheduler once per row. For each row, Numba launches threads, distributes `A.shape[1]` iterations, and synchronises. If the matrix has many short rows, the thread-launch overhead per row dominates. The canonical fix is to swap: put `prange` on the outer loop (over rows) so that the thread pool is launched once and each thread processes one or more complete rows.
- C) Incorrect — `prange` only affects the loop it directly annotates. The outer `range` remains serial; there is no automatic promotion of the outer loop and no race condition on `result[i]` (each outer iteration writes to a distinct `result[i]`).
- D) Incorrect — this is a well-known Numba anti-pattern. Parallelising the innermost loop with a serial outer loop leads to high scheduling-to-work ratios and often results in worse performance than a fully serial implementation.

---

## Q10 — Object Mode Fallback (Plain @jit)

```python
from numba import jit

@jit   # No nopython=True
def analyse(data):
    counts = {}          # Plain Python dict
    for x in data:
        key = int(x) % 10
        counts[key] = counts.get(key, 0) + 1
    return counts

result = analyse([1.1, 2.2, 3.3, 11.1, 21.1])
print(type(result))
```

**What does this code print, and what mode does Numba use?**

- A) Raises `TypingError` — `@jit` cannot handle Python dicts
- B) Prints `<class 'dict'>` — Numba falls back to object mode and runs the function using the Python interpreter
- C) Prints `<class 'numba.typed.Dict'>` — Numba automatically converts the dict to a typed dict
- D) Prints `<class 'dict'>` — Numba compiles the dict in nopython mode using type inference

**Answer: B**

- A) Incorrect — `TypingError` is raised by `@njit` (nopython=True). Plain `@jit` without `nopython=True` does not raise a `TypingError`; it silently falls back to object mode when it encounters unsupported types.
- B) Correct — plain `@jit` detects that the Python `dict` and `.get()` method are not supported in nopython mode and falls back to object mode, executing the function body through the Python interpreter. The result is a plain Python `dict`, and `type(result)` prints `<class 'dict'>`. No speedup is gained; the fallback is completely silent by default.
- C) Incorrect — Numba never automatically upgrades plain Python dicts to `numba.typed.Dict`. The programmer must explicitly use `numba.typed.Dict` with declared key and value types. There is no automatic conversion.
- D) Incorrect — Numba cannot type-infer a plain `dict` in nopython mode because it lacks static element-type information. If nopython compilation succeeded, the return type would still be a Python dict (not a Numba-specific type), but compilation would fail before reaching that point.

---

## Q11 — numba.typed.List in @njit

```python
import numba
import numba.typed
import numpy as np
from numba import njit

@njit
def running_sum(arr):
    results = numba.typed.List.empty_list(numba.float64)
    total = 0.0
    for x in arr:
        total += x
        results.append(total)
    return results

arr = np.array([1.0, 2.0, 3.0, 4.0])
out = running_sum(arr)
print(list(out))
```

**What does this code print?**

- A) `[1.0, 2.0, 3.0, 4.0]` — `results.append` overwrites rather than accumulates
- B) Raises `TypingError` because `numba.typed.List` is not supported inside `@njit`
- C) `[1.0, 3.0, 6.0, 10.0]` — the running cumulative sum
- D) `[10.0, 10.0, 10.0, 10.0]` — Numba evaluates the append lazily at function exit

**Answer: C**

- A) Incorrect — `results.append(total)` correctly appends the current value of `total` to the list at each iteration. `total` accumulates (`1`, `3`, `6`, `10`), and each intermediate value is appended, not overwritten.
- B) Incorrect — `numba.typed.List` is fully supported inside `@njit`. It is the recommended replacement for plain Python lists in nopython mode. `numba.typed.List.empty_list(numba.float64)` creates a typed list with `float64` elements, which Numba can compile.
- C) Correct — `total` starts at `0.0`. After each element: `1.0→1.0`, `2.0→3.0`, `3.0→6.0`, `4.0→10.0`. Each step appends the cumulative sum to `results`. The final list contains `[1.0, 3.0, 6.0, 10.0]`. Converting the `numba.typed.List` back to a Python list with `list(out)` gives the standard Python list representation.
- D) Incorrect — Numba does not use lazy evaluation for `typed.List.append`. Each `append` call executes immediately in sequence during the loop, adding the current `total` to the list at that moment. There is no deferred or batched append mechanism.

---

## Q12 — @numba.vectorize Type Signature

```python
import numpy as np
from numba import vectorize

@vectorize(['float64(float64, float64)',
            'float32(float32, float32)'])
def clipped_add(x, y):
    result = x + y
    if result > 1.0:
        return 1.0
    return result

a = np.array([0.3, 0.7, 0.9], dtype=np.float32)
b = np.array([0.4, 0.4, 0.4], dtype=np.float32)
print(clipped_add(a, b))
```

**What does this code print?**

- A) `[0.7 1.0 1.0]` — float32 arrays use the float32 specialisation; values above 1.0 are clipped
- B) `[0.7 1.1 1.3]` — the `if` branch is never taken because Numba ignores conditionals in ufuncs
- C) Raises `TypeError` — `@vectorize` does not support `if` statements inside the kernel
- D) `[0.7 1.0 1.3]` — only the middle element clips because it is exactly 1.1

**Answer: A**

- A) Correct — `a` and `b` are `float32` arrays. Numba dispatches to the `float32(float32, float32)` specialisation. Element-wise: `0.3+0.4=0.7` (below 1.0, kept), `0.7+0.4=1.1` (above 1.0, clipped to 1.0), `0.9+0.4=1.3` (above 1.0, clipped to 1.0). Output: `[0.7 1.0 1.0]` as a float32 array.
- B) Incorrect — Numba fully supports `if` statements inside `@vectorize` kernels. Conditional logic is compiled to native branch instructions. There is no restriction on control flow inside a vectorize kernel.
- C) Incorrect — `@vectorize` supports arbitrary scalar logic including conditionals, loops, and function calls, as long as the types are supported in nopython mode. The restriction is on the *types* used, not the *control flow*.
- D) Incorrect — `0.9+0.4=1.3`, which is above 1.0, so it is also clipped. Both the second and third elements exceed 1.0 and are clipped; the output is `[0.7, 1.0, 1.0]`, not `[0.7, 1.0, 1.3]`.

---

## Q13 — prange Reduction: Is it Safe?

```python
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def parallel_sum(arr):
    total = 0.0
    for i in prange(len(arr)):
        total += arr[i]
    return total

arr = np.ones(1_000_000)
print(parallel_sum(arr))
```

**What does this code print?**

- A) A non-deterministic value close to but not exactly 1000000.0, due to a data race
- B) `1000000.0` — Numba detects the scalar reduction pattern and handles it safely
- C) Raises a `RuntimeError` — `prange` cannot be used with shared scalar accumulators
- D) `0.0` — `total` is a local variable so each thread has its own copy and the main thread's copy is never updated

**Answer: B**

- A) Incorrect — Numba specifically detects the `total += arr[i]` pattern inside `prange` as a scalar reduction. It transforms this into a private accumulator per thread, then merges (sums) the private totals at the end. No data race occurs for this recognised pattern.
- B) Correct — Numba's automatic parallelisation recognises `scalar += array[i]` inside `prange` as a parallel reduction. Each thread accumulates into a private copy of `total`, and the thread-private results are combined (summed) at loop exit. The final result is correct and deterministic (subject to floating-point rounding order, which may produce results very close to but not exactly equal to 1000000.0 due to non-associative summation — but `np.ones` sums exactly to 1000000.0 in float64). The print output is `1000000.0`.
- C) Incorrect — shared scalar accumulators in `prange` are a recognised pattern in Numba. No `RuntimeError` is raised. The reduction is handled automatically.
- D) Incorrect — `total` is not a thread-local copy in the naïve sense. Numba's reduction transformation specifically creates per-thread private copies and merges them. The main-thread view of `total` is updated with the merged result after the parallel loop completes.

---

## Q14 — Two Type Specializations, One Function

```python
import numpy as np
from numba import njit

@njit
def square_sum(arr):
    s = 0.0
    for x in arr:
        s += x * x
    return s

a32 = np.ones(100, dtype=np.float32)
a64 = np.ones(100, dtype=np.float64)

r1 = square_sum(a32)   # Call 1
r2 = square_sum(a64)   # Call 2
r3 = square_sum(a32)   # Call 3

print(r1, r2, r3)
```

**How many times does Numba compile `square_sum` across these three calls?**

- A) Once — Numba generates one generic version for all array types
- B) Three times — Numba compiles on every call
- C) Twice — once for `float32` and once for `float64`; Call 3 reuses the `float32` specialisation
- D) Once — only on Call 1; subsequent calls with different types raise a `TypingError`

**Answer: C**

- A) Incorrect — Numba compiles a separate specialisation per type signature, not one generic version. The `float32` and `float64` paths produce different LLVM IR (different register widths, different SIMD patterns) and are stored separately in the function's dispatch table.
- B) Incorrect — Numba caches compiled specialisations by type signature. Call 3 uses the same `float32` signature as Call 1; the cached specialisation is retrieved without recompilation. Recompilation only happens for previously unseen type combinations.
- C) Correct — Call 1 (`float32`) triggers the first compilation. Call 2 (`float64`) triggers the second compilation. Call 3 (`float32`) hits the cached `float32` specialisation from Call 1 — no compilation occurs. Total compilations: 2. The dispatch table now contains two entries; future calls use whichever matches the argument type.
- D) Incorrect — `@njit` does not raise a `TypingError` when called with a different (but supported) type. It compiles a new specialisation for each new supported type. `TypingError` is raised only when Numba cannot infer a valid type at all (e.g., passing a string or an arbitrary Python object).

---

## Q15 — cuda.synchronize() Timing Trap

```python
from numba import cuda
import numpy as np
from time import perf_counter

@cuda.jit
def add_kernel(x, y, out):
    i = cuda.grid(1)
    if i < x.shape[0]:
        out[i] = x[i] + y[i]

n = 1_000_000
x = np.random.rand(n).astype(np.float32)
y = np.random.rand(n).astype(np.float32)
out = np.empty_like(x)

tpb = 256
bpg = (n + tpb - 1) // tpb

# Warmup
add_kernel[bpg, tpb](x, y, out)
cuda.synchronize()

t0 = perf_counter()
for _ in range(100):
    add_kernel[bpg, tpb](x, y, out)
t1 = perf_counter()
print(f"Avg: {(t1 - t0) / 100 * 1000:.3f} ms")
```

**What is wrong with this benchmark?**

- A) The warmup call should be omitted; it corrupts the compiled kernel state
- B) `cuda.synchronize()` is missing after the timed loop; the measured time reflects only kernel launch time, not actual GPU execution time
- C) `perf_counter()` is not accurate enough for GPU timing; `cuda.event_elapsed_time` must be used
- D) Nothing is wrong; this is a valid GPU benchmark

**Answer: B**

- A) Incorrect — the warmup call is correct and necessary. CUDA kernels are also JIT compiled by Numba on first launch. The warmup call ensures the kernel is compiled before timing begins. Removing it would make the first timed iteration include compilation overhead.
- B) Correct — CUDA kernel launches are **asynchronous**. The line `add_kernel[bpg, tpb](x, y, out)` enqueues the kernel on the GPU and returns immediately to the CPU, before the GPU has finished executing. Without `cuda.synchronize()` after the loop, `t1 = perf_counter()` captures only the time to issue 100 kernel launch commands (microseconds), not the time for the GPU to complete 100 executions. The fix is to add `cuda.synchronize()` immediately before `t1 = perf_counter()`.
- C) Incorrect — `perf_counter()` is accurate enough for CPU-side timing. With `cuda.synchronize()` added, it correctly measures wall-clock time including GPU execution. `cuda.event_elapsed_time` is an alternative that measures GPU-side time directly, but `perf_counter` + `synchronize` is a valid and common approach.
- D) Incorrect — the missing `cuda.synchronize()` before `t1` is a real bug that will produce systematically underestimated (and meaningless) timing results. The measured value would reflect CPU launch overhead (~microseconds) rather than GPU execution time (~milliseconds).

---

## Q16 — Loop-Carried Dependency with prange

```python
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def prefix_sum(arr):
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in prange(1, len(arr)):
        out[i] = out[i - 1] + arr[i]   # depends on previous output
    return out

arr = np.array([1.0, 2.0, 3.0, 4.0])
print(prefix_sum(arr))
```

**What is the most accurate description of what happens?**

- A) Raises `ParallelError` — Numba detects the loop-carried dependency and refuses to compile
- B) Produces the correct prefix sum `[1. 3. 6. 10.]` because Numba serialises prange when it detects dependencies
- C) Produces an incorrect or non-deterministic result because different threads read `out[i-1]` before it has been written by the thread responsible for iteration `i-1`
- D) Produces the correct result because each thread atomically reads and writes `out[i-1]`

**Answer: C**

- A) Incorrect — Numba does not perform dependency analysis to detect loop-carried hazards in `prange`. It trusts the programmer to ensure independence. No error is raised; the code compiles and runs with a data race.
- B) Incorrect — Numba does not silently serialise `prange` when it detects dependencies. `prange` always attempts parallel execution. If the loop has a dependency, that is the programmer's error, and the result is undefined behaviour (a race condition), not a correct serial fallback.
- C) Correct — with `prange`, threads execute iterations in an unspecified order and concurrently. Thread computing `out[5]` reads `out[4]`, which may not yet have been written by the thread computing `out[4]`. The value read is stale (whatever was in `np.empty_like`), so `out[5]` is computed from garbage. The result is non-deterministic and incorrect. Prefix sum has an inherent sequential dependency and cannot be correctly parallelised with `prange` in this naive form.
- D) Incorrect — Numba does not insert atomic operations for `prange` body accesses. Atomic reads and writes would be needed to avoid the race but would not fix the logical dependency anyway: even with atomics, thread 5 could atomically read `out[4]` before thread 4 has written it.

---

## Q17 — cache=True and __pycache__ Location

```python
# File: /home/user/project/compute.py
from numba import njit

@njit(cache=True)
def heavy(arr):
    s = 0.0
    for x in arr:
        s += x * x
    return s
```

**After calling `heavy(arr)` for the first time, where does Numba store the compiled cache files?**

- A) In `/tmp/numba_cache/` — a system-wide cache directory
- B) In `~/.numba/cache/` — a per-user cache directory in the home folder
- C) In `/home/user/project/__pycache__/` — adjacent to the source file
- D) In memory only — `cache=True` does not write to disk

**Answer: C**

- A) Incorrect — Numba does not use a system-wide `/tmp/numba_cache/`. There is no global cache directory shared across users or projects. Each source file's cache lives alongside that source file.
- B) Incorrect — Numba does not use `~/.numba/cache/`. Numba's disk cache is co-located with the source file, following the same convention as Python's `.pyc` bytecode cache.
- C) Correct — Numba stores compiled binaries in a `__pycache__` subdirectory adjacent to the source `.py` file. For `/home/user/project/compute.py`, the cache files are written to `/home/user/project/__pycache__/`. The files have extensions `.nbi` (type information index) and `.nbc` (compiled native binary). This mirrors CPython's `.pyc` caching convention and makes the cache easy to clear (delete `__pycache__`).
- D) Incorrect — `cache=True` explicitly enables disk persistence. Without `cache=True`, compiled specialisations exist only in memory for the lifetime of the Python process. `cache=True` adds the disk-write step to persist across process restarts. Claiming it does not write to disk contradicts its entire purpose.

---

## Q18 — @numba.stencil Output

```python
import numpy as np
import numba

@numba.stencil
def diff1d(arr):
    return arr[1] - arr[-1]   # forward difference: arr[i+1] - arr[i-1]

a = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
result = diff1d(a)
print(result)
```

**What does `result` contain?**

- A) `[3. 5. 7. 9. 0.]` — one-sided forward differences, with the last element zero
- B) `[0. 8. 12. 16. 0.]` — central differences with boundary zeros at both ends
- C) Raises `IndexError` for the boundary elements where a neighbour is out of bounds
- D) `[3. 8. 12. 16. 9.]` — indices wrap around at the boundaries

**Answer: B**

- A) Incorrect — `[3. 5. 7. 9. 0.]` would come from `a[i+1] - a[i]`, a one-sided forward difference. The stencil here uses `arr[-1]` (left neighbour, `a[i-1]`) and `arr[1]` (right neighbour, `a[i+1]`), making it a central difference. The element values in `a` are `[1, 4, 9, 16, 25]`, so differences of adjacent pairs are `3, 5, 7, 9` — this matches a one-sided pattern, not the central-difference pattern with boundary zeros.
- B) Correct — the stencil computes `a[i+1] - a[i-1]` at each index `i`. Boundary elements where a neighbour is out of bounds receive zero by `@stencil`'s default `const_border` policy. Results: `i=0`: left OOB → `0`; `i=1`: `a[2]-a[0]=9-1=8`; `i=2`: `a[3]-a[1]=16-4=12`; `i=3`: `a[4]-a[2]=25-9=16`; `i=4`: right OOB → `0`. Output: `[0. 8. 12. 16. 0.]`.
- C) Incorrect — `@numba.stencil` handles out-of-bounds stencil accesses gracefully by returning the boundary value (zero by default, the `const_border` policy). No `IndexError` is raised. Automatic boundary handling is one of the main conveniences the decorator provides over a manual loop.
- D) Incorrect — `@stencil` does not use periodic/wrap-around boundary conditions by default. Out-of-bounds accesses use the `const_border` policy (zero-fill). To use wrap-around, you would need to implement the boundary logic manually inside the kernel.

---

## Q19 — dict Literal in @njit

```python
from numba import njit

@njit
def histogram(data):
    counts = {}
    for x in data:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts
```

**What happens when this is called?**

- A) Compiles and runs correctly — plain Python dicts work in nopython mode
- B) TypingError at compilation — plain `{}` dict literals are not supported in nopython mode; use `numba.typed.Dict`
- C) Falls back to object mode silently and runs correctly
- D) Compiles fine but raises `KeyError` at runtime on the first new key

**Answer: B**

- A) Incorrect — plain Python dicts (`{}`) are not supported in `@njit` (nopython mode). Numba requires `numba.typed.Dict` for dict-like containers in compiled functions.
- B) Correct — Numba raises a `TypingError` during JIT compilation because it cannot infer a type for a plain Python dict literal. Fix: use `numba.typed.Dict.empty(key_type, value_type)` instead.
- C) Incorrect — `@njit` (shorthand for `@jit(nopython=True)`) never falls back to object mode; it errors immediately if compilation fails. Plain `@jit` without `nopython=True` would silently fall back, but not `@njit`.
- D) Incorrect — the error is raised at compile time (JIT compilation on first call), not at runtime. Numba type-checks the entire function body before executing any of it.

---

## Q20 — Calling Undecorated Python Function from @njit

```python
import numpy as np
from numba import njit

def square(x):
    return x * x   # plain Python, no decorator

@njit
def sum_of_squares(arr):
    total = 0.0
    for x in arr:
        total += square(x)
    return total

a = np.arange(4, dtype=np.float64)   # [0., 1., 2., 3.]
print(sum_of_squares(a))
```

**What happens?**

- A) TypingError — `@njit` cannot call functions that are not decorated with `@jit` or `@njit`
- B) Runs and prints `14.0` — Numba traces into and compiles `square` as part of the same compilation pass
- C) Falls back to object mode for the `square(x)` call, slowing down the whole function
- D) Prints `0.0` — undecorated calls return `None` in nopython mode, and `None` is treated as 0

**Answer: B**

- A) Incorrect — Numba does not require called functions to be pre-decorated. When compiling `sum_of_squares`, it traces into `square`, infers its types from the call site (`x` is float64), and compiles it inline. No decorator on `square` is needed.
- B) Correct — Numba compiles called Python functions on demand as part of the same JIT pass. `square(x)` where `x` is float64 compiles to `x * x`. `0² + 1² + 2² + 3² = 0 + 1 + 4 + 9 = 14.0`.
- C) Incorrect — `@njit` never silently drops to object mode; it either compiles the entire call graph in nopython mode or raises a TypingError. There is no partial object-mode fallback.
- D) Incorrect — Numba does not substitute `None` for undecorated calls. The function is compiled and returns the correct numeric result.

---

## Q21 — parallel=True with range vs prange

```python
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def sum_range(arr):
    total = 0.0
    for i in range(len(arr)):      # regular range
        total += arr[i]
    return total

@njit(parallel=True)
def sum_prange(arr):
    total = 0.0
    for i in prange(len(arr)):     # prange
        total += arr[i]
    return total
```

**Which function benefits from CPU parallelization, and why?**

- A) Both — `parallel=True` parallelizes all loops regardless of whether `range` or `prange` is used
- B) Only `sum_prange` — regular `range` loops are not parallelized; only `prange` explicitly requests parallel execution
- C) Neither — scalar reductions cannot be parallelized in Numba at all
- D) Only `sum_range` — `prange` is reserved for GPU kernels; on CPU it has no effect

**Answer: B**

- A) Incorrect — `parallel=True` enables two distinct mechanisms: (1) `prange` loops for explicit parallelism, and (2) automatic parallelization of NumPy-style array expressions. A regular `range` loop with a scalar accumulator is not recognized as a parallel pattern by the auto-parallelizer.
- B) Correct — `prange` is Numba's signal to the auto-parallelizer that loop iterations are independent. Numba detects `total += arr[i]` inside `prange` as a scalar reduction, creates private accumulators per thread, and merges them. With `range`, the same loop is compiled serially.
- C) Incorrect — `prange` reductions on scalars are explicitly supported. Numba performs automatic parallel scalar reductions for `+=`, `*=`, etc. when used with `prange`.
- D) Incorrect — `prange` is for CPU parallelism via Numba's threading backend. GPU kernels use `@cuda.jit` with `cuda.grid()`; `prange` is entirely unrelated to GPU execution.

---

## Q22 — print() Inside @njit

```python
from numba import njit
import numpy as np

@njit
def verbose_max(arr):
    best = arr[0]
    for x in arr:
        if x > best:
            best = x
            print("New best:", best)
    return best

a = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
result = verbose_max(a)
```

**Does `print()` work inside `@njit`, and what does it output?**

- A) TypingError — `print()` is a Python built-in and is not supported in nopython mode
- B) Compiles and prints `New best: 3.0` then `New best: 5.0` — Numba supports `print()` with numeric scalars and string literals in nopython mode
- C) The `print()` calls compile but produce no output — stdout is suppressed inside nopython kernels
- D) Only works if the argument is a string; `print("New best:", best)` raises TypingError because `best` is a float

**Answer: B**

- A) Incorrect — Numba supports `print()` in nopython mode for numeric scalars (int, float) and string literals. It maps to a low-level `printf`-style call rather than the Python built-in.
- B) Correct — The kernel compiles. Scanning `[1.0, 3.0, 2.0, 5.0, 4.0]`: 3.0 > 1.0 → prints `New best: 3.0`; 5.0 > 3.0 → prints `New best: 5.0`. Output is two lines.
- C) Incorrect — output is not suppressed. Numba's `print()` writes directly to stdout via printf, so output appears immediately (useful for debugging inside JIT-compiled kernels).
- D) Incorrect — mixing a string literal and a numeric argument in `print()` IS supported in nopython mode. Numba handles the common `print("label:", value)` pattern.

---

## Q23 — Default Arguments in @njit

```python
from numba import njit

@njit
def clip(x, lo=0.0, hi=1.0):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

print(clip(2.5))
print(clip(-1.0))
print(clip(0.5))
```

**What is printed?**

- A) TypingError — `@njit` does not support default argument values
- B) `1.0`, `0.0`, `0.5` — Numba supports default float arguments in nopython mode
- C) `2.5`, `-1.0`, `0.5` — default values are ignored; all inputs pass through unchanged
- D) `1.0`, `0.0`, `1.0` — `clip(0.5)` returns `hi` because 0.5 is the upper half of [0, 1]

**Answer: B**

- A) Incorrect — Numba supports default argument values for `@njit` functions. They behave identically to regular Python defaults.
- B) Correct — `clip(2.5)`: 2.5 > 1.0 → return 1.0. `clip(-1.0)`: -1.0 < 0.0 → return 0.0. `clip(0.5)`: 0.5 is not < 0.0 and not > 1.0 → return 0.5.
- C) Incorrect — the `if` branches are compiled and execute correctly. Inputs are not passed through unchanged; values outside [lo, hi] are clipped to the boundary.
- D) Incorrect — `clip(0.5)` hits neither branch (0.5 is within [0.0, 1.0]), so it falls through to `return x`, returning 0.5 exactly.

---

## Q24 — fastmath=True: Purpose and Risk

```python
import numpy as np
from numba import njit

@njit(fastmath=False)
def safe_sum(arr):
    total = 0.0
    for x in arr:
        total += x
    return total

@njit(fastmath=True)
def fast_sum(arr):
    total = 0.0
    for x in arr:
        total += x
    return total
```

**Which statement about `fastmath=True` is correct?**

- A) `fast_sum` is guaranteed to return the same numeric result as `safe_sum` — `fastmath` only affects throughput, never accuracy
- B) `fastmath=True` allows the compiler to treat floating-point addition as associative and reorder operations, potentially producing a numerically different result while enabling vectorization
- C) `fastmath=True` enables multi-core parallelism for the loop — it is equivalent to adding `parallel=True`
- D) `fastmath=True` is only meaningful for GPU kernels; on CPU it has no effect

**Answer: B**

- A) Incorrect — `fastmath=True` explicitly relaxes IEEE 754 guarantees. The compiler may reorder additions (e.g., sum in pairs, use FMA), which changes rounding behaviour and can produce different results for cancellation-heavy inputs.
- B) Correct — `fastmath=True` passes LLVM's fast-math flags: assume no NaN/Inf, treat FP arithmetic as associative, allow contraction (FMA), etc. This enables SIMD vectorization of the reduction loop. The tradeoff is that results may differ from strict left-to-right IEEE 754 evaluation, especially with large cancellations.
- C) Incorrect — `fastmath` and `parallel` are orthogonal flags. `fastmath=True` does not spawn threads; it only changes code generation assumptions for the existing single-threaded (or prange-parallel) loop.
- D) Incorrect — `fastmath=True` targets the CPU LLVM backend and is fully effective on CPU. It is not GPU-specific; CUDA kernels have their own fast-math settings.

---

## Q25 — In-Place Array Modification Return Value

```python
import numpy as np
from numba import njit

@njit
def fill_zeros(arr):
    for i in range(len(arr)):
        arr[i] = 0.0
    # no return statement

a = np.ones(4, dtype=np.float64)
r = fill_zeros(a)
print(r)
print(a)
```

**What is printed?**

- A) `[0. 0. 0. 0.]` then `[0. 0. 0. 0.]` — Numba automatically returns the modified array
- B) `None` then `[0. 0. 0. 0.]` — the array is modified in-place; the function returns `None` with no explicit return
- C) `None` then `[1. 1. 1. 1.]` — `@njit` operates on a copy; the original `a` is unchanged
- D) TypingError — `@njit` functions must have an explicit return statement

**Answer: B**

- A) Incorrect — a function with no `return` statement returns `None` in Python, and `@njit` preserves this. Numba does not automatically return the first array argument.
- B) Correct — NumPy arrays are passed by reference to `@njit` functions; in-place writes to `arr[i]` modify the original array `a`. The function has no `return`, so Python returns `None`. `print(r)` prints `None`; `print(a)` prints `[0. 0. 0. 0.]`.
- C) Incorrect — `@njit` does NOT copy input arrays. Arrays are passed as pointers to the underlying buffer, so all writes are immediately visible in the caller's NumPy array.
- D) Incorrect — an explicit `return` is not required. `@njit` functions can return `None` implicitly, just like regular Python functions.

---

## Key Facts Summary

| Concept | Rule |
|---------|------|
| Warmup | ALWAYS call `@jit` function once before timing to exclude compilation |
| `nogil=True` | Releases the GIL — enables real thread parallelism with `ThreadPool` |
| `@njit` | Alias for `@jit(nopython=True)` — errors immediately if Numba cannot compile |
| `nopython=True` dicts | Not supported — use `numba.typed.Dict` instead |
| JIT on vectorized NumPy | Minimal benefit — `np.sum`, `np.dot`, etc. are already compiled C |
| `cache=True` | Saves compiled binary to `__pycache__` — faster startup on subsequent runs |
| CUDA kernel launches | Host-only — cannot launch from inside a `@jit`/nopython function |
| Typical speedup | 100-200x for tight Python loops vs `@njit` |
| `prange` placement | Annotate the outermost loop — inner-only `prange` adds overhead |
| `@jit` object mode | Plain `@jit` silently falls back to interpreter — use `@njit` to surface errors |
| `numba.typed.List` | Required for mutable sequences in `@njit` — plain `list` raises `TypingError` |
| `@vectorize` | Creates a NumPy ufunc from a scalar function — declare type signatures explicitly |
| Type specialisation | Each unique argument type triggers one compilation; subsequent same-type calls reuse cache |
| `cuda.synchronize()` | MUST call before stopping the timer — CUDA launches are asynchronous |
| Loop-carried dependency | Never use `prange` when iteration i+1 depends on iteration i — data race, wrong result |
| `cache=True` invalidation | Source code change or Numba version upgrade invalidates the disk cache |
