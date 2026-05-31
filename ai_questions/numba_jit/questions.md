# Numba JIT — MCQ Practice

> Topics: @jit warmup, nogil, nopython, speedup cases, @cuda.jit.
> Exam frequency: **2024 exam + F25**.

---

## Q1 — First Call Compilation Overhead

> **Week reference:** Week 9

You run the following code to benchmark a Numba-JIT compiled function:

```python
from numba import jit
import numpy as np, time

@jit
def sum_squares(arr):
    total = 0.0
    for x in arr:
        total += x * x
    return total

arr = np.random.rand(1_000_000)
t0 = time.perf_counter()
result = sum_squares(arr)
t1 = time.perf_counter()
print(f"Elapsed: {t1 - t0:.4f}s")
```

What does the measured time primarily reflect?

- A) The execution time of the compiled native code
- B) The time to compile `sum_squares` plus execution time
- C) Python interpreter overhead for the loop
- D) NumPy array allocation time

**Answer: B**

- A) Incorrect — the first call triggers JIT compilation; execution time alone is not what is measured
- B) Correct — the first call to a `@jit` function incurs compilation; the measured time is dominated by that compilation cost
- C) Incorrect — after `@jit` compilation, the Python interpreter does not run the loop body
- D) Incorrect — the array is allocated before timing begins; allocation is not included in the measured interval

---

## Q2 — Warm-Up Best Practice

> **Week reference:** Week 9

A student wants to measure the steady-state runtime of a Numba function `compute(x)`. Which approach gives a reliable execution time?

- A) Call `compute(x)` once and record that time
- B) Call `compute(x)` once to warm up, then time a second call
- C) Use `timeit` without any warm-up calls
- D) Decorate with `@jit(cache=False)` and time the first call

**Answer: B**

- A) Incorrect — the first call includes compilation overhead, making it unreliably slow
- B) Correct — the warm-up call triggers compilation; the second call measures only native execution time
- C) Incorrect — `timeit` by default may include compilation in its first iteration if no warm-up is done
- D) Incorrect — disabling cache does not remove the first-call compilation cost

---

## Q3 — @njit vs @jit

> **Week reference:** Week 9

What is the key behavioural difference between `@jit` and `@njit`?

- A) `@njit` is faster because it uses CUDA; `@jit` uses the CPU
- B) `@jit` releases the GIL automatically; `@njit` does not
- C) `@jit` can fall back to the Python interpreter if compilation fails; `@njit` raises an error instead
- D) `@njit` caches compiled code to disk by default; `@jit` does not

**Answer: C**

- A) Incorrect — `@njit` targets the CPU just like `@jit`; CUDA requires `@cuda.jit`
- B) Incorrect — neither decorator releases the GIL by default; `nogil=True` is required for that
- C) Correct — `@njit` is shorthand for `@jit(nopython=True)`, which disallows Python-object fallback and raises a `TypingError` if Numba cannot compile the function
- D) Incorrect — disk caching requires the explicit `cache=True` argument on either decorator

---

## Q4 — nogil and ThreadPool

> **Week reference:** Week 9

You have a Numba function decorated with `@jit` (no extra arguments) and you call it from multiple threads using `concurrent.futures.ThreadPoolExecutor`. What happens?

- A) The threads run truly in parallel because Numba releases the GIL automatically
- B) The threads run sequentially because the GIL is still held during JIT execution
- C) The code raises a `RuntimeError` because Numba is not thread-safe
- D) The threads run in parallel only on the first call, then sequentially afterwards

**Answer: B**

- A) Incorrect — `@jit` alone does not release the GIL; `nogil=True` is required
- B) Correct — without `nogil=True`, Numba holds the GIL during execution, so threaded calls serialize
- C) Incorrect — Numba does not raise an error for threaded use; it simply does not parallelise
- D) Incorrect — GIL behaviour is consistent across all calls regardless of call count

---

## Q5 — nogil=True Effect

> **Week reference:** Week 9

Which decorator enables a Numba function to execute in parallel across multiple OS threads?

- A) `@jit(parallel=True)`
- B) `@jit(nopython=True)`
- C) `@jit(nogil=True)`
- D) `@njit(cache=True)`

**Answer: C**

- A) Incorrect — `parallel=True` auto-parallelises `prange` loops within a single function call but does not release the GIL for external threads
- B) Incorrect — `nopython=True` enforces strict compilation but does not release the GIL
- C) Correct — `nogil=True` tells Numba to release the Python GIL during execution, allowing true thread-level parallelism when used with a `ThreadPool`
- D) Incorrect — `cache=True` saves compiled artifacts to disk; it has no effect on the GIL

---

## Q6 — JIT Speedup for Python Loops

> **Week reference:** Week 9

A function contains a tight Python `for`-loop performing per-element floating-point arithmetic on one million elements. After applying `@njit`, which speedup over plain Python is most typical?

- A) 2–5×
- B) 10–20×
- C) 100–200×
- D) 10 000–100 000×

**Answer: C**

- A) Incorrect — 2–5× is typical for interpreted overhead elimination in simple cases; Numba achieves much more for tight arithmetic loops
- B) Incorrect — 10–20× underestimates the benefit; Numba generates native LLVM code eliminating Python's per-iteration overhead
- C) Correct — for tight numerical loops, `@njit` typically delivers ~100–200× speedup over pure Python by compiling to optimised native code
- D) Incorrect — this range is unrealistically high; modern Python loops run at ~10–100 ns/iteration, leaving no room for such extreme gains

---

## Q7 — JIT vs Already-Vectorized NumPy

> **Week reference:** Week 9

A colleague applies `@njit` to a function that consists entirely of NumPy array operations (e.g., `np.dot`, `np.sum`, slicing). What is the most likely outcome?

- A) The function runs ~100× faster because Numba further optimises NumPy
- B) Performance is similar to or slightly worse than the plain NumPy version
- C) The function raises a `TypingError` because Numba cannot handle NumPy calls
- D) The GIL is released automatically, giving linear speedup with threads

**Answer: B**

- A) Incorrect — NumPy operations are already compiled C/Fortran; Numba cannot improve upon them and adds dispatch overhead
- B) Correct — Numba does support many NumPy functions but for already-vectorized code it typically matches NumPy at best; for some operations it adds overhead
- C) Incorrect — Numba supports a large subset of NumPy; it will not error on standard array ops
- D) Incorrect — `@njit` alone does not release the GIL

---

## Q8 — When JIT Helps Most

> **Week reference:** Week 9

For which of the following workloads does Numba JIT provide the greatest benefit over plain Python?

- A) Calling `np.linalg.eig` on a large matrix
- B) Reading a CSV file with `pandas.read_csv`
- C) A Python `for`-loop computing a running cumulative product element by element
- D) Vectorised string operations on a Pandas `Series`

**Answer: C**

- A) Incorrect — `np.linalg.eig` is already an optimised LAPACK routine; Numba cannot improve it
- B) Incorrect — file I/O is not a CPU-bound numerical loop; JIT compilation offers no benefit
- C) Correct — element-wise loops that resist simple vectorisation are exactly where Numba excels, converting interpreted per-iteration overhead into native machine code
- D) Incorrect — Pandas string operations involve Python objects that Numba cannot compile in `nopython` mode

---

## Q9 — cache=True

> **Week reference:** Week 9

What is the effect of using `@jit(cache=True)` on a Numba function?

- A) Results are cached in memory so repeated calls with the same inputs skip execution
- B) The compiled native code is saved to disk so the first call on subsequent Python runs avoids recompilation
- C) The function is compiled ahead of time at import, eliminating first-call latency entirely
- D) Numba caches intermediate LLVM IR to speed up future compilations of similar functions

**Answer: B**

- A) Incorrect — `cache=True` caches the compiled *binary*, not the function results; inputs are not memoized
- B) Correct — `cache=True` writes the compiled artifact to `__pycache__`; on the next Python process startup the cached binary is loaded, skipping recompilation
- C) Incorrect — compilation still happens on the first call in a new process if no cached artifact exists yet
- D) Incorrect — Numba saves the final compiled machine code, not intermediate LLVM IR

---

## Q10 — parallel=True and prange

> **Week reference:** Week 9

Which of the following correctly uses Numba's `parallel=True` option to parallelise a loop across CPU cores?

- A) `@jit(parallel=True)` with a regular `range` loop
- B) `@njit(parallel=True)` with `numba.prange` replacing `range`
- C) `@jit(nogil=True)` called from multiple threads, each handling a slice
- D) `@cuda.jit` targeting the GPU with a grid of threads

**Answer: B**

- A) Incorrect — `parallel=True` with a regular `range` does not auto-parallelise; `prange` is required to mark the loop for parallel execution
- B) Correct — `@njit(parallel=True)` combined with `numba.prange` instructs Numba to distribute loop iterations across available CPU cores automatically
- C) Incorrect — `nogil=True` enables external threading; it does not use Numba's internal parallel scheduler
- D) Incorrect — `@cuda.jit` targets GPU threads, not CPU core parallelism via `prange`

---

## Q11 — Type Restrictions in @njit

> **Week reference:** Week 9

A developer tries to pass a Python dictionary `{"key": 1.0}` as an argument to a function decorated with `@njit`. What happens?

- A) Numba infers the dictionary type and compiles successfully
- B) Numba falls back to object mode and runs the function in the Python interpreter
- C) Numba raises a compilation error because arbitrary Python dicts are not supported in nopython mode
- D) The function runs correctly but without any speedup

**Answer: C**

- A) Incorrect — while Numba has limited support for typed dictionaries via `numba.typed.Dict`, a plain Python `dict` with inferred types is not supported in nopython mode
- B) Incorrect — `@njit` (`nopython=True`) never falls back to object mode; that fallback only exists in `@jit` without `nopython=True`
- C) Correct — `@njit` raises a `TypingError` when it encounters types it cannot compile, such as arbitrary Python dictionaries or class instances with unsupported methods
- D) Incorrect — `@njit` does not silently degrade; it raises an error rather than running uncompiled code

---
