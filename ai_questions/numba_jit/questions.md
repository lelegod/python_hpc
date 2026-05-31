# Numba JIT — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — First Call Compilation Overhead](#q1-first-call-compilation-overhead)
- [Q2 — Warm-Up Best Practice](#q2-warm-up-best-practice)
- [Q3 — @njit vs @jit](#q3-njit-vs-jit)
- [Q4 — nogil and ThreadPool](#q4-nogil-and-threadpool)
- [Q5 — nogil=True Effect](#q5-nogiltrue-effect)
- [Q6 — JIT Speedup for Python Loops](#q6-jit-speedup-for-python-loops)
- [Q7 — JIT vs Already-Vectorized NumPy](#q7-jit-vs-already-vectorized-numpy)
- [Q8 — When JIT Helps Most](#q8-when-jit-helps-most)
- [Q9 — cache=True](#q9-cachetrue)
- [Q10 — parallel=True and prange](#q10-paralleltrue-and-prange)
- [Q11 — Type Restrictions in @njit](#q11-type-restrictions-in-njit)

---

> Topics: @jit warmup, nogil, nopython, speedup cases, @cuda.jit.
> Exam frequency: **2024 exam + F25**.

---

## Q1 — First Call Compilation Overhead

> **Week reference:** Week 9

**Mental Model:** `@jit` is lazy — compilation is deferred to the first call, not import time. The first timing measurement is almost always dominated by LLVM compilation, not computation. Trap: assuming you're measuring execution speed.

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

- A) Incorrect — the first call triggers JIT compilation; the compiled native execution time is only a small fraction of what is measured. On a million-element array, the execution alone might take ~1 ms but compilation can add hundreds of milliseconds.
- B) Correct — the first call to a `@jit` function incurs LLVM compilation of the function for the specific argument types. This compilation cost dominates the measured interval; subsequent calls (after warm-up) are much faster.
- C) Incorrect — after `@jit` compilation, the Python interpreter does not run the loop body; native machine code runs instead. The loop overhead is eliminated, not measured.
- D) Incorrect — the array `arr` is allocated on the line before `t0 = time.perf_counter()`, so array allocation is completely outside the timed interval.

---

## Q2 — Warm-Up Best Practice

> **Week reference:** Week 9

**Mental Model:** Think of JIT as "compile on first use." To measure execution speed, you must separate the one-time compilation cost from the repeated execution cost. Call once to compile, then time the second call.

A student wants to measure the steady-state runtime of a Numba function `compute(x)`. Which approach gives a reliable execution time?

- A) Call `compute(x)` once and record that time
- B) Call `compute(x)` once to warm up, then time a second call
- C) Use `timeit` without any warm-up calls
- D) Decorate with `@jit(cache=False)` and time the first call

**Answer: B**

- A) Incorrect — the first call includes compilation overhead that can be 10–1000× longer than the actual execution. For a function that takes 1 ms to run, compilation might take 500 ms, making the measurement useless as a performance indicator.
- B) Correct — the warm-up call triggers compilation and caches the compiled artifact in memory; the second call only pays the native execution cost. For any benchmarking of Numba functions, one warm-up call is the minimum required.
- C) Incorrect — `timeit` by default runs many iterations but starts from the first call. Without a prior warm-up, its first iteration includes compilation, skewing the mean upward, especially if the number of repetitions is small.
- D) Incorrect — `cache=False` (which is already the default) does not remove first-call compilation cost; it only affects whether compiled artifacts are written to disk. Setting it explicitly changes nothing about the warm-up requirement.

---

## Q3 — @njit vs @jit

> **Week reference:** Week 9

**Mental Model:** `@njit` = `@jit(nopython=True)`. The "nopython" name tells you the key: it refuses to run any Python-object-mode code. This is stricter but gives you a clear compile-time error rather than a silent performance regression from fallback.

What is the key behavioural difference between `@jit` and `@njit`?

- A) `@njit` is faster because it uses CUDA; `@jit` uses the CPU
- B) `@jit` releases the GIL automatically; `@njit` does not
- C) `@jit` can fall back to the Python interpreter if compilation fails; `@njit` raises an error instead
- D) `@njit` caches compiled code to disk by default; `@jit` does not

**Answer: C**

- A) Incorrect — `@njit` targets the CPU just like `@jit`; both compile to native CPU code via LLVM. GPU execution requires the entirely separate `@cuda.jit` decorator.
- B) Incorrect — neither decorator releases the GIL by default. Both hold the GIL during execution unless the explicit `nogil=True` argument is added.
- C) Correct — `@njit` is shorthand for `@jit(nopython=True)`, which disallows the "object mode" fallback. If Numba cannot compile a type or operation, `@jit` silently falls back to the Python interpreter (slower, but no error), while `@njit` raises a `TypingError` immediately, making the compilation failure visible.
- D) Incorrect — disk caching requires the explicit `cache=True` argument on either decorator; neither enables it by default. `@njit` without `cache=True` recompiles on every new Python process start.

---

## Q4 — nogil and ThreadPool

> **Week reference:** Week 9

**Mental Model:** Python's GIL allows only one thread to execute Python bytecode at a time. Numba's `@jit` still holds the GIL by default. Without `nogil=True`, a ThreadPool calling a Numba function serialises — threads queue up rather than run in parallel.

You have a Numba function decorated with `@jit` (no extra arguments) and you call it from multiple threads using `concurrent.futures.ThreadPoolExecutor`. What happens?

- A) The threads run truly in parallel because Numba releases the GIL automatically
- B) The threads run sequentially because the GIL is still held during JIT execution
- C) The code raises a `RuntimeError` because Numba is not thread-safe
- D) The threads run in parallel only on the first call, then sequentially afterwards

**Answer: B**

- A) Incorrect — `@jit` alone does not release the GIL. GIL release requires the explicit `nogil=True` keyword argument. Without it, all threads compete for the same GIL, serialising execution despite there being multiple OS threads.
- B) Correct — without `nogil=True`, Numba holds the GIL during execution. The ThreadPoolExecutor creates real OS threads, but they cannot run simultaneously; each waits for the GIL, turning apparent parallelism into serial execution.
- C) Incorrect — Numba does not raise any error for threaded use. It silently serialises rather than crashing, which is actually a more dangerous failure mode because you lose performance without any error signal.
- D) Incorrect — GIL behaviour is consistent across all calls. There is no "first call parallel, rest sequential" mechanism; the GIL is held on every call when `nogil` is not set.

---

## Q5 — nogil=True Effect

> **Week reference:** Week 9

**Mental Model:** `nogil=True` is the bridge between Numba and Python threading. It signals Numba to release the GIL before entering compiled code, allowing OS threads to run simultaneously. Without it, a ThreadPool is no faster than serial.

Which decorator enables a Numba function to execute in parallel across multiple OS threads?

- A) `@jit(parallel=True)`
- B) `@jit(nopython=True)`
- C) `@jit(nogil=True)`
- D) `@njit(cache=True)`

**Answer: C**

- A) Incorrect — `parallel=True` auto-parallelises `prange` loops *within a single function call* using Numba's internal thread pool, but does not release the GIL for external threads calling the function concurrently.
- B) Incorrect — `nopython=True` enforces strict compilation without Python-object fallback. It improves speed and correctness guarantees but has absolutely no effect on GIL management; the GIL is still held.
- C) Correct — `nogil=True` explicitly tells Numba to release the Python GIL when entering the compiled function body. This allows a `ThreadPoolExecutor` (or any other Python threading mechanism) to run multiple calls truly in parallel on separate CPU cores.
- D) Incorrect — `cache=True` saves compiled machine code to disk (in `__pycache__`) to avoid recompilation on subsequent runs. It has no effect on GIL behaviour or thread parallelism.

---

## Q6 — JIT Speedup for Python Loops

> **Week reference:** Week 9

**Mental Model:** Python's interpreter overhead per loop iteration is ~50–100 ns. Native LLVM-compiled code for the same arithmetic runs in ~0.5–1 ns per iteration — a factor of ~100. The speedup primarily comes from eliminating interpreter overhead, not from any algorithmic change.

A function contains a tight Python `for`-loop performing per-element floating-point arithmetic on one million elements. After applying `@njit`, which speedup over plain Python is most typical?

- A) 2–5×
- B) 10–20×
- C) 100–200×
- D) 10 000–100 000×

**Answer: C**

- A) Incorrect — 2–5× is typical for simpler optimisations like avoiding attribute lookups or using local variable caching. Numba eliminates the entire Python bytecode interpreter for the loop body, achieving far more.
- B) Incorrect — 10–20× underestimates the benefit; a typical Python loop iteration costs ~50–100 ns while Numba-compiled arithmetic costs ~0.5–1 ns, a ratio of ~100×.
- C) Correct — for tight numerical loops, `@njit` typically delivers ~100–200× speedup over pure Python. This comes from eliminating per-iteration interpreter overhead and enabling LLVM optimisations like loop unrolling and vectorisation.
- D) Incorrect — this range is unrealistically high. Modern CPython loops run at ~10–100 ns/iteration, and native code cannot be faster than physically possible clock cycles. Even at 10 ns vs 0.1 ns, that is only 100×.

---

## Q7 — JIT vs Already-Vectorized NumPy

> **Week reference:** Week 9

**Mental Model:** NumPy operations are already compiled C/Fortran code (BLAS, MKL, etc.). Numba competes with, not improves upon, these. The JIT benefit applies to *Python loop overhead*, not to code that already avoids Python loops.

A colleague applies `@njit` to a function that consists entirely of NumPy array operations (e.g., `np.dot`, `np.sum`, slicing). What is the most likely outcome?

- A) The function runs ~100× faster because Numba further optimises NumPy
- B) Performance is similar to or slightly worse than the plain NumPy version
- C) The function raises a `TypingError` because Numba cannot handle NumPy calls
- D) The GIL is released automatically, giving linear speedup with threads

**Answer: B**

- A) Incorrect — NumPy operations are already compiled C/Fortran routines (often using highly optimised BLAS or MKL). Numba cannot improve upon them; it may even add function call dispatch overhead compared to calling NumPy directly.
- B) Correct — Numba supports a large subset of NumPy, but for already-vectorised code it typically matches NumPy at best. For some operations Numba's code generation may be slightly less optimised than vendor-tuned NumPy libraries, resulting in a minor slowdown.
- C) Incorrect — Numba supports most standard NumPy functions and array operations in nopython mode. A `TypingError` would only occur if the function used unsupported Python objects or operations.
- D) Incorrect — `@njit` alone does not release the GIL. GIL release requires the explicit `nogil=True` argument, regardless of what the function contains.

---

## Q8 — When JIT Helps Most

> **Week reference:** Week 9

**Mental Model:** JIT excels where Python's interpreter overhead dominates: tight element-wise loops that cannot be expressed as a single NumPy call. If the work is already a library call (LAPACK, I/O, string ops), JIT adds nothing because the bottleneck is not Python bytecode.

For which of the following workloads does Numba JIT provide the greatest benefit over plain Python?

- A) Calling `np.linalg.eig` on a large matrix
- B) Reading a CSV file with `pandas.read_csv`
- C) A Python `for`-loop computing a running cumulative product element by element
- D) Vectorised string operations on a Pandas `Series`

**Answer: C**

- A) Incorrect — `np.linalg.eig` is already an optimised LAPACK routine (DGEEV or similar) running compiled Fortran. Numba cannot improve upon it; the bottleneck is the algorithm's O(n³) complexity, not Python overhead.
- B) Incorrect — file I/O in `pandas.read_csv` is dominated by disk bandwidth and CSV parsing in C. JIT compilation offers zero benefit because there is no Python numerical loop to eliminate.
- C) Correct — a running cumulative product loop has a strict serial dependency (each step depends on the previous), making it impossible to vectorise with standard NumPy. This is exactly the pattern where Numba excels: turning an O(n)-loop with Python interpreter overhead into native machine code.
- D) Incorrect — Pandas string operations work on Python `str` objects, which Numba cannot compile in `nopython` mode. Attempting to JIT-compile such a function would raise a `TypingError`.

---

## Q9 — cache=True

> **Week reference:** Week 9

**Mental Model:** `cache=True` is disk persistence of the compiled binary — like `.pyc` files for Python bytecode. It eliminates recompilation across Python processes but does not affect the first-ever compilation or in-memory behaviour.

What is the effect of using `@jit(cache=True)` on a Numba function?

- A) Results are cached in memory so repeated calls with the same inputs skip execution
- B) The compiled native code is saved to disk so the first call on subsequent Python runs avoids recompilation
- C) The function is compiled ahead of time at import, eliminating first-call latency entirely
- D) Numba caches intermediate LLVM IR to speed up future compilations of similar functions

**Answer: B**

- A) Incorrect — `cache=True` caches the compiled *binary machine code*, not the function results. Inputs are not memoized; the function still executes every time it is called.
- B) Correct — `cache=True` writes the compiled artifact to `__pycache__` on disk. On the next Python process start, Numba finds the cached binary and loads it without recompiling, eliminating the first-call latency for that process.
- C) Incorrect — compilation still happens on the very first call in a brand-new Python process where no cached artifact exists yet. `cache=True` only helps from the second process run onwards.
- D) Incorrect — Numba saves the final compiled machine code (ELF binary), not intermediate LLVM IR. LLVM IR is an intermediate representation; caching it would still require LLVM to do back-end compilation, defeating the purpose.

---

## Q10 — parallel=True and prange

> **Week reference:** Week 9

**Mental Model:** `parallel=True` alone does nothing without `prange`. Think of `prange` as the explicit annotation marking which loop iterations are independent. Without it, Numba has no loop to parallelise and the option is a no-op.

Which of the following correctly uses Numba's `parallel=True` option to parallelise a loop across CPU cores?

- A) `@jit(parallel=True)` with a regular `range` loop
- B) `@njit(parallel=True)` with `numba.prange` replacing `range`
- C) `@jit(nogil=True)` called from multiple threads, each handling a slice
- D) `@cuda.jit` targeting the GPU with a grid of threads

**Answer: B**

- A) Incorrect — `parallel=True` with a regular `range` does not auto-parallelise the loop. Numba only parallelises loops explicitly annotated with `prange`; a regular `range` loop is compiled as a serial loop even with `parallel=True`.
- B) Correct — `@njit(parallel=True)` combined with `numba.prange` (parallel range) instructs Numba's automatic parallelisation pass to distribute loop iterations across available CPU cores, similar to OpenMP `#pragma omp parallel for`.
- C) Incorrect — `nogil=True` enables external threading at the Python level; it does not use Numba's internal parallel scheduler or distribute iterations across cores within a single function call.
- D) Incorrect — `@cuda.jit` targets GPU threads, which is a completely different execution model. It requires explicit grid/block configuration and does not use CPU core parallelism via `prange`.

---

## Q11 — Type Restrictions in @njit

> **Week reference:** Week 9

**Mental Model:** `@njit` needs to infer static types for everything. Plain Python dicts, lists of mixed types, and class instances are dynamic Python objects — Numba cannot assign them a fixed type at compile time, so it raises a `TypingError` instead of silently degrading.

A developer tries to pass a Python dictionary `{"key": 1.0}` as an argument to a function decorated with `@njit`. What happens?

- A) Numba infers the dictionary type and compiles successfully
- B) Numba falls back to object mode and runs the function in the Python interpreter
- C) Numba raises a compilation error because arbitrary Python dicts are not supported in nopython mode
- D) The function runs correctly but without any speedup

**Answer: C**

- A) Incorrect — while Numba has limited support for typed dictionaries via `numba.typed.Dict` (with explicit key/value types), a plain Python `dict` created with literal syntax does not have a fixed static type and is not supported in nopython mode.
- B) Incorrect — `@njit` (`nopython=True`) never falls back to object mode. That fallback only exists in `@jit` without `nopython=True`. The entire point of `@njit` is to make the fallback impossible and surface compilation failures as errors.
- C) Correct — `@njit` raises a `TypingError` when it encounters types it cannot compile, such as arbitrary Python dictionaries, class instances with dynamic attributes, or generator expressions. The error message identifies the unsupported type.
- D) Incorrect — `@njit` does not silently degrade to slower uncompiled code. It either compiles successfully or raises an error; there is no middle ground of "runs but slowly."

---
