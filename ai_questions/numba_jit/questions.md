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

## Set 3 — Extended Practice

- [Q12 — prange and Nested Loops](#q12--prange-and-nested-loops)
- [Q13 — Object Mode Fallback with Plain @jit](#q13--object-mode-fallback-with-plain-jit)
- [Q14 — numba.typed.List vs Python List](#q14--numbatypedlist-vs-python-list)
- [Q15 — @numba.vectorize for Ufuncs](#q15--numbavectorize-for-ufuncs)
- [Q16 — Loop-Carried Dependency and prange](#q16--loop-carried-dependency-and-prange)
- [Q17 — Type Specialization and Recompilation](#q17--type-specialization-and-recompilation)
- [Q18 — cache=True Invalidation](#q18--cachetrue-invalidation)
- [Q19 — numba.get_num_threads()](#q19--numbaget_num_threads)
- [Q20 — @numba.stencil Decorator](#q20--numbastencil-decorator)
- [Q21 — Supported NumPy Subset in nopython Mode](#q21--supported-numpy-subset-in-nopython-mode)

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

## Set 3 — Extended Practice

---

## Q12 — prange and Nested Loops

> **Week reference:** Week 9

**Mental Model:** `prange` parallelises the loop it directly replaces — Numba's automatic parallelisation pass targets the outermost annotated loop only. Inner loops run serially within each parallel worker. If you annotate only an inner loop with `prange`, the outer loop still runs serially and the inner parallelism is launched from a single thread, which typically adds overhead without benefit.

A function has two nested loops. A developer replaces only the inner loop's `range` with `prange` while keeping `parallel=True`. What is the most likely outcome?

- A) Both loops run in parallel, doubling the available parallelism
- B) Only the inner loop is parallelised; the outer loop runs serially, potentially adding scheduling overhead with little benefit
- C) Numba raises an error because `prange` is only valid in the outermost loop position
- D) The function behaves identically to using `range` in both loops

**Answer: B**

- A) Incorrect — `prange` does not recursively parallelise outer loops. Only the loop body it immediately surrounds is scheduled across threads; the outer loop continues to call that body serially, one iteration at a time.
- B) Correct — Numba's parallel scheduler is launched once per outer-loop iteration. For a tight inner loop with small iteration counts, the thread-pool dispatch overhead can exceed the computation time, making performance worse than serial code. The canonical pattern is to annotate the outermost loop with `prange` so that the largest grain of work is distributed.
- C) Incorrect — Numba does not raise an error for `prange` in inner loops; it silently compiles and runs, but with suboptimal (or even negative) performance. There is no syntactic restriction on position.
- D) Incorrect — using `prange` always engages the parallel scheduler. Even if performance is poor, the execution model differs from serial `range`; threads are launched and synchronised, which changes timing and resource usage.

---

## Q13 — Object Mode Fallback with Plain @jit

> **Week reference:** Week 9

**Mental Model:** Plain `@jit` (without `nopython=True`) has two modes: nopython mode (fast, compiled) and object mode (slow, interpreted fallback). When Numba cannot infer types for part of a function in nopython mode, `@jit` silently compiles that part in object mode and continues. This is the most dangerous Numba trap: you get no error, no warning by default, and the function runs — just as slowly as pure Python.

A function decorated with `@jit` (no arguments) contains a line that uses an unsupported Python object. What happens at call time?

- A) Numba raises a `TypingError` immediately, like `@njit`
- B) Numba silently falls back to object mode and executes the function using the Python interpreter, with no speedup for the unsupported parts
- C) Numba skips the unsupported line and compiles the rest in nopython mode
- D) Python raises an `AttributeError` because `@jit` cannot decorate functions with unsupported types

**Answer: B**

- A) Incorrect — `TypingError` on encountering an unsupported type is the behaviour of `@njit` (nopython=True). Plain `@jit` without `nopython=True` does not raise this error; it falls back gracefully (but silently) to object mode.
- B) Correct — This is the most important Numba gotcha: `@jit` without `nopython=True` will happily compile and run in object mode if nopython compilation fails. The function returns a correct result but with Python-interpreter-level performance. The only visible symptom is that it is slow. This is why `@njit` (or `@jit(nopython=True)`) is strongly recommended — it makes the failure loud rather than silent.
- C) Incorrect — Numba does not selectively skip lines. When object mode is triggered, the entire function (or a region containing the unsupported type) runs through the Python interpreter; there is no mixed line-by-line compilation.
- D) Incorrect — `@jit` is a valid decorator for any Python function regardless of what types it uses. The decorator itself never raises `AttributeError`; it defers error handling (or fallback) to compile time, which happens lazily on the first call.

---

## Q14 — numba.typed.List vs Python List

> **Week reference:** Week 9

**Mental Model:** Numba requires all types to be statically inferrable in nopython mode. A plain Python `list` can hold mixed types, so Numba cannot assign it a fixed element type at compile time. `numba.typed.List` is a Numba-native container that enforces a homogeneous element type (e.g., all `float64`), allowing it to be used inside `@njit` functions. The trade-off: `numba.typed.List` is slower than a NumPy array for numerical work and should be used only when a dynamically-growing sequence is truly needed.

Which of the following is the correct way to pass a growing list of floats into an `@njit` function?

- A) Pass a plain Python `list` — Numba infers the element type automatically
- B) Use `numba.typed.List` with a declared element type
- C) Convert the list to a Python `tuple` before passing it
- D) Use `@jit` without `nopython=True` so the list is handled in object mode

**Answer: B**

- A) Incorrect — a plain Python `list` is a dynamically-typed object. `@njit` raises a `TypingError` because Numba cannot statically infer a uniform element type for a generic Python list. Even if all elements happen to be the same type at runtime, Numba performs type inference at compile time and cannot guarantee homogeneity.
- B) Correct — `numba.typed.List` enforces a fixed element dtype. You create one with `numba.typed.List.empty_list(numba.float64)` or by reflection from a regular list using `numba.typed.List([1.0, 2.0])`. Inside `@njit`, Numba knows every element is `float64` and can compile efficiently. This is the supported pattern for mutable sequences in nopython mode.
- C) Incorrect — a Python `tuple` of uniform numeric types IS supported in `@njit` (Numba can infer the type of each fixed-position element). However, tuples are immutable and fixed-length, so they do not support `.append()`. For a growing sequence, `numba.typed.List` is the correct choice.
- D) Incorrect — using object mode via plain `@jit` works around the type restriction but gives no compiled speedup for the list-processing code. This defeats the purpose of using Numba. The correct solution is `numba.typed.List`, not object mode fallback.

---

## Q15 — @numba.vectorize for Ufuncs

> **Week reference:** Week 9

**Mental Model:** `@numba.vectorize` creates a NumPy universal function (ufunc) from a scalar Python function. You write the function for a single element, and Numba generates a compiled ufunc that broadcasts over arrays of any shape — identical to built-in NumPy ufuncs like `np.add`. You must declare the type signature(s) explicitly. The result is a true NumPy ufunc: it supports broadcasting, reduction (`.reduce()`), and accumulation (`.accumulate()`).

What is the primary purpose of the `@numba.vectorize` decorator?

- A) To parallelise a function using multiple CPU cores via `prange`
- B) To create a compiled NumPy ufunc from a scalar function, enabling it to broadcast over arrays
- C) To compile a function for GPU execution using CUDA threads
- D) To cache the function's results in memory so repeated calls with the same inputs are skipped

**Answer: B**

- A) Incorrect — CPU parallelism across cores uses `@njit(parallel=True)` combined with `prange`. `@vectorize` does not introduce multi-core parallelism by default; it targets element-wise array operations (though `@vectorize(target='parallel')` does enable parallel CPU execution as a secondary feature).
- B) Correct — `@numba.vectorize` compiles a scalar function into a NumPy ufunc. You write: `def f(x, y): return x + y * 2.0` and decorate it with `@vectorize(['float64(float64, float64)'])`. The resulting object `f` accepts arrays of any shape and broadcasts exactly like `np.add`. This is the idiomatic Numba way to create custom ufuncs without writing C extensions.
- C) Incorrect — GPU execution uses `@numba.cuda.jit` or `@vectorize(target='cuda')`. The default target for `@vectorize` is `'cpu'`, which generates native CPU ufunc code, not GPU kernels.
- D) Incorrect — `@vectorize` has no memoization functionality. The compiled function executes on every call; results are not stored between calls. Memoization would require a separate mechanism like `functools.lru_cache`, which is incompatible with array inputs.

---

## Q16 — Loop-Carried Dependency and prange

> **Week reference:** Week 9

**Mental Model:** `prange` is only safe when loop iterations are independent — the result of iteration `i` must not feed into iteration `i+1`. A loop-carried dependency (each iteration reads a value written by the previous one) cannot be safely parallelised: different threads would race to read and write the same memory location, producing non-deterministic results. The one exception is simple reductions (e.g., summing into a scalar), which Numba handles with a built-in reduction pattern using `prange`.

A function uses `prange` to parallelise a loop where each iteration updates a shared accumulator: `total += arr[i]`. Ignoring Numba's built-in reduction handling, what is the risk of naively using `prange` for this pattern?

- A) The loop will not compile because `prange` forbids any use of variables outside the loop
- B) Multiple threads write to `total` concurrently, causing a data race and an incorrect result
- C) The loop will run correctly but serially, identical to `range`
- D) Numba automatically inserts a lock so the result is always correct

**Answer: B**

- A) Incorrect — `prange` does not forbid access to outer-scope variables. Simple reductions (`total += arr[i]`) are a documented pattern that Numba supports via a special reduction transformation. The question asks about the risk when this handling is not assumed.
- B) Correct — without synchronisation, multiple threads read the current value of `total`, add their `arr[i]`, and write back — the classic read-modify-write race condition. If thread A reads `total=5`, thread B simultaneously reads `total=5`, both add their values and write back, one update is lost. In practice, Numba detects simple scalar reductions and handles them safely, but any non-trivial shared accumulation (e.g., updating a shared array element) remains a race hazard with `prange`.
- C) Incorrect — `prange` with `parallel=True` genuinely launches multiple threads; it does not silently degrade to serial execution like plain `range`. If there is a race, the result is non-deterministic, not deterministically correct.
- D) Incorrect — Numba does not insert locks around arbitrary shared writes. The only automatic safety mechanism is the scalar reduction pattern (`total += ...`) which Numba rewrites as a private-accumulator + merge pattern. Arbitrary shared mutable state is the programmer's responsibility.

---

## Q17 — Type Specialization and Recompilation

> **Week reference:** Week 9

**Mental Model:** Numba compiles a separate specialisation of a `@jit` function for each unique combination of argument types it encounters. Calling `f(int_array)` and then `f(float_array)` produces two compiled versions stored in the same function object. Each new type combination triggers a fresh LLVM compilation. This is transparent to the user but means: (a) the first call for each new type is slow, and (b) a function called with many different types accumulates many compiled versions.

A `@njit` function `process(arr)` is called three times: first with a `float64` array, then with a `float32` array, then with a `float64` array again. How many times does Numba compile the function?

- A) Once — Numba compiles one generic version that handles all numeric types
- B) Twice — once for `float64` and once for `float32`
- C) Three times — once per call
- D) Zero times — `@njit` compiles eagerly at decoration time

**Answer: B**

- A) Incorrect — Numba does not generate generic code. Each type specialisation produces a separate LLVM IR and machine code path tuned for that exact type (including SIMD widths, register usage, etc.). There is no single "generic numeric" version.
- B) Correct — Numba caches compiled specialisations by argument type signature. The first `float64` call triggers compilation of the `float64` specialisation. The `float32` call triggers a second compilation. The second `float64` call hits the already-compiled `float64` version in the cache — no recompilation occurs. Result: exactly two compilations.
- C) Incorrect — the third call reuses the `float64` specialisation compiled during the first call. Numba only recompiles when a previously unseen type combination is encountered; it does not recompile on every call.
- D) Incorrect — `@njit` is lazy by default; it defers compilation to the first call with a given type. Eager compilation requires explicitly calling the function with typed arguments or using `@njit(cache=True)` combined with an ahead-of-time compilation step. Even with `cache=True`, the initial compilation is still deferred to first use.

---

## Q18 — cache=True Invalidation

> **Week reference:** Week 9

**Mental Model:** Numba's `cache=True` stores the compiled binary alongside a hash of the source function. On the next Python process start, Numba checks: (1) does a cached file exist? (2) does the source hash still match? If the function's source code has changed since the last run, the cached binary is invalidated and Numba recompiles from scratch. Similarly, updating Numba itself invalidates all caches because the binary format may change.

Under which circumstance does `cache=True` NOT eliminate the first-call compilation cost?

- A) When the function is called a second time in the same Python process
- B) When the function's source code has changed since the cache was written
- C) When the same script is run a second time without any code changes
- D) When `parallel=True` is also specified alongside `cache=True`

**Answer: B**

- A) Incorrect — within the same Python process, the compiled specialisation is already resident in memory from the first call. There is no disk lookup at all for subsequent calls in the same process; `cache=True` only matters across separate process starts.
- B) Correct — Numba stores a hash of the function's source code with the cached binary. When the source changes (even a single character), the hash no longer matches and Numba discards the stale cache and recompiles. This is the intended behaviour: you should never run stale compiled code against changed source. A Numba version upgrade also invalidates the cache.
- C) Incorrect — running the same unchanged script a second time is exactly the scenario where `cache=True` provides its benefit. Numba finds the valid cached binary, verifies the source hash matches, and loads the pre-compiled artifact, skipping the LLVM compilation step entirely.
- D) Incorrect — `cache=True` is fully compatible with `parallel=True`. The parallel compilation result is also cached to disk. Combining both options is common and valid; neither flag inhibits the other.

---

## Q19 — numba.get_num_threads()

> **Week reference:** Week 9

**Mental Model:** Numba's parallel scheduler uses a thread pool whose size defaults to the number of logical CPU cores. `numba.get_num_threads()` queries the current pool size and `numba.set_num_threads(n)` changes it. On an HPC cluster where your job requests fewer cores than the node has, Numba may still launch a full-node-width thread pool and oversubscribe the CPUs — causing context-switching overhead and interference with other jobs. Always set the thread count to match your allocated core count on shared systems.

On an HPC cluster, a job requests 4 CPU cores via `#BSUB -n 4`. A `@njit(parallel=True)` function is called without any thread configuration. What problem might arise?

- A) The function raises a `RuntimeError` because `parallel=True` requires explicit thread count configuration
- B) Numba defaults to using all logical cores on the node (e.g., 32 or 64), oversubscribing the CPU and interfering with other jobs
- C) The function runs serially because Numba detects the HPC environment and disables threading
- D) The function uses exactly 4 threads because LSF automatically sets the `OMP_NUM_THREADS` environment variable

**Answer: B**

- A) Incorrect — Numba does not require explicit thread configuration. It silently defaults to the node's total logical core count (via `os.cpu_count()`) if no limit is set. No error is raised; the problem is silent oversubscription.
- B) Correct — Numba reads the number of available CPUs from the OS, which on a multi-user HPC node reports the full node count regardless of how many cores your job was allocated. A 32-core node with a 4-core job allocation will still launch 32 Numba threads. These extra threads compete for CPU time slots allocated to other jobs, slowing your job and degrading cluster performance for others. The fix: call `numba.set_num_threads(4)` at the start of your script, or set the `NUMBA_NUM_THREADS` environment variable in your job script.
- C) Incorrect — Numba has no HPC-awareness. It does not detect LSF/SLURM environments and does not auto-disable threading. It always defaults to the full logical CPU count.
- D) Incorrect — LSF does not automatically set `OMP_NUM_THREADS` for Python jobs. Even if it did, Numba does not read `OMP_NUM_THREADS`; it reads `NUMBA_NUM_THREADS` or queries `os.cpu_count()`. You must explicitly set the thread count.

---

## Q20 — @numba.stencil Decorator

> **Week reference:** Week 9

**Mental Model:** `@numba.stencil` is a decorator for functions that compute a single output element as a function of its neighbours in an input array (a stencil operation, like a convolution or finite-difference). The decorated function writes the kernel for one element — Numba handles the loop over the entire array and boundary management. It is conceptually similar to a ufunc but for neighbourhood operations. Boundary elements that reference out-of-bounds neighbours are set to zero by default.

What type of computation is `@numba.stencil` specifically designed to express?

- A) Reductions that collapse an entire array to a scalar value
- B) Element-wise operations where each output element depends only on the corresponding input element
- C) Neighbourhood operations where each output element depends on nearby input elements (e.g., finite differences, convolutions)
- D) Scatter operations that write a single input value to multiple output positions

**Answer: C**

- A) Incorrect — reductions (sum, max, min) are not stencil operations. A stencil reads from a local neighbourhood and writes to one output element; it does not accumulate across the entire array into a scalar. Reductions use `prange` with an accumulator variable or NumPy reduction functions.
- B) Incorrect — purely element-wise operations (each output depends on exactly one input element at the same index) are handled by `@numba.vectorize`. `@stencil` is specifically for the case where the output at index `i` depends on elements like `arr[i-1]`, `arr[i]`, `arr[i+1]` — a neighbourhood, not a single point.
- C) Correct — `@numba.stencil` is purpose-built for stencil computations: each output element is computed from a fixed neighbourhood pattern of input elements. Classic examples are 1D finite differences (`arr[i+1] - arr[i-1]`), 2D Laplacians, and image convolutions with small kernels. You define the kernel for a single output position, and Numba generates an efficient loop over the full array.
- D) Incorrect — scatter operations (one input to many outputs) are the opposite of stencil operations (many inputs to one output). Numba has no dedicated scatter decorator; scatter is implemented manually with index arithmetic in a regular `@njit` loop.

---

## Q21 — Supported NumPy Subset in nopython Mode

> **Week reference:** Week 9

**Mental Model:** Numba supports a large but not complete subset of NumPy in nopython mode. Well-supported operations include array creation (`np.zeros`, `np.ones`, `np.empty`), arithmetic, indexing, slicing, `np.sum`, `np.dot`, `np.sqrt`, and many more. Operations that are NOT supported include functions that return Python objects (e.g., `np.unique` with `return_counts=True`), string operations, and most `np.linalg` routines (though `np.linalg.norm` is supported). The practical test: if the return type is a plain numeric array or scalar, it is likely supported; if it returns structured Python objects, it probably is not.

Which of the following NumPy operations is supported inside a `@njit` function?

- A) `np.unique(arr, return_counts=True)` — returns a tuple of arrays including a count array
- B) `np.linalg.eig(A)` — returns eigenvalues and eigenvectors
- C) `np.zeros((n, m), dtype=np.float64)` — allocates a zero-filled 2D array
- D) `np.savetxt('output.txt', arr)` — writes an array to a text file

**Answer: C**

- A) Incorrect — `np.unique` with `return_counts=True` returns a Python tuple of arrays with varying structure. Numba's nopython mode does not support this variant of `np.unique` as of recent versions. Even basic `np.unique` support is limited; the multi-return form is not in the supported subset.
- B) Incorrect — `np.linalg.eig` is not in Numba's supported NumPy subset for nopython mode. It calls into LAPACK via Python object machinery. Numba supports a limited set of `np.linalg` functions (`np.linalg.norm`, `np.linalg.solve`, `np.linalg.inv`), but eigendecomposition is not among them. Calling it inside `@njit` raises a `TypingError`.
- C) Correct — `np.zeros`, `np.ones`, `np.empty`, and their shape/dtype variants are fully supported in nopython mode. Allocating and initialising arrays inside `@njit` functions is common and efficient. Numba translates these directly to memory allocation calls without going through the Python interpreter.
- D) Incorrect — `np.savetxt` involves file I/O and string formatting, both of which require the Python interpreter and are entirely outside Numba's compiled subset. Calling it inside `@njit` raises a `TypingError`. File I/O must be done in regular Python code outside the JIT-compiled function.

---
