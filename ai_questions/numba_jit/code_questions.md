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

## Key Facts Summary

| Concept | Rule |
|---------|------|
| Warmup | ALWAYS call `@jit` function once before timing to exclude compilation |
| `nogil=True` | Releases the GIL — enables real thread parallelism with `ThreadPool` |
| `@njit` | Alias for `@jit(nopython=True)` — errors immediately if Numba cannot compile |
| `nopython=True` dicts | Not supported — use `numba.typed.Dict` instead |
| JIT on vectorized NumPy | Minimal benefit — `np.sum`, `np.dot`, etc. are already compiled C |
| `cache=True` | Saves compiled binary to disk — faster startup on subsequent runs |
| CUDA kernel launches | Host-only — cannot launch from inside a `@jit`/nopython function |
| Typical speedup | 100-200x for tight Python loops vs `@njit` |
