# Numba JIT — Code-Based MCQ Practice

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

> The first call to a `@jit`-decorated function triggers compilation. The measured time includes compilation overhead, not just execution time. To benchmark correctly, call `dot(a, b)` once as a warmup before starting the timer, then time the second (compiled) call.

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

> Without `nogil=True`, Numba holds the Python GIL during the compiled function's execution. Even though the function runs in nopython mode, the GIL is still acquired before entry. All 4 threads will run, but only one at a time — no true parallelism.

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

> `@jit(nopython=True, nogil=True)` compiles to native code and releases the GIL for the duration of the call. `ThreadPool` threads can then execute `simulate` concurrently on separate CPU cores, achieving true parallelism for CPU-bound work.

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

> Standard Python `dict` objects are not supported in `nopython=True` mode. Numba cannot infer a static type for a generic Python dict and will raise a `numba.core.errors.TypingError` at compilation time. To use a dictionary-like structure, you would need `numba.typed.Dict` with explicit key/value types.

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

> A 100-200x speedup is entirely typical when comparing an interpreted pure Python loop to the same loop compiled to native machine code via `@njit`. The warmup on a smaller array is valid — it triggers compilation for the correct input type (`float64` NumPy array), and the compiled code is reused for the full-size benchmark.

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

> `np.sum` is a highly optimized, pre-compiled C routine. Wrapping it in `@njit` adds JIT compilation overhead on the first call and provides no meaningful runtime benefit — the bottleneck is already eliminated. `@njit` delivers the largest gains when replacing interpreted Python loops, not already-vectorized NumPy calls.

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

> `cache=True` instructs Numba to write the compiled native binary (`.nbi`/`.nbc` files) to a `__pycache__` directory. On subsequent runs of the same script, Numba detects the cached binary and loads it directly, skipping the JIT compilation step. This reduces startup time for scripts that call expensive-to-compile functions. It does NOT cache results (that would be memoization).

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

> CUDA kernel launches (`kernel[blocks, threads](args)`) are host-side operations that invoke the CUDA runtime API. They cannot be called from inside a Numba `nopython` JIT-compiled function. The `@cuda.jit` kernel must be launched directly from regular host Python code. To use both CPU JIT and GPU computation, the CPU function should call a Python wrapper that launches the GPU kernel.

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
