# Week 9 — Numba JIT & CUDA Syntax Reference

## Numba CPU JIT

```python
from numba import jit, njit

@jit(nopython=True)     # compile to native code, no Python fallback
def fast_func(A, B):
    ...

@njit                   # same as @jit(nopython=True)
def fast_func(A, B):
    ...

@jit(nopython=True, nogil=True)   # releases GIL → ThreadPool can parallelize
def fast_func(arr):
    ...

@jit(nopython=True, cache=True)   # saves compiled binary to disk
def fast_func(arr):
    ...
```

**ALWAYS warm up before timing:**
```python
fast_func(x)           # first call = compilation (slow)
t = perf_counter()
fast_func(x)           # second call = fast
elapsed = perf_counter() - t
```

**JIT helps:** tight Python loops with per-element arithmetic.
**JIT does NOT help:** already-vectorized NumPy (already compiled C).

---

## CUDA Kernel

```python
from numba import cuda

@cuda.jit
def kernel(x, y, out):
    i = cuda.grid(1)              # global 1D thread index
    if i < x.shape[0]:            # ALWAYS bounds check
        out[i] = x[i] + y[i]

@cuda.jit
def kernel_2d(A, B, C):
    i, j = cuda.grid(2)           # global 2D thread indices
    if i < C.shape[0] and j < C.shape[1]:
        ...

@cuda.jit(device=True)            # helper — callable from kernel only, NOT from host
def helper(a, b):
    return a + b
```

---

## Thread Index Variables (inside kernel)

| Variable | Meaning |
|---|---|
| `cuda.threadIdx.x` | index within block (0 to blockDim-1) |
| `cuda.blockIdx.x` | which block (0 to bpg-1) |
| `cuda.blockDim.x` | threads per block |
| `cuda.gridDim.x` | blocks per grid |
| `cuda.grid(1)` | blockIdx.x * blockDim.x + threadIdx.x |
| `cuda.grid(2)` | (row, col) — global 2D position |

---

## Launching a Kernel

```python
# 1D
tpb = 256                                  # threads per block (multiple of 32!)
bpg = (n + tpb - 1) // tpb                # blocks per grid — always ROUND UP
kernel[bpg, tpb](x, y, out)

# 2D
tpb = (16, 16)
bpg = ((rows + tpb[0] - 1) // tpb[0],
       (cols + tpb[1] - 1) // tpb[1])
kernel_2d[bpg, tpb](A, B, C)
```

**Max threads per block = 1024. Warp size = 32. Always use multiples of 32.**

---

## Memory Transfers

```python
from numba import cuda

d_x = cuda.to_device(x)           # HtoD: NumPy → GPU
d_out = cuda.device_array(n)       # allocate on GPU, no transfer (0 bytes HtoD)
d_out = cuda.device_array_like(x)  # allocate same shape/dtype as x, no transfer
result = d_out.copy_to_host()      # DtoH: GPU → NumPy

# Pinned memory (faster DMA)
with cuda.pinned(x, y, out):
    kernel[bpg, tpb](x, y, out)
    cuda.synchronize()
```

**Numba auto-transfers (implicit mode):** calling kernel with NumPy arrays → ALL args HtoD before + ALL args DtoH after. For 2 args: 2 HtoD + 2 DtoH = 4 transfers. Optimal: 1 HtoD per input + 1 DtoH per output.

---

## cuda.synchronize()

```python
cuda.synchronize()    # CPU blocks here until all GPU work is complete
```

**Required before:** timing with `perf_counter()`, reading results from explicit device arrays.
**Not the same as** `cuda.syncthreads()` which is INSIDE the kernel (synchronizes threads within a block).

---

## Timing a Kernel

```python
from time import perf_counter
from numba import cuda

# Warmup
kernel[bpg, tpb](x, y, out)
cuda.synchronize()

rep = 200
t = perf_counter()
for _ in range(rep):
    kernel[bpg, tpb](x, y, out)
cuda.synchronize()                    # wait for ALL 200 runs to finish
print((perf_counter() - t) / rep * 1000, 'ms')
```

Note: `cuda.synchronize()` is OUTSIDE the loop — GPU queues all 200 kernels back-to-back.

---

## Exam Traps

| Trap | Correct |
|---|---|
| Time first @jit call | First call = compilation — warm up first |
| JIT always faster than NumPy | Not for already-vectorized code |
| `nogil` not needed for ThreadPool | GIL held without it — add `nogil=True` |
| `@njit` silently falls back to Python | Raises TypingError |
| `cuda.synchronize()` = `cuda.syncthreads()` | Different scope — host vs within block |
| Bounds check not needed | Grid rounds up → some threads are out of bounds |
| 256×1 block gives best coalescing | 1×256 is best — threads differ in col (last axis) |
