# Week 10 — CuPy, GPU Reductions & nsys Syntax Reference

## CuPy — Drop-in NumPy Replacement

```python
import cupy as cp

# Load directly into GPU memory
data = cp.loadtxt('file.csv', delimiter=',', skiprows=1, usecols=(1, 2))
d_arr = cp.array(np_arr)           # NumPy → GPU
np_arr = cp.asnumpy(d_arr)         # GPU → NumPy (also: d_arr.get())

# All NumPy operations work identically
cp.sin(x), cp.cos(x), cp.arctan2(a, b)
cp.radians(x), cp.sqrt(x)
cp.triu_indices(n, k=1)            # upper triangular indices

# Convert GPU scalar back to Python float
float(d_arr.mean())
```

**Rule:** Replace `import numpy as np` with `import cupy as cp`. That's it.

**Critical difference from NumPy:** Python loops over CuPy arrays launch a separate kernel per iteration → huge `cuLaunchKernel` overhead. Always use vectorized (no-loop) operations.

---

## Shared Memory in CUDA Kernel

```python
from numba import cuda

TPB = 128

@cuda.jit
def kernel(data, out, n):
    # Declare shared array — size must be compile-time constant (TPB)
    sdata = cuda.shared.array(TPB, dtype=data.dtype)

    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    # Load from global → shared (pad with 0.0 if out of bounds)
    sdata[tid] = data[i] if i < n else 0.0
    cuda.syncthreads()     # WAIT — all threads must load before anyone reads

    # Do work in shared memory
    s = 1
    while s < cuda.blockDim.x:
        idx = 2 * s * tid          # strided index — avoids warp divergence
        if idx < cuda.blockDim.x:
            sdata[idx] += sdata[idx + s]
        s *= 2
        cuda.syncthreads()         # sync after each reduction step

    if tid == 0:
        out[cuda.blockIdx.x] = sdata[0]
```

### `cuda.shared.array(size, dtype)`
- **What**: allocates fast per-block shared memory
- **Scope**: visible only to threads in the SAME block
- **Size**: must be a compile-time constant (use `TPB` variable, not `cuda.blockDim.x`)
- **Gotcha**: very limited — < 100 KB per SM; too large → kernel launch fails

### `cuda.syncthreads()`
- **What**: barrier — ALL threads in block must reach this line before any proceeds
- **Scope**: within one block only (NOT across blocks)
- **When**: after writing to shared memory before reading it

---

## Warp Divergence

A **warp** = 32 threads executing the same instruction simultaneously.

**Divergence:** some threads take `if`, others take `else` → GPU serializes both branches → wasted cycles.

**Original (diverged):**
```python
if tid % (2 * s) == 0:          # active/idle threads interleaved within warp
    sdata[tid] += sdata[tid + s]
```

**Fixed (no divergence):**
```python
idx = 2 * s * tid               # active threads packed at front of warp
if idx < cuda.blockDim.x:
    sdata[idx] += sdata[idx + s]
```

Result: ~2.3× kernel speedup.

---

## GPU Profiling with nsys

```bash
# Record
nsys profile -o profile_name python script.py args

# View statistics
nsys stats profile_name.nsys-rep
```

### Key sections

| Section key | What it shows |
|---|---|
| `gpukernsum` | GPU kernel execution times (actual compute) |
| `gpumemtimesum` | HtoD and DtoH transfer times |
| `gpumemsizesum` | bytes transferred HtoD / DtoH |
| `cudaapisum` | CPU-side API overhead (kernel launches, malloc) |

**`cuLaunchKernel` in `cudaapisum` taking > kernel time → too many small kernels (CuPy loop pattern).**

---

## GPU Reduction Pattern

```python
TPB = 64

def get_grid(n, tpb):
    return (n + tpb - 1) // tpb

def reduce(x):          # x must be a device array
    n = len(x)
    bpg = get_grid(n, TPB)
    out = cuda.device_array(bpg, dtype=x.dtype)
    while bpg > 1:
        reduce_kernel[bpg, TPB](x, out, n)
        n = bpg
        bpg = get_grid(n, TPB)
        x[:n] = out[:n]
    reduce_kernel[bpg, TPB](x, out, n)
    return out
```

---

## Exam Traps

| Trap | Correct |
|---|---|
| `cuda.syncthreads()` = `cuda.synchronize()` | `syncthreads` = within block; `synchronize` = CPU waits for GPU |
| Shared memory is global to all blocks | Shared memory is PER BLOCK only |
| CuPy loop is fine like NumPy loop | CuPy loop = many kernel launches → slow |
| `cuda.shared.array(cuda.blockDim.x, ...)` | Size must be compile-time constant like `TPB` |
