# Week 6 — Parallel Reductions & Shared Memory Syntax Reference

## Parallel Reduction Requirements

An operator can be used in a parallel reduction if it is:
1. **Associative**: `(a op b) op c == a op (b op c)` — REQUIRED
2. **Commutative**: `a op b == b op a` — also required for unordered reductions

**Test with a counterexample.** For `abssum(x,y) = abs(x+y)`:
- a=1, b=2, c=-3
- Left: `abssum(abssum(1,2), -3) = abssum(3,-3) = 0`
- Right: `abssum(1, abssum(2,-3)) = abssum(1,1) = 2`
- `0 ≠ 2` → NOT associative → cannot use in parallel reduction

**Valid:** sum, product, min, max, AND, OR, XOR, set intersection
**Invalid:** `abs(x+y)` (not associative), matrix multiply (not commutative)

---

## Reduction Complexity

**Binary tree reduction:**
```
Depth  = ceil(log2(N))
Speedup = N / log2(N)   (with unlimited processors)
```

**Flat (two-level) reduction:**
```
Time    = N/T + T       (N elements, T chunks)
Optimal T = sqrt(N)
Speedup   = sqrt(N) / 2
```

---

## Shared Memory with multiprocessing

```python
import ctypes
import multiprocessing as mp
import numpy as np

def init(shared_arr_):
    global shared_arr
    shared_arr = shared_arr_

# Allocate shared memory block
data = np.load('data.npy')
shared_arr = mp.RawArray(ctypes.c_float, data.size)
arr = np.frombuffer(shared_arr, dtype='float32').reshape(data.shape)
np.copyto(arr, data)   # copy data into shared block

# Workers access via module global — no pickling/IPC overhead
pool = mp.Pool(n, initializer=init, initargs=(shared_arr,))
```

### `mp.RawArray(typecode, size_or_initializer)`
- **What**: allocates a C array in shared memory accessible by all processes
- **typecode**: `ctypes.c_float`, `ctypes.c_double`, `ctypes.c_int`, etc.
- **Gotcha**: must wrap with `np.frombuffer` to use as NumPy array

### `np.frombuffer(buffer, dtype)`
- **What**: creates NumPy array sharing memory with a raw buffer
- **Returns**: 1D NumPy array (reshape if needed)

---

## NUMA (Non-Uniform Memory Access)

On dual-socket servers, memory lands on one socket by default → speedup plateaus at ~50% of cores.

```bash
numactl --interleave=all python script.py   # spread memory across all sockets
```

---

## modern Python SharedMemory (Python 3.8+)

```python
from multiprocessing import shared_memory
import numpy as np

# Create
shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
np.copyto(shared_arr, arr)

# In worker (attach by name)
existing_shm = shared_memory.SharedMemory(name=shm.name)
worker_arr = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

# Cleanup (must do both)
shm.close()
shm.unlink()
```

---

## Exam Traps

| Trap | Correct |
|---|---|
| Commutativity alone is sufficient for reduction | Need ASSOCIATIVITY too |
| `abs(x+y)` fails because not commutative | It IS commutative; fails ASSOCIATIVITY |
| Matrix multiply fails because not associative | MM IS associative; fails COMMUTATIVITY |
| Binary tree speedup = log2(N) | Speedup = N/log2(N) |
