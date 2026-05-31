# Week 3 — Cache Effects & Blosc Syntax Reference

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [NumPy Strides](#numpy-strides)
  - [`arr.strides`](#arrstrides)
- [Blosc Compression](#blosc-compression)
- [Cache Hierarchy (DTU XeonGold6226R)](#cache-hierarchy-dtu-xeongold6226r)
- [Row vs Column Access](#row-vs-column-access)
- [Exam Traps](#exam-traps)

---

## NumPy Strides

### `arr.strides`
- **What**: tuple of bytes to step in each dimension
- **Returns**: tuple of ints (bytes)
- For C-order (row-major) float64 shape (M, N): strides = `(N*8, 8)`
- Last axis always has smallest stride in C-order

```python
import numpy as np

a = np.ones((3, 4))         # float64
a.strides                    # (32, 8) — 32 bytes per row, 8 bytes per element

# Byte offset for a[i, j] = i * strides[0] + j * strides[1]

a.T.strides                  # (8, 32) — strides reversed, same data
np.shares_memory(a, a.T)     # True — same buffer
```

**Loop ordering rule:** innermost loop → smallest stride → sequential memory access.

Given strides `(600, 40, 8, 200)`:
- Sort ascending: 8(axis2) < 40(axis1) < 200(axis3) < 600(axis0)
- Inner→outer loop order: axis2, axis1, axis3, axis0

---

## Blosc Compression

```python
import blosc

# Compress a NumPy array
arr = np.random.rand(1000, 1000).astype('float32')
packed = blosc.pack_array(arr)      # returns bytes

# Decompress
arr2 = blosc.unpack_array(packed)   # returns NumPy array

# Settings
blosc.set_nthreads(4)               # parallel compression threads
blosc.compress(arr.tobytes(),       # compress raw bytes
               typesize=4,          # bytes per element (float32 = 4)
               clevel=5,            # compression level 0-9
               cname='lz4')        # codec: lz4, blosclz, zstd, snappy
```

---

## Cache Hierarchy (DTU XeonGold6226R)

| Level | Size | Latency |
|---|---|---|
| L1d | 32 KB / core | ~1 ns |
| L2 | 1 MB / core | ~4 ns |
| L3 | 19.7 MB shared | ~10 ns |
| RAM | GBs | ~100 ns |

Cache line = **64 bytes** = **8 float64** values.

When you access `x[0]`, the CPU automatically loads `x[0]..x[7]` into cache.
The next 7 sequential accesses are free cache hits.

```bash
lscpu | grep cache    # read cache sizes on any node
```

---

## Row vs Column Access

```python
mat = np.random.rand(N, N)   # row-major (C-order)

mat[0, :]    # ROW access: contiguous in memory → FAST (cache-friendly)
mat[:, 0]    # COL access: stride N*8 bytes    → SLOW (cache misses)
```

Real difference on large arrays: column access ~10x slower than row access.

---

## Exam Traps

| Trap | Correct |
|---|---|
| Largest stride → innermost loop | SMALLEST stride → innermost loop |
| `a.T` changes memory layout | `a.T` only reverses strides — same data |
| CPU optimal = channels first | CPU optimal = channels LAST (HWC) |
