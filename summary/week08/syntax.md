# Week 8 — Memory-Mapped Files & Zarr Syntax Reference

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [np.memmap](#npmemmap)
  - [`np.memmap(filename, dtype, mode, shape)`](#npmemmapfilename-dtype-mode-shape)
  - [Mode summary](#mode-summary)
  - [Strided access — memory magic](#strided-access-memory-magic)
- [Zarr](#zarr)
  - [`zarr.open(path, mode, shape, chunks, dtype)`](#zarropenpath-mode-shape-chunks-dtype)
  - [Chunk shape for access patterns](#chunk-shape-for-access-patterns)
  - [Nested directory store (avoid too many files)](#nested-directory-store-avoid-too-many-files)
- [Pandas Chunked CSV](#pandas-chunked-csv)
  - [Chunked reader limitations](#chunked-reader-limitations)
- [Memory Budget Formula](#memory-budget-formula)
- [Exam Traps](#exam-traps)

---

## np.memmap

### `np.memmap(filename, dtype, mode, shape)`
- **What**: opens/creates a file as a NumPy array backed by disk
- **RAM used**: only pages you actually ACCESS — not the whole file
- **Gotcha**: ALWAYS specify both `dtype` AND `shape` — default dtype is `uint8` which silently misinterprets data

```python
import numpy as np

# Create new file (overwrites if exists)
mm = np.memmap('data.raw', dtype='int32', mode='w+', shape=(1000, 1000))
mm[:] = some_array
del mm      # flush to disk (also happens on gc)

# Open existing for reading
mm = np.memmap('data.raw', dtype='int32', mode='r', shape=(1000, 1000))

# Open existing for read/write
mm = np.memmap('data.raw', dtype='int32', mode='r+', shape=(1000, 1000))

# Copy-on-write (changes NOT saved to disk)
mm = np.memmap('data.raw', dtype='int32', mode='c', shape=(1000, 1000))
```

### Mode summary

| Mode | Creates file | Reads | Writes | Persists changes |
|---|---|---|---|---|
| `w+` | Yes | Yes | Yes | Yes |
| `r` | No | Yes | No | N/A |
| `r+` | No | Yes | Yes | Yes |
| `c` | No | Yes | Yes (copy) | No |

### Strided access — memory magic

```python
# 10^10 element file — only accessed pages load into RAM
big = np.memmap('huge.raw', dtype='int32', mode='r', shape=(10**10,))
y = np.array(big[::100_000])   # loads 10^5 elements = 400 KB RAM only
```

---

## Zarr

### `zarr.open(path, mode, shape, chunks, dtype)`
- **What**: chunked, compressed N-dimensional array stored on disk
- **Chunk size**: tuning parameter — too small = overhead, too large = poor parallelism

**`zarr.open` modes:**

| Mode | Store must exist? | Creates store? | Truncates existing? | Notes |
|------|:-----------------:|:--------------:|:-------------------:|-------|
| `'r'` | Yes | No | No | Read-only |
| `'r+'` | Yes | No | No | Read-write; error if missing |
| `'w'` | No | Yes | **Yes** | Create or overwrite |
| `'w-'` / `'x'` | No | Yes | No | Exclusive create — **error if exists** |
| `'a'` | No | Yes | No | **Default** — create if absent, else read-write |

> Exam trap: omitting `mode=` gives `'a'` (not `'r'` or `'w'`). Use `'w-'` in pipelines where overwriting previous results would be a bug.

```python
import zarr

# Create (overwrite if exists)
z = zarr.open('data.zarr', mode='w', shape=(1000, 1000),
              chunks=(50, 50), dtype='int32')
z[0:50, 0:50] = chunk_data    # write one chunk

# Read
z = zarr.open('data.zarr', mode='r')
data = z[100:200, :]

# Default mode='a': create if missing, open read-write if present
z = zarr.open('data.zarr', shape=(1000, 1000), chunks=(50, 50), dtype='int32')

# Exclusive create: fail if already exists (safe for write-once pipelines)
z = zarr.open('output.zarr', mode='w-', shape=(1000, 1000), chunks=(50, 50), dtype='int32')

# Multiple processes can write to DIFFERENT chunks simultaneously — no locking needed
```

### Chunk shape for access patterns

| Access pattern | Optimal chunk |
|---|---|
| Row by row `z[i, :]` | `(1, full_width)` — 1 chunk per row |
| Column by column `z[:, j]` | `(full_height, few_cols)` — 1 chunk per column |

### Nested directory store (avoid too many files)
```python
store = zarr.storage.NestedDirectoryStore('data.zarr')
z = zarr.open(store, mode='w', shape=shape, chunks=chunks)
```

---

## Pandas Chunked CSV

```python
import pandas as pd

# Process CSV too large for RAM
total = 0.0
for chunk in pd.read_csv('big.csv.zip', chunksize=100_000):
    total += chunk[chunk['parameterId'] == 'precip_past10min']['value'].sum()
print(total)
```

### Chunked reader limitations
- Forward-only iterator — cannot `len()`, random access, or re-iterate
- Re-open the file to iterate again

---

## Memory Budget Formula

```
bytes_per_row = sum(dtype_bytes for each column)
max_rows      = available_bytes / bytes_per_row
```

Example: 3 columns (uint32=4, float64=8, int16=2) = 14 bytes/row.
200 MB = 200,000,000 bytes → max_rows = 200,000,000 / 14 ≈ 14,285,714

---

## Exam Traps

| Trap | Correct |
|---|---|
| memmap loads entire file into RAM | Only accessed pages load |
| Mode `'c'` saves changes to disk | Copy-on-write — changes discarded on close |
| Zarr requires locking for parallel writes | Each chunk is independent — no locking needed |
| Default dtype when opening memmap | uint8 — always specify dtype explicitly |
| Can omit shape when opening memmap | Shape is required — file has no metadata |
