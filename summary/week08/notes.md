# Week 8 — Storing Big Data

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Week 8 — Storing Big Data](#week-8--storing-big-data)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Theory \& Concepts](#theory--concepts)
    - [The Problem: Data Too Large for RAM](#the-problem-data-too-large-for-ram)
    - [Pandas Chunking](#pandas-chunking)
    - [Memory Mapping](#memory-mapping)
      - [Motivation](#motivation)
    - [numpy.memmap](#numpymemmap)
    - [Zarr Arrays](#zarr-arrays)
    - [When to Use What](#when-to-use-what)
  - [Key Code Examples](#key-code-examples)
    - [Mandelbrot reference implementation (mandelbrotref.py)](#mandelbrot-reference-implementation-mandelbrotrefpy)
    - [Creating a memmap and writing results](#creating-a-memmap-and-writing-results)
    - [Reading a memmap (e.g., for downsampling)](#reading-a-memmap-eg-for-downsampling)
    - [Pandas chunked CSV processing](#pandas-chunked-csv-processing)
    - [CSV to chunked Parquet conversion](#csv-to-chunked-parquet-conversion)
    - [Reading chunked Parquet with column pruning](#reading-chunked-parquet-with-column-pruning)
  - [Exercise Highlights](#exercise-highlights)
    - [Exercise 1 — Dataframe Chunking (DMI precipitation data)](#exercise-1--dataframe-chunking-dmi-precipitation-data)
    - [Exercise 2 — Mandelbrot memmap](#exercise-2--mandelbrot-memmap)
    - [Exercise 3 — Mandelbrot Zarr](#exercise-3--mandelbrot-zarr)
  - [Key Takeaways](#key-takeaways)

---

## Overview

Week 8 addresses the core problem of handling data that does not fit into RAM. The lecture opens with the motivating quote: "My job was killed because it used too much memory" — and rather than simply  memory, the week introduces three principled techniques for working within memory constraints: **Pandas chunking**, **NumPy memory mapping**, and **Zarr arrays**. The exercises apply all three to the running Mandelbrot and DMI weather datasets.

---

## Theory & Concepts

### The Problem: Data Too Large for RAM

The conventional workflow is:
1. Load data from storage into RAM
2. Load data from RAM into CPU caches (via cache lines)
3. Perform computation

The fundamental constraint is that the entire dataset must fit in RAM. Memory mapping breaks this constraint.

---

### Pandas Chunking

When a CSV is too large to load entirely, `pd.read_csv` accepts a `chunksize` parameter that returns a `TextFileReader` iterator instead of a DataFrame. Each iteration yields a chunk of `chunksize` rows.

```python
import pandas as pd

# Returns a TextFileReader iterator, not a DataFrame
dfc = pd.read_csv('bighuge.csv', chunksize=100)

# Accumulate result across chunks
coordsum = 0.0
for df in dfc:
    coordsum += df['coordsx'].sum()
```

**Pro:** Only one chunk resides in memory at a time — no need to hold the full DataFrame.

**Con:** More complex code. The iterator can only be traversed once; re-reading requires re-opening the file. Random access (`dfc[1]`) and `len(dfc)` are not supported.

**Parquet chunking with PyArrow** is significantly faster. A chunked parquet file stores metadata so you can query `num_row_groups` and access any group by index:

```python
import pyarrow.parquet as pq

pf = pq.ParquetFile('bighuge.parquet')
pf.num_row_groups  # e.g., 82

g = pf.read_row_group(2)   # Read 3rd row group
df = g.to_pandas()          # Convert to Pandas
```

Additional speedup comes from **column pruning** — only loading the columns you need via the `columns` argument to `read_table` (PyArrow) or `usecols` (Pandas).

**Measured performance on DMI data (XeonGold6226R):**

| Method | Runtime | Peak Memory |
|--------|---------|-------------|
| Pandas, no chunking | 17.19 s | ~2,000 MB |
| Pandas chunked (chunksize=10,000) | 16.70 s | ~134 MB |
| Parquet chunked, all columns | 5.09 s | ~213 MB |
| Parquet chunked, relevant columns only | 1.17 s | ~140 MB |

Column pruning alone gave a **13.6x speedup** over the non-chunked Pandas baseline.

---

### Memory Mapping

#### Motivation

Normal RAM access: CPU requests address → data loaded from storage into RAM → loaded into caches → computation runs. The entire dataset must fit in RAM.

Memory mapping introduces a "fake" memory region backed directly by a file on disk. The OS maps virtual address space to a file. The CPU does not know the difference — it issues normal memory requests, and the OS pages data in on demand from the file.

**Key points:**
1. Data is loaded *on demand* from disk — only the pages actually accessed are read
2. You do not need space for all data simultaneously
3. The file can be treated as normal memory using array indexing

**Important nuance:** In reality data still passes through RAM — RAM acts as an *extra cache* for the file. Consequences:
- Reading/writing already-cached data is fast
- Reading/writing from file for the first time is slow (triggers a page fault)
- Data is evicted from RAM once memory is full, just like CPU caches

---

### numpy.memmap

`np.memmap` creates a NumPy array backed by a file. **You must always specify shape and dtype** — the file is raw binary and NumPy has no way to infer these from the file itself (the default dtype is `uint8`, which will silently misinterpret data).

**Modes:**

| Mode | Meaning |
|------|---------|
| `'w+'` | Create new file (or overwrite existing), read/write |
| `'r'` | Open existing file, read-only |
| `'r+'` | Open existing file, read and write |
| `'c'` | Copy-on-write (changes not persisted to disk) |

**Critical:** Writes to a `'w+'` or `'r+'` memmap are immediately reflected in the underlying file. There is no separate flush step needed (changes are written out by the OS), but this means accidental writes corrupt the saved data.

```python
import numpy as np

# Create a new 10x10 memory-mapped array
mm = np.memmap('tmp.raw', mode='w+', shape=(10, 10), dtype='float64')
mm[:5, :5] = 1   # This change is also written to disk

# Open for read-only (writes will raise ValueError)
mm = np.memmap('tmp.raw', mode='r')  # Default dtype=uint8 — wrong!

# Open correctly for read/write
mm = np.memmap('tmp.raw', mode='r+', dtype='float64', shape=(10, 10))
# Now behaves like a normal NumPy array, but changes go to disk
```

**Gotcha illustrated in lecture:** Opening a `float64` file without specifying dtype defaults to `uint8`. A 10x10 float64 array is 800 bytes, so the memmap shows shape `(800,)` and garbage data. Always specify both `dtype` and `shape`.

---

### Zarr Arrays

Zarr is a file storage format for **chunked, compressed, N-dimensional arrays**, based on an open specification. It is a modern alternative to HDF5.

The array is divided into rectangular chunks. Each chunk is stored as a separate file and can be independently compressed. This enables:
- Partial reads (only load the chunks you need)
- Parallel writes (different processes can write different chunks simultaneously without locking)
- Significant storage savings through compression, especially in regions with uniform values

**Zarr vs memmap trade-off:**

| Aspect | memmap (.raw) | Zarr |
|--------|--------------|------|
| Format | Raw binary, no metadata | Chunked + compressed |
| Storage (1000x1000 Mandelbrot) | 5.3 MB | 2.4 MB (chunk=100) |
| Parallel write runtime (32 cores) | 0.8 s | 1.1 s (chunk=50) |
| Chunk size effect on runtime | N/A | Optimal chunk exists; too small = overhead, too large = poor parallelism |
| Chunk size effect on storage | N/A | Larger chunks compress better |

**Chunk size tuning (1000x1000 Mandelbrot, 24 cores):**

| Chunk size | Runtime | Stored size |
|------------|---------|------------|
| 10x10 | 24.2 s | 285 MB |
| 25x25 | 3.6 s | 38 MB |
| 50x50 | 1.1 s | 9.5 MB |
| 100x100 | 1.3 s | 2.4 MB |
| 200x200 | 4.3 s | 656 KB |

Fastest execution at chunk=50; smallest storage at chunk=200. There is a sweet spot because: too-small chunks create excessive overhead, too-large chunks cannot be parallelized effectively.

---

### When to Use What

| Technique | Use Case |
|-----------|----------|
| `pd.read_csv(chunksize=N)` | CSV too large for RAM, sequential aggregation |
| Parquet + PyArrow chunking | Large tabular data, fast random chunk access, column pruning |
| `np.memmap` | Large numerical arrays, disk-backed, parallel writes, raw format |
| Zarr | Large numerical arrays, compressed storage, parallel writes, interoperable |
| Regular in-memory array | Small data, single process, no memory pressure |

---

## Key Code Examples

### Mandelbrot reference implementation (mandelbrotref.py)

```python
import numpy as np

def mandelbrot_escape_time(c):
    z = 0
    for i in range(100):
        z = z**2 + c
        if np.abs(z) > 2:
            return i
    return 100


def mandelbrot(size):
    xpts, ypts = np.meshgrid(
        # Add 1 to size to make it compatible with previous weeks
        np.linspace(-2, 2, size+1)[:-1],
        np.linspace(-2, 2, size+1)[:-1],
    )
    points = 1j * xpts.ravel() + ypts.ravel()
    mandelbrot = np.array([mandelbrot_escape_time(c) for c in points])
    mandelbrot = mandelbrot.reshape((size, size))
    return mandelbrot
```

### Creating a memmap and writing results

```python
import numpy as np

# Create a new memmap (w+ = create/overwrite)
mm = np.memmap('mandelbrot.raw', mode='w+', shape=(1000, 1000), dtype='int32')

# Write computed data — changes go directly to disk
mm[:] = mandelbrot(1000)

# No explicit flush needed; OS handles it when mm goes out of scope
del mm
```

### Reading a memmap (e.g., for downsampling)

```python
import numpy as np

# Must specify exact dtype and shape used when writing
mm = np.memmap('mandelbrot.raw', mode='r', dtype='int32', shape=(4000, 4000))

# Downsampling: only reads the pages touched by this slice
step = 4
downsampled = mm[::step, ::step]
```

### Pandas chunked CSV processing

```python
import pandas as pd

dfc = pd.read_csv('/path/to/file.csv.zip', chunksize=100_000)

total = 0.0
for chunk in dfc:
    total += chunk[chunk['parameterId'] == 'precip_past10min']['value'].sum()

print(total)
```

### CSV to chunked Parquet conversion

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

df_chunks = pd.read_csv('data.csv.zip', chunksize=100_000)

writer = None
for chunk in df_chunks:
    table = pa.Table.from_pandas(chunk)
    if writer is None:
        writer = pq.ParquetWriter('data.parquet', schema=table.schema)
    writer.write_table(table)
writer.close()
```

### Reading chunked Parquet with column pruning

```python
import pyarrow.parquet as pq

pf = pq.ParquetFile('data.parquet')
total = 0.0
for i in range(pf.num_row_groups):
    group = pf.read_row_group(i, columns=['parameterId', 'value'])
    df = group.to_pandas()
    total += df[df['parameterId'] == 'precip_past10min']['value'].sum()

print(total)
```

---

## Exercise Highlights

### Exercise 1 — Dataframe Chunking (DMI precipitation data)

Process a large CSV of weather observations in chunks without loading the full file. Key steps:
1. Use `pd.read_csv(..., chunksize=N)` and accumulate the precipitation sum across chunks
2. Benchmark chunk sizes 1,000 / 10,000 / 100,000 / 1,000,000 — observe that chunk size 1,000 is slow (overhead), while 10,000+ matches the non-chunked runtime with a fraction of the memory
3. Convert the CSV to a chunked Parquet file using PyArrow's `ParquetWriter`
4. Re-implement using `pq.ParquetFile` to iterate row groups — achieves ~3x speedup over Pandas
5. Add column pruning (only load `parameterId` and `value`) — final result is **13.6x faster** than the Pandas baseline and uses ~93% less memory

### Exercise 2 — Mandelbrot memmap

Extend the weeks 5/6 Mandelbrot parallelization to write results directly into a memory-mapped array:
1. Create an NxN memmap with `mode='w+'`; compute and fill it sequentially
2. Parallelize: each process writes its assigned row slice directly into the shared memmap — achieves near-perfect speed-up (0.8 s at 32 processes vs 0.9 s for in-memory parallel version)
3. Implement downsampling: open a large memmap (4000x4000) with `mode='r'`, slice every nth row and column (`mm[::n, ::n]`), save as PNG. Observe that memory and runtime both decrease as step size increases — confirming that only accessed pages are loaded, not the whole array.

### Exercise 3 — Mandelbrot Zarr

Replace the memmap output with a Zarr array:
1. Create a Zarr array with configurable CxC chunk size; fill chunks one at a time
2. Parallelize: each process fills one chunk independently (no locking needed)
3. Sweep chunk sizes 10/25/50/100/200 — fastest at 50x50 (1.1 s); smallest storage at 200x200 (656 KB vs 5.3 MB for raw memmap)
4. Larger chunks compress better because Mandelbrot values are uniform away from the boundary

---

## Key Takeaways

- **Chunking** lets you process arbitrarily large tabular data with bounded memory use, at the cost of sequential-only access and more complex code. Parquet + column pruning can give an order-of-magnitude speedup over Pandas CSV chunking.
- **Memory mapping** is the OS-level mechanism that lets a file act as an extension of RAM. Only pages that are actually accessed are loaded, so you can work with arrays much larger than RAM. NumPy's `np.memmap` exposes this as a standard array — but you must always specify `dtype` and `shape` when opening an existing file.
- **Writes to a memmap go directly to disk.** This makes memmaps suitable for parallel output (multiple processes can write different regions simultaneously) but also means accidental writes corrupt the file.
- **Zarr** adds chunking and compression on top of the memmap concept. It trades a small runtime overhead for significant storage savings and better interoperability. For the Mandelbrot dataset, chunk size 50x50 was optimal for speed; larger chunks gave smaller files.
- **RAM acts as a cache for memory-mapped files.** Repeatedly accessing the same region of a memmap is fast (served from RAM); first access is slow (page fault from disk). This is why the downsampling experiment shows memory and runtime both scaling with the number of pages read.
- **Chunk size is a tuning parameter** for both Pandas chunking (affects overhead vs memory trade-off) and Zarr (affects parallelism granularity vs compression efficiency).
