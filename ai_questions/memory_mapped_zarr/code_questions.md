# Memory-Mapped Files & Zarr — Code-Based MCQ Practice

> Format: Each question shows np.memmap or Zarr code to analyse.
> Exam frequency: **F25 exam** — np.memmap and Zarr.

---

## Q1 — How many bytes does `y` use?

```python
x = np.memmap('big.raw', mode='r', dtype='int32', shape=(10**10,))
y = np.array(x[::100_000])
print(y.nbytes)
```

**A)** 400 bytes
**B)** 400 000 bytes
**C)** 40 000 000 bytes
**D)** 4 000 000 000 bytes

**Answer: B**

- A) Incorrect — 400 bytes would be only 100 elements × 4 bytes, but slicing 10^10 by 100 000 yields 100 000 elements.
- B) Correct — `x[::100_000]` yields 10^10 / 10^5 = 100 000 elements; `np.array(...)` copies them into RAM; 100 000 × 4 bytes = 400 000 bytes.
- C) Incorrect — 40 000 000 bytes would imply 10 million elements were selected, off by a factor of 100.
- D) Incorrect — 4 000 000 000 bytes is the full on-disk size of `x`; `np.array(x[::100_000])` does not copy the whole file.

---

## Q2 — What mode opens an existing file for modification without truncating it?

```python
mm = np.memmap('data.raw', mode='w+', shape=(1000, 1000), dtype='float64')
mm[0, :] = np.ones(1000)
del mm   # flushes to disk
```

You later want to open `data.raw` again to modify the values you wrote, without losing the existing data.

Which mode should you use?

**A)** `'r'`
**B)** `'w+'`
**C)** `'r+'`
**D)** `'c'`

**Answer: C**

- A) Incorrect — `'r'` is read-only; any attempted write will raise an error.
- B) Incorrect — `'w+'` creates or truncates the file, so all previously written data is lost.
- C) Correct — `'r+'` opens an existing file for both reading and writing without truncating it.
- D) Incorrect — `'c'` (copy-on-write) keeps writes in RAM only and never persists them to disk.

---

## Q3 — What shape does `mm.shape` report?

```python
mm = np.memmap('data.raw', mode='r')   # No dtype or shape specified
print(mm.shape)
```

The file `data.raw` was originally created as `dtype='float64'`, `shape=(100, 100)`.

**A)** `(100, 100)`
**B)** `(10000,)`
**C)** `(80000,)`
**D)** `(1250,)`

**Answer: C**

- A) Incorrect — `np.memmap` has no way to recover the original shape metadata; it is not stored in the raw file.
- B) Incorrect — `(10000,)` would be the element count if `dtype='float64'` were specified, but the default is `uint8`.
- C) Correct — default `dtype=uint8` (1 byte/element); file is 100 × 100 × 8 = 80 000 bytes → shape `(80000,)`.
- D) Incorrect — `(1250,)` has no basis; it would require a dtype of 64 bytes per element which doesn't exist.

---

## Q4 — What does `mm2[0, 0]` print?

```python
mm = np.memmap('data.raw', mode='c', dtype='float64', shape=(1000, 1000))
mm[0, 0] = 999.0
del mm

mm2 = np.memmap('data.raw', mode='r', dtype='float64', shape=(1000, 1000))
print(mm2[0, 0])
```

Assume `data.raw` originally contains all zeros.

**A)** `999.0`
**B)** `0.0`
**C)** `RuntimeError` — copy-on-write mode cannot be deleted
**D)** Undefined behaviour — the file is corrupted

**Answer: B**

- A) Incorrect — mode `'c'` never writes back to disk, so the file still contains the original zero.
- B) Correct — `'c'` (copy-on-write) keeps the write `mm[0, 0] = 999.0` in a private in-memory page only; `del mm` discards it; re-reading the file yields the original `0.0`.
- C) Incorrect — `del mm` on a memmap is valid and simply flushes/frees the mapping; it raises no error.
- D) Incorrect — the file is never modified by copy-on-write mode, so it remains intact.

---

## Q5 — Which chunk shape minimises I/O for row-by-row access?

```python
z = zarr.open('data.zarr', mode='w', shape=(512, 512),
              chunks=???, dtype='float64')

for i in range(512):
    result[i] = np.sum(z[i, :])   # reads entire row each iteration
```

**A)** `(1, 512)`
**B)** `(512, 1)`
**C)** `(64, 64)`
**D)** `(512, 512)`

**Answer: A**

- A) Correct — each chunk is exactly 1 row × 512 columns, so each `z[i, :]` loads exactly one chunk with no wasted I/O.
- B) Incorrect — `(512, 1)` is column-oriented; reading a full row spans all 512 chunks, each contributing only 1 useful element.
- C) Incorrect — `(64, 64)` causes each row access to load 8 chunks and discard 63/64 of each chunk's data.
- D) Incorrect — `(512, 512)` loads the entire array on the very first row access, wasting all subsequent I/O.

---

## Q6 — How many chunks are accessed to compute `col_sum`?

```python
z = zarr.open('data.zarr', mode='w', shape=(1000, 100_000),
              chunks=(1000, 100), dtype='float64')

col_sum = z[:, 50].sum()   # sum entire column 50
```

**A)** 1000
**B)** 100
**C)** 10
**D)** 1

**Answer: D**

- A) Incorrect — 1000 would be one chunk per row, but each chunk already spans all 1000 rows.
- B) Incorrect — 100 is the column width of a single chunk, not the number of chunks accessed.
- C) Incorrect — 10 would apply if the chunk height were 100, not 1000.
- D) Correct — each chunk is 1000 rows × 100 columns; column 50 falls in column range [0, 99] (chunk col-index 0); with chunk height = array height = 1000, exactly 1 chunk covers the full column.

---

## Q7 — Is this parallel Zarr write safe?

```python
import zarr
from multiprocessing import Pool

z = zarr.open('output.zarr', mode='w', shape=(100, 100), chunks=(10, 10))

def write_chunk(i):
    z[i*10:(i+1)*10, :] = compute_rows(i)

with Pool(10) as pool:
    pool.map(write_chunk, range(10))
```

**A)** No — multiple processes writing to the same file will cause data corruption
**B)** No — `zarr.open` cannot be used inside worker processes
**C)** Yes — each process writes to a non-overlapping chunk, which is safe
**D)** Yes — but only if a `synchronizer` is passed to `zarr.open`

**Answer: C**

- A) Incorrect — Zarr stores each chunk as an independent file; disjoint chunk writes do not race or corrupt each other.
- B) Incorrect — `zarr.open` works fine inside worker processes; there is no such restriction.
- C) Correct — each worker writes rows `[i*10:(i+1)*10, :]`, mapping to entirely separate chunk files; no shared state, no locking needed.
- D) Incorrect — a synchronizer is only necessary when two workers could write to the same chunk simultaneously, which is not the case here.

---

## Q8 — How much RAM does `small` use?

```python
big = np.memmap('huge.raw', mode='r', dtype='float32', shape=(4000, 4000))
small = big[::4, ::4]
print(f"RAM used by small: {small.nbytes} bytes")
```

**A)** 64 000 000 bytes (64 MB) — the full `big` array is loaded
**B)** 4 000 000 bytes (4 MB) — `small` is a view with a strided layout
**C)** 0 bytes — `small` is a view; no data is copied into RAM
**D)** 1 000 000 bytes (1 MB) — only the selected elements are stored

**Answer: C**

- A) Incorrect — creating a view never triggers a full load; the memmap is backed by OS virtual memory and pages are only brought in when accessed.
- B) Incorrect — `small.nbytes` reports the logical size (1000 × 1000 × 4 = 4 MB), but that is not the RAM actually allocated; no copy is made.
- C) Correct — `big[::4, ::4]` is a strided view into the memory-mapped file; creating it copies nothing into RAM.
- D) Incorrect — there is no separate 1 MB allocation; the view shares the underlying file mapping with no independent buffer.

---
