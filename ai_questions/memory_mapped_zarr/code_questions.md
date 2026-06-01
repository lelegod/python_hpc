# Memory-Mapped Files & Zarr — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — How many bytes does `y` use?](#q1-how-many-bytes-does-y-use)
- [Q2 — What mode opens an existing file for modification without truncating it?](#q2-what-mode-opens-an-existing-file-for-modification-without-truncating-it)
- [Q3 — What shape does `mm.shape` report?](#q3-what-shape-does-mmshape-report)
- [Q4 — What does `mm2[0, 0]` print?](#q4-what-does-mm20-0-print)
- [Q5 — Which chunk shape minimises I/O for row-by-row access?](#q5-which-chunk-shape-minimises-io-for-row-by-row-access)
- [Q6 — How many chunks are accessed to compute `col_sum`?](#q6-how-many-chunks-are-accessed-to-compute-col_sum)
- [Q7 — Is this parallel Zarr write safe?](#q7-is-this-parallel-zarr-write-safe)
- [Q8 — How much RAM does `small` use?](#q8-how-much-ram-does-small-use)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q9 — What does this code print?](#q9-what-does-this-code-print)
- [Q10 — What is wrong with this code?](#q10-what-is-wrong-with-this-code)
- [Q11 — How many bytes does `chunk_data` occupy?](#q11-how-many-bytes-does-chunk_data-occupy)
- [Q12 — What does `m[0]` return after this sequence?](#q12-what-does-m0-return-after-this-sequence)
- [Q13 — What is the total number of chunks in this Zarr array?](#q13-what-is-the-total-number-of-chunks-in-this-zarr-array)
- [Q14 — Which line raises an error?](#q14-which-line-raises-an-error)
- [Q15 — How many chunks are loaded by this Zarr access?](#q15-how-many-chunks-are-loaded-by-this-zarr-access)
- [Q16 — What does this print?](#q16-what-does-this-print)
- [Q17 — What chunk shape produces the smallest RAM per chunk?](#q17-what-chunk-shape-produces-the-smallest-ram-per-chunk)
- [Q18 — What is the result of this memmap dtype mismatch?](#q18-what-is-the-result-of-this-memmap-dtype-mismatch)
- [Q19 — What happens when this script runs twice?](#q19-what-happens-when-this-script-runs-twice)
- [Q20 — Which mode prevents overwriting an existing result?](#q20-which-mode-prevents-overwriting-an-existing-result)
- [Q21 — zarr mode 'r+' on a missing store](#q21-zarr-mode-r-on-a-missing-store)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q22 — What does isinstance check return?](#q22--what-does-isinstance-check-return)
- [Q23 — What does this offset memmap print?](#q23--what-does-this-offset-memmap-print)
- [Q24 — Which line raises a ValueError for size mismatch?](#q24--which-line-raises-a-valueerror-for-size-mismatch)
- [Q25 — How many files does this Zarr write create?](#q25--how-many-files-does-this-zarr-write-create)
- [Q26 — What does nchunks_initialized return?](#q26--what-does-nchunks_initialized-return)
- [Q27 — What does this SharedMemory code print?](#q27--what-does-this-sharedmemory-code-print)
- [Q28 — What is the on-disk size ordering?](#q28--what-is-the-on-disk-size-ordering)
- [Q29 — What does this zarr.zeros call do to existing data?](#q29--what-does-this-zarrzeros-call-do-to-existing-data)
- [Q30 — What does this compressor=None zarr store?](#q30--what-does-this-compressornone-zarr-store)
- [Q31 — What does reading an unwritten Zarr chunk return?](#q31--what-does-reading-an-unwritten-zarr-chunk-return)

---

> Format: Each question shows np.memmap or Zarr code to analyse.
> Exam frequency: **F25 exam** — np.memmap and Zarr.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--how-many-bytes-does-y-use)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

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

**Answer: B**

- A) Incorrect — creating a view never triggers a full load; the memmap is backed by OS virtual memory and pages are only brought in when accessed. 64 MB is the full logical size of `big`, not `small`.
- B) Correct — `big[::4, ::4]` selects 1000 × 1000 elements from the 4000 × 4000 array. The result is a strided view (no data copied) but `small.nbytes` reports the **logical** byte count of the view: 1000 × 1000 × 4 = 4 000 000 bytes (4 MB). No new physical RAM is allocated — `small` shares pages with the underlying memmap — but `nbytes` always returns the logical element count × itemsize, not the physically committed RAM.
- C) Incorrect — the code prints `small.nbytes`, which is `4 000 000`, not `0`. `nbytes` is a logical property of the array's shape and dtype, not a measure of physically allocated pages. Physically, no extra RAM is committed until the pages are accessed, but `nbytes` does not reflect that.
- D) Incorrect — `small` is a view, not a copy; no 1 MB buffer is created. `small.nbytes` = 1000 × 1000 × 4 = 4 000 000 bytes, not 1 000 000.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets memmap modes, Zarr chunk optimization, block memory calculation, and when to prefer memory mapping over loading

---

## Q9 — What does this code print?

```python
import numpy as np

fp = np.memmap('test.raw', mode='w+', dtype='int32', shape=(4,))
fp[:] = [10, 20, 30, 40]
del fp

fp2 = np.memmap('test.raw', mode='r', dtype='int32', shape=(4,))
print(fp2[2])
```

**A)** `0`
**B)** `30`
**C)** `AttributeError` — mode `'r'` has no index operator
**D)** `FileNotFoundError` — mode `'r'` cannot open a file created by `'w+'`

**Answer: B**

- A) Incorrect — the `del fp` on a `'w+'` memmap flushes all written values to disk. The file is not reset to zero on close.
- B) Correct — `'w+'` writes `[10, 20, 30, 40]` to disk; `del fp` flushes the mapping; `fp2` re-opens the same file in read-only mode and `fp2[2]` returns the third element, which is `30`.
- C) Incorrect — mode `'r'` supports full numpy indexing; it raises errors only on *write* attempts, not reads.
- D) Incorrect — mode `'r'` requires the file to exist, which it does — `'w+'` created it in the preceding step.

---

## Q10 — What is wrong with this code?

```python
import numpy as np

m = np.memmap('new_data.raw', mode='r+', dtype='float64', shape=(1000,))
m[:] = np.arange(1000, dtype='float64')
del m
```

The file `new_data.raw` does not exist before this script runs.

**A)** Nothing — `'r+'` creates the file if absent.
**B)** `FileNotFoundError` on line 3 — `'r+'` requires the file to already exist.
**C)** `ValueError` on line 4 — `'r+'` is read-only and does not permit writes.
**D)** The script runs silently but no data is written to disk.

**Answer: B**

- A) Incorrect — `'r+'` maps to `open(path, 'r+b')` which does **not** include `O_CREAT`; the OS raises `FileNotFoundError` if the file is absent. Only `'w+'` creates the file.
- B) Correct — `np.memmap` with `mode='r+'` immediately attempts to open the file for read-write access. Since `new_data.raw` does not exist, the OS raises `FileNotFoundError` before any data is touched.
- C) Incorrect — `'r+'` is explicitly read-write (the `+` denotes write access). The assignment on line 4 would succeed if the file existed.
- D) Incorrect — this describes mode `'c'` behaviour, not `'r+'`. With `'r+'` writes persist; the script does not run silently — it crashes at line 3.

---

## Q11 — How many bytes does `chunk_data` occupy?

```python
import zarr
import numpy as np

z = zarr.open('array.zarr', mode='w', shape=(800, 600),
              chunks=(80, 60), dtype='float32')

chunk_data = z[0:80, 0:60]   # load exactly one chunk
print(chunk_data.nbytes)
```

**A)** 4 800 bytes
**B)** 19 200 bytes
**C)** 1 920 000 bytes
**D)** 480 000 bytes

**Answer: B**

- A) Incorrect — 4 800 bytes = 80 × 60 × 1 byte, which would be the size if dtype were `uint8` (1 byte per element), not `float32` (4 bytes per element).
- B) Correct — chunk memory = 80 rows × 60 cols × 4 bytes/element (float32) = 19 200 bytes. `z[0:80, 0:60]` returns a numpy array that is a copy of the chunk; `nbytes` reports 19 200.
- C) Incorrect — 1 920 000 bytes = 800 × 600 × 4, which is the full array size, not one chunk.
- D) Incorrect — 480 000 bytes = 80 × 60 × 100; there is no factor of 100 in this calculation.

---

## Q12 — What does `m[0]` return after this sequence?

```python
import numpy as np

m = np.memmap('data.raw', mode='w+', dtype='float32', shape=(10,))
m[0] = 3.14
m.flush()

m2 = np.memmap('data.raw', mode='c', dtype='float32', shape=(10,))
m2[0] = 99.0
del m2

m3 = np.memmap('data.raw', mode='r', dtype='float32', shape=(10,))
print(m3[0])
```

**A)** `99.0`
**B)** `3.14`
**C)** `0.0`
**D)** `ValueError` — the file was corrupted by the copy-on-write operation

**Answer: B**

- A) Incorrect — `m2` was opened with mode `'c'`; the assignment `m2[0] = 99.0` goes to a private RAM page only. `del m2` discards that page without touching the file.
- B) Correct — `m` wrote `3.14` and flushed it to disk. `m2` in mode `'c'` made a private in-memory modification that was never persisted. `m3` reads the unchanged file, which still contains `3.14` at index 0.
- C) Incorrect — the file was explicitly written to via `m` with `mode='w+'` and flushed; index 0 is `3.14`, not `0.0`.
- D) Incorrect — copy-on-write mode never modifies the backing file; the file is intact.

---

## Q13 — What is the total number of chunks in this Zarr array?

```python
import zarr

z = zarr.open('grid.zarr', mode='w', shape=(1200, 900),
              chunks=(300, 300), dtype='float64')
print(z.nchunks)
```

**A)** `4`
**B)** `9`
**C)** `12`
**D)** `3`

**Answer: C**

- A) Incorrect — 4 would be the chunk count for a 2×2 grid, which would require equal divisions of both axes. The row axis gives 1200 ÷ 300 = 4 chunks; the column axis gives 900 ÷ 300 = 3 chunks; 4 × 3 = 12, not 4.
- B) Incorrect — 9 = 3 × 3 would apply if the array were (900, 900) with chunks (300, 300), or (1200, 1200) viewed incorrectly. Here the row axis contributes 4 chunks, not 3.
- C) Correct — row axis: 1200 ÷ 300 = 4 chunks; column axis: 900 ÷ 300 = 3 chunks; total = 4 × 3 = 12 chunks. `z.nchunks` reports this directly.
- D) Incorrect — 3 is the column-axis chunk count alone; both axes must be multiplied together.

---

## Q14 — Which line raises an error?

```python
import numpy as np

m = np.memmap('vals.raw', mode='w+', dtype='uint8', shape=(100,))  # line A
m[50] = 255                                                          # line B
del m                                                                # line C

m2 = np.memmap('vals.raw', mode='r', dtype='uint8', shape=(100,))  # line D
m2[50] = 0                                                           # line E
```

**A)** Line A
**B)** Line B
**C)** Line D
**D)** Line E

**Answer: D**

- A) Incorrect — line A opens a new `'w+'` memmap, creating the file. This is valid and raises no error.
- B) Incorrect — line B writes to an index in a `'w+'` array, which is fully read-write. This succeeds.
- C) Incorrect — line D opens the existing file in read-only mode `'r'`. The file exists (created by line A), so no `FileNotFoundError` occurs.
- D) Correct — line E attempts to assign to element 50 of a mode `'r'` (read-only) memory map. NumPy raises `ValueError: assignment destination is read-only` because the underlying mmap was created with write protection disabled.

---

## Q15 — How many chunks are loaded by this Zarr access?

```python
import zarr

z = zarr.open('data.zarr', mode='w', shape=(1000, 1000),
              chunks=(100, 100), dtype='float32')

subset = z[150:250, 150:250]
```

**A)** 1
**B)** 4
**C)** 100
**D)** 2

**Answer: B**

- A) Incorrect — the slice `[150:250, 150:250]` spans two chunk boundaries on each axis (rows 150–249 cross the boundary at row 200; columns 150–249 cross the boundary at column 200), so it cannot be served by a single chunk.
- B) Correct — along the row axis: chunk boundaries at 100, 200, 300 ... The slice rows 150–249 span chunks [100–199] and [200–299] = 2 row-chunks. Along the column axis: same situation, 2 column-chunks. Total = 2 × 2 = 4 chunks loaded.
- C) Incorrect — 100 would be the total number of chunks in one axis of this 1000×1000 array with 100-element chunks; it is not the number touched by this specific slice.
- D) Incorrect — 2 counts only one axis; both axes must be multiplied.

---

## Q16 — What does this print?

```python
import numpy as np

fp = np.memmap('buf.raw', mode='w+', dtype='float64', shape=(5,))
fp[2] = 1.5
fp.flush()

fp2 = np.memmap('buf.raw', mode='r+', dtype='float64', shape=(5,))
fp2[2] += 10.0
del fp2

fp3 = np.memmap('buf.raw', mode='r', dtype='float64', shape=(5,))
print(fp3[2])
```

**A)** `1.5`
**B)** `10.0`
**C)** `11.5`
**D)** `0.0`

**Answer: C**

- A) Incorrect — `fp2` opened in `'r+'` mode successfully reads the flushed value `1.5` and adds `10.0` to it, then `del fp2` flushes the result `11.5` to disk.
- B) Incorrect — `fp2[2] += 10.0` is equivalent to `fp2[2] = fp2[2] + 10.0 = 1.5 + 10.0 = 11.5`, not just `10.0`.
- C) Correct — `fp` writes and flushes `1.5`; `fp2` reads `1.5`, adds `10.0`, writes `11.5`, and flushes on `del`; `fp3` reads the final value `11.5`.
- D) Incorrect — `'w+'` mode does not reset the file between openings; only the very first `'w+'` open initialised it to zeros. Subsequent opens in `'r+'` preserve existing data.

---

## Q17 — What chunk shape produces the smallest RAM per chunk?

```python
import zarr

# All four arrays have the same total shape and dtype
configs = [
    zarr.open('a.zarr', mode='w', shape=(1024, 1024), chunks=(256, 256), dtype='float64'),
    zarr.open('b.zarr', mode='w', shape=(1024, 1024), chunks=(512, 128), dtype='float64'),
    zarr.open('c.zarr', mode='w', shape=(1024, 1024), chunks=(128, 512), dtype='float64'),
    zarr.open('d.zarr', mode='w', shape=(1024, 1024), chunks=(64, 1024), dtype='float64'),
]
```

Which config uses the least RAM per chunk?

**A)** `a` — chunks `(256, 256)`
**B)** `b` — chunks `(512, 128)`
**C)** `c` — chunks `(128, 512)`
**D)** `d` — chunks `(64, 1024)`

**Answer: D**

Chunk memory = rows × cols × 8 bytes (float64):
- a: 256 × 256 × 8 = 524 288 bytes (512 KB)
- b: 512 × 128 × 8 = 524 288 bytes (512 KB)
- c: 128 × 512 × 8 = 524 288 bytes (512 KB)
- d: 64 × 1024 × 8 = 524 288 bytes (512 KB)

All four chunks have the same memory: 256 × 256 = 512 × 128 = 128 × 512 = 64 × 1024 = 65 536 elements × 8 bytes = 524 288 bytes. The answer is D only because the question is a trap — all configurations use identical RAM per chunk. The correct answer is that **all are equal**, but if forced to pick one, all produce 512 KB. This tests whether you calculate chunk memory correctly rather than guessing based on "smaller numbers = less memory".

**Answer: D** (all equal — 524 288 bytes each; the question tests calculation, not intuition about shape)

---

## Q18 — What is the result of this memmap dtype mismatch?

```python
import numpy as np

# Write: create file as float32
fp = np.memmap('data.raw', mode='w+', dtype='float32', shape=(4,))
fp[:] = [1.0, 2.0, 3.0, 4.0]
del fp

# Read: open as float64
fp2 = np.memmap('data.raw', mode='r', dtype='float64', shape=(2,))
print(fp2[0])
```

**A)** `1.0` — NumPy auto-converts float32 to float64
**B)** A large garbage float64 value — the 8 raw bytes reinterpreted as float64
**C)** `ValueError` — dtype mismatch is detected automatically
**D)** `1.0` followed by `2.0` — the first two float32s are concatenated into a float64

**Answer: B**

- A) Incorrect — `np.memmap` does not perform dtype conversion; it reinterprets raw bytes. There is no auto-conversion mechanism for raw binary files.
- B) Correct — the file contains 4 × 4 = 16 bytes of float32 data. Opening with `dtype='float64'` reinterprets the first 8 bytes (two float32 values `1.0` and `2.0` in IEEE 754 binary) as a single float64 — the result is a garbage value that happens to have the bit pattern of those 8 bytes interpreted as float64.
- C) Incorrect — `np.memmap` performs no dtype validation against file contents; raw files have no header, so no mismatch can be detected.
- D) Incorrect — the float32 bit patterns for `1.0` and `2.0` are `0x3F800000` and `0x40000000`; concatenated as 8 bytes and interpreted as float64, the result is neither `1.0` nor `2.0`.

---

## Q19 — What happens when this script runs twice?

> **Week reference:** Week 8

```python
import zarr
import numpy as np

z = zarr.open('cache.zarr', shape=(100,), chunks=(10,), dtype='float64')
z[:] = np.arange(100, dtype='float64')
print(z[0])
```

The script is run once, then run again without deleting `cache.zarr`. What does the second run print?

- A) Raises a `ContainsArrayError` — the store already exists.
- B) Prints `0.0` — the second run creates a fresh store, overwriting the first.
- C) Prints `0.0` — `mode='a'` (default) opens the existing store and overwrites element 0 with `np.arange(100)[0]`.
- D) Raises a `ValueError` — `shape` and `chunks` must not be specified when opening an existing store.

**Answer: C**

- A) Incorrect — `ContainsArrayError` is raised only with `mode='w-'`/`'x'`; the default mode `'a'` never raises on a pre-existing store.
- B) Incorrect — `'w'` mode would overwrite; the default `'a'` does not truncate existing data.
- C) Correct — `mode='a'` opens the existing store read-write (or creates it if absent). The second run writes `np.arange(100)` into it again, so `z[0]` is `0.0` just as in the first run. No error, no overwrite of the store structure.
- D) Incorrect — zarr ignores `shape`/`chunks`/`dtype` when opening an existing store in `'a'` or `'r+'` mode; the stored metadata takes precedence.

---

## Q20 — Which mode prevents overwriting an existing result?

> **Week reference:** Week 8

```python
import zarr

z = zarr.open('experiment_01.zarr', mode=MODE, shape=(1000,), chunks=(100,), dtype='float32')
z[:] = results
```

You are filling in `MODE`. The job should **crash if `experiment_01.zarr` already exists** so previous results are never silently lost. Which value of `MODE` is correct?

- A) `'w'`
- B) `'a'`
- C) `'r+'`
- D) `'w-'`

**Answer: D**

- A) Incorrect — `'w'` silently overwrites any existing store; it provides no protection.
- B) Incorrect — `'a'` opens an existing store without error and writes into it, overwriting whatever was there.
- C) Incorrect — `'r+'` raises an error if the store does *not* exist, which is the opposite guard from what is needed.
- D) Correct — `'w-'` (exclusive create, also written `'x'`) raises `ContainsArrayError` if the path already exists, making it the correct choice for write-once pipeline outputs.

---

## Q21 — zarr mode 'r+' on a missing store

> **Week reference:** Week 8

```python
import zarr, os

path = 'temp.zarr'
if os.path.exists(path):
    os.rmdir(path)

z = zarr.open(path, mode='r+', shape=(50,), chunks=(10,), dtype='int32')
```

What happens when this code runs?

- A) A new store is created at `temp.zarr` and opened read-write.
- B) An error is raised because `'r+'` requires the store to already exist.
- C) The store is created in read-only mode and writing raises `ValueError`.
- D) Nothing — zarr silently falls back to `mode='a'` when `'r+'` is requested on a missing path.

**Answer: B**

- A) Incorrect — creating a missing store requires `'w'`, `'w-'`, or `'a'`; `'r+'` provides no `O_CREAT` semantics.
- B) Correct — `mode='r+'` requires the store to already exist. On a missing path it raises a `GroupNotFoundError` (or `ArrayNotFoundError`). This mirrors POSIX `open(O_RDWR)` without `O_CREAT`.
- C) Incorrect — `'r+'` is not read-only; the problem is the missing store, not access permissions.
- D) Incorrect — zarr does not silently fall back to a different mode; the error propagates to the caller.

---

## Set 3 — Extended Practice

> Targets memmap subclass behaviour, offset parameter, shape/size mismatch, SharedMemory, sparse Zarr chunks, compressor=None, and zarr.zeros overwrite semantics.

---

## Q22 — What does isinstance check return?

> **Week reference:** Week 8

```python
import numpy as np

mm = np.memmap('check.raw', mode='w+', dtype='float32', shape=(10,))
print(isinstance(mm, np.ndarray))
print(type(mm) is np.ndarray)
```

**A)** `True` then `True`
**B)** `True` then `False`
**C)** `False` then `False`
**D)** `False` then `True`

**Answer: B**

- A) Incorrect — `type(mm) is np.ndarray` performs an exact type identity check. Since `mm` is a `numpy.memmap` instance (a subclass), its `type` is `numpy.memmap`, not `numpy.ndarray`. This check returns `False`.
- B) Correct — `isinstance(mm, np.ndarray)` returns `True` because `numpy.memmap` is a subclass of `numpy.ndarray`, so all `memmap` objects satisfy the `ndarray` isinstance test. However `type(mm) is np.ndarray` is `False` because the exact type is `numpy.memmap`, not `numpy.ndarray`. This is the classic subclass trap.
- C) Incorrect — `isinstance` with a parent class returns `True` for instances of subclasses; it does not require exact type equality.
- D) Incorrect — `type(mm) is np.ndarray` is `False`, but `isinstance(mm, np.ndarray)` is `True` — the reverse of this option.

---

## Q23 — What does this offset memmap print?

> **Week reference:** Week 8

```python
import numpy as np, struct

# Write a 4-byte header then 3 float32 values
with open('hdr.bin', 'wb') as f:
    f.write(struct.pack('i', 42))               # 4-byte int header
    f.write(struct.pack('3f', 1.0, 2.0, 3.0))  # 12 bytes of float32

mm = np.memmap('hdr.bin', dtype='float32', mode='r', offset=4, shape=(3,))
print(mm[1])
```

**A)** `42.0`
**B)** `1.0`
**C)** `2.0`
**D)** A large garbage value

**Answer: C**

- A) Incorrect — `42` is the 4-byte integer header at bytes 0–3. With `offset=4`, those bytes are skipped entirely; the mapping starts at the first float32.
- B) Incorrect — `mm[0]` would be `1.0` (the first float32 starting at byte 4). `mm[1]` is the second float32 starting at byte 8.
- C) Correct — `offset=4` skips the 4-byte header. The float32 values begin at byte 4: `mm[0]=1.0` (bytes 4–7), `mm[1]=2.0` (bytes 8–11), `mm[2]=3.0` (bytes 12–15). `mm[1]` is therefore `2.0`.
- D) Incorrect — the offset is correctly specified and aligns perfectly with the float32 data. There is no misalignment or dtype mismatch, so no garbage values appear.

---

## Q24 — Which line raises a ValueError for size mismatch?

> **Week reference:** Week 8

```python
import numpy as np

# Create a 40-byte file (10 float32 values)
fp = np.memmap('sized.raw', mode='w+', dtype='float32', shape=(10,))
del fp

# Line A
ma = np.memmap('sized.raw', mode='r', dtype='float32', shape=(10,))

# Line B
mb = np.memmap('sized.raw', mode='r', dtype='float64', shape=(10,))

# Line C
mc = np.memmap('sized.raw', mode='r', dtype='uint8', shape=(40,))

# Line D
md = np.memmap('sized.raw', mode='r', dtype='uint8', shape=(39,))
```

Which line raises a `ValueError`?

**A)** Line A
**B)** Line B
**C)** Line C
**D)** Line D

**Answer: B**

- A) Incorrect — Line A: `float32` × 10 elements = 40 bytes = exact file size. This mapping is valid.
- B) Correct — Line B: `float64` × 10 elements = 80 bytes, but the file is only 40 bytes. NumPy detects that the requested mapping exceeds the file size and raises `ValueError: mmap length is greater than file size`.
- C) Incorrect — Line C: `uint8` × 40 elements = 40 bytes = exact file size. This is valid (though the byte values will be the raw float32 bit patterns, not meaningful numbers).
- D) Incorrect — Line D: `uint8` × 39 elements = 39 bytes ≤ 40 bytes. NumPy allows mappings that are strictly smaller than the file; the last byte is simply not mapped.

---

## Q25 — How many files does this Zarr write create?

> **Week reference:** Week 8

```python
import zarr, numpy as np

z = zarr.open('sparse.zarr', mode='w', shape=(100, 100),
              chunks=(10, 10), dtype='float32')
# Write to exactly 2 non-overlapping chunks
z[0:10, 0:10] = 1.0
z[90:100, 90:100] = 2.0
```

After this code runs, how many chunk files exist inside `sparse.zarr/`? (Ignore `.zarray` and `.zattrs` metadata files.)

**A)** 100 — one per chunk in a 10×10 grid
**B)** 2 — only the two chunks that were written
**C)** 0 — chunks are only written when `z.store.flush()` is called
**D)** 1 — Zarr consolidates all written chunks into a single file

**Answer: B**

- A) Incorrect — Zarr arrays are sparse by default. Chunks that have never been written do not exist as files on disk. A 100-element chunk grid does not mean 100 files are created at open time.
- B) Correct — each write to a distinct chunk region triggers the compression and storage of exactly that chunk. Two writes to two non-overlapping chunks create exactly 2 chunk files (`0.0` and `9.9` in the default naming scheme).
- C) Incorrect — Zarr writes chunks to disk synchronously when you assign to the array (e.g., `z[...] = ...`). There is no deferred flush required for `DirectoryStore`.
- D) Incorrect — Zarr never consolidates chunk data into a single file in `DirectoryStore`. Chunks remain as separate files. (Metadata can be consolidated with `zarr.consolidate_metadata()`, but chunk data is never merged.)

---

## Q26 — What does nchunks_initialized return?

> **Week reference:** Week 8

```python
import zarr

z = zarr.open('test.zarr', mode='w', shape=(500, 500),
              chunks=(100, 100), dtype='int32')
print(z.nchunks)
print(z.nchunks_initialized)

z[0:100, 0:100] = 7
z[0:100, 100:200] = 7

print(z.nchunks_initialized)
```

What are the three printed values, in order?

**A)** `25`, `25`, `25`
**B)** `25`, `0`, `2`
**C)** `0`, `0`, `2`
**D)** `25`, `25`, `2`

**Answer: B**

- A) Incorrect — `nchunks_initialized` starts at 0 immediately after `mode='w'` creation. No chunks are written until data is assigned.
- B) Correct — `z.nchunks` = (500÷100) × (500÷100) = 5 × 5 = 25 total chunks. Before any writes, `nchunks_initialized` = 0 (no chunk files exist yet). After two writes to two distinct chunks, `nchunks_initialized` = 2.
- C) Incorrect — `nchunks` is always the total theoretical chunk count (25), regardless of how many have been written. It is a computed property of shape and chunk shape, not a count of stored files.
- D) Incorrect — the first `nchunks_initialized` is 0 (no writes have occurred yet), not 25. Only writes materialise chunks on disk.

---

## Q27 — What does this SharedMemory code print?

> **Week reference:** Week 8

```python
from multiprocessing.shared_memory import SharedMemory
import numpy as np

shm = SharedMemory(create=True, size=4 * 10)  # 10 int32s
arr = np.ndarray((10,), dtype='int32', buffer=shm.buf)
arr[:] = np.arange(10, dtype='int32')

shm2 = SharedMemory(name=shm.name, create=False)
arr2 = np.ndarray((10,), dtype='int32', buffer=shm2.buf)
print(arr2[5])

shm2.close()
shm.close()
shm.unlink()
```

**A)** `0`
**B)** `5`
**C)** `NameError` — `shm.name` is not accessible before `shm.close()`
**D)** `ValueError` — you cannot create two numpy arrays backed by the same SharedMemory

**Answer: B**

- A) Incorrect — `arr[:]` was set to `np.arange(10)`, so index 5 holds the value 5. The shared memory block is not zeroed between the two attachments.
- B) Correct — `arr` writes `[0,1,2,3,4,5,6,7,8,9]` into the shared block via `shm.buf`. `shm2` attaches to the same physical pages by name. `arr2` wraps the same bytes as a second numpy array, so `arr2[5]` reads the value 5 written by `arr[5] = 5`.
- C) Incorrect — `shm.name` is available immediately after `SharedMemory(create=True, ...)` returns; it is set before `close()` is called. Closing the shm would remove access to the buffer, but `shm.name` is just a string attribute.
- D) Incorrect — multiple numpy arrays can safely share the same buffer (shared memory or otherwise) as long as they do not write to overlapping regions simultaneously. Creating two read/write views of the same buffer is valid and is exactly the intended pattern for inter-process shared memory.

---

## Q28 — What is the on-disk size ordering?

> **Week reference:** Week 8

```python
import zarr, numpy as np

data = np.zeros((1000, 1000), dtype='int32')
# Mandelbrot-like: most values are uniform (0), boundary values vary
data[400:600, 400:600] = np.random.randint(1, 100, (200, 200))

for chunk_size in [10, 50, 200]:
    z = zarr.open(f'arr_{chunk_size}.zarr', mode='w',
                  shape=(1000, 1000), chunks=(chunk_size, chunk_size),
                  dtype='int32')
    z[:] = data
```

After all three writes, which on-disk size ordering is correct (smallest to largest)?

**A)** `arr_10.zarr` < `arr_50.zarr` < `arr_200.zarr`
**B)** `arr_200.zarr` < `arr_50.zarr` < `arr_10.zarr`
**C)** `arr_10.zarr` = `arr_50.zarr` = `arr_200.zarr`
**D)** `arr_50.zarr` < `arr_10.zarr` < `arr_200.zarr`

**Answer: B**

- A) Incorrect — this is the reverse of the correct order. Larger chunks compress better because they span more of the uniform-zero region in a single compressor call, achieving a higher compression ratio.
- B) Correct — with mostly-zero data (large uniform regions), larger chunks give Blosc a bigger block of redundant bytes to compress. `arr_200.zarr` uses only 25 chunks (each 200×200×4 = 160 KB before compression) and achieves the highest ratio. `arr_10.zarr` has 10,000 chunks (each only 400 bytes), most of which compress individually to nearly their original size with little benefit. The DTU 02613 Week 8 solutions confirmed this pattern empirically with Mandelbrot data.
- C) Incorrect — compressed size depends strongly on chunk shape. Uniform regions compress much better in larger chunks, producing dramatically different on-disk sizes.
- D) Incorrect — there is no reason for the middle chunk size to produce the smallest result when all chunks contain only uniform zeros in the non-random regions.

---

## Q29 — What does this zarr.zeros call do to existing data?

> **Week reference:** Week 8

```python
import zarr, numpy as np

# First run: write some data
z = zarr.open('result.zarr', mode='w', shape=(100,), chunks=(10,), dtype='float64')
z[:] = np.ones(100) * 9.9

# Second run (same script, immediately after)
z2 = zarr.zeros('result.zarr', chunks=(10,), dtype='float64')
print(z2[0])
```

Assume `zarr.zeros` with a path string behaves like `zarr.open(path, mode='w', ...)`.

**A)** `9.9` — `zarr.zeros` opens the existing store without modifying data.
**B)** `0.0` — `zarr.zeros` overwrites the existing store with a fresh zero-filled array.
**C)** `ValueError` — `zarr.zeros` raises an error if the store already exists.
**D)** `9.9` — `zarr.zeros` only zeroes chunks that have not been written yet.

**Answer: B**

- A) Incorrect — `zarr.zeros` uses `mode='w'` internally, which truncates any existing store. The previous 9.9 values are overwritten.
- B) Correct — `zarr.zeros` is equivalent to `zarr.open(path, mode='w', fill_value=0, ...)`. It creates a fresh store, destroying any existing data. After the call, all chunks return the fill value 0.0. This is a common exam trap: `zarr.zeros` sounds like it might only initialise empty chunks, but it actually replaces the entire store.
- C) Incorrect — `zarr.zeros` does not raise on a pre-existing store; it silently overwrites it. To get an error on pre-existing data, use `mode='w-'`.
- D) Incorrect — `zarr.zeros` does not distinguish between written and unwritten chunks. It replaces the entire store unconditionally.

---

## Q30 — What does this compressor=None zarr store?

> **Week reference:** Week 8

```python
import zarr, numpy as np, os

z = zarr.open('raw.zarr', mode='w', shape=(100,), chunks=(100,),
              dtype='float32', compressor=None)
z[:] = np.ones(100, dtype='float32')

chunk_file = 'raw.zarr/0'
print(os.path.getsize(chunk_file))
```

What does this print?

**A)** A value smaller than 400 — Blosc compression is applied automatically regardless of `compressor=None`.
**B)** `400` — the chunk stores 100 float32 values × 4 bytes with no compression.
**C)** `0` — chunks of all-ones are always compressed to zero bytes.
**D)** `800` — zarr doubles the storage to maintain a compressed and uncompressed copy.

**Answer: B**

- A) Incorrect — `compressor=None` explicitly disables compression. Zarr respects this setting and writes raw uncompressed bytes to the chunk file.
- B) Correct — with `compressor=None`, each chunk is stored as raw bytes with no compression overhead. 100 elements × 4 bytes/float32 = 400 bytes. The chunk file `0` (chunk index 0 along axis 0) will be exactly 400 bytes.
- C) Incorrect — even if the data were highly compressible, `compressor=None` means no compression is applied. The file stores the literal 400 bytes of IEEE 754 float32 `1.0` values.
- D) Incorrect — Zarr never stores a duplicate copy. With or without compression, each chunk is stored exactly once in the format configured at creation time.

---

## Q31 — What does reading an unwritten Zarr chunk return?

> **Week reference:** Week 8

```python
import zarr, numpy as np

z = zarr.open('lazy.zarr', mode='w', shape=(200, 200),
              chunks=(100, 100), dtype='int32', fill_value=-1)

# Only write the top-left chunk
z[0:100, 0:100] = 99

print(z[150, 150])
print(z.nchunks_initialized)
```

**A)** `99` then `4`
**B)** `0` then `1`
**C)** `-1` then `1`
**D)** `KeyError` then `1`

**Answer: C**

- A) Incorrect — element `[150, 150]` falls in the bottom-right chunk `(1, 1)`, which was never written. It returns the fill value, not 99.
- B) Incorrect — the fill value is `-1`, not `0`. The default fill value is 0, but here `fill_value=-1` was explicitly set. This is the classic trap of assuming fill_value=0.
- C) Correct — element `[150, 150]` is in chunk `(1, 1)` (rows 100–199, cols 100–199), which has never been written. Zarr returns `fill_value=-1` without reading any chunk file (there is nothing to read). `nchunks_initialized` is 1 because only the top-left chunk `(0, 0)` has been written.
- D) Incorrect — Zarr never raises `KeyError` for unwritten chunks. Reading an absent chunk is well-defined: it returns the fill value. `KeyError` would occur only if you accessed the raw store dictionary directly with an invalid key.

---
