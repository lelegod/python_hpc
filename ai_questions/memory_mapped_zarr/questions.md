# Memory-Mapped Files & Zarr — MCQ Practice

> Topics: np.memmap modes, RAM footprint, Zarr chunk shapes, shared memory.
> Exam frequency: **F25 exam** — np.memmap Q19, Zarr in re-exam and F25.

---

## Q1 — np.memmap Mode 'r'
> **Week reference:** Week 8

You open a memory-mapped file with `np.memmap('data.raw', mode='r', dtype='float32', shape=(1000,))`. What is true about this operation?

- A) The file is created if it does not exist, then opened for reading.
- B) The file must already exist; you can read but not modify the data.
- C) The file must already exist; you can read and write, but changes are not persisted.
- D) The file is overwritten with zeros and opened for reading.

**Answer: B**

- A) Incorrect — mode 'r' never creates a file; if the file is absent, a `FileNotFoundError` is raised.
- B) Correct — 'r' is read-only and requires the file to already exist on disk.
- C) Incorrect — that describes mode 'c' (copy-on-write), not 'r'.
- D) Incorrect — that describes mode 'w+', which creates or overwrites; 'r' leaves the file intact.

---

## Q2 — np.memmap Mode 'w+'
> **Week reference:** Week 8

Which statement correctly describes `np.memmap('out.raw', mode='w+', dtype='float32', shape=(500, 500))`?

- A) The file must exist; it is opened for both reading and writing.
- B) The file is opened read-only; writes raise a `ValueError`.
- C) The file is created if absent (or truncated if present) and opened for reading and writing.
- D) Changes are kept in RAM and never written to disk.

**Answer: C**

- A) Incorrect — that describes mode 'r+'; 'w+' does not require a pre-existing file.
- B) Incorrect — 'w+' allows both reading and writing.
- C) Correct — 'w+' creates a new file or truncates an existing one, then opens it read/write.
- D) Incorrect — that describes mode 'c'; 'w+' flushes changes to disk.

---

## Q3 — np.memmap Mode 'r+'
> **Week reference:** Week 8

A colleague runs `m = np.memmap('existing.raw', mode='r+', dtype='int16', shape=(200,))` and then sets `m[0] = 42`. What happens?

- A) A `TypeError` is raised because 'r+' is read-only.
- B) The value 42 is stored in RAM only and lost when the script exits.
- C) The value 42 is written to the file on disk (after flush or gc).
- D) A new copy of the file is created with the modification.

**Answer: C**

- A) Incorrect — 'r+' allows writing; only mode 'r' is read-only.
- B) Incorrect — that describes mode 'c'; 'r+' persists changes.
- C) Correct — 'r+' requires the file to exist and writes changes back to disk.
- D) Incorrect — no copy is created; the original file is modified in place.

---

## Q4 — np.memmap Mode 'c' (Copy-on-Write)
> **Week reference:** Week 8

You open a memmap with `m = np.memmap('reference.raw', mode='c', dtype='float64', shape=(1000,))`, modify `m[0] = 99.0`, then close the script. What is the state of `reference.raw` after the script exits?

- A) `reference.raw` is unchanged; the modification existed only in RAM.
- B) `reference.raw` now has 99.0 at index 0.
- C) `reference.raw` is deleted and replaced with a new file containing 99.0.
- D) A `PermissionError` is raised when you try to assign to `m[0]`.

**Answer: A**

- A) Correct — mode 'c' is copy-on-write: writes go to a private RAM copy and are never flushed to disk.
- B) Incorrect — the original file is never modified in 'c' mode.
- C) Incorrect — the original file is untouched; no replacement is created.
- D) Incorrect — the assignment succeeds in RAM; no error is raised.

---

## Q5 — memmap RAM Footprint
> **Week reference:** Week 8

You create `m = np.memmap('huge.raw', mode='r', dtype='float32', shape=(10**9,))` — a 4 GB file. You then read only `m[0]`. How much RAM does the OS actually load?

- A) 4 GB — the entire file is loaded when the memmap is created.
- B) 4 bytes — only the single float32 is loaded.
- C) One OS memory page (typically 4 KB), because the OS loads pages on demand.
- D) 512 MB — NumPy preloads 1/8 of the file as a read-ahead buffer.

**Answer: C**

- A) Incorrect — memmap uses virtual memory mapping; the OS only loads pages that are actually accessed.
- B) Incorrect — the OS always loads a full page (minimum granularity), not a single element.
- C) Correct — the OS demand-pages only the 4 KB page containing `m[0]`.
- D) Incorrect — NumPy does no special read-ahead buffering for memmap arrays.

---

## Q6 — Strided memmap Access and RAM Usage
> **Week reference:** Week 8

Consider:
```python
x = np.memmap('big.raw', mode='r', dtype='int32', shape=(10**10,))
y = np.array(x[::100_000])
```
Approximately how much RAM does `y` occupy?

- A) 40 GB (the entire `x` array).
- B) 400 KB (100,000 elements × 4 bytes).
- C) 400 MB (100,000,000 bytes).
- D) 4 MB (1,000,000 elements × 4 bytes).

**Answer: B**

- A) Incorrect — `x` itself occupies virtually no RAM; only the pages touched when creating `y` are loaded.
- B) Correct — `10^10 / 10^5 = 10^5` elements × 4 bytes/element = 400,000 bytes ≈ 400 KB.
- C) Incorrect — that would be 100 million elements, not 100,000.
- D) Incorrect — the stride selects 10^5 elements, not 10^6.

---

## Q7 — dtype and shape Must Be Specified
> **Week reference:** Week 8

A 10×10 array of `float64` values was saved to disk. A new developer opens it with:
```python
m = np.memmap('matrix.raw', mode='r', shape=(10, 10))
```
What happens?

- A) NumPy auto-detects `float64` from the file header and reads it correctly.
- B) The default dtype is `uint8`, so the 800-byte file is interpreted as 100 `uint8` values with wrong data.
- C) A `ValueError` is raised because dtype is required.
- D) The default dtype is `float64`, so the data is read correctly.

**Answer: B**

- A) Incorrect — raw `.npy`-less memmap files have no header; NumPy cannot auto-detect dtype.
- B) Correct — `np.memmap` defaults to `dtype='uint8'`; the 800-byte file is silently misread as 100 uint8 values (shape (10,10) of uint8 = 100 bytes, so the shape would actually be mismatched too).
- C) Incorrect — NumPy does not raise an error for a missing dtype; it silently falls back to uint8.
- D) Incorrect — the default dtype is `uint8`, not `float64`.

---

## Q8 — Zarr Chunk Shape for Row-by-Row Access
> **Week reference:** Week 8

You have a 1024×1024 Zarr array with chunk shape `(1, 1024)`. A loop reads each row one at a time: `for i in range(1024): process(arr[i, :])`. How many chunks are loaded per row?

- A) 1024 chunks per row — one chunk per column element.
- B) 1 chunk per row — the chunk spans the entire row width.
- C) 2 chunks per row — Zarr always splits rows in half.
- D) 0 chunks — Zarr caches the entire array on first access.

**Answer: B**

- A) Incorrect — that would occur with chunk shape `(1024, 1)`, not `(1, 1024)`.
- B) Correct — chunk `(1, 1024)` covers one full row, so each `arr[i, :]` loads exactly one chunk.
- C) Incorrect — there is no automatic halving; chunk shape is user-defined.
- D) Incorrect — Zarr loads chunks on demand; it does not cache the entire array upfront.

---

## Q9 — Zarr Chunk Shape Mismatched to Access Pattern
> **Week reference:** Week 8

The same 1024×1024 Zarr array now has chunk shape `(1024, 1)`. A loop reads each row: `for i in range(1024): process(arr[i, :])`. How many chunks must be loaded for a single row read?

- A) 1 chunk.
- B) 128 chunks.
- C) 1024 chunks.
- D) 0 chunks, because Zarr preloads the array.

**Answer: C**

- A) Incorrect — chunk `(1024, 1)` covers one full column, not one full row.
- B) Incorrect — 128 would apply to a different chunk width like 8.
- C) Correct — each chunk is one element wide, so reading all 1024 columns requires 1024 chunk loads — extremely inefficient.
- D) Incorrect — Zarr loads lazily; no preloading occurs.

---

## Q10 — Zarr Chunk Size Too Small
> **Week reference:** Week 8

You store a 1024×1024 float32 Zarr array with chunk shape `(10, 10)`. What is the primary performance problem?

- A) Each chunk is too large, causing excessive memory usage.
- B) There are too many chunks (~10,500), leading to high metadata and I/O overhead per operation.
- C) Zarr cannot handle chunk shapes that are not powers of two.
- D) Small chunks prevent compression from working at all.

**Answer: B**

- A) Incorrect — the chunks are small (400 bytes each), not large.
- B) Correct — ~10,500 tiny chunks means enormous overhead in chunk lookups, file-system operations, and decompression calls per array operation.
- C) Incorrect — Zarr accepts arbitrary chunk shapes; powers of two are not required.
- D) Incorrect — compression still works on small chunks, but is less effective and has higher per-chunk overhead.

---

## Q11 — Zarr Chunk Size Too Large
> **Week reference:** Week 8

You store a 1024×1024 float32 array as a single Zarr chunk `(512, 512)` and process it with 16 parallel workers, each responsible for a different quadrant. What is the main drawback?

- A) Zarr raises a `ChunkError` when the chunk is larger than 256×256.
- B) With only 4 chunks total, workers frequently contend for the same chunk, reducing parallelism and causing one process to block a large memory region.
- C) The file cannot be compressed when chunk size exceeds 64 KB.
- D) Reading a single element requires loading 512×512 × 4 = 1 MB into RAM, but this is unavoidable.

**Answer: B**

- A) Incorrect — Zarr imposes no maximum chunk size limit.
- B) Correct — large chunks reduce available parallelism because fewer chunks exist and each one covers a large region; workers processing adjacent data must wait or load the same chunk.
- C) Incorrect — Zarr supports compression regardless of chunk size.
- D) Incorrect — while true, this is not specific to "too large" chunks and is a secondary concern; parallelism degradation is the primary problem stated.

---

## Q12 — Parallel Writes to Different Zarr Chunks
> **Week reference:** Week 8

Four worker processes simultaneously write to four non-overlapping regions of a Zarr array, each writing to a distinct chunk. Which statement is correct?

- A) All four writes will corrupt the data because Zarr arrays are not thread/process safe.
- B) A file lock must be acquired before each write to prevent chunk-level race conditions.
- C) The writes are safe without locking because each chunk maps to an independent storage block; there is no shared state between different chunks.
- D) Zarr uses a global write lock internally, so only one process writes at a time.

**Answer: C**

- A) Incorrect — Zarr is safe for concurrent writes as long as different processes write to different chunks.
- B) Incorrect — locking is only needed if multiple processes write to the *same* chunk simultaneously; different chunks are fully independent.
- C) Correct — each Zarr chunk is stored as a separate file (or object), so writes to distinct chunks have no shared state and require no locking.
- D) Incorrect — Zarr has no global write lock; it relies on chunk-level independence for concurrency safety.

---
