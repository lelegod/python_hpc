# Memory-Mapped Files & Zarr — MCQ Practice

> Topics: np.memmap modes, RAM footprint, Zarr chunk shapes, shared memory.
> Exam frequency: **F25 exam** — np.memmap Q19, Zarr in re-exam and F25.

---

## Q1 — np.memmap Mode 'r'
> **Week reference:** Week 8

**Mental Model:** Mode letters map to standard Unix file semantics: 'r' = read-only, must exist. The key trap is confusing 'r' (read-only, file must exist) with 'w+' (create/overwrite, read-write). If you open in 'r' and the file is missing, you get an immediate FileNotFoundError.

You open a memory-mapped file with `np.memmap('data.raw', mode='r', dtype='float32', shape=(1000,))`. What is true about this operation?

- A) The file is created if it does not exist, then opened for reading.
- B) The file must already exist; you can read but not modify the data.
- C) The file must already exist; you can read and write, but changes are not persisted.
- D) The file is overwritten with zeros and opened for reading.

**Answer: B**

- A) Incorrect — mode 'r' never creates a file; if the file is absent, a `FileNotFoundError` is raised immediately. File creation only occurs with modes 'w+' (create or truncate) and implicitly with 'r+' if misused.
- B) Correct — 'r' is the read-only mode. The file must already exist on disk, and any attempt to assign to the array (e.g. `m[0] = 1`) raises a `ValueError: assignment destination is read-only`.
- C) Incorrect — that describes mode 'c' (copy-on-write): writes go to a private RAM copy and are never flushed to disk. Mode 'r' does not allow writes at all.
- D) Incorrect — that describes mode 'w+', which creates the file (or truncates it to zeros) and opens it for reading and writing. Mode 'r' leaves the existing file completely intact.

---

## Q2 — np.memmap Mode 'w+'
> **Week reference:** Week 8

**Mental Model:** 'w+' = create-or-overwrite, read-write. The '+' means both read and write. The 'w' means create fresh (destroying existing content). This is how you write a new large array to disk without allocating it all in RAM first.

Which statement correctly describes `np.memmap('out.raw', mode='w+', dtype='float32', shape=(500, 500))`?

- A) The file must exist; it is opened for both reading and writing.
- B) The file is opened read-only; writes raise a `ValueError`.
- C) The file is created if absent (or truncated if present) and opened for reading and writing.
- D) Changes are kept in RAM and never written to disk.

**Answer: C**

- A) Incorrect — that describes mode 'r+', which requires a pre-existing file and opens it read-write. Mode 'w+' does not require a pre-existing file; it creates one (or destroys the existing one).
- B) Incorrect — 'w+' is explicitly read-write; the '+' in all memmap modes indicates write access is permitted. Trying to describe 'w+' as read-only confuses it with plain 'r'.
- C) Correct — 'w+' creates a new file of the correct size (500×500×4 = 1 MB of float32) filled with zeros, or truncates and overwrites an existing file. The resulting array supports both reading and writing.
- D) Incorrect — that describes mode 'c' (copy-on-write). Mode 'w+' flushes changes back to disk, either explicitly via `m.flush()` or automatically when the object is garbage collected.

---

## Q3 — np.memmap Mode 'r+'
> **Week reference:** Week 8

**Mental Model:** 'r+' = open existing file, read-write, changes persist. The key distinction from 'c': changes go to disk. The key distinction from 'w+': the file must already exist (no creation/truncation).

A colleague runs `m = np.memmap('existing.raw', mode='r+', dtype='int16', shape=(200,))` and then sets `m[0] = 42`. What happens?

- A) A `TypeError` is raised because 'r+' is read-only.
- B) The value 42 is stored in RAM only and lost when the script exits.
- C) The value 42 is written to the file on disk (after flush or gc).
- D) A new copy of the file is created with the modification.

**Answer: C**

- A) Incorrect — 'r+' allows writing; the '+' explicitly grants write access. Only mode 'r' (without '+') is read-only. A `TypeError` would be raised for mode 'r', not 'r+'.
- B) Incorrect — that describes mode 'c' (copy-on-write), where all writes go to a private RAM copy that is discarded on exit. In 'r+' mode, the OS memory map is backed by the actual file on disk.
- C) Correct — 'r+' requires the file to exist and maps it read-write to disk. When `m[0] = 42` is executed, the OS schedules a write-back to the file; calling `m.flush()` ensures it is committed immediately, but it will persist regardless on clean process exit.
- D) Incorrect — no copy is created; the original file is modified in place. Copy-on-write behaviour only occurs with mode 'c'.

---

## Q4 — np.memmap Mode 'c' (Copy-on-Write)
> **Week reference:** Week 8

**Mental Model:** Mode 'c' = read the real file, but redirect all writes to a private RAM buffer that vanishes when the process exits. It is useful for safely experimenting with large datasets without risk of corrupting the original file.

You open a memmap with `m = np.memmap('reference.raw', mode='c', dtype='float64', shape=(1000,))`, modify `m[0] = 99.0`, then close the script. What is the state of `reference.raw` after the script exits?

- A) `reference.raw` is unchanged; the modification existed only in RAM.
- B) `reference.raw` now has 99.0 at index 0.
- C) `reference.raw` is deleted and replaced with a new file containing 99.0.
- D) A `PermissionError` is raised when you try to assign to `m[0]`.

**Answer: A**

- A) Correct — mode 'c' is copy-on-write: the OS creates a private, per-process copy of any written page in RAM. When the process exits, this private copy is discarded. The original `reference.raw` on disk is never written to.
- B) Incorrect — the original file is never modified in 'c' mode; this is the entire point of copy-on-write. If you need writes to persist, use mode 'r+' or 'w+' instead.
- C) Incorrect — the original file is untouched; no replacement file is created. Copy-on-write only affects the in-memory page table mapping, leaving disk state unchanged.
- D) Incorrect — mode 'c' allows the assignment to succeed in RAM with no error. The write is accepted silently but redirected to a private page rather than the file-backed page.

---

## Q5 — memmap RAM Footprint
> **Week reference:** Week 8

**Mental Model:** memmap = virtual memory mapping, not eager loading. Creating a memmap only allocates virtual address space (essentially free). Physical RAM pages are loaded on demand by the OS, one 4 KB page at a time, only when you actually access that memory region.

You create `m = np.memmap('huge.raw', mode='r', dtype='float32', shape=(10**9,))` — a 4 GB file. You then read only `m[0]`. How much RAM does the OS actually load?

- A) 4 GB — the entire file is loaded when the memmap is created.
- B) 4 bytes — only the single float32 is loaded.
- C) One OS memory page (typically 4 KB), because the OS loads pages on demand.
- D) 512 MB — NumPy preloads 1/8 of the file as a read-ahead buffer.

**Answer: C**

- A) Incorrect — memmap uses virtual memory mapping via `mmap(2)`; the OS only loads pages that are actually touched. Creating the memmap object allocates virtual address space only, with zero physical RAM pages loaded.
- B) Incorrect — the OS always loads a full memory page (typically 4 096 bytes) as the minimum I/O granularity; it cannot load less than one page. A single float32 (4 bytes) cannot be loaded in isolation below this page boundary.
- C) Correct — when `m[0]` is first accessed, the OS demand-pages the single 4 KB page containing that address. For the rest of the 4 GB file, no physical RAM is consumed until those pages are accessed.
- D) Incorrect — NumPy does no special read-ahead buffering for memmap arrays; that is the OS's responsibility. The OS may perform sequential read-ahead heuristics, but this is not NumPy behaviour and amounts to at most a few pages.

---

## Q6 — Strided memmap Access and RAM Usage
> **Week reference:** Week 8

**Mental Model:** The array `y` holds the *copied* values, not a view into the memmap. So `y`'s RAM = number_of_selected_elements × bytes_per_element. The memmap `x` itself uses only the pages actually read during the strided access.

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

- A) Incorrect — `x` itself occupies virtually no RAM (it is a virtual memory mapping); and `y` is a copy of only the selected elements. Even `x`'s touched pages during the strided access are small compared to 40 GB.
- B) Correct — `10^10 / 10^5 = 10^5` elements are selected by the stride. Each int32 is 4 bytes, giving `10^5 × 4 = 400,000 bytes = 400 KB` for array `y`. The fact that `x` is a 40 GB file is irrelevant to `y`'s size.
- C) Incorrect — 400 MB would require `10^8` elements; the stride `::100_000` on `10^10` elements gives `10^10 / 10^5 = 10^5` elements, not `10^8`.
- D) Incorrect — 4 MB would require `10^6` elements; the correct count is `10^10 / 10^5 = 10^5 = 100,000` elements. Easy off-by-one: the stride step is `100_000`, not `10_000`.

---

## Q7 — dtype and shape Must Be Specified
> **Week reference:** Week 8

**Mental Model:** Raw `.raw` files have no header — they are just bytes on disk. NumPy has no way to detect the dtype from file content alone. Without specifying dtype, it defaults to `uint8`, silently misinterpreting all your data.

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

- A) Incorrect — raw memmap files (`.raw`) have no header; only `.npy` files include dtype/shape metadata. NumPy cannot auto-detect dtype from a headerless binary file.
- B) Correct — `np.memmap` defaults to `dtype='uint8'`. The 800-byte file (10×10×8 bytes of float64) would be interpreted as 100 uint8 values instead of 100 float64 values — shape (10,10) of uint8 = 100 bytes, which mismatches the 800-byte file size. All data is silently corrupted.
- C) Incorrect — NumPy does not raise an error for a missing dtype; it silently falls back to uint8. This silent corruption is more dangerous than a raised error because the code appears to run successfully.
- D) Incorrect — the default dtype is `uint8`, not `float64`. There is no mechanism by which NumPy could infer float64 from a raw byte file; always specify both dtype AND shape explicitly.

---

## Q8 — Zarr Chunk Shape for Row-by-Row Access
> **Week reference:** Week 8

**Mental Model:** Align chunk shape to access pattern. If you read row-by-row, make each chunk one full row: `(1, width)`. This guarantees one chunk I/O operation per `arr[i, :]` access. Misalignment (e.g. column-oriented chunks) causes many chunks per access.

You have a 1024×1024 Zarr array with chunk shape `(1, 1024)`. A loop reads each row one at a time: `for i in range(1024): process(arr[i, :])`. How many chunks are loaded per row?

- A) 1024 chunks per row — one chunk per column element.
- B) 1 chunk per row — the chunk spans the entire row width.
- C) 2 chunks per row — Zarr always splits rows in half.
- D) 0 chunks — Zarr caches the entire array on first access.

**Answer: B**

- A) Incorrect — that would occur with chunk shape `(1024, 1)` (each chunk is one element tall and one element wide). With `(1, 1024)` the chunk covers the full row width of 1024 elements.
- B) Correct — chunk `(1, 1024)` means each chunk covers exactly 1 row × 1024 columns. Reading `arr[i, :]` retrieves all 1024 elements of row i, which is exactly one chunk. This is the optimal alignment for row-by-row access.
- C) Incorrect — Zarr does no automatic halving; chunk shape is entirely user-defined and Zarr respects it exactly. There is no built-in "split in half" behaviour.
- D) Incorrect — Zarr loads chunks on demand (lazily); it does not cache the entire array on first access. Each chunk is only loaded when a read/write operation touches it.

---

## Q9 — Zarr Chunk Shape Mismatched to Access Pattern
> **Week reference:** Week 8

**Mental Model:** If chunks are column-oriented `(1024, 1)` but you access rows, every row access spans all 1024 column-chunks. Each chunk load requires a separate decompression + I/O operation — a 1024× amplification of I/O cost per row read.

The same 1024×1024 Zarr array now has chunk shape `(1024, 1)`. A loop reads each row: `for i in range(1024): process(arr[i, :])`. How many chunks must be loaded for a single row read?

- A) 1 chunk.
- B) 128 chunks.
- C) 1024 chunks.
- D) 0 chunks, because Zarr preloads the array.

**Answer: C**

- A) Incorrect — chunk `(1024, 1)` covers one full column (1024 rows × 1 column), not one full row. A single row spans all 1024 columns, so it crosses all 1024 column-chunks simultaneously.
- B) Incorrect — 128 would apply if each chunk were 8 columns wide `(1024, 8)`: 1024/8 = 128 chunks per row. With chunk width 1, each column is its own chunk, giving 1024 chunks per row.
- C) Correct — each chunk covers 1 column of all 1024 rows. To read row i across all 1024 columns, Zarr must load 1024 separate chunk files, decompress each one, and extract the single element at row i. This is catastrophically inefficient — 1024× the I/O of the aligned case.
- D) Incorrect — Zarr loads lazily; no preloading occurs. Chunk data is only read from disk when a read operation touches it.

---

## Q10 — Zarr Chunk Size Too Small
> **Week reference:** Week 8

**Mental Model:** Each chunk is a separate file (or storage object). Very small chunks = very many files = very high filesystem metadata overhead. A 1024×1024 array with (10,10) chunks = ~10,500 chunk files for one array. Every operation touching multiple chunks pays the overhead of multiple file opens, decompressions, and closes.

You store a 1024×1024 float32 Zarr array with chunk shape `(10, 10)`. What is the primary performance problem?

- A) Each chunk is too large, causing excessive memory usage.
- B) There are too many chunks (~10,500), leading to high metadata and I/O overhead per operation.
- C) Zarr cannot handle chunk shapes that are not powers of two.
- D) Small chunks prevent compression from working at all.

**Answer: B**

- A) Incorrect — the chunks are tiny (10×10×4 = 400 bytes each), not large. Memory pressure from individual chunks is negligible; the problem is the opposite.
- B) Correct — 1024/10 ≈ 103 chunks per dimension, giving ~103² ≈ 10,500 chunk files. Every operation on the array requires traversing metadata for thousands of chunks, opening/closing thousands of files, and running the compression algorithm thousands of times. This overhead dominates actual computation time.
- C) Incorrect — Zarr accepts arbitrary integer chunk shapes; powers of two are a common choice for alignment but are not required. Shape `(10, 10)` is syntactically and semantically valid.
- D) Incorrect — compression still works on small chunks (they are just compressed as small independent blocks), but it is less effective (less data to find patterns in) and carries higher per-call overhead. The primary problem is I/O metadata overhead, not compression failure.

---

## Q11 — Zarr Chunk Size Too Large
> **Week reference:** Week 8

**Mental Model:** Large chunks = fewer parallel units of work. With a (512,512) chunk on a 1024×1024 array, you have only 4 chunks total — 4 workers is the maximum parallelism. Workers wanting adjacent data must load the same big chunk, causing contention and excess memory usage.

You store a 1024×1024 float32 array as a single Zarr chunk `(512, 512)` and process it with 16 parallel workers, each responsible for a different quadrant. What is the main drawback?

- A) Zarr raises a `ChunkError` when the chunk is larger than 256×256.
- B) With only 4 chunks total, workers frequently contend for the same chunk, reducing parallelism and causing one process to block a large memory region.
- C) The file cannot be compressed when chunk size exceeds 64 KB.
- D) Reading a single element requires loading 512×512 × 4 = 1 MB into RAM, but this is unavoidable.

**Answer: B**

- A) Incorrect — Zarr imposes no maximum chunk size limit. Any chunk shape is valid as long as it fits in memory; there is no built-in `ChunkError` for large chunks.
- B) Correct — a 1024×1024 array with chunk size (512,512) has only 4 chunks total (2×2 grid). With 16 workers, 12 workers must share chunks with others. Each shared chunk must be fully loaded into memory before any worker can access its sub-region, and writes to the same chunk require coordination or produce corrupted output.
- C) Incorrect — Zarr supports compression for any chunk size. A (512,512) float32 chunk = 1 MB, which compresses fine. No such size limit exists in any Zarr compressor.
- D) Incorrect — while loading 1 MB per element access is indeed a consequence of large chunks and is a real drawback, the question asks for the *main* drawback in the context of 16 parallel workers. Parallelism degradation (B) is the more significant problem being tested here.

---

## Q12 — Parallel Writes to Different Zarr Chunks
> **Week reference:** Week 8

**Mental Model:** Zarr chunks are independent storage objects (separate files in DirectoryStore, separate keys in ZipStore, etc.). Writing to chunk A never touches chunk B's bytes. This independence makes parallel writes to distinct chunks inherently safe with no coordination needed.

Four worker processes simultaneously write to four non-overlapping regions of a Zarr array, each writing to a distinct chunk. Which statement is correct?

- A) All four writes will corrupt the data because Zarr arrays are not thread/process safe.
- B) A file lock must be acquired before each write to prevent chunk-level race conditions.
- C) The writes are safe without locking because each chunk maps to an independent storage block; there is no shared state between different chunks.
- D) Zarr uses a global write lock internally, so only one process writes at a time.

**Answer: C**

- A) Incorrect — Zarr is safe for concurrent writes as long as different processes write to different chunks. The safety guarantee comes from each chunk being a separate storage object with no shared bytes.
- B) Incorrect — locking is only needed if multiple processes write to the *same* chunk simultaneously, which would race on the same underlying file or object. Different chunks have completely independent storage, so no locking is required between them.
- C) Correct — each Zarr chunk is stored as a separate file (in DirectoryStore) or object (in S3Store, etc.). Writing chunk i modifies only chunk i's bytes on disk; chunk j's bytes are physically stored elsewhere and are completely unaffected. Four workers writing to four different chunks have zero shared state.
- D) Incorrect — Zarr has no global write lock. Its concurrency model relies entirely on chunk-level independence for safety. Adding a global lock would defeat the purpose of chunked storage for parallel I/O.

---
