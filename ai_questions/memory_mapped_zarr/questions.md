# Memory-Mapped Files & Zarr — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — np.memmap Mode 'r'](#q1-npmemmap-mode-r)
- [Q2 — np.memmap Mode 'w+'](#q2-npmemmap-mode-w)
- [Q3 — np.memmap Mode 'r+'](#q3-npmemmap-mode-r)
- [Q4 — np.memmap Mode 'c' (Copy-on-Write)](#q4-npmemmap-mode-c-copy-on-write)
- [Q5 — memmap RAM Footprint](#q5-memmap-ram-footprint)
- [Q6 — Strided memmap Access and RAM Usage](#q6-strided-memmap-access-and-ram-usage)
- [Q7 — dtype and shape Must Be Specified](#q7-dtype-and-shape-must-be-specified)
- [Q8 — Zarr Chunk Shape for Row-by-Row Access](#q8-zarr-chunk-shape-for-row-by-row-access)
- [Q9 — Zarr Chunk Shape Mismatched to Access Pattern](#q9-zarr-chunk-shape-mismatched-to-access-pattern)
- [Q10 — Zarr Chunk Size Too Small](#q10-zarr-chunk-size-too-small)
- [Q11 — Zarr Chunk Size Too Large](#q11-zarr-chunk-size-too-large)
- [Q12 — Parallel Writes to Different Zarr Chunks](#q12-parallel-writes-to-different-zarr-chunks)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q13 — Which memmap Mode Creates a New File?](#q13-which-memmap-mode-creates-a-new-file)
- [Q14 — What Happens When You `del` a Mode 'r+' memmap?](#q14-what-happens-when-you-del-a-mode-r-memmap)
- [Q15 — Zarr Chunk Memory Calculation](#q15-zarr-chunk-memory-calculation)
- [Q16 — When Is Memory Mapping Better Than Loading?](#q16-when-is-memory-mapping-better-than-loading)
- [Q17 — Multiple Processes Sharing a Read-Only memmap](#q17-multiple-processes-sharing-a-read-only-memmap)
- [Q18 — Choosing Between memmap and Loading for Sequential Access](#q18-choosing-between-memmap-and-loading-for-sequential-access)
- [Q19 — Mode 'c' After Flush](#q19-mode-c-after-flush)
- [Q20 — Zarr Chunk Count for a Given Array](#q20-zarr-chunk-count-for-a-given-array)
- [Q21 — What Happens When memmap 'r+' File Does Not Exist?](#q21-what-happens-when-memmap-r-file-does-not-exist)
- [Q22 — Zarr Chunk Shape for Column-by-Column Access](#q22-zarr-chunk-shape-for-column-by-column-access)
- [Q23 — zarr.open Default Mode](#q23-zarropen-default-mode)
- [Q24 — zarr.open Mode 'w-' / 'x'](#q24-zarropen-mode-w-x)
- [Q25 — zarr.open Mode Comparison: 'r+' vs 'a'](#q25-zarropen-mode-comparison-r-vs-a)

---

> Topics: np.memmap modes, RAM footprint, Zarr chunk shapes, shared memory.
> Exam frequency: **F25 exam** — np.memmap Q19, Zarr in re-exam and F25.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--npmemmap-mode-r)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

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

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets memmap modes, Zarr chunk optimization, block memory calculation, and when to prefer memory mapping over loading

---

## Q13 — Which memmap Mode Creates a New File?

> **Week reference:** Week 8

You need to create a new large binary file `output.raw` and fill it with computed float32 values. The file does not exist yet. Which `np.memmap` mode is correct?

- A) `'r'`
- B) `'r+'`
- C) `'c'`
- D) `'w+'`

**Answer: D**

Mode `'w+'` is the only mode that creates a new file (or overwrites an existing one) and opens it for both reading and writing. It allocates the full byte extent on disk immediately (filled with zeros), allowing you to write elements into any position. Mode `'r'` is read-only and requires the file to already exist. Mode `'r+'` also requires the file to already exist and would raise `FileNotFoundError` if it is absent. Mode `'c'` requires the file to exist and keeps all writes in private RAM, never persisting them — useless for creating a new file.

---

## Q14 — What Happens When You `del` a Mode 'r+' memmap?

> **Week reference:** Week 8

A script opens a memory-mapped file with `m = np.memmap('data.raw', mode='r+', dtype='float32', shape=(1000,))`, sets several elements, then executes `del m`. What is the effect on `data.raw`?

- A) The file is deleted from disk.
- B) The modified values are lost; the file remains unchanged.
- C) Any pending writes are flushed to disk and the mapping is released.
- D) The file is truncated to zero bytes.

**Answer: C**

When a `numpy.memmap` object opened in `'r+'` mode is deleted (or garbage-collected), Python's finalizer calls the underlying `mmap` object's close, which causes the OS to flush any dirty memory pages back to the backing file. The result is that all modifications made through the array are durably written to `data.raw`. The file is neither deleted nor truncated. If you need a guaranteed flush before `del`, call `m.flush()` explicitly. Mode `'c'` is the case where writes are lost on deletion, not `'r+'`.

---

## Q15 — Zarr Chunk Memory Calculation

> **Week reference:** Week 8

A Zarr array has shape `(2000, 4000)`, dtype `float64` (8 bytes per element), and chunk shape `(200, 400)`. How many bytes of RAM are loaded when a single chunk is accessed?

- A) 80 bytes
- B) 80 000 bytes
- C) 640 000 bytes
- D) 6 400 000 bytes

**Answer: C**

Chunk memory = chunk_rows × chunk_cols × bytes_per_element = 200 × 400 × 8 = 640 000 bytes (640 KB). Option A (80 bytes) would be 200 × 400 ÷ 1 000 — nonsensical. Option B (80 000 bytes) omits the 8-byte factor for float64, as if dtype were uint8. Option D (6 400 000 bytes) multiplies by 80 instead of 8 — a ×10 error.

---

## Q16 — When Is Memory Mapping Better Than Loading?

> **Week reference:** Week 8

You have a 50 GB float64 array on disk. Your algorithm reads a random 0.1% of elements. Which approach is most appropriate?

- A) Load the full array into RAM with `np.load`, then index it; 50 GB fits in typical HPC node memory.
- B) Use `np.memmap` in mode `'r'`; only the OS pages containing accessed elements are brought into RAM.
- C) Use `np.memmap` in mode `'c'`; copy-on-write prevents any disk reads.
- D) Split the file into 1 000 pieces and load each piece sequentially.

**Answer: B**

When only a small fraction of a large array is needed, memory mapping is ideal: the OS demand-pages only the 4 KB pages that contain accessed elements. For 0.1% of 50 GB, roughly 50 MB of pages are actually read — a 1000× reduction versus loading the whole file. Option A is wrong because loading 50 GB just to use 50 MB wastes enormous time and RAM. Option C is wrong because copy-on-write still reads from disk on page faults — it does not skip disk I/O. Option D is a valid but slower workaround that still loads far more data than needed.

---

## Q17 — Multiple Processes Sharing a Read-Only memmap

> **Week reference:** Week 8

Eight worker processes all open the same large file with `np.memmap('data.raw', mode='r', ...)`. What is true about physical RAM usage?

- A) Each process loads its own independent copy of the file, using 8× the file size in total RAM.
- B) The OS page cache is shared; all eight processes share the same physical RAM pages for file-backed read-only mappings.
- C) Only the first process to open the file gets a real mapping; the others receive empty arrays.
- D) Each process uses exactly one OS page (4 KB) regardless of how much data they access.

**Answer: B**

File-backed read-only memory mappings are backed by the OS page cache. When multiple processes map the same file with `MAP_SHARED` (which is what `mmap(2)` uses for file-backed maps), the kernel serves all of them from the same physical pages. There is no per-process duplication of data in RAM — the total RAM used is proportional to the pages actually accessed, not to the number of processes. Option A describes `np.load` behavior (each process loads a private copy). Option C is false — all processes get valid mappings. Option D confuses "pages loaded on open" (zero) with steady-state usage.

---

## Q18 — Choosing Between memmap and Loading for Sequential Access

> **Week reference:** Week 8

Your script reads a 2 GB float32 array sequentially from start to finish — every element, in order. Which statement best describes the trade-off between `np.memmap` and `np.load`?

- A) memmap is always faster because it avoids all disk I/O via the page cache.
- B) `np.load` is faster in this case because it reads the file in large sequential chunks; memmap's per-page demand-paging overhead adds latency.
- C) memmap is faster because it uses direct memory access (DMA) while `np.load` does not.
- D) Both take identical time because the total bytes read from disk is the same.

**Answer: B**

When every element is read sequentially, the total I/O is the same for both approaches, but `np.load` (via `np.fromfile`) uses large buffered reads that amortize syscall overhead. Memory mapping adds demand-paging overhead: the OS must handle a page fault for every new 4 KB page encountered, which adds minor but real overhead compared to a single large sequential read. For purely sequential full-file access, loading is typically equal or slightly faster. Memory mapping's advantage is for large files where only a subset is accessed. Option A is false — memmap does not avoid disk I/O; it defers it. Option C is false — both methods use DMA for disk transfers. Option D is close but ignores the differing overheads.

---

## Q19 — Mode 'c' After Flush

> **Week reference:** Week 8

A developer runs:
```python
m = np.memmap('ref.raw', mode='c', dtype='float64', shape=(500,))
m[0] = 42.0
m.flush()
del m
```
After this, what is the value at index 0 in `ref.raw`?

- A) 42.0 — `flush()` forces the write to disk.
- B) 0.0 (or original value) — mode `'c'` never writes to disk regardless of `flush()`.
- C) Undefined — calling `flush()` on a copy-on-write map raises a `ValueError`.
- D) 42.0 — `del` triggers a final write-back for copy-on-write maps.

**Answer: B**

Mode `'c'` (copy-on-write) maps the file privately. All writes go to anonymous RAM pages, never to the file-backed pages. Calling `flush()` on a mode `'c'` memmap is a no-op with respect to disk — it does not raise an error, but it also writes nothing to the file. The original `ref.raw` retains whatever value was at index 0 before the script ran. Neither `flush()` nor `del` changes this — the private pages are simply discarded. Options A and D are the classic exam trap: assuming `flush()` or `del` materialises copy-on-write writes to disk.

---

## Q20 — Zarr Chunk Count for a Given Array

> **Week reference:** Week 8

A Zarr array has shape `(600, 800)` and chunk shape `(100, 200)`. How many chunks does it contain in total?

- A) 24
- B) 48
- C) 6
- D) 12

**Answer: A**

Number of chunks = (600 ÷ 100) × (800 ÷ 200) = 6 × 4 = 24 total chunks. Each axis is divided independently: 6 chunks along the row axis, 4 chunks along the column axis. Option B (48) doubles the correct answer — a common error of multiplying (6 × 4 × 2). Option C (6) gives only the row-axis chunk count. Option D (12) halves the correct answer.

---

## Q21 — What Happens When memmap 'r+' File Does Not Exist?

> **Week reference:** Week 8

A script attempts:
```python
m = np.memmap('missing.raw', mode='r+', dtype='float32', shape=(100,))
```
The file `missing.raw` does not exist. What is the result?

- A) The file is created, filled with zeros, and opened for reading and writing.
- B) A `FileNotFoundError` is raised because mode `'r+'` requires the file to already exist.
- C) An empty array is returned with shape `(100,)` and all zeros.
- D) The operation succeeds silently, and `missing.raw` is created the first time a write is performed.

**Answer: B**

Mode `'r+'` maps directly to the POSIX `O_RDWR` flag without `O_CREAT` — the file must exist. If `missing.raw` is absent, the OS raises `FileNotFoundError` immediately when the file is opened. This is the key distinction between `'r+'` and `'w+'`: `'w+'` adds `O_CREAT | O_TRUNC` (create or truncate), while `'r+'` does not. Options A and D describe `'w+'` behavior. Option C is impossible — NumPy never silently returns zeros for a missing file in `'r+'` mode.

---

## Q22 — Zarr Chunk Shape for Column-by-Column Access

> **Week reference:** Week 8

You have a 512×1024 Zarr array (512 rows, 1024 columns) of int32. Your workload reads one full column at a time: `arr[:, j]` for each j. Which chunk shape minimises the number of chunks loaded per column access?

- A) `(1, 1024)` — row-aligned chunks
- B) `(512, 1)` — each chunk is a single full column
- C) `(64, 64)` — square chunks balance row and column access
- D) `(512, 1024)` — the entire array in one chunk

**Answer: B**

For column-by-column access, align chunks to the column dimension: `(512, 1)` means each chunk spans all 512 rows and exactly 1 column, so `arr[:, j]` loads exactly 1 chunk. Option A `(1, 1024)` is row-aligned — reading a full column would require loading 512 separate chunks (one per row). Option C `(64, 64)` requires 8 chunks per column (512 ÷ 64 = 8). Option D `(512, 1024)` loads the full array for every column access (1 chunk but 1024× the data needed). Note that `(512, 1)` chunk memory = 512 × 1 × 4 = 2 048 bytes per chunk — very small, so in practice a slightly wider column chunk like `(512, 8)` may be more efficient, but `(512, 1)` gives the minimum chunk count per access.

---

## Q23 — zarr.open Default Mode

> **Week reference:** Week 8

You call `z = zarr.open('results.zarr', shape=(500, 500), chunks=(50, 50), dtype='float32')` without specifying a `mode` argument. Which statement is correct?

- A) The call raises a `TypeError` because `mode` is a required argument.
- B) The store is opened in `'r'` (read-only) mode; writing to `z` will raise a `ValueError`.
- C) The store is opened in `'a'` (append/create) mode; it is created if absent, or opened read-write if it already exists without truncating data.
- D) The store is opened in `'w'` mode; if the file already exists it is silently overwritten.

**Answer: C**

- A) Incorrect — `mode` has a default value of `'a'` in zarr; omitting it is valid.
- B) Incorrect — `'r'` mode is not the default; read-only mode must be requested explicitly.
- C) Correct — zarr's default mode is `'a'` (append). It creates the store if it does not exist, or opens it read-write if it does, without destroying existing data. This is intentionally safe for interactive use.
- D) Incorrect — `'w'` would overwrite an existing store; the default `'a'` does not. If you need guaranteed-fresh creation, pass `mode='w'` explicitly.

---

## Q24 — zarr.open Mode 'w-' / 'x'

> **Week reference:** Week 8

A data pipeline writes results to `output.zarr`. You want the job to **fail immediately** if `output.zarr` already exists, to prevent silently overwriting previous results. Which `zarr.open` mode achieves this?

- A) `mode='w'`
- B) `mode='r+'`
- C) `mode='w-'` (equivalently `mode='x'`)
- D) `mode='a'`

**Answer: C**

- A) Incorrect — `'w'` creates or **overwrites** an existing store without any error. It provides no protection against accidental overwrites.
- B) Incorrect — `'r+'` opens an existing store read-write; it raises an error only if the store does *not* exist, which is the opposite of what is wanted.
- C) Correct — `'w-'` (also written `'x'`, like the POSIX `O_EXCL` flag) creates the store only if it does not already exist. If the path already exists, it raises a `ContainsArrayError` (or equivalent), making it safe for write-once pipeline outputs.
- D) Incorrect — `'a'` silently opens the existing store read-write; it never raises an error due to pre-existence.

---

## Q25 — zarr.open Mode Comparison: 'r+' vs 'a'

> **Week reference:** Week 8

Which of the following correctly distinguishes `zarr.open(..., mode='r+')` from `zarr.open(..., mode='a')`?

- A) `'r+'` creates the store if absent; `'a'` raises an error if the store is missing.
- B) `'r+'` raises an error if the store is absent; `'a'` creates the store if absent.
- C) Both modes create the store if absent; they differ only in whether existing data can be deleted.
- D) `'r+'` allows appending new arrays; `'a'` only allows reading.

**Answer: B**

- A) Incorrect — this has the behaviour of the two modes exactly reversed.
- B) Correct — `'r+'` requires the store to already exist (analogous to POSIX `O_RDWR` without `O_CREAT`); it raises an error on a missing path. `'a'` adds `O_CREAT` semantics — it creates the store if absent, then opens it read-write.
- C) Incorrect — `'r+'` does not create the store if absent; that is the key distinction from `'a'`.
- D) Incorrect — both modes allow writing; the distinction is purely about whether the store must pre-exist.

---
