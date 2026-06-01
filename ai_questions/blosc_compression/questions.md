# Blosc Compression — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — When Blosc Helps vs Hurts](#q1--when-blosc-helps-vs-hurts)
- [Q2 — LZ4 vs ZSTD Tradeoff](#q2--lz4-vs-zstd-tradeoff)
- [Q3 — SHUFFLE Filter Effect](#q3--shuffle-filter-effect)
- [Q4 — BITSHUFFLE vs SHUFFLE](#q4--bitshuffle-vs-shuffle)
- [Q5 — Zeros Array Compression Ratio](#q5--zeros-array-compression-ratio)
- [Q6 — Random Data and Blosc](#q6--random-data-and-blosc)
- [Q7 — blosc.pack_array API](#q7--bloscpack_array-api)
- [Q8 — nthreads and Blosc Parallelism](#q8--nthreads-and-blosc-parallelism)
- [Q9 — Chunk Size Effect on Compression](#q9--chunk-size-effect-on-compression)
- [Q10 — blosclz Codec Characteristics](#q10--blosclz-codec-characteristics)
- [Q11 — Compression Level 1 vs 9](#q11--compression-level-1-vs-9)
- [Q12 — Why os.sync() Is Used](#q12--why-ossync-is-used)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2--generated-practice-questions-exam-day-focus)
- [Q13 — Disk I/O Bottleneck and Blosc Benefit](#q13--disk-io-bottleneck-and-blosc-benefit)
- [Q14 — Tiled Array Compressibility](#q14--tiled-array-compressibility)
- [Q15 — zlib vs lz4 Speed vs Ratio](#q15--zlib-vs-lz4-speed-vs-ratio)
- [Q16 — NOSHUFFLE for Integer Data](#q16--noshuffle-for-integer-data)
- [Q17 — blosc.compress vs blosc.pack_array](#q17--blosccompress-vs-bloscpack_array)
- [Q18 — nthreads Default Value](#q18--nthreads-default-value)
- [Q19 — uint8 Data and Shuffle Filter](#q19--uint8-data-and-shuffle-filter)
- [Q20 — Compression Level 0 Meaning](#q20--compression-level-0-meaning)
- [Q21 — lz4hc Codec Purpose](#q21--lz4hc-codec-purpose)
- [Q22 — Blosc2 vs Blosc1 API Compatibility](#q22--blosc2-vs-blosc1-api-compatibility)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q23 — typesize Parameter Role](#q23--typesize-parameter-role)
- [Q24 — zstd Write Speed vs lz4 Tradeoff (Course Quiz Fact)](#q24--zstd-write-speed-vs-lz4-tradeoff-course-quiz-fact)
- [Q25 — Blosc Internal Chunk Default Size](#q25--blosc-internal-chunk-default-size)
- [Q26 — SHUFFLE Filter Integer Benefit](#q26--shuffle-filter-integer-benefit)
- [Q27 — blosc.compress shuffle Parameter Default](#q27--blosccompress-shuffle-parameter-default)
- [Q28 — Effect of clevel on Decompression Speed](#q28--effect-of-clevel-on-decompression-speed)
- [Q29 — snappy Codec Availability](#q29--snappy-codec-availability)
- [Q30 — Blosc Thread Safety in Multiprocessing](#q30--blosc-thread-safety-in-multiprocessing)
- [Q31 — Reading Back with Wrong Function](#q31--reading-back-with-wrong-function)
- [Q32 — Compressibility and Data Entropy Relationship](#q32--compressibility-and-data-entropy-relationship)

---

> Topics: Blosc codecs, shuffle filters, chunk sizing, compression ratio vs speed, nthreads.
> Exam frequency: **Week 3 topic**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--when-blosc-helps-vs-hurts)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 3 — Extended Practice](#set-3--extended-practice)

---

## Q1 — When Blosc Helps vs Hurts

> **Week reference:** Week 3

**Mental Model:** Blosc is worth using when disk I/O is the bottleneck and the data is compressible — fewer bytes read from disk outweighs the CPU cost of decompression. For incompressible data (e.g., uniformly random bytes), Blosc only adds overhead.

For which array type does Blosc compression provide the **greatest read speedup** compared to plain `np.load`?

- A) A large array of `float64` values drawn from `np.random.rand`
- B) A large array of `uint8` zeros
- C) A large array of `float32` values from a standard normal distribution
- D) A large array whose values are the output of `os.urandom`

**Answer: B**

- A) Incorrect — uniformly random float64 values have high entropy and compress very poorly. The compressed file is nearly the same size as the raw `.npy` file, so Blosc adds CPU overhead with negligible I/O savings.
- B) Correct — an array of all zeros is maximally compressible. The compressed file is tiny (near-zero bytes), so reading it is extremely fast. Decompression is trivial, so the total read time is far below that of reading a raw gigabyte-scale array.
- C) Incorrect — standard-normal float32 values are pseudo-random and have high entropy. They compress poorly, so Blosc gives little to no read speedup and may slow things down due to (de)compression overhead.
- D) Incorrect — `os.urandom` produces cryptographically random bytes, which are essentially incompressible. Blosc cannot reduce the file size at all, making it strictly slower than `np.load`.

---

## Q2 — LZ4 vs ZSTD Tradeoff

> **Week reference:** Week 3

**Mental Model:** LZ4 prioritises throughput (compress/decompress speed) while ZSTD prioritises compression ratio. ZSTD produces smaller files but is computationally heavier — the tradeoff depends on whether you are CPU-bound or I/O-bound.

You switch `cname="lz4"` to `cname="zstd"` in `blosc.pack_array`. Compared to LZ4, what does ZSTD typically do?

- A) Compresses faster but produces larger files
- B) Produces smaller files but takes more CPU time to compress and decompress
- C) Produces the same file size but uses more threads internally
- D) Decompresses faster but compresses slower than LZ4

**Answer: B**

- A) Incorrect — ZSTD is not faster than LZ4 at compression. LZ4 is specifically designed for maximum throughput; ZSTD trades speed for ratio.
- B) Correct — ZSTD achieves better compression ratios than LZ4, producing smaller on-disk files. The cost is higher CPU time for both the compress (write) and decompress (read) phases. This makes ZSTD the better choice when I/O is the bottleneck and CPU is plentiful.
- C) Incorrect — the number of threads Blosc uses is controlled separately via `blosc.set_nthreads`, not by the codec choice.
- D) Incorrect — LZ4 is generally faster than ZSTD at decompression too. ZSTD's advantage is ratio, not any speed metric relative to LZ4.

---

## Q3 — SHUFFLE Filter Effect

> **Week reference:** Week 3

**Mental Model:** The SHUFFLE filter rearranges bytes of multi-byte elements so that byte 0 of all elements comes first, then byte 1 of all elements, etc. This groups similar-valued bytes together, improving entropy coding by the subsequent codec.

Why does the SHUFFLE filter improve compression ratio for floating-point arrays?

- A) It sorts the array values in ascending order before compressing
- B) It groups together bytes at the same significance position across all elements, making runs of similar bytes longer and more compressible
- C) It converts float64 values to float32 before passing them to the codec
- D) It removes duplicate values from the array to reduce data volume

**Answer: B**

- A) Incorrect — SHUFFLE does not reorder the elements; it reorders the bytes within the packed binary representation. The element order is preserved.
- B) Correct — for a float64 array, SHUFFLE interleaves the 8 bytes of each element so that all "byte 0" values appear consecutively, then all "byte 1" values, etc. The most-significant bytes of floating-point numbers tend to change slowly (exponent bits are often identical), so grouping them creates long runs that LZ4 or ZSTD can compress heavily.
- C) Incorrect — SHUFFLE is a byte-level transformation only; it does not change the data type or truncate precision.
- D) Incorrect — SHUFFLE is a lossless byte reordering. It does not remove or deduplicate values.

---

## Q4 — BITSHUFFLE vs SHUFFLE

> **Week reference:** Week 3

**Mental Model:** BITSHUFFLE extends SHUFFLE to the bit level — it groups bit `n` of all elements together. This is more effective for data with many identical or slowly-varying bits, but has higher CPU overhead than SHUFFLE.

Which statement best describes when BITSHUFFLE outperforms SHUFFLE?

- A) When the array has dtype `uint8`, because uint8 values have no byte-level structure
- B) When the array contains floating-point data with many near-identical values (e.g., a nearly-constant array), maximising identical-bit runs
- C) When the array is already compressed and BITSHUFFLE avoids recompressing it
- D) When using the `blosclz` codec, which requires bit-level input

**Answer: B**

- A) Incorrect — for `uint8` data, each element is only 1 byte, so SHUFFLE and BITSHUFFLE operate on the same granularity. BITSHUFFLE has higher CPU overhead and provides no meaningful advantage over SHUFFLE for single-byte dtypes.
- B) Correct — BITSHUFFLE is most effective when data has many identical bits across elements, such as floating-point values that are nearly equal (shared sign, exponent, and upper mantissa bits). Grouping all bit-0 values, all bit-1 values, etc., can create extremely long runs of identical bits, which delta-coding or entropy coding handles very efficiently.
- C) Incorrect — BITSHUFFLE is applied before the codec in the compression pipeline. It is not a post-processing step for already-compressed data.
- D) Incorrect — the shuffle filter is independent of the codec selection. Any codec (`lz4`, `zstd`, `blosclz`, etc.) can be combined with any shuffle mode.

---

## Q5 — Zeros Array Compression Ratio

> **Week reference:** Week 3

**Mental Model:** Compression ratio = uncompressed size / compressed size. All-zeros data compresses to a constant number of bytes (essentially a run-length description of "N zeros"), so very large arrays yield astronomical ratios.

A 512 × 512 × 512 `uint8` array of zeros occupies 128 MiB uncompressed. After `blosc.pack_array` with `cname="lz4"`, the file is approximately 1 KiB. What is the approximate compression ratio?

- A) 2:1
- B) 100:1
- C) ~131,000:1
- D) 1:1 (LZ4 cannot compress integer arrays)

**Answer: C**

- A) Incorrect — a 2:1 ratio means the compressed file is 64 MiB. An all-zeros array compresses far more aggressively than that because it has zero entropy.
- B) Incorrect — 100:1 would mean a ~1.28 MiB compressed file. While impressive for typical data, all-zeros data achieves far better ratios because every byte is identical.
- C) Correct — 128 MiB = 134,217,728 bytes. At ~1 KiB (1,024 bytes) compressed, the ratio is ~131,072:1. All-zeros is the best case for any lossless compressor because the entire content can be described in a handful of bytes.
- D) Incorrect — LZ4 operates on arbitrary byte streams, including integer arrays. It compresses by finding repeated byte sequences, and an all-zeros stream is the ultimate repeated sequence.

---

## Q6 — Random Data and Blosc

> **Week reference:** Week 3

**Mental Model:** Maximum-entropy data (uniform random) has no repeated patterns — every byte is independent and unpredictable. No lossless compressor can reduce it; some even expand it slightly due to header overhead.

You call `blosc.pack_array` on a 1024 × 1024 × 1024 `uint8` array filled with `np.random.randint(1, 256)`. What do you expect?

- A) Blosc compresses it to around 50% of its original size because LZ4 is very efficient
- B) The compressed output is approximately the same size as or slightly larger than the uncompressed array
- C) Blosc raises a `ValueError` because it cannot compress random data
- D) The file is smaller because Blosc removes statistically unlikely byte sequences

**Answer: B**

- A) Incorrect — uniformly random data has maximum entropy (~8 bits per byte). LZ4 finds no repeated patterns to compress, so the ratio is approximately 1:1.
- B) Correct — for near-maximum-entropy data, Blosc achieves a compression ratio of ~1:1. The compressed file may even be fractionally larger than the raw array due to the Blosc frame header and block-level metadata overhead.
- C) Incorrect — Blosc does not raise an error for incompressible data. It simply stores the data with no reduction (or passes through uncompressed blocks when compression would expand them).
- D) Incorrect — lossless compressors never discard data. They can only rearrange or represent it more compactly when structure exists.

---

## Q7 — blosc.pack_array API

> **Week reference:** Week 3

**Mental Model:** `blosc.pack_array` is a convenience function that serialises a NumPy array (including its dtype, shape, and strides) into a bytes object ready to write to disk. The reverse is `blosc.unpack_array`, which reconstructs the original array.

What does `blosc.pack_array(arr, cname="lz4")` return?

- A) A compressed file path as a string
- B) A `bytes` object containing the Blosc-compressed representation of the array, including metadata
- C) A new NumPy array with dtype `uint8` representing the compressed bytes
- D) A dictionary with keys `"data"` and `"meta"` holding compressed data and shape information

**Answer: B**

- A) Incorrect — `blosc.pack_array` does not write any files. It returns a Python object; the caller is responsible for writing it to disk (e.g., with `open(..., "wb").write(...)`).
- B) Correct — `pack_array` returns a `bytes` object. This bytes object encodes the Blosc frame, which includes the array's shape, dtype, strides, and the compressed data blocks. Passing it to `blosc.unpack_array` reconstructs the original array exactly.
- C) Incorrect — the return value is a Python `bytes` object, not a NumPy array. NumPy arrays and Python bytes objects are distinct types.
- D) Incorrect — `pack_array` returns a single `bytes` object, not a dictionary. Shape and dtype information is embedded in the Blosc frame header, not exposed as a separate dict.

---

## Q8 — nthreads and Blosc Parallelism

> **Week reference:** Week 3

**Mental Model:** Blosc splits data into chunks and compresses/decompresses each chunk in parallel using its internal thread pool. `blosc.set_nthreads(n)` controls how many threads it uses. More threads help when chunks are large enough to amortise thread overhead.

Which call correctly sets Blosc to use 4 threads for all subsequent operations?

- A) `blosc.nthreads = 4`
- B) `blosc.set_nthreads(4)`
- C) `blosc.compress(data, nthreads=4)`
- D) `import blosc; blosc.BLOSC_MAX_THREADS = 4`

**Answer: B**

- A) Incorrect — `blosc.nthreads` is not a settable module-level attribute in the Blosc Python API. Assigning to it silently creates a new attribute but does not affect the internal thread pool.
- B) Correct — `blosc.set_nthreads(n)` is the documented API function to configure the Blosc thread pool. It returns the previous thread count and takes effect for all subsequent `compress`, `decompress`, `pack_array`, and `unpack_array` calls.
- C) Incorrect — `blosc.compress` does not accept an `nthreads` keyword argument. Thread count is a global setting, not per-call.
- D) Incorrect — `BLOSC_MAX_THREADS` is a C-level compile-time constant that caps the maximum allowable thread count. It is not a runtime setter for the active thread count.

---

## Q9 — Chunk Size Effect on Compression

> **Week reference:** Week 3

**Mental Model:** Blosc compresses data in fixed-size chunks. Chunks must fit in CPU cache for best performance — too small and thread overhead dominates; too large and cache thrashing hurts. The default chunk size is chosen to fit in L2/L3 cache.

What is the primary consequence of setting Blosc's chunk size too small (e.g., 512 bytes per chunk)?

- A) Blosc raises a `RuntimeError` because the minimum chunk size is 64 KB
- B) Compression ratio improves because each chunk is independently optimal
- C) Thread scheduling and per-chunk metadata overhead dominate, reducing throughput even though data fits in L1 cache
- D) The shuffle filter is automatically disabled because it requires at least one full cache line

**Answer: C**

- A) Incorrect — Blosc does not raise an error for small chunk sizes. It accepts user-specified sizes, though performance may degrade.
- B) Incorrect — smaller chunks generally worsen compression ratio because each chunk is compressed independently without context from neighbouring chunks. Cross-chunk patterns cannot be exploited.
- C) Correct — for very small chunks, the per-chunk overhead (thread synchronisation, block header writes, metadata) constitutes a large fraction of the work per chunk. There are also more function calls and more context switches, reducing the effective throughput significantly.
- D) Incorrect — the shuffle filter is not automatically disabled based on chunk size. It operates byte-by-byte within whatever chunk size is set.

---

## Q10 — blosclz Codec Characteristics

> **Week reference:** Week 3

**Mental Model:** `blosclz` is Blosc's default built-in codec, derived from FastLZ. It is tuned for speed at compression and decompression, with a compression ratio comparable to LZ4. It is not as fast as LZ4 in modern benchmarks but requires no external dependency.

Which statement about the `blosclz` codec is most accurate?

- A) `blosclz` is a lossless codec based on the Zstandard algorithm optimised for scientific arrays
- B) `blosclz` is Blosc's default codec, derived from FastLZ, prioritising compression and decompression speed over maximum compression ratio
- C) `blosclz` achieves the highest compression ratios of all Blosc codecs for floating-point data
- D) `blosclz` is not available in the Python `blosc` package and must be installed separately

**Answer: B**

- A) Incorrect — `blosclz` is based on FastLZ, not Zstandard. Zstandard is a separate algorithm available as `cname="zstd"`.
- B) Correct — `blosclz` is the original default codec in Blosc, adapted from FastLZ. It prioritises speed (both compression and decompression) over maximising compression ratio. It is suitable when CPU time is at a premium and moderate compression is sufficient.
- C) Incorrect — for floating-point data, `zstd` (especially at higher compression levels) typically achieves better ratios than `blosclz`. `lz4hc` also outperforms `blosclz` at ratio.
- D) Incorrect — `blosclz` is compiled into the Blosc C library itself and is available whenever the `blosc` Python package is installed. No additional installation is required.

---

## Q11 — Compression Level 1 vs 9

> **Week reference:** Week 3

**Mental Model:** Compression level (1–9) controls the effort the codec expends searching for patterns. Level 1 is fastest with the worst ratio; level 9 takes the most CPU time for the best ratio. For I/O-bound workloads, a moderate level (e.g., 5) often gives the best end-to-end throughput.

In `blosc.compress(data, clevel=9, cname="lz4")` vs `clevel=1`, what is the main tradeoff?

- A) `clevel=9` uses 9 threads; `clevel=1` uses 1 thread
- B) `clevel=9` applies BITSHUFFLE; `clevel=1` applies NOSHUFFLE
- C) `clevel=9` spends more CPU time searching for repeated patterns, yielding a smaller compressed output; `clevel=1` compresses quickly with a larger output
- D) `clevel=9` compresses data lossily to achieve smaller files; `clevel=1` is always lossless

**Answer: C**

- A) Incorrect — compression level has nothing to do with thread count. Thread count is controlled exclusively by `blosc.set_nthreads`.
- B) Incorrect — the shuffle filter (`shuffle` parameter) is independent of `clevel`. Both `clevel=1` and `clevel=9` use the same shuffle mode unless you change the `shuffle` argument.
- C) Correct — `clevel` controls how hard the codec works to find compressible patterns. Higher levels search more thoroughly (e.g., with a longer match window in LZ4), producing smaller files at the cost of more CPU time. Decompression speed is usually not significantly affected by `clevel`.
- D) Incorrect — Blosc is always lossless regardless of compression level. `clevel=9` does not lose any data; it simply invests more CPU cycles to find a more compact lossless representation.

---

## Q12 — Why os.sync() Is Used

> **Week reference:** Week 3

**Mental Model:** Without `os.sync()`, the OS may cache writes in its page cache and return from `write()` before the data hits the disk. Timing "write + sync" gives an honest measure of disk write latency rather than memory-copy latency.

In the course exercises, `os.sync()` is called after writing a file. Why is this necessary for accurate benchmarking?

- A) It forces NumPy to release its internal array buffer so memory is freed before timing the next operation
- B) It flushes the OS page cache to disk, ensuring that the measured time includes actual disk I/O and not just in-memory buffering
- C) It synchronises all Blosc threads so the compression result is consistent across runs
- D) It prevents the OS from caching the read that follows, so the read benchmark also measures disk access

**Answer: B**

- A) Incorrect — `os.sync()` is a system call that flushes dirty kernel buffers to storage. It has no direct effect on NumPy's memory allocator or array buffers.
- B) Correct — the OS write path buffers data in memory before writing it to disk. `os.sync()` blocks until all buffered writes are committed to the storage device, so the elapsed time from `open().write()` to after `os.sync()` reflects true disk write latency.
- C) Incorrect — Blosc thread synchronisation is handled internally by the Blosc library. `os.sync()` is a kernel-level operation unrelated to Blosc's thread pool.
- D) Incorrect — `os.sync()` flushes writes; it does not drop the read cache (that would require `echo 3 > /proc/sys/vm/drop_caches` on Linux, which requires root). Subsequent reads may still be served from the OS page cache.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

---

## Q13 — Disk I/O Bottleneck and Blosc Benefit

> **Week reference:** Week 3

**Mental Model:** Blosc is a speed-oriented compressor — its primary benefit is reducing bytes transferred from disk, not minimising file size at all costs. The break-even point is when the time saved on I/O exceeds the time spent on (de)compression.

For Blosc to make reading data faster than `np.load`, which condition must hold?

- A) The array must have dtype `float64`, because Blosc is only optimised for 8-byte elements
- B) The time saved by reading fewer bytes from disk must exceed the CPU time added by decompression
- C) Blosc must be run with at least 2 threads, otherwise decompression is slower than disk I/O
- D) The array must fit entirely in RAM, otherwise Blosc falls back to `np.memmap`

**Answer: B**

- A) Incorrect — Blosc works with any dtype. The SHUFFLE filter is most effective for multi-byte types, but the core benefit (fewer bytes read from disk) applies to any compressible data regardless of dtype.
- B) Correct — this is the fundamental tradeoff. Disk bandwidth is far lower than CPU/RAM bandwidth. If the data compresses well, reading fewer compressed bytes and decompressing in fast RAM is faster than reading all uncompressed bytes from the slow disk.
- C) Incorrect — single-threaded Blosc decompression can still be faster than disk I/O for highly compressible data. Thread count affects throughput but is not a hard requirement for the speedup condition.
- D) Incorrect — Blosc operates on in-memory bytes objects. It does not interact with `np.memmap` and has no requirement about whether the array fits in RAM.

---

## Q14 — Tiled Array Compressibility

> **Week reference:** Week 3

**Mental Model:** A tiled array repeating `[0, 1, 2, ..., 255, 0, 1, ...]` is highly structured — the same 256-byte pattern repeats millions of times. LZ4 and similar codecs excel at this: after the first occurrence, each subsequent tile is a back-reference to a previous match.

The exercise creates a tiled array with:
```python
np.tile(np.arange(256, dtype='uint8'), (n // 256) * n * n).reshape(n, n, n)
```
Compared to a zeros array of the same size and codec, how does this array's compression ratio compare?

- A) Better than zeros, because the repeating pattern has more structure
- B) Worse than zeros but still much better than random data, because the repeating 256-byte tile is easy to back-reference
- C) Identical to zeros, because all uint8 values have the same bit width
- D) Worse than random data, because alternating ascending values confuse the codec

**Answer: B**

- A) Incorrect — an all-zeros array is maximally compressible: the entire content is a single run of one value. The tiled array's 256-byte period requires the compressor to store one full period and then back-reference it, which is very efficient but not as extreme as all-zeros.
- B) Correct — the tiled pattern is highly compressible because after the first 256 bytes, every subsequent tile is an exact copy of those bytes. A run-length or LZ back-reference encoder encodes this as "repeat this block N times," yielding a very high ratio — just not quite as high as all-zeros.
- C) Incorrect — compression ratio depends on data entropy and pattern structure, not on dtype element size. An all-zeros uint8 array is far more compressible than a tiled uint8 array.
- D) Incorrect — regular repeating patterns are ideal for dictionary-based codecs like LZ4. The ascending sequence `[0..255]` is not confusing; it is a short, predictable pattern that compressors handle very well.

---

## Q15 — zlib vs lz4 Speed vs Ratio

> **Week reference:** Week 3

**Mental Model:** `zlib` (DEFLATE) achieves better compression ratios than `lz4` but is much slower at both compression and decompression. `lz4` is optimised for throughput, making it the default in Blosc for I/O-speed use cases.

You need to store a large array that will be read thousands of times per day. Storage space is plentiful but read latency must be minimised. Which codec is the best choice?

- A) `zlib`, because higher compression ratio means fewer bytes to read
- B) `lz4`, because its very fast decompression minimises the CPU portion of read latency
- C) `lz4hc`, because it achieves maximum compression like zlib but decompresses as fast as lz4
- D) `blosclz`, because it is the only codec that supports multi-threaded decompression

**Answer: B**

- A) Incorrect — for many reads per day, decompression speed matters as much as or more than file size. `zlib`'s slow decompression would add significant latency on every read even if the file is smaller.
- B) Correct — `lz4` decompresses at rates comparable to memory bandwidth (~1–4 GB/s per thread). For frequent reads, the ultra-fast decompression keeps the CPU-side latency minimal. If the ratio is still sufficient to keep the file smaller than the raw array, `lz4` wins overall.
- C) Incorrect — `lz4hc` is a high-compression variant of LZ4: it achieves better ratios than plain LZ4 but at the cost of slower compression (write-time). Its decompression speed is similar to LZ4 since the decompressor logic is the same. However, if storage is plentiful, lz4hc's higher write cost without further read benefit makes lz4 still preferable here.
- D) Incorrect — all Blosc codecs support multi-threaded operation via Blosc's internal chunk-parallel framework. `blosclz` is not special in this regard.

---

## Q16 — NOSHUFFLE for Integer Data

> **Week reference:** Week 3

**Mental Model:** The SHUFFLE filter is most beneficial for multi-byte dtypes where byte significance varies strongly (e.g., float32, float64). For `uint8` data, each element is 1 byte, so SHUFFLE has no effect — there is nothing to interleave across bytes within a single element.

You have a `uint8` array and try `blosc.compress(data, shuffle=blosc.SHUFFLE)` versus `shuffle=blosc.NOSHUFFLE`. What do you expect?

- A) SHUFFLE greatly improves compression ratio because it rearranges elements into ascending order
- B) The compression ratio is identical for both, because SHUFFLE on a 1-byte dtype has no effect
- C) NOSHUFFLE always gives better ratio for integer arrays because the codec works directly on raw values
- D) SHUFFLE raises a `TypeError` because it requires at least 2 bytes per element

**Answer: B**

- A) Incorrect — SHUFFLE does not sort elements. It reorders bytes within multi-byte elements to group bytes of the same significance. For 1-byte elements, there is only one byte per element, so the operation is a no-op.
- B) Correct — for `uint8` (1 byte per element), the SHUFFLE filter has nothing to interleave. The input to the codec is identical whether SHUFFLE or NOSHUFFLE is specified. Compression ratio and speed are therefore the same.
- C) Incorrect — for integer types wider than 1 byte (e.g., `int32`, `int64`), SHUFFLE often improves ratio by grouping the slowly-varying high bytes. The claim that NOSHUFFLE is universally better for integers is false.
- D) Incorrect — `blosc.compress` does not raise a `TypeError` for `uint8` data with `SHUFFLE`. It silently applies a no-op shuffle and proceeds normally.

---

## Q17 — blosc.compress vs blosc.pack_array

> **Week reference:** Week 3

**Mental Model:** `blosc.compress` operates on raw bytes and produces a Blosc-compressed buffer without NumPy metadata. `blosc.pack_array` wraps `blosc.compress` and also pickles NumPy shape/dtype/strides metadata, making round-trip reconstruction with `unpack_array` straightforward.

What is the key difference between `blosc.compress(data.tobytes())` and `blosc.pack_array(data)`?

- A) `compress` is faster because it skips computing the shuffle filter
- B) `pack_array` also stores the array's shape, dtype, and strides, enabling exact reconstruction via `unpack_array`; `compress` only stores raw bytes
- C) `pack_array` supports more codecs than `compress`
- D) `compress` stores the result in a temporary file; `pack_array` returns bytes in memory

**Answer: B**

- A) Incorrect — both functions apply the same shuffle filter pipeline. `pack_array` does add a small overhead for pickling metadata, but both use the same Blosc compression core.
- B) Correct — `pack_array` stores NumPy array metadata (shape, dtype, strides) alongside the compressed data so that `unpack_array` can reconstruct the exact original array object. `compress` returns a Blosc frame of the raw bytes; the caller must track shape and dtype separately.
- C) Incorrect — both `compress` and `pack_array` support the same set of codecs: `blosclz`, `lz4`, `lz4hc`, `zstd`, `zlib`, `snappy` (if available). Codec availability is a library-level feature, not function-specific.
- D) Incorrect — both `blosc.compress` and `blosc.pack_array` return Python `bytes` objects in memory. Neither writes files automatically.

---

## Q18 — nthreads Default Value

> **Week reference:** Week 3

**Mental Model:** Blosc defaults to using 1 thread for safety (to avoid unexpected parallelism in multi-process workloads). You must explicitly call `blosc.set_nthreads(n)` to enable multi-threaded compression/decompression.

What is the default number of threads Blosc uses if you never call `blosc.set_nthreads`?

- A) The number of logical CPU cores on the machine
- B) 4 threads
- C) 1 thread
- D) 2 threads

**Answer: C**

- A) Incorrect — Blosc does not automatically detect and use all available cores by default. This would be unsafe in multi-process environments (e.g., when Blosc is used inside a multiprocessing worker). The user must opt in to parallelism.
- B) Incorrect — 4 is not the default. Some users assume Blosc matches common thread-pool defaults, but the Blosc library defaults to single-threaded operation.
- C) Correct — Blosc defaults to 1 thread. This ensures deterministic, safe behaviour in all environments. To leverage multi-core compression, the user must explicitly call `blosc.set_nthreads(n)` where `n > 1`.
- D) Incorrect — the default is 1 thread, not 2.

---

## Q19 — uint8 Data and Shuffle Filter

> **Week reference:** Week 3

**Mental Model:** Choosing the right shuffle filter requires knowing the dtype. For multi-byte types like `float64`, SHUFFLE rearranges the 8 bytes per element to group significance levels; for `uint8` (1 byte), no rearrangement is possible and SHUFFLE is a no-op.

In the course exercise, the Blosc benchmark uses `dtype='uint8'` arrays. Why might BITSHUFFLE be more useful than SHUFFLE for this dtype specifically?

- A) BITSHUFFLE sorts the array values in place, while SHUFFLE does not sort anything
- B) BITSHUFFLE operates at the bit level, so it can still group bits of equal significance across elements even for 1-byte dtypes where SHUFFLE is a no-op
- C) BITSHUFFLE is required for all integer dtypes when using the `zstd` codec
- D) BITSHUFFLE halves the apparent dtype width from `uint8` to `uint4`, reducing data volume

**Answer: B**

- A) Incorrect — neither SHUFFLE nor BITSHUFFLE sort array values. Both are byte- or bit-level reorderings that group corresponding positional bits/bytes together.
- B) Correct — for `uint8`, SHUFFLE has nothing to transpose at the byte level (each element is already 1 byte). BITSHUFFLE, however, transposes at the bit level: bit 7 of all elements, then bit 6, etc. For structured uint8 data (e.g., values clustered in a narrow range), the high bits may be highly correlated, and BITSHUFFLE can expose this for the downstream codec.
- C) Incorrect — BITSHUFFLE is independent of codec choice. It can be combined with any Blosc codec.
- D) Incorrect — BITSHUFFLE is a lossless reordering and does not change the dtype, the number of bits, or the data volume before passing to the codec.

---

## Q20 — Compression Level 0 Meaning

> **Week reference:** Week 3

**Mental Model:** Compression level 0 is a special value in Blosc meaning "no compression" — data is stored as-is in the Blosc frame. This is useful when you want Blosc's chunking and shuffling infrastructure without the compression overhead.

What does `blosc.compress(data, clevel=0)` do?

- A) Uses the minimum non-zero compression effort, equivalent to `clevel=1`
- B) Raises a `ValueError` because 0 is not a valid compression level
- C) Stores data uncompressed inside the Blosc frame — shuffle filters may still be applied, but no codec compression is performed
- D) Compresses data with ZLIB at level 0, which applies Huffman coding only

**Answer: C**

- A) Incorrect — `clevel=0` is not the same as `clevel=1`. It is a distinct mode that bypasses codec compression entirely.
- B) Incorrect — `clevel=0` is a valid and documented value. Blosc treats it as "store without compression."
- C) Correct — with `clevel=0`, Blosc skips the codec compression step. The shuffle filter can still be applied (the data is still byte-shuffled if `shuffle != NOSHUFFLE`), but the shuffled bytes are stored directly in the Blosc frame without running them through LZ4, ZSTD, or any other codec.
- D) Incorrect — `clevel=0` bypasses all codec compression. It is not a ZLIB-specific mode. ZLIB is only used when `cname="zlib"` is specified.

---

## Q21 — lz4hc Codec Purpose

> **Week reference:** Week 3

**Mental Model:** `lz4hc` is the "High Compression" variant of LZ4. It uses a slower, more exhaustive pattern-matching algorithm at compression time but decompresses at the same speed as plain LZ4. It is ideal for write-once, read-many workloads where write time is not critical.

When is `cname="lz4hc"` a better choice than `cname="lz4"`?

- A) When you need faster compression than LZ4 provides
- B) When you write data once and read it many times, and storage space is more constrained than write time
- C) When the data contains random floating-point values that LZ4 cannot compress
- D) When using more than 4 threads, because lz4hc scales better with thread count

**Answer: B**

- A) Incorrect — `lz4hc` is slower to compress than plain `lz4`. It trades compression speed for compression ratio.
- B) Correct — `lz4hc` is the right choice for write-once, read-many scenarios where you want the smallest possible file (to minimise disk reads) but can afford a longer write phase. Decompression speed is identical to LZ4, so read performance is the same or better (due to smaller files).
- C) Incorrect — `lz4hc` cannot compress genuinely random data any better than `lz4`. If the data has no structure, neither variant helps.
- D) Incorrect — thread scaling in Blosc is determined by the chunk-parallel framework, which is the same for all codecs. `lz4hc` does not scale differently with thread count than `lz4`.

---

## Q22 — Blosc2 vs Blosc1 API Compatibility

> **Week reference:** Week 3

**Mental Model:** `blosc2` is the successor to the original `blosc` package. The top-level API for basic operations (`compress`, `decompress`, `pack_array`, `unpack_array`) is largely backwards-compatible, but `blosc2` adds new features (super-chunks, two-level compression, improved codecs) and has a different package name.

Which statement about using `import blosc2` vs `import blosc` in the course exercises is most accurate?

- A) `blosc2` is completely incompatible with `blosc`; you must rewrite all `pack_array` calls
- B) `blosc2` provides a `blosc2.compress`/`blosc2.decompress` API that is largely compatible with `blosc`, but adds features like NDim arrays and super-chunks not available in `blosc`
- C) `blosc2` only supports the `zstd` codec; all other codecs require the original `blosc` package
- D) `blosc2` removes multi-threading support and requires manual chunking instead

**Answer: B**

- A) Incorrect — `blosc2` maintains backwards compatibility for the core compression API. Basic calls like `blosc2.compress`, `blosc2.decompress`, and their equivalents work similarly to `blosc`'s API. No complete rewrite is required for simple use cases.
- B) Correct — `blosc2` is a drop-in replacement for most `blosc` usage, but it extends the API with new abstractions: `SChunk` (super-chunks supporting multiple compression levels), native N-dimensional array support, and improved codec selection. The basic `compress`/`decompress` interface is compatible.
- C) Incorrect — `blosc2` supports all the same codecs as `blosc` (`blosclz`, `lz4`, `lz4hc`, `zstd`, `zlib`) and adds new ones. It does not restrict codec availability.
- D) Incorrect — `blosc2` retains and improves Blosc's internal parallelism. Multi-threading is still controlled via `blosc2.set_nthreads` (or equivalents) and is a core feature of the library.

---

## Set 3 — Extended Practice

---

## Q23 — typesize Parameter Role

> **Week reference:** Week 3

**Mental Model:** The `typesize` parameter in `blosc.compress` tells the SHUFFLE filter how many bytes make up one element. SHUFFLE needs this to know the stride at which to reorder bytes. Passing the wrong `typesize` causes the shuffle to be applied incorrectly, degrading compression ratio.

Which value should `typesize` be set to when compressing the raw bytes of a `float64` array?

- A) 1 — because `tobytes()` produces a flat `bytes` object of individual bytes
- B) 4 — because Blosc internally converts float64 to float32 before shuffling
- C) 8 — because each `float64` element occupies 8 bytes, and SHUFFLE uses this stride to group bytes by significance
- D) 64 — because `typesize` is the total size of the array's dtype in bits

**Answer: C**

- A) Incorrect — `tobytes()` does produce a flat byte stream, but `typesize` does not describe the output of `tobytes()`; it describes the size of the original element so that SHUFFLE can correctly interleave bytes at 8-byte strides. Passing `typesize=1` tells SHUFFLE there is nothing to interleave and effectively disables it.
- B) Incorrect — Blosc never converts dtype silently. The caller controls dtype entirely. `float64` stays `float64` throughout; `typesize=4` would mis-stride the shuffle and corrupt the grouping.
- C) Correct — `float64` is 8 bytes per element (`np.dtype('float64').itemsize == 8`). Setting `typesize=8` tells the SHUFFLE filter to treat every 8 consecutive bytes as one logical element and transpose bytes at that granularity, grouping all sign/exponent bytes together for maximum compressibility.
- D) Incorrect — `typesize` is measured in bytes, not bits. A 64-bit float is 8 bytes; passing `typesize=64` would tell Blosc each element is 64 bytes wide, which is nonsensical and would disable effective shuffling.

---

## Q24 — zstd Write Speed vs lz4 Tradeoff (Course Quiz Fact)

> **Week reference:** Week 3

**Mental Model:** The Week 3 quiz result confirms: switching from lz4 to zstd makes writing (compression) slower but reading (decompression) roughly the same speed, while producing smaller files. This is the canonical tradeoff to know for the exam.

According to the Week 3 exercise results, what changes when switching from `cname="lz4"` to `cname="zstd"` for zeros and tiled data?

- A) Both reading and writing are faster, and file size is smaller
- B) Reading is slower, writing is the same speed, file size is smaller
- C) Reading is about the same speed, writing is slower, and file size is smaller
- D) Reading is faster, writing is slower, and file size is larger

**Answer: C**

- A) Incorrect — writing (compression) is not faster with zstd; it is slower because zstd performs more exhaustive pattern matching. Reading is not faster either — decompression speeds for lz4 and zstd are comparable.
- B) Incorrect — reading speed is similar between lz4 and zstd for decompression. The bottleneck difference is at the compression (write) stage, not decompression (read).
- C) Correct — this is the documented quiz answer for the course exercise. ZSTD's more thorough encoding takes longer to compress (slower writes) but decompresses at a similar speed to LZ4 (same reads). The payoff is smaller files on disk.
- D) Incorrect — zstd produces smaller files than lz4, not larger. That smaller size is the entire motivation for accepting the slower write speed.

---

## Q25 — Blosc Internal Chunk Default Size

> **Week reference:** Week 3

**Mental Model:** Blosc splits input data into chunks before compressing them in parallel. The default chunk size is chosen to fit in the CPU's L2/L3 cache — typically around 256 KB to 1 MB. Chunks that fit in cache allow the codec to run at near-memory-bandwidth speeds.

What is the rationale behind Blosc's default chunk size being in the range of hundreds of kilobytes?

- A) The chunk size must equal the POSIX page size (4 KiB) for alignment with OS memory management
- B) Chunks are sized to fit in CPU L2/L3 cache, so that each chunk can be compressed/decompressed entirely in fast cache memory without hitting RAM bandwidth
- C) The chunk size must be at least 1 GiB so that Blosc's thread pool can process the entire array in one pass
- D) Chunks must be a power of two in bytes because the SHUFFLE filter requires power-of-two strides

**Answer: B**

- A) Incorrect — 4 KiB chunks would be far too small for efficient Blosc operation. While 4 KiB is the OS page size, Blosc's design goal is cache residency at the L2/L3 level, not page alignment.
- B) Correct — Blosc is designed as a "blocker, shuffler, and compressor." The blocking step (chunking) is specifically sized so each chunk fits in L2 or L3 cache. When the compressor operates on cache-resident data, it runs at cache bandwidth (~100–500 GB/s) rather than RAM bandwidth (~20–50 GB/s), making compression fast enough to exceed disk bandwidth.
- C) Incorrect — 1 GiB chunks would defeat the purpose of chunking: they would not fit in cache, and the entire benefit of chunk-level cache residency would be lost. Smaller chunks are better for cache utilisation.
- D) Incorrect — the SHUFFLE filter does not require power-of-two chunk sizes. It requires only that `typesize` is specified correctly. Chunk size is a separate parameter.

---

## Q26 — SHUFFLE Filter Integer Benefit

> **Week reference:** Week 3

**Mental Model:** SHUFFLE is most beneficial for multi-byte types where upper bytes (more significant) are nearly constant across elements. For wide integer types like `int32` or `int64` storing small values (e.g., counts or IDs), the upper bytes are all zeros, creating long runs after shuffling.

For which `int32` array does the SHUFFLE filter provide the most compression benefit?

- A) An array of values drawn from `np.random.randint(0, 2**31)`
- B) An array of consecutive small integers `[0, 1, 2, 3, ..., N]` where N < 65536
- C) An array where each element equals `2**31 - 1` (max int32 value)
- D) An array of values drawn from a uniform distribution over the full int32 range

**Answer: B**

- A) Incorrect — values spread uniformly across the full int32 range use all 4 bytes heavily and randomly. After SHUFFLE, the byte streams are still random with no long runs. SHUFFLE provides minimal benefit.
- B) Correct — for small consecutive integers (N < 65536), the upper 2 bytes of each int32 element are all zeros. After SHUFFLE groups the upper bytes together, the codec sees a long run of 0x00 bytes followed by the slowly-varying lower bytes. This is highly compressible.
- C) Incorrect — an array of all identical values (all `2**31 - 1`) is compressible with or without SHUFFLE — the entire array is the same repeated bytes. SHUFFLE does not provide additional benefit over NOSHUFFLE for constant data.
- D) Incorrect — a full-range uniform distribution produces high-entropy bytes in all 4 positions. After SHUFFLE, each byte stream is still random. The same outcome as option A: minimal benefit from SHUFFLE.

---

## Q27 — blosc.compress shuffle Parameter Default

> **Week reference:** Week 3

**Mental Model:** The default shuffle mode in `blosc.compress` is `blosc.SHUFFLE` (value=1) when `typesize > 1`, and `blosc.NOSHUFFLE` (value=0) when `typesize=1`. Knowing defaults matters for predicting behaviour when arguments are omitted.

What shuffle mode does `blosc.compress(data, typesize=8, cname='lz4')` use if `shuffle` is not specified?

- A) `blosc.NOSHUFFLE` — the safe default is no filtering
- B) `blosc.BITSHUFFLE` — bit-level shuffling is the default for all dtypes
- C) `blosc.SHUFFLE` — byte-level shuffling is the default when `typesize > 1`
- D) No shuffling at any level, because `shuffle` is a required argument

**Answer: C**

- A) Incorrect — `NOSHUFFLE` is not the default for `typesize > 1`. Blosc defaults to SHUFFLE when the element size is more than 1 byte because it almost always improves compression for scientific data without significant overhead.
- B) Incorrect — BITSHUFFLE is not the default. It must be explicitly requested with `shuffle=blosc.BITSHUFFLE`. BITSHUFFLE has higher CPU cost and is not the safe default.
- C) Correct — when `typesize > 1`, `blosc.compress` defaults to `shuffle=blosc.SHUFFLE`. For `typesize=1`, the default is `blosc.NOSHUFFLE` (since there is nothing to shuffle at the byte level). This behaviour is consistent with the Blosc C library defaults.
- D) Incorrect — `shuffle` is an optional keyword argument with a sensible default. Omitting it does not raise an error.

---

## Q28 — Effect of clevel on Decompression Speed

> **Week reference:** Week 3

**Mental Model:** Compression level (clevel) controls how hard the codec searches during compression. Decompression always reads the exact format written by the compressor — it does not re-search. Therefore, clevel has a negligible effect on decompression speed.

How does increasing `clevel` from 1 to 9 affect decompression speed for `cname="lz4"`?

- A) Decompression is significantly slower at clevel=9 because LZ4 must reverse the more complex encoding
- B) Decompression is significantly faster at clevel=9 because the smaller compressed file requires less data to read from disk
- C) Decompression speed is nearly unchanged by clevel; only compression (write) speed is affected
- D) clevel=9 disables multi-threaded decompression to ensure correctness at high compression levels

**Answer: C**

- A) Incorrect — the LZ4 decompressor always reads the same binary frame format regardless of how hard the compressor worked. The decompressor's job is to follow back-references and literal copies, which is the same cost no matter how many passes the compressor made to find those references.
- B) Partially true reasoning but wrong framing — a smaller file does read faster from disk (fewer bytes), but that is an indirect I/O effect, not a change in decompression throughput. The question asks about decompression speed, and the CPU decompression speed itself is unaffected by clevel.
- C) Correct — clevel only affects the compression path (encoder). The decoder always performs the same operation: read the Blosc frame, decompress each block using the codec's fixed decompression algorithm. Decompression speed is determined by data size and codec, not by what clevel was used during compression.
- D) Incorrect — clevel has no interaction with thread count. Multi-threaded decompression works at any clevel. Thread count is controlled solely by `blosc.set_nthreads`.

---

## Q29 — snappy Codec Availability

> **Week reference:** Week 3

**Mental Model:** Blosc bundles `blosclz`, `lz4`, `lz4hc`, `zstd`, and `zlib` as built-in codecs. `snappy` is an optional codec that requires a separate C library and may not be available in all installations. Calling `blosc.compress` with `cname="snappy"` on an installation without it raises an error.

Which of the following codecs is NOT guaranteed to be available in a standard `blosc` Python package installation?

- A) `lz4`
- B) `zstd`
- C) `snappy`
- D) `blosclz`

**Answer: C**

- A) Incorrect — `lz4` is compiled into the Blosc C library and is always available in the Python `blosc` package.
- B) Incorrect — `zstd` is also compiled into the Blosc C library as a standard included codec.
- C) Correct — `snappy` (Google's Snappy codec) is an optional dependency. It is not bundled with the Blosc C library by default. On a standard DTU HPC environment or `pip install blosc` installation, `snappy` may not be available. Using `cname="snappy"` when it is not installed raises a `ValueError`.
- D) Incorrect — `blosclz` is the original default codec, compiled directly into the Blosc library itself. It is always available.

---

## Q30 — Blosc Thread Safety in Multiprocessing

> **Week reference:** Week 3

**Mental Model:** Blosc uses an internal C-level thread pool. When a Python `multiprocessing` worker process starts, it gets its own copy of the Blosc library with the default thread count (1). Each worker's Blosc state is independent. Setting nthreads in the parent process does not propagate to child processes.

You have a `multiprocessing.Pool` with 4 workers. You call `blosc.set_nthreads(4)` in the main process before creating the pool. How many Blosc threads does each worker process use?

- A) 4 — the pool inherits the parent's Blosc state via fork
- B) 1 — each worker starts with Blosc's default of 1 thread; the parent's setting is not reliably inherited
- C) 16 — each worker uses all available cores divided evenly
- D) 0 — workers cannot use Blosc because the thread pool conflicts with multiprocessing

**Answer: B**

- A) Incorrect — on Linux, `fork()` does copy memory state including the thread count setting, but the Blosc internal thread pool is reinitialised in the child after fork because inheriting a live thread pool across fork is unsafe (POSIX warns against it). On other platforms (spawn-based), no state is copied at all. The safe assumption is that each worker starts at Blosc's default of 1 thread.
- B) Correct — `multiprocessing` workers (especially with the `spawn` start method used on macOS and Windows by default) start fresh Python processes. Each new process loads Blosc fresh, getting the default of 1 thread. Even on Linux with `fork`, the Blosc thread pool state should not be relied upon. Workers should each call `blosc.set_nthreads` explicitly if multi-threaded Blosc is needed.
- C) Incorrect — Blosc does not auto-detect or distribute cores across workers. Thread count is only set via `blosc.set_nthreads`. There is no mechanism that divides cores among processes.
- D) Incorrect — Blosc can be used inside multiprocessing workers. The concern is only about thread count initialisation, not a fundamental incompatibility.

---

## Q31 — Reading Back with Wrong Function

> **Week reference:** Week 3

**Mental Model:** `blosc.pack_array` and `blosc.compress` produce different frame layouts. `pack_array` embeds a pickle payload; `compress` stores raw bytes. The inverse must always match: `unpack_array` for `pack_array`, and `decompress` for `compress`. Mixing them does not crash Blosc but returns wrong data types.

Data was compressed with `blosc.compress(arr.tobytes(), ...)` and saved to disk. You read the file and call `blosc.unpack_array(raw_bytes)`. What happens?

- A) You get back the original NumPy array correctly, because `unpack_array` can handle both frame types
- B) You get a `ValueError` because the Blosc frame magic number differs between `compress` and `pack_array`
- C) You get an error or garbled result because `unpack_array` expects a pickle header that `compress` never wrote
- D) You get back a `bytes` object instead of a NumPy array, which is the same as calling `decompress`

**Answer: C**

- A) Incorrect — `unpack_array` is not polymorphic. It specifically calls `blosc.decompress` and then `pickle.loads` on the result. If the data was written with `blosc.compress` (no pickle header), `pickle.loads` will fail on the raw decompressed bytes.
- B) Incorrect — both `compress` and `pack_array` write valid Blosc frames with the same magic number. The Blosc frame format is the same at the outer level; the difference is in the decompressed payload (raw bytes vs pickled array).
- C) Correct — `unpack_array` decompress the Blosc frame to get raw bytes, then calls `pickle.loads` on those bytes to reconstruct the NumPy array. If the original data was written with `blosc.compress`, the decompressed bytes are the raw array data (not a pickle stream). `pickle.loads` will raise an `UnpicklingError` or similar exception.
- D) Incorrect — `unpack_array` does not return a `bytes` object. It specifically calls `pickle.loads` on the decompressed data, which will either return an ndarray (if the frame was from `pack_array`) or raise an exception (if from `compress`).

---

## Q32 — Compressibility and Data Entropy Relationship

> **Week reference:** Week 3

**Mental Model:** Lossless compression ratio is bounded by data entropy. High-entropy data (random, encrypted, already-compressed) is incompressible. Low-entropy data (zeros, repeated patterns, slowly varying values) compresses well. This is a fundamental information-theoretic limit, not a codec limitation.

Which of the following is the most accurate statement about the theoretical limit of lossless compression?

- A) Any data can be compressed by at least 50% with a sufficiently advanced codec
- B) The maximum compression ratio achievable is determined by the data's entropy — high-entropy data cannot be significantly compressed regardless of codec
- C) ZSTD at clevel=9 can always achieve at least 2:1 compression because it uses entropy coding
- D) Blosc can always compress NumPy arrays to at least 10% of their original size because it knows the dtype

**Answer: B**

- A) Incorrect — Shannon's source coding theorem proves that no lossless code can compress data below its entropy rate. High-entropy data (e.g., uniformly random bytes at ~8 bits/byte) cannot be losslessly compressed at all, let alone by 50%.
- B) Correct — information theory establishes that the minimum lossless representation of data is its entropy. A uniformly random byte sequence has ~8 bits/byte of entropy — already optimal, nothing to compress. Only data with redundancy (repeated patterns, biased distributions, correlations) can be compressed. This applies universally to all codecs including Blosc's.
- C) Incorrect — entropy coding (Huffman, ANS) only helps when the input symbol distribution is non-uniform. For uniformly random data, entropy coding produces output the same size as or larger than the input. ZSTD's entropy coder cannot create compression from true randomness.
- D) Incorrect — knowing the dtype does not help compress high-entropy data. An array of random float64 values has high entropy regardless of dtype annotation. Blosc's knowledge of the dtype only helps the SHUFFLE filter rearrange bytes, but if those bytes are random, shuffling random bytes produces random bytes.

---
