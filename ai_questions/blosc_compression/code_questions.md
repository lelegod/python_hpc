# Blosc Compression — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Lossless Round-Trip (Zeros Array)](#q1--lossless-round-trip-zeros-array)
- [Q2 — Compression Ratio for Random Data](#q2--compression-ratio-for-random-data)
- [Q3 — SHUFFLE vs NOSHUFFLE for All-Zeros](#q3--shuffle-vs-noshuffle-for-all-zeros)
- [Q4 — pack_array Preserves dtype and Shape](#q4--pack_array-preserves-dtype-and-shape)
- [Q5 — Codec Comparison for Tiled Data](#q5--codec-comparison-for-tiled-data)
- [Q6 — Timing compress vs decompress for Random float64](#q6--timing-compress-vs-decompress-for-random-float64)
- [Q7 — Thread Count Does Not Affect Compressed Content](#q7--thread-count-does-not-affect-compressed-content)
- [Q8 — SHUFFLE vs BITSHUFFLE for Linearly Spaced Floats](#q8--shuffle-vs-bitshuffle-for-linearly-spaced-floats)
- [Q9 — zstd Produces Smaller File Than lz4 for Zeros](#q9--zstd-produces-smaller-file-than-lz4-for-zeros)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2--generated-practice-questions-exam-day-focus)
- [Q10 — pack_array/unpack_array Round-Trip Correctness](#q10--pack_arrayunpack_array-round-trip-correctness)
- [Q11 — bytes Input to compress()](#q11--bytes-input-to-compress)
- [Q12 — Parallel Compression Speedup](#q12--parallel-compression-speedup)
- [Q13 — float64 vs float32 Compressed Size](#q13--float64-vs-float32-compressed-size)
- [Q14 — Random Data: Blosc vs NumPy File Size](#q14--random-data-blosc-vs-numpy-file-size)
- [Q15 — decompress vs unpack_array Return Type](#q15--decompress-vs-unpack_array-return-type)
- [Q16 — clevel Comparison: lz4 vs zstd](#q16--clevel-comparison-lz4-vs-zstd)
- [Q17 — Thread Count at Compress vs Decompress](#q17--thread-count-at-compress-vs-decompress)
- [Q18 — Zeros vs Random Compression Ratio Thresholds](#q18--zeros-vs-random-compression-ratio-thresholds)

---

> Format: Each question shows Blosc Python code with a specific behaviour to predict.
> Exam frequency: **Week 3 topic**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#question-1)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — Lossless Round-Trip (Zeros Array)

```python
import blosc
import numpy as np

arr = np.zeros((512, 512, 512), dtype='uint8')
compressed = blosc.pack_array(arr, cname='lz4')
recovered = blosc.unpack_array(compressed)
print(np.array_equal(arr, recovered))
```

What does this code print?

**A)** `False` — the compression is lossy and some zeros become non-zero after round-trip

**B)** `True` — `pack_array`/`unpack_array` is a lossless round-trip that exactly reconstructs the original array

**C)** A `ValueError` because a 512^3 uint8 array is too large for `pack_array`

**D)** `True`, but only if the array contains no NaN or Inf values

**Answer: B**

- A) Incorrect — Blosc is a **lossless** compressor. Every bit of the original array is preserved exactly. The output of `unpack_array` is identical to the input of `pack_array`.
- B) Correct — `pack_array` compresses and serialises the array including its metadata (shape, dtype, strides). `unpack_array` reverses this exactly. `np.array_equal` returns `True` because every element is identical.
- C) Incorrect — there is no upper size limit in `pack_array` beyond available RAM. A 512^3 uint8 array is 128 MiB, which is easily handled.
- D) Incorrect — losslessness is unconditional for Blosc. The NaN/Inf caveat applies to lossy compressors (e.g., `float16` downcast) but not to Blosc.

---

## Q2 — Compression Ratio for Random Data

```python
import blosc
import numpy as np

arr = np.random.randint(1, 256, size=(1024, 1024, 1024), dtype='uint8')
compressed = blosc.pack_array(arr, cname='lz4')
ratio = len(arr.tobytes()) / len(compressed)
print(f"Compression ratio: {ratio:.2f}")
```

Which output is most likely for a large uniform-random uint8 array?

**A)** `Compression ratio: 128.00` — LZ4 achieves 128:1 for byte arrays

**B)** `Compression ratio: ~1.00` — random data has maximum entropy and LZ4 finds no patterns to compress

**C)** `Compression ratio: 0.98` — the Blosc frame header makes compressed output slightly larger than raw data

**D)** `Compression ratio: 10.00` — LZ4 always achieves at least 10:1 for uint8 arrays

**Answer: B**

- A) Incorrect — 128:1 is characteristic of maximally compressible data (e.g., all-zeros). Random data yields essentially 1:1.
- B) Correct — uniform random integers have maximum entropy (~8 bits per byte). LZ4 finds no repeated sequences to back-reference, so the compressed size is approximately equal to the uncompressed size, giving a ratio near 1.00. `len(compressed)` includes the Blosc frame header (a few hundred bytes), so the ratio may be slightly below 1.
- C) Partially correct reasoning but the answer is misleadingly presented — the ratio is indeed near 1.00 (or fractionally below 1 due to headers), but Blosc actually stores the uncompressed block when compression would expand the data. The ratio would be ≈1.00, not specifically 0.98.
- D) Incorrect — LZ4 guarantees no fixed minimum ratio for arbitrary data. For random data, it achieves close to 1:1.

---

## Q3 — SHUFFLE vs NOSHUFFLE for All-Zeros

```python
import blosc
import numpy as np

blosc.set_nthreads(4)
arr = np.zeros((256, 256, 256), dtype='float64')
c1 = blosc.compress(arr.tobytes(), shuffle=blosc.SHUFFLE, typesize=8, cname='lz4')
c2 = blosc.compress(arr.tobytes(), shuffle=blosc.NOSHUFFLE, typesize=8, cname='lz4')
print(len(c1) < len(c2))
```

What does this code print for an all-zeros float64 array?

**A)** `False` — SHUFFLE always increases compressed size by adding byte-reordering overhead

**B)** `True` or `False` — outcome depends on `nthreads`, which was set to 4

**C)** `True` — SHUFFLE groups the zero bytes together; but for all-zeros data, both produce near-identical tiny outputs, so the result is unpredictable

**D)** Either `True` or `False`, but both `c1` and `c2` are very small because all-zeros compresses to near-zero bytes regardless of shuffle mode

**Answer: D**

- A) Incorrect — SHUFFLE does not increase compressed size; it typically helps or is neutral. For all-zeros data, every byte is 0x00 regardless of how they are reordered, so both shuffle modes produce essentially the same input to the codec.
- B) Incorrect — `nthreads` affects compression throughput, not the compressed output content. The compressed bytes are deterministic given the same data, codec, level, and shuffle mode.
- C) Partially correct intuition — for all-zeros data, the shuffle rearrangement has no effect because every byte is already 0x00. Both `c1` and `c2` will be near-identical in size. Whether `len(c1) < len(c2)` evaluates to `True` or `False` depends on implementation details of the Blosc frame header for each mode, not on meaningful compression differences.
- D) Correct — both compressed outputs are tiny (an all-zeros array compresses to a handful of bytes). The `print` statement may output `True` or `False` depending on minor header size differences, but the important result is that both are near-zero in size. The comparison is essentially meaningless for this data.

---

## Q4 — pack_array Preserves dtype and Shape

```python
import blosc
import numpy as np
import os

def write_blosc(arr, file_name, cname="lz4"):
    b_arr = blosc.pack_array(arr, cname=cname)
    with open(f"{file_name}.bl", "wb") as w:
        w.write(b_arr)
    if hasattr(os, 'sync'):
        os.sync()

def read_blosc(file_name):
    with open(f"{file_name}.bl", "rb") as r:
        b_arr = r.read()
    return blosc.unpack_array(b_arr)

arr = np.zeros((512, 512, 512), dtype='uint8')
write_blosc(arr, 'test', cname='zstd')
result = read_blosc('test')
print(result.dtype, result.shape)
```

What does this code print?

**A)** `uint8 (512, 512, 512)` — `pack_array`/`unpack_array` preserves dtype and shape

**B)** `float64 (512, 512, 512)` — `unpack_array` always returns float64

**C)** An error, because `write_blosc` uses `cname='zstd'` but `read_blosc` does not specify a codec

**D)** `uint8 (134217728,)` — the array is flattened during Blosc serialisation

**Answer: A**

- A) Correct — `blosc.pack_array` serialises the array along with its metadata (dtype, shape, strides) using pickle. `blosc.unpack_array` deserialises this metadata and reconstructs the array with the original dtype and shape. Output is `uint8 (512, 512, 512)`.
- B) Incorrect — `unpack_array` uses the pickled metadata to reconstruct the original dtype. It does not default to float64.
- C) Incorrect — the codec is encoded in the Blosc frame header. `unpack_array` reads the header to determine which codec was used for compression; the caller does not need to specify it.
- D) Incorrect — `pack_array` uses pickle for shape/stride metadata, which includes the original 3D shape. The reconstructed array has the same shape as the original.

---

## Q5 — Codec Comparison for Tiled Data

```python
import blosc
import numpy as np

arr = np.arange(256, dtype='uint8')
arr = np.tile(arr, 1024 * 1024).reshape(256, 256, 256)

c_lz4   = blosc.pack_array(arr, cname='lz4')
c_zstd  = blosc.pack_array(arr, cname='zstd')
c_bclz  = blosc.pack_array(arr, cname='blosclz')

print(len(c_zstd) < len(c_lz4))
print(len(c_lz4) < len(c_bclz) or len(c_bclz) < len(c_lz4))
```

For a highly repetitive tiled array, which output is most likely?

**A)** `False` then `False` — all codecs produce identical output for repetitive data

**B)** `True` then `True` — ZSTD beats LZ4 in ratio, and LZ4 and blosclz differ

**C)** `True` then `False` — ZSTD beats LZ4, and LZ4 and blosclz are identical

**D)** `False` then `True` — LZ4 beats ZSTD for repetitive data because LZ4 uses a longer match window

**Answer: B**

- A) Incorrect — different codecs use different matching algorithms with different match-window sizes and entropy coding. For repetitive data, ZSTD typically achieves better ratio than LZ4 or blosclz.
- B) Correct — for highly repetitive data, ZSTD's more exhaustive pattern matching and entropy coding (Huffman/ANS) yields a smaller output than LZ4. LZ4 and blosclz use similar but distinct algorithms and will generally produce different (though comparably-sized) outputs. The second `print` evaluates `True` because the two sizes are not equal.
- C) Incorrect — LZ4 and blosclz use different algorithms (FastLZ-derived vs LZ4's hash-chain approach) and will produce different compressed sizes for structured data. They are not identical.
- D) Incorrect — ZSTD uses a much longer effective match window than LZ4 and adds entropy coding on top of LZ77 matching. For repetitive data, ZSTD consistently achieves better ratios than LZ4.

---

## Q6 — Timing compress vs decompress for Random float64

```python
import blosc
import numpy as np
from time import perf_counter

blosc.set_nthreads(1)
arr = np.random.rand(500, 500, 500)  # float64

t0 = perf_counter()
c = blosc.compress(arr.tobytes(), typesize=8, cname='lz4', clevel=9)
t1 = perf_counter()

t2 = perf_counter()
d = blosc.decompress(c)
t3 = perf_counter()

print(f"compress: {t1-t0:.3f}s, decompress: {t3-t2:.3f}s")
print(f"ratio: {len(arr.tobytes())/len(c):.2f}")
```

For random float64 data with `clevel=9`, which outcome is most accurate?

**A)** Compression is ~9x slower than decompression and ratio is ~10:1

**B)** Compression and decompression are similar in speed and ratio is ~1:1, because random floats are incompressible

**C)** Compression is slower than decompression, but ratio is ~1:1 because random floats have maximum entropy

**D)** `clevel=9` causes Blosc to switch to ZSTD internally, producing a ~5:1 ratio

**Answer: C**

- A) Incorrect — random float64 data is incompressible regardless of `clevel`. A 10:1 ratio is impossible for maximum-entropy data.
- B) Incorrect — compression is always slower than decompression for LZ4. At `clevel=9`, LZ4 spends significantly more time searching for matches (which it won't find), making compression slower. Decompression is very fast. The ratio is indeed ~1:1, but the speed claim is wrong.
- C) Correct — `clevel=9` makes LZ4 search harder for patterns, which costs more CPU time for compression. Decompression speed is essentially unaffected by `clevel` since the decompressor just reads the format, not re-searches. The ratio is ~1:1 because random float64 values have maximum entropy. Compression is noticeably slower than decompression.
- D) Incorrect — `clevel` does not change the codec. The codec is fixed by `cname`. `clevel=9` with `cname='lz4'` uses LZ4 at its highest effort level, not ZSTD.

---

## Q7 — Thread Count Does Not Affect Compressed Content

```python
import blosc
import numpy as np

arr = np.zeros((1024, 1024, 1024), dtype='uint8')  # 1 GiB

blosc.set_nthreads(1)
c1 = blosc.pack_array(arr)

blosc.set_nthreads(8)
c2 = blosc.pack_array(arr)

print(len(c1) == len(c2))
```

What does this code print?

**A)** `False` — more threads process more chunks simultaneously, producing a different compressed stream

**B)** `True` — the compressed content is identical regardless of thread count; only throughput changes

**C)** `False` — the default codec (`blosclz`) is not thread-safe and produces non-deterministic output with 8 threads

**D)** This raises a `RuntimeError` because 8 threads exceeds Blosc's default limit

**Answer: B**

- A) Incorrect — thread count does not affect the content of the compressed output. Blosc partitions data into fixed-size chunks and compresses each chunk independently. The chunks and their order are the same regardless of how many threads process them.
- B) Correct — Blosc's chunk-based parallel design ensures that the compressed output is identical for any thread count ≥ 1. Each chunk is compressed independently using the same algorithm and parameters; the multi-threaded result is the same as single-threaded, just computed faster.
- C) Incorrect — Blosc's codecs are thread-safe by design. Each thread operates on its own independent chunk buffer. The output is fully deterministic.
- D) Incorrect — Blosc's maximum thread count is a compile-time constant (typically 256). Setting 8 threads is well within this limit and does not raise any error.

---

## Q8 — SHUFFLE vs BITSHUFFLE for Linearly Spaced Floats

```python
import blosc
import numpy as np

arr = np.linspace(0, 1, 1000000, dtype='float32')
c_shuffle    = blosc.compress(arr.tobytes(), typesize=4, shuffle=blosc.SHUFFLE,    cname='lz4')
c_bitshuffle = blosc.compress(arr.tobytes(), typesize=4, shuffle=blosc.BITSHUFFLE, cname='lz4')
c_noshuffle  = blosc.compress(arr.tobytes(), typesize=4, shuffle=blosc.NOSHUFFLE,  cname='lz4')

print(len(c_noshuffle) > len(c_shuffle))
print(len(c_shuffle) > len(c_bitshuffle))
```

For a linearly spaced float32 array, which output is most likely?

**A)** `True` then `True` — NOSHUFFLE > SHUFFLE > BITSHUFFLE in compressed size

**B)** `True` then `False` — NOSHUFFLE > SHUFFLE, but BITSHUFFLE does not help more than SHUFFLE for linearly spaced data

**C)** `False` then `False` — all three modes produce the same compressed size for float32

**D)** `False` then `True` — SHUFFLE compresses better than NOSHUFFLE and BITSHUFFLE for float arrays

**Answer: A**

- A) Correct — for linearly spaced float32 data, the bytes are highly structured. SHUFFLE groups bytes by significance level, creating long runs of slowly-varying bytes (especially the exponent/sign bytes) that LZ4 compresses well. BITSHUFFLE goes further by transposing at the bit level, grouping identical bits even more tightly, which can achieve even better ratio for smoothly varying values. So `NOSHUFFLE > SHUFFLE > BITSHUFFLE` in output size, making both prints `True`.
- B) Incorrect — for smoothly varying float32 data (slowly changing values), BITSHUFFLE often outperforms SHUFFLE because the slowly changing mantissa bits produce very long uniform-bit runs after bit-level transposition.
- C) Incorrect — the three modes produce different outputs for structured floating-point data. NOSHUFFLE typically compresses the worst for multi-byte floating-point types.
- D) Incorrect — while SHUFFLE does compress better than NOSHUFFLE (`False` would be wrong for the first print), BITSHUFFLE typically also outperforms SHUFFLE for structured float data.

---

## Q9 — zstd Produces Smaller File Than lz4 for Zeros

```python
import blosc
import numpy as np
import os

def write_blosc(arr, file_name, cname="lz4"):
    b_arr = blosc.pack_array(arr, cname=cname)
    with open(f"{file_name}.bl", "wb") as w:
        w.write(b_arr)
    if hasattr(os, 'sync'):
        os.sync()

n = 512
A = np.zeros((n, n, n), dtype='uint8')
write_blosc(A, 'data', cname='lz4')
write_blosc(A, 'data', cname='zstd')

import os.path
size_lz4  = os.path.getsize('data.bl')  # after first write
# (Note: second write overwrites the file)
```

After the second call (`cname='zstd'`), what is `data.bl`'s expected size relative to the `lz4` version?

**A)** Larger — ZSTD is slower and writes more metadata than LZ4

**B)** Smaller or equal — ZSTD achieves a higher compression ratio than LZ4 for structured data like zeros

**C)** Identical — the underlying data is all-zeros, which both codecs compress to the same size

**D)** The file does not exist because `write_blosc` with `cname='zstd'` creates a `.zst` file instead

**Answer: B**

- A) Incorrect — more metadata is not added just because a codec is slower. ZSTD's compressed output is smaller, not larger, for compressible data.
- B) Correct — ZSTD achieves a better compression ratio than LZ4 for all-zeros data. The resulting file written by the second call (`cname='zstd'`) is smaller than what the first call (`cname='lz4'`) produced, because ZSTD's entropy coding more efficiently represents the highly compressible zeros.
- C) Incorrect — LZ4 and ZSTD use different algorithms. For all-zeros data, both produce very small output, but ZSTD typically produces slightly fewer bytes due to its more efficient entropy coder.
- D) Incorrect — the output file name is determined by the caller (`file_name + ".bl"`). The codec name (`cname`) affects only the compression algorithm inside the Blosc frame; it does not change the file extension.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

---

## Q10 — pack_array/unpack_array Round-Trip Correctness

```python
import blosc
import numpy as np

arr = np.zeros((256, 256, 256), dtype='uint8')
blosc.set_nthreads(4)

compressed = blosc.pack_array(arr, cname='lz4', clevel=5)
arr2 = blosc.unpack_array(compressed)

print(arr.shape == arr2.shape)
print(arr.dtype == arr2.dtype)
print(np.all(arr == arr2))
```

What are the three printed values?

**A)** `True`, `True`, `True`

**B)** `True`, `False`, `True` — dtype changes during compression

**C)** `False`, `True`, `True` — shape is lost but values and dtype are preserved

**D)** `True`, `True`, `False` — some zeros may flip to 1 due to Blosc's lossy mode at `clevel=5`

**Answer: A**

- A) Correct — `pack_array` preserves shape, dtype, and exact values through a lossless round-trip. All three assertions are `True`. Thread count and compression level do not affect correctness.
- B) Incorrect — dtype is stored in the Blosc frame's pickled metadata and is preserved exactly. No dtype conversion occurs.
- C) Incorrect — shape is stored in the pickled metadata alongside dtype. The reconstructed array has the same shape as the original.
- D) Incorrect — Blosc is always lossless. There is no lossy mode. `clevel=5` controls how hard the codec searches for patterns; it never introduces data loss.

---

## Q11 — bytes Input to compress()

```python
import blosc

data = b'\x00' * (10 * 1024 * 1024)  # 10 MiB of zeros

c = blosc.compress(data, typesize=1, cname='lz4', clevel=1)
d = blosc.decompress(c)

print(len(d) == len(data))
print(d == data)
```

What does this code print?

**A)** `True` then `True` — lossless round-trip is guaranteed regardless of `clevel`

**B)** `True` then `False` — the byte values may change but the length is preserved

**C)** `False` then `False` — `clevel=1` skips the decompression metadata, causing a size mismatch

**D)** A `TypeError` — `blosc.compress` requires a NumPy array, not a Python bytes object

**Answer: A**

- A) Correct — `blosc.compress` accepts any bytes-like object. `clevel=1` simply means the codec uses minimum effort to find patterns; it does not affect correctness. The decompressed bytes are bit-for-bit identical to the input, and the length is preserved. Both prints are `True`.
- B) Incorrect — Blosc compression is always lossless. The byte values are identical after decompression.
- C) Incorrect — `clevel` does not affect metadata completeness. The Blosc frame always contains enough metadata for full reconstruction regardless of level.
- D) Incorrect — `blosc.compress` operates on any bytes-like object (`bytes`, `bytearray`, `memoryview`). It does not require a NumPy array; that is what `blosc.pack_array` is for.

---

## Q12 — Parallel Compression Speedup

```python
import blosc
import numpy as np
from time import perf_counter

n = 1024
arr = np.zeros((n, n, n), dtype='uint8')  # 1 GiB

blosc.set_nthreads(1)
t0 = perf_counter()
c1 = blosc.pack_array(arr, cname='lz4')
t1 = perf_counter()

blosc.set_nthreads(8)
t2 = perf_counter()
c2 = blosc.pack_array(arr, cname='lz4')
t3 = perf_counter()

print((t3 - t2) < (t1 - t0))
```

What does this code most likely print on a machine with 8 physical cores?

**A)** `False` — more threads introduce synchronisation overhead that dominates for 1 GiB arrays

**B)** `True` — 8 threads compress the 1 GiB array faster than 1 thread because Blosc parallelises across chunks

**C)** `True` — but only because the second call benefits from OS page cache warming of the array

**D)** `False` — Blosc only parallelises decompression; compression is always single-threaded

**Answer: B**

- A) Incorrect — for 1 GiB of data, chunk-level parallelism in Blosc provides significant speedup. The data is split into many independent chunks (each typically 128 KB–1 MB), and 8 threads can process 8 chunks simultaneously. Synchronisation overhead is negligible relative to 1 GiB of compression work.
- B) Correct — Blosc uses a thread pool to compress independent chunks in parallel. For a 1 GiB array with default chunk sizes, 8 threads should compress approximately 4–7x faster than 1 thread (accounting for memory bandwidth saturation and overhead).
- C) Incorrect — the page cache warming is a minor effect and not the primary reason. The answer is `True` because of genuine thread-level parallelism.
- D) Incorrect — Blosc parallelises both compression and decompression. The thread pool is used for both `compress`/`pack_array` and `decompress`/`unpack_array`.

---

## Q13 — float64 vs float32 Compressed Size

```python
import blosc
import numpy as np

arr = np.random.rand(1000, 1000)  # float64, values in [0, 1)

# Approach A
c_a = blosc.compress(arr.tobytes(), typesize=8, shuffle=blosc.SHUFFLE, cname='lz4')

# Approach B
arr_f32 = arr.astype('float32')
c_b = blosc.compress(arr_f32.tobytes(), typesize=4, shuffle=blosc.SHUFFLE, cname='lz4')

print(len(c_a) > len(c_b))
```

What does this code most likely print?

**A)** `False` — float32 is less compressible than float64 because it packs more values per byte

**B)** `True` — the float64 array is 8 MiB while float32 is 4 MiB; the larger raw size means more compressed bytes even at the same ratio

**C)** `False` — both compressed outputs have the same size because SHUFFLE equalises dtype differences

**D)** `True` only if the values happen to be exactly representable in both float32 and float64

**Answer: B**

- A) Incorrect — float32 uses 4 bytes per element vs float64's 8 bytes. The raw data is half the size, and since random floats compress at approximately the same ratio regardless of precision, the float32 compressed output is smaller. So `len(c_a) > len(c_b)` is `True`, not `False`.
- B) Correct — the float64 array occupies 8 MiB; float32 occupies 4 MiB. For random data, both compress poorly (~1:1 ratio). The float64 compressed output is roughly twice as large as float32. `len(c_a) > len(c_b)` prints `True`.
- C) Incorrect — SHUFFLE rearranges bytes within each element but does not change the total byte count. The compressed sizes differ by roughly a factor of 2.
- D) Incorrect — the comparison depends on the total bytes, not representability. Even for values that are exactly representable in float32, the compressed sizes differ because of the 2x difference in raw data volume.

---

## Q14 — Random Data: Blosc vs NumPy File Size

```python
import blosc
import numpy as np
import os

def write_numpy(arr, file_name):
    np.save(f"{file_name}.npy", arr)
    if hasattr(os, 'sync'):
        os.sync()

def write_blosc(arr, file_name, cname="lz4"):
    b_arr = blosc.pack_array(arr, cname=cname)
    with open(f"{file_name}.bl", "wb") as w:
        w.write(b_arr)
    if hasattr(os, 'sync'):
        os.sync()

n = 512
arr = np.random.randint(1, 256, size=(n, n, n), dtype='uint8')
write_numpy(arr, 'rand')
write_blosc(arr, 'rand')

import os.path
npy_size   = os.path.getsize('rand.npy')
blosc_size = os.path.getsize('rand.bl')
print(blosc_size < npy_size)
```

What does this code most likely print?

**A)** `True` — Blosc always produces smaller files than NumPy save

**B)** `False` — random data is incompressible; the Blosc file may be similar in size or slightly larger due to frame overhead

**C)** `True` — SHUFFLE filter rearranges random bytes into ascending order, making them compressible

**D)** `False` — `np.save` compresses data internally, so `.npy` is already compressed

**Answer: B**

- A) Incorrect — Blosc only produces smaller files when the data is compressible. For random uint8 data with maximum entropy, Blosc achieves ~1:1 ratio, and the Blosc frame header adds a small overhead.
- B) Correct — random uniform integers have maximum entropy. LZ4 finds no patterns to compress. The `.bl` file is approximately the same size as the `.npy` file (both roughly n^3 bytes), plus the Blosc frame header overhead. `blosc_size < npy_size` may evaluate to `False`.
- C) Incorrect — SHUFFLE reorders bytes by significance position within multi-byte elements; it does not sort element values. For uint8 (1-byte elements), SHUFFLE is a no-op. Even for larger dtypes, it does not sort values.
- D) Incorrect — `np.save` writes an uncompressed binary format (the `.npy` format includes a header with dtype/shape info but no compression). The file size is approximately `n^3` bytes plus a small header.

---

## Q15 — decompress vs unpack_array Return Type

```python
import blosc
import numpy as np

# What is wrong with this code?
arr = np.arange(1000, dtype='int32')
compressed_bytes = blosc.pack_array(arr)

with open('data.bl', 'wb') as f:
    f.write(compressed_bytes)

with open('data.bl', 'rb') as f:
    raw = f.read()

recovered = blosc.decompress(raw)
print(type(recovered))
```

What does `type(recovered)` print, and is this the intended way to recover the array?

**A)** `<class 'numpy.ndarray'>` — `decompress` is equivalent to `unpack_array` for NumPy arrays

**B)** `<class 'bytes'>` — `decompress` returns raw bytes, not a NumPy array; `unpack_array` should be used instead

**C)** A `TypeError` is raised because `decompress` cannot process output from `pack_array`

**D)** `<class 'numpy.ndarray'>` with dtype `uint8` — `decompress` always returns uint8 arrays

**Answer: B**

- A) Incorrect — `blosc.decompress` is the inverse of `blosc.compress`; it returns raw `bytes`. `blosc.unpack_array` is the inverse of `blosc.pack_array`; it returns a NumPy array. Using `decompress` on `pack_array` output gives bytes containing a pickled NumPy array, not the array itself.
- B) Correct — `blosc.pack_array` serialises the array as pickle + Blosc-compressed bytes. `blosc.decompress` reverses only the Blosc compression layer, returning the raw pickled bytes. The returned object is `bytes`, not a NumPy array. To recover the array, `blosc.unpack_array(raw)` must be called instead.
- C) Incorrect — `decompress` does not raise a `TypeError` here. The Blosc frame format from `pack_array` is valid input to `decompress`; it just returns the pickle bytes rather than the reconstructed array.
- D) Incorrect — `decompress` returns a `bytes` object, not a NumPy array of any dtype.

---

## Q16 — clevel Comparison: lz4 vs zstd

```python
import blosc
import numpy as np

arr = np.tile(np.arange(256, dtype='uint8'), 256 * 256).reshape(256, 256, 256)

c_lz4_1  = blosc.pack_array(arr, cname='lz4',  clevel=1)
c_lz4_9  = blosc.pack_array(arr, cname='lz4',  clevel=9)
c_zstd_1 = blosc.pack_array(arr, cname='zstd', clevel=1)
c_zstd_9 = blosc.pack_array(arr, cname='zstd', clevel=9)

print(len(c_zstd_9) < len(c_lz4_9))
print(len(c_lz4_9)  < len(c_lz4_1))
```

For a tiled (repetitive) uint8 array, which output is most likely?

**A)** `True` then `True` — ZSTD at level 9 beats LZ4 at level 9; LZ4 at level 9 beats LZ4 at level 1

**B)** `True` then `False` — ZSTD beats LZ4, but LZ4 compression ratio does not improve with higher levels

**C)** `False` then `True` — LZ4 beats ZSTD because tiled data favours back-reference length; and level 9 beats level 1 for LZ4

**D)** `False` then `False` — tiled data compresses to 0 bytes in all cases

**Answer: A**

- A) Correct — ZSTD's superior entropy coding and longer match window yield a better ratio than LZ4 for tiled data (`True`). Within LZ4, higher compression levels search more exhaustively for back-references, so `clevel=9` achieves a better ratio than `clevel=1` for structured data (`True`).
- B) Incorrect — for structured (tiled) data, LZ4 does show meaningful ratio improvement as `clevel` increases, because higher levels find longer and more distant matches.
- C) Incorrect — ZSTD consistently outperforms LZ4 in compression ratio for all data types tested in the course. The first print is `True`, not `False`.
- D) Incorrect — no codec can compress data to literally 0 bytes (the Blosc frame header alone occupies multiple bytes). The compressed sizes are small but non-zero.

---

## Q17 — Thread Count at Compress vs Decompress

```python
import blosc
import numpy as np

blosc.set_nthreads(4)

arr = np.zeros((512, 512, 512), dtype='uint8')
compressed = blosc.pack_array(arr, cname='lz4')

# Now change thread count and decompress
blosc.set_nthreads(1)
recovered = blosc.unpack_array(compressed)

print(np.array_equal(arr, recovered))
```

What does this code print?

**A)** `False` — decompressing with 1 thread produces different results than compressing with 4 threads

**B)** A `RuntimeError` — the Blosc frame was created with 4 threads and cannot be decoded with 1

**C)** `True` — the number of threads used for compression vs decompression does not affect correctness

**D)** `True` — but only for all-zeros arrays; for other data it would be `False`

**Answer: C**

- A) Incorrect — thread count is not stored in the Blosc frame. The compressed data is a valid Blosc stream that any number of threads can decompress. The result is always identical.
- B) Incorrect — there is no "threads used during compression" metadata in a Blosc frame. Any thread count can decompress any Blosc frame.
- C) Correct — Blosc's correctness is completely independent of thread count. Threads only affect throughput. Compressing with 4 threads and decompressing with 1 thread produces the exact same result as any other combination.
- D) Incorrect — thread-count independence applies to all data, not just all-zeros arrays. The lossless guarantee is unconditional.

---

## Q18 — Zeros vs Random Compression Ratio Thresholds

```python
import blosc
import numpy as np
import os

n = 1024

# Zeros: maximum compressibility
A_zeros  = np.zeros((n, n, n), dtype='uint8')
# Random: minimum compressibility
A_random = np.random.randint(0, 256, (n, n, n), dtype='uint8')

def blosc_file_size(arr, fname, cname='lz4'):
    b = blosc.pack_array(arr, cname=cname)
    with open(fname, 'wb') as f:
        f.write(b)
    return os.path.getsize(fname)

sz_zeros  = blosc_file_size(A_zeros,  'z.bl')
sz_random = blosc_file_size(A_random, 'r.bl')

raw_size = n ** 3  # bytes for uint8

print(sz_zeros  < raw_size * 0.001)   # < 0.1% of raw
print(sz_random > raw_size * 0.95)    # > 95% of raw
```

Which output is most likely?

**A)** `True` then `True` — zeros compress to near-zero; random data compresses to ~1:1

**B)** `False` then `False` — both arrays compress to approximately 50% of raw size with LZ4

**C)** `True` then `False` — zeros compress well; random data also compresses well because LZ4 is efficient

**D)** `False` then `True` — zeros do not compress below 0.1% due to Blosc frame overhead; random data exceeds 95%

**Answer: A**

- A) Correct — a 1 GiB all-zeros array compresses to a handful of bytes with LZ4, easily below 0.1% of raw size (the first threshold is ~1 MiB; actual output is ~kilobytes). Random uint8 data achieves ~1:1 compression, so the compressed file is roughly the full 1 GiB, well above 95% of raw. Both prints are `True`.
- B) Incorrect — zeros compress far better than 50%. LZ4 represents the entire array as "N repetitions of 0x00" in essentially a constant number of bytes.
- C) Incorrect — random data is incompressible for any lossless codec. LZ4 cannot achieve meaningful compression for maximum-entropy data regardless of its efficiency.
- D) Incorrect — a 1 GiB all-zeros array easily compresses below 0.1% of raw size. The Blosc frame overhead is tiny (a few hundred bytes vs ~1 GiB raw), so the first threshold is easily met. The second part is correct (random > 95%), making this answer incorrect overall.

---
