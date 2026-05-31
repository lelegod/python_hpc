# Week 3 Exercises — Cache Effects + Blosc Compression

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Section 1: Cache Effects](#section-1-cache-effects)
- [Exercise 1.1 `[PRACTICE]`](#exercise-11-practice)
- [Exercise 1.2 `[AUTOLAB]`](#exercise-12-autolab)
- [Exercise 1.3 `[PRACTICE]`](#exercise-13-practice)
- [Exercise 1.4 `[PRACTICE]`](#exercise-14-practice)
  - [Exercise 1.4a `[PRACTICE]`](#exercise-14a-practice)
- [Exercise 1.5 `[PRACTICE]`](#exercise-15-practice)
- [Exercise 1.6 `[AUTOLAB]`](#exercise-16-autolab)
  - [Exercise 1.6a `[PRACTICE]` (Optional)](#exercise-16a-practice-optional)
- [Section 2: Efficient Data Storage with Blosc](#section-2-efficient-data-storage-with-blosc)
- [Exercise 2.1 `[AUTOLAB]`](#exercise-21-autolab)
- [Exercise 2.2 `[AUTOLAB]`](#exercise-22-autolab)
- [Exercise 2.3 `[PRACTICE]`](#exercise-23-practice)
- [Exercise 2.4 `[PRACTICE]`](#exercise-24-practice)
- [Exercise 2.5 `[AUTOLAB]`](#exercise-25-autolab)
  - [Exercise 2.5a `[PRACTICE]`](#exercise-25a-practice)

---

---

## Section 1: Cache Effects

The exercise expands on section 6.1.1 in *Fast Python*. The starting code is:

```python
import numpy as np

SIZE = 100

mat = np.random.rand(SIZE, SIZE)
double_column = 2 * mat[:, 0]
double_row = 2 * mat[0, :]
```

---

## Exercise 1.1 `[PRACTICE]`

Make a Python program that measures the execution time of `2 * mat[:, 0]` and `2 * mat[0, :]`. Measure the time for at least 1000 repetitions. Hint: remember what you did in Week 2, Exercise 2.5.

> **Solution:**
>
> Use `perf_counter` to bracket both operations separately, then divide by the number of repetitions to get average time per call.
>
> ```python
> from time import perf_counter as time
> import numpy as np
>
> SIZE = 100
> n_repeat = int(1e3)
> mat = np.random.rand(SIZE, SIZE)
>
> trow = time()
> for _ in range(n_repeat):
>     mat[0, :] * 1.01
> trow = time() - trow
>
> tcol = time()
> for _ in range(n_repeat):
>     mat[:, 0] * 1.01
> tcol = time() - tcol
>
> print('trow =', trow / n_repeat)
> print('tcol =', tcol / n_repeat)
> ```

---

## Exercise 1.2 `[AUTOLAB]`

Make a job script that runs your program. Submit it to the `hpc` queue, use a node with an Intel Xeon Gold 6126, Intel Xeon Gold 6142 or Intel Xeon Gold 6226R processor, and request a single core. Hint: remember the exercises from week 1.

> **Solution:**
>
> The student job script (`cache.sh`) submits the cache measurement program to the HPC cluster, requesting 1 core and 16 GB of memory on the `hpc` queue.
>
> ```bash
> #!/bin/bash
> #BSUB -J python
> #BSUB -q hpc
> #BSUB -W 10
> #BSUB -R "rusage[mem=16384MB]"
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -o python%J.out
> #BSUB -e python_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> python cache.py
> ```

---

## Exercise 1.3 `[PRACTICE]`

Modify your Python program and/or your jobscript to measure the time for `SIZE` ranging from 10^1 to 10^4.5. Do the measurement from a batch job — this is important, so you have control over the hardware! Hint: Use `np.logspace` to get logarithmically spaced numbers for `SIZE`.

> **Solution:**
>
> Use `np.logspace(1, 4.5, 30)` to sweep across matrix sizes and collect timings for both row and column access patterns.
>
> ```python
> from time import perf_counter as time
> import numpy as np
>
> ns = np.round(np.logspace(1, 4.5, 30))
> trows = []
> tcols = []
> n_repeat = int(1e3)
>
> for n in ns:
>     n = int(n)
>     mat = np.random.rand(n, n)
>
>     trow = time()
>     for _ in range(n_repeat):
>         mat[0, :] * 1.01
>     trow = time() - trow
>
>     tcol = time()
>     for _ in range(n_repeat):
>         mat[:, 0] * 1.01
>     tcol = time() - tcol
>
>     trows.append(trow / n_repeat)
>     tcols.append(tcol / n_repeat)
>
> print(mat.dtype)
> print('ns =', list(ns))
> print('trows =', trows)
> print('tcols =', tcols)
> ```
>
> The student implementation in `cache.py` computes MFLOP/s directly and overlays L1/L2/L3 cache boundary lines on the plot.

---

## Exercise 1.4 `[PRACTICE]`

Make a loglog plot of the performance of the row and column doubling as MFLOP/s over the size of the matrix in kilobytes. Do they perform the same? How does their performance align with the sizes of the CPU caches? Hint: you can read the cache sizes from the `lscpu` output.

> **Solution:**
>
> On an Intel Xeon Gold 6126 (representative cluster node), the cache hierarchy from `lscpu` is:
> - L1d cache: 32 KB
> - L2 cache: 1024 KB (1 MB)
> - L3 cache: 19712 KB (~19.25 MB)
>
> When the matrix is small enough to fit entirely in L1, both row and column scaling perform similarly. As the matrix grows beyond L1 (but fits in L2), performance diverges: **column scaling degrades** because accessing a column reads almost every cache line of the array (strided access with stride = `SIZE * 8` bytes), causing frequent L1 evictions. Row scaling remains fast because the entire row still fits in L1.
>
> As the matrix grows into L2 and L3 territory, column scaling continues to suffer from increasingly expensive cache misses, widening the performance gap further.

### Exercise 1.4a `[PRACTICE]`

To make the performance difference clearer, plot the ratio of MFLOP/s.

> **Solution:**
>
> Plot `MFLOP/s_row / MFLOP/s_col` on a loglog scale. The ratio grows with matrix size, showing that row access becomes disproportionately faster than column access as the matrix exceeds cache sizes. The ratio is close to 1 for small (L1-resident) matrices and increases significantly once the matrix no longer fits in L1.

---

## Exercise 1.5 `[PRACTICE]`

Let us focus on the row scaling. Modify the code so `mat` is a row vector, i.e., `mat = np.random.rand(1, SIZE)`. Measure the time for `SIZE` ranging from 10^2 to 10^8 using at least 100 repetitions. Again, do the measurement from a batch job. Hint: Use `np.logspace` to generate values for `SIZE`.

> **Solution:**
>
> ```python
> from time import perf_counter as time
> import numpy as np
>
> ns = np.ceil(np.logspace(2, 8, 30))
> trows = []
> n_repeat = int(1e2)
>
> for n in ns:
>     n = int(n)
>     mat = np.random.rand(1, n)
>
>     trow = time()
>     for _ in range(n_repeat):
>         mat[0, :] * 2
>     trow = time() - trow
>     trows.append(trow / n_repeat)
>
> print('ns =', list(ns))
> print('trows =', trows)
> ```

---

## Exercise 1.6 `[AUTOLAB]`

Make a loglog plot of the performance of the row doubling as MFLOP/s over the size of the row vector in kilobytes. What do you see? Do the performance changes align with the cache sizes?

> **Solution:**
>
> The plot shows performance decreasing in a **stair-step fashion** at cache boundaries (L1 → L2 → L3 → RAM). When the row vector fits in L1, performance is high (though the initial regime may show a rising trend due to Python overhead dominating at very small sizes). Each time the vector grows beyond a cache level, performance drops noticeably, then stabilizes until the next cache boundary.
>
> The stair-step pattern confirms that cache effects are the primary driver of performance — not algorithmic complexity.

### Exercise 1.6a `[PRACTICE]` (Optional)

Redo the experiment using `mat = mat.astype('float32')`. Do you observe the same pattern?

> **Solution:**
>
> Yes, the same stair-step pattern is observed with `float32`. Since `float32` uses half the memory of `float64`, the cache boundaries occur at twice the array length (in elements), but the performance profile shape is essentially the same.

---

## Section 2: Efficient Data Storage with Blosc

The exercise analyzes when reading compressed data + decompressing is faster than reading raw data. Based on section 6.2.1 in *Fast Python*. Supporting functions (provided):

```python
import os
import blosc
import numpy as np


def write_numpy(arr, file_name):
    np.save(f"{file_name}.npy", arr)
    os.sync()


def write_blosc(arr, file_name, cname="lz4"):
    b_arr = blosc.pack_array(arr, cname=cname)
    with open(f"{file_name}.bl", "wb") as w:
        w.write(b_arr)
    os.sync()


def read_numpy(file_name):
    return np.load(f"{file_name}.npy")


def read_blosc(file_name):
    with open(f"{file_name}.bl", "rb") as r:
        b_arr = r.read()
    return blosc.unpack_array(b_arr)
```

`os.sync()` forces the OS to flush IO buffers to disk, ensuring a fair comparison between methods.

---

## Exercise 2.1 `[AUTOLAB]`

Write a Python program that takes a number `n` as a command line argument. It must then generate a 3-dimensional NumPy array of zeros with size `n × n × n` and `dtype='uint8'`. The program must then measure and print the time it takes to perform each of the following operations:

- Save the array to a file using the provided `write_numpy`.
- Save the array to a file using the provided `write_blosc`.
- Read the array from the created file using the provided `read_numpy`.
- Read the array from the created file using the provided `read_blosc`.

Note: you must print the time for each operation separately, i.e., four time values should be printed. You may print them on a single line separated by a space or on 4 separate lines.

> **Solution:**
>
> The student implementation in `blosc_ex.py` wraps each I/O function with a `time_it` decorator that prints elapsed time, then calls them in sequence for the zeros array.
>
> ```python
> import sys
> import os
> import blosc
> from time import perf_counter
> import numpy as np
> from functools import wraps
>
> def time_it(func):
>     @wraps(func)
>     def wrapper(*args):
>         start = perf_counter()
>         result = func(*args)
>         end = perf_counter()
>         print(f"{end - start}")
>         return result
>     return wrapper
>
> @time_it
> def write_numpy(arr, file_name):
>     np.save(f"{file_name}.npy", arr)
>     if hasattr(os, 'sync'):
>         os.sync()
>
> @time_it
> def write_blosc(arr, file_name, cname="lz4"):
>     b_arr = blosc.pack_array(arr, cname=cname)
>     with open(f"{file_name}.bl", "wb") as w:
>         w.write(b_arr)
>     if hasattr(os, 'sync'):
>         os.sync()
>
> @time_it
> def read_numpy(file_name):
>     return np.load(f"{file_name}.npy")
>
> @time_it
> def read_blosc(file_name):
>     with open(f"{file_name}.bl", "rb") as r:
>         b_arr = r.read()
>     return blosc.unpack_array(b_arr)
>
> def main():
>     n = int(sys.argv[1])
>     A = np.zeros((n, n, n), dtype='uint8')
>
>     write_numpy(A, 'write')
>     write_blosc(A, 'write')
>     read_numpy('write')
>     read_blosc('write')
>
> if __name__ == '__main__':
>     main()
> ```

---

## Exercise 2.2 `[AUTOLAB]`

Make a job script that runs your program with `n` = 256, 512 and 1024 and submit it to the `hpc` queue. Request 1 core and remember to request enough memory to keep the arrays in memory.

> **Solution:**
>
> A 1024 × 1024 × 1024 `uint8` array requires 1 GB of RAM; the job script requests 4 GB to be safe. The three array sizes are run sequentially.
>
> ```bash
> #!/bin/bash
> #BSUB -J python
> #BSUB -q hpc
> #BSUB -W 10
> #BSUB -R "rusage[mem=4096MB]"
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -o python%J.out
> #BSUB -e python_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> python blosc_ex.py 256
> python blosc_ex.py 512
> python blosc_ex.py 1024
> ```

---

## Exercise 2.3 `[PRACTICE]`

Do the same, but now, instead of zero entries, the array will have tiled values of integers up to 256. Hint: You may use the following snippet to generate the array:

```python
tiled_array = np.tile(
    np.arange(256, dtype='uint8'),
    (n // 256) * n * n,
).reshape(n, n, n)
```

> **Solution:**
>
> Use the provided snippet as-is. The tiled array has a repeating `[0, 1, 2, ..., 255, 0, 1, ...]` pattern, which is highly compressible (highly structured/repetitive data). Blosc with LZ4 should compress this very well, and the read+decompress time will be much shorter than reading the raw numpy file.
>
> The student implementation in `blosc_quiz.py` wraps this in a `tiled(n)` function:
>
> ```python
> def tiled(n):
>     tiled_array = np.tile(
>         np.arange(256, dtype='uint8'),
>         (n // 256) * n * n,
>     ).reshape(n, n, n)
>     print(f"Tiled ({n}): ")
>     write_read(tiled_array)
> ```

---

## Exercise 2.4 `[PRACTICE]`

Do it one last time, but now using random integer entries from 0 to 256. Hint: Use the `np.random.randint` function.

> **Solution:**
>
> ```python
> rand_array = np.random.randint(
>     0, 256, size=(n,) * 3, dtype='uint8'
> )
> ```
>
> Random data has essentially no compressible structure, so Blosc will achieve little to no compression. The compressed file will be nearly the same size as the raw `.npy` file, and the extra CPU time spent on (de)compression will make Blosc slower than plain numpy for random data.

---

## Exercise 2.5 `[AUTOLAB]`

Compare the time it takes to write and read the files in the three items above for the different `n`. When should we use Blosc? Why? Compare also the sizes of the generated files.

> **Solution:**
>
> Key observations across the three data types:
>
> | Data type | Compressibility | Blosc file size | Blosc read faster? |
> |-----------|----------------|-----------------|-------------------|
> | Zeros     | Extremely high | Tiny (near 0)   | Yes, strongly     |
> | Tiled     | High           | Much smaller    | Yes               |
> | Random    | None           | ~Same as .npy   | No (overhead only)|
>
> **When to use Blosc:** Use Blosc when your data is compressible (structured, repetitive, or sparse) **and** disk I/O is the bottleneck (i.e., reading raw bytes from disk is slower than the CPU overhead of decompression). For random or near-incompressible data, Blosc adds CPU overhead with no I/O savings, making it slower overall.
>
> **Why it works:** Compression reduces the number of bytes written/read from disk. Since disk bandwidth is much lower than CPU/memory bandwidth, reading fewer bytes and then decompressing in RAM is faster than reading many bytes directly — as long as the data compresses well.

### Exercise 2.5a `[PRACTICE]`

In the blosc implementation, change the compression algorithm from `cname="lz4"` to `cname="zstd"`. What improvements do you observe? What was the cost of these improvements?

> **Solution:**
>
> `zstd` (Zstandard) achieves **better compression ratios** than `lz4` — compressed files are smaller, and read times improve because fewer bytes are fetched from disk.
>
> The **cost** is increased CPU time for both compression (write) and decompression (read). `zstd` is computationally heavier than `lz4`. For latency-sensitive workloads or when CPU is the bottleneck (e.g., data already fits in cache), `lz4` may be preferable. For I/O-bound workloads with highly compressible data, `zstd` wins because the extra CPU work is more than offset by the reduced I/O.
