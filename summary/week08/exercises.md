# Week 8 Exercises — Memory-Mapped Files, Zarr, Shared Memory

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Exercise 1.1 `[AUTOLAB]`](#exercise-11-autolab)
- [Exercise 1.2 `[PRACTICE]`](#exercise-12-practice)
- [Exercise 1.3 `[PRACTICE]`](#exercise-13-practice)
- [Exercise 1.4 `[PRACTICE]`](#exercise-14-practice)
- [Exercise 1.5 `[PRACTICE]`](#exercise-15-practice)
- [Exercise 2.1 `[AUTOLAB]`](#exercise-21-autolab)
- [Exercise 2.2 `[PRACTICE]`](#exercise-22-practice)
  - [2.2a `[PRACTICE]`](#22a-practice)
- [Exercise 2.3 `[AUTOLAB]`](#exercise-23-autolab)
- [Exercise 2.4 `[PRACTICE]`](#exercise-24-practice)
- [Exercise 3.1 `[AUTOLAB]`](#exercise-31-autolab)
- [Exercise 3.2 `[PRACTICE]`](#exercise-32-practice)
- [Exercise 3.3 `[PRACTICE]`](#exercise-33-practice)
  - [3.3a `[PRACTICE]`](#33a-practice)
  - [3.3b `[PRACTICE]`](#33b-practice)
- [Exercise 3.4 `[PRACTICE]`](#exercise-34-practice)

---

---

## Exercise 1.1 `[AUTOLAB]`

Make a Python program which takes two command line arguments: a path to a Pandas data frame and a chunk size. Compute the total precipitation where you process the dataframe in chunks of the given size — do *not* load the entire dataframe. Also, do *not* perform any memory reducing operations (like in week 7). Finally, print the result (and only the result).

The dataframe is at: `/dtu/projects/02613_2025/data/dmi/2023_01.csv.zip`

> **Solution:**
>
> Process each chunk independently, accumulate the precipitation sum, and print the total. `pd.read_csv` with `chunksize` returns an iterator — never a full dataframe.
>
> ```python
> import sys
> import pandas as pd
>
> def main():
>     path = sys.argv[1]
>     chunksize = int(sys.argv[2])
>
>     total = 0.0
>     for chunk in pd.read_csv(path, chunksize=chunksize):
>         total += float(chunk[chunk['parameterId'] == 'precip_past10min']['value'].sum())
>
>     print(total)
>
> if __name__ == '__main__':
>     main()
> ```

---

## Exercise 1.2 `[PRACTICE]`

Measure the run time and memory use for chunk sizes 1,000, 10,000, 100,000 and 1,000,000. Use:

```
/usr/bin/time -f"mem=%M KB runtime=%e s" python script.py
```

Perform the measurement as a batch job. Select 1 core and a specific CPU so the results are repeatable. What do you observe? How does the run time compare to the non-chunked version?

Hint: `time` prints its output to `stderr` and will be added to the `.err` file. If you want, you can redirect it to `stdout` by adding `2>&1` at the end of your timing command.

> **Solution:**
>
> Sample batch job (XeonGold6226R):
>
> ```bash
> #!/bin/sh
> #BSUB -q hpc
> #BSUB -J ppandas
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -R "rusage[mem=16GB]"
> #BSUB -R "select[model == XeonGold6226R]"
> #BSUB -W 00:10
> #BSUB -o batch_output/precip_pandas_%J.out
> #BSUB -e batch_output/precip_pandas_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> echo $CPUTYPE
>
> set -e
>
> for c in 1000 10000 100000 1000000; do
>     echo $c
>     /usr/bin/time -f"mem=%M KB runtime=%e s" \
>     python precip_pandas.py \
>         /dtu/projects/02613_2025/data/dmi/2023_01.csv.zip $c \
>     2>&1
> done
> ```
>
> Results (XeonGold6226R):
>
> | Chunk size | Memory    | Runtime  |
> |-----------|-----------|----------|
> | 1,000     | 131,712 KB | 27.61 s |
> | 10,000    | 137,628 KB | 16.70 s |
> | 100,000   | 197,120 KB | 15.92 s |
> | 1,000,000 | 559,220 KB | 16.50 s |
>
> A chunk size of 1,000 is slower due to iteration overhead. From 10,000 upward the runtime is roughly equal — no performance penalty from chunking. Memory grows with chunk size as expected.
>
> The non-chunked version ran in 17.19 s with 2,040,052 KB memory — slightly slower than chunked and with ~15x higher peak memory.

---

## Exercise 1.3 `[PRACTICE]`

Convert the CSV file to a chunked parquet file. Use a chunksize of 100,000. Hint: See section 8.3.2 in Fast Python.

> **Solution:**
>
> Write each CSV chunk as a Parquet row group using a `ParquetWriter` that is kept open across chunks:
>
> ```python
> import sys
> import pandas as pd
> import pyarrow as pa
> import pyarrow.parquet as pq
>
> infile = sys.argv[1]
> outfile = sys.argv[2]
>
> df_chunks = pd.read_csv(
>     infile,
>     dtype={'parameterId': 'category'},
>     chunksize=100_000
> )
>
> first = True
> writer = None
> for chunk in df_chunks:
>     chunk_table = pa.Table.from_pandas(chunk)
>     schema = chunk_table.schema
>     if first:
>         first = False
>         writer = pq.ParquetWriter(outfile, schema=schema)
>     writer.write_table(chunk_table)
> writer.close()
> ```
>
> Run as:
> ```
> python pandas_to_parquet.py \
>     /dtu/projects/02613_2025/data/dmi/2023_01.csv.zip dmi_chunks.parquet
> ```

---

## Exercise 1.4 `[PRACTICE]`

Modify your Python program from exercise 1.1 to receive the path to a chunked parquet file instead of a path to a CSV file. What is the new run time and memory use? Perform your measurement with a batch script and select the *same* CPU model.

> **Solution:**
>
> Use `pq.ParquetFile` and iterate over row groups with `read_row_group(i)`:
>
> ```python
> import sys
> import pyarrow.parquet as pq
>
> def precip(df):
>     return float(df[df['parameterId'] == 'precip_past10min']['value'].sum())
>
> if __name__ == '__main__':
>     fname = sys.argv[1]
>
>     total = 0
>     pf = pq.ParquetFile(fname)
>     for i in range(pf.num_row_groups):
>         group = pf.read_row_group(i)
>         total += precip(group.to_pandas())
>
>     print(total)
> ```
>
> Results (XeonGold6226R, chunksize 100,000 parquet):
>
> - Memory: 218,484 KB
> - Runtime: **5.09 s** (~3x faster than the Pandas CSV-chunked version)

---

## Exercise 1.5 `[PRACTICE]`

Finally, modify your program further to only load the relevant columns. What is your final run time?

> **Solution:**
>
> The `precip` function only accesses `parameterId` and `value`. Pass `columns=['parameterId', 'value']` to `read_row_group` (or `usecols` for `pd.read_csv`):
>
> ```python
> group = pf.read_row_group(i, columns=['parameterId', 'value'])
> ```
>
> Results (XeonGold6226R):
>
> - Memory: 143,160 KB
> - Runtime: **1.17 s**
>
> That is 4.3x faster than loading all columns and **13.6x faster** than the original Pandas CSV approach.

---

## Exercise 2.1 `[AUTOLAB]`

Make a Python program that creates an N×N array with the Mandelbrot set. It must take the number N as a command line argument. The limits for the real and imaginary values must be the same as in weeks 5 & 6. The program must store the data for the Mandelbrot array in a memory mapped array.

> **Solution:**
>
> Create the memmap with `mode='w+'` (create/overwrite), compute the full Mandelbrot array using `mandelbrot()` from `mandelbrotref.py`, write it to the memmap, then `del` to flush to disk.
>
> ```python
> import sys
> import numpy as np
> import os
> sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
> from mandelbrotref import mandelbrot
>
> def main():
>     N = int(sys.argv[1])
>
>     mm = np.memmap('mandelbrot.raw', dtype='int32', mode='w+', shape=(N, N))
>     result = mandelbrot(N)
>     mm[:] = result
>     del mm  # flush to disk
>
> if __name__ == '__main__':
>     main()
> ```

---

## Exercise 2.2 `[PRACTICE]`

Parallelize your program for computing the Mandelbrot set in a memory mapped array by modifying your solution from weeks 5 and 6. Each process should write its part of the Mandelbrot array directly to the memory mapped array.

### 2.2a `[PRACTICE]`

Run the program for a 1000×1000 array. What is your speed-up? What is your run time compared to the in-memory version from weeks 5 and 6? Run your measurement as a batch job and select a specific CPU model so your results are repeatable.

> **Solution:**
>
> Divide the rows of the N×N memmap across worker processes (same row-chunking strategy as weeks 5 & 6). Each process opens the memmap with `mode='r+'`, computes its row slice, writes it, then `del`s to flush.
>
> Sample batch job:
>
> ```bash
> #!/bin/sh
> #BSUB -q hpc
> #BSUB -J memmap_p
> #BSUB -n 32
> #BSUB -R "span[hosts=1]"
> #BSUB -R "rusage[mem=1GB]"
> #BSUB -R "select[model == XeonGold6226R]"
> #BSUB -W 00:10
> #BSUB -o batch_output/memmap_parallel_%J.out
> #BSUB -e batch_output/memmap_parallel_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> set -e
> for n in 1 2 4 8 16 24 32; do
>     echo $n
>     python mandelbrot_memmap.py 1000 $n
> done
> ```
>
> Results: near-perfect linear speed-up up to 32 processes. Final runtime at 32 processes: **0.8 s**. The in-memory RAM version from weeks 5 & 6 achieved 0.9 s — essentially the same. Writing to a memmap adds negligible overhead compared to the computation itself.

---

## Exercise 2.3 `[AUTOLAB]`

Make a Python program which reads the saved Mandelbrot array, downsamples it, and saves the result as a PNG image. It must take three command line arguments: the path to the saved Mandelbrot array from exercise 2.1, the size N of the N×N array and an integer step length n. It must then open the specified array as a NumPy memmap, load every nth row and column, and then save the resulting array as a PNG file.

*Example:* For a 5×5 array with step n=3, the downsampled result is a 2×2 array containing elements at positions (0,0), (0,3), (3,0), (3,3). With n=2, the result is a 3×3 array containing every other row and column.

> **Solution:**
>
> Open the raw file as a read-only memmap, use NumPy stride slicing `[::n, ::n]` to select every nth row and column (this does not load the skipped rows), then save with matplotlib.
>
> ```python
> import sys
> import numpy as np
> import matplotlib.pyplot as plt
>
> def main():
>     path = sys.argv[1]
>     N = int(sys.argv[2])
>     n = int(sys.argv[3])
>
>     mm = np.memmap(path, dtype='int32', mode='r', shape=(N, N))
>     downscaled = mm[::n, ::n]
>
>     plt.imshow(downscaled, cmap='hot', extent=(-2, 2, -2, 2))
>     plt.axis('off')
>     plt.savefig('mandelbrot.png', bbox_inches='tight', pad_inches=0)
>
> if __name__ == '__main__':
>     main()
> ```

---

## Exercise 2.4 `[PRACTICE]`

At `/dtu/projects/02613_2025/data/mandelbrot/mandelbrot.raw` is a file with a 4000×4000 Mandelbrot set array. Run the downsampling program on this array for step lengths 1, 2, 4, 8, 16. If you hardcoded the size as 1000×1000, remember to change it.

Measure the run time and peak memory use with `/usr/bin/time`. Plot the results. Perform the measurements as a batch job and select a CPU model so the results are repeatable. What happens to the memory and run time as the step length n increases?

> **Solution:**
>
> Sample batch job:
>
> ```bash
> #!/bin/sh
> #BSUB -q hpc
> #BSUB -J mandelpic
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -R "rusage[mem=2GB]"
> #BSUB -R "select[model == XeonGold6226R]"
> #BSUB -W 00:10
> #BSUB -o batch_output/mandelbrot_pic_%J.out
> #BSUB -e batch_output/mandelbrot_pic_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> for n in 1 2 4 8 16; do
>     echo $n
>     /usr/bin/time -f"mem=%M KB runtime=%e s" \
>     python mandelbrot_downscale.py \
>     /dtu/projects/02613_2025/data/mandelbrot/mandelbrot.raw 4000 $n \
>     2>&1
> done
> ```
>
> As the step length n increases, both **run time and peak memory decrease**. This is the key advantage of memmap: only the accessed pages are read from disk into memory. With a larger step, fewer rows and columns are touched, so the OS loads fewer pages. The entire 4000×4000 array (61 MB) is never fully loaded into RAM.

---

## Exercise 3.1 `[AUTOLAB]`

Make a Python program that creates a Zarr array containing the Mandelbrot set. The size of the array must be N×N. The program must take two command line arguments: the size N and a positive integer C which is the size of the C×C chunks. Structure your program so it fills out the chunks one at a time.

> **Solution:**
>
> Open a Zarr store with `zarr.open(..., chunks=(C, C))`, then iterate over all (row_start, col_start) chunk origins and compute each C×C block independently.
>
> ```python
> import sys
> import numpy as np
> import zarr
> import os
> sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
> from mandelbrotref import mandelbrot_escape_time
>
> def compute_chunk(row_start, col_start, chunk_rows, chunk_cols, N):
>     all_pts = np.linspace(-2, 2, N + 1)[:-1]
>     x = all_pts[col_start:col_start + chunk_cols]
>     y = all_pts[row_start:row_start + chunk_rows]
>     xpts, ypts = np.meshgrid(x, y)
>     points = 1j * xpts.ravel() + ypts.ravel()
>     result = np.array([mandelbrot_escape_time(c) for c in points])
>     return result.reshape(chunk_rows, chunk_cols)
>
> def main():
>     N = int(sys.argv[1])
>     C = int(sys.argv[2])
>
>     z = zarr.open('mandelbrot.zarr', mode='w', shape=(N, N),
>                   chunks=(C, C), dtype='int32')
>
>     for row_start in range(0, N, C):
>         for col_start in range(0, N, C):
>             chunk_rows = min(C, N - row_start)
>             chunk_cols = min(C, N - col_start)
>             z[row_start:row_start + chunk_rows,
>               col_start:col_start + chunk_cols] = compute_chunk(
>                 row_start, col_start, chunk_rows, chunk_cols, N)
>
> if __name__ == '__main__':
>     main()
> ```

---

## Exercise 3.2 `[PRACTICE]`

Parallelize your program using multiprocessing. Each process should fill one chunk of the Zarr array.

> **Solution:**
>
> Build a list of all (row_start, col_start) chunk coordinates, then use `multiprocessing.Pool` to dispatch `compute_chunk` calls in parallel. Each worker computes its chunk independently and writes it to the shared Zarr store (Zarr uses file-level locking per chunk, so concurrent writes to different chunks are safe).
>
> Combine your code from exercise 3.1 with the `Pool.starmap` / `Pool.map` pattern from weeks 5 and 6. The number of processes should equal the number of available cores (`os.cpu_count()`).

---

## Exercise 3.3 `[PRACTICE]`

Time the runtime of the program for array size 1000 and chunk sizes 10, 25, 50, 100, and 200. Use a process for each available core (e.g., 32 processes for a 32-core CPU). Run your timing as a batch job and specify a CPU model so the result is repeatable.

### 3.3a `[PRACTICE]`

What happens with the run times? What chunk size gives the fastest execution?

### 3.3b `[PRACTICE]`

What happens with the size of the saved Zarr array? What chunk size gives the smallest result? Hint: measure directory size with `du -sh <filename>.zarr`.

> **Solution:**
>
> Sample batch job:
>
> ```bash
> #!/bin/sh
> #BSUB -q hpc
> #BSUB -J zarr
> #BSUB -n 24
> #BSUB -R "span[hosts=1]"
> #BSUB -R "rusage[mem=1GB]"
> #BSUB -R "select[model == XeonGold6226R]"
> #BSUB -W 00:10
> #BSUB -o batch_output/zarr_%J.out
> #BSUB -e batch_output/zarr_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> set -e
> for n in 10 25 50 100 200; do
>     echo $n
>     python mandelbrot_zarr.py 1000 $n mandelbrot_${LSB_JOBID}.zarr
>     du -sh mandelbrot_${LSB_JOBID}.zarr
> done
> ```
>
> Results (XeonGold6226R, 24 processes):
>
> | Chunk size | Runtime  | Zarr size |
> |-----------|----------|-----------|
> | 10×10     | 24.2 s   | 285 MB    |
> | 25×25     | 3.6 s    | 38 MB     |
> | 50×50     | **1.07 s** | 9.5 MB  |
> | 100×100   | 1.25 s   | 2.4 MB    |
> | 200×200   | 4.3 s    | 656 KB    |
>
> **Runtime:** Fastest at chunk size 50×50. Very small chunks create too much overhead (many tiny tasks and file writes). Very large chunks reduce parallelism since fewer chunks are available for the pool to distribute.
>
> **Storage:** Larger chunks compress better. Zarr compresses each chunk independently; regions away from the Mandelbrot boundary have uniform values and compress very efficiently. The memmap raw file takes ~5.3 MB — at chunk size 100 Zarr uses only 2.4 MB with comparable speed.

---

## Exercise 3.4 `[PRACTICE]`

How does the performance compare with your memmap implementation?

> **Solution:**
>
> The memmap parallel implementation achieved **0.8 s** at 32 processes. The fastest Zarr implementation achieved **1.1 s** at 32 processes — about 35% slower. However, Zarr stores the result with significantly less disk space (2.4 MB vs 5.3 MB at comparable speed, down to 656 KB at larger chunk sizes). As dataset sizes grow, this storage-vs-speed trade-off becomes increasingly relevant.
