# Week 13 Exercises — HPC Pitfalls: Common Errors

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Exercise 1 — Excessive I/O `[PRACTICE]`](#exercise-1-excessive-io-practice)
  - [1.1 `[PRACTICE]`](#11-practice)
  - [1.2 `[PRACTICE]`](#12-practice)
  - [1.3 `[PRACTICE]`](#13-practice)
- [Exercise 2 — Multi-Threaded NumPy `[AUTOLAB]`](#exercise-2-multi-threaded-numpy-autolab)
  - [2.1 `[PRACTICE]`](#21-practice)
  - [2.2 `[PRACTICE]`](#22-practice)
  - [2.3 `[AUTOLAB]`](#23-autolab)
  - [2.4 `[PRACTICE]`](#24-practice)
  - [2.5 `[PRACTICE]`](#25-practice)

---

---

## Exercise 1 — Excessive I/O `[PRACTICE]`

Consider the following Python program that prints a short story about a number for the first 100,000 integers:

```python
for i in range(100_000):
    # Story by ChatGPT
    print(
        f"Once upon a time, {i} felt lonely, being just one among many. "
        f"Dreaming of distinction, {i} doubled itself, becoming 2*{i}. "
        f"Still, it wasn't enough, so {i} squared itself, turning into {i}^2. "
        f" Realizing every transformation retained its essence, "
        f"{i} learned to cherish its identity, understanding it was unique, "
        f"infinite, essential-forever {i}, the protagonist of its own "
        f"mathematical saga."
    )
```

### 1.1 `[PRACTICE]`

Make a job script that runs the above Python program. Add the `-u` option to Python to run with unbuffered output. You must submit the job to the `hpc` queue, request 1 core and a CPU model for repeatability. Also, make sure you specify output files with the `-o/-e` options and that the files are under `/work3/02613/dump/`.

> **Solution:**
>
> ```bash
> #!/bin/bash
> #BSUB -J print
> #BSUB -q hpc
> #BSUB -W 00:10
> #BSUB -n 1
> #BSUB -R "select[model==XeonE5_2650v4]"
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -o /work3/02613/dump/printlots_%J.out
> #BSUB -e /work3/02613/dump/printlots_%J.err
>
> # Initialize Python environment
> source /dtu/projects/02613_2024/conda/conda_init.sh
> conda activate 02613
>
> python -u printlots.py
> ```

### 1.2 `[PRACTICE]`

Make another job script where you manually redirect the output of the Python program to a separate set of files. Again, make sure these files are also under `/work3/02613/dump/`.
Hint: see the slides.

> **Solution:**
>
> Instead of relying solely on LSF's `-o/-e` channels, manually redirect stdout and stderr from the Python process itself:
>
> ```bash
> #!/bin/bash
> #BSUB -J print
> #BSUB -q hpc
> #BSUB -W 00:10
> #BSUB -n 1
> #BSUB -R "select[model==XeonE5_2650v4]"
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -o /work3/02613/dump/printlots_%J.out
> #BSUB -e /work3/02613/dump/printlots_%J.err
>
> # Initialize Python environment
> source /dtu/projects/02613_2024/conda/conda_init.sh
> conda activate 02613
>
> python -u printlots.py \
>     1> /work3/02613/dump/output_${LSB_JOBID}.txt \
>     2> /work3/02613/dump/error_${LSB_JOBID}.txt
> ```

### 1.3 `[PRACTICE]`

Submit both jobs. Check the job resource summaries at the end of the output files. Did the output get redirected as expected? What is the run time difference?

> **Solution:**
>
> The job **without** manual redirection (output goes through LSF's `-o/-e` channels):
>
> ```
> CPU time :                                   7.37 sec.
> Max Memory :                                 6 MB
> Run time :                                   80 sec.
> ```
>
> The job **with** manual redirection (Python writes directly to files):
>
> ```
> CPU time :                                   1.48 sec.
> Max Memory :                                 -
> Run time :                                   3 sec.
> ```
>
> Without manual redirection the job took 80 seconds. With manual redirection the run time dropped to 3 seconds — **26x faster**. The pitfall is that routing large volumes of output through LSF's own I/O channels is extremely slow. Writing directly to files from the shell bypasses that overhead entirely.

---

## Exercise 2 — Multi-Threaded NumPy `[AUTOLAB]`

Consider the following Python program, which creates two sets of 100 random 1000x1000 matrices and then multiplies them together. At the end, it prints out the time it took.

```python
from time import perf_counter as time
import numpy as np

def matmuls(A, B):
    assert A.shape[0] == B.shape[0], "A and B must hold same number of matrices"
    assert A.shape[2] == B.shape[1], "A and B must hold compatible matrices"
    n = A.shape[0]
    C = np.empty((n, A.shape[1], B.shape[2]))
    for i in range(n):
        C[i] = np.matmul(A[i], B[i])
    return C

A = np.random.rand(100, 1000, 1000)
B = np.random.rand(100, 1000, 1000)
t0 = time()
C = matmuls(A, B)
t1 = time()
print(t1 - t0)
```

### 2.1 `[PRACTICE]`

Write a job script that runs the Python program. You must submit the job to the `hpc` queue, request 1 core and a CPU model for repeatability. Submit the job. What is the run time?

> **Solution:**
>
> ```bash
> #!/bin/bash
> #BSUB -J matmuls
> #BSUB -q hpc
> #BSUB -W 00:10
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -R "select[model==XeonE5_2650v4]"
> #BSUB -R "rusage[mem=4GB]"
> #BSUB -o batch_output/matmuls_%J.out
> #BSUB -e batch_output/matmuls_%J.err
>
> source /dtu/projects/02613_2024/conda/conda_init.sh
> conda activate 02613
>
> python -u matmuls.py
> ```
>
> With the `XeonE5_2650v4` model the run time was approximately **5.87 seconds**.

### 2.2 `[PRACTICE]`

Change the number of cores to 8 cores and submit again. Did the run time change?

> **Solution:**
>
> Change `#BSUB -n 1` to `#BSUB -n 8`. The run time was approximately **5.96 seconds** — effectively no change. NumPy does **not** automatically use the extra cores just because they were allocated; you must explicitly tell it to.

### 2.3 `[AUTOLAB]`

Change the job script to enable NumPy multi-threading with 8 threads (one for each requested core). Submit again. Did the run time change?
Hint: see the slides.

> **Solution:**
>
> Set the thread-count environment variables before calling Python. The student's `matrix_multiplication_job.sh` demonstrates this:
>
> ```bash
> #!/bin/bash
> #BSUB -J matmuls
> #BSUB -q hpc
> #BSUB -n 8
> #BSUB -R "span[hosts=1]"
> #BSUB -R "rusage[mem=4GB]"
> #BSUB -R "select[model==XeonGold6226R]"
> #BSUB -W 00:10
> #BSUB -o matmuls_%J.out
> #BSUB -e matmuls_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> # Enable NumPy multi-threading — set to match the number of allocated cores
> NUM_THREADS=8
> OMP_NUM_THREADS=$NUM_THREADS
> MKL_NUM_THREADS=$NUM_THREADS
> OPENBLAS_NUM_THREADS=$NUM_THREADS
>
> python matmul.py
> ```
>
> With 8 threads the run time dropped to approximately **1.28 seconds** — 4.6x faster. The key is that `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` must match the number of allocated cores; otherwise NumPy's backend either spawns too few or too many threads.

### 2.4 `[PRACTICE]`

Change the Python program such that the `matmuls` function is parallelized with multi-threading using a [`ThreadPool`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.ThreadPool) from the `multiprocessing` module. Submit again. What is the run time now? Why can we use multi-threading for this program instead of using multi-*processing*?
Hint: see the slides from week 5.

> **Solution:**
>
> Use `ThreadPool.starmap` to distribute individual matrix multiplications across threads:
>
> ```python
> from time import perf_counter as time
> import numpy as np
> from multiprocessing.pool import ThreadPool
>
> def matmuls(A, B):
>     assert A.shape[0] == B.shape[0], "A and B must hold same number of mats"
>     assert A.shape[2] == B.shape[1], "A and B must hold compatible matrices"
>     n = A.shape[0]
>     with ThreadPool(8) as p:
>         C = np.concatenate(p.starmap(np.matmul, zip(A, B)))
>     return C
>
> A = np.random.rand(100, 1000, 1000)
> B = np.random.rand(100, 1000, 1000)
> t0 = time()
> C = matmuls(A, B)
> t1 = time()
> print(t1 - t0)
> ```
>
> Running with 8 threads in the pool **and** NumPy multi-threading still enabled gives approximately **1.87 seconds** — slower than expected because each pool thread also spawns 8 NumPy threads internally, creating far more threads than available cores. This leads to heavy scheduling overhead.
>
> We can use a `ThreadPool` (rather than a `ProcessPool`) here because `np.matmul` **releases the GIL** during computation. Since each thread spends most of its time inside native BLAS code with the GIL released, multiple threads can run truly in parallel despite Python's GIL.

### 2.5 `[PRACTICE]`

Change your job script to disable NumPy multi-threading (i.e., remove what you added in exercise 2.3). Submit the job one last time. What is the run time now?

> **Solution:**
>
> Remove (or set to 1) the `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` variables so each matrix multiplication runs single-threaded, while the `ThreadPool` provides the outer parallelism. The run time is approximately **1.33 seconds** — on par with the fully multi-threaded NumPy approach from 2.3, and faster than the doubly-parallel version in 2.4.
>
> **Key takeaway:** When writing parallel code, be aware that libraries like NumPy may themselves spawn threads. Naively combining a thread pool with multi-threaded NumPy leads to over-subscription (more threads than cores) and poor performance. Always match the total number of active threads to the number of allocated cores.
