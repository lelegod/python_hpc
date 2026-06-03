# HPC Pitfalls — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Missing export on Thread Env Vars](#q1--missing-export-on-thread-env-vars)
- [Q2 — Shell Redirection vs LSF -o](#q2--shell-redirection-vs-lsf--o)
- [Q3 — ThreadPool + Multi-Threaded NumPy Oversubscription](#q3--threadpool--multi-threaded-numpy-oversubscription)
- [Q4 — Requesting Cores Without span[hosts=1]](#q4--requesting-cores-without-spanhosts1)
- [Q5 — Diagnosing No Speedup: Cores vs Threads](#q5--diagnosing-no-speedup-cores-vs-threads)
- [Q6 — Buffered vs Unbuffered Python Output](#q6--buffered-vs-unbuffered-python-output)
- [Q7 — Memory Request Calculation](#q7--memory-request-calculation)
- [Q8 — Subshell Export Trap](#q8--subshell-export-trap)
- [Q9 — Fixing Oversubscription by Setting OMP to 1](#q9--fixing-oversubscription-by-setting-omp-to-1)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2--generated-practice-questions-exam-day-focus)
- [Q10 — Login Node Computation](#q10--login-node-computation)
- [Q11 — Multiple Pitfalls in One Script](#q11--multiple-pitfalls-in-one-script)
- [Q12 — Correct ThreadPool + NumPy Setup](#q12--correct-threadpool--numpy-setup)
- [Q13 — Identifying the Fastest I/O Approach](#q13--identifying-the-fastest-io-approach)
- [Q14 — ProcessPool vs ThreadPool for NumPy](#q14--processpool-vs-threadpool-for-numpy)
- [Q15 — MKL_NUM_THREADS Alone Is Not Enough](#q15--mkl_num_threads-alone-is-not-enough)
- [Q16 — Output File Location Pitfall](#q16--output-file-location-pitfall)
- [Q17 — Predicting Run Time from Thread Count](#q17--predicting-run-time-from-thread-count)
- [Q18 — Fixing All Pitfalls in a Combined Script](#q18--fixing-all-pitfalls-in-a-combined-script)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q19 — rusage Per-Core Trap](#q19--rusage-per-core-trap)
- [Q20 — Pure Python ThreadPool Speedup Prediction](#q20--pure-python-threadpool-speedup-prediction)
- [Q21 — time.time vs time.perf_counter Output](#q21--timetime-vs-timeperf_counter-output)
- [Q22 — ProcessPool with Large Arrays](#q22--processpool-with-large-arrays)
- [Q23 — What os.environ.get Sees When export Is Missing](#q23--what-osenvironget-sees-when-export-is-missing)
- [Q24 — Wall Time Kill: What Output Remains](#q24--wall-time-kill-what-output-remains)
- [Q25 — Counting Active Threads from Config Variables](#q25--counting-active-threads-from-config-variables)
- [Q26 — OMP=0 Undefined Behaviour vs OMP=1](#q26--omp0-undefined-behaviour-vs-omp1)
- [Q27 — LSB_JOBID in Shell Redirect vs %J in BSUB](#q27--lsb_jobid-in-shell-redirect-vs-j-in-bsub)
- [Q28 — Amdahl Calculation from Timing Output](#q28--amdahl-calculation-from-timing-output)
- [Q29 — Process Spawn Overhead for Tiny Tasks](#q29--process-spawn-overhead-for-tiny-tasks)
- [Q30 — @njit First-Call Timing Trap](#q30--njit-first-call-timing-trap)
- [Q31 — time.time() Resolution for Sub-ms Benchmarks](#q31--timetime-resolution-for-sub-ms-benchmarks)
- [Q32 — arr.tolist() in a Tight Loop](#q32--arrtolist-in-a-tight-loop)
- [Q33 — Cold Cache vs Warm Cache in Profiling](#q33--cold-cache-vs-warm-cache-in-profiling)

---

> Format: Each question shows a buggy or suboptimal job script or Python code — identify the pitfall or fix.
> Exam frequency: **Week 13 topic — common exam target**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#question-1)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — Missing export on Thread Env Vars

```bash
#!/bin/bash
#BSUB -J matmuls
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 00:10
#BSUB -o matmuls_%J.out
#BSUB -e matmuls_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

NUM_THREADS=8
OMP_NUM_THREADS=$NUM_THREADS
MKL_NUM_THREADS=$NUM_THREADS
OPENBLAS_NUM_THREADS=$NUM_THREADS

python matmul.py
```

This script requests 8 cores and sets thread count variables. Yet NumPy still runs with only 1 thread. What is the single bug?

**A)** `NUM_THREADS=8` should be `NUM_THREADS="8"` — environment variables require quoted strings

**B)** The thread variables are assigned but never exported; Python inherits only the parent shell's exported environment, so it never sees `OMP_NUM_THREADS`

**C)** `span[hosts=1]` conflicts with multi-threading and must be removed before the thread vars take effect

**D)** The conda environment must be activated before setting thread variables, not after

**Answer: B**

- A) Incorrect — Bash does not require quotes around integer values in variable assignment; `NUM_THREADS=8` and `NUM_THREADS="8"` are equivalent.
- B) Correct — `OMP_NUM_THREADS=$NUM_THREADS` stores the value in the shell's local variable table but does NOT add it to the environment passed to child processes. Each variable needs `export` (e.g., `export OMP_NUM_THREADS=$NUM_THREADS`) for Python to inherit it. This is the exact pitfall from the Week 13 `matrix_multiplication_job.sh` example.
- C) Incorrect — `span[hosts=1]` constrains node placement and is unrelated to environment variable export mechanics.
- D) Incorrect — Variable assignments are order-independent with respect to conda activation; the bug is purely the missing `export` keyword.

---

## Q2 — Shell Redirection vs LSF -o

```bash
#!/bin/bash
#BSUB -J print
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -R "rusage[mem=512MB]"
#BSUB -o /work3/02613/dump/printlots_%J.out
#BSUB -e /work3/02613/dump/printlots_%J.err

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

python -u printlots.py
```

The program `printlots.py` prints 100,000 verbose lines. This script takes ~80 seconds to run. Which change reduces the run time to ~3 seconds?

**A)** Change `#BSUB -n 1` to `#BSUB -n 4` to allocate more cores for I/O

**B)** Remove the `-u` flag from Python, enabling buffered output to batch writes

**C)** Add shell redirection: `python -u printlots.py 1> /work3/02613/dump/output_${LSB_JOBID}.txt 2> /work3/02613/dump/error_${LSB_JOBID}.txt`

**D)** Change the queue from `hpc` to `gpuv100` to use GPU-accelerated I/O

**Answer: C**

- A) Incorrect — More cores do not accelerate I/O; the bottleneck is the path stdout travels, not CPU availability. Adding cores would not change the routing of print output through LSF's channels.
- B) Incorrect — Removing `-u` would add buffering, meaning Python batches writes before sending them. While this slightly reduces write frequency, the bottleneck is LSF's channel overhead, not write frequency per se. The correct fix is bypassing LSF's channel entirely.
- C) Correct — Redirecting stdout and stderr with `>` and `2>` causes the shell to connect `printlots.py`'s file descriptors directly to files on disk, bypassing LSF's I/O infrastructure entirely. This is what reduced the time from 80 s to 3 s in the Week 13 exercise.
- D) Incorrect — GPU queues are for GPU computation, not I/O acceleration. Stdout redirection works identically on GPU and CPU nodes.

---

## Q3 — ThreadPool + Multi-Threaded NumPy Oversubscription

```python
from time import perf_counter as time
import numpy as np
from multiprocessing.pool import ThreadPool

def matmuls(A, B):
    n = A.shape[0]
    with ThreadPool(8) as p:
        C = np.concatenate(p.starmap(np.matmul, zip(A, B)))
    return C

A = np.random.rand(100, 1000, 1000)
B = np.random.rand(100, 1000, 1000)
t0 = time()
C = matmuls(A, B)
t1 = time()
print(t1 - t0)
```

The job script exports `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, and `OPENBLAS_NUM_THREADS=8`. The measured run time is ~1.87 s — slower than expected. Pure multi-threaded NumPy alone takes ~1.28 s. What is the problem?

**A)** `ThreadPool` cannot call `np.matmul` because NumPy is not thread-safe

**B)** `np.concatenate` at the end is the bottleneck and should be replaced with `np.stack`

**C)** Thread oversubscription: each of the 8 pool threads spawns 8 NumPy BLAS threads internally, creating 64 active threads on 8 cores

**D)** The `zip(A, B)` call copies all data before the pool starts, creating unnecessary memory overhead

**Answer: C**

- A) Incorrect — NumPy releases the GIL during `np.matmul`, making it explicitly safe and effective with `ThreadPool`. This is why ThreadPool is the recommended approach.
- B) Incorrect — `np.concatenate` on 100 results of shape (1000, 1000) is a minor operation compared to the 100 matrix multiplications; it is not the bottleneck.
- C) Correct — With `OMP_NUM_THREADS=8`, each call to `np.matmul` internally spawns 8 BLAS threads. With 8 pool threads each doing this simultaneously, there are 8 × 8 = 64 active threads competing for 8 cores. The context-switching overhead explains why ~1.87 s is slower than the 1.28 s measured with BLAS threading alone.
- D) Incorrect — `zip(A, B)` creates an iterator of array slices without copying data (NumPy slices are views); memory overhead is minimal.

---

## Q4 — Requesting Cores Without span[hosts=1]

```bash
#!/bin/bash
#BSUB -J matmuls
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 00:10
#BSUB -o matmuls_%J.out
#BSUB -e matmuls_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

python matmul_shared.py
```

`matmul_shared.py` uses Python's `multiprocessing.shared_memory` to share a large array between worker processes. On some runs, workers cannot access the shared memory and crash. What BSUB directive is missing?

**A)** `#BSUB -x` — to give the job exclusive access to the node

**B)** `#BSUB -R "span[hosts=1]"` — to ensure all 8 cores are on the same physical node

**C)** `#BSUB -M 4096` — to set the per-process memory limit

**D)** `#BSUB -P 02613` — to associate the job with the correct project account

**Answer: B**

- A) Incorrect — Exclusive node access (`-x`) prevents other users' jobs from sharing the node, but it does not guarantee all allocated cores are on one node. A job without `span[hosts=1]` can still be split across nodes even with `-x`.
- B) Correct — Without `span[hosts=1]`, LSF may allocate the 8 cores across two or more nodes. Shared memory (`/dev/shm`, POSIX shared memory) is a per-node resource; processes on different nodes cannot access each other's shared memory segments, causing crashes.
- C) Incorrect — `-M` sets the per-process memory limit in kilobytes; it does not affect node placement and would not fix a cross-node shared memory problem.
- D) Incorrect — `-P` specifies the project for accounting purposes and has no bearing on node placement or shared memory accessibility.

---

## Q5 — Diagnosing No Speedup: Cores vs Threads

```bash
#!/bin/bash
#BSUB -J matmuls_8core
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:10
#BSUB -o matmuls_%J.out
#BSUB -e matmuls_%J.err

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

python matmul.py
```

`matmul.py` runs 100 × `np.matmul` on 1000×1000 matrices in a Python for loop. The user expects ~8x speedup over the 1-core run (~5.87 s) but measures ~5.96 s. What is the cause?

**A)** The `XeonE5_2650v4` model does not support multi-threading

**B)** NumPy does not automatically use the 8 allocated cores; no thread-count env vars were exported

**C)** `span[hosts=1]` prevents NumPy from using more than 1 thread

**D)** The Python for loop is I/O bound, so adding cores has no effect

**Answer: B**

- A) Incorrect — The XeonE5_2650v4 is a multi-core CPU that supports multi-threading; the hardware is not the limitation.
- B) Correct — This reproduces the exact Week 13 Exercise 2.2 result. Allocating 8 cores with LSF does not instruct NumPy to use them. Without `export OMP_NUM_THREADS=8` (and MKL/OpenBLAS equivalents), NumPy's BLAS defaults to 1 thread, giving essentially the same run time as the 1-core job.
- C) Incorrect — `span[hosts=1]` constrains where cores are placed, not how many threads NumPy uses.
- D) Incorrect — `np.matmul` on 1000×1000 matrices is compute-bound and memory-bandwidth-bound, not I/O-bound. Adding threads does help — but only when the thread vars are exported.

---

## Q6 — Buffered vs Unbuffered Python Output

```bash
python printlots.py > /work3/02613/dump/out_${LSB_JOBID}.txt
```

versus

```bash
python -u printlots.py > /work3/02613/dump/out_${LSB_JOBID}.txt
```

`printlots.py` uses a tight loop with `print()`. Why does the second command produce more accurate timing measurements in the job's resource summary?

**A)** `-u` enables unicode, which is required for LSF to parse the output file

**B)** Without `-u`, Python buffers stdout in memory and flushes in large batches; the write pattern is less predictable and can inflate CPU time measurements, while `-u` writes each line immediately

**C)** `-u` compresses the output file, reducing the total I/O time measured

**D)** The first command redirects stderr as well, but the second only redirects stdout, isolating the measurement

**Answer: B**

- A) Incorrect — `-u` stands for "unbuffered," not "unicode."
- B) Correct — Python's default I/O buffering collects output in an 8 KB buffer before writing. When measuring I/O performance (specifically the pitfall of routing output through LSF channels), buffering changes the write pattern and can produce misleading timings. `-u` forces each `print()` call to write through immediately, making the measurement reflect real per-write overhead. This is why the Week 13 exercise specifies `-u` explicitly.
- C) Incorrect — `-u` has no compression functionality.
- D) Incorrect — Both commands redirect only stdout (`>`); neither redirects stderr. The `2>` operator would be needed for stderr.

---

## Q7 — Memory Request Calculation

A job creates the following NumPy arrays:

```python
A = np.random.rand(100, 1000, 1000)   # float64
B = np.random.rand(100, 1000, 1000)   # float64
C = np.empty((100, 1000, 1000))       # float64
```

Which `#BSUB` memory request is the smallest that safely covers these arrays plus ~1 GB Python runtime overhead?

**A)** `#BSUB -R "rusage[mem=512MB]"`

**B)** `#BSUB -R "rusage[mem=1GB]"`

**C)** `#BSUB -R "rusage[mem=4GB]"`

**D)** `#BSUB -R "rusage[mem=40GB]"`

**Answer: C**

- A) Incorrect — 512 MB is less than a single array (100 × 1000 × 1000 × 8 bytes = 800 MB); LSF would kill the job immediately when A is allocated.
- B) Incorrect — 1 GB covers slightly more than one array but not all three (3 × 800 MB = 2.4 GB) plus Python overhead.
- C) Correct — Each array is ~800 MB; three arrays total ~2.4 GB. Adding ~1 GB for the Python interpreter, NumPy internals, and temporary BLAS buffers gives ~3.4 GB. 4 GB is the standard choice used in the Week 13 `matrix_multiplication_job.sh`. Requesting less risks OOM termination.
- D) Incorrect — 40 GB is approximately 17x more than needed. Over-requesting memory wastes cluster resources, reduces your job's queue priority, and may cause your job to wait longer for a node with enough free memory.

---

## Q8 — Subshell Export Trap

```bash
#!/bin/bash
#BSUB -J matmuls
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:10
#BSUB -o matmuls_%J.out
#BSUB -e matmuls_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

(
  export OMP_NUM_THREADS=8
  export MKL_NUM_THREADS=8
  export OPENBLAS_NUM_THREADS=8
)

python matmul.py
```

The thread variables are exported inside parentheses but NumPy still uses 1 thread. Why?

**A)** Parentheses around `export` statements prevent them from taking effect

**B)** The exports happen inside a subshell; the parent shell (where `python matmul.py` runs) never receives these variables

**C)** Thread variables must be exported using `env` instead of `export`

**D)** The exports must come after `conda activate` to override conda's default thread settings

**Answer: B**

- A) Incorrect — Parentheses DO allow `export` to work within the subshell and its children. The issue is not that `export` fails, but that its effect is confined to the subshell's scope.
- B) Correct — `(...)` in bash spawns a subshell (a child process). Environment changes in a child never propagate back to the parent. After the subshell exits, the parent shell's environment is unchanged. `python matmul.py` runs in the parent shell and inherits its unexported environment, so `OMP_NUM_THREADS` is absent.
- C) Incorrect — `env VAR=val command` is an alternative syntax for inline environment passing, but `export` in the same shell scope works correctly — the problem is purely the subshell boundary.
- D) Incorrect — Variable export order relative to conda activation does not matter; conda does not reset user-defined thread env vars.

---

## Q9 — Fixing Oversubscription by Setting OMP to 1

```python
from multiprocessing.pool import ThreadPool
import numpy as np

def matmuls(A, B):
    n = A.shape[0]
    with ThreadPool(8) as p:
        C = np.concatenate(p.starmap(np.matmul, zip(A, B)))
    return C
```

The job script currently exports `OMP_NUM_THREADS=8`. Run time is ~1.87 s. A colleague suggests setting `OMP_NUM_THREADS=1` (and similarly for MKL/OpenBLAS). What run time should be expected after this change, and why?

**A)** ~5.87 s — disabling BLAS threading removes all parallelism

**B)** ~1.33 s — total active threads drop from 64 to 8 (one per pool thread), eliminating oversubscription

**C)** ~0.17 s — the pool alone achieves near-linear 8x speedup once BLAS overhead is removed

**D)** ~1.87 s — the run time is determined by the pool size, not by BLAS thread settings

**Answer: B**

- A) Incorrect — Setting `OMP_NUM_THREADS=1` disables BLAS's internal parallelism, but the `ThreadPool(8)` still provides 8-way parallelism across the 100 matrix multiplications. The computation is still parallel.
- B) Correct — With `OMP_NUM_THREADS=1`, each pool thread executes `np.matmul` using 1 BLAS thread. The 8 pool threads collectively use exactly 8 threads on 8 cores — no oversubscription. The Week 13 exercise measured ~1.33 s for this configuration.
- C) Incorrect — Linear 8x speedup from 5.87 s would be ~0.73 s. Real scaling is limited by Amdahl's law, memory bandwidth, and thread coordination overhead.
- D) Incorrect — Pool size determines coarse-grained parallelism; BLAS thread settings determine fine-grained parallelism within each pool call. Both interact to determine total thread count and thus oversubscription.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

---

## Q10 — Login Node Computation

```bash
# On the login node, user types:
$ conda activate 02613
$ python matmul.py
```

`matmul.py` runs 100 × `np.matmul` on 1000×1000 matrices and prints the time. What is wrong with this workflow?

**A)** `conda activate` cannot be run on login nodes; you must use `source activate` instead

**B)** Heavy computation is being run directly on the login node; this should be submitted as a batch job with `bsub`

**C)** The script will fail because numpy is not installed in the default conda environment

**D)** Login nodes do not have Python available; you must use the module system

**Answer: B**

- A) Incorrect — `conda activate` works on login nodes; the issue is not the conda command.
- B) Correct — Login nodes are shared infrastructure for file management, editing, and job submission only. Running multi-second or multi-minute computations on login nodes consumes shared CPU resources that other users depend on, and may get your session killed by administrators. The correct approach is `bsub < submit.sh`.
- C) Incorrect — The course conda environment (`02613`) is specifically set up with numpy and all required packages.
- D) Incorrect — Login nodes have Python available; the problem is the inappropriate use of login nodes for computation, not a software availability issue.

---

## Q11 — Multiple Pitfalls in One Script

```bash
#!/bin/bash
#BSUB -J bad_job
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=512MB]"
#BSUB -W 00:10
#BSUB -o logs/output.out
#BSUB -e logs/error.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

OMP_NUM_THREADS=8
MKL_NUM_THREADS=8

python -u heavy_print.py
```

`heavy_print.py` creates `A = np.random.rand(100, 1000, 1000)` and `B = np.random.rand(100, 1000, 1000)`, multiplies them, and prints 50,000 lines of output. Identify ALL pitfalls in this script.

**A)** Thread vars not exported; output goes through LSF channel (no shell redirect)

**B)** Thread vars not exported; output goes through LSF channel; memory request too low; missing `span[hosts=1]`

**C)** Missing `span[hosts=1]`; output goes through LSF channel; wrong queue

**D)** Memory request too low; wrong queue; missing `-u` flag

**Answer: B**

- A) Partially correct but incomplete — the export and I/O pitfalls are identified, but memory and span issues are also present.
- B) Correct — Four pitfalls: (1) `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are assigned without `export`, so Python sees neither. (2) 50,000 print lines go through LSF's `-o` channel with no shell redirect — potentially adding significant run time overhead. (3) `rusage[mem=512MB]` is far below the ~2.4 GB needed for two 100×1000×1000 float64 arrays; LSF will kill the job. (4) Without `#BSUB -R "span[hosts=1]"`, the 8 cores may be on different nodes, breaking any shared memory or efficient thread communication.
- C) Incorrect — The queue (`hpc`) is correct for CPU jobs; the pitfall is not the queue choice but the missing span, I/O routing, and export issues.
- D) Incorrect — `-u` IS present on the Python invocation. Memory is indeed too low, but the queue is appropriate and `-u` is not missing.

---

## Q12 — Correct ThreadPool + NumPy Setup

```bash
#!/bin/bash
#BSUB -J matmuls_correct
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 00:10
#BSUB -o matmuls_%J.out
#BSUB -e matmuls_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MPI_NUM_THREADS=8

python matmul.py
```

`matmul.py` runs a serial for loop over 100 `np.matmul` calls. Which statement is true about this script?

**A)** This script is buggy — exporting `OMP_NUM_THREADS=8` with a serial for loop causes oversubscription

**B)** This script correctly enables 8-thread NumPy BLAS for each `np.matmul` call and should achieve ~4.6x speedup over the 1-thread baseline

**C)** The script needs a `ThreadPool(8)` added to `matmul.py` to actually use the 8 cores

**D)** `MPI_NUM_THREADS` is invalid and will cause the script to fail at runtime

**Answer: B**

- A) Incorrect — A serial for loop calling `np.matmul` with `OMP_NUM_THREADS=8` runs one `np.matmul` at a time, each using 8 BLAS threads. There is no oversubscription since only one matmul runs at a time and it uses exactly 8 threads on 8 cores.
- B) Correct — This matches the corrected `matrix_multiplication_job.sh` from Week 13. The exported thread vars tell NumPy's BLAS backend to use 8 threads per call. With a serial loop, each call gets 8 threads and total active threads never exceed 8. Expected speedup: ~1.28 s vs ~5.87 s baseline ≈ 4.6x.
- C) Incorrect — A ThreadPool is not required; letting BLAS handle parallelism internally is equally effective and avoids the complexity of managing a pool.
- D) Incorrect — `MPI_NUM_THREADS` is a valid environment variable used by some MPI-integrated BLAS builds. Exporting it is harmless even if the particular BLAS backend doesn't read it; it will not cause a script failure.

---

## Q13 — Identifying the Fastest I/O Approach

Three job scripts submit the same print-heavy Python program:

- **Script A:** Uses `#BSUB -o` only; no shell redirect
- **Script B:** Uses `#BSUB -o` AND shell redirect: `python -u script.py > /work3/output_${LSB_JOBID}.txt`
- **Script C:** Uses shell redirect only (`#BSUB -o` is omitted): `python -u script.py > /work3/output_${LSB_JOBID}.txt`

Rank these from fastest to slowest for a program printing 100,000 lines.

**A)** A = B = C — all approaches write to the same filesystem

**B)** B and C are equally fast, and both are faster than A

**C)** A is fastest because LSF optimises its own I/O channel

**D)** B is fastest, C is second, A is slowest

**Answer: B**

- A) Incorrect — The path the output travels matters enormously. LSF's own I/O channel adds significant per-write overhead; direct file descriptors do not.
- B) Correct — Both Script B and Script C bypass LSF's I/O channel for the actual program output by connecting stdout directly to a file. The presence or absence of `#BSUB -o` in Script C does not affect the shell redirect performance. Both achieve ~3 s in the Week 13 scenario. Script A routes all output through LSF's channel at ~80 s.
- C) Incorrect — LSF's I/O channel is slower for high-volume output, not optimised. This is the core Week 13 pitfall.
- D) Incorrect — There is no reason for Script B to be faster than Script C; the shell redirect mechanism is identical in both.

---

## Q14 — ProcessPool vs ThreadPool for NumPy

```python
from multiprocessing.pool import Pool
import numpy as np

def single_matmul(args):
    a, b = args
    return np.matmul(a, b)

def matmuls(A, B):
    n = A.shape[0]
    with Pool(8) as p:
        C = np.concatenate(p.map(single_matmul, zip(A, B)))
    return C
```

A colleague proposes this `Pool`-based (process pool) version instead of `ThreadPool`. What is the main performance concern compared to a `ThreadPool` solution?

**A)** `Pool` (process pool) cannot use NumPy because each worker gets a separate Python interpreter

**B)** `Pool` (process pool) requires serializing (pickling) each 1000×1000 float64 array for inter-process communication, adding significant data transfer overhead not present with ThreadPool

**C)** `Pool` (process pool) workers cannot run on HPC clusters because they require root privileges

**D)** `Pool` (process pool) ignores `OMP_NUM_THREADS` so BLAS always uses 1 thread in each worker

**Answer: B**

- A) Incorrect — Process pool workers do have separate Python interpreters, but they can absolutely use NumPy. Each worker imports and uses NumPy independently.
- B) Correct — Process pool workers are separate OS processes. Data passed to them must be serialized (pickled) and sent through an IPC pipe. Each 1000×1000 float64 array is 8 MB; serializing and transmitting 100 such arrays adds substantial overhead compared to `ThreadPool`, where threads share the same memory address space and array slices are just views — no copying required.
- C) Incorrect — `multiprocessing.pool.Pool` is a standard Python library feature with no special privilege requirements; it works on HPC clusters without issue.
- D) Incorrect — Process pool workers inherit the parent's exported environment, including `OMP_NUM_THREADS`. BLAS in each worker respects the thread count directive.

---

## Q15 — MKL_NUM_THREADS Alone Is Not Enough

```bash
export MKL_NUM_THREADS=8
python matmul.py
```

The system running this script uses NumPy compiled against OpenBLAS (not Intel MKL). What is the likely outcome?

**A)** NumPy uses 8 threads because OpenBLAS reads `MKL_NUM_THREADS` as a fallback

**B)** NumPy uses 1 thread because OpenBLAS reads `OPENBLAS_NUM_THREADS` (or `OMP_NUM_THREADS`), not `MKL_NUM_THREADS`

**C)** The script fails with an error because `MKL_NUM_THREADS` is only valid on Intel hardware

**D)** NumPy automatically detects the OpenBLAS backend and converts `MKL_NUM_THREADS` to the correct variable

**Answer: B**

- A) Incorrect — OpenBLAS does not read `MKL_NUM_THREADS` as a fallback; each library reads only its own designated environment variable.
- B) Correct — OpenBLAS checks `OPENBLAS_NUM_THREADS` first, then `OMP_NUM_THREADS`. `MKL_NUM_THREADS` is Intel MKL's proprietary variable. Since this system uses OpenBLAS, `MKL_NUM_THREADS=8` is simply ignored, and OpenBLAS defaults to 1 (or to the hardware thread count depending on version). This is why the Week 13 solution exports ALL three variables: you don't always know which BLAS backend is active.
- C) Incorrect — Setting `MKL_NUM_THREADS` on a non-MKL system is harmless; it is simply an unused environment variable.
- D) Incorrect — NumPy has no conversion logic between BLAS thread variables; each library reads its own specific variable independently.

---

## Q16 — Output File Location Pitfall

```bash
#!/bin/bash
#BSUB -J print
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -R "rusage[mem=512MB]"
#BSUB -o /home/s252786/logs/printlots_%J.out
#BSUB -e /home/s252786/logs/printlots_%J.err

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

python -u printlots.py \
    1> /home/s252786/logs/output_${LSB_JOBID}.txt \
    2> /home/s252786/logs/error_${LSB_JOBID}.txt
```

The script writes output to the user's home directory `/home/s252786/`. What is the pitfall?

**A)** The `-o` and `-e` flags conflict with shell redirection, causing the program to hang

**B)** Home directories on DTU HPC clusters have strict quotas and are not designed for large output; output files for jobs should be written to `/work3/` scratch space

**C)** `LSB_JOBID` is not defined when using shell redirection; only `%J` works in that context

**D)** Using `/home/` paths with `-u` flag disables buffering on the output file

**Answer: B**

- A) Incorrect — `#BSUB -o` and shell redirection coexist without conflict; the LSF output file will simply be empty or contain only LSF resource summary lines.
- B) Correct — Home directories on DTU HPC are NFS-mounted with small quotas (often a few GB) and are not intended for job output. Large output files should go to `/work3/02613/` (or similar scratch space), which is specifically provisioned for high-volume HPC output. The Week 13 exercise solution explicitly uses `/work3/02613/dump/` for output files.
- C) Incorrect — `$LSB_JOBID` is a valid LSF environment variable set when the job runs; it works in shell redirects. `%J` is LSF's own substitution token for `#BSUB` lines only.
- D) Incorrect — `-u` controls Python's buffer behaviour and has no interaction with the output file path.

---

## Q17 — Predicting Run Time from Thread Count

Based on the Week 13 exercise, match each configuration to its approximate run time for 100 × `np.matmul` on 1000×1000 matrices using 8 cores:

| Configuration | Run time |
|---|---|
| 1 core, no thread vars | ? |
| 8 cores, no thread vars exported | ? |
| 8 cores, `OMP_NUM_THREADS=8` exported | ? |
| 8 cores, `ThreadPool(8)` + `OMP_NUM_THREADS=8` | ? |
| 8 cores, `ThreadPool(8)` + `OMP_NUM_THREADS=1` | ? |

Which answer correctly matches all five configurations?

**A)** 5.87 s, 2.93 s, 1.28 s, 0.64 s, 1.33 s

**B)** 5.87 s, 5.96 s, 1.28 s, 1.87 s, 1.33 s

**C)** 5.87 s, 5.96 s, 1.28 s, 1.28 s, 1.87 s

**D)** 5.87 s, 5.87 s, 2.93 s, 1.87 s, 1.28 s

**Answer: B**

- A) Incorrect — 8 cores with no thread vars does not halve the run time (no env vars = 1 BLAS thread regardless of core count). And `ThreadPool(8)` + `OMP_NUM_THREADS=8` is the oversubscribed case, not the fastest.
- B) Correct — Directly from the Week 13 exercise: 1 core baseline = 5.87 s; 8 cores no vars = 5.96 s (no change); 8-thread NumPy = 1.28 s; ThreadPool(8) + 8-thread NumPy = 1.87 s (oversubscribed, slowest of the parallel options); ThreadPool(8) + 1-thread NumPy = 1.33 s (correct fix).
- C) Incorrect — The doubly-parallel case (1.87 s) and the fixed ThreadPool case (1.33 s) are swapped.
- D) Incorrect — This implies no benefit from exporting thread vars with 8 cores, and shows 2.93 s for the 8-thread case, neither of which matches the exercise data.

---

## Q18 — Fixing All Pitfalls in a Combined Script

The following script has multiple pitfalls. Rewrite it mentally and identify the corrected version:

```bash
#!/bin/bash
#BSUB -J broken
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 00:10
#BSUB -o broken_%J.out

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
OPENBLAS_NUM_THREADS=8

python matmul_and_print.py
```

`matmul_and_print.py` creates two 100×1000×1000 float64 arrays, multiplies them in a for loop, and prints 80,000 lines. Which corrected script fixes ALL pitfalls?

**A)**
```bash
#BSUB -n 8
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"
export OMP_NUM_THREADS=8
python matmul_and_print.py
```

**B)**
```bash
#BSUB -n 8
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
python -u matmul_and_print.py 1> /work3/02613/out_${LSB_JOBID}.txt 2> /work3/02613/err_${LSB_JOBID}.txt
```

**C)**
```bash
#BSUB -n 8
#BSUB -R "rusage[mem=4GB]"
export OMP_NUM_THREADS=8
python -u matmul_and_print.py > /work3/02613/out_${LSB_JOBID}.txt
```

**D)**
```bash
#BSUB -n 8
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
python -u matmul_and_print.py > /work3/02613/out_${LSB_JOBID}.txt
```

**Answer: B**

- A) Incorrect — Only exports `OMP_NUM_THREADS` (missing `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS`), memory is still only 1 GB (too low for two 800 MB arrays), and high-volume print output still goes through LSF's channel.
- B) Correct — Fixes all four pitfalls from the original: (1) `export` on all three thread vars so Python inherits them; (2) `rusage[mem=4GB]` to safely cover two 800 MB arrays plus Python overhead; (3) `span[hosts=1]` to keep all cores on one node; (4) shell redirection to `/work3/` scratch space with `-u` to bypass LSF's I/O channel for the 80,000 print statements.
- C) Incorrect — Exports only `OMP_NUM_THREADS` (not MKL or OpenBLAS), and is missing `span[hosts=1]`. Memory is correct and redirection is correct, but thread coverage and node placement are incomplete.
- D) Incorrect — `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are assigned without `export`, so Python still cannot see them — the core pitfall from the original script is unchanged.

---

## Set 3 — Extended Practice

---

## Q19 — rusage Per-Core Trap

```bash
#!/bin/bash
#BSUB -J simulate
#BSUB -q hpc
#BSUB -W 10:00
#BSUB -R "rusage[mem=100GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o sim_%J.out
#BSUB -e sim_%J.err

python simulate.py initconds.npy
```

The `simulate.py` script needs at least 100 GB of RAM in total. How much total memory does this job script actually reserve, and is it correct?

**A)** 100 GB total — the script is correct

**B)** 25 GB total — the script under-requests by 4×, and the job will be OOM-killed

**C)** 400 GB total — the script over-requests by 4× because `rusage[mem=X]` is per core

**D)** 400 GB total — the script is correct because LSF divides the memory evenly among cores

**Answer: C**

- A) Incorrect — `rusage[mem=100GB]` is not a total-job specification; it is a per-slot (per-core) specification on LSF. With 4 cores, LSF reserves 4 × 100 GB = 400 GB.
- B) Incorrect — The memory reservation is too large (400 GB), not too small. The program would not be OOM-killed; instead, the job would wait a very long time in the queue while LSF searches for a node with 400 GB free.
- C) Correct — This is the exact scenario from the F25 exam Q1. On DTU's LSF cluster, `rusage[mem=X]` is a per-core specification. Four cores × 100 GB = 400 GB reserved. The correct directive to reserve exactly 100 GB total across 4 cores is `rusage[mem=25GB]` (100 GB / 4 = 25 GB per core).
- D) Incorrect — LSF does not divide the requested memory among cores; it multiplies it. Each core is independently allocated the requested `rusage` amount.

---

## Q20 — Pure Python ThreadPool Speedup Prediction

```python
from multiprocessing.pool import ThreadPool
from time import perf_counter

def sum_range(n):
    s = 0
    for i in range(n):
        s += i
    return s

numbers = list(range(1, 100_001))  # [1, 2, ..., 100000]

t0 = perf_counter()
with ThreadPool(8) as p:
    results = p.map(sum_range, numbers)
t1 = perf_counter()
print(f"{t1 - t0:.2f}s")
```

This job runs on a node with 8 cores. Compared to a serial `for` loop calling `sum_range` over all 100,000 numbers, what speedup does the `ThreadPool(8)` version achieve?

**A)** Approximately 8× — each thread handles 12,500 numbers independently

**B)** Approximately 1× or less — `sum_range` is pure Python and holds the GIL, so all 8 threads serialize at the interpreter level

**C)** Approximately 4× — the GIL allows two threads to overlap at a time

**D)** Approximately 64× — ThreadPool combines 8 threads with Python's internal BLAS optimisation

**Answer: B**

- A) Incorrect — `sum_range` is pure Python bytecode (a for loop with integer addition). The GIL means only one thread executes Python bytecode at any instant. With 8 threads all trying to run Python, the effective execution is largely serial plus the overhead of thread creation, context switching, and GIL acquisition/release. Real performance is typically equal to or worse than single-threaded.
- B) Correct — `sum_range` never releases the GIL: it only does Python integer arithmetic and loop iteration, both of which hold the GIL throughout. A `ThreadPool` therefore provides no meaningful parallelism here. For such workloads, `multiprocessing.pool.Pool` (process pool) is needed to bypass the GIL entirely.
- C) Incorrect — The GIL is a mutual exclusion lock; at most one thread holds it and executes Python bytecode at any given time. There is no "two threads overlap" mode for pure Python.
- D) Incorrect — `sum_range` does not use NumPy or BLAS at all. There is no BLAS optimisation involved.

---

## Q21 — time.time vs time.perf_counter Output

```python
import time

A_time   = time.time()
B_time   = time.perf_counter()
C_time   = time.process_time()
D_time   = time.monotonic()

# ... run a 2-second computation ...

print(time.time()         - A_time)   # Line A
print(time.perf_counter() - B_time)   # Line B
print(time.process_time() - C_time)   # Line C
print(time.monotonic()    - D_time)   # Line D
```

The computation runs for approximately 2 seconds of wall time and uses 100% of one CPU core throughout. Which line is most suitable for use in the Week 13 HPC benchmarks, and approximately what value does it print?

**A)** Line A — `time.time()` is the most accurate; prints ~2.0

**B)** Line B — `time.perf_counter()` is the highest-resolution monotonic clock for benchmarking; prints ~2.0

**C)** Line C — `time.process_time()` measures true CPU usage; prints ~2.0, and is the right choice for HPC

**D)** Line D — `time.monotonic()` is the only clock suitable for multi-threaded work; prints ~2.0

**Answer: B**

- A) Incorrect — `time.time()` returns the system clock which can be adjusted by NTP, potentially jumping backward. While it would print ~2.0 in this case, it is not the recommended benchmarking tool. The Week 13 `matmul.py` specifically imports `perf_counter`, not `time`.
- B) Correct — `time.perf_counter()` is the Python standard for benchmarking. It is monotonic, has the highest platform resolution (often nanoseconds), and cannot be adjusted by the system clock. For a 2-second single-threaded computation it prints ~2.0. This is exactly what `from time import perf_counter as time` in `matmul.py` uses.
- C) Incorrect — `time.process_time()` excludes time when the process is not scheduled (e.g., when the OS runs other processes). For a 2-second wall-time computation it would also print ~2.0 if the core was 100% busy, but it would underreport if there was any scheduling preemption or sleep. More importantly, it is not the right tool for measuring what users actually experience (wall-clock latency).
- D) Incorrect — `time.monotonic()` is suitable for measuring elapsed time and cannot go backward, but it has lower resolution than `perf_counter` on many platforms. It is not the standard choice for HPC benchmarks. It would also print ~2.0 here but is not preferred.

---

## Q22 — ProcessPool with Large Arrays

```python
from multiprocessing.pool import Pool
import numpy as np
from time import perf_counter

def multiply(args):
    a, b = args
    return np.matmul(a, b)

A = np.random.rand(100, 1000, 1000)   # 100 matrices, each 8 MB
B = np.random.rand(100, 1000, 1000)

t0 = perf_counter()
with Pool(8) as p:
    results = p.map(multiply, zip(A, B))
C = np.concatenate(results)
t1 = perf_counter()
print(t1 - t0)
```

This uses `Pool` (process pool) instead of `ThreadPool`. What is the primary performance concern compared to the `ThreadPool` version?

**A)** `Pool` cannot concatenate results because each worker returns a different array type

**B)** Each 8 MB matrix slice must be pickled and sent through an IPC pipe to a worker process, and the result must be pickled back — adding ~1.6 GB of serialisation overhead for 100 pairs

**C)** `Pool` workers do not inherit `OMP_NUM_THREADS`, so BLAS uses all available cores in each worker, causing oversubscription

**D)** `Pool(8)` on an 8-core node always produces the same run time as `ThreadPool(8)` because both use 8 parallel execution units

**Answer: B**

- A) Incorrect — `Pool` workers can return any picklable object, including NumPy arrays. `np.concatenate` works on a list of arrays regardless of how they were produced.
- B) Correct — Each 1000×1000 float64 matrix is 8 MB. `Pool` must pickle the input arrays (A[i] and B[i]) in the parent and send them through an OS pipe to the worker, then pickle the 8 MB result back. For 100 pairs: 100 × 2 × 8 MB in + 100 × 8 MB out = ~2.4 GB of data through IPC. `ThreadPool` workers share the same address space, so `A[i]` is simply a pointer view with zero copying. This pickle overhead is why `ThreadPool` is strongly preferred for NumPy parallelism.
- C) Incorrect — `Pool` workers inherit the parent's exported environment, including `OMP_NUM_THREADS`. BLAS threading is controlled correctly in each worker.
- D) Incorrect — `Pool` and `ThreadPool` have fundamentally different communication mechanisms. `Pool` has IPC serialisation overhead; `ThreadPool` does not. Their run times differ significantly for large-array workloads.

---

## Q23 — What os.environ.get Sees When export Is Missing

```bash
#!/bin/bash
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"

OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
OPENBLAS_NUM_THREADS=8

python - <<'EOF'
import os
print("OMP:", os.environ.get("OMP_NUM_THREADS", "NOT SET"))
print("MKL:", os.environ.get("MKL_NUM_THREADS", "NOT SET"))
print("OBL:", os.environ.get("OPENBLAS_NUM_THREADS", "NOT SET"))
EOF
```

What does this script print?

**A)**
```
OMP: 8
MKL: 8
OBL: 8
```

**B)**
```
OMP: NOT SET
MKL: NOT SET
OBL: NOT SET
```

**C)**
```
OMP: 8
MKL: NOT SET
OBL: NOT SET
```

**D)**
```
OMP: NOT SET
MKL: 8
OBL: NOT SET
```

**Answer: B**

- A) Incorrect — The variables are assigned in the shell but not exported. Python's `os.environ` reflects the process environment inherited at launch, which only includes *exported* variables. Without `export`, none of the three variables reach the Python process.
- B) Correct — `OMP_NUM_THREADS=8` (without `export`) sets a local shell variable. When the shell forks to run Python (even via the heredoc `python - <<'EOF'`), the child inherits only the exported environment. Since none of the thread variables were exported, `os.environ.get(...)` returns the default `"NOT SET"` for all three. This directly demonstrates the Week 13 core pitfall.
- C) Incorrect — There is no mechanism by which `OMP_NUM_THREADS` would be inherited but not `MKL_NUM_THREADS`. All three were assigned identically (no export), so all three are equally invisible to Python.
- D) Incorrect — Same reasoning as C: all three variables were set without `export` in the same fashion; none are inherited.

---

## Q24 — Wall Time Kill: What Output Remains

```bash
#!/bin/bash
#BSUB -J crunch
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:02
#BSUB -o crunch_%J.out
#BSUB -e crunch_%J.err

python -u long_job.py
```

`long_job.py` runs a loop that writes progress lines and takes 5 minutes to complete:

```python
import time
for i in range(300):        # one iteration per second
    time.sleep(1)
    print(f"Step {i+1}/300 done")
```

After the job finishes (or is terminated), what does `crunch_%J.out` contain?

**A)** All 300 lines — LSF buffers output and writes it all at job end regardless of the `-W` limit

**B)** Nothing — LSF discards buffered output when a job is killed

**C)** Lines for steps 1 through approximately 120, followed by the LSF resource summary showing the job was killed

**D)** Lines for steps 1 through approximately 120; the file ends mid-job because LSF killed the process at the 2-minute wall time limit

**Answer: D**

- A) Incorrect — The job is killed at the 2-minute wall time limit. Only the output produced before the kill signal reaches the output file. There is no post-kill buffering.
- B) Incorrect — Output already written to the file before the kill signal is preserved. LSF does not retroactively erase completed output lines.
- C) Incorrect — The LSF resource summary appears at the end of the `-o` file when a job *completes normally*. A job killed by the wall time limit may receive a brief LSF notice, but not a full resource summary in the same format as a successful completion.
- D) Correct — The job runs for 2 minutes (120 seconds), printing one line per second. At the 2-minute mark, LSF sends SIGKILL. Because `-u` (unbuffered output) is used, each `print()` was flushed immediately to the LSF `-o` channel. Approximately 120 lines are preserved in the output file. The file then ends abruptly — no "Step 121" onward, and no clean resource summary.

---

## Q25 — Counting Active Threads from Config Variables

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

python - <<'EOF'
import numpy as np
from multiprocessing.pool import ThreadPool

A = np.random.rand(50, 500, 500)
B = np.random.rand(50, 500, 500)

with ThreadPool(4) as p:
    C = np.concatenate(p.starmap(np.matmul, zip(A, B)))
EOF
```

This job runs on a node with 8 allocated cores (`#BSUB -n 8`). How many active threads are running at peak during the `ThreadPool` execution, and is the job oversubscribed relative to allocated cores?

**A)** 4 threads total — the ThreadPool uses 4 workers and each runs single-threaded NumPy

**B)** 8 threads total — 4 pool threads + 4 BLAS threads shared across the pool; matches the 8 allocated cores exactly

**C)** 16 threads total — each of the 4 pool threads spawns 4 BLAS threads; 16 > 8 cores, so the job is oversubscribed 2×

**D)** 4 threads total — NumPy detects the ThreadPool context and automatically limits BLAS to 1 thread per call

**Answer: C**

- A) Incorrect — `OMP_NUM_THREADS=4` means *each* BLAS call spawns 4 threads internally. With 4 pool workers each making a `np.matmul` call simultaneously, the total is 4 × 4 = 16 threads.
- B) Incorrect — BLAS threads are not shared across pool workers; each pool worker independently spawns its own set of BLAS threads. 4 pool threads × 4 BLAS threads each = 16 total, not 8.
- C) Correct — This is the oversubscription pattern from Week 13 at a smaller scale. Each of the 4 `ThreadPool` workers calls `np.matmul`, which spawns `OMP_NUM_THREADS=4` BLAS threads internally. At peak, 4 × 4 = 16 threads are active on 8 cores — 2× oversubscribed. The fix is to set `OMP_NUM_THREADS=1` and keep `ThreadPool(8)`, or use `ThreadPool(1)` and `OMP_NUM_THREADS=8`, or use `ThreadPool(4)` with `OMP_NUM_THREADS=2` (4 × 2 = 8).
- D) Incorrect — NumPy has no detection logic for external ThreadPool contexts. It always spawns the number of BLAS threads specified by the environment variable, regardless of whether a ThreadPool is active.

---

## Q26 — OMP=0 Undefined Behaviour vs OMP=1

```bash
# Script A
export OMP_NUM_THREADS=0
python matmul.py

# Script B
export OMP_NUM_THREADS=1
python matmul.py
```

A developer wants to ensure NumPy runs single-threaded. They are deciding between `OMP_NUM_THREADS=0` and `OMP_NUM_THREADS=1`. Which is correct, and what does `OMP_NUM_THREADS=0` actually do?

**A)** Both are equivalent — setting `OMP_NUM_THREADS` to 0 or 1 both produce single-threaded NumPy execution

**B)** `OMP_NUM_THREADS=1` is correct and produces single-threaded execution; `OMP_NUM_THREADS=0` is undefined behaviour and may cause a crash, run with the default thread count, or behave unpredictably depending on the OpenMP implementation

**C)** `OMP_NUM_THREADS=0` disables OpenMP entirely and is the recommended way to force single-threaded execution

**D)** `OMP_NUM_THREADS=0` means "use all available cores" in OpenMP's convention

**Answer: B**

- A) Incorrect — `0` and `1` have different semantics. `OMP_NUM_THREADS=1` explicitly requests 1 thread and is well-defined. `OMP_NUM_THREADS=0` is outside the valid positive-integer range; its behaviour is implementation-defined.
- B) Correct — The OpenMP specification defines `OMP_NUM_THREADS` as a positive integer. Setting it to `0` is technically undefined behaviour. In practice, different OpenMP runtimes handle it differently: some fall back to the default (hardware thread count), some crash, and some silently treat it as 1. For reliable single-threaded execution, `OMP_NUM_THREADS=1` is the correct and portable choice.
- C) Incorrect — There is no standard OpenMP or BLAS convention where `0` means "disable threading." The safe way to force single-threaded execution is `OMP_NUM_THREADS=1`.
- D) Incorrect — "Use all available cores" is not an OpenMP convention for any value of `OMP_NUM_THREADS`. The default thread count (when the variable is unset or set to `0`) varies by implementation; it is often the hardware core count, but this is not guaranteed.

---

## Q27 — LSB_JOBID in Shell Redirect vs %J in BSUB

```bash
#!/bin/bash
#BSUB -J myrun
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=512MB]"
#BSUB -W 00:10
#BSUB -o output_%J.out

python -u script.py > output_${LSB_JOBID}.txt
```

The job gets ID 12345. What are the names of the two output files created, and what does each contain?

**A)** Only `output_12345.out` is created; the shell redirect overwrites it

**B)** `output_12345.out` (LSF's output file, contains the LSF resource summary only) and `output_12345.txt` (shell-redirected stdout from `script.py`)

**C)** `output_%J.out` (a file literally named with `%J`) and `output_12345.txt`

**D)** `output_12345.out` containing all output from `script.py`, and `output_12345.txt` which is empty

**Answer: B**

- A) Incorrect — The shell redirect (`> output_${LSB_JOBID}.txt`) and the LSF `-o` flag write to *different files*. They do not interfere with each other.
- B) Correct — `%J` is LSF's own substitution token, expanded at job submission time to the job ID (12345), so `-o output_%J.out` creates `output_12345.out`. This file receives the LSF resource summary (and any stdout not redirected by the shell). `$LSB_JOBID` is a shell environment variable set at runtime to the same job ID, so the shell redirect `> output_${LSB_JOBID}.txt` creates `output_12345.txt` and writes `script.py`'s stdout there. The key insight: `%J` is a LSF directive token, `$LSB_JOBID` is a shell variable — both resolve to the same number but in different contexts.
- C) Incorrect — `%J` is expanded by LSF at submission time, not kept as a literal string. The file will be named `output_12345.out`, not `output_%J.out`.
- D) Incorrect — The shell redirect captures stdout from `script.py` into the `.txt` file. The LSF `-o` file receives LSF's own output (resource summary and any stdout not captured by the redirect), not the full script output.

---

## Q28 — Amdahl Calculation from Timing Output

```python
# Serial baseline
serial_time = 12.0   # seconds, 1 core

# Parallel run on 3 cores
parallel_time = 4.8  # seconds, 3 cores

speedup = serial_time / parallel_time
print(f"Speedup: {speedup:.1f}x")

# Amdahl: 1/S(p) = (1 - F) + F/p  →  solve for F
# Then max speedup = 1 / (1 - F)
p = 3
S = speedup
F = p * (1 - 1/S) / (p - 1)
max_speedup = 1 / (1 - F)
print(f"Parallel fraction F: {F:.2f}")
print(f"Theoretical max speedup: {max_speedup:.1f}x")
```

What does this code print?

**A)**
```
Speedup: 2.5x
Parallel fraction F: 0.90
Theoretical max speedup: 10.0x
```

**B)**
```
Speedup: 2.5x
Parallel fraction F: 0.75
Theoretical max speedup: 4.0x
```

**C)**
```
Speedup: 3.0x
Parallel fraction F: 1.00
Theoretical max speedup: inf x
```

**D)**
```
Speedup: 2.5x
Parallel fraction F: 0.80
Theoretical max speedup: 5.0x
```

**Answer: A**

- A) Correct — Step by step: `speedup = 12.0 / 4.8 = 2.5`. Then F = p × (1 - 1/S) / (p - 1) = 3 × (1 - 1/2.5) / (3 - 1) = 3 × (1 - 0.4) / 2 = 3 × 0.6 / 2 = 1.8 / 2 = 0.9. Maximum speedup = 1 / (1 - 0.9) = 1 / 0.1 = 10.0. This matches the F25 exam Q8 Amdahl calculation exactly.
- B) Incorrect — F=0.75 would give S(3) = 1/(0.25 + 0.25) = 2.0×, not 2.5×.
- C) Incorrect — F=1.0 would require `parallel_time = serial_time / p = 4.0s`, giving speedup=3.0. But 12.0/4.8=2.5, not 3.0.
- D) Incorrect — F=0.80 gives S(3) = 1/(0.2 + 0.8/3) = 1/0.467 ≈ 2.14×, not 2.5×.

---

## Q29 — Process Spawn Overhead for Tiny Tasks

> **Week reference:** Week 13

```python
import multiprocessing as mp
import time

def add_one(x):
    return x + 1

start = time.perf_counter()
with mp.Pool(4) as p:
    results = p.map(add_one, range(10))
elapsed = time.perf_counter() - start
```

On a typical laptop, which elapsed time is most realistic, and why?

- A) ~0.0001 s — `Pool.map` overhead for 10 trivial tasks is negligible
- B) ~0.1–0.5 s — process spawning and IPC (pickle/unpickle) dominate; the actual `x + 1` computation is irrelevant
- C) ~4.0 s — each of the 4 worker processes takes ~1 s to spawn
- D) ~10.0 s — each of the 10 tasks takes ~1 s due to GIL contention between workers

**Answer: B**

- A) Incorrect — process-based `Pool` has significant fixed costs: spawning worker processes (fork or spawn), pickling arguments, sending them via IPC pipes, executing, pickling results, and returning them. For `x + 1`, this overhead is orders of magnitude larger than the computation.
- B) Correct — on macOS (spawn start method) or Linux, `Pool` creation takes ~50–500 ms just for worker process startup. Each `p.map` call also serializes and deserializes all inputs and outputs. For 10 trivially cheap tasks, total elapsed time is dominated by this overhead, typically ~0.1–0.5 s. This is the canonical "don't use Pool for tiny tasks" pitfall from Week 13.
- C) Incorrect — process spawn time is typically 10–100 ms per process, not 1 s. Four workers would take ~40–400 ms total, not 4 s.
- D) Incorrect — `multiprocessing.Pool` uses separate processes, not threads. There is no GIL contention between worker processes. Each process has its own GIL.

---

## Q30 — @njit First-Call Timing Trap

> **Week reference:** Week 9

```python
from numba import njit
import numpy as np, time

@njit
def fast_dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

a = np.random.rand(10**6)
b = np.random.rand(10**6)

t = time.perf_counter()
r1 = fast_dot(a, b)
t1 = time.perf_counter() - t

t = time.perf_counter()
r2 = fast_dot(a, b)
t2 = time.perf_counter() - t

print(t1 > t2)
```

**What is printed, and why?**

- A) `False` — the first call is faster because the CPU is not yet thermally throttled
- B) `True` — the first call includes JIT compilation time (typically 0.1–3 s); the second reuses the compiled binary
- C) `False` — JIT compilation is cached from previous runs via `cache=True`, so both calls are equally fast
- D) `True` — the second call is slower because the CPU cache is cold after the first call completes

**Answer: B**

- A) Incorrect — thermal throttling does not cause a significant speed difference between two consecutive calls on the timescale of a few seconds.
- B) Correct — the first call to a `@njit` function triggers JIT compilation. Compilation takes 0.1–3 s depending on function complexity and NumPy imports. `t1` therefore includes compilation time and is much larger than `t2` (pure execution). `t1 > t2` is `True`.
- C) Incorrect — `cache=True` is not set here. Without it, compiled binaries are only held in memory for the current process lifetime. No disk cache is consulted.
- D) Incorrect — after the first call, the array `a` and `b` data is warm in L3/DRAM from the first pass. The second call is faster, not slower, than the first (ignoring compilation time). And even if caches were cold, that effect is tiny compared to compilation overhead.

---

## Q31 — time.time() Resolution for Sub-ms Benchmarks

> **Week reference:** Week 2 / Week 13

```python
import time
import numpy as np

a = np.ones((50, 50))
b = np.ones((50, 50))

times = []
for _ in range(100):
    t0 = time.time()
    c = a @ b
    times.append(time.time() - t0)

print(min(times))
```

**What is the pitfall with using `time.time()` here?**

- A) `time.time()` measures CPU time, not wall time — use `time.process_time()` instead
- B) `time.time()` resolution on some platforms (notably older Windows) can be as coarse as 10–15 ms — for operations taking microseconds, many measurements will read `0.0`; use `time.perf_counter()`
- C) There is no pitfall — `time.time()` has nanosecond resolution on all modern operating systems
- D) The loop must use `timeit.repeat()` instead; `time.time()` is not safe to call in a loop

**Answer: B**

- A) Incorrect — `time.time()` returns wall-clock (real elapsed) time, not CPU time. `time.process_time()` measures CPU time excluding sleep. The issue here is resolution, not the time domain.
- B) Correct — `time.time()` delegates to the OS system clock, which on some platforms (Windows legacy, some embedded systems) has 10–15 ms granularity. A 50×50 matrix multiply takes ~1–10 µs — far below 10 ms — so many measurements appear as `0.0`. `time.perf_counter()` uses the highest-resolution hardware counter available and is the correct choice for sub-millisecond benchmarking.
- C) Incorrect — nanosecond resolution is not guaranteed by `time.time()`. It is only guaranteed by `time.perf_counter()` on Python ≥ 3.3 on supported hardware.
- D) Incorrect — `time.time()` is safe to call in a loop. `timeit.repeat()` is a convenience wrapper that also handles warmup and GC disabling, but it's not the only correct approach.

---

## Q32 — arr.tolist() in a Tight Loop

> **Week reference:** Week 13

```python
import numpy as np

arr = np.arange(10**6, dtype=np.float64)
total = 0.0

for x in arr.tolist():
    total += x

print(total)
```

**What is the HPC pitfall in this code?**

- A) No pitfall — iterating over a Python list is faster than iterating over a NumPy array directly
- B) `arr.tolist()` converts the entire 8 MB array into ~24 MB of Python float objects before the loop starts, then the loop runs at slow Python speed; `np.sum(arr)` is ~100× faster
- C) The loop will silently overflow float64 before reaching 10^6 elements
- D) `arr.tolist()` is not supported for float64 arrays and raises `TypeError`

**Answer: B**

- A) Incorrect — a Python list of Python floats is *slower* than iterating `arr` directly. Iterating a NumPy array also yields Python float objects (one boxed object per element), so the per-element overhead is similar. The real harm of `.tolist()` is the upfront allocation of the entire Python list in memory.
- B) Correct — `arr.tolist()` allocates a Python list containing 10^6 Python float objects. Each Python float is ~24 bytes on CPython 64-bit, totalling ~24 MB (vs 8 MB for the NumPy buffer). The subsequent loop iterates at ~50–100 ns/element in Python. `np.sum(arr)` uses vectorized C code and processes the same array in ~1 ms — roughly 100× faster — without any extra allocation.
- C) Incorrect — float64 range is ±1.8×10^308. The sum 0+1+2+...+(10^6-1) = ~5×10^11, far below float64's maximum. No overflow occurs.
- D) Incorrect — `.tolist()` is supported for all numeric NumPy dtypes including float64. It returns a standard Python list.

---

## Q33 — Cold Cache vs Warm Cache in Profiling

> **Week reference:** Week 2 / Week 13

```python
import cProfile
import numpy as np

arr = np.random.rand(5000, 5000)   # ~200 MB

def row_sums():
    return arr.sum(axis=1)

cProfile.run('row_sums()', sort='cumulative')
```

**Why might the profiling result for this single call to `row_sums()` be misleading for production performance estimates?**

- A) `cProfile` adds so much overhead that it slows NumPy C extensions by 10–100×
- B) The first call reads 200 MB from RAM (cold cache); subsequent calls hit L3 cache and are faster — a single profiled call captures worst-case (cold) latency, not steady-state throughput
- C) `arr.sum(axis=1)` is not visible in `cProfile` output because it is a C extension; you see only the Python wrapper overhead
- D) The result is always accurate because `cProfile` measures wall-clock time

**Answer: B**

- A) Incorrect — `cProfile` instruments Python function calls, not C extension internals. Its overhead is ~1 µs per Python call. For a bulk C operation like `np.sum`, the instrumentation overhead is negligible relative to the computation.
- B) Correct — a 5000×5000 float64 array is 200 MB. The first call fetches 200 MB from DRAM at ~20–50 GB/s (cold), taking ~4–10 ms. If the same data is re-accessed immediately (warm, fitting in L3), it runs at ~100–500 GB/s — 5–25× faster. Profiling a single cold call overstates the steady-state cost for workloads where data is reused. Always warm up (call once, then measure) to get representative numbers.
- C) Incorrect — `cProfile` does show C extension calls under their Python wrapper names (e.g., `<built-in method numpy.core._multiarray_umath.ndarray.sum>`). The call does appear in the profile; the issue is not invisibility.
- D) Incorrect — wall-clock time is accurate for what it measures (elapsed real time of that one call), but accurate measurement of a single cold-cache call does not generalize to steady-state performance. "Accurate" and "representative" are different things.

---
