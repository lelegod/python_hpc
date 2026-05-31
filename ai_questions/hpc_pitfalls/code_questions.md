# HPC Pitfalls — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Question 1 — Missing export on Thread Env Vars](#question-1)
- [Question 2 — Shell Redirection vs LSF -o](#question-2)
- [Question 3 — ThreadPool + Multi-Threaded NumPy Oversubscription](#question-3)
- [Question 4 — Requesting Cores Without span[hosts=1]](#question-4)
- [Question 5 — Diagnosing No Speedup: Cores vs Threads](#question-5)
- [Question 6 — Buffered vs Unbuffered Python Output](#question-6)
- [Question 7 — Memory Request Calculation](#question-7)
- [Question 8 — Subshell Export Trap](#question-8)
- [Question 9 — Fixing Oversubscription by Setting OMP to 1](#question-9)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2--generated-practice-questions-exam-day-focus)
- [Question 10 — Login Node Computation](#question-10)
- [Question 11 — Multiple Pitfalls in One Script](#question-11)
- [Question 12 — Correct ThreadPool + NumPy Setup](#question-12)
- [Question 13 — Identifying the Fastest I/O Approach](#question-13)
- [Question 14 — ProcessPool vs ThreadPool for NumPy](#question-14)
- [Question 15 — MKL_NUM_THREADS Alone Is Not Enough](#question-15)
- [Question 16 — Output File Location Pitfall](#question-16)
- [Question 17 — Predicting Run Time from Thread Count](#question-17)
- [Question 18 — Fixing All Pitfalls in a Combined Script](#question-18)

---

> Format: Each question shows a buggy or suboptimal job script or Python code — identify the pitfall or fix.
> Exam frequency: **Week 13 topic — common exam target**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#question-1)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Question 1

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

## Question 2

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

## Question 3

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

## Question 4

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

## Question 5

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

## Question 6

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

## Question 7

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

## Question 8

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

## Question 9

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

## Question 10

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

## Question 11

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

## Question 12

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

`matmul.py` runs a for loop over 100 `np.matmul` calls with single-threaded BLAS. Which statement is true about this script?

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

## Question 13

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

## Question 14

```python
from multiprocessing.pool import ProcessPool
import numpy as np

def single_matmul(args):
    a, b = args
    return np.matmul(a, b)

def matmuls(A, B):
    n = A.shape[0]
    with ProcessPool(8) as p:
        C = np.concatenate(p.map(single_matmul, zip(A, B)))
    return C
```

A colleague proposes this `ProcessPool`-based version instead of `ThreadPool`. What is the main performance concern compared to a `ThreadPool` solution?

**A)** `ProcessPool` cannot use NumPy because each worker gets a separate Python interpreter

**B)** `ProcessPool` requires serializing (pickling) each 1000×1000 float64 array for inter-process communication, adding significant data transfer overhead not present with ThreadPool

**C)** `ProcessPool` workers cannot run on HPC clusters because they require root privileges

**D)** `ProcessPool` ignores `OMP_NUM_THREADS` so BLAS always uses 1 thread in each worker

**Answer: B**

- A) Incorrect — `ProcessPool` workers do have separate Python interpreters, but they can absolutely use NumPy. Each worker imports and uses NumPy independently.
- B) Correct — `ProcessPool` workers are separate OS processes. Data passed to them must be serialized (pickled) and sent through an IPC pipe. Each 1000×1000 float64 array is 8 MB; serializing and transmitting 100 such arrays adds substantial overhead compared to `ThreadPool`, where threads share the same memory address space and array slices are just views — no copying required.
- C) Incorrect — `ProcessPool` is a standard Python library feature with no special privilege requirements; it works on HPC clusters without issue.
- D) Incorrect — `ProcessPool` workers inherit the parent's exported environment, including `OMP_NUM_THREADS`. BLAS in each worker respects the thread count directive.

---

## Question 15

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

## Question 16

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

## Question 17

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

## Question 18

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
