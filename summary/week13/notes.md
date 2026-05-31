# Week 13 — HPC Pitfalls

## Overview

Week 13 covers common mistakes that HPC users make — mistakes that either waste your own compute time or degrade the shared cluster for everyone. The lecture walks through six numbered pitfalls (excessive I/O, duplicate I/O via `tee`, spamming scheduler commands, too many files per folder, mis-configured multi-threading, and leaving background processes running) and gives concrete measurements showing how bad each one is. The exercises let you reproduce the I/O and multi-threading pitfalls yourself on DTU HPC.

Key framing from the lecture: these are guidelines and warning signs, not hard rules. Your code may be fine, but a dependency you did not write may not be.

---

## Theory & Concepts

### Categories of Pitfalls

| Category | Scope |
|---|---|
| Issues that affect only you | Wrong results, wasted walltime, slow jobs |
| Issues that affect everyone on the system | Scheduler overload, shared filesystem strain |
| Issues specific to DTU HPC | LSF/LSF channel behaviour, `/work3` vs home directories |
| General Linux / cloud issues | Too many files per directory, orphaned background processes |

The critical insight is that issues affecting the system affect you too — a degraded scheduler means your jobs queue longer.

---

## Pitfall Catalogue

### Pitfall 1 — Excessive I/O via `-o`/`-e` Channels

**What it is:**
LSF captures stdout and stderr through its own internal channels when you use `#BSUB -o` and `#BSUB -e`. These channels have significant overhead for high-volume or high-frequency output.

**Why it is a problem:**
The `-o`/`-e` mechanism is slow for large output. The lecture example showed a job printing 100,000 lines taking **80 seconds** via the `-o`/`-e` channel vs only **3 seconds** with manual shell redirection — 26x slower.

**How to avoid it:**
Manually redirect stdout and stderr in the job script using shell operators. Output must go to a fast filesystem (`/work3`, not home directories).

**Wrong way:**
```bash
#BSUB -o name_%J.out
#BSUB -e name_%J.err

python -u script.py input.npy 4
```

**Right way:**
```bash
#BSUB -o /work3/02613/dump/name_%J.out
#BSUB -e /work3/02613/dump/name_%J.err

python -u script.py input.npy 4 \
    1> /work3/02613/dump/output_${LSB_JOBID}.txt \
    2> /work3/02613/dump/error_${LSB_JOBID}.txt
```

Note: you still keep `-o`/`-e` for the LSF job summary that appears after completion — the trick is to redirect the actual program output separately so the `-o`/`-e` channel carries almost nothing.

**Caveat:** the speed difference only appears on fast filesystems. On DTU HPC, use `/work3`. Home directories are slow for this purpose.

---

### Pitfall 2 — Duplicate I/O via `tee`

**What it is:**
Using `tee` in a job script pipes program output to a file while simultaneously letting it flow through to stdout. LSF then captures that stdout into the `-o` file, so every byte is written twice.

**Why it is a problem:**
Direct quote from HPC Support: "tee: don't use tee in batch jobs! Doesn't make sense, but duplicates amount of data!" You double disk usage and double I/O load on the shared filesystem.

**How to avoid it:**
Either print to stdout or write to a file/logging system — never both via `tee`. Choose one output path.

**Wrong way:**
```bash
python -u script.py | tee output.txt
```

**Right way:**
```bash
# Option A: let LSF capture it
python -u script.py

# Option B: redirect manually
python -u script.py > /work3/02613/dump/output_${LSB_JOBID}.txt
```

**Moral from lecture:** "either print or log to file/web/other, but not both!"

Also worth noting: your code may be fine, but a Python package you import may internally use `tee` or equivalent patterns. Always inspect your dependencies.

---

### Pitfall 3 — Spamming `bjobs`/`bstat`

**What it is:**
Repeatedly calling `bstat` or `bjobs` in a tight loop (e.g., `watch bstat` with the default 2-second refresh interval) to monitor job status.

**Why it is a problem:**
Every call to `bstat`/`bjobs` interrupts the LSF scheduler. This degrades scheduling performance for everyone on the cluster. Additionally, the scheduler only updates job data 1–2 times per minute, so polling every 2 seconds is pointless as well as harmful.

`watch -n 0.1 bstat` is described in the lecture as "totally insane."

**How to avoid it:**
- Be patient
- If you must automate polling, use `watch -n30 bstat` (30-second interval) or better `watch -n60 bstat`
- Ask yourself: "do I really need this?" — you will get an email notification when the job finishes

**Wrong way:**
```bash
watch bstat         # polls every 2 seconds by default
watch -n 0.1 bstat  # totally insane
```

**Right way:**
```bash
watch -n60 bstat    # once per minute is reasonable
# or simply: bstat  # check manually when you need to
```

---

### Pitfall 4 — Too Many Files in One Folder

**What it is:**
Placing tens of thousands of files in a single directory on a shared filesystem.

**Why it is a problem:**
Shared filesystems use metadata servers to track directory contents. Thousands of files in one directory causes slowdowns for all operations on that directory — including `ls` which can take seconds. Beyond a certain threshold, shell glob expansion fails entirely:
```
$ ls bigdata/*.jpg
bash: /usr/bin/ls: Argument list too long
```

The lecture example: a directory with 100,000 files takes 2.6 seconds just to `ls`.

**How to avoid it:**
- Organise large datasets into subfolders, each containing at most ~1000 files
- For scientific array data, use `zarr.storage.NestedDirectoryStore` instead of `DirectoryStore` — the nested variant distributes chunk files across a directory hierarchy rather than dumping them all in one place

**Wrong way:**
```
bigdata/
    000001.jpg
    000002.jpg
    ...
    100000.jpg     # 100K files in one directory
```

**Right way:**
```
bigdata/
    000000/        # subfolders with ~1000 files each
        000000.jpg
        000059.jpg
        ...
    001000/
        001000.jpg
        ...
```

The CelebA dataset in the course project (`/dtu/projects/02613_2024/data/celeba/images/`) is already structured this way — subfolders named `000000`, `008000`, `016000`, etc.

---

### Pitfall 5 — Wrong Number of Threads/Processes

This pitfall has two opposite failure modes.

#### 5a — Too Many Threads (Oversubscription)

**What it is:**
Running more threads than cores allocated by LSF. The CPU model may have 32 cores total, but your job was allocated only 4 (`#BSUB -n 4`). Most scientific packages (NumPy, SciPy, scikit-learn, etc.) do not read the LSF allocation — they query the hardware and spawn threads for all 32 cores. If you are also running your own thread/process pool on top, threads multiply further.

**Why it is a problem:**
Threads fight over cores → context-switching overhead → all jobs running on that node slow down, including other users' jobs.

**Common causes:**
1. Packages that auto-detect hardware thread count instead of the LSF-allocated count. `multiprocessing.Pool()` uses `os.cpu_count()` which returns the hardware count, not the allocation.
2. You run a thread pool whose workers also call a multi-threaded package — thread count multiplies.

**Solution:**
Disable package-level multi-threading by setting environment variables before launching Python:
```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MPI_NUM_THREADS=1
```
Or, if you want to use the package's threading with N cores:
```bash
NUM_THREADS=4
OMP_NUM_THREADS=$NUM_THREADS
MKL_NUM_THREADS=$NUM_THREADS
MKL_NUM_THREADS=$NUM_THREADS
OPENBLAS_NUM_THREADS=$NUM_THREADS
```

**Wrong way:**
```bash
#BSUB -n 4
# No thread count set — NumPy will use all 32 hardware cores
python -u script.py
```

**Right way (disable package threading, use your own pool):**
```bash
#BSUB -n 8
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -u script.py
```

**Wrong way (double parallelism):**
```python
# BAD: each ThreadPool worker calls np.matmul which is itself multi-threaded
with ThreadPool(8) as p:
    C = np.concatenate(p.starmap(np.matmul, zip(A, B)))
# Result: way more threads than cores, slower than either alone
```

#### 5b — Too Few Threads (Under-utilisation)

**What it is:**
Requesting multiple cores but not telling the package to use them. Packages that support multi-threading often default to single-threaded mode and require explicit activation.

**Why it is a problem:**
You pay for cores you do not use, and your job runs slower than it could.

**Solution:**
Set the thread-count environment variables to the number of allocated cores.

**NumPy example — the right way:**
```bash
#BSUB -n 8
NUM_THREADS=8
OMP_NUM_THREADS=$NUM_THREADS
MPI_NUM_THREADS=$NUM_THREADS
MKL_NUM_THREADS=$NUM_THREADS
OPENBLAS_NUM_THREADS=$NUM_THREADS
python mm.py
```
Result from lecture: 1.72 seconds (1 thread) → 0.23 seconds (8 threads) = **7.4x speed-up**.

**Moral from lecture:** "be careful with parallel code. Check your packages!"

**NumPy thread stack:** Your code → NumPy Python API → LAPACK → BLAS → Threading system. The exact environment variable depends on which BLAS backend is installed (Netlib, OpenBLAS, ATLAS, MKL, Veclib). Setting all four variables is the safest approach.

---

### Pitfall 6 — Leaving Background Processes Running

**What it is:**
Starting a background monitoring or logging process in a job script (with `&`) that runs an infinite loop. When the main Python script finishes, the background process keeps running. LSF waits for all child processes to finish before ending the job, so the job does not terminate until it hits the walltime limit (`-W 24:00`).

**Why it is a problem:**
A job that should have taken 1 hour occupies resources for 24 hours. Node resources are wasted; other jobs cannot run there.

**Example of the problem:**
```bash
#BSUB -W 24:00

mymonitor > monitor_$LSB_JOBID.out &   # infinite loop!

# script takes approx. 1 hour
python -u script.py input.npy 4
# job now waits 23 more hours for mymonitor to exit
```

**Solution:**
Design monitoring scripts to terminate when the main work is done. Signal the monitor when the main process finishes, or have the monitor check for a sentinel file/process.

**Right way:**
```bash
mymonitor > monitor_$LSB_JOBID.out &
MONITOR_PID=$!

python -u script.py input.npy 4

kill $MONITOR_PID   # clean up monitor when work is done
wait $MONITOR_PID
```

---

## The "Good HPC Citizen" Rules

These rules protect the shared cluster for all users:

1. **Do not spam the scheduler.** Call `bstat`/`bjobs` infrequently (at most once per minute). Each call interrupts LSF scheduling.
2. **Do not oversubscribe cores.** Always set thread-count environment variables to match your LSF allocation, not the hardware count.
3. **Do not hammer the shared filesystem.** Avoid placing 100,000+ files in a single directory. Keep output on `/work3`, not home directories, and do not generate unnecessary I/O.
4. **Do not use `tee` in batch jobs.** It doubles filesystem writes with no benefit in a non-interactive context.
5. **Do not leave orphaned processes.** Kill background processes before your job ends.
6. **Request realistic walltimes.** Over-requesting walltime blocks resources from others.

**Overall principle from lecture:** treat these as guidelines and warning signs, not hard rules. Always test — different HPC systems have different bottlenecks and what is a problem on DTU HPC may not be elsewhere, and vice versa.

---

## Debugging HPC Issues

### Diagnosing I/O slowness
- Compare run times between `-o`/`-e` channel and manual redirection
- Check which filesystem your output goes to (`df -h <path>`) — only `/work3` is fast enough for high-volume output
- Check `Max Processes` and `Max Threads` in the LSF job summary at the end of the output file

### Diagnosing thread count problems
- Run `top` or `htop` on the compute node (via `ssh` into the execution host listed in `bjobs`) during a running job and check the thread count
- Check the LSF resource summary: `Max Threads` in the job output will tell you how many threads ran
- If `Max Threads` is much larger than `#BSUB -n`, you have an oversubscription problem

### Diagnosing too many files
- `ls <dir> | wc -l` to count files in a directory
- If `ls <dir>/*.ext` returns "Argument list too long", you have too many files in one folder

### Checking what a package does internally
- For NumPy: `numpy.show_config()` shows which BLAS backend is linked — this tells you which `*_NUM_THREADS` variable controls it
- Resource: https://superfastpython.com/multithreaded-numpy-functions/ lists which NumPy functions are multi-threaded

### Useful LSF commands reference
| Task | Command |
|---|---|
| Submit job | `bsub < submit.sh` |
| Check job status | `bstat` or `bjobs` |
| Peek at running job output | `bpeek <JOBID>` |
| Kill a job | `bkill <JOBID>` |

---

## Key Code Examples

### matmul.py — the baseline (from course file)

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

if __name__ == '__main__':
    A = np.random.rand(100, 1000, 1000)
    B = np.random.rand(100, 1000, 1000)
    t0 = time()
    C = matmuls(A, B)
    t1 = time()
    print(t1 - t0)
```

This serial loop over 100 matrix multiplications is the starting point. Each `np.matmul` call can internally use multiple threads if NumPy multi-threading is enabled.

### Solution: parallelise with ThreadPool (exercise 2.4)

```python
from time import perf_counter as time
import numpy as np
from multiprocessing.pool import ThreadPool

def matmuls(A, B):
    assert A.shape[0] == B.shape[0], "A and B must hold same number of mats"
    assert A.shape[2] == B.shape[1], "A and B must hold compatible matrices"
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

**Why ThreadPool works here (not ProcessPool):** NumPy releases the GIL during BLAS calls. Since most time is spent inside multi-threaded C/Fortran code, threads can run in true parallel even under the GIL. ProcessPool would require copying the large arrays to each worker process (expensive), while ThreadPool shares memory.

**Gotcha:** if you combine ThreadPool(8) with NumPy multi-threading enabled (OMP_NUM_THREADS=8), you get 8 × 8 = 64 threads fighting over 8 cores. The solutions show this is slower (1.87s) than either strategy alone. Disable package threading when using a thread pool:

```bash
# In job script, when using ThreadPool in Python:
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MPI_NUM_THREADS=1
python -u matmuls_threadpool.py
```
Result: ~1.33 seconds — comparable to pure NumPy multi-threading (1.28s).

### Complete job script template (good practices)

```bash
#!/bin/bash
#BSUB -J matmuls
#BSUB -q hpc
#BSUB -W 00:10
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o /work3/02613/dump/matmuls_%J.out
#BSUB -e /work3/02613/dump/matmuls_%J.err

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

NUM_THREADS=8
OMP_NUM_THREADS=$NUM_THREADS
MPI_NUM_THREADS=$NUM_THREADS
MKL_NUM_THREADS=$NUM_THREADS
OPENBLAS_NUM_THREADS=$NUM_THREADS

python -u matmuls.py \
    1> /work3/02613/dump/output_${LSB_JOBID}.txt \
    2> /work3/02613/dump/error_${LSB_JOBID}.txt
```

### Measured performance summary (from exercises and lecture)

| Configuration | Run time |
|---|---|
| Excessive I/O via `-o`/`-e`, 100k lines | 80 sec |
| Manual redirect to `/work3` | 3 sec (26x faster) |
| matmul, 1 core, NumPy single-threaded | ~5.87 sec |
| matmul, 8 cores, NumPy still single-threaded | ~5.96 sec (no change) |
| matmul, 8 cores, NumPy multi-threading enabled | ~1.28 sec (4.6x faster) |
| matmul, ThreadPool(8) + NumPy multi-threading | ~1.87 sec (slower — oversubscribed) |
| matmul, ThreadPool(8) + NumPy single-threaded | ~1.33 sec (back to good) |

---

## Key Takeaways

1. **Redirect output manually to `/work3`.** The LSF `-o`/`-e` channels are slow for high-volume output. Manual shell redirection to `/work3` can be 26x faster. Keep `-o`/`-e` for the summary, redirect actual program output separately.

2. **Never use `tee` in batch jobs.** It duplicates all output data — once to the file and once to stdout which LSF captures again.

3. **Do not spam `bstat`/`bjobs`.** Every call interrupts the scheduler for everyone. Use `watch -n60 bstat` at most; better yet, wait for the email notification.

4. **Keep files spread across subdirectories.** No more than ~1000 files per directory to avoid shared filesystem metadata bottlenecks. Use `zarr.NestedDirectoryStore` for array data.

5. **Always set thread-count environment variables.** Packages like NumPy use BLAS/LAPACK which can auto-detect hardware thread counts (ignoring your LSF allocation). Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MPI_NUM_THREADS` explicitly to match `#BSUB -n`.

6. **Do not stack parallelism layers.** A thread pool whose workers call multi-threaded NumPy functions results in O(n²) threads. Disable package threading when running your own pool, or let the package handle threading and keep your code serial.

7. **ThreadPool works for NumPy because it releases the GIL.** BLAS calls run in C/Fortran and release the Python GIL, so Python threads achieve true parallelism for this workload without the copy overhead of ProcessPool.

8. **Kill background processes before job exit.** A monitoring script with an infinite loop will prevent your job from terminating until the walltime limit, wasting hours of allocated resources.

9. **Your code may be fine, but a dependency may not be.** The central lesson of the lecture is that problems often come from packages you did not write and did not inspect. Always check what your packages are doing under the hood.

10. **These are guidelines, not laws.** Test on the actual system. Different HPC clusters have different bottlenecks — what is critical on DTU HPC may not matter elsewhere.
