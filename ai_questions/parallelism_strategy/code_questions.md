# Parallelism Strategy — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — GIL with pure Python loops](#q1-gil-with-pure-python-loops)
- [Q2 — NumPy releases the GIL](#q2-numpy-releases-the-gil)
- [Q3 — Reading `time` command output](#q3-reading-time-command-output)
- [Q4 — Predicting `time` output after scaling up cores](#q4-predicting-time-output-after-scaling-up-cores)
- [Q5 — IPC overhead from too many small tasks](#q5-ipc-overhead-from-too-many-small-tasks)
- [Q6 — Numba `nogil=True` with `ThreadPool`](#q6-numba-nogiltrue-with-threadpool)
- [Q7 — Static chunking with variable-runtime tasks](#q7-static-chunking-with-variable-runtime-tasks)
- [Q8 — Missing `if __name__ == '__main__':` guard](#q8-missing-if-__name__-__main__-guard)
- [Q9 — Choosing static vs dynamic scheduling](#q9-choosing-static-vs-dynamic-scheduling)
- [Q10 — `Pool()` ignores LSF allocation](#q10-pool-ignores-lsf-allocation)
- [Key Facts Summary](#key-facts-summary)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q11 — Threading a Pure Python Accumulator](#q11-threading-a-pure-python-accumulator)
- [Q12 — NumPy Thread Parallelism: Predicting `time` Output](#q12-numpy-thread-parallelism-predicting-time-output)
- [Q13 — Numba `nogil=True` Thread Count Scaling](#q13-numba-nogiltrue-thread-count-scaling)
- [Q14 — Detecting Oversubscription on an HPC Node](#q14-detecting-oversubscription-on-an-hpc-node)
- [Q15 — Choosing `chunksize` for a Mandelbrot Computation](#q15-choosing-chunksize-for-a-mandelbrot-computation)
- [Q16 — `ThreadPool` vs `Pool` for Mixed NumPy and Python Workload](#q16-threadpool-vs-pool-for-mixed-numpy-and-python-workload)
- [Q17 — Race Condition with Shared Counter in `multiprocessing`](#q17-race-condition-with-shared-counter-in-multiprocessing)
- [Q18 — Interpreting Speedup from `time` with 8 Workers](#q18-interpreting-speedup-from-time-with-8-workers)
- [Q19 — When Does `nogil=True` Not Help?](#q19-when-does-nogiltrue-not-help)
- [Q20 — Combining `imap_unordered` and Task Batching](#q20-combining-imap_unordered-and-task-batching)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q21 — `pool.starmap` vs Lambda with `spawn`](#q21--poolstarmap-vs-lambda-with-spawn)
- [Q22 — `pool.imap` Return Type and Memory](#q22--poolimap-return-type-and-memory)
- [Q23 — `apply_async` Pattern from `pi_parallel.py`](#q23--apply_async-pattern-from-pi_parallelpy)
- [Q24 — Chunked vs Per-Sample Parallelism](#q24--chunked-vs-per-sample-parallelism)
- [Q25 — `fork` Inherits Parent State](#q25--fork-inherits-parent-state)
- [Q26 — `ProcessPoolExecutor.map` vs `Pool.starmap`](#q26--processpoolexecutormap-vs-poolstarmap)
- [Q27 — Lambda Pickling Failure with `spawn`](#q27--lambda-pickling-failure-with-spawn)
- [Q28 — `RawArray` Shared Memory Pattern](#q28--rawarray-shared-memory-pattern)
- [Q29 — Non-Associative Operator in Reduction](#q29--non-associative-operator-in-reduction)
- [Q30 — `pool.map` with `chunksize` Effect on `time` Output](#q30--poolmap-with-chunksize-effect-on-time-output)

---

> Format: Each question shows Python parallelism code or shell output to analyse.
> Exam frequency: **Every exam**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--gil-with-pure-python-loops)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — GIL with pure Python loops

```python
def process_number(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

with Pool(8) as pool:
    results = pool.map(process_number, range(1, 100_000_001))
```

**Should this use `ThreadPool` or `Pool` (multiprocessing)?**

A) `ThreadPool` — threads are lighter weight and share memory  
B) `Pool` — each process gets its own GIL, so all 8 workers run truly in parallel  
C) Either — Python's GIL is only a problem for I/O-bound work  
D) Neither — this workload cannot be parallelised  

**Answer: B**

- A) Incorrect — threads share the GIL, so pure Python threads run sequentially, not in parallel
- B) Correct — multiprocessing spawns separate interpreter processes each with their own GIL, enabling true parallelism
- C) Incorrect — the GIL is a problem for CPU-bound pure Python work, not just I/O-bound work
- D) Incorrect — the workload is embarrassingly parallel and can be split across processes

---

## Q2 — NumPy releases the GIL

```python
from multiprocessing.pool import ThreadPool
import numpy as np

def row_sum(row):
    return np.sum(row)

with ThreadPool(8) as pool:
    results = pool.map(row_sum, matrix)
```

**Is `ThreadPool` appropriate here? Will the threads actually run in parallel?**

A) No — threading is never appropriate for CPU-bound work in Python  
B) No — `np.sum` still holds the GIL like any other Python function  
C) Yes — NumPy releases the GIL during its C-level computation, so all 8 threads run truly in parallel  
D) Yes — but only if `matrix` is a list of Python lists, not a NumPy array  

**Answer: C**

- A) Incorrect — threading is appropriate for CPU-bound work when the underlying code releases the GIL (as NumPy does)
- B) Incorrect — `np.sum` is a C extension that explicitly releases the GIL during its inner loop
- C) Correct — NumPy's C extension functions release the GIL, allowing all 8 threads to run `np.sum` simultaneously
- D) Incorrect — NumPy releases the GIL regardless of the input type; using a NumPy array is actually more efficient

---

## Q3 — Reading `time` command output

```
real    0m6.240s
user    0m23.814s
sys     0m0.142s
```

**What does this output tell you about how the program ran?**

A) The program ran for 23.8 seconds wall-clock time using a single core  
B) The program ran for 6.2 seconds wall-clock time; `user` time of 23.8 s spread over ~4 cores confirms parallel execution  
C) The program spent 23.8 s waiting for I/O  
D) `real` > `user` would indicate parallelism; here `user` > `real`, so something is wrong  

**Answer: B**

- A) Incorrect — `real` is wall-clock time (6.24 s), not `user`; 23.8 s is summed CPU time across all cores
- B) Correct — `real`=6.24 s is elapsed time, `user`/`real`≈3.8 indicates ~4 cores ran in parallel
- C) Incorrect — `user` time is CPU computation time, not I/O wait; I/O wait appears as `real` >> `user`
- D) Incorrect — `user` > `real` is the normal and expected signature of a parallel program using multiple cores

---

## Q4 — Predicting `time` output after scaling up cores

```
# Single-threaded baseline:
real    0m12.000s
user    0m11.980s
sys     0m0.020s
```

**If the same program is re-run with 2 parallel workers (perfectly parallelisable workload), what would the `time` output look like?**

A) `real≈12s`, `user≈6s`, `sys≈0.01s`  
B) `real≈6s`, `user≈6s`, `sys≈0.01s`  
C) `real≈6s`, `user≈12s`, `sys≈0.02s`  
D) `real≈6s`, `user≈24s`, `sys≈0.04s`  

**Answer: C**

- A) Incorrect — `real` would halve with 2 workers, and `user` does not halve (it represents total CPU work done)
- B) Incorrect — `user` stays approximately constant because the same total computation is performed; it does not halve
- C) Correct — `real` halves to ~6 s as work is split across 2 cores; `user` stays ~12 s as total CPU time is conserved
- D) Incorrect — `user` would not double; adding more workers conserves total CPU time, it does not multiply it

---

## Q5 — IPC overhead from too many small tasks

```python
with Pool(4) as pool:
    results = pool.map(process_item, range(1_000_000))
```

**Why might this be *slower* than a serial implementation?**

A) `Pool(4)` can only use 4 CPUs; a serial program can use all available CPUs  
B) 1 million tasks means 1 million inter-process messages; IPC serialisation overhead dominates for small tasks  
C) `pool.map` blocks until all results are ready, preventing any parallelism  
D) `range(1_000_000)` is a generator and cannot be passed to `pool.map`  

**Answer: B**

- A) Incorrect — a serial program uses only 1 CPU; Pool(4) uses up to 4, which is more, not fewer
- B) Correct — each of 1,000,000 items must be pickled, sent via IPC, computed, pickled again, and returned; this overhead dominates for tiny tasks
- C) Incorrect — `pool.map` blocking for results does not prevent parallelism; workers execute concurrently while the main process waits
- D) Incorrect — `pool.map` accepts any iterable including `range` objects

---

## Q6 — Numba `nogil=True` with `ThreadPool`

```python
from numba import jit

@jit(nopython=True, nogil=True)
def simulate(n):
    x = 0.0
    for i in range(n):
        x += i * 0.001
    return x

from multiprocessing.pool import ThreadPool
with ThreadPool(8) as pool:
    results = pool.map(simulate, [10**6]*8)
```

**Is `ThreadPool` appropriate here? Will threads run in parallel?**

A) No — it is still a Python loop inside `simulate`, so the GIL is held  
B) No — `nopython=True` removes Python objects, but the GIL is still held  
C) Yes — `nogil=True` instructs Numba to release the GIL during JIT-compiled execution, so 8 threads run truly in parallel  
D) Yes — but only after the first call warms up the JIT compilation  

**Answer: C**

- A) Incorrect — the loop is compiled to native machine code by Numba; it is no longer a Python bytecode loop and does not hold the GIL
- B) Incorrect — `nopython=True` alone does not release the GIL; the additional `nogil=True` flag is required for that
- C) Correct — `nogil=True` explicitly instructs Numba to release the GIL during JIT-compiled execution, enabling true thread parallelism
- D) Incorrect — while JIT warmup does occur on the first call, the parallelism itself does not depend on warmup state

---

## Q7 — Static chunking with variable-runtime tasks

```python
# simulate_ball has highly variable runtime (seconds to hours per call)
with Pool(8) as pool:
    results = pool.map(simulate_ball, all_params, chunksize=10)
```

**Is `chunksize=10` a good choice here? Why or why not?**

A) Yes — larger chunks always reduce IPC overhead and improve performance  
B) Yes — 10 is small enough to adapt to runtime variance  
C) No — `chunksize` must equal 1 for multiprocessing to work correctly  
D) No — static chunks cause load imbalance; some workers will finish early and idle while others run long tasks  

**Answer: D**

- A) Incorrect — larger chunks reduce IPC overhead only when task runtimes are uniform; with high variance they cause idle workers
- B) Incorrect — a chunksize of 10 is still static pre-assignment; it does not adapt dynamically to which tasks run long or short
- C) Incorrect — `chunksize` can be any positive integer; 1 is valid but not a requirement for correctness
- D) Correct — with highly variable runtimes, static pre-assignment leaves workers idle when their assigned tasks happen to be fast while others still run long tasks

---

## Q8 — Missing `if __name__ == '__main__':` guard

```python
import multiprocessing as mp

def worker(x):
    return x ** 2

pool = mp.Pool(4)
results = pool.map(worker, range(10))
```

**What is missing that would cause infinite process spawning on Windows and macOS?**

A) `pool.close()` and `pool.join()` are not called  
B) The `if __name__ == '__main__':` guard is missing  
C) `mp.set_start_method('fork')` must be called before creating the pool  
D) `worker` must be defined inside the `if __name__ == '__main__':` block  

**Answer: B**

- A) Incorrect — missing `pool.close()`/`pool.join()` is a resource leak but does not cause infinite process spawning
- B) Correct — on Windows/macOS (`spawn` start method) each worker re-imports the module, re-executing `mp.Pool(4)` and creating a recursive fork bomb
- C) Incorrect — setting the start method to `fork` would sidestep the issue, but it is not the standard fix and is unavailable on Windows
- D) Incorrect — `worker` can be defined at module level; it is the `mp.Pool()` call that must be inside the guard, not the function definition

---

## Q9 — Choosing static vs dynamic scheduling

```
kernel1 runtimes (ms): [42, 38, 41, 39, 40, 43, 38]  # stddev ≈ 1.8 ms
kernel2 runtimes (ms): [12, 890, 45, 340, 5, 720, 88]  # stddev ≈ 350 ms
```

**Which scheduling strategy should be used for `kernel1` and `kernel2`?**

A) Both should use dynamic scheduling (chunksize=1) for maximum flexibility  
B) Both should use static scheduling (large chunksize) to minimise IPC overhead  
C) `kernel1` → static (large chunksize); `kernel2` → dynamic (chunksize=1 / imap_unordered)  
D) `kernel1` → dynamic; `kernel2` → static  

**Answer: C**

- A) Incorrect — dynamic scheduling adds IPC overhead per task; for uniform `kernel1` this overhead is wasteful with no load-balancing benefit
- B) Incorrect — static scheduling for `kernel2` would leave workers idle while others process the 890 ms outliers
- C) Correct — uniform runtimes (kernel1) suit static pre-assignment; high-variance runtimes (kernel2) require dynamic scheduling to keep all cores busy
- D) Incorrect — this reverses the correct pairing; dynamic is needed where variance is high, not where it is low

---

## Q10 — `Pool()` ignores LSF allocation

```bash
#BSUB -n 8
python script.py
```

```python
from multiprocessing import Pool
import os

pool = Pool()  # No argument passed
print(f"Workers: {pool._processes}")
```

**What does `Workers:` print when this job runs on a 32-core machine?**

A) `Workers: 8` — Pool() respects the LSF `-n 8` CPU allocation  
B) `Workers: 1` — Pool() defaults to a single process for safety  
C) `Workers: 32` — Pool() calls `os.cpu_count()` which returns the total machine CPUs, not the LSF allocation  
D) `Workers: 4` — Pool() uses half of `os.cpu_count()` by default  

**Answer: C**

- A) Incorrect — `Pool()` has no awareness of LSF scheduler directives; it never reads `#BSUB -n` or `LSB_DJOB_NUMPROC` automatically
- B) Incorrect — `Pool()` does not default to 1; it defaults to `os.cpu_count()` which reflects the physical machine
- C) Correct — `Pool()` calls `os.cpu_count()` returning the machine total (32), oversubscribing the 8 allocated cores and hurting performance
- D) Incorrect — `Pool()` uses the full `os.cpu_count()` value, not half of it

---

## Key Facts Summary

| Concept | Rule |
|---------|------|
| **GIL + pure Python** | Holds GIL → `ThreadPool` is sequential → use `multiprocessing.Pool` |
| **NumPy** | Releases GIL in C code → `ThreadPool` works for CPU-bound NumPy |
| **Numba `nogil=True`** | Releases GIL in JIT code → `ThreadPool` works |
| **`time` output** | `real` ↓ with parallelism; `user` ≈ constant (summed over cores) |
| **IPC overhead** | Too many tiny tasks → serialisation cost dominates → chunk the work |
| **Load balancing** | Uniform runtimes → static (large chunksize); variable → dynamic (`imap_unordered`) |
| **`Pool()` default** | Uses `os.cpu_count()` (machine total), NOT LSF `-n` allocation |
| **`__main__` guard** | Required on Windows/macOS (`spawn`); prevents recursive process spawning |

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets GIL behavior with threads vs multiprocessing, NumPy GIL release, Numba nogil, static vs dynamic scheduling, and choosing the right parallelism strategy

---

## Q11 — Threading a Pure Python Accumulator

> **Week reference:** Week 5

```python
import threading

total = 0

def accumulate(n):
    global total
    for i in range(n):
        total += i

threads = [threading.Thread(target=accumulate, args=(10**6,)) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(total)
```

**Which two problems does this code have?**

- A) No GIL issue; race condition on `total` only
- B) GIL serialises threads (no speedup); AND unsynchronised increments of `total` cause a race condition — the result is non-deterministic
- C) GIL prevents all execution; the script hangs
- D) Using `global` causes a NameError; the race condition is the only issue

**Answer: B**

The GIL means only one thread runs Python bytecode at a time — four threads on a pure Python loop give no speedup over serial. Additionally, `total += i` is not atomic: it involves a `LOAD_FAST`, `INPLACE_ADD`, and `STORE_FAST`. The GIL can be released between these instructions, so two threads can read the same value of `total`, both add their `i`, and write back conflicting results — a classic read-modify-write race condition. The final `total` will be less than the correct value and will differ between runs.

- A) Incorrect — the GIL is absolutely an issue here: threads compete for the GIL and run sequentially, so there is no parallel speedup.
- B) Correct — GIL blocks parallel speedup; unsynchronised `total +=` produces non-deterministic, incorrect results.
- C) Incorrect — the GIL does not prevent execution; it serialises threads. The script runs to completion but gives wrong results and no speedup.
- D) Incorrect — `global total` is syntactically valid. The race condition exists alongside the GIL serialisation issue.

---

## Q12 — NumPy Thread Parallelism: Predicting `time` Output

> **Week reference:** Week 5

```python
from multiprocessing.pool import ThreadPool
import numpy as np

def compute(arr):
    return np.dot(arr, arr)

arrays = [np.random.rand(5000, 5000) for _ in range(4)]

# Baseline (serial): real=20s, user=20s
# Now run with ThreadPool(4):
with ThreadPool(4) as pool:
    results = pool.map(compute, arrays)
```

**What do you expect the `time` output to look like after the ThreadPool change (on a 4-core machine)?**

- A) `real≈20s`, `user≈20s` — threading with NumPy is still serialised by the GIL
- B) `real≈5s`, `user≈20s` — NumPy releases the GIL; 4 threads run in parallel; user time is conserved
- C) `real≈5s`, `user≈5s` — both real and user time halve proportionally with more cores
- D) `real≈20s`, `user≈80s` — 4 threads run in parallel causing user time to quadruple while real stays flat

**Answer: B**

`np.dot` is a BLAS-backed C extension that releases the GIL during matrix multiplication. All 4 threads run simultaneously on their separate arrays, reducing wall-clock time from 20s to ~5s. Total CPU work is unchanged — each thread still performs the same computation — so `user` time stays ~20s (sum of 4 × ~5s per thread).

- A) Incorrect — NumPy releases the GIL; threading with NumPy achieves genuine parallelism.
- B) Correct — `real` halves to ~5s (4× speedup); `user` stays ~20s (CPU work is conserved across threads).
- C) Incorrect — `user` time represents total CPU work, which does not change with parallelism. Only `real` (wall clock) decreases.
- D) Incorrect — `real` would decrease, not stay flat. And while `user` can exceed `real` with parallelism, it stays approximately constant, not quadruple.

---

## Q13 — Numba `nogil=True` Thread Count Scaling

> **Week reference:** Week 9

```python
from numba import njit
from concurrent.futures import ThreadPoolExecutor
import time

@njit(nogil=True)
def heavy_compute(n):
    acc = 0.0
    for i in range(n):
        acc += i ** 0.5
    return acc

# Serial baseline: 1 call with n=10**8 → 2.0s
# Parallel: 4 calls with n=10**8 each, using ThreadPoolExecutor(4)
with ThreadPoolExecutor(4) as ex:
    futs = [ex.submit(heavy_compute, 10**8) for _ in range(4)]
    results = [f.result() for f in futs]
```

**What is the expected wall-clock time for the parallel version on a 4-core machine?**

- A) ~8.0s — 4 calls × 2.0s each, still serialised by the GIL
- B) ~2.0s — `nogil=True` releases the GIL; all 4 threads run truly in parallel
- C) ~0.5s — ThreadPoolExecutor splits each call across 4 cores internally
- D) ~4.0s — threading halves the time but cannot achieve full 4× speedup

**Answer: B**

`@njit(nogil=True)` compiles to native machine code and releases the GIL during execution. All 4 `ThreadPoolExecutor` threads run their `heavy_compute` call simultaneously on separate cores, taking ~2.0s wall-clock (same as one serial call). Total user CPU time would be ~8.0s (4 × 2.0s), but real time stays at ~2.0s.

- A) Incorrect — `nogil=True` prevents GIL serialisation; threads run in parallel, not sequentially.
- B) Correct — 4 parallel threads each take 2.0s; wall-clock time ≈ 2.0s (not 4 × 2.0s).
- C) Incorrect — `ThreadPoolExecutor` does not split a single function call across cores; it submits whole calls to separate threads.
- D) Incorrect — with `nogil=True` and 4 independent calls on 4 cores, the speedup is ~4×, not ~2×.

---

## Q14 — Detecting Oversubscription on an HPC Node

> **Week reference:** Week 5

```python
# BSUB script:  #BSUB -n 4
# Python script running on a 64-core HPC node:

from multiprocessing import Pool, cpu_count
import os

print(f"cpu_count={cpu_count()}")
print(f"LSB_DJOB_NUMPROC={os.environ.get('LSB_DJOB_NUMPROC', 'not set')}")

with Pool() as pool:
    print(f"workers={pool._processes}")
```

**On a 64-core HPC node with `#BSUB -n 4`, what does this script print?**

- A) `cpu_count=4`, `LSB_DJOB_NUMPROC=4`, `workers=4`
- B) `cpu_count=64`, `LSB_DJOB_NUMPROC=4`, `workers=64`
- C) `cpu_count=4`, `LSB_DJOB_NUMPROC=not set`, `workers=4`
- D) `cpu_count=64`, `LSB_DJOB_NUMPROC=not set`, `workers=64`

**Answer: B**

`cpu_count()` queries the OS for physical CPUs on the node — it returns 64. LSF sets `LSB_DJOB_NUMPROC=4` in the job's environment, so that prints correctly. `Pool()` with no argument calls `cpu_count()` and creates 64 workers — oversubscribing the 4 allocated cores by 16×, which degrades performance and steals resources from other jobs.

- A) Incorrect — `cpu_count()` returns the node's total CPUs (64), not the LSF allocation (4). LSF does not make `cpu_count()` allocation-aware.
- B) Correct — `cpu_count()=64` (node total), `LSB_DJOB_NUMPROC=4` (LSF sets this), `Pool()` → 64 workers (oversubscribed).
- C) Incorrect — LSF always sets `LSB_DJOB_NUMPROC` when a job allocates cores; it is not "not set" within an LSF job.
- D) Incorrect — LSF does set `LSB_DJOB_NUMPROC`; the correct fix is to read it with `os.environ.get('LSB_DJOB_NUMPROC')`.

---

## Q15 — Choosing `chunksize` for a Mandelbrot Computation

> **Week reference:** Week 6

```python
from multiprocessing import Pool

def mandelbrot_pixel(coord):
    x, y = coord
    c = complex(x, y)
    z = 0
    for i in range(1000):
        z = z*z + c
        if abs(z) > 2:
            return i
    return 1000

coords = [(x/500, y/500) for x in range(-500, 500) for y in range(-500, 500)]

# Option A:
with Pool(8) as pool:
    result_a = pool.map(mandelbrot_pixel, coords, chunksize=len(coords)//8)

# Option B:
with Pool(8) as pool:
    result_b = list(pool.imap_unordered(mandelbrot_pixel, coords, chunksize=1))
```

**Which option gives better performance and why?**

- A) Option A — equal static chunks reduce IPC overhead for this uniform computation
- B) Option B — Mandelbrot pixels have highly variable iteration counts; dynamic scheduling prevents idle workers
- C) Option A — large chunksize is always faster than chunksize=1
- D) Both perform identically — chunksize only affects result ordering, not performance

**Answer: B**

Mandelbrot iteration counts vary from 1 (pixels that escape immediately) to 1000 (pixels inside the set). A static chunk that happens to contain many interior-set pixels takes 1000× longer than a chunk of exterior pixels. Workers that finish exterior-heavy chunks sit idle while one worker grinds through interior pixels. `imap_unordered` with `chunksize=1` assigns one pixel at a time, so finishing workers immediately pick up the next task. For 1,000,000 pixels the IPC overhead of chunksize=1 can be reduced by using a small non-1 chunksize (e.g. 100), but dynamic scheduling remains the right strategy.

- A) Incorrect — Mandelbrot is not a uniform computation; static equal chunks create severe load imbalance.
- B) Correct — highly variable iteration counts make dynamic scheduling the right choice to prevent idle workers.
- C) Incorrect — large chunksize is only faster when task runtimes are uniform. With variable runtimes it causes load imbalance.
- D) Incorrect — chunksize directly controls scheduling strategy and has a major effect on performance when runtimes vary.

---

## Q16 — `ThreadPool` vs `Pool` for Mixed NumPy and Python Workload

> **Week reference:** Week 5

```python
def mixed_task(data):
    # Step 1: pure Python preprocessing (loops, string operations) — 20% of time
    processed = [str(x) for x in data]
    # Step 2: NumPy computation — 80% of time
    arr = np.array([float(s) for s in processed])
    return np.fft.fft(arr)
```

**You want to run 100 independent calls. Which parallelisation is most appropriate?**

- A) `multiprocessing.Pool` — the 20% pure Python ensures threads are always slower
- B) `ThreadPoolExecutor` — NumPy's `fft` releases the GIL and dominates runtime (80%), so threads achieve near-linear speedup
- C) `ThreadPoolExecutor` for the NumPy step; `multiprocessing.Pool` for the preprocessing step
- D) Neither — tasks with mixed Python and NumPy cannot be parallelised

**Answer: B**

80% of `mixed_task` is `np.fft.fft`, which releases the GIL. The remaining 20% (pure Python list comprehensions) is GIL-held and serialises across threads, but it is a minor fraction. The effective parallel fraction is ~80%, giving up to ~5× speedup with enough workers (Amdahl ceiling: 1/0.2 = 5×). `ThreadPoolExecutor` avoids process-spawn and IPC pickling overhead, making it more efficient than multiprocessing for this mixed workload.

- A) Incorrect — 20% serial fraction limits speedup to ~5× but does not make threads always slower. The dominant 80% NumPy work enables real thread parallelism.
- B) Correct — 80% GIL-releasing NumPy work enables meaningful thread parallelism; `ThreadPoolExecutor` is the low-overhead choice.
- C) Incorrect — you cannot easily split a single function into separate pool calls for each phase without restructuring the code. The function is called as a unit.
- D) Incorrect — mixed workloads benefit from threading if the GIL-releasing portion dominates runtime.

---

## Q17 — Race Condition with Shared Counter in `multiprocessing`

> **Week reference:** Week 5

```python
from multiprocessing import Pool, Value

counter = Value('i', 0)

def increment(n):
    for _ in range(n):
        counter.value += 1

with Pool(4) as pool:
    pool.map(increment, [250_000] * 4)

print(counter.value)  # Expected: 1_000_000
```

**What will `counter.value` most likely print, and why?**

- A) `1000000` — `Value` provides automatic atomic access across processes
- B) A value less than `1000000` — `counter.value += 1` is not atomic; without a lock, concurrent increments from different processes overwrite each other
- C) `1000000` — processes have separate memory spaces so there is no race condition
- D) `0` — `Value` objects cannot be shared between Pool workers spawned with `spawn` start method

**Answer: B**

`multiprocessing.Value` creates a shared memory segment accessible across processes, but `counter.value += 1` is still not atomic: it is a read-modify-write of the shared integer. Without a lock (`Value('i', 0, lock=True)` with explicit `acquire()`/`release()`), concurrent processes can read the same value, both add 1, and write back the same result — losing increments. The fix is `with counter.get_lock(): counter.value += 1`.

- A) Incorrect — `Value` provides a lock attribute but does not make `+=` atomic by default. You must explicitly acquire the lock.
- B) Correct — unsynchronised `+=` on a `Value` is a race condition; the result will be less than 1,000,000.
- C) Incorrect — `multiprocessing.Value` is specifically designed for shared memory across processes; it is shared, not separate.
- D) Incorrect — `Value` objects created before `Pool` is created are accessible to workers on POSIX systems. On Windows/macOS with `spawn`, you must pass the `Value` via initializer arguments, but this code pattern works on Linux with `fork`.

---

## Q18 — Interpreting Speedup from `time` with 8 Workers

> **Week reference:** Week 5

```
# Serial run:
real    0m40.0s
user    0m39.8s
sys     0m0.2s

# Parallel run with Pool(8):
real    0m6.2s
user    0m46.5s
sys     0m1.8s
```

**What is the observed speedup, and what does `user > 8 × real` signal?**

- A) Speedup = 6.45×; `user` slightly above `8 × real` is normal — small overhead from IPC and scheduling
- B) Speedup = 6.45×; `user > real` means the program has a bug causing wasted CPU work
- C) Speedup = 8×; `user` should equal `real` for correct parallel programs
- D) No speedup — `user` time increased from 39.8s to 46.5s, indicating more work was done

**Answer: A**

Observed speedup = 40.0s / 6.2s ≈ 6.45×. Less than 8× is expected due to Amdahl's Law (serial fraction), process-spawn overhead, and IPC. `user` = 46.5s vs `8 × real` = 49.6s: user is slightly below 8 × real, meaning workers are not 100% utilised (some idle time, spawn overhead). `user > real` is the normal signature of successful multi-process parallelism — total CPU-seconds accumulate across all 8 worker processes.

- A) Correct — 40/6.2 ≈ 6.45× speedup; `user` ≈ 8 × real confirms ~8 parallel workers with small overhead.
- B) Incorrect — `user > real` is not a bug signal; it is the expected pattern for multi-process programs. A bug would cause user >> N × real.
- C) Incorrect — `user` equals `real` only for single-threaded programs. With N workers, `user ≈ N × real` when workers are busy.
- D) Incorrect — `user` increasing is expected and correct; real time (wall clock) decreased from 40s to 6.2s showing clear speedup.

---

## Q19 — When Does `nogil=True` Not Help?

> **Week reference:** Week 9

```python
from numba import njit
from concurrent.futures import ThreadPoolExecutor

@njit(nogil=True)
def process(data):
    results = []          # creates a Python list
    for x in data:
        results.append(x * 2.0)
    return results

with ThreadPoolExecutor(4) as ex:
    futs = [ex.submit(process, [1.0]*10**6) for _ in range(4)]
```

**Will this code compile with `nopython=True` / `@njit`, and does `nogil=True` achieve parallelism?**

- A) Yes, compiles fine; `nogil=True` enables parallel execution across 4 threads
- B) No, `@njit` will raise a `TypingError` — Python lists with `append` are not supported in nopython mode; code must be rewritten with NumPy arrays
- C) Yes, compiles fine; but `nogil=True` is ignored because the function returns a Python list
- D) No, `@njit` silently falls back to object mode and `nogil=True` has no effect

**Answer: B**

`@njit` is equivalent to `@jit(nopython=True)`. In nopython mode, Numba cannot work with dynamically-typed Python lists containing mixed or untyped items in all contexts, and using `list.append` on a reflected list raises a `TypingError` in newer Numba versions, or at minimum requires special handling. The standard fix is to replace the Python list with a pre-allocated `numpy.zeros` array and index into it. Once rewritten with NumPy arrays, `@njit(nogil=True)` correctly releases the GIL and enables 4× parallel speedup.

- A) Incorrect — `@njit` with Python lists and `append` typically fails to compile in nopython mode with a TypingError.
- B) Correct — Python dynamic lists are not cleanly supported by `@njit` nopython mode; NumPy arrays are the required substitute.
- C) Incorrect — the issue occurs at compile time (TypingError), not silently at runtime.
- D) Incorrect — `@njit` does not fall back to object mode; it raises an error if nopython compilation fails (unlike `@jit` which can fall back).

---

## Q20 — Combining `imap_unordered` and Task Batching

> **Week reference:** Week 6

```python
from multiprocessing import Pool

def process_batch(items):
    return [item ** 2 for item in items]

data = list(range(1_000_000))
batch_size = 1000
batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

with Pool(8) as pool:
    results_nested = list(pool.imap_unordered(process_batch, batches, chunksize=1))
```

**What problem does this code solve compared to `pool.map(lambda x: x**2, data)`?**

- A) It uses `imap_unordered` which returns results faster by avoiding sorting overhead
- B) It batches 1,000,000 individual tasks into 1,000 batches of 1,000, reducing IPC round-trips from 1,000,000 to 1,000 while using dynamic scheduling across the 1,000 batches
- C) `imap_unordered` is the only correct way to use multiprocessing for large datasets
- D) It avoids the `__main__` guard requirement by using `imap_unordered` instead of `map`

**Answer: B**

`pool.map(f, data)` with 1,000,000 items creates 1,000,000 IPC round-trips (each item pickled, dispatched, computed, result pickled, returned). At ~50 µs IPC overhead per item, that is ~50 seconds of pure overhead for trivial work. Batching into 1,000 groups of 1,000 reduces IPC calls to 1,000, each carrying more work. `imap_unordered` with `chunksize=1` over the 1,000 batches then provides dynamic scheduling — if some batches happen to take longer (though here all are equal), idle workers pick up remaining batches.

- A) Incorrect — avoiding sort overhead is a minor benefit of `imap_unordered`; the primary design goal here is IPC reduction through batching.
- B) Correct — batching reduces IPC round-trips by 1,000×; dynamic scheduling across batches keeps workers busy.
- C) Incorrect — `pool.map` is valid and commonly used; `imap_unordered` is not a requirement.
- D) Incorrect — the `__main__` guard is required for all `multiprocessing.Pool` usage on Windows/macOS regardless of which map variant is used.

---

## Set 3 — Extended Practice

- [Q21 — `pool.starmap` vs Lambda with `spawn`](#q21--poolstarmap-vs-lambda-with-spawn)
- [Q22 — `pool.imap` Return Type and Memory](#q22--poolimap-return-type-and-memory)
- [Q23 — `apply_async` Pattern from `pi_parallel.py`](#q23--apply_async-pattern-from-pi_parallelpy)
- [Q24 — Chunked vs Per-Sample Parallelism](#q24--chunked-vs-per-sample-parallelism)
- [Q25 — `fork` Inherits Parent State](#q25--fork-inherits-parent-state)
- [Q26 — `ProcessPoolExecutor.map` vs `Pool.starmap`](#q26--processpoolexecutormap-vs-poolstarmap)
- [Q27 — Lambda Pickling Failure with `spawn`](#q27--lambda-pickling-failure-with-spawn)
- [Q28 — `RawArray` Shared Memory Pattern](#q28--rawarray-shared-memory-pattern)
- [Q29 — Non-Associative Operator in Reduction](#q29--non-associative-operator-in-reduction)
- [Q30 — `pool.map` with `chunksize` Effect on `time` Output](#q30--poolmap-with-chunksize-effect-on-time-output)

---

## Q21 — `pool.starmap` vs Lambda with `spawn`

> **Week reference:** Week 5

```python
import multiprocessing

def score(x, weight):
    return x * weight

pairs = [(i, 0.5) for i in range(100)]

if __name__ == '__main__':
    with multiprocessing.Pool(4) as pool:
        # Option A:
        results_a = pool.map(lambda p: score(*p), pairs)

        # Option B:
        results_b = pool.starmap(score, pairs)
```

**On macOS (default start method: `spawn`), which option works and which fails?**

- A) Both work — `spawn` can handle lambdas since Python 3.8
- B) Option A fails with `AttributeError: Can't pickle local object '<lambda>'`; Option B works correctly
- C) Option B fails because `starmap` is not compatible with the `spawn` start method
- D) Both fail — multi-argument functions require `apply_async` with explicit `args` tuples on `spawn`

**Answer: B**

- A) Incorrect — lambdas have never been picklable in CPython regardless of Python version. The `spawn` start method requires pickling the task function; lambdas fail at this step.
- B) Correct — `spawn` pickles the function to send it to workers. `lambda p: score(*p)` is anonymous and not importable by name → `AttributeError`. `pool.starmap(score, pairs)` sends the named function `score` (importable from the module) and the argument list; this pickles cleanly.
- C) Incorrect — `starmap` is a standard `multiprocessing.Pool` method that works with all start methods including `spawn`. It is unrelated to start method compatibility.
- D) Incorrect — `starmap` solves multi-argument functions correctly on all start methods. `apply_async` with `args=` tuples also works, but it is per-task and lower-level. Neither requires `spawn` workarounds.

---

## Q22 — `pool.imap` Return Type and Memory

> **Week reference:** Week 6

```python
from multiprocessing import Pool
import sys

def square(x):
    return x * x

if __name__ == '__main__':
    with Pool(4) as pool:
        result = pool.imap(square, range(10_000_000))
        print(type(result))
        print(sys.getsizeof(result))
        first_ten = [next(result) for _ in range(10)]
        print(first_ten)
```

**What does this code print for `type(result)` and `sys.getsizeof(result)`, and what is `first_ten`?**

- A) `<class 'list'>`, ~80 MB (stores all 10M results), `[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]`
- B) `<class 'multiprocessing.pool.IMapIterator'>`, a small fixed size (~100 bytes), `[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]`
- C) `<class 'generator'>`, a small fixed size, `[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]`
- D) `<class 'multiprocessing.pool.IMapIterator'>`, a small fixed size, but `next(result)` raises `StopIteration` — `imap` results must be consumed with `for r in result:`

**Answer: B**

- A) Incorrect — `imap` does not materialise all results into a list. The iterator object itself is tiny regardless of dataset size; this is the key memory advantage over `pool.map`.
- B) Correct — `pool.imap` returns an `IMapIterator` (a lazy iterator). The object itself is small (just state and a queue reference). `next(result)` fetches results one by one in submission order; the first 10 values are `[0, 1, 4, ..., 81]`.
- C) Incorrect — `IMapIterator` is not a plain Python generator; it is a multiprocessing-specific class. However, the memory behaviour (lazy, small footprint) is conceptually similar.
- D) Incorrect — `IMapIterator` supports `next()` via the iterator protocol. Both `next(result)` and `for r in result:` are valid. `StopIteration` is only raised when the iterator is exhausted (after all 10M values have been consumed).

---

## Q23 — `apply_async` Pattern from `pi_parallel.py`

> **Week reference:** Week 5

```python
import random
import multiprocessing

def sample():
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)
    return 1 if x**2 + y**2 <= 1 else 0

if __name__ == '__main__':
    samples = 1_000_000
    n_proc = 10
    pool = multiprocessing.Pool(n_proc)
    results_async = [pool.apply_async(sample) for i in range(samples)]
    hits = sum(r.get() for r in results_async)
    pi = 4.0 * hits / samples
    print(pi)
```

**This code is correct but slow. What is the primary performance problem?**

- A) `random.uniform` is not thread-safe and causes incorrect results under multiprocessing
- B) `pool.apply_async` is called 1,000,000 times — one task per sample — causing 1,000,000 IPC round-trips; the overhead far exceeds the cost of one sample
- C) The `if __name__ == '__main__':` guard prevents tasks from running in worker processes
- D) `sum(r.get() ...)` is sequential and blocks parallel execution of the samples

**Answer: B**

- A) Incorrect — `random.uniform` is not shared between processes; each worker has its own independent RNG state (forked or re-seeded at spawn). There is no thread-safety issue.
- B) Correct — this is the exact anti-pattern the course quiz demonstrates. One `apply_async` call per sample = 1,000,000 IPC messages. Each sample takes ~1 µs to compute; each IPC round-trip costs ~10–100 µs. The overhead is 10–100× the useful work. The fix is `pi_chunked.py`: submit 10 tasks of 100,000 samples each.
- C) Incorrect — the `__main__` guard is present and correctly wraps the pool creation and usage. Workers receive tasks via the queue and execute `sample()` without issues.
- D) Incorrect — `sum(r.get() ...)` does collect results sequentially, but workers run concurrently while the main process iterates. The bottleneck is the 1,000,000-task IPC overhead, not the sequential result collection.

---

## Q24 — Chunked vs Per-Sample Parallelism

> **Week reference:** Week 5

```python
import random, multiprocessing

def sample_multiple(n):
    hits = 0
    for _ in range(n):
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        if x**2 + y**2 <= 1:
            hits += 1
    return hits

if __name__ == '__main__':
    samples = 1_000_000
    n_proc = 10
    chunk = samples // n_proc          # 100_000 per worker
    pool = multiprocessing.Pool(n_proc)
    results = [pool.apply_async(sample_multiple, (chunk,))
               for _ in range(n_proc)]
    hits = sum(r.get() for r in results)
    pi = 4.0 * hits / samples
    print(pi)
```

**Compared to the per-sample version (Q23), why is this code significantly faster?**

- A) `sample_multiple` uses a compiled C extension that bypasses the GIL
- B) Only 10 IPC round-trips are made instead of 1,000,000; each worker does 100,000 samples locally, so the IPC overhead is amortised over 100,000 units of work per trip
- C) Using `range` instead of `random.uniform` in a loop is faster
- D) `pool.apply_async` with a tuple argument is faster than without arguments

**Answer: B**

- A) Incorrect — `sample_multiple` is pure Python, just like `sample`. No C extensions are involved. The GIL is not the relevant factor here (multiprocessing bypasses it regardless).
- B) Correct — 10 IPC calls instead of 1,000,000: overhead drops by 100,000×. Each worker receives one argument (`chunk = 100,000`), loops 100,000 times locally, and returns one integer. Total IPC cost ≈ 10 × 100 µs = 1 ms vs 1,000,000 × 100 µs = 100 seconds for the per-sample version.
- C) Incorrect — both versions call `random.uniform` per sample; the loop structure is essentially identical. The speedup comes from reduced IPC, not loop mechanics.
- D) Incorrect — `apply_async` with or without arguments has essentially the same overhead. The argument tuple `(chunk,)` is trivially small to pickle; this is not the source of the speedup.

---

## Q25 — `fork` Inherits Parent State

> **Week reference:** Week 5

```python
import multiprocessing
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def worker_task(x):
    logger.info(f"processing {x}")
    return x * 2

if __name__ == '__main__':
    multiprocessing.set_start_method('fork')
    with multiprocessing.Pool(4) as pool:
        results = pool.map(worker_task, range(8))
    print(results)
```

**On macOS, what risk does `set_start_method('fork')` introduce in this code?**

- A) `fork` cannot be used on macOS at all; this raises `OSError`
- B) The `logging` module uses internal locks; if a lock is held in the parent at fork time, the child inherits a locked lock with no thread to release it, potentially causing a deadlock on the first `logger.info` call in a worker
- C) `fork` causes all workers to share the same RNG state, making `random` calls non-random
- D) `set_start_method` must be called before importing `logging`; calling it after causes undefined behaviour

**Answer: B**

- A) Incorrect — `fork` is available on macOS (POSIX-compliant). Python emits a `DeprecationWarning` in newer versions but it does not raise `OSError`.
- B) Correct — the `logging` module's `StreamHandler` uses a lock to prevent interleaved output. If a log message is in progress in the parent when `fork` fires, the child inherits the locked lock. The lock's owning thread does not exist in the child, so when the child calls `logger.info`, it tries to acquire the lock and hangs forever. This is a real production issue on macOS and the reason `spawn` is the macOS default.
- C) Incorrect — `fork` does copy the parent's RNG state to all children (so each child starts with the same seed). However this causes correlated random numbers, not a deadlock. It is a correctness issue for Monte Carlo work but not the "risk" the question targets.
- D) Incorrect — `set_start_method` can be called at any point before the first `Pool` is created. The import order does not affect its behaviour.

---

## Q26 — `ProcessPoolExecutor.map` vs `Pool.starmap`

> **Week reference:** Week 5

```python
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

def distance(x, y):
    return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5

points_a = [(0,0), (1,1), (2,2)]
points_b = [(3,3), (4,4), (5,5)]

if __name__ == '__main__':
    # Attempt 1: ProcessPoolExecutor
    with ProcessPoolExecutor(4) as ex:
        results1 = list(ex.map(distance, points_a, points_b))

    # Attempt 2: multiprocessing.Pool
    pairs = list(zip(points_a, points_b))
    with Pool(4) as pool:
        results2 = pool.starmap(distance, pairs)

    print(results1)
    print(results2)
```

**What does each approach print?**

- A) Both print `[4.243, 4.243, 4.243]` — both correctly compute distances between paired points
- B) `results1` prints `[4.243, 4.243, 4.243]`; `results2` raises `TypeError` because `starmap` cannot unpack coordinate tuples
- C) `results1` raises `TypeError`; `results2` prints `[4.243, 4.243, 4.243]`
- D) Both raise `TypeError` — neither `ex.map` nor `pool.starmap` can handle tuple arguments for functions expecting two arguments

**Answer: A**

- A) Correct — `ProcessPoolExecutor.map(func, iter1, iter2)` accepts multiple iterables and passes one element from each as separate positional arguments (like Python's built-in `map`). `distance(points_a[i], points_b[i])` is called for each `i`. `Pool.starmap(func, [(a0,b0), (a1,b1), ...])` unpacks each tuple as positional arguments: `distance(*pair)`. Both call `distance((0,0), (3,3))` etc., computing `sqrt(18) ≈ 4.243`. Both produce identical results.
- B) Incorrect — `pool.starmap(distance, pairs)` with `pairs = [((0,0),(3,3)), ...]` correctly unpacks each 2-tuple as `distance((0,0), (3,3))`. The function receives two tuple arguments, which is what it expects.
- C) Incorrect — `ex.map(distance, points_a, points_b)` passes elements from both iterables as separate arguments. This is the standard multi-iterable `map` protocol and works correctly.
- D) Incorrect — both approaches are designed for multi-argument functions. `ex.map` with multiple iterables and `pool.starmap` with a list of argument tuples both work correctly.

---

## Q27 — Lambda Pickling Failure with `spawn`

> **Week reference:** Week 5

```python
import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(4) as pool:
        results = pool.map(lambda x: x**3, range(10))
    print(results)
```

**What error does this raise, and on which line does it occur?**

- A) `RuntimeError: context has already been set` — `set_start_method('spawn')` cannot be called inside `if __name__ == '__main__':`
- B) `AttributeError: Can't pickle local object '<lambda>'` — raised when `pool.map` tries to pickle the lambda to send it to workers
- C) `PicklingError: Can't pickle <class 'function'>` — raised when the pool is created with `spawn`
- D) The code runs correctly and prints `[0, 1, 8, 27, 64, 125, 216, 343, 512, 729]`

**Answer: B**

- A) Incorrect — `set_start_method` can be called inside `if __name__ == '__main__':` as long as it is called before any Pool is created. No `RuntimeError` here.
- B) Correct — `pool.map` attempts to pickle the `lambda x: x**3` to serialize it into the task queue for workers. Python's `pickle` cannot serialize lambdas because they are anonymous and have no importable name. The error is `AttributeError: Can't pickle local object '<lambda>'`, raised at the `pool.map(...)` call.
- C) Incorrect — the pool itself is created successfully; workers are spawned without issue. The pickling error occurs specifically when the task function (the lambda) is serialized, not at pool creation time.
- D) Incorrect — this would be the correct output if the code worked. But with `spawn` and a lambda, it always fails with a pickling error.

---

## Q28 — `RawArray` Shared Memory Pattern

> **Week reference:** Week 6

```python
import ctypes
import multiprocessing as mp
import numpy as np

def init(shared_):
    global shared_arr
    shared_arr = shared_

def add_one(i):
    arr = np.frombuffer(shared_arr, dtype='float32')
    arr[i] += 1.0

if __name__ == '__main__':
    data = np.array([10.0, 20.0, 30.0, 40.0], dtype='float32')
    raw = mp.RawArray(ctypes.c_float, data.size)
    buf = np.frombuffer(raw, dtype='float32')
    np.copyto(buf, data)

    with mp.Pool(4, initializer=init, initargs=(raw,)) as pool:
        pool.map(add_one, range(4))

    final = np.frombuffer(raw, dtype='float32')
    print(final)
```

**What does `final` print?**

- A) `[10. 20. 30. 40.]` — workers operate on their own copies; the `RawArray` in the main process is unchanged
- B) `[11. 21. 31. 41.]` — workers modify the shared `RawArray` directly; changes are visible in the main process
- C) `[14. 24. 34. 44.]` — each of 4 workers adds 1.0 to all 4 elements
- D) A random result — concurrent writes to shared memory without locks cause data races

**Answer: B**

- A) Incorrect — `RawArray` is a shared memory object. Workers do not receive copies; they access the same physical memory pages as the main process. Changes made by workers are immediately visible to the main process.
- B) Correct — each worker is assigned a unique index `i` (0, 1, 2, 3) via `pool.map(add_one, range(4))`. Worker 0 adds 1 to index 0, worker 1 to index 1, etc. Since each worker writes to a distinct array element, there are no data races. `final = [11., 21., 31., 41.]`.
- C) Incorrect — each worker receives a single index `i` and only modifies `arr[i]`. Worker 0 only touches index 0, not all 4 elements.
- D) Incorrect — a data race would occur only if multiple workers wrote to the same index simultaneously. Here each index is assigned to exactly one worker (`range(4)` distributes indices 0–3 one per worker), so no race condition exists.

---

## Q29 — Non-Associative Operator in Reduction

> **Week reference:** Week 6

```python
def abssum(x, y):
    return abs(x + y)

# Tree reduction applied to [-3, 2, -5, 4]:
# Round 1 (parallel): abssum(-3, 2) = 1,  abssum(-5, 4) = 1
# Round 2 (parallel): abssum(1, 1) = 2

# Serial left-to-right application:
# abssum(abssum(abssum(-3, 2), -5), 4)
# = abssum(abssum(1, -5), 4)
# = abssum(4, 4) = 8    ← different from 2!
```

**What does this demonstrate about using `abssum` as a parallel reduction operator?**

- A) The tree reduction is correct; the serial version is wrong because it uses a different ordering
- B) `abssum` is not associative: different groupings give different results (2 vs 8), making it invalid for parallel reduction trees that apply the operator in unpredictable groupings
- C) Both results are wrong; the correct answer is `abs(-3+2-5+4) = 2`, which equals the tree result, so the tree reduction is valid
- D) The parallel tree is valid because it always produces the minimum possible result, which is desirable for numerical stability

**Answer: B**

- A) Incorrect — a valid combining operator must give the same result regardless of grouping. "Different ordering" is exactly the problem: a correct operator (like `+`) gives the same answer regardless of grouping order. `abssum` does not.
- B) Correct — the tree gives 2; serial left-to-right gives 8; other groupings give other values. `abssum` is not associative: `abs(abs(a+b)+c) ≠ abs(a+abs(b+c))` in general. Parallel reduction trees apply the operator in a tree structure that changes grouping; a non-associative operator produces incorrect and unpredictable results.
- C) Incorrect — `abs(-3+2-5+4) = abs(-2) = 2`. The tree result happens to equal this, but that is coincidental for this specific input. For other inputs the tree result would differ. The operator's non-associativity means you cannot rely on the result being correct.
- D) Incorrect — there is no mathematical guarantee that a parallel reduction tree with `abssum` produces a minimum, maximum, or any other useful value. The result is simply unpredictable and incorrect.

---

## Q30 — `pool.map` with `chunksize` Effect on `time` Output

> **Week reference:** Week 6

```python
# Scenario A: chunksize=1 (one task per IPC call)
# 10,000 tasks each taking 1ms
# pool.map(func, range(10_000), chunksize=1)
# Result: real=5.2s, user=19.8s

# Scenario B: chunksize=250 (40 IPC calls total)
# same 10,000 tasks, same 1ms each
# pool.map(func, range(10_000), chunksize=250)
# Result: real=2.6s, user=10.2s
```

**Why does Scenario B have lower `user` time than Scenario A, even though both do the same computation?**

- A) `chunksize=250` forces workers to run faster by using SIMD instructions
- B) Scenario A makes 10,000 IPC round-trips (each with pickle/unpickle overhead); Scenario B makes 40 round-trips. The IPC overhead is CPU work counted in `user` time; reducing IPC calls reduces `user` time by removing that overhead
- C) `chunksize=1` causes context switching which is counted as `sys` time, not `user` time
- D) Both `user` times should be identical; the numbers given are physically impossible

**Answer: B**

- A) Incorrect — `chunksize` controls scheduling granularity, not instruction-level optimisation. No SIMD or compiler effects are involved.
- B) Correct — IPC overhead (pickling, queue writes, queue reads, unpickling) is Python/C code that runs on the CPU and is counted as `user` time. Scenario A: 10,000 round-trips × ~100 µs IPC each = ~1 second of pure IPC overhead added to `user`. Scenario B: 40 round-trips × ~100 µs = ~4 ms of IPC overhead — negligible. The reduction in IPC work reduces `user` time proportionally. Both real and user time decrease because idle-free workers do less overhead work per unit of useful computation.
- C) Incorrect — IPC queue operations involve kernel calls (pipe reads/writes) which do add `sys` time, but the dominant cost of pickling and unpickling is pure `user`-space CPU work. Context switching overhead between the pool workers themselves is also a `sys`-time effect, but the primary explanation is IPC pickle overhead in user space.
- D) Incorrect — the numbers are physically plausible. A 2× difference in `user` time between `chunksize=1` and `chunksize=250` for 1ms tasks is consistent with IPC overhead dominating at fine granularity.

---
