# Parallelism Strategy — Code-Based MCQ Practice

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
