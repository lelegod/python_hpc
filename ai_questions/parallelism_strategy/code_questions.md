# Parallelism Strategy — Code-Based MCQ Practice

> Format: Each question shows Python parallelism code or shell output to analyse.
> Exam frequency: **Every exam**.

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
