# Parallelism Strategy — MCQ Practice

> Topics: GIL, threading vs multiprocessing, static vs dynamic scheduling, time command output.
> Exam frequency: **Every exam**.

---

## Q1 — Pure Python Loop: Threading vs Multiprocessing

> **Week reference:** Week 5

You have a CPU-bound function that runs a pure Python loop computing a large sum (no NumPy or Numba). You want to parallelise it across 4 workers. Which approach gives a real speedup?

- A) `ThreadPoolExecutor(max_workers=4)` — threads share memory and avoid spawn overhead
- B) `multiprocessing.Pool(4)` — each process has its own GIL, enabling true parallelism
- C) `threading.Thread` with a shared counter — avoids the GIL by using atomic operations
- D) Either threading or multiprocessing — both bypass the GIL for CPU-bound work

**Answer: B**

- A) Incorrect — the GIL prevents true parallel execution of pure Python code in threads; only one thread runs at a time
- B) Correct — separate processes each have their own GIL, so all 4 can execute Python bytecode simultaneously
- C) Incorrect — Python's GIL still applies to all threads in the same process; there are no truly atomic Python operations
- D) Incorrect — threading does NOT bypass the GIL for pure Python; only multiprocessing (separate processes) does

---

## Q2 — NumPy `.sum()` in a ThreadPool

> **Week reference:** Week 5

A function calls `numpy.sum()` on a large array. You want to parallelise multiple independent calls. Which is true?

- A) Use `multiprocessing.Pool` — NumPy operations hold the GIL, so threads cannot run in parallel
- B) Use `ThreadPoolExecutor` — NumPy releases the GIL during its C-level operations, allowing true thread parallelism
- C) NumPy never releases the GIL, so neither threading nor multiprocessing can speed up `numpy.sum()`
- D) Use `multiprocessing.Pool` because threads are always slower than processes for array operations

**Answer: B**

- A) Incorrect — NumPy releases the GIL during its C-level operations, so threads CAN run in parallel
- B) Correct — NumPy's internal C code releases the GIL, enabling true parallel execution with threads and avoiding costly process-spawn overhead
- C) Incorrect — NumPy is specifically designed to release the GIL during computation
- D) Incorrect — for NumPy workloads, ThreadPool avoids process-spawn/IPC overhead and works fine since the GIL is released

---

## Q3 — Numba `@jit(nogil=True)` and ThreadPool

> **Week reference:** Week 6

You have a Numba-compiled function decorated with `@jit(nopython=True, nogil=True)`. You want to parallelise multiple independent calls. Which is the recommended approach?

- A) `multiprocessing.Pool` — Numba JIT functions always hold the GIL even with `nogil=True`
- B) `ThreadPoolExecutor` — `nogil=True` causes Numba to release the GIL during JIT execution, enabling true thread parallelism
- C) Neither threading nor multiprocessing helps — Numba JIT runs on a single core only
- D) `ThreadPoolExecutor` — but only works if you also set `nopython=False`

**Answer: B**

- A) Incorrect — `nogil=True` explicitly instructs Numba to release the GIL during compiled function execution
- B) Correct — `nogil=True` releases the GIL in the Numba-compiled code, so a ThreadPool achieves true parallel execution with low overhead
- C) Incorrect — Numba JIT functions with `nogil=True` can run in parallel across threads
- D) Incorrect — `nopython=True` is required for `nogil=True` to work; `nopython=False` would be a step backward

---

## Q4 — Parallelising `simulate_ball` with Sequential Internal Steps

> **Week reference:** Week 5

`simulate_ball(n_steps)` is a pure Python function that runs `n_steps` sequential simulation steps — each step depends on the previous. You want to run 1,000 independent simulations. What is the correct parallelisation strategy?

- A) Use `ThreadPool` to parallelise the steps inside a single `simulate_ball` call
- B) Use `multiprocessing.Pool.map` to run multiple independent `simulate_ball` calls in parallel
- C) Use `asyncio` to overlap the sequential steps within one simulation
- D) The sequential dependency means no parallelism is possible at all

**Answer: B**

- A) Incorrect — the steps inside `simulate_ball` are sequentially dependent; you cannot parallelise within a single call
- B) Correct — the 1,000 simulations are independent of each other; multiprocessing can run them in parallel across processes
- C) Incorrect — `asyncio` is for I/O-bound concurrency; sequential CPU-bound steps cannot be overlapped this way
- D) Incorrect — even though internal steps are sequential, the 1,000 separate simulation runs are fully independent and can be parallelised

---

## Q5 — Static Scheduling and Uniform Task Times

> **Week reference:** Week 6

You are parallelising 800 independent tasks using `pool.map` with a large `chunksize`. Each task takes approximately the same time (standard deviation < 1% of mean). What scheduling strategy is this, and is it appropriate?

- A) Dynamic scheduling; appropriate because it minimises IPC overhead
- B) Static scheduling; appropriate because uniform task times mean each worker finishes at roughly the same time with minimal scheduling overhead
- C) Static scheduling; inappropriate because workers may finish at different times causing load imbalance
- D) Dynamic scheduling; inappropriate because it causes excessive context switching for uniform tasks

**Answer: B**

- A) Incorrect — large `chunksize` with `pool.map` is static scheduling (work is pre-divided); dynamic uses `imap_unordered` with chunksize=1
- B) Correct — static scheduling pre-divides work and has low overhead; with uniform task times there is negligible load imbalance
- C) Incorrect — with near-identical task times, static scheduling produces very balanced loads; imbalance is only a problem with high variance
- D) Incorrect — `pool.map` with large chunksize is static, not dynamic

---

## Q6 — Dynamic Scheduling and Variable Task Times

> **Week reference:** Week 6

You have tasks whose runtimes vary significantly (some tasks take 10× longer than others). You use `pool.imap_unordered(func, tasks, chunksize=1)`. What is the advantage of this approach?

- A) Static pre-division is better because it avoids repeated IPC calls to the scheduler
- B) Dynamic scheduling prevents load imbalance: fast workers pick up new tasks as soon as they finish, rather than sitting idle
- C) `imap_unordered` speeds up individual task execution by using multiple CPU cores per task
- D) Dynamic scheduling is always faster than static regardless of task variance

**Answer: B**

- A) Incorrect — static scheduling with highly variable tasks leaves fast workers idle while slow workers are still running
- B) Correct — `imap_unordered` with chunksize=1 implements dynamic scheduling: each worker fetches the next task upon completion, balancing load automatically
- C) Incorrect — `imap_unordered` distributes tasks across workers; it does not use multiple cores for a single task
- D) Incorrect — for uniform task times, static scheduling has lower overhead; dynamic is only preferable when variance is high

---

## Q7 — Choosing Scheduling Based on Kernel Variance

> **Week reference:** Week 6

You benchmark two kernel functions:
- **Kernel A**: mean runtime 200ms, std = 40ms (high variance)
- **Kernel B**: mean runtime 200ms, std = 0.05ms (negligible variance)

Which kernel benefits most from dynamic scheduling, and why?

- A) Kernel B — low variance means workers always finish together, so dynamic scheduling keeps them synchronised
- B) Kernel A — high variance means some workers finish much earlier than others; dynamic scheduling reassigns them immediately
- C) Both benefit equally — dynamic scheduling always outperforms static
- D) Neither — dynamic scheduling only helps when tasks have different mean runtimes, not different variances

**Answer: B**

- A) Incorrect — if all workers finish together (low variance), static scheduling already achieves perfect balance with less overhead
- B) Correct — with std=40ms (20% of mean), some tasks take 2× longer than others; dynamic scheduling ensures fast workers are not idle while slow tasks linger
- C) Incorrect — for Kernel B's near-uniform times, static scheduling's lower IPC overhead makes it preferable
- D) Incorrect — variance (not just mean differences) is the key factor; high variance within a pool of tasks causes load imbalance under static scheduling

---

## Q8 — Interpreting `time` Command Output

> **Week reference:** Week 5

Running `time python script.py` on a single-threaded script gives: `real 12.0s, user 12.0s, sys 0.1s`. You refactor it to use `multiprocessing.Pool(4)` and run again on a 4-core machine: `real 3.2s, user 12.4s, sys 0.3s`. Why does `user` time increase while `real` time decreases?

- A) `user` time is a bug in the `time` command; it should decrease proportionally to `real`
- B) `real` is wall-clock time (decreases with parallelism); `user` is the sum of CPU time across all cores (stays roughly constant or increases with overhead)
- C) `user` time includes I/O wait; parallelism introduces more I/O, increasing `user`
- D) `real` and `user` both decrease with parallelism; the increase in `user` indicates a bug in the code

**Answer: B**

- A) Incorrect — this is the expected and correct behaviour of the `time` command with parallelism
- B) Correct — `real` is elapsed wall-clock time; `user` accumulates CPU seconds across all processes, so 4 parallel processes each doing ~3s of CPU work contribute ~12s total `user` time
- C) Incorrect — `user` time is CPU time spent in user-space code; I/O wait is reflected in `sys` and idle time, not `user`
- D) Incorrect — `user` time increasing when using multiple processes is completely normal and expected behaviour

---

## Q9 — `Pool()` Default Workers and HPC Oversubscription

> **Week reference:** Week 5

You submit an LSF job with `#BSUB -n 4` (requesting 4 cores) and your Python script calls `multiprocessing.Pool()` with no argument. How many worker processes are created, and what is the risk?

- A) 4 workers — `Pool()` reads the LSF `#BSUB -n` directive automatically
- B) `os.cpu_count()` workers (e.g., 32 on a 32-core node) — this oversubscribes the allocated cores, slowing all jobs on that node
- C) 1 worker — `Pool()` defaults to single-process mode in batch environments
- D) `os.cpu_count()` workers, but LSF automatically limits execution to the 4 allocated cores

**Answer: B**

- A) Incorrect — `Pool()` calls `os.cpu_count()` and has no knowledge of LSF job allocations
- B) Correct — `Pool()` defaults to `os.cpu_count()`, which returns the node's total CPUs; using more than allocated processes starves other users' jobs sharing the node
- C) Incorrect — `Pool()` with no argument uses `os.cpu_count()`, not 1
- D) Incorrect — LSF allocates cores but does not prevent a process from using more; it is the programmer's responsibility to limit workers to the allocation

---

## Q10 — The `if __name__ == '__main__':` Guard

> **Week reference:** Week 5

On macOS, a Python script using `multiprocessing.Pool` runs correctly from the command line but crashes with a `RuntimeError` about recursive spawning when imported as a module. What is the fix?

- A) Switch from `multiprocessing` to `threading` — threads do not require this guard
- B) Wrap all `Pool` usage inside `if __name__ == '__main__':` to prevent the spawned child processes from re-executing the pool creation code
- C) Add `multiprocessing.set_start_method('fork')` — the `spawn` method causes recursive imports
- D) The error is unrelated to multiprocessing; it is caused by a circular import

**Answer: B**

- A) Incorrect — while threads avoid this issue (no new processes), it does not fix the multiprocessing spawning problem
- B) Correct — on macOS/Windows, `multiprocessing` uses `spawn` by default; child processes re-import the script, and without the guard they re-execute pool creation, causing infinite recursion
- C) Incorrect — using `fork` on macOS can cause deadlocks (e.g., with Objective-C runtimes); the correct fix is the `__name__` guard, not changing the start method
- D) Incorrect — the error is specifically caused by unguarded pool creation being re-executed in spawned child processes

---

## Q11 — Task Granularity and IPC Overhead

> **Week reference:** Week 5

You have 1,000,000 independent tasks, each taking approximately 1 microsecond. You submit them all to `pool.map(func, range(1_000_000))`. The parallel version is slower than a serial loop. What is the most likely cause and fix?

- A) The GIL is blocking all threads; fix by switching to `ThreadPoolExecutor`
- B) IPC overhead per task (pickling, queue communication) dominates the 1µs task time; fix by chunking into fewer, larger tasks (e.g., 8 tasks of 125,000 operations each)
- C) 1,000,000 processes are being spawned; fix by using `chunksize=1000`
- D) NumPy is not being used; fix by vectorising with `numpy.vectorize`

**Answer: B**

- A) Incorrect — the problem is not the GIL but task granularity; switching to threads would have the same overhead problem
- B) Correct — each `pool.map` call incurs IPC overhead (pickling arguments, queue operations, result retrieval) that far exceeds 1µs; batching into 8 tasks of 125,000 ops each amortises this overhead
- C) Incorrect — `Pool` reuses a fixed number of worker processes; it does not spawn 1,000,000 processes; but the per-task IPC cost still dominates
- D) Incorrect — the issue is parallelism overhead, not vectorisation; chunking the tasks is the correct solution

---

## Q12 — I/O-Bound Tasks and the GIL

> **Week reference:** Week 5

A script reads 500 CSV files from disk sequentially, taking 60 seconds total. Each file read is I/O-bound (the disk is the bottleneck, not the CPU). Which parallelisation strategy is appropriate?

- A) `multiprocessing.Pool` only — the GIL blocks threads even during disk I/O
- B) `ThreadPoolExecutor` is appropriate — the GIL is released during I/O operations, so threads can overlap waiting for disk reads
- C) Neither threading nor multiprocessing helps — disk I/O is inherently sequential
- D) Only `asyncio` can overlap I/O operations; threads and processes cannot

**Answer: B**

- A) Incorrect — the GIL is released during I/O system calls, so threads CAN run in parallel during file reads
- B) Correct — Python releases the GIL during blocking I/O (read/write syscalls), allowing multiple threads to overlap disk waits; `ThreadPoolExecutor` is ideal and avoids process-spawn overhead
- C) Incorrect — overlapping I/O waits across threads (or processes) reduces total wall-clock time by keeping the disk pipeline full
- D) Incorrect — `asyncio` with non-blocking I/O is one option, but `ThreadPoolExecutor` is simpler and equally effective for synchronous file reads

---

## Q13 — Parallelising `@jit(nogil=True)` Across Outer Loop

> **Week reference:** Week 6

`simulate_single(params)` is compiled with `@jit(nopython=True, nogil=True)`. It runs sequential internal steps but you need to call it 10,000 times with different parameters. What is the correct strategy?

- A) Use `multiprocessing.Pool` to run 10,000 parallel calls — Numba JIT functions require separate processes
- B) Use `ThreadPoolExecutor` to parallelise the outer loop of 10,000 independent calls — `nogil=True` releases the GIL so threads execute truly in parallel
- C) Parallelise the internal steps of `simulate_single` by passing a thread pool into the function
- D) Use `pool.map` with `chunksize=1` — Numba automatically detects multiprocessing and optimises accordingly

**Answer: B**

- A) Incorrect — while multiprocessing would work, it is unnecessarily heavy; `nogil=True` means ThreadPool achieves true parallelism with much lower overhead
- B) Correct — the 10,000 calls are independent (outer loop), and `nogil=True` means threads are not serialised by the GIL; `ThreadPoolExecutor` parallelises the outer loop efficiently
- C) Incorrect — the internal steps are sequentially dependent on each other; you cannot parallelise within a single `simulate_single` call
- D) Incorrect — `pool.map` creates multiprocessing workers; this works but is overkill; the `nogil=True` flag specifically enables thread-based parallelism without process-spawn overhead

---
