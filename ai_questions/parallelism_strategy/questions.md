# Parallelism Strategy — MCQ Practice

> Topics: GIL, threading vs multiprocessing, static vs dynamic scheduling, time command output.
> Exam frequency: **Every exam**.

---

## Q1 — Pure Python Loop: Threading vs Multiprocessing

> **Week reference:** Week 5
> **Mental Model:** The GIL is a per-interpreter lock, not a per-thread lock. Threads share the GIL and take turns; processes each own one. For pure Python CPU-bound code, threads are serialized — only multiprocessing escapes the GIL.

You have a CPU-bound function that runs a pure Python loop computing a large sum (no NumPy or Numba). You want to parallelise it across 4 workers. Which approach gives a real speedup?

- A) `ThreadPoolExecutor(max_workers=4)` — threads share memory and avoid spawn overhead
- B) `multiprocessing.Pool(4)` — each process has its own GIL, enabling true parallelism
- C) `threading.Thread` with a shared counter — avoids the GIL by using atomic operations
- D) Either threading or multiprocessing — both bypass the GIL for CPU-bound work

**Answer: B**

- A) Incorrect — the GIL prevents true parallel execution of pure Python code in threads. All 4 threads live in the same interpreter and compete for the same lock, so only one thread runs Python bytecode at a time. Wall-clock time barely improves over serial.
- B) Correct — separate processes each have their own Python interpreter and their own GIL, so all 4 can execute Python bytecode simultaneously. The spawn overhead is a one-time cost; it pays off for long-running CPU-bound tasks.
- C) Incorrect — Python's GIL applies to all threads in the same process regardless of how they communicate. There are no truly atomic Python-level operations that bypass the GIL; even incrementing a shared counter requires the GIL.
- D) Incorrect — threading does NOT bypass the GIL for pure Python. Only multiprocessing (separate processes) does. This is the most common parallelism misconception tested on exams.

---

## Q2 — NumPy `.sum()` in a ThreadPool

> **Week reference:** Week 5
> **Mental Model:** NumPy's C-extension code releases the GIL before entering tight numerical loops, so threads genuinely run in parallel. The decision rule: does the heavy work happen in C (NumPy/Numba)? If yes, ThreadPool works. If it's pure Python bytecode? Use multiprocessing.

A function calls `numpy.sum()` on a large array. You want to parallelise multiple independent calls. Which is true?

- A) Use `multiprocessing.Pool` — NumPy operations hold the GIL, so threads cannot run in parallel
- B) Use `ThreadPoolExecutor` — NumPy releases the GIL during its C-level operations, allowing true thread parallelism
- C) NumPy never releases the GIL, so neither threading nor multiprocessing can speed up `numpy.sum()`
- D) Use `multiprocessing.Pool` because threads are always slower than processes for array operations

**Answer: B**

- A) Incorrect — NumPy releases the GIL during its C-level operations, so threads CAN run in parallel. The premise is backwards. Using multiprocessing here would work but wastes resources copying large arrays between processes.
- B) Correct — NumPy's internal C code (written in C/Fortran) releases the GIL before entering computation loops, enabling true parallel execution with threads. ThreadPool also avoids the costly process-spawn and array-serialization overhead of multiprocessing.
- C) Incorrect — NumPy is specifically designed to release the GIL during computation. This is a core reason NumPy + threads is a recommended pattern for numerical parallelism in Python.
- D) Incorrect — for NumPy workloads, ThreadPool avoids process-spawn overhead and IPC array copying. Processes must pickle/unpickle large arrays to send them across process boundaries, which can dominate compute time for large arrays.

---

## Q3 — Numba `@jit(nogil=True)` and ThreadPool

> **Week reference:** Week 6
> **Mental Model:** `nogil=True` is the explicit GIL release switch for Numba JIT code. Once set, a ThreadPool gives true parallelism with near-zero overhead vs multiprocessing. Remember: `nogil=True` requires `nopython=True` — object mode cannot release the GIL.

You have a Numba-compiled function decorated with `@jit(nopython=True, nogil=True)`. You want to parallelise multiple independent calls. Which is the recommended approach?

- A) `multiprocessing.Pool` — Numba JIT functions always hold the GIL even with `nogil=True`
- B) `ThreadPoolExecutor` — `nogil=True` causes Numba to release the GIL during JIT execution, enabling true thread parallelism
- C) Neither threading nor multiprocessing helps — Numba JIT runs on a single core only
- D) `ThreadPoolExecutor` — but only works if you also set `nopython=False`

**Answer: B**

- A) Incorrect — `nogil=True` explicitly instructs Numba to release the GIL during compiled function execution. The whole point of the flag is to enable thread-based parallelism. If it still held the GIL, the flag would be meaningless.
- B) Correct — `nogil=True` releases the GIL in the Numba-compiled code, so a ThreadPool achieves true parallel execution with low overhead. No array copying across process boundaries, no spawn cost — just threads running compiled machine code in parallel.
- C) Incorrect — Numba JIT functions with `nogil=True` can absolutely run in parallel across threads. Numba compiles to native machine code that runs on any available core.
- D) Incorrect — `nopython=True` is required for `nogil=True` to work; `nopython=False` (object mode) cannot release the GIL because it still calls Python C-API functions. Setting `nopython=False` would be a step backward that breaks `nogil`.

---

## Q4 — Parallelising `simulate_ball` with Sequential Internal Steps

> **Week reference:** Week 5
> **Mental Model:** Parallelism lives at the level where tasks are independent. Internal sequential dependencies prevent intra-call parallelism, but 1,000 independent calls are fully parallel at the call level. Find the right granularity — don't try to parallelize within a sequential dependency chain.

`simulate_ball(n_steps)` is a pure Python function that runs `n_steps` sequential simulation steps — each step depends on the previous. You want to run 1,000 independent simulations. What is the correct parallelisation strategy?

- A) Use `ThreadPool` to parallelise the steps inside a single `simulate_ball` call
- B) Use `multiprocessing.Pool.map` to run multiple independent `simulate_ball` calls in parallel
- C) Use `asyncio` to overlap the sequential steps within one simulation
- D) The sequential dependency means no parallelism is possible at all

**Answer: B**

- A) Incorrect — the steps inside `simulate_ball` are sequentially dependent (step n+1 uses step n's result). Threading inside a single call cannot parallelize a data dependency chain; it would require serial execution anyway.
- B) Correct — the 1,000 simulations are completely independent of each other (each has its own state). multiprocessing.Pool.map distributes these independent calls across worker processes, each running their own sequential simulation in parallel with others.
- C) Incorrect — `asyncio` is for I/O-bound concurrency where tasks can yield while waiting for I/O. Sequential CPU-bound computation steps with dependencies cannot be overlapped via async/await.
- D) Incorrect — even though internal steps within one simulation are sequential, the 1,000 separate simulation runs are fully independent and can be parallelised. Amdahl's law applies to the serial fraction within each task, not to independent tasks.

---

## Q5 — Static Scheduling and Uniform Task Times

> **Week reference:** Week 6
> **Mental Model:** Static = divide the work upfront into equal chunks. Works perfectly when all chunks take the same time (low stddev). Dynamic = workers pull new tasks on demand. Only pays off when task times vary widely — otherwise the per-task IPC cost exceeds the load-balancing benefit.

You are parallelising 800 independent tasks using `pool.map` with a large `chunksize`. Each task takes approximately the same time (standard deviation < 1% of mean). What scheduling strategy is this, and is it appropriate?

- A) Dynamic scheduling; appropriate because it minimises IPC overhead
- B) Static scheduling; appropriate because uniform task times mean each worker finishes at roughly the same time with minimal scheduling overhead
- C) Static scheduling; inappropriate because workers may finish at different times causing load imbalance
- D) Dynamic scheduling; inappropriate because it causes excessive context switching for uniform tasks

**Answer: B**

- A) Incorrect — large `chunksize` with `pool.map` is static scheduling (work is pre-divided upfront before any worker starts). Dynamic scheduling uses `imap_unordered` with chunksize=1 so workers pull individual tasks as they finish.
- B) Correct — static scheduling pre-divides work once and has low per-task IPC overhead. With std < 1% of mean, all workers take nearly identical time per chunk, finishing at roughly the same time. Idle time at the end is negligible, making static optimal here.
- C) Incorrect — load imbalance from static scheduling requires high variance in task times. With std < 1%, the worst-case imbalance is < 1% of mean time — completely negligible. Static is the right choice.
- D) Incorrect — `pool.map` with large chunksize is static, not dynamic. Context switching is also a thread concern; multiprocessing workers are separate processes and don't context-switch between each other in the same way.

---

## Q6 — Dynamic Scheduling and Variable Task Times

> **Week reference:** Week 6
> **Mental Model:** High variance in task times = some workers finish early and sit idle under static scheduling. Dynamic scheduling (chunksize=1) keeps all workers busy by having them pull new tasks on demand. The cost is more IPC calls; the benefit is no idle workers.

You have tasks whose runtimes vary significantly (some tasks take 10× longer than others). You use `pool.imap_unordered(func, tasks, chunksize=1)`. What is the advantage of this approach?

- A) Static pre-division is better because it avoids repeated IPC calls to the scheduler
- B) Dynamic scheduling prevents load imbalance: fast workers pick up new tasks as soon as they finish, rather than sitting idle
- C) `imap_unordered` speeds up individual task execution by using multiple CPU cores per task
- D) Dynamic scheduling is always faster than static regardless of task variance

**Answer: B**

- A) Incorrect — static scheduling with highly variable tasks causes severe load imbalance. A worker that gets all the 10× tasks finishes long after others are idle. The IPC savings from static scheduling are outweighed by idle worker time.
- B) Correct — `imap_unordered` with chunksize=1 implements dynamic scheduling: each worker fetches the next available task upon completing its current one. Fast workers automatically take more tasks, slow tasks don't leave others idle. Result: near-perfect utilization.
- C) Incorrect — `imap_unordered` distributes tasks across workers (one task per worker at a time); it does not split a single task across multiple cores. Multi-core within a single task requires different techniques (e.g., Numba parallel or OpenMP).
- D) Incorrect — for uniform task times, static scheduling has lower overhead because it avoids the per-task IPC cost of workers querying for new work. Dynamic is preferable specifically when variance is high, not universally.

---

## Q7 — Choosing Scheduling Based on Kernel Variance

> **Week reference:** Week 6
> **Mental Model:** The scheduling decision is a variance test. High stddev / mean ratio (coefficient of variation) → dynamic. Low CV → static. A CV of 20% (Kernel A: 40ms std on 200ms mean) means some tasks take 2× the mean — those stragglers idle all other workers under static scheduling.

You benchmark two kernel functions:
- **Kernel A**: mean runtime 200ms, std = 40ms (high variance)
- **Kernel B**: mean runtime 200ms, std = 0.05ms (negligible variance)

Which kernel benefits most from dynamic scheduling, and why?

- A) Kernel B — low variance means workers always finish together, so dynamic scheduling keeps them synchronised
- B) Kernel A — high variance means some workers finish much earlier than others; dynamic scheduling reassigns them immediately
- C) Both benefit equally — dynamic scheduling always outperforms static
- D) Neither — dynamic scheduling only helps when tasks have different mean runtimes, not different variances

**Answer: B**

- A) Incorrect — if all workers finish together (low variance), static scheduling already achieves perfect balance with no IPC overhead for task dispatch. Dynamic scheduling adds unnecessary overhead when Kernel B tasks are essentially identical in cost.
- B) Correct — with std=40ms on a 200ms mean (CV = 20%), the longest tasks can take ~360ms (mean + 4σ) while average tasks take 200ms. Under static scheduling, workers with slow tasks become stragglers; dynamic scheduling reassigns finishing workers immediately, cutting idle time.
- C) Incorrect — for Kernel B's near-zero variance (CV = 0.025%), static scheduling produces essentially perfect balance. Adding dynamic IPC overhead would make it slower, not faster.
- D) Incorrect — variance within a pool of same-mean tasks is exactly the key factor. High variance within identically-defined tasks (not just between different tasks) causes load imbalance. Kernel A and B have the same mean; only variance differs.

---

## Q8 — Interpreting `time` Command Output

> **Week reference:** Week 5
> **Mental Model:** real = wall clock (what you experience); user = sum of CPU seconds across all cores. With N parallel workers each doing T seconds of CPU work: real ≈ T, user ≈ N×T. User time staying flat or rising while real drops is the signature of successful parallelism, not a bug.

Running `time python script.py` on a single-threaded script gives: `real 12.0s, user 12.0s, sys 0.1s`. You refactor it to use `multiprocessing.Pool(4)` and run again on a 4-core machine: `real 3.2s, user 12.4s, sys 0.3s`. Why does `user` time increase while `real` time decreases?

- A) `user` time is a bug in the `time` command; it should decrease proportionally to `real`
- B) `real` is wall-clock time (decreases with parallelism); `user` is the sum of CPU time across all cores (stays roughly constant or increases with overhead)
- C) `user` time includes I/O wait; parallelism introduces more I/O, increasing `user`
- D) `real` and `user` both decrease with parallelism; the increase in `user` indicates a bug in the code

**Answer: B**

- A) Incorrect — this is the expected and correct behaviour of the `time` command with parallelism. user > real is the normal signature of multi-core execution, not a bug.
- B) Correct — `real` is elapsed wall-clock time: 12s → 3.2s, showing ~3.75× speedup from 4 workers. `user` accumulates CPU seconds across all processes: 4 parallel processes each doing ~3s of CPU work contribute ~12s total user time, plus a small overhead (12.4s > 12.0s due to IPC and scheduling).
- C) Incorrect — `user` time is CPU time spent in user-space code. I/O wait does not count toward user time; it appears as idle time or in `sys` (kernel time). More I/O would increase `sys`, not `user`.
- D) Incorrect — `user` time increasing when using multiple processes is completely normal and expected. It would be a bug if user time dramatically exceeded ncores × real time (indicating runaway processes), but 12.4s ≈ 4 × 3.2s is exactly right.

---

## Q9 — `Pool()` Default Workers and HPC Oversubscription

> **Week reference:** Week 5
> **Mental Model:** Pool() is ignorant of LSF, SLURM, or any job scheduler. It calls os.cpu_count() which returns the physical node's total cores — often 32-64 on an HPC node. Always pass an explicit worker count matching your #BSUB -n allocation to avoid starving other users.

You submit an LSF job with `#BSUB -n 4` (requesting 4 cores) and your Python script calls `multiprocessing.Pool()` with no argument. How many worker processes are created, and what is the risk?

- A) 4 workers — `Pool()` reads the LSF `#BSUB -n` directive automatically
- B) `os.cpu_count()` workers (e.g., 32 on a 32-core node) — this oversubscribes the allocated cores, slowing all jobs on that node
- C) 1 worker — `Pool()` defaults to single-process mode in batch environments
- D) `os.cpu_count()` workers, but LSF automatically limits execution to the 4 allocated cores

**Answer: B**

- A) Incorrect — `Pool()` calls `os.cpu_count()` and has no knowledge of LSF, SLURM, or any job scheduler. It reads the hardware, not the job manifest. LSF directives are only enforced at the OS scheduling level, not exposed to Python.
- B) Correct — `Pool()` defaults to `os.cpu_count()`, which returns the node's total physical CPUs (e.g., 32 or 64). If you allocated 4 but spawn 32 workers, your job uses 8× its allocation, starving other users' jobs sharing that node and potentially causing policy violations.
- C) Incorrect — `Pool()` with no argument uses `os.cpu_count()`, not 1. Single-process mode would be `Pool(1)` or just not using a Pool at all.
- D) Incorrect — LSF allocates cores and accounts for usage, but does not prevent a process from using more than allocated. It is entirely the programmer's responsibility to call `Pool(int(os.environ.get('LSB_DJOB_NUMPROC', os.cpu_count())))` or simply `Pool(4)`.

---

## Q10 — The `if __name__ == '__main__':` Guard

> **Week reference:** Week 5
> **Mental Model:** On macOS/Windows, multiprocessing uses the `spawn` start method: each worker imports the script from scratch. Without the guard, the worker re-executes the pool creation code, spawning more workers, which spawn more... infinite recursion. The guard breaks this loop by running pool code only in the original process.

On macOS, a Python script using `multiprocessing.Pool` runs correctly from the command line but crashes with a `RuntimeError` about recursive spawning when imported as a module. What is the fix?

- A) Switch from `multiprocessing` to `threading` — threads do not require this guard
- B) Wrap all `Pool` usage inside `if __name__ == '__main__':` to prevent the spawned child processes from re-executing the pool creation code
- C) Add `multiprocessing.set_start_method('fork')` — the `spawn` method causes recursive imports
- D) The error is unrelated to multiprocessing; it is caused by a circular import

**Answer: B**

- A) Incorrect — while threads avoid this specific issue (threads don't re-import the module), switching to threading doesn't fix the multiprocessing spawning problem and would remove process-level parallelism.
- B) Correct — on macOS/Windows, `multiprocessing` defaults to `spawn`: each worker process re-imports the top-level module. Without `if __name__ == '__main__':`, the worker sees the pool creation code and tries to spawn more workers, triggering infinite recursion. The guard ensures pool creation only runs in the parent.
- C) Incorrect — using `fork` on macOS can cause deadlocks with Objective-C runtimes (e.g., if the parent imported Foundation frameworks before forking). The `spawn` method is safer on macOS; the correct fix is the `__name__` guard.
- D) Incorrect — the error is specifically caused by unguarded pool creation being re-executed in spawned child processes. The traceback will reference multiprocessing internals, not circular import chains.

---

## Q11 — Task Granularity and IPC Overhead

> **Week reference:** Week 5
> **Mental Model:** IPC overhead per task (pickle + queue send + queue recv + unpickle) is on the order of microseconds to tens of microseconds. If your task is shorter than this overhead, parallelism makes things worse. The fix is always to batch: trade many small tasks for fewer large tasks.

You have 1,000,000 independent tasks, each taking approximately 1 microsecond. You submit them all to `pool.map(func, range(1_000_000))`. The parallel version is slower than a serial loop. What is the most likely cause and fix?

- A) The GIL is blocking all threads; fix by switching to `ThreadPoolExecutor`
- B) IPC overhead per task (pickling, queue communication) dominates the 1µs task time; fix by chunking into fewer, larger tasks (e.g., 8 tasks of 125,000 operations each)
- C) 1,000,000 processes are being spawned; fix by using `chunksize=1000`
- D) NumPy is not being used; fix by vectorising with `numpy.vectorize`

**Answer: B**

- A) Incorrect — the problem is not the GIL but task granularity. Switching to threads would have the same IPC overhead problem (pickling arguments and results through the task queue is necessary regardless of threading vs multiprocessing).
- B) Correct — each `pool.map` task incurs IPC overhead for pickling arguments, enqueuing the task, dequeueing by the worker, executing (1µs), pickling the result, and returning it. This overhead (typically ~10-100µs) far exceeds 1µs of actual work. Batching into 8 tasks of 125,000 ops each means only 8 IPC roundtrips instead of 1,000,000.
- C) Incorrect — `Pool` creates a fixed pool of worker processes (e.g., 8) and reuses them; it does not spawn 1,000,000 processes. However, the 1,000,000 per-task IPC communications still dominate. Using `chunksize=1000` would actually help by reducing IPC calls to ~1000, but the root cause explanation is wrong.
- D) Incorrect — the issue is parallelism IPC overhead, not lack of vectorisation. While vectorising with NumPy would be the right fix for serial code, it doesn't address the multiprocessing overhead problem.

---

## Q12 — I/O-Bound Tasks and the GIL

> **Week reference:** Week 5
> **Mental Model:** The GIL is released during blocking syscalls (read, write, recv). I/O-bound code is therefore GIL-friendly for threads: while one thread waits for disk, another thread runs. For I/O-bound work, ThreadPool avoids process-spawn overhead and is the right tool.

A script reads 500 CSV files from disk sequentially, taking 60 seconds total. Each file read is I/O-bound (the disk is the bottleneck, not the CPU). Which parallelisation strategy is appropriate?

- A) `multiprocessing.Pool` only — the GIL blocks threads even during disk I/O
- B) `ThreadPoolExecutor` is appropriate — the GIL is released during I/O operations, so threads can overlap waiting for disk reads
- C) Neither threading nor multiprocessing helps — disk I/O is inherently sequential
- D) Only `asyncio` can overlap I/O operations; threads and processes cannot

**Answer: B**

- A) Incorrect — the GIL is released during blocking I/O system calls (read/write). While the thread is waiting for the OS to return data from disk, Python releases the GIL and another thread can run. Threads genuinely run in parallel during I/O.
- B) Correct — Python releases the GIL during blocking I/O (read/write syscalls), allowing multiple threads to overlap disk waits. With 500 files, ThreadPoolExecutor can have N files in flight simultaneously, reducing total wall time. No process-spawn overhead, no array serialization needed.
- C) Incorrect — overlapping I/O waits across threads (or processes) reduces total wall-clock time by keeping the I/O pipeline full. If the disk can serve multiple concurrent reads, threading directly exploits this. Even with a single disk, threads reduce time by overlapping OS latency (seek, buffer fill) across files.
- D) Incorrect — `asyncio` with non-blocking I/O is one valid option, but it requires async-compatible file I/O libraries (e.g., `aiofiles`). `ThreadPoolExecutor` is simpler, works with standard `open()`/`read()`, and is equally effective for synchronous file reads.

---

## Q13 — Parallelising `@jit(nogil=True)` Across Outer Loop

> **Week reference:** Week 6
> **Mental Model:** Always parallelize at the outermost independent level. Sequential internal steps → can't parallelize inside. Independent outer calls → parallelize there. With `nogil=True`, threads at the outer level get true parallelism at the lowest overhead.

`simulate_single(params)` is compiled with `@jit(nopython=True, nogil=True)`. It runs sequential internal steps but you need to call it 10,000 times with different parameters. What is the correct strategy?

- A) Use `multiprocessing.Pool` to run 10,000 parallel calls — Numba JIT functions require separate processes
- B) Use `ThreadPoolExecutor` to parallelise the outer loop of 10,000 independent calls — `nogil=True` releases the GIL so threads execute truly in parallel
- C) Parallelise the internal steps of `simulate_single` by passing a thread pool into the function
- D) Use `pool.map` with `chunksize=1` — Numba automatically detects multiprocessing and optimises accordingly

**Answer: B**

- A) Incorrect — while multiprocessing would work functionally, it is unnecessarily heavy. Each process would need to JIT-compile the function separately, and parameters/results must be serialized across process boundaries. `nogil=True` exists precisely to enable efficient thread-based parallelism.
- B) Correct — the 10,000 calls are independent at the outer loop level. `nogil=True` means the GIL is released while each thread executes `simulate_single`, so threads are not serialized. `ThreadPoolExecutor` with N workers gives ~N× speedup with near-zero IPC overhead (no pickling, no process spawn).
- C) Incorrect — the internal steps of `simulate_single` are sequentially dependent on each other (step n+1 uses step n's result). Passing a thread pool into the function cannot parallelize a data dependency chain. Parallelism must happen at the outer loop of independent calls.
- D) Incorrect — `pool.map` creates multiprocessing workers (separate processes), which works but incurs unnecessary overhead. `chunksize=1` with multiprocessing is dynamic scheduling — reasonable for variable-time tasks but overkill here. The `nogil=True` flag specifically enables thread-based parallelism as the preferred approach.

---
