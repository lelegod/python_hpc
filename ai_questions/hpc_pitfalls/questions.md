# HPC Pitfalls — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — LSF I/O Channels vs Shell Redirection](#q1--lsf-io-channels-vs-shell-redirection)
- [Q2 — Why LSF -o/-e Is Slow](#q2--why-lsf--o-e-is-slow)
- [Q3 — Python -u Flag](#q3--python--u-flag)
- [Q4 — NumPy Threads: Allocating Cores Is Not Enough](#q4--numpy-threads-allocating-cores-is-not-enough)
- [Q5 — Which Env Vars Control NumPy Threading](#q5--which-env-vars-control-numpy-threading)
- [Q6 — VAR=val vs export VAR=val](#q6--varval-vs-export-varval)
- [Q7 — Child Process Env Var Inheritance](#q7--child-process-env-var-inheritance)
- [Q8 — Thread Oversubscription Defined](#q8--thread-oversubscription-defined)
- [Q9 — ThreadPool + Multi-Threaded NumPy Outcome](#q9--threadpool--multi-threaded-numpy-outcome)
- [Q10 — When to Use ThreadPool vs ProcessPool for NumPy](#q10--when-to-use-threadpool-vs-processpool-for-numpy)
- [Q11 — GIL and np.matmul](#q11--gil-and-npmatmul)
- [Q12 — span[hosts=1] and Shared Memory](#q12--spanhosts1-and-shared-memory)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2--generated-practice-questions-exam-day-focus)
- [Q13 — Login Node Abuse](#q13--login-node-abuse)
- [Q14 — Memory Estimation Errors](#q14--memory-estimation-errors)
- [Q15 — Optimal Thread Count for Oversubscription Fix](#q15--optimal-thread-count-for-oversubscription-fix)
- [Q16 — I/O Runtime Comparison Numbers](#q16--io-runtime-comparison-numbers)
- [Q17 — Multi-Threaded NumPy Speedup](#q17--multi-threaded-numpy-speedup)
- [Q18 — Diagnosing No Speedup from Extra Cores](#q18--diagnosing-no-speedup-from-extra-cores)
- [Q19 — export in a Subshell vs Current Shell](#q19--export-in-a-subshell-vs-current-shell)
- [Q20 — Choosing the Right Parallelism Strategy](#q20--choosing-the-right-parallelism-strategy)
- [Q21 — Combining ThreadPool and Single-Threaded NumPy](#q21--combining-threadpool-and-single-threaded-numpy)
- [Q22 — Recognising Multiple Pitfalls in One Script](#q22--recognising-multiple-pitfalls-in-one-script)

---

> Topics: Excessive I/O, unexported env vars, thread oversubscription, login node abuse, memory errors.
> Exam frequency: **Week 13 topic — common exam target**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--lsf-io-channels-vs-shell-redirection)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — LSF I/O Channels vs Shell Redirection

> **Week reference:** Week 13

**Mental Model:** LSF's `-o`/`-e` flags route output through the scheduler's own I/O infrastructure, which adds significant overhead for high-volume output. Direct shell redirection bypasses the scheduler entirely and writes straight to disk.

A Python job prints 100,000 long lines. One version uses only LSF's `#BSUB -o` flag; a second version additionally redirects stdout with `python -u script.py > /work3/output.txt`. Which statement best describes the performance difference observed in Week 13?

- A) Both approaches have identical run times because LSF ultimately writes to the same filesystem
- B) The version using LSF `-o` alone is faster because the scheduler optimises disk writes
- C) The version with shell redirection (`>`) is roughly 26x faster (3 s vs 80 s) because it bypasses LSF's I/O channels
- D) Shell redirection is slightly faster, but only for files larger than 1 GB

**Answer: C**

- A) Incorrect — Even though the same Lustre/NFS filesystem is used, routing through LSF's I/O channels introduces massive scheduling overhead that accumulates with every line printed.
- B) Incorrect — LSF does not optimise disk writes; its I/O channel is a bottleneck, not an accelerator, for high-volume output.
- C) Correct — The Week 13 exercise measured 80 s with LSF-only output vs 3 s with direct shell redirection — a ~26x difference. The bottleneck is the overhead of routing each line through LSF's infrastructure.
- D) Incorrect — The effect is proportional to the number of I/O operations, not file size, and was clearly observed on a relatively small output from 100,000 print statements.

---

## Q2 — Why LSF -o/-e Is Slow

> **Week reference:** Week 13

**Mental Model:** LSF's `-o`/`-e` output channels were designed for moderate log volumes, not for the continuous stream produced by programs that print on every iteration. Each write round-trips through the LSF daemon, compounding latency.

Why is routing large volumes of print output through LSF's `-o`/`-e` flags so much slower than shell redirection?

- A) LSF compresses stdout before writing it, which is CPU-intensive
- B) LSF's I/O channels add per-write overhead that accumulates when a program prints many lines, whereas shell redirection writes directly to a file descriptor
- C) LSF writes to a remote NFS share, but shell redirection writes to local SSD
- D) The `-o` flag is rate-limited by the scheduler to prevent disk saturation

**Answer: B**

- A) Incorrect — LSF does not compress stdout/stderr output; there is no compression step involved.
- B) Correct — Each line printed by the program travels through LSF's daemon infrastructure, adding latency per operation. With 100,000 lines, this overhead multiplies to tens of seconds. Direct shell redirection (`>`) skips this path entirely.
- C) Incorrect — Both methods typically write to the same shared filesystem; the difference is the software path taken, not the physical storage device.
- D) Incorrect — LSF does not impose an explicit rate limit on `-o`; the slowness comes from architectural overhead, not a deliberate throttle.

---

## Q3 — Python -u Flag

> **Week reference:** Week 13

**Mental Model:** Python buffers stdout by default — output accumulates in memory and is flushed in batches. In HPC scripts where you redirect to a file and want to see real-time progress (or avoid data loss on crash), unbuffered output is essential.

What does running `python -u script.py` do, and why is it recommended in HPC job scripts?

- A) It enables unicode support, which is required for non-ASCII output in LSF logs
- B) It disables Python's stdout/stderr buffering so output is written immediately, which is important for real-time log files and measuring I/O performance accurately
- C) It runs Python in single-threaded mode, preventing accidental multi-threading
- D) It unlocks additional CPU registers for numerical computation

**Answer: B**

- A) Incorrect — The `-u` flag has nothing to do with unicode; Python handles unicode via encoding settings, not this flag.
- B) Correct — `-u` stands for "unbuffered." Without it, Python collects output in a buffer and flushes periodically, which can distort timing measurements and delay log visibility. In HPC scripts this is especially important when measuring I/O overhead.
- C) Incorrect — `-u` affects I/O buffering only, not thread scheduling or Python's threading model.
- D) Incorrect — Python has no flag that unlocks CPU registers; this is a hardware concern, not a Python interpreter option.

---

## Q4 — NumPy Threads: Allocating Cores Is Not Enough

> **Week reference:** Week 13

**Mental Model:** LSF allocates cores, but NumPy's BLAS backend only uses multiple threads if told to via environment variables. Allocation and utilisation are separate concerns — allocating 8 cores does not automatically make NumPy use 8 threads.

A job script requests 8 cores with `#BSUB -n 8` and runs a NumPy matrix multiplication loop. No environment variables are set. Compared to the 1-core run, what happens to the run time?

- A) It drops to approximately 1/8 of the original time because LSF distributes work across cores
- B) It stays roughly the same (~5.96 s vs ~5.87 s) because NumPy does not automatically use the extra cores
- C) It doubles because LSF's scheduler overhead increases with more allocated cores
- D) It halves because the OS automatically load-balances threads across available CPUs

**Answer: B**

- A) Incorrect — LSF allocates cores but does not parallelize NumPy's internal loops. NumPy's threading is controlled by environment variables, not core allocation.
- B) Correct — The Week 13 exercise confirmed this: with 8 cores allocated but no thread env vars set, the run time was ~5.96 s vs ~5.87 s with 1 core — effectively identical. The extra cores sit idle.
- C) Incorrect — LSF's scheduling overhead is negligible compared to the computation time; it does not cause a 2x slowdown.
- D) Incorrect — The OS can migrate threads across CPUs, but NumPy uses a single thread unless its BLAS backend is explicitly told otherwise.

---

## Q5 — Which Env Vars Control NumPy Threading

> **Week reference:** Week 13

**Mental Model:** NumPy delegates dense linear algebra to a BLAS library (MKL, OpenBLAS, or the system BLAS). Each library reads a different environment variable. Setting all of them ensures coverage regardless of which backend is compiled in.

Which set of environment variables should be exported to enable 8-thread NumPy execution on a node with 8 allocated cores?

- A) `PYTHON_THREADS=8` and `NUMPY_WORKERS=8`
- B) `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=8`
- C) `LSF_THREADS=8` and `BLAS_CORES=8`
- D) `NUMBA_NUM_THREADS=8` and `CUDA_VISIBLE_DEVICES=8`

**Answer: B**

- A) Incorrect — `PYTHON_THREADS` and `NUMPY_WORKERS` are not real environment variables recognised by NumPy or its BLAS backends.
- B) Correct — `OMP_NUM_THREADS` controls OpenMP (used by many numerical libraries), `MKL_NUM_THREADS` controls Intel MKL, and `OPENBLAS_NUM_THREADS` controls OpenBLAS. Setting all three ensures the active BLAS backend picks up the thread count. The course job script also includes `MPI_NUM_THREADS=8`.
- C) Incorrect — `LSF_THREADS` and `BLAS_CORES` are not recognised variables; LSF does not expose a thread-count variable of this name.
- D) Incorrect — `NUMBA_NUM_THREADS` controls Numba's JIT parallelism (a different framework), and `CUDA_VISIBLE_DEVICES` selects GPUs — neither controls NumPy's BLAS threads.

---

## Q6 — VAR=val vs export VAR=val

> **Week reference:** Week 13

**Mental Model:** In bash, `VAR=val` sets a variable in the current shell's environment only. `export VAR=val` marks it for inheritance by child processes. Since Python is a child process of the job script shell, only exported variables reach it.

In a bash job script, what is the difference between `OMP_NUM_THREADS=8` and `export OMP_NUM_THREADS=8`?

- A) There is no difference — all shell variables are automatically inherited by child processes
- B) `OMP_NUM_THREADS=8` makes the variable available only in the current shell; `export OMP_NUM_THREADS=8` makes it available to child processes like the Python interpreter
- C) `export` encrypts the variable value to prevent other users from seeing it
- D) `export` sets the variable globally across all nodes in the cluster

**Answer: B**

- A) Incorrect — This is the classic pitfall. Variables set without `export` exist only in the current shell's scope and are NOT passed to child processes.
- B) Correct — The bash `export` built-in marks a variable as part of the environment that is passed to any child process (via `execve`). Without `export`, Python launches without seeing `OMP_NUM_THREADS`, so NumPy defaults to 1 thread.
- C) Incorrect — `export` has nothing to do with encryption or access control; it purely controls environment inheritance.
- D) Incorrect — `export` is a shell-level concept affecting only child processes of the current shell; it has no cluster-wide scope.

---

## Q7 — Child Process Env Var Inheritance

> **Week reference:** Week 13

**Mental Model:** The UNIX process model passes a copy of the parent's exported environment to each child. Variables that exist in the shell but were not exported are invisible to any spawned program, including Python.

A job script contains the following lines:

```bash
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
python matmul.py
```

Why does NumPy still run with a single thread despite these assignments?

- A) The variable names are misspelled — the correct names use lowercase letters
- B) The variables are assigned but not exported, so the Python child process never receives them
- C) Python ignores environment variables unless they are passed with `python --env OMP_NUM_THREADS=8`
- D) The assignments must appear inside the `#BSUB` directives to take effect

**Answer: B**

- A) Incorrect — `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are the correct uppercase names used by OpenMP and MKL respectively.
- B) Correct — Without `export`, these variables exist only in the job script's own shell environment. When the shell forks and execs `python`, the child's environment does not contain these variables, so NumPy's BLAS backend sees no thread-count directive and defaults to 1.
- C) Incorrect — Python reads environment variables directly from the process environment inherited at launch; there is no `--env` flag.
- D) Incorrect — `#BSUB` directives are parsed by LSF before the script runs; they cannot set arbitrary environment variables for the shell execution phase.

---

## Q8 — Thread Oversubscription Defined

> **Week reference:** Week 13

**Mental Model:** A CPU core can only execute one hardware thread at a time. When more software threads are active than there are cores, the OS must context-switch between them, introducing scheduling overhead that can exceed the benefit of parallelism.

What is thread oversubscription, and why does it degrade performance?

- A) Using more memory than the node has RAM, causing swap thrashing
- B) Creating more active threads than available CPU cores, causing the OS to context-switch threads and adding scheduling overhead that outweighs the parallelism benefit
- C) Subscribing to more LSF queues than allowed, causing job scheduling delays
- D) Importing more Python modules than the interpreter can handle simultaneously

**Answer: B**

- A) Incorrect — This describes memory overcommitment / swap thrashing, which is a memory pitfall, not a threading pitfall.
- B) Correct — When active thread count > core count, threads compete for cores. The OS scheduler constantly switches between them, and all the cache warm-up, register saves, and scheduling overhead can make a 2-level parallel program slower than either parallelism level alone.
- C) Incorrect — LSF queue subscriptions are a job scheduling concept unrelated to thread counts inside a running job.
- D) Incorrect — Python imports are a module loading concern, not a threading concept.

---

## Q9 — ThreadPool + Multi-Threaded NumPy Outcome

> **Week reference:** Week 13

**Mental Model:** If you have 8-thread NumPy AND an 8-thread pool, each pool thread spawns 8 NumPy threads, giving 64 active threads on 8 cores — 8x oversubscribed. The Week 13 experiment showed this is actually slower than either approach alone.

A job with 8 cores runs a `ThreadPool(8)` to parallelize `np.matmul` calls, and also has `OMP_NUM_THREADS=8` exported. What is the expected run time compared to ThreadPool alone (with `OMP_NUM_THREADS=1`) and to multi-threaded NumPy alone (no ThreadPool)?

- A) Fastest of all three approaches because both levels of parallelism are active simultaneously
- B) Slowest of all three approaches (~1.87 s) due to oversubscription — each pool thread spawns 8 NumPy threads, creating 64 threads on 8 cores
- C) Identical to ThreadPool alone because NumPy automatically detects that threads are already in use and disables its own threading
- D) Identical to multi-threaded NumPy alone because Python's GIL prevents ThreadPool from adding parallelism

**Answer: B**

- A) Incorrect — Combining both levels does not multiply their benefits; it multiplies thread count beyond core count, introducing heavy overhead.
- B) Correct — The Week 13 exercise measured ~1.87 s for this doubly-parallel version, compared to ~1.28 s for multi-threaded NumPy alone and ~1.33 s for ThreadPool with single-threaded NumPy. The oversubscription overhead makes it the slowest option.
- C) Incorrect — NumPy's BLAS backend does not detect external threading contexts; it spawns its configured number of threads regardless.
- D) Incorrect — `np.matmul` releases the GIL during computation, so multiple ThreadPool threads CAN run truly in parallel. The GIL is not the limiting factor here.

---

## Q10 — When to Use ThreadPool vs ProcessPool for NumPy

> **Week reference:** Week 13

**Mental Model:** Python's GIL blocks two Python threads from running Python bytecode simultaneously. But NumPy's BLAS operations release the GIL during the heavy computation, so threads can overlap in C code even when the GIL would otherwise block them.

Why is a `ThreadPool` preferable to a `ProcessPool` for parallelizing `np.matmul` calls?

- A) `ThreadPool` is always faster than `ProcessPool` regardless of the workload
- B) `np.matmul` releases the GIL during computation, so multiple threads can execute BLAS code truly in parallel without the overhead of forking separate processes and copying large arrays
- C) `ProcessPool` cannot call NumPy functions because they require a shared memory address space
- D) LSF does not support running multiple processes within a single job

**Answer: B**

- A) Incorrect — `ThreadPool` is only preferable for GIL-releasing workloads. For pure-Python CPU-bound work, `ProcessPool` is needed to achieve true parallelism.
- B) Correct — Because `np.matmul` drops the GIL during its BLAS call, threads can overlap their computation in native code. `ProcessPool` would also work but adds significant overhead from forking processes and pickling/copying large NumPy arrays between processes.
- C) Incorrect — `ProcessPool` workers CAN call NumPy; they just each get their own copy of arrays (via pickling), which is the inefficiency, not an impossibility.
- D) Incorrect — LSF supports multiprocessing within a job; the choice between ThreadPool and ProcessPool is a Python-level design decision.

---

## Q11 — GIL and np.matmul

> **Week reference:** Week 13

**Mental Model:** The GIL is held while Python bytecode executes but can be released by C extensions that do not touch Python objects. NumPy deliberately releases the GIL during matrix operations so that threading is effective.

Which statement correctly explains how Python's GIL interacts with `np.matmul` in a multi-threaded context?

- A) The GIL is held throughout `np.matmul`, making ThreadPool ineffective for NumPy workloads
- B) The GIL is released during `np.matmul`'s underlying BLAS computation, allowing multiple threads to execute matrix multiplications concurrently
- C) `np.matmul` uses the GIL only when writing results back to the output array, not during the computation phase
- D) NumPy bypasses the GIL entirely by using a separate Python interpreter for each thread

**Answer: B**

- A) Incorrect — This is the naive assumption. NumPy deliberately releases the GIL in its C extension code during compute-heavy operations to enable thread-level parallelism.
- B) Correct — When `np.matmul` enters its BLAS routine (dgemm etc.), it calls `Py_BEGIN_ALLOW_THREADS` to release the GIL, then reacquires it on return. This is the key reason ThreadPool works for NumPy without needing ProcessPool.
- C) Incorrect — The GIL release covers the entire BLAS computation, not just the write-back phase.
- D) Incorrect — There is only one Python interpreter per process; NumPy does not launch sub-interpreters. It simply releases the GIL in C code.

---

## Q12 — span[hosts=1] and Shared Memory

> **Week reference:** Week 13

**Mental Model:** Shared memory (including NumPy shared arrays and Python's `multiprocessing.shared_memory`) only works when all processes are on the same physical node, because memory is not shared across nodes. Without `span[hosts=1]`, LSF may spread a multi-core job across multiple nodes.

Why must `#BSUB -R "span[hosts=1]"` be added to a job script that uses Python's shared memory or a `ThreadPool`?

- A) It forces all 8 cores to run at the same clock speed for reproducible timing
- B) It ensures all allocated cores are on the same physical node, which is required for shared memory access between processes and for meaningful thread parallelism
- C) It enables hyperthreading on the allocated cores
- D) It reserves the entire node exclusively, preventing other users' jobs from running on it

**Answer: B**

- A) Incorrect — `span[hosts=1]` constrains node placement, not CPU frequency; clock speed is a separate concern controlled by CPU model selection.
- B) Correct — Without `span[hosts=1]`, LSF could split 8 cores across two or more nodes. Shared memory and thread pools require a single address space, which only exists within one node. Cross-node communication requires MPI or similar.
- C) Incorrect — `span[hosts=1]` is a placement constraint, not a CPU feature flag; hyperthreading is a BIOS/OS setting.
- D) Incorrect — `span[hosts=1]` does not give exclusive node access; other jobs can still share the node. For exclusivity, you would use `#BSUB -x`.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

---

## Q13 — Login Node Abuse

> **Week reference:** Week 13

**Mental Model:** Login nodes are shared infrastructure used by all cluster users for file management, job submission, and light editing. Running computation-heavy jobs on them degrades the experience for every other user and may result in your session being killed.

Which of the following actions is considered a serious HPC pitfall when working on a login node?

- A) Running `ls -lh` to check file sizes in your home directory
- B) Editing a Python script with `nano` or `vim`
- C) Submitting a job with `bsub < submit.sh`
- D) Running a 100 x 1000×1000 matrix multiplication loop directly in the terminal without submitting a job

**Answer: D**

- A) Incorrect — `ls` is a trivial file system operation with negligible CPU cost; it is entirely appropriate on a login node.
- B) Incorrect — Text editing is a light interactive task and is an expected use of the login node.
- C) Incorrect — `bsub` is exactly how jobs are supposed to be launched; it hands off work to the cluster, not the login node.
- D) Correct — A matrix multiplication loop over 100 pairs of 1000×1000 matrices is heavy computation that should run on a compute node via LSF. Running it on the login node consumes shared CPU resources and can disrupt other users.

---

## Q14 — Memory Estimation Errors

> **Week reference:** Week 13

**Mental Model:** Under-requesting memory causes LSF to kill the job when usage exceeds the limit. Over-requesting wastes cluster resources and reduces queue priority. The Week 13 matmul exercise requests 4 GB for three arrays of 100 × 1000 × 1000 float64 values — each array is ~800 MB.

A job creates three `float64` NumPy arrays of shape `(100, 1000, 1000)`. What is the minimum memory request that safely covers all three arrays?

- A) 300 MB — each float64 element is 3 bytes
- B) 800 MB — one array alone
- C) ~2.4 GB — three arrays × ~800 MB each, with some headroom
- D) 40 GB — 100 matrices × 1000 × 1000 × 8 bytes × 3 arrays, converted to GB

**Answer: C**

- A) Incorrect — `float64` is 8 bytes per element, not 3. The 3 likely comes from confusing dtype with something else.
- B) Incorrect — 800 MB covers only one array; the job creates three (A, B, and C), so total data is ~2.4 GB.
- C) Correct — Each array: 100 × 1000 × 1000 × 8 bytes = 800,000,000 bytes ≈ 800 MB. Three arrays = ~2.4 GB. The job script uses 4 GB to add headroom for Python runtime and temporary allocations.
- D) Incorrect — The arithmetic is correct (100 × 1000 × 1000 × 8 × 3 = 2,400,000,000 bytes ≈ 2.4 GB), but the stated result of 40 GB is off by ~17x, suggesting a unit conversion error.

---

## Q15 — Optimal Thread Count for Oversubscription Fix

> **Week reference:** Week 13

**Mental Model:** The safest rule is: total active threads = number of allocated cores. If you use a ThreadPool of size N, each NumPy call should use 1 thread, so NumPy env vars should be set to 1 (or unset). If you want NumPy multi-threading, skip the ThreadPool entirely.

A job requests 8 cores. You want to use a `ThreadPool` to parallelize `np.matmul`. What should `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` be set to?

- A) 8 — to maximise NumPy's internal parallelism in each thread
- B) 1 — to ensure each pool thread runs single-threaded NumPy, keeping total threads at 8
- C) 64 — the thread count auto-scales to the available cores
- D) 0 — to disable NumPy threading entirely and rely solely on the OS scheduler

**Answer: B**

- A) Incorrect — Setting NumPy threads to 8 with a ThreadPool of 8 creates 64 active threads on 8 cores (8x oversubscription), which the Week 13 exercise showed is slower (~1.87 s) than either approach alone.
- B) Correct — With `OMP_NUM_THREADS=1` (and equivalents), each pool thread runs a single NumPy thread, totalling exactly 8 active threads across 8 cores. This was measured at ~1.33 s in the exercise — on par with the pure multi-threaded NumPy approach.
- C) Incorrect — Thread counts do not auto-scale based on available cores; they default to 1 or to the hardware thread count depending on the library, neither of which is what you want here.
- D) Incorrect — Setting to 0 is typically undefined behaviour for these variables; use 1 to explicitly request single-threaded execution.

---

## Q16 — I/O Runtime Comparison Numbers

> **Week reference:** Week 13

**Mental Model:** The specific numbers from the exercise (80 s vs 3 s) are regularly cited in exam questions. Knowing the order of magnitude and the mechanism is more important than memorising exact seconds.

In the Week 13 I/O exercise, the job that routes 100,000 print statements through LSF's `-o` channel ran in ____ seconds, while the job with shell redirection ran in ____ seconds.

- A) 3 s and 80 s (redirection was slower)
- B) 80 s and 3 s (LSF channel was slower)
- C) 10 s and 10 s (no measurable difference)
- D) 0.1 s and 0.1 s (both near-instant at the scale of 100,000 lines)

**Answer: B**

- A) Incorrect — The order is reversed. Shell redirection was the fast option, not the slow one.
- B) Correct — LSF's own I/O channel: 80 s. Direct shell redirection to a file: 3 s. The 26x speedup comes from bypassing LSF's per-write infrastructure.
- C) Incorrect — There was a clear and dramatic 26x performance difference between the two approaches.
- D) Incorrect — 100,000 verbose print statements through LSF's channel is far from near-instant; the overhead accumulates to 80 seconds in the exercise.

---

## Q17 — Multi-Threaded NumPy Speedup

> **Week reference:** Week 13

**Mental Model:** Perfect scaling would give 8x speedup for 8 threads, but memory bandwidth, synchronisation overhead, and Amdahl's law cap real speedup. The exercise measured ~4.6x with 8 threads — reasonable for a memory-intensive workload.

In the Week 13 matmul exercise, running with 8 threads (via `OMP_NUM_THREADS=8`) reduced the run time from ~5.87 s to ~1.28 s. What is the approximate speedup, and why doesn't it reach the theoretical 8x?

- A) 8x speedup was achieved; the theoretical maximum matches observed results for this workload
- B) ~4.6x speedup; the gap from 8x is due to memory bandwidth saturation, synchronisation overhead, and the serial fraction of the loop
- C) ~2x speedup; NumPy's threading is limited to 2 threads regardless of the env var setting
- D) No speedup; the 5.87 s figure already includes multi-threaded NumPy

**Answer: B**

- A) Incorrect — 8x would require perfect scaling. Real BLAS workloads are limited by memory bandwidth and coordination overhead; 5.87 / 1.28 ≈ 4.6x, not 8x.
- B) Correct — 5.87 / 1.28 ≈ 4.6x. This is typical for memory-bandwidth-bound BLAS operations. The serial loop overhead (iterating over 100 matrices) also contributes an Amdahl-law serial fraction.
- C) Incorrect — NumPy with OpenBLAS or MKL scales beyond 2 threads; the env var is respected up to the available hardware thread count.
- D) Incorrect — The 5.87 s figure is from the 1-core baseline. Multi-threading was not active at that point.

---

## Q18 — Diagnosing No Speedup from Extra Cores

> **Week reference:** Week 13

**Mental Model:** When adding cores produces no speedup, the first question is: does the software know the cores are there? For NumPy, the answer is no unless thread-count env vars are explicitly exported.

A user submits a NumPy job requesting 16 cores but measures the same run time as a 1-core run. What is the most likely explanation?

- A) The HPC cluster's scheduler did not actually allocate 16 cores
- B) The NumPy operation is memory-bandwidth-bound and saturates at 1 core
- C) The thread-count environment variables (`OMP_NUM_THREADS` etc.) were not exported, so NumPy still runs single-threaded
- D) NumPy multi-threading requires GPU hardware and does not work on CPU-only nodes

**Answer: C**

- A) Incorrect — While possible in theory, LSF reliably allocates requested cores; the more common cause is software not using them.
- B) Incorrect — Memory bandwidth saturation causes diminishing returns beyond a few threads, but it does not explain identical performance from 1 to 16 cores. Saturation would show some speedup at low thread counts.
- C) Correct — This is the canonical Week 13 pitfall. Without `export OMP_NUM_THREADS=N`, NumPy's BLAS backend sees no directive and defaults to 1 thread. The cores are allocated but sit idle, producing no speedup.
- D) Incorrect — NumPy's BLAS multi-threading is a CPU feature. It has nothing to do with GPU hardware.

---

## Q19 — export in a Subshell vs Current Shell

> **Week reference:** Week 13

**Mental Model:** Variables exported in a subshell (e.g., inside `$(...)` or a child bash process) do NOT propagate back to the parent shell. Environment propagation is strictly downward: parent to child, never child to parent.

Consider this script:

```bash
#!/bin/bash
(export OMP_NUM_THREADS=8)
python matmul.py
```

Why does `python matmul.py` still see `OMP_NUM_THREADS` as unset?

- A) Parentheses change `export` into a local variable assignment with no effect
- B) The `export` runs in a subshell (created by `(...)`); the variable is never added to the parent shell's environment, so Python doesn't inherit it
- C) `OMP_NUM_THREADS` must be set before the `#!/bin/bash` shebang line
- D) Python ignores `OMP_NUM_THREADS` when set inside bash scripts

**Answer: B**

- A) Incorrect — Parentheses create a subshell; `export` within that subshell does correctly mark the variable for that subshell's children, but the parent shell is unaffected.
- B) Correct — `(...)` spawns a subshell. The subshell exports `OMP_NUM_THREADS` into its own environment and any children of the subshell, but changes to a child's environment never propagate back to the parent. When the subshell exits, the variable is gone. `python` runs in the parent shell's environment, which was never modified.
- C) Incorrect — Environment variables are not part of the shebang mechanism; there is no "before the shebang" execution context.
- D) Incorrect — Python reads environment variables from the process environment it inherits; it does not discriminate based on how they were set in bash.

---

## Q20 — Choosing the Right Parallelism Strategy

> **Week reference:** Week 13

**Mental Model:** Three strategies exist for NumPy parallelism: (1) let BLAS use multiple threads internally, (2) use a ThreadPool with single-threaded BLAS, (3) use both together. Option 3 causes oversubscription. Options 1 and 2 are roughly equivalent in practice.

A user has 8 cores and wants to parallelize 100 independent `np.matmul` calls. Which two strategies are approximately equivalent in performance and avoid oversubscription?

- A) `ThreadPool(8)` with `OMP_NUM_THREADS=8`, and `ThreadPool(1)` with `OMP_NUM_THREADS=1`
- B) `ThreadPool(8)` with `OMP_NUM_THREADS=1`, and `ThreadPool(1)` with `OMP_NUM_THREADS=8`
- C) `ThreadPool(8)` with `OMP_NUM_THREADS=8`, and `ThreadPool(64)` with `OMP_NUM_THREADS=1`
- D) `Pool(8)` (process pool) with `OMP_NUM_THREADS=8`, and `ThreadPool(8)` with `OMP_NUM_THREADS=8`

**Answer: B**

- A) Incorrect — The first combination creates 64 threads on 8 cores (oversubscribed); the second is just 1 thread — both are poor choices.
- B) Correct — `ThreadPool(8)` with single-threaded NumPy uses exactly 8 threads. `ThreadPool(1)` (effectively no pool) with 8-thread NumPy also uses exactly 8 threads. Both measured ~1.28–1.33 s in the Week 13 exercise. Neither oversubscribes.
- C) Incorrect — `ThreadPool(64)` with `OMP_NUM_THREADS=1` creates 64 threads on 8 cores — massively oversubscribed.
- D) Incorrect — Both combinations in D use Pool(8)/ThreadPool(8) with `OMP_NUM_THREADS=8`, creating 64 active threads — the oversubscription scenario measured at ~1.87 s.

---

## Q21 — Combining ThreadPool and Single-Threaded NumPy

> **Week reference:** Week 13

**Mental Model:** Removing multi-threaded NumPy from the ThreadPool scenario gives each pool thread its own single BLAS thread, for 8 total — matching the core count exactly. The measured result (~1.33 s) is essentially as fast as letting BLAS use 8 threads alone (~1.28 s).

In the Week 13 exercise, what happened when `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` were set to 1 while using `ThreadPool(8)`?

- A) The run time increased dramatically because BLAS was disabled, leaving Python loops to perform the computation
- B) The run time was approximately 1.33 s — similar to the multi-threaded NumPy approach (~1.28 s) and faster than the doubly-parallel approach (~1.87 s)
- C) The run time was identical to the single-core, single-thread baseline (~5.87 s)
- D) The run time was faster than any other approach because single-threaded BLAS has lower overhead per call

**Answer: B**

- A) Incorrect — Setting `OMP_NUM_THREADS=1` reduces BLAS to single-threaded operation, but BLAS is still used. The ThreadPool provides the parallelism that compensates.
- B) Correct — With single-threaded BLAS and 8 pool threads, total active threads = 8 = core count. No oversubscription. The measured time (~1.33 s) is close to pure 8-thread NumPy (~1.28 s), confirming both approaches are valid and roughly equivalent.
- C) Incorrect — `ThreadPool(8)` parallelizes the 100 matmul calls across 8 workers; even with single-threaded BLAS, there is significant speedup over the serial baseline.
- D) Incorrect — Single-threaded BLAS per call does have lower per-call coordination overhead, but the ~1.33 s result is not the fastest observed; ~1.28 s (pure multi-threaded NumPy) is slightly faster.

---

## Q22 — Recognising Multiple Pitfalls in One Script

> **Week reference:** Week 13

**Mental Model:** Real HPC bugs often combine multiple pitfalls simultaneously. A script can waste cores (no thread env vars), oversubscribe (both ThreadPool and multi-threaded NumPy), AND push I/O through LSF channels — each compounding the others.

A colleague's job script: (1) requests 8 cores but sets thread vars without `export`, (2) uses `ThreadPool(8)` AND `OMP_NUM_THREADS=8` exported, (3) prints 200,000 lines of output through LSF `-o` only. How many distinct HPC pitfalls does this script contain?

- A) One — only the thread oversubscription matters
- B) Two — unexported env vars and thread oversubscription
- C) Three — unexported env vars (no-op since (2) does export), thread oversubscription, and excessive I/O through LSF channels
- D) Zero — the script correctly uses 8 cores and handles output

**Answer: C**

- A) Incorrect — Thread oversubscription is one pitfall, but the script has additional problems worth identifying separately.
- B) Incorrect — The env vars in point (1) without `export` are indeed a pitfall, but note that point (2) separately exports correctly — making (2) itself the oversubscription pitfall. And point (3) adds a third distinct pitfall. Count carefully: (1) is the export pitfall, (2) is the oversubscription pitfall, (3) is the I/O pitfall.
- C) Correct — Three distinct pitfalls: (1) thread count variables set without `export` are invisible to Python; (2) combining `ThreadPool(8)` with `OMP_NUM_THREADS=8` creates 64 threads on 8 cores; (3) routing 200,000 output lines through LSF's `-o` channel could take many minutes instead of seconds.
- D) Incorrect — The script contains multiple serious performance bugs; it is far from correct.

---
