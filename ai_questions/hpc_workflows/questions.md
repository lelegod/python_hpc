# HPC Workflows — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Job Array Index Range](#q1-job-array-index-range)
- [Q2 — Job Array Step Syntax](#q2-job-array-step-syntax)
- [Q3 — Zero-Based Array Access in Python](#q3-zero-based-array-access-in-python)
- [Q4 — Per-Element Log Files](#q4-per-element-log-files)
- [Q5 — Email Notification with Job Arrays](#q5-email-notification-with-job-arrays)
- [Q6 — Orphan Background Process](#q6-orphan-background-process)
- [Q7 — Correct Cleanup of Background Process](#q7-correct-cleanup-of-background-process)
- [Q8 — Thread Environment Variables and Oversubscription](#q8-thread-environment-variables-and-oversubscription)
- [Q9 — Fixing Thread Oversubscription](#q9-fixing-thread-oversubscription)
- [Q10 — multiprocessing.Pool Default Worker Count](#q10-multiprocessingpool-default-worker-count)
- [Q11 — Map-Reduce Job Dependency](#q11-map-reduce-job-dependency)
- [Q12 — Login Node vs Compute Node](#q12-login-node-vs-compute-node)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q13 — done() vs ended() Critical Distinction](#q13-done-vs-ended-critical-distinction)
- [Q14 — ended() Permissive Dependency](#q14-ended-permissive-dependency)
- [Q15 — Output File %J vs %I](#q15-output-file-j-vs-i)
- [Q16 — Three-Stage Pipeline Construction](#q16-three-stage-pipeline-construction)
- [Q17 — Concurrent Array Job Limit](#q17-concurrent-array-job-limit)
- [Q18 — LSB_JOBINDEX Off-by-One in Python](#q18-lsb_jobindex-off-by-one-in-python)
- [Q19 — Dependency on Specific Job ID](#q19-dependency-on-specific-job-id)
- [Q20 — AND Dependency: Both Jobs Must Succeed](#q20-and-dependency-both-jobs-must-succeed)
- [Q21 — What %I Expands To in a Running Job](#q21-what-i-expands-to-in-a-running-job)
- [Q22 — Danger of Missing %I in Output Directive](#q22-danger-of-missing-i-in-output-directive)

---

> Topics: Job arrays, $LSB_JOBINDEX, background processes, thread env vars, oversubscription.
> Exam frequency: **2024 exam + F25** (job arrays, dependencies).

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--job-array-index-range)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — Job Array Index Range

> **Week reference:** Week 11

**Mental Model:** LSF arrays are 1-based — the range in `[1-10]` is both the index range and the literal values `$LSB_JOBINDEX` takes. Python lists are 0-based. The mismatch means you always need a `- 1` when indexing Python structures from `$LSB_JOBINDEX`.

You submit a job array with `#BSUB -J myjob[1-10]`. What values does `$LSB_JOBINDEX` take across the 10 array elements?

- A) 0, 1, 2, ..., 9
- B) 1, 2, 3, ..., 10
- C) 0, 1, 2, ..., 10 (11 values)
- D) 1, 2, 3, ..., 9 (9 values)

**Answer: B**

- A) Incorrect — `$LSB_JOBINDEX` is 1-based in LSF, not 0-based. Index 0 does not exist; the first array element always has index 1. Choosing A is the most common off-by-one error when coming from zero-indexed programming languages.
- B) Correct — LSF numbers array elements starting from 1, matching the range specified in the directive. The range `[1-10]` is inclusive on both ends, giving exactly 10 elements with indices 1 through 10.
- C) Incorrect — the range `[1-10]` produces exactly 10 elements (10 − 1 + 1 = 10). Including index 0 would require the directive `[0-10]`, giving 11 elements.
- D) Incorrect — `[1-10]` is inclusive on the upper bound; index 10 is included. The range `[1-9]` would give 9 elements.

---

## Q2 — Job Array Step Syntax

> **Week reference:** Week 11

**Mental Model:** The syntax `[start-end:step]` mirrors Python's slice notation but is 1-based. Starting at 1 with step 2 gives odd numbers. This is useful for partitioning work: one array gets odds, another gets evens, covering all files without overlap.

You submit `#BSUB -J sim[1-100:2]`. Which indices are created?

- A) 1, 2, 3, ..., 100 (all 100)
- B) 2, 4, 6, ..., 100 (even numbers)
- C) 1, 3, 5, ..., 99 (odd numbers)
- D) 0, 2, 4, ..., 98 (even numbers starting at 0)

**Answer: C**

- A) Incorrect — the `:2` step means every second value is skipped; not all 100 are created. Without a step, `[1-100]` would create all 100. The `:2` reduces the count to 50.
- B) Incorrect — starting from 1 (not 0) with a step of 2 gives 1, 3, 5, ... (odd numbers). Even numbers would require starting at 2: `[2-100:2]`.
- C) Correct — starting at 1 and incrementing by 2 yields 1, 3, 5, ..., 99, giving 50 elements total. These are the odd-numbered indices in the 1–100 range.
- D) Incorrect — LSF arrays are 1-based; index 0 never exists in any LSF job array. The lowest possible index is always 1.

---

## Q3 — Zero-Based Array Access in Python

> **Week reference:** Week 11

**Mental Model:** `$LSB_JOBINDEX` is 1-based, Python lists are 0-based. Always subtract 1. If element 1 should process `file_list[0]`, element 2 should process `file_list[1]`, etc. — that is `file_list[index - 1]`. This is the most commonly tested indexing trap on the exam.

A Python script is submitted as part of a job array. The script reads a list of input files called `file_list`. Which line correctly picks the file for the current array element?

- A) `fname = file_list[int(os.environ["LSB_JOBINDEX"])]`
- B) `fname = file_list[int(os.environ["LSB_JOBINDEX"]) - 1]`
- C) `fname = file_list[int(os.environ["LSB_JOBINDEX"]) + 1]`
- D) `fname = file_list[int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1]`

**Answer: B**

- A) Incorrect — `$LSB_JOBINDEX` starts at 1, so for element 1, index 1 skips `file_list[0]`. The last element would try `file_list[N]` which raises an `IndexError` (off by one at both ends).
- B) Correct — subtracting 1 converts the 1-based LSF index to a 0-based Python list index. Element 1 → `file_list[0]`, element 2 → `file_list[1]`, ..., element N → `file_list[N-1]`. This is the correct mapping.
- C) Incorrect — adding 1 compounds the off-by-one error in the wrong direction. Element 1 accesses `file_list[2]`, skipping `file_list[0]` and `file_list[1]`, and the last element would cause an `IndexError`.
- D) Incorrect — `SLURM_ARRAY_TASK_ID` is the SLURM scheduler's equivalent variable; DTU HPC uses LSF/BSUB where the variable is `LSB_JOBINDEX`. This line would raise a `KeyError` because the environment variable does not exist in LSF.

---

## Q4 — Per-Element Log Files

> **Week reference:** Week 11

**Mental Model:** `%J` = job ID (same for all elements in the array), `%I` = array index (unique per element). To get unique filenames per run AND per element, you need both. Using only `%J` causes all 50 elements to write to the same file simultaneously, producing garbled output.

You want each element of a 50-element job array to write to its own log file. Which `#BSUB` output directive achieves this?

- A) `#BSUB -o results_%J.out`
- B) `#BSUB -o results_%I.out`
- C) `#BSUB -o results_%J_%I.out`
- D) `#BSUB -o results.out`

**Answer: C**

- A) Incorrect — `%J` expands to the parent job ID (e.g. `12345`), which is the same for all 50 array elements. All elements would write to `results_12345.out` simultaneously, interleaving output unpredictably and making debugging impossible.
- B) Incorrect — `%I` gives the array index (1, 2, ..., 50), producing unique files per element. However, if you rerun the job (new job ID), the new run's output overwrites the old one since the filenames are the same. Missing `%J` causes collisions across runs.
- C) Correct — combining `%J` (job ID, unique per submission) and `%I` (array index, unique per element within a run) produces a fully unique filename like `results_12345_3.out` per element per run. This is the standard pattern for job array logging.
- D) Incorrect — a fixed filename means all 50 elements compete to write to `results.out` at the same time, producing a race condition on the file and mixed-up output from different array elements.

---

## Q5 — Email Notification with Job Arrays

> **Week reference:** Week 11

**Mental Model:** LSF treats each array element as an independent job for all purposes, including notifications. `-N` on an array of 200 = 200 separate "job finished" emails. Never use `-N` with large arrays — use a dependency job to monitor completion instead.

A developer adds `#BSUB -N` to a job array script `#BSUB -J process[1-200]` so they know when the work is done. What is the consequence?

- A) One summary email is sent when all 200 elements complete.
- B) One email is sent per array element, totalling 200 emails.
- C) No email is sent; -N is not valid with job arrays.
- D) Two emails per element (start + end), totalling 400 emails.

**Answer: B**

- A) Incorrect — LSF treats each array element as an independent job for notification purposes; there is no "all elements done" aggregation built into `-N`. A single summary email would require a dependency job to detect completion.
- B) Correct — `-N` sends one end-of-job notification email per array element. For `process[1-200]`, this means 200 emails flood the inbox, one per completed element. This is a well-known LSF gotcha.
- C) Incorrect — `-N` is syntactically valid with job arrays; LSF accepts the directive without error. It just behaves unexpectedly at scale, sending one email per element instead of one total.
- D) Incorrect — `-N` sends only an end notification, not a start notification. Start notifications require a separate flag (`-B`). Even then, 200 elements × 1 email each = 200 emails, not 400.

---

## Q6 — Orphan Background Process

> **Week reference:** Week 11

**Mental Model:** A backgrounded process (`&`) becomes an orphan when the shell script exits — it keeps running under the init/systemd process tree. LSF's job slot stays occupied until the process exits or the wall-clock limit is reached. This wastes allocated compute time.

A job script launches a monitoring process in the background and then runs the main computation:

```bash
mymonitor &
python train.py
```

The Python script finishes in 2 hours. The wall-clock limit is 4 hours. What happens?

- A) The job ends cleanly as soon as `python train.py` finishes.
- B) `mymonitor` is automatically killed when `python train.py` exits.
- C) The job slot stays occupied by `mymonitor` until the 4-hour wall limit is hit.
- D) LSF detects the orphan and terminates it within a few seconds.

**Answer: C**

- A) Incorrect — the shell script exits after Python finishes, but the backgrounded `mymonitor` process was not a child of the shell's foreground process group in a way that LSF monitors directly. The job remains "running" as long as any process in the job's cgroup is alive.
- B) Incorrect — background processes are not automatically killed when the foreground script ends on most HPC systems. The process is detached from the shell but continues running in the same LSF job cgroup, holding the slot open.
- C) Correct — `mymonitor` holds the LSF job slot open for the remaining 2 hours (from job end to wall limit). This wastes 2 CPU-hours of allocated time and delays other users' jobs from starting on that slot.
- D) Incorrect — LSF does not automatically reap background child processes spawned by the job script unless the wall-clock limit is explicitly reached. There is no "orphan detection" grace period.

---

## Q7 — Correct Cleanup of Background Process

> **Week reference:** Week 11

**Mental Model:** Capture PID immediately after `&` with `$!`, run the main work, then `kill $PID` and `wait $PID`. The `wait` ensures the kill completes before the script exits. Missing any step leaves an orphan or a race condition.

Which job script pattern correctly ensures `mymonitor` is killed when the main script finishes?

- A) `mymonitor & ; python train.py`
- B) `mymonitor & MONITOR_PID=$! ; python train.py ; kill $MONITOR_PID`
- C) `MONITOR_PID=$! ; mymonitor & ; python train.py ; kill $MONITOR_PID ; wait $MONITOR_PID`
- D) `mymonitor & MONITOR_PID=$! ; python train.py ; kill $MONITOR_PID ; wait $MONITOR_PID`

**Answer: D**

- A) Incorrect — no PID is captured and no `kill` is issued. `mymonitor` runs as a background process with no cleanup mechanism, leaving a guaranteed orphan when the script exits.
- B) Incorrect — `$!` correctly captures `mymonitor`'s PID (it is evaluated after the `&` runs), but there is no `wait $MONITOR_PID` after the `kill`. Without `wait`, `kill` sends SIGTERM and returns immediately; the script then exits before `mymonitor` has actually terminated, which can leave it briefly as an orphan or produce a race condition.
- C) Incorrect — `$!` is captured *before* `mymonitor &` is executed, so it refers to the PID of whatever previously ran in the background (possibly none), not to `mymonitor`. The kill targets the wrong process.
- D) Correct — `mymonitor &` runs first; `$!` is immediately captured into `MONITOR_PID`; Python runs; then `kill $MONITOR_PID` sends SIGTERM; `wait $MONITOR_PID` blocks until the process actually exits. This is the complete and correct pattern.

---

## Q8 — Thread Environment Variables and Oversubscription

> **Week reference:** Week 11

**Mental Model:** OpenBLAS, MKL, and OpenMP query `os.cpu_count()` (the full hardware count) by default — they have no knowledge of LSF allocations. On a 32-core node where you requested 8 cores, each NumPy call can spawn 32 threads, using 4× your allocation and degrading everyone else on that node.

A job requests `#BSUB -n 8` (8 CPU cores). The job runs on a node with 32 physical cores. No thread count variables are set. A NumPy operation internally uses OpenBLAS. How many threads does OpenBLAS likely spawn?

- A) 8 — it reads the LSF allocation automatically.
- B) 1 — OpenBLAS defaults to single-threaded without explicit configuration.
- C) 32 — it defaults to the number of hardware cores on the node.
- D) 4 — OpenBLAS uses half the available cores as a conservative default.

**Answer: C**

- A) Incorrect — OpenBLAS has no awareness of LSF allocations or cgroups. It queries `sysconf(_SC_NPROCESSORS_ONLN)` directly, which returns the total hardware core count on the node, not the LSF-allocated count.
- B) Incorrect — OpenBLAS defaults to multi-threaded using all detectable cores. Single-threaded mode requires explicitly setting `OPENBLAS_NUM_THREADS=1` or using a single-threaded build.
- C) Correct — without `OPENBLAS_NUM_THREADS` being set, OpenBLAS spawns up to the hardware core count (32 in this case), causing oversubscription: the job uses 4× its allocated resources and interferes with co-resident jobs on the same node.
- D) Incorrect — there is no half-core heuristic in OpenBLAS; it uses all available hardware threads. The "use half" behaviour is not a feature of any standard BLAS library.

---

## Q9 — Fixing Thread Oversubscription

> **Week reference:** Week 11

**Mental Model:** Three independent threading backends — OpenMP, MKL, OpenBLAS — each respect only their own environment variable. Setting just one leaves the others unconstrained. Always set all three to match your `#BSUB -n` allocation.

A job requests 8 cores (`#BSUB -n 8`). To prevent NumPy/SciPy from oversubscribing, which environment variables should be set to `8` in the job script?

- A) `LSF_NUM_THREADS` only
- B) `OMP_NUM_THREADS` only
- C) `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS`
- D) `PYTHON_NUM_THREADS` and `NUMPY_NUM_THREADS`

**Answer: C**

- A) Incorrect — `LSF_NUM_THREADS` is not a real environment variable. LSF does not expose a standard variable that BLAS libraries read for thread limits. Setting this variable does nothing.
- B) Incorrect — `OMP_NUM_THREADS` caps OpenMP-based parallelism (used by some NumPy builds and SciPy routines) but leaves MKL-based and OpenBLAS-based thread pools uncapped. Those can still spawn up to 32 threads each.
- C) Correct — `OMP_NUM_THREADS` caps OpenMP; `MKL_NUM_THREADS` caps Intel MKL (used when NumPy is linked against MKL, e.g. on Anaconda); `OPENBLAS_NUM_THREADS` caps OpenBLAS (used in many conda/pip builds). All three must be set to prevent any backend from oversubscribing.
- D) Incorrect — neither `PYTHON_NUM_THREADS` nor `NUMPY_NUM_THREADS` exist as standard environment variables. NumPy does not expose a unified thread-count variable; you must configure each underlying library separately.

---

## Q10 — multiprocessing.Pool Default Worker Count

> **Week reference:** Week 11

**Mental Model:** `Pool()` calls `os.cpu_count()` internally, which returns the total hardware logical CPUs — not the LSF-allocated count. On a 32-core node with an 8-core allocation, `Pool()` spawns 32 workers, oversubscribing by 4× and slowing down all co-resident jobs.

A script submitted with `#BSUB -n 8` on a 32-core node contains:

```python
from multiprocessing import Pool
with Pool() as pool:
    results = pool.map(compute, data)
```

How many worker processes does `Pool()` create?

- A) 8 — it reads the LSF `-n` allocation.
- B) 4 — Python uses half the detected cores by default.
- C) 32 — it calls `os.cpu_count()` which returns the total hardware cores.
- D) 1 — `Pool()` with no arguments falls back to serial execution.

**Answer: C**

- A) Incorrect — `Pool()` with no arguments calls `os.cpu_count()`, which is completely unaware of LSF allocations or cgroups. Python has no built-in mechanism to read the LSF `-n` allocation.
- B) Incorrect — there is no half-core default in Python's multiprocessing. `os.cpu_count()` returns the full logical CPU count without any reduction factor.
- C) Correct — `os.cpu_count()` queries the OS for total logical CPUs (32 on a typical HPC node), so `Pool()` spawns 32 workers. The fix is to use `Pool(processes=int(os.environ.get("LSB_DJOB_NUMPROC", os.cpu_count())))` or hard-code the allocated count.
- D) Incorrect — `Pool()` with no arguments does NOT default to serial execution. It spawns `os.cpu_count()` worker processes, which is the oversubscription problem being tested.

---

## Q11 — Map-Reduce Job Dependency

> **Week reference:** Week 11

**Mental Model:** `done()` = all specified jobs reached DONE (success only). `ended()` = any termination including EXIT (failure). For a cleanup/reduce job that should only run after all map jobs succeed, use `done()`. Using `ended()` would run reduce even if half the map jobs failed.

You have a map phase as job array `#BSUB -J map[1-20]` and a reduce job that should run only after ALL map elements succeed. Which `#BSUB` directive achieves this on the reduce job?

- A) `#BSUB -w "done(map)"`
- B) `#BSUB -w "ended(map[1-20])"`
- C) `#BSUB -w "done(map[*])"`
- D) `#BSUB -hold_jid map`

**Answer: A**

- A) Correct — `done(map)` waits until ALL elements of the array named `map` reach DONE status (successful completion). Using the array name without a subscript applies the condition to the entire array, making the reduce job wait for every element.
- B) Incorrect — `ended` is satisfied when jobs end in any state, including EXIT (failure). If 5 out of 20 map jobs fail, the reduce job would still run on incomplete data, which is almost certainly wrong for a reduce step.
- C) Incorrect — `map[*]` is not valid LSF dependency syntax. The correct form uses the array name without any subscript notation: `done(map)`. The `[*]` glob is a shell or SLURM construct, not LSF.
- D) Incorrect — `-hold_jid` is a Sun Grid Engine (SGE/UGE) directive, not LSF/BSUB syntax. On DTU HPC (which uses LSF), this directive is silently ignored or causes a submission error. The LSF equivalent uses `-w`.

---

## Q12 — Login Node vs Compute Node

> **Week reference:** Week 11

**Mental Model:** The login node is a shared gateway used by all users for file transfers, job submission, and light editing. Running computation there degrades the experience for everyone and violates HPC cluster etiquette. Even "just 5 minutes" of CPU-intensive work is unacceptable — use `bsub` or `linuxsh` for interactive work.

A student connects to `login.hpc.dtu.dk` and runs a CPU-intensive simulation directly in the terminal. What is the correct response?

- A) This is fine for short jobs under 10 minutes.
- B) The login node is shared; computation there degrades performance for all users and violates HPC policy.
- C) The login node has more RAM so it is actually better for memory-heavy workloads.
- D) Computation on the login node is automatically migrated to a compute node by LSF.

**Answer: B**

- A) Incorrect — even "short" computation on the login node is not acceptable practice. The 10-minute threshold does not exist in DTU HPC policy; even a 2-minute heavy job can noticeably slow login and file operations for all users. Use `linuxsh` for any interactive computation.
- B) Correct — the login node is a lightweight shared gateway. Running CPU-intensive work there consumes shared resources (CPU time, RAM), slows down SSH logins for everyone, and violates cluster usage policy. The correct approach is `bsub` for batch jobs or `linuxsh`/`bsub -Is` for interactive sessions on compute nodes.
- C) Incorrect — login nodes are not provisioned with extra RAM for computation. They are typically configured with moderate RAM for handling many concurrent SSH sessions, not for running large simulations.
- D) Incorrect — LSF does not auto-migrate processes started outside of `bsub`. Any process started directly in the terminal runs on the login node and stays there; LSF only manages jobs explicitly submitted via `bsub`.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets job arrays with $LSB_JOBINDEX, job dependencies (done vs ended), output file naming, and pipeline construction

---

## Q13 — done() vs ended() Critical Distinction

> **Week reference:** Week 11

A 50-element job array named `preprocess` is submitted. Due to a data error, 4 of the 50 jobs exit with a non-zero return code (EXIT state). A downstream job was submitted with `#BSUB -w "done(preprocess)"`. What happens to the downstream job?

- A) It starts after the 46 successful jobs complete; the 4 failures are tolerated.
- B) It remains permanently stuck in PEND state because `done()` requires every element to reach DONE.
- C) It starts after all 50 jobs have left RUN state, regardless of exit code.
- D) It starts after a 10-minute timeout, even though some jobs failed.

**Answer: B**

`done()` is an all-or-nothing condition: every element of the named array must reach DONE (exit code 0). If even one element reaches EXIT, the `done()` condition is permanently false and the downstream job waits forever in PEND. There is no timeout or partial-success tolerance. A) describes the behaviour of `ended()`, not `done()`. C) also describes `ended()`. D) is invented; LSF has no dependency timeout mechanism.

---

## Q14 — ended() Permissive Dependency

> **Week reference:** Week 11

A pipeline runs preprocessing then aggregation. The aggregation step must run even if some preprocessing jobs fail (it will handle missing inputs gracefully). Which dependency directive on the aggregation job is correct?

- A) `#BSUB -w "done(preprocess)"`
- B) `#BSUB -w "ended(preprocess)"`
- C) `#BSUB -w "exit(preprocess)"`
- D) `#BSUB -w "finished(preprocess)"`

**Answer: B**

`ended()` is satisfied when all jobs in the named group have finished — whether DONE or EXIT. This is the correct choice when the downstream job should always run regardless of upstream failures. A) uses `done()`, which blocks permanently if any job fails. C) `exit()` is not a valid LSF dependency keyword. D) `finished()` does not exist in LSF dependency syntax.

---

## Q15 — Output File %J vs %I

> **Week reference:** Week 11

A 100-element job array is submitted with `#BSUB -o log_%I.out`. Job array index 47 runs as part of job submission ID 98765. What is the output filename for that element?

- A) `log_98765.out`
- B) `log_47_98765.out`
- C) `log_47.out`
- D) `log_98765_47.out`

**Answer: C**

`%I` expands to the array index, so the file is `log_47.out`. A) would result from `%J` alone (just the job ID). B) and D) would require `%I_%J` or `%J_%I` respectively. The directive only contains `%I`, so only the array index appears in the filename.

---

## Q16 — Three-Stage Pipeline Construction

> **Week reference:** Week 11

You need to build a three-stage pipeline: stage1 (array [1-20]), stage2 (array [1-20], runs after stage1 succeeds), stage3 (single job, runs after stage2 succeeds). Which sequence of `bsub` commands is correct?

- A) `bsub -J "s1[1-20]" s1.sh` then `bsub -w "done(s1)" -J "s2[1-20]" s2.sh` then `bsub -w "done(s2)" s3.sh`
- B) `bsub -J "s1[1-20]" s1.sh` then `bsub -J "s2[1-20]" -w "ended(s1)" s2.sh` then `bsub -w "ended(s2)" s3.sh`
- C) `bsub -J "s1[1-20]" s1.sh && bsub -J "s2[1-20]" s2.sh && bsub s3.sh`
- D) `bsub -J "s1[1-20]" s1.sh` then `bsub -hold_jid s1 -J "s2[1-20]" s2.sh` then `bsub -hold_jid s2 s3.sh`

**Answer: A**

A) correctly uses `done()` at both dependency stages, ensuring each stage only runs when the previous stage completed successfully. B) uses `ended()` which allows downstream stages to run even after failures — incorrect when success is required. C) uses shell `&&` which chains submission commands, not job execution order — all three jobs are submitted immediately and run in parallel. D) uses `-hold_jid` which is SGE/UGE syntax, not valid LSF/BSUB.

---

## Q17 — Concurrent Array Job Limit

> **Week reference:** Week 11

A job array is submitted with `#BSUB -J "bigrun[1-500]%20"`. What does the `%20` control?

- A) The step size — only indices 1, 21, 41, ... are created.
- B) The maximum number of array elements that can run concurrently (at most 20 at a time).
- C) The wall-clock limit in minutes for each array element.
- D) The number of CPU cores requested per array element.

**Answer: B**

The `%N` suffix after the range in LSF job array syntax limits concurrency: at most N elements run simultaneously. This prevents flooding the cluster with all 500 jobs at once. A) describes step syntax, which uses `[start-end:step]` (a colon, not a percent). C) wall-clock limit uses `#BSUB -W`. D) core count uses `#BSUB -n`.

---

## Q18 — LSB_JOBINDEX Off-by-One in Python

> **Week reference:** Week 11

A Python script accesses a list of 8 configuration files. The job array is `#BSUB -J "run[1-8]"`. Which line correctly selects the configuration for each array element?

- A) `config = configs[os.environ["LSB_JOBINDEX"]]`
- B) `config = configs[int(os.environ["LSB_JOBINDEX"])]`
- C) `config = configs[int(os.environ["LSB_JOBINDEX"]) - 1]`
- D) `config = configs[int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1]`

**Answer: C**

`$LSB_JOBINDEX` is 1-based (1 through 8) but Python list indices are 0-based (0 through 7). Subtracting 1 maps them correctly: element 1 → `configs[0]`, element 8 → `configs[7]`. A) passes a string to the index operator, raising `TypeError`. B) passes the integer directly without subtracting 1 — element 1 accesses `configs[1]` (skips first), and element 8 raises `IndexError`. D) uses the SLURM variable name, which does not exist in LSF and raises `KeyError`.

---

## Q19 — Dependency on Specific Job ID

> **Week reference:** Week 11

A user submits a preprocessing job and it is assigned job ID 54321. They want to submit an analysis job that only runs after job 54321 completes successfully. Which directive is correct?

- A) `#BSUB -w "done(preprocess)"`
- B) `#BSUB -w "done(54321)"`
- C) `#BSUB -w "ended(54321)"`
- D) `#BSUB -after 54321`

**Answer: B**

LSF `-w` accepts either a job name or a numeric job ID inside `done()` or `ended()`. `done(54321)` waits for job ID 54321 to reach DONE (success). A) uses a name — if the submitted job was not named `preprocess`, this would either match a different job or block indefinitely. C) uses `ended()` which would also run if job 54321 failed. D) `-after` is not a valid LSF directive.

---

## Q20 — AND Dependency: Both Jobs Must Succeed

> **Week reference:** Week 11

A final report job must run only after BOTH a data job (named `data`) and a model job (named `model`) complete successfully. Which directive expresses this?

- A) `#BSUB -w "done(data) || done(model)"`
- B) `#BSUB -w "done(data,model)"`
- C) `#BSUB -w "done(data) && done(model)"`
- D) `#BSUB -w "done(data+model)"`

**Answer: C**

LSF dependency expressions support `&&` (AND) and `||` (OR) logical operators. `done(data) && done(model)` requires both to reach DONE before the downstream job starts. A) uses `||` (OR), meaning the report starts as soon as either one finishes — incorrect. B) a comma inside `done()` is not valid LSF dependency syntax. D) uses `+` which is not a valid operator in LSF dependency expressions.

---

## Q21 — What %I Expands To in a Running Job

> **Week reference:** Week 11

A job array is submitted with `#BSUB -J "exp[1-30]"` and `#BSUB -o results_%J_%I.out`. The entire job array is assigned submission ID 11111. When array element 15 runs, what is its output filename?

- A) `results_11111_15.out`
- B) `results_15_11111.out`
- C) `results_11111.out`
- D) `results_15.out`

**Answer: A**

The directive `results_%J_%I.out` expands `%J` to the parent job ID (11111) and `%I` to the array index (15), giving `results_11111_15.out`. B) reverses the order from what the directive specifies. C) omits `%I` expansion — that would result from `results_%J.out`. D) omits `%J` expansion — that would result from `results_%I.out`.

---

## Q22 — Danger of Missing %I in Output Directive

> **Week reference:** Week 11

A job array `#BSUB -J "proc[1-10]"` uses the output directive `#BSUB -o output_%J.out`. All 10 elements run concurrently on different nodes. What is the result?

- A) Each element creates a uniquely named file; output is clean.
- B) All 10 elements write concurrently to the same file, producing interleaved and potentially corrupted output.
- C) LSF automatically appends the array index to the filename even without `%I`.
- D) Only the element with the lowest index writes output; the others discard their stdout.

**Answer: B**

Without `%I`, `%J` expands to the same parent job ID for all 10 elements. Every element therefore opens the identical file `output_<jobID>.out`. With 10 processes writing simultaneously to the same file from different nodes, writes interleave unpredictably and output is corrupted or mixed. The fix is to use `%J_%I`. C) is false — LSF never auto-appends the index. D) is false — all elements write output, causing the collision.

---
