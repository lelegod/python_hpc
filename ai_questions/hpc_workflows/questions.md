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
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q23 — Explicit Index List in Job Array](#q23--explicit-index-list-in-job-array)
- [Q24 — Per-Core Memory Calculation With rusage](#q24--per-core-memory-calculation-with-rusage)
- [Q25 — span[hosts=1] Purpose in Job Scripts](#q25--spanhosts1-purpose-in-job-scripts)
- [Q26 — Element-Level Dependency With done(array[*])](#q26--element-level-dependency-with-donearray)
- [Q27 — LSB_DJOB_NUMPROC for Correct Core Count](#q27--lsb_djob_numproc-for-correct-core-count)
- [Q28 — Rerunning a Single Failed Array Element](#q28--rerunning-a-single-failed-array-element)
- [Q29 — -B vs -N Notification Flags](#q29---b-vs--n-notification-flags)
- [Q30 — OR Dependency With Two Named Jobs](#q30--or-dependency-with-two-named-jobs)
- [Q31 — Fan-Out Fan-In: Naming Output Files Per Element](#q31--fan-out-fan-in-naming-output-files-per-element)
- [Q32 — Maximum Array Size Limit Consequence](#q32--maximum-array-size-limit-consequence)

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

## Set 3 — Extended Practice

> Targets explicit index lists, per-core memory, span[hosts=1], element-level dependencies, LSB_DJOB_NUMPROC, rerunning failed elements, notification flags, OR dependencies, fan-out/fan-in output naming, and array size limits.

---

## Q23 — Explicit Index List in Job Array

> **Week reference:** Week 11

**Mental Model:** LSF allows a comma-separated list of specific indices instead of a range. This is useful when you need to rerun only a handful of failed elements, or when your dataset has gaps. The indices do not have to be contiguous or sorted.

A developer wants to submit a job array that runs only for indices 2, 29, 71, 73, and 127 — matching specific files in a dataset. Which `#BSUB` directive achieves this?

- A) `#BSUB -J "proc[2-127:25]"`
- B) `#BSUB -J "proc[2,29,71,73,127]"`
- C) `#BSUB -J "proc[2..127]" -subset 2,29,71,73,127`
- D) `#BSUB -J "proc[1-5]"` with a lookup table inside the script

**Answer: B**

- A) Incorrect — `[2-127:25]` is step syntax starting at 2 with step 25, giving indices 2, 27, 52, 77, 102, 127. These do not match the desired set; index 29, 71, and 73 are missing.
- B) Correct — LSF supports an explicit comma-separated index list inside the brackets. `proc[2,29,71,73,127]` creates exactly 5 array elements with those specific `$LSB_JOBINDEX` values. This is the standard way to rerun a known set of failed elements.
- C) Incorrect — `..` is not valid LSF range syntax, and `-subset` is not an LSF directive. This would cause a job submission error.
- D) Incorrect — using `[1-5]` with an internal lookup table works but is a workaround that introduces extra complexity and still creates indices 1–5 rather than the semantically meaningful target indices. The direct syntax in B is the correct approach.

---

## Q24 — Per-Core Memory Calculation With rusage

> **Week reference:** Week 11

**Mental Model:** The `rusage[mem=X]` value in LSF is the memory **per slot (core)**, not the total job memory. If your job requests `#BSUB -n 4` and needs 100 GB total, you must request 25 GB per core. Setting the full amount per core would over-request 4× and waste resources or prevent scheduling.

A job script contains `#BSUB -n 4` and the Python script it runs requires at least 100 GB of RAM in total. What value should replace `???` in `#BSUB -R "rusage[mem=???GB]"`?

- A) 25GB — divide total memory by the number of requested cores
- B) 100GB — always specify the total memory the job needs
- C) 400GB — LSF divides by cores automatically so you must multiply up
- D) 50GB — use half the total as a conservative estimate

**Answer: A**

- A) Correct — `rusage[mem=X]` is per core in LSF. With 4 cores requested, the per-core value is 100 GB ÷ 4 = 25 GB. LSF multiplies this by the core count when enforcing the allocation, so the total reserved memory is 100 GB as required.
- B) Incorrect — specifying 100 GB per core on a 4-core job would request 400 GB total from the scheduler. This wastes three times as much memory as needed and may delay or prevent scheduling if nodes do not have 400 GB free.
- C) Incorrect — LSF does not divide your value by the core count. What you specify is already the per-core amount. Providing 400 GB per core would request 1600 GB total, which is almost certainly unavailable on any node.
- D) Incorrect — there is no "half as conservative" heuristic in LSF memory requests. Under-requesting memory risks the job being killed by the OOM killer when it exceeds the reserved amount.

---

## Q25 — span[hosts=1] Purpose in Job Scripts

> **Week reference:** Week 11

**Mental Model:** By default, LSF may spread a multi-core job across multiple nodes (e.g., 2 cores on node A and 2 cores on node B). `span[hosts=1]` forces all requested cores to be allocated on a single node. This is required for shared-memory parallelism (multiprocessing, threading) because processes on different nodes cannot share memory.

A job script requests 8 cores with `#BSUB -n 8` and also includes `#BSUB -R "span[hosts=1]"`. What does the `span[hosts=1]` resource requirement do?

- A) It limits the job to run on exactly 1 compute node total, ensuring all 8 cores are on the same physical machine.
- B) It requests 1 dedicated GPU host in addition to the 8 CPU cores.
- C) It restricts the job to a single rack of hosts to minimise network latency.
- D) It submits one copy of the job to each available host simultaneously.

**Answer: A**

- A) Correct — `span[hosts=1]` is a placement constraint that forces LSF to allocate all `n` cores on a single node. This is essential for any shared-memory parallel program (Python multiprocessing, OpenMP, threading) because inter-node communication is not possible through shared memory.
- B) Incorrect — `span[hosts=1]` has nothing to do with GPU allocation. GPU resources are requested separately via `rusage[ngpus_excl_p=N]` or similar directives depending on the cluster configuration.
- C) Incorrect — LSF does not have a rack-level placement constraint expressed through `span[hosts=1]`. Rack-level affinity would require a different resource string if supported at all.
- D) Incorrect — `span[hosts=1]` is a constraint that concentrates resources, not a fan-out directive. It reduces the host count to exactly 1, the opposite of distributing across many hosts.

---

## Q26 — Element-Level Dependency With done(array[*])

> **Week reference:** Week 11

**Mental Model:** `done(jobname[*])` is a special form that lets each element of a new array wait for the corresponding element of a previously submitted array. Element 1 of the new array waits for element 1 of the old array, element 2 waits for element 2, and so on. This enables element-wise pipelines without a single bottleneck reduce step.

An array `stage1[1-5]` is already running. A new array `stage2[1-5]` is submitted where each element must wait only for its matching `stage1` element (element 2 waits for element 2, not all five). Which dependency directive achieves this?

- A) `#BSUB -w "done(stage1)"`
- B) `#BSUB -w "done(stage1[*])"`
- C) `#BSUB -w "done(stage1[1-5])"`
- D) `#BSUB -w "done(stage1) && started(stage2)"`

**Answer: B**

- A) Incorrect — `done(stage1)` without a subscript means every element of `stage2` waits for ALL elements of `stage1` to reach DONE before any `stage2` element can start. This is a fan-in, not a per-element dependency.
- B) Correct — `done(stage1[*])` uses the `[*]` wildcard to express an element-wise dependency: each element of the new array waits for the correspondingly indexed element of `stage1`. This is the standard LSF syntax for element-level chaining.
- C) Incorrect — `done(stage1[1-5])` is not valid LSF dependency syntax. Range subscripts inside dependency expressions are not supported; only `[*]` and explicit single indices like `[3]` are valid.
- D) Incorrect — `started(stage2)` is not a valid LSF dependency keyword. Valid keywords are `done`, `ended`, `exit`, `started` is not supported in this context. The expression is syntactically invalid.

---

## Q27 — LSB_DJOB_NUMPROC for Correct Core Count

> **Week reference:** Week 11

**Mental Model:** `LSB_DJOB_NUMPROC` is the LSF environment variable that contains the number of CPU slots actually allocated to the job (matching `#BSUB -n`). Reading this at runtime instead of calling `os.cpu_count()` lets Python multiprocessing respect the LSF allocation regardless of the total hardware on the node.

A job script requests 8 cores. To fix the `Pool()` oversubscription problem, which Python line correctly creates a pool with exactly the LSF-allocated core count?

- A) `with Pool(processes=os.cpu_count()) as pool:`
- B) `with Pool(processes=8) as pool:`
- C) `with Pool(processes=int(os.environ["LSB_DJOB_NUMPROC"])) as pool:`
- D) `with Pool(processes=int(os.environ["OMP_NUM_THREADS"])) as pool:`

**Answer: C**

- A) Incorrect — `os.cpu_count()` returns the total hardware logical CPUs on the node (e.g., 32), not the LSF-allocated count. This is the original oversubscription problem.
- B) Incorrect — hard-coding 8 works for this specific job but is fragile: if someone changes `#BSUB -n` without updating the Python script, the mismatch silently returns. It is not portable across different job submissions.
- C) Correct — `LSB_DJOB_NUMPROC` is set by LSF to the number of allocated CPU slots, matching the `#BSUB -n` value. Reading it dynamically ensures the pool size always equals the actual allocation, regardless of what hardware is under the job or what `#BSUB -n` is set to.
- D) Incorrect — `OMP_NUM_THREADS` controls OpenMP thread count, not Python multiprocessing worker count. `Pool()` does not read `OMP_NUM_THREADS`. If `OMP_NUM_THREADS` is unset, this raises `KeyError`.

---

## Q28 — Rerunning a Single Failed Array Element

> **Week reference:** Week 11

**Mental Model:** A job array is one logical submission, but individual elements can be rerun by specifying the parent job ID with a subscript index: `bsub -J "name[k]" ... script.sh` does not work for rerunning; instead you resubmit with an explicit single-element array or use `bkill` + resubmit. The cleanest pattern is to submit a new single-element array with the exact failed index.

A 100-element job array (`proc[1-100]`) finishes, but elements 7, 43, and 88 failed. What is the correct command to rerun only those three elements?

- A) `brerun proc[7,43,88]`
- B) `bsub -J "proc[7,43,88]" proc.sh`
- C) `bsub -restart proc 7 43 88`
- D) `bjobs -rerun 7,43,88`

**Answer: B**

- A) Incorrect — `brerun` is not a valid LSF command. LSF does not have a built-in `brerun` utility; failed elements must be resubmitted as a new job.
- B) Correct — submitting a new job array with the explicit index list `[7,43,88]` creates a new 3-element array using exactly those `$LSB_JOBINDEX` values. The script receives the same indices as before and can reprocess the same data. This is the standard LSF approach to rerunning specific failed elements.
- C) Incorrect — `bsub -restart` is not a valid LSF option. There is no restart mechanism built into LSF for rerunning specific failed elements of a previously submitted array.
- D) Incorrect — `bjobs` is a query command for viewing job status; it does not accept a `-rerun` flag. `bjobs` cannot submit, restart, or requeue jobs.

---

## Q29 — -B vs -N Notification Flags

> **Week reference:** Week 11

**Mental Model:** LSF has two separate email notification flags: `-B` sends an email when the job transitions from PEND to RUN (job starts), and `-N` sends an email when the job ends. Both can be combined. On large job arrays, both flags multiply by the element count, potentially causing hundreds of emails.

A job array `#BSUB -J "run[1-10]"` includes both `#BSUB -B` and `#BSUB -N`. How many emails are sent in total when all 10 elements start and finish successfully?

- A) 2 emails total — one start email and one end email for the whole array
- B) 10 emails — one end email per element (only `-N` sends email; `-B` is for start events but counts once per array)
- C) 20 emails — `-B` sends 10 start emails and `-N` sends 10 end emails, one per element
- D) 1 email — LSF batches all notifications into a single summary when both flags are used together

**Answer: C**

- A) Incorrect — like `-N`, the `-B` flag sends notifications per array element, not per array. There is no array-level aggregation for either flag.
- B) Incorrect — `-B` does send per-element start notifications exactly like `-N` sends per-element end notifications. Both flags operate at the element level for job arrays.
- C) Correct — `-B` sends one email per element when it starts (10 emails) and `-N` sends one email per element when it ends (10 emails), totalling 20 emails. This is why both flags should be avoided on large arrays.
- D) Incorrect — there is no batching or summary mode when both `-B` and `-N` are used together. They operate independently and each fires once per element event.

---

## Q30 — OR Dependency With Two Named Jobs

> **Week reference:** Week 11

**Mental Model:** LSF dependency expressions support `||` (OR) in addition to `&&` (AND). An OR dependency starts the job as soon as either condition is satisfied. This is useful for "whichever data source finishes first" patterns, or for resilient pipelines where the downstream job can tolerate incomplete upstream data.

A reporting job should start as soon as EITHER a `modelA` job OR a `modelB` job reaches DONE (the report can work with either model's output). Which dependency directive is correct?

- A) `#BSUB -w "done(modelA) && done(modelB)"`
- B) `#BSUB -w "done(modelA,modelB)"`
- C) `#BSUB -w "done(modelA) || done(modelB)"`
- D) `#BSUB -w "ended(modelA) || ended(modelB)"`

**Answer: C**

- A) Incorrect — `&&` (AND) requires both `modelA` AND `modelB` to reach DONE before the report starts. This is the opposite of the desired "either one" behaviour.
- B) Incorrect — a comma inside `done()` is not valid LSF dependency syntax. This would cause a submission error or be misinterpreted; the correct multi-job AND form uses `&&`.
- C) Correct — `||` (OR) means the condition is satisfied as soon as either `done(modelA)` or `done(modelB)` becomes true. The reporting job starts as soon as the first model completes successfully.
- D) Incorrect — `ended()` is satisfied by any terminal state including EXIT (failure). If either model job fails, `ended(modelA) || ended(modelB)` would trigger the report even though neither model produced valid output. The question requires success (`done()`), not just termination.

---

## Q31 — Fan-Out Fan-In: Naming Output Files Per Element

> **Week reference:** Week 11

**Mental Model:** In a fan-out/fan-in pattern, each array element writes a partial result to its own file, then a single reduce job loads all partial files. The naming convention must be deterministic and based on the array index so the reduce job can discover all partial files with a glob pattern like `subhist_*.npy`.

In a fan-out workflow, each element of a 203-element array processes one folder and saves a partial histogram. Which Python line correctly saves the result so that the reduce job can later load all files with `glob("subhist_*.npy")`?

- A) `np.save("subhist.npy", hist)`
- B) `np.save(f"subhist_{idx}.npy", hist)` where `idx = int(sys.argv[1]) - 1`
- C) `np.save(f"subhist_{os.environ['LSB_JOBINDEX']}.npy", hist)`
- D) Both B and C are correct; either naming convention works equally well

**Answer: D**

- A) Incorrect — a fixed filename means all 203 elements overwrite each other's output. Only the last element's histogram would survive; the reduce job would load a single file instead of 203.
- B) Correct on its own — `idx` here is the 0-based Python index (after subtracting 1), giving filenames `subhist_0.npy` through `subhist_202.npy`. These are all found by `glob("subhist_*.npy")`.
- C) Correct on its own — using `$LSB_JOBINDEX` directly (1-based) gives `subhist_1.npy` through `subhist_203.npy`. These are also found by `glob("subhist_*.npy")`. The starting index differs from B but both sets are unique and complete.
- D) Correct — both B and C produce 203 uniquely named files that `glob("subhist_*.npy")` will find. The reduce job sums all loaded arrays regardless of the numbering scheme. Either convention is valid; consistency within the project is what matters.

---

## Q32 — Maximum Array Size Limit Consequence

> **Week reference:** Week 11

**Mental Model:** LSF clusters impose a maximum array size (e.g., 1000 elements on some clusters). Attempting to submit an array larger than the limit causes the submission to fail immediately with an error, not a silent reduction. The fix is to either split the work into multiple smaller arrays submitted sequentially, or use the `%N` concurrency limit to throttle a large array if the cluster permits it.

A cluster has a maximum job array size of 1000 elements. A user tries to submit `#BSUB -J "sim[1-5000]"`. What happens?

- A) LSF silently truncates the array to 1000 elements and runs those.
- B) The submission fails immediately with an error; the job is not queued at all.
- C) LSF queues all 5000 elements but runs only 1000 concurrently until the others are promoted.
- D) The array is split automatically into 5 batches of 1000 and submitted as separate jobs.

**Answer: B**

- A) Incorrect — LSF does not silently truncate arrays. Silent data loss would be catastrophic in scientific workflows. The scheduler rejects the submission outright so the user knows to fix the script.
- B) Correct — exceeding the cluster's maximum array size causes an immediate submission error (e.g., "array size exceeds maximum"). The job is not queued, not partially queued, and not silently modified. The user must split the work manually or use the `%N` concurrency throttle if a larger array is permitted with throttling.
- C) Incorrect — the `%N` concurrency throttle limits simultaneous running elements, but only after a valid array is successfully submitted. It does not allow bypassing the maximum array size limit during submission.
- D) Incorrect — LSF has no automatic batch-splitting feature. The user must manually submit multiple smaller arrays or write a wrapper script to chain them with dependencies.

---
