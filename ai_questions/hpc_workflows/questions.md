# HPC Workflows — MCQ Practice

> Topics: Job arrays, $LSB_JOBINDEX, background processes, thread env vars, oversubscription.
> Exam frequency: **2024 exam + F25** (job arrays, dependencies).

---

## Q1 — Job Array Index Range

You submit a job array with `#BSUB -J myjob[1-10]`. What values does `$LSB_JOBINDEX` take across the 10 array elements?

- A) 0, 1, 2, ..., 9
- B) 1, 2, 3, ..., 10
- C) 0, 1, 2, ..., 10 (11 values)
- D) 1, 2, 3, ..., 9 (9 values)

**Answer: B**

- A) Incorrect — $LSB_JOBINDEX is 1-based, not 0-based; index 0 does not exist.
- B) Correct — LSF numbers array elements starting from 1, giving exactly the range specified.
- C) Incorrect — The range [1-10] produces exactly 10 elements, not 11.
- D) Incorrect — [1-10] is inclusive on both ends, so index 10 is included.

---

## Q2 — Job Array Step Syntax

You submit `#BSUB -J sim[1-100:2]`. Which indices are created?

- A) 1, 2, 3, ..., 100 (all 100)
- B) 2, 4, 6, ..., 100 (even numbers)
- C) 1, 3, 5, ..., 99 (odd numbers)
- D) 0, 2, 4, ..., 98 (even numbers starting at 0)

**Answer: C**

- A) Incorrect — The `:2` step means every second index is skipped; not all 100 are created.
- B) Incorrect — Starting from 1 with a step of 2 gives odd numbers, not even.
- C) Correct — Starting at 1 and incrementing by 2 yields 1, 3, 5, ..., 99 (50 elements total).
- D) Incorrect — LSF arrays are 1-based; index 0 never exists.

---

## Q3 — Zero-Based Array Access in Python

A Python script is submitted as part of a job array. The script reads a list of input files called `file_list`. Which line correctly picks the file for the current array element?

- A) `fname = file_list[int(os.environ["LSB_JOBINDEX"])]`
- B) `fname = file_list[int(os.environ["LSB_JOBINDEX"]) - 1]`
- C) `fname = file_list[int(os.environ["LSB_JOBINDEX"]) + 1]`
- D) `fname = file_list[int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1]`

**Answer: B**

- A) Incorrect — $LSB_JOBINDEX starts at 1; index 1 on a list skips the first element.
- B) Correct — Subtracting 1 converts the 1-based LSF index to a 0-based Python list index.
- C) Incorrect — Adding 1 would skip one extra element and cause an off-by-one error.
- D) Incorrect — SLURM_ARRAY_TASK_ID is a SLURM variable; DTU HPC uses LSF/BSUB.

---

## Q4 — Per-Element Log Files

You want each element of a 50-element job array to write to its own log file. Which `#BSUB` output directive achieves this?

- A) `#BSUB -o results_%J.out`
- B) `#BSUB -o results_%I.out`
- C) `#BSUB -o results_%J_%I.out`
- D) `#BSUB -o results.out`

**Answer: C**

- A) Incorrect — `%J` expands to the parent job ID only; all 50 elements share the same file and outputs are interleaved.
- B) Incorrect — `%I` gives the array index but omits the job ID, which can cause collisions between different runs.
- C) Correct — `%J` (job ID) + `%I` (array index) together produce a unique filename per element per run.
- D) Incorrect — A fixed filename means all 50 elements write to the same file, mixing output unpredictably.

---

## Q5 — Email Notification with Job Arrays

A developer adds `#BSUB -N` to a job array script `#BSUB -J process[1-200]` so they know when the work is done. What is the consequence?

- A) One summary email is sent when all 200 elements complete.
- B) One email is sent per array element, totalling 200 emails.
- C) No email is sent; -N is not valid with job arrays.
- D) Two emails per element (start + end), totalling 400 emails.

**Answer: B**

- A) Incorrect — LSF treats each array element as an independent job for notification purposes.
- B) Correct — `-N` sends one email per array element, which means 200 emails flood the inbox.
- C) Incorrect — `-N` is syntactically valid with job arrays; it just behaves unexpectedly at scale.
- D) Incorrect — `-N` sends an end notification only, not a start notification; but there is still one per element.

---

## Q6 — Orphan Background Process

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

- A) Incorrect — The shell script exits after Python finishes, but the backgrounded `mymonitor` process keeps running.
- B) Incorrect — Background processes are not automatically killed when the foreground script ends on most HPC systems.
- C) Correct — The orphan process holds the job slot open, wasting 2 hours of allocated compute time.
- D) Incorrect — LSF does not automatically reap background child processes spawned by the job script.

---

## Q7 — Correct Cleanup of Background Process

Which job script pattern correctly ensures `mymonitor` is killed when the main script finishes?

- A) `mymonitor & ; python train.py`
- B) `mymonitor & MONITOR_PID=$! ; python train.py ; kill $MONITOR_PID`
- C) `MONITOR_PID=$! ; mymonitor & ; python train.py ; kill $MONITOR_PID ; wait $MONITOR_PID`
- D) `mymonitor & MONITOR_PID=$! ; python train.py ; kill $MONITOR_PID ; wait $MONITOR_PID`

**Answer: D**

- A) Incorrect — No PID is captured and no kill is issued, leaving an orphan process.
- B) Incorrect — `$!` must be captured immediately after `&`; this line assigns `$!` before launching `mymonitor`.
- C) Incorrect — `$!` is captured before `mymonitor &`, so it refers to a different (or no) process.
- D) Correct — `mymonitor &` runs first, `$!` captures its PID immediately, Python runs, then the monitor is killed and waited on.

---

## Q8 — Thread Environment Variables and Oversubscription

A job requests `#BSUB -n 8` (8 CPU cores). The job runs on a node with 32 physical cores. No thread count variables are set. A NumPy operation internally uses OpenBLAS. How many threads does OpenBLAS likely spawn?

- A) 8 — it reads the LSF allocation automatically.
- B) 1 — OpenBLAS defaults to single-threaded without explicit configuration.
- C) 32 — it defaults to the number of hardware cores on the node.
- D) 4 — OpenBLAS uses half the available cores as a conservative default.

**Answer: C**

- A) Incorrect — OpenBLAS has no awareness of LSF allocations; it queries the OS directly.
- B) Incorrect — OpenBLAS defaults to multi-threaded using all cores it can detect.
- C) Correct — Without `OPENBLAS_NUM_THREADS` being set, OpenBLAS spawns threads up to the hardware core count (32).
- D) Incorrect — There is no half-core heuristic; OpenBLAS uses all available cores by default.

---

## Q9 — Fixing Thread Oversubscription

A job requests 8 cores (`#BSUB -n 8`). To prevent NumPy/SciPy from oversubscribing, which environment variables should be set to `8` in the job script?

- A) `LSF_NUM_THREADS` only
- B) `OMP_NUM_THREADS` only
- C) `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS`
- D) `PYTHON_NUM_THREADS` and `NUMPY_NUM_THREADS`

**Answer: C**

- A) Incorrect — `LSF_NUM_THREADS` is not a real variable; LSF does not set thread limits through such an env var.
- B) Incorrect — Setting only OpenMP threads leaves MKL and OpenBLAS free to use all hardware cores.
- C) Correct — All three commonly used threading backends must be explicitly capped to match the allocated core count.
- D) Incorrect — Neither `PYTHON_NUM_THREADS` nor `NUMPY_NUM_THREADS` exist as standard environment variables.

---

## Q10 — multiprocessing.Pool Default Worker Count

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

- A) Incorrect — `Pool()` with no arguments calls `os.cpu_count()`, which is unaware of LSF allocations.
- B) Incorrect — There is no half-core default in Python's multiprocessing.
- C) Correct — `os.cpu_count()` returns the total logical CPUs on the node (32), causing oversubscription.
- D) Incorrect — `Pool()` with no arguments does NOT default to serial; it spawns `os.cpu_count()` workers.

---

## Q11 — Map-Reduce Job Dependency

You have a map phase as job array `#BSUB -J map[1-20]` and a reduce job that should run only after ALL map elements succeed. Which `#BSUB` directive achieves this on the reduce job?

- A) `#BSUB -w "done(map)"`
- B) `#BSUB -w "ended(map[1-20])"`
- C) `#BSUB -w "done(map[*])"`
- D) `#BSUB -hold_jid map`

**Answer: A**

- A) Correct — `done(map)` waits until ALL elements of the array named `map` reach DONE status.
- B) Incorrect — `ended` is satisfied when jobs end in any state (including EXIT/failure), not just successful completion.
- C) Incorrect — `map[*]` is not valid LSF dependency syntax; the correct form uses the array name without subscript.
- D) Incorrect — `-hold_jid` is a Sun Grid Engine / UGE directive, not LSF/BSUB syntax.

---

## Q12 — Login Node vs Compute Node

A student connects to `login.hpc.dtu.dk` and runs a CPU-intensive simulation directly in the terminal. What is the correct response?

- A) This is fine for short jobs under 10 minutes.
- B) The login node is shared; computation there degrades performance for all users and violates HPC policy.
- C) The login node has more RAM so it is actually better for memory-heavy workloads.
- D) Computation on the login node is automatically migrated to a compute node by LSF.

**Answer: B**

- A) Incorrect — Even "short" computation on the login node is not acceptable; use `linuxsh` for interactive work.
- B) Correct — The login node is a shared gateway; running computation there slows login/file operations for every user and violates cluster usage policy.
- C) Incorrect — Login nodes are not provisioned with more RAM for computation; they are lightweight gateway machines.
- D) Incorrect — LSF does not auto-migrate processes started outside of `bsub`; the process simply runs on the login node.

---
