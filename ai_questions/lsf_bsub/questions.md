# LSF / BSUB Job Scripts — MCQ Practice

> Topics: BSUB flags, resource calculation, GPU configuration, job dependencies, job arrays.
> Exam frequency: **Every exam** — second highest priority topic.

---

## Q1 — Memory Per Core Calculation
> **Week reference:** Week 1

A program requires **32 GB** of total RAM and will run on **8 cores**. What is the correct `rusage` specification in the `#BSUB` header?

- A) `#BSUB -R "rusage[mem=32GB]"`
- B) `#BSUB -R "rusage[mem=4GB]"`
- C) `#BSUB -R "rusage[mem=256GB]"`
- D) `#BSUB -R "rusage[mem=8GB]"`

**Answer: B**

- A) Incorrect — 32 GB is the total; rusage[mem=] specifies per-core memory, so this would over-request by 8×.
- B) Correct — rusage[mem=XGB] is per core; 32 GB ÷ 8 cores = 4 GB per core.
- C) Incorrect — 256 GB is 32 × 8, which would request 256× too much memory.
- D) Incorrect — 8 GB per core × 8 cores = 64 GB total, which is double what is needed.

---

## Q2 — Identifying Wrong Memory Value
> **Week reference:** Week 1

A colleague's job script contains `#BSUB -n 8` and `#BSUB -R "rusage[mem=16GB]"`. The program actually needs **16 GB total**. What is the problem, and what is the corrected `rusage` value?

- A) No problem — 16 GB per core on 8 cores is correct for a 16 GB program.
- B) The script requests too little memory; it should be `rusage[mem=128GB]`.
- C) The script over-requests memory; the correct value is `rusage[mem=2GB]`.
- D) The `-n 8` flag must be removed so the rusage value is interpreted as total.

**Answer: C**

- A) Incorrect — rusage[mem=16GB] with 8 cores requests 128 GB total, far exceeding the 16 GB needed.
- B) Incorrect — 128 GB would be even more excessive; the program only needs 16 GB total.
- C) Correct — 16 GB total ÷ 8 cores = 2 GB per core; the script over-requests by 8×.
- D) Incorrect — removing -n 8 does not change how LSF interprets rusage; it is always per core.

---

## Q3 — Shared Memory Parallelism Flag
> **Week reference:** Week 1

You are running a Python `multiprocessing` program that uses a shared memory array across 16 worker processes. All processes must be on the same node to access shared memory. Which `#BSUB` directive is required?

- A) `#BSUB -R "select[ncpus=16]"`
- B) `#BSUB -R "span[ptile=16]"`
- C) `#BSUB -R "span[hosts=1]"`
- D) `#BSUB -allhosts`

**Answer: C**

- A) Incorrect — select[ncpus=16] filters nodes by CPU count but does not force all cores onto one node.
- B) Incorrect — span[ptile=N] sets cores per host but does not restrict to exactly one host.
- C) Correct — span[hosts=1] forces all requested cores to be allocated on a single node, required for shared-memory parallelism.
- D) Incorrect — -allhosts is not a valid LSF BSUB directive.

---

## Q4 — GPU Queue Configuration
> **Week reference:** Week 1

You need to convert a CPU job script to run on a GPU. The original script has `#BSUB -q hpc`. Which pair of changes correctly configures the script for a single V100 GPU in exclusive process mode?

- A) Change to `#BSUB -q gpu` and add `#BSUB -R "select[gpu=1]"`
- B) Change to `#BSUB -q gpuv100` and add `#BSUB -gpu "num=1:mode=exclusive_process"`
- C) Change to `#BSUB -q gpuv100` and add `#BSUB -R "rusage[gpu=1]"`
- D) Change to `#BSUB -q gpuv100`; no other changes are needed.

**Answer: B**

- A) Incorrect — -q gpu is not the correct queue name and select[gpu=1] is not how GPUs are requested in LSF.
- B) Correct — -q gpuv100 selects the V100 queue and -gpu "num=1:mode=exclusive_process" reserves one GPU exclusively.
- C) Incorrect — rusage[gpu=1] is not valid LSF syntax for GPU reservation.
- D) Incorrect — switching the queue alone does not allocate a GPU; the -gpu flag is also required.

---

## Q5 — Job Dependency done() Semantics
> **Week reference:** Week 11

Job `analysis` has `#BSUB -w "done(prepare)"`. Five jobs named `prepare` are submitted. Under what condition does `analysis` start running?

- A) As soon as any single job named `prepare` reaches the DONE state.
- B) Only when all jobs named `prepare` have reached the DONE state successfully.
- C) When all jobs named `prepare` have terminated, regardless of exit status.
- D) Immediately, since `done()` is only a soft dependency.

**Answer: B**

- A) Incorrect — done() requires ALL matching jobs to complete successfully, not just one.
- B) Correct — done(name) waits until every job with that name finishes with DONE (exit code 0).
- C) Incorrect — that describes ended(), not done(); done() ignores jobs that exit with failure.
- D) Incorrect — done() is a hard dependency; the job will not start until the condition is satisfied.

---

## Q6 — Job Dependency ended() vs done()
> **Week reference:** Week 11

A cleanup job must remove temporary files regardless of whether the preceding compute jobs succeeded or failed. Which dependency expression should be used?

- A) `#BSUB -w "done(compute)"`
- B) `#BSUB -w "exit(compute)"`
- C) `#BSUB -w "ended(compute)"`
- D) `#BSUB -w "started(compute)"`

**Answer: C**

- A) Incorrect — done() only triggers if all compute jobs succeed; cleanup would never run after a failure.
- B) Incorrect — exit() only triggers if jobs fail, which is the opposite of what is needed.
- C) Correct — ended() triggers when jobs reach any terminal state (DONE or EXIT), ensuring cleanup always runs.
- D) Incorrect — started() is not a standard LSF dependency condition.

---

## Q7 — EXIT State Blocks done() Dependency
> **Week reference:** Week 11

You submit a job `postprocess` with `#BSUB -w "done(prepare)"`. After a while, `bjobs` shows one `prepare` job in EXIT state and the rest in DONE. What happens to `postprocess`?

- A) It starts immediately because the majority of `prepare` jobs succeeded.
- B) It starts once the failed job is resubmitted and completes successfully.
- C) It will never start because one `prepare` job exited with failure.
- D) It starts after a 10-minute retry timeout configured by LSF.

**Answer: C**

- A) Incorrect — done() has no majority rule; all jobs must reach DONE for the dependency to clear.
- B) Incorrect — LSF does not automatically resubmit failed jobs; postprocess remains pending indefinitely.
- C) Correct — because done() requires every matching job to reach DONE, a single EXIT causes the dependent job to stay pending forever.
- D) Incorrect — LSF has no automatic retry timeout for dependency conditions; the job simply waits.

---

## Q8 — Job Array Syntax
> **Week reference:** Week 11

Which `#BSUB` directive creates a job array named `simulation` with 10 elements indexed 1 through 10?

- A) `#BSUB -J simulation -n 10`
- B) `#BSUB -J "simulation[0-9]"`
- C) `#BSUB -J "simulation[1-10]"`
- D) `#BSUB -J "simulation" -array 10`

**Answer: C**

- A) Incorrect — -n sets the number of CPU cores, not the number of array jobs.
- B) Incorrect — [0-9] produces indices 0 through 9, not 1 through 10; the question asks for 10 jobs indexed 1–10.
- C) Correct — -J "name[1-10]" is the standard LSF syntax for a job array with indices 1 to 10.
- D) Incorrect — -array is not a valid BSUB flag; the array specification belongs inside -J.

---

## Q9 — $LSB_JOBINDEX Is 1-Based
> **Week reference:** Week 11

A job array is submitted with `#BSUB -J "run[1-10]"`. The script reads a list of parameter files stored in a Python list `params`. Which line correctly reads the file for the current array element?

- A) `param = params[$LSB_JOBINDEX]`
- B) `idx = int(os.environ['LSB_JOBINDEX']); param = params[idx]`
- C) `idx = int(os.environ['LSB_JOBINDEX']) - 1; param = params[idx]`
- D) `idx = int(os.environ['LSB_JOBINDEX']) + 1; param = params[idx]`

**Answer: C**

- A) Incorrect — shell variable syntax in Python; this would raise a NameError and also has an off-by-one error.
- B) Incorrect — $LSB_JOBINDEX starts at 1, so index 1 accesses the second element of a 0-based Python list.
- C) Correct — subtracting 1 converts from LSF's 1-based index to Python's 0-based list index.
- D) Incorrect — adding 1 would skip an additional element and cause an IndexError for the last job.

---

## Q10 — Per-Element Log Output
> **Week reference:** Week 11

A job array `run[1-20]` is submitted. You want each array element to write its output to a separate file named `run_<jobid>_<index>.out`. Which output directive is correct?

- A) `#BSUB -o run_%J.out`
- B) `#BSUB -o run_%J_%I.out`
- C) `#BSUB -o run_%I.out`
- D) `#BSUB -o run_%J_%A.out`

**Answer: B**

- A) Incorrect — %J expands to the job ID only; all 20 array elements would append to the same file.
- B) Correct — %J is the job ID and %I is the array index; together they produce one unique file per element.
- C) Incorrect — %I alone omits the job ID, so re-runs would overwrite previous output files.
- D) Incorrect — %A is not a standard LSF output filename substitution token.

---

## Q11 — Wall Time Format
> **Week reference:** Week 1

A job needs to run for 1 hour and 30 minutes. Which pair of `#BSUB -W` values are both valid and equivalent ways to specify this wall time?

- A) `#BSUB -W 1:30` and `#BSUB -W 90`
- B) `#BSUB -W 1.5` and `#BSUB -W 90m`
- C) `#BSUB -W 130` and `#BSUB -W 1:30`
- D) `#BSUB -W 5400` and `#BSUB -W 1:30`

**Answer: A**

- A) Correct — -W 1:30 means 1 hour 30 minutes in HH:MM format; -W 90 means 90 minutes; both are 90 minutes.
- B) Incorrect — LSF does not accept decimal hours or the 'm' suffix for wall time.
- C) Incorrect — -W 130 means 130 minutes (2 h 10 min), not 90 minutes.
- D) Incorrect — -W 5400 would be interpreted as 5400 minutes (90 hours), not 5400 seconds.

---

## Q12 — Pinning to a CPU Model
> **Week reference:** Week 1

You are benchmarking a NumPy routine and need reproducible timings. Different node generations in the cluster give different results. Which `#BSUB` directive ensures the job always runs on nodes with an Intel Xeon Gold 6226R processor?

- A) `#BSUB -R "select[cpu=XeonGold6226R]"`
- B) `#BSUB -R "rusage[cpu=XeonGold6226R]"`
- C) `#BSUB -R "select[model==XeonGold6226R]"`
- D) `#BSUB -processor XeonGold6226R`

**Answer: C**

- A) Incorrect — the key for CPU model selection is `model`, not `cpu`, and the operator should be `==`.
- B) Incorrect — rusage is for resource consumption (memory, GPU slots), not hardware selection.
- C) Correct — select[model==XeonGold6226R] filters the scheduler to only allocate nodes matching that CPU model string.
- D) Incorrect — -processor is not a valid BSUB directive.

---
