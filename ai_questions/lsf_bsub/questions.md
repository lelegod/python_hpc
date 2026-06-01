# LSF / BSUB Job Scripts — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Memory Per Core Calculation](#q1-memory-per-core-calculation)
- [Q2 — Identifying Wrong Memory Value](#q2-identifying-wrong-memory-value)
- [Q3 — Shared Memory Parallelism Flag](#q3-shared-memory-parallelism-flag)
- [Q4 — GPU Queue Configuration](#q4-gpu-queue-configuration)
- [Q5 — Job Dependency done() Semantics](#q5-job-dependency-done-semantics)
- [Q6 — Job Dependency ended() vs done()](#q6-job-dependency-ended-vs-done)
- [Q7 — EXIT State Blocks done() Dependency](#q7-exit-state-blocks-done-dependency)
- [Q8 — Job Array Syntax](#q8-job-array-syntax)
- [Q9 — $LSB_JOBINDEX Is 1-Based](#q9-lsb_jobindex-is-1-based)
- [Q10 — Per-Element Log Output](#q10-per-element-log-output)
- [Q11 — Wall Time Format](#q11-wall-time-format)
- [Q12 — Pinning to a CPU Model](#q12-pinning-to-a-cpu-model)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q13 — Total Memory for a 4-Core Job](#q13-total-memory-for-a-4-core-job)
- [Q14 — Per-Core Value for a Target Total](#q14-per-core-value-for-a-target-total)
- [Q15 — GPU Job: Which Flag Is Mandatory?](#q15-gpu-job-which-flag-is-mandatory)
- [Q16 — GPU Mode: exclusive_process vs shared](#q16-gpu-mode-exclusive_process-vs-shared)
- [Q17 — span[hosts=1] vs span[ptile=N]](#q17-spanhosts1-vs-spanptilen)
- [Q18 — Wall Time: Minutes-Only Format](#q18-wall-time-minutes-only-format)
- [Q19 — bjobs Status Column Meanings](#q19-bjobs-status-column-meanings)
- [Q20 — -n Flag vs -R Flag: What Each Controls](#q20-n-flag-vs-r-flag-what-each-controls)
- [Q21 — bkill and Stuck PEND Jobs](#q21-bkill-and-stuck-pend-jobs)
- [Q22 — Email Notification Flags](#q22-email-notification-flags)
- [Set 3 — Extended Practice](#set-3-extended-practice)
- [Q23 — bhist: Inspecting Completed Jobs](#q23--bhist-inspecting-completed-jobs)
- [Q24 — OMP_NUM_THREADS vs -n Interaction](#q24--omp_num_threads-vs--n-interaction)
- [Q25 — Dependency by Job ID vs Job Name](#q25--dependency-by-job-id-vs-job-name)
- [Q26 — Non-Contiguous Job Array Indices](#q26--non-contiguous-job-array-indices)
- [Q27 — bsub Submission Syntax: < Is Required](#q27--bsub-submission-syntax--is-required)
- [Q28 — Memory Charged for Unused Cores](#q28--memory-charged-for-unused-cores)
- [Q29 — USUSP and SSUSP States](#q29--ususp-and-ssusp-states)
- [Q30 — Multiple -R Strings: AND Semantics](#q30--multiple--r-strings-and-semantics)
- [Q31 — gpu32 Queue vs gpuv100 Queue](#q31--gpu32-queue-vs-gpuv100-queue)
- [Q32 — Separating stdout and stderr with -e](#q32--separating-stdout-and-stderr-with--e)

---

> Topics: BSUB flags, resource calculation, GPU configuration, job dependencies, job arrays.
> Exam frequency: **Every exam** — second highest priority topic.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--memory-per-core-calculation)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — Memory Per Core Calculation
> **Week reference:** Week 1

**Mental Model:** rusage[mem=] is always per-core — total_needed ÷ n_cores = per-core value — the trap is requesting the total directly, which over-allocates by a factor of n_cores.

A program requires **32 GB** of total RAM and will run on **8 cores**. What is the correct `rusage` specification in the `#BSUB` header?

- A) `#BSUB -R "rusage[mem=32GB]"`
- B) `#BSUB -R "rusage[mem=4GB]"`
- C) `#BSUB -R "rusage[mem=256GB]"`
- D) `#BSUB -R "rusage[mem=8GB]"`

**Answer: B**

- A) Incorrect — rusage[mem=32GB] reserves 32 GB per core, so LSF allocates 32 × 8 = 256 GB total. The job needs only 32 GB total, making this an 8× over-request that wastes cluster resources and may prevent scheduling.
- B) Correct — rusage[mem=X] is per core; 32 GB total ÷ 8 cores = 4 GB per core. LSF then reserves 4 × 8 = 32 GB total, exactly matching the requirement.
- C) Incorrect — 256 GB = 32 GB × 8, so someone mistakenly multiplied total by cores instead of dividing. That would request 256 × 8 = 2048 GB total — wildly excessive.
- D) Incorrect — 8 GB per core × 8 cores = 64 GB total, which is double what the program needs. Halving the total-to-per-core conversion was not applied correctly.

---

## Q2 — Identifying Wrong Memory Value
> **Week reference:** Week 1

**Mental Model:** Spot the direction of the rusage error — always ask "is this the per-core value or the total?" and verify total = per-core × n_cores — the trap is treating the program's total need as the per-core value.

A colleague's job script contains `#BSUB -n 8` and `#BSUB -R "rusage[mem=16GB]"`. The program actually needs **16 GB total**. What is the problem, and what is the corrected `rusage` value?

- A) No problem — 16 GB per core on 8 cores is correct for a 16 GB program.
- B) The script requests too little memory; it should be `rusage[mem=128GB]`.
- C) The script over-requests memory; the correct value is `rusage[mem=2GB]`.
- D) The `-n 8` flag must be removed so the rusage value is interpreted as total.

**Answer: C**

- A) Incorrect — rusage[mem=16GB] with -n 8 reserves 16 × 8 = 128 GB total. The program only needs 16 GB, so the script over-requests by exactly 8×. This wastes scheduler allocations on every run.
- B) Incorrect — 128 GB per core would allocate 128 × 8 = 1024 GB total, which is 64× more than needed. The script already over-requests; increasing the value makes it far worse.
- C) Correct — 16 GB total ÷ 8 cores = 2 GB per core. With rusage[mem=2GB] and -n 8, LSF reserves 2 × 8 = 16 GB total, matching the program's actual need.
- D) Incorrect — removing -n 8 allocates only 1 core, causing the program to fail for lack of workers. LSF always interprets rusage[mem=] as per-core regardless of whether -n is present; it cannot be switched to a total interpretation.

---

## Q3 — Shared Memory Parallelism Flag
> **Week reference:** Week 1

**Mental Model:** Shared-memory requires all processes on the same physical node — span[hosts=1] is the only LSF directive that enforces single-node allocation — the trap is confusing CPU count selection with node co-location.

You are running a Python `multiprocessing` program that uses a shared memory array across 16 worker processes. All processes must be on the same node to access shared memory. Which `#BSUB` directive is required?

- A) `#BSUB -R "select[ncpus=16]"`
- B) `#BSUB -R "span[ptile=16]"`
- C) `#BSUB -R "span[hosts=1]"`
- D) `#BSUB -allhosts`

**Answer: C**

- A) Incorrect — select[ncpus=16] is a filter that restricts jobs to nodes that have at least 16 CPUs available, but it does not prevent LSF from spreading the 16 cores across multiple such nodes. Shared memory across nodes is impossible.
- B) Incorrect — span[ptile=N] sets the maximum cores per host (tile size), which caps how many cores per node are used. With ptile=16 and -n 16, all cores could land on one node, but this is not guaranteed if the scheduler sees two 8-core nodes. span[hosts=1] is the explicit guarantee.
- C) Correct — span[hosts=1] forces all -n cores to be allocated on exactly one node. This is the required directive whenever shared-memory (multiprocessing, shared arrays, mmap) is used and processes must physically co-reside.
- D) Incorrect — -allhosts is not a valid LSF BSUB directive. It does not exist and would cause a script parse error.

---

## Q4 — GPU Queue Configuration
> **Week reference:** Week 1

**Mental Model:** GPU jobs need both the right queue AND an explicit -gpu resource request — the queue selects the hardware class, the -gpu flag allocates the device — the trap is thinking queue alone is sufficient.

You need to convert a CPU job script to run on a GPU. The original script has `#BSUB -q hpc`. Which pair of changes correctly configures the script for a single V100 GPU in exclusive process mode?

- A) Change to `#BSUB -q gpu` and add `#BSUB -R "select[gpu=1]"`
- B) Change to `#BSUB -q gpuv100` and add `#BSUB -gpu "num=1:mode=exclusive_process"`
- C) Change to `#BSUB -q gpuv100` and add `#BSUB -R "rusage[gpu=1]"`
- D) Change to `#BSUB -q gpuv100`; no other changes are needed.

**Answer: B**

- A) Incorrect — -q gpu is not the correct DTU HPC queue name for V100s (it's gpuv100), and select[gpu=1] is not valid LSF GPU syntax. GPU allocation requires the -gpu flag, not a select expression.
- B) Correct — -q gpuv100 routes the job to nodes with V100 GPUs, and -gpu "num=1:mode=exclusive_process" allocates exactly one GPU in exclusive process mode (no GPU sharing with other jobs).
- C) Incorrect — rusage[gpu=1] is not valid LSF syntax for GPU reservation. rusage is for consumable resources like memory; GPUs are allocated via the dedicated -gpu flag.
- D) Incorrect — switching the queue selects the right class of nodes but does not actually reserve a GPU. Without the -gpu flag, the job lands on a GPU node but no GPU is allocated, so CUDA calls will fail.

---

## Q5 — Job Dependency done() Semantics
> **Week reference:** Week 11

**Mental Model:** done() = ALL matching jobs must reach DONE successfully — it is an AND across every job with that name — the trap is confusing it with ended() (any terminal state) or thinking majority-done is sufficient.

Job `analysis` has `#BSUB -w "done(prepare)"`. Five jobs named `prepare` are submitted. Under what condition does `analysis` start running?

- A) As soon as any single job named `prepare` reaches the DONE state.
- B) Only when all jobs named `prepare` have reached the DONE state successfully.
- C) When all jobs named `prepare` have terminated, regardless of exit status.
- D) Immediately, since `done()` is only a soft dependency.

**Answer: B**

- A) Incorrect — done() is an AND condition, not an OR. If any single completion were sufficient, one quick job could trigger downstream analysis before the others finish, corrupting results. LSF requires every matching job to complete.
- B) Correct — done(name) resolves to true only when every job with that name has exited with DONE status (exit code 0). All five prepare jobs must complete successfully before analysis is released.
- C) Incorrect — "any terminal state" describes ended(), not done(). If a prepare job exits with a non-zero code (EXIT state), done() remains false and analysis stays pending indefinitely.
- D) Incorrect — done() is a hard dependency enforced by the LSF scheduler. The dependent job is held in PEND state until the condition is fully satisfied; it cannot start early.

---

## Q6 — Job Dependency ended() vs done()
> **Week reference:** Week 11

**Mental Model:** Choose the dependency based on whether you care about success — done() = success only, ended() = any exit (success or failure) — the trap is using done() for cleanup tasks that must run even after failures.

A cleanup job must remove temporary files regardless of whether the preceding compute jobs succeeded or failed. Which dependency expression should be used?

- A) `#BSUB -w "done(compute)"`
- B) `#BSUB -w "exit(compute)"`
- C) `#BSUB -w "ended(compute)"`
- D) `#BSUB -w "started(compute)"`

**Answer: C**

- A) Incorrect — done() only triggers when compute jobs exit successfully (DONE state). If any compute job fails (EXIT state), done() stays false and the cleanup job never runs, leaving temporary files on disk.
- B) Incorrect — exit() only triggers when jobs fail (EXIT state). This is the opposite extreme: cleanup would only run after failures, not after successful runs that also leave temp files.
- C) Correct — ended() triggers when jobs reach any terminal state (DONE or EXIT). This ensures cleanup runs after compute finishes, regardless of success or failure — exactly the semantics needed for unconditional teardown.
- D) Incorrect — started() triggers when the named jobs begin running, not when they finish. A cleanup job triggered at start would run while compute is still in progress, which is the opposite of what is needed for post-processing teardown.

---

## Q7 — EXIT State Blocks done() Dependency
> **Week reference:** Week 11

**Mental Model:** A single EXIT in a done() dependency permanently blocks the downstream job — there is no timeout or partial-completion logic — the trap is thinking LSF auto-retries or uses majority rules.

You submit a job `postprocess` with `#BSUB -w "done(prepare)"`. After a while, `bjobs` shows one `prepare` job in EXIT state and the rest in DONE. What happens to `postprocess`?

- A) It starts immediately because the majority of `prepare` jobs succeeded.
- B) It starts once the failed job is resubmitted and completes successfully.
- C) It will never start because one `prepare` job exited with failure.
- D) It starts after a 10-minute retry timeout configured by LSF.

**Answer: C**

- A) Incorrect — done() has no majority-rule or quorum logic. Even if 99 out of 100 prepare jobs are DONE, a single EXIT keeps the condition false. LSF evaluates done() as a strict AND across all matching jobs.
- B) Incorrect — LSF does not automatically resubmit failed jobs; resubmission is entirely the user's responsibility. Until the user manually resubmits the failed prepare job and it succeeds, postprocess remains stuck in PEND.
- C) Correct — because done() requires every matching job to be in DONE state, the single EXIT job permanently prevents the condition from resolving. postprocess stays in PEND indefinitely unless the user intervenes.
- D) Incorrect — LSF has no built-in retry timeout for dependency conditions. Dependency evaluation is event-driven (job state changes), not time-driven. The job simply waits until the condition becomes true or the user kills it.

---

## Q8 — Job Array Syntax
> **Week reference:** Week 11

**Mental Model:** Job array indices go inside the -J name in square brackets — [start-end] syntax, 1-based by convention — the trap is using -n for array count (that sets cores) or using 0-based indexing.

Which `#BSUB` directive creates a job array named `simulation` with 10 elements indexed 1 through 10?

- A) `#BSUB -J simulation -n 10`
- B) `#BSUB -J "simulation[0-9]"`
- C) `#BSUB -J "simulation[1-10]"`
- D) `#BSUB -J "simulation" -array 10`

**Answer: C**

- A) Incorrect — -n 10 requests 10 CPU cores for a single job, not 10 separate array elements. This would submit one job that uses 10 cores, not an array of 10 jobs each using 1 core.
- B) Incorrect — [0-9] creates indices 0 through 9 (10 elements), but the question specifically asks for indices 1 through 10. Using [0-9] means $LSB_JOBINDEX starts at 0, causing an off-by-one when accessing 1-based parameter lists.
- C) Correct — -J "simulation[1-10]" is standard LSF job array syntax. It creates 10 array elements with $LSB_JOBINDEX ranging from 1 to 10 inclusive.
- D) Incorrect — -array is not a valid BSUB directive. The array specification must be embedded within the -J job name string; there is no separate flag for it.

---

## Q9 — $LSB_JOBINDEX Is 1-Based
> **Week reference:** Week 11

**Mental Model:** $LSB_JOBINDEX is 1-based, Python lists are 0-based — always subtract 1 when indexing into a Python sequence — the trap is using the index directly and silently reading the wrong element.

A job array is submitted with `#BSUB -J "run[1-10]"`. The script reads a list of parameter files stored in a Python list `params`. Which line correctly reads the file for the current array element?

- A) `param = params[$LSB_JOBINDEX]`
- B) `idx = int(os.environ['LSB_JOBINDEX']); param = params[idx]`
- C) `idx = int(os.environ['LSB_JOBINDEX']) - 1; param = params[idx]`
- D) `idx = int(os.environ['LSB_JOBINDEX']) + 1; param = params[idx]`

**Answer: C**

- A) Incorrect — $LSB_JOBINDEX is a shell variable, not a Python variable. Using it directly in Python raises a NameError. Even if it were valid syntax, the missing -1 would read params[1] for the first array element instead of params[0].
- B) Incorrect — os.environ['LSB_JOBINDEX'] is correctly read as a string and converted to int, but no offset is applied. For array element 1, idx=1 accesses params[1] (the second item), skipping params[0]. For element 10, idx=10 raises IndexError on a 10-element list.
- C) Correct — subtracting 1 maps $LSB_JOBINDEX (1-based: 1…10) to Python list indices (0-based: 0…9). Element 1 → params[0], element 10 → params[9].
- D) Incorrect — adding 1 shifts the index up by 2 positions from where it should be (idx = LSB_JOBINDEX + 1 = 2 for element 1, accessing params[2]). For element 10, idx=11 causes IndexError on a 10-element list.

---

## Q10 — Per-Element Log Output
> **Week reference:** Week 11

**Mental Model:** %J = job ID, %I = array index — you need both to create unique filenames per element — the trap is using only %J (all elements write to the same file) or using an invalid token like %A.

A job array `run[1-20]` is submitted. You want each array element to write its output to a separate file named `run_<jobid>_<index>.out`. Which output directive is correct?

- A) `#BSUB -o run_%J.out`
- B) `#BSUB -o run_%J_%I.out`
- C) `#BSUB -o run_%I.out`
- D) `#BSUB -o run_%J_%A.out`

**Answer: B**

- A) Incorrect — %J expands to the job ID (same for all array elements), so all 20 elements write to the same file run_12345.out. Their output lines interleave and become impossible to separate.
- B) Correct — %J is the job ID (unique per array submission) and %I is the array index (1–20). Together they produce run_12345_1.out, run_12345_2.out, etc. — one unique file per element.
- C) Incorrect — %I alone gives run_1.out through run_20.out, which is unique within one submission. However, if you resubmit the array, the new run gets a different job ID but the same index numbers, silently overwriting the previous output files.
- D) Incorrect — %A is not a recognized LSF output filename token. Using an invalid token causes LSF to write a literal "%A" in the filename or fall back to a default, not the array index.

---

## Q11 — Wall Time Format
> **Week reference:** Week 1

**Mental Model:** LSF -W accepts HH:MM or plain minutes — 1:30 = 90 minutes = -W 90 — the trap is using seconds, decimal hours, or unit suffixes that LSF does not recognize.

A job needs to run for 1 hour and 30 minutes. Which pair of `#BSUB -W` values are both valid and equivalent ways to specify this wall time?

- A) `#BSUB -W 1:30` and `#BSUB -W 90`
- B) `#BSUB -W 1.5` and `#BSUB -W 90m`
- C) `#BSUB -W 130` and `#BSUB -W 1:30`
- D) `#BSUB -W 5400` and `#BSUB -W 1:30`

**Answer: A**

- A) Correct — -W 1:30 uses HH:MM format: 1 hour + 30 minutes = 90 minutes. -W 90 uses plain integer format interpreted as minutes: 90 minutes. Both specify exactly 90 minutes of wall time.
- B) Incorrect — LSF does not parse decimal hours (1.5) or minute suffixes (90m). These values would either be rejected or misinterpreted as 1 minute and 90 minutes respectively, depending on LSF version.
- C) Incorrect — -W 130 is 130 minutes (2 hours 10 minutes), not 90 minutes. The plain integer format does not do HH:MM parsing; 130 means 130 minutes flat.
- D) Incorrect — -W 5400 is interpreted as 5400 minutes (90 hours), not 5400 seconds. LSF wall-time integers are always in minutes, not seconds. This would over-reserve the job by a factor of 60.

---

## Q12 — Pinning to a CPU Model
> **Week reference:** Week 1

**Mental Model:** Hardware selection uses select[model==X] with double-equals — rusage is for consumable resources, select is for hardware properties — the trap is using = instead of == or using rusage for non-consumable hardware attributes.

You are benchmarking a NumPy routine and need reproducible timings. Different node generations in the cluster give different results. Which `#BSUB` directive ensures the job always runs on nodes with an Intel Xeon Gold 6226R processor?

- A) `#BSUB -R "select[cpu=XeonGold6226R]"`
- B) `#BSUB -R "rusage[cpu=XeonGold6226R]"`
- C) `#BSUB -R "select[model==XeonGold6226R]"`
- D) `#BSUB -processor XeonGold6226R`

**Answer: C**

- A) Incorrect — the LSF resource attribute for CPU model is `model`, not `cpu`. Using `cpu` would either be ignored or match a different attribute. Additionally, selection predicates require `==` (double-equals) for string comparison, not `=`.
- B) Incorrect — rusage is for consumable resources that are decremented when allocated (like memory slots or GPU slots). CPU model is a static hardware property, not a consumable; it belongs in a select expression, not rusage.
- C) Correct — select[model==XeonGold6226R] filters node selection to only allocate nodes whose reported CPU model string matches exactly. The double-equals is required syntax for string equality in LSF resource expressions.
- D) Incorrect — -processor is not a valid BSUB directive. There is no such flag in LSF; using it would cause a script parsing error and the job would not be submitted.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets rusage[mem=X] per-core semantics, GPU queue flags, -n vs span[hosts=1], wall time formats, and bjobs output interpretation

---

## Q13 — Total Memory for a 4-Core Job

> **Week reference:** Week 1

A job script contains `#BSUB -n 4` and `#BSUB -R "rusage[mem=8192]"`. Assuming the memory unit is MB, what is the **total** memory reserved by LSF for this job?

- A) 8192 MB (8 GB)
- B) 16384 MB (16 GB)
- C) 32768 MB (32 GB)
- D) 2048 MB (2 GB)

**Answer: C**

rusage[mem=X] is per-core. With 4 cores: 8192 MB × 4 = 32768 MB = 32 GB total. A is the most common trap — treating the value as the total rather than per-core. B would be 8192 × 2 cores. D would be 8192 / 4, which inverts the direction of the error.

---

## Q14 — Per-Core Value for a Target Total

> **Week reference:** Week 1

You need exactly **24 GB** of total memory for a job running on **6 cores**. What value should you specify for `rusage[mem=X]` (in MB)?

- A) 24576
- B) 4096
- C) 147456
- D) 12288

**Answer: B**

Per-core = total ÷ n_cores = 24 GB ÷ 6 = 4 GB = 4096 MB. A (24576 MB = 24 GB) treats the total as the per-core value, which would actually allocate 24 × 6 = 144 GB. C (147456 MB) is the result of that over-allocation. D (12288 MB = 12 GB) halves correctly once but that would give 12 × 6 = 72 GB total, not 24 GB.

---

## Q15 — GPU Job: Which Flag Is Mandatory?

> **Week reference:** Week 1

A student switches their queue to `#BSUB -q gpuv100` but does not add any other GPU-related line to the script. The job runs. What actually happens?

- A) The job gets a GPU automatically because it is in the GPU queue
- B) The job lands on a GPU node but no GPU is allocated; CUDA calls will fail
- C) LSF rejects the script with a submission error because -gpu is required
- D) The job is automatically given one GPU in shared mode

**Answer: B**

The GPU queue routes the job to nodes that have GPUs installed, but it does not allocate a GPU device to the process. Only `#BSUB -gpu "num=1:mode=exclusive_process"` (or similar) actually reserves a GPU. Without it, CUDA initialization finds no device and the program fails at runtime. LSF does not reject the script — the directive is optional, just useless here.

---

## Q16 — GPU Mode: exclusive_process vs shared

> **Week reference:** Week 1

Two students both request `#BSUB -gpu "num=1"` without specifying a mode. A third student uses `#BSUB -gpu "num=1:mode=exclusive_process"`. What is the practical difference?

- A) The third student's job will always be scheduled faster because exclusive mode has higher priority
- B) Without specifying mode, the GPU may be shared with other jobs; exclusive_process guarantees no other process uses the same GPU concurrently
- C) exclusive_process reserves the entire node, not just the GPU
- D) There is no difference — all GPU jobs are run in exclusive mode by default on DTU HPC

**Answer: B**

`mode=exclusive_process` means the GPU is not shared with any other process for the duration of the job. Without it, the GPU may be in a shared or default mode where multiple jobs can time-share the device, which causes non-deterministic performance and can corrupt memory-heavy workloads. It does not affect node-level exclusivity or scheduling priority.

---

## Q17 — span[hosts=1] vs span[ptile=N]

> **Week reference:** Week 1

A job uses `#BSUB -n 8` and `#BSUB -R "span[ptile=4]"`. How many nodes does LSF allocate, and is this safe for `multiprocessing.shared_memory`?

- A) 1 node with 8 cores — safe for shared memory
- B) 2 nodes with 4 cores each — NOT safe for shared memory across nodes
- C) 8 nodes with 1 core each — safe because all cores are pinned
- D) 4 nodes with 2 cores each — safe because ptile balances the load

**Answer: B**

`span[ptile=4]` sets the maximum cores per host to 4. With `-n 8` total cores, LSF distributes across 2 nodes (4 cores each). Shared memory (e.g., `multiprocessing.shared_memory.SharedMemory`) exists only within a single node's physical RAM. Processes on different nodes cannot access it. Only `span[hosts=1]` forces all 8 cores onto one node, which is the requirement for shared-memory parallelism.

---

## Q18 — Wall Time: Minutes-Only Format

> **Week reference:** Week 1

A researcher wants to set a wall time limit of exactly **3 hours and 15 minutes**. Which `#BSUB -W` values are both valid and equivalent?

- A) `-W 3:15` and `-W 195`
- B) `-W 3.25` and `-W 195`
- C) `-W 315` and `-W 3:15`
- D) `-W 3h15m` and `-W 195`

**Answer: A**

`-W 3:15` uses HH:MM format (3 hours, 15 minutes = 195 minutes). `-W 195` uses the plain-integer minutes format. Both specify identical wall time. B uses decimal hours (3.25), which LSF does not parse. C uses 315 which means 315 minutes (5 hours 15 minutes), not 3:15 — a classic trap where people read the digits literally. D uses a unit-suffix format that LSF does not recognize.

---

## Q19 — bjobs Status Column Meanings

> **Week reference:** Week 1

You run `bjobs` and see jobs in states PEND, RUN, DONE, and EXIT. A colleague says "my job is PEND — it must be paused mid-run." Is this correct, and what does PEND actually mean?

- A) Correct — PEND means the job is paused at a checkpoint
- B) Incorrect — PEND means the job is waiting in the queue for resources or an unsatisfied dependency; it has not started running yet
- C) Incorrect — PEND means the job finished but the exit code is pending evaluation
- D) Correct — PEND means LSF is pending a node health check before resuming the job

**Answer: B**

PEND (pending) is the state a job is in before it has ever started running. It waits in the queue either because no suitable resources are available yet or because a dependency condition (`-w "done(...)"`) is not yet satisfied. It is not a paused-mid-run state — that would be USUSP (user-suspended) or SSUSP (system-suspended). DONE means finished successfully; EXIT means terminated with a non-zero exit code.

---

## Q20 — -n Flag vs -R Flag: What Each Controls

> **Week reference:** Week 1

What is the functional difference between `#BSUB -n 8` and `#BSUB -R "rusage[mem=4GB]"`?

- A) `-n` sets the number of tasks, `-R` sets the number of CPU cores per task
- B) `-n` sets the total number of CPU cores to request; `-R` specifies resource requirements (memory per core, node constraints, hardware selection)
- C) `-n` sets the number of nodes; `-R` sets resources per node
- D) Both flags control CPU count; `-R` overrides `-n` when they conflict

**Answer: B**

`-n` is the core count flag — it tells LSF how many CPU slots to reserve in total. `-R` is the resource requirement string — it can specify memory via `rusage[mem=X]` (per core), node topology via `span[hosts=1]`, hardware filters via `select[model==X]`, and more. They serve completely different purposes. `-n` sets how many; `-R` sets what kind and how much per unit.

---

## Q21 — bkill and Stuck PEND Jobs

> **Week reference:** Week 1

A job is stuck in PEND state because its `done()` dependency can never be satisfied (the prerequisite job exited with an error). What is the correct action to remove the stuck job?

- A) Wait — LSF will automatically detect the unsatisfied dependency and kill the job after 24 hours
- B) `bdel <jobid>` — the dedicated command for deleting pending jobs
- C) `bkill <jobid>` — terminates or removes any job regardless of its current state
- D) `bstop <jobid>` — stops the dependency check and releases the job to run immediately

**Answer: C**

`bkill <jobid>` is the standard command to terminate or remove LSF jobs in any state: PEND, RUN, DONE, or EXIT. LSF does not auto-kill jobs with permanently unsatisfied dependencies — they sit in PEND forever until the user acts. `bdel` is not a standard LSF command. `bstop` suspends a running job (sets it to USUSP); it does not release dependencies or kill the job.

---

## Q22 — Email Notification Flags

> **Week reference:** Week 1

Which pair of `#BSUB` directives together sends an email to `user@dtu.dk` when the job **ends** (not when it starts)?

- A) `#BSUB -B` and `#BSUB -u user@dtu.dk`
- B) `#BSUB -N` and `#BSUB -u user@dtu.dk`
- C) `#BSUB -mail user@dtu.dk` alone
- D) `#BSUB -N` alone (LSF infers the address from the submitting user's login)

**Answer: B**

`-N` sends an email notification when the job **ends** (reaches any terminal state). `-u` specifies the email address. Both are needed together: `-N` enables end-of-job notification and `-u` directs it to the right address. `-B` sends notification at job **start**, not end. `-mail` is not a valid BSUB flag. While `-N` alone might default to the submitter's system account on some clusters, DTU HPC requires `-u` to specify an actual email address.

---

## Set 3 — Extended Practice

- [Q23 — bhist: Inspecting Completed Jobs](#q23--bhist-inspecting-completed-jobs)
- [Q24 — OMP_NUM_THREADS vs -n Interaction](#q24--omp_num_threads-vs--n-interaction)
- [Q25 — Dependency by Job ID vs Job Name](#q25--dependency-by-job-id-vs-job-name)
- [Q26 — Non-Contiguous Job Array Indices](#q26--non-contiguous-job-array-indices)
- [Q27 — bsub Submission Syntax: < Is Required](#q27--bsub-submission-syntax--is-required)
- [Q28 — Memory Charged for Unused Cores](#q28--memory-charged-for-unused-cores)
- [Q29 — USUSP and SSUSP States](#q29--ususp-and-ssusp-states)
- [Q30 — Multiple -R Strings: AND Semantics](#q30--multiple--r-strings-and-semantics)
- [Q31 — gpu32 Queue vs gpuv100 Queue](#q31--gpu32-queue-vs-gpuv100-queue)
- [Q32 — Separating stdout and stderr with -e](#q32--separating-stdout-and-stderr-with--e)

---

## Q23 — bhist: Inspecting Completed Jobs

> **Week reference:** Week 1

**Mental Model:** `bjobs` only shows active jobs (PEND/RUN/USUSP); once a job reaches DONE or EXIT it disappears from `bjobs` output — `bhist` is the dedicated command for querying the historical record of completed jobs.

A job finished an hour ago and is no longer visible in `bjobs`. You need to check whether it completed successfully and how long it ran. Which command retrieves this information?

- A) `bjobs -a`
- B) `bhist <jobid>`
- C) `bpeek <jobid>`
- D) `bjobs -hist <jobid>`

**Answer: B**

- A) Incorrect — `-a` shows all jobs including recently finished ones within the current session window, but once the job has cleared LSF's active record, even `bjobs -a` will not find it. `bhist` queries the persistent accounting log.
- B) Correct — `bhist <jobid>` queries the LSF accounting database and reports the full lifecycle of a completed job: submission time, start time, end time, exit status, and resources consumed. It works for jobs that finished long ago.
- C) Incorrect — `bpeek` is used to preview the live stdout/stderr of a currently running job. It has no meaning for completed jobs and will return an error.
- D) Incorrect — `-hist` is not a valid `bjobs` flag. The correct tool for historical data is the separate `bhist` command.

---

## Q24 — OMP_NUM_THREADS vs -n Interaction

> **Week reference:** Week 1

**Mental Model:** `-n` tells LSF how many CPU slots to reserve; `OMP_NUM_THREADS` tells OpenMP (and similar threading libraries) how many threads to spawn at runtime — they must match or the job either under-uses reserved cores or over-subscribes them.

A job script requests `#BSUB -n 8` but the script body never sets `OMP_NUM_THREADS`. The Python program uses a multi-threaded NumPy routine that respects `OMP_NUM_THREADS`. What is the most likely outcome?

- A) NumPy automatically detects the 8 reserved cores and uses exactly 8 threads
- B) The program uses however many threads the library's default selects, which may be the total physical core count of the node — potentially over-subscribing the allocation
- C) LSF sets `OMP_NUM_THREADS=8` automatically whenever `-n 8` is specified
- D) The program uses 1 thread because LSF does not export any thread-count environment variable

**Answer: B**

- A) Incorrect — LSF does not expose the `-n` value to the running process as a thread-count hint. NumPy/OpenMP have no awareness of LSF's slot reservation.
- B) Correct — OpenMP and MKL (used by NumPy) read `OMP_NUM_THREADS` or fall back to detecting the node's physical core count. On a 40-core node with `-n 8`, the library may spawn 40 threads, causing 5× over-subscription of the 8 reserved slots and poor performance for all jobs on that node.
- C) Incorrect — LSF does not automatically set `OMP_NUM_THREADS`. The user must explicitly add `export OMP_NUM_THREADS=8` (or equivalent) in the job script to match the `-n` value.
- D) Incorrect — With no `OMP_NUM_THREADS` set, the library does not default to 1; it defaults to the hardware thread count detected at runtime.

---

## Q25 — Dependency by Job ID vs Job Name

> **Week reference:** Week 11

**Mental Model:** `done(name)` matches ALL currently submitted jobs with that name — if multiple jobs share a name, the dependency waits for all of them; using a numeric job ID `done(12345)` pins the dependency to exactly one specific job submission.

You submit two separate jobs both named `preprocess`. Later you submit a `report` job with `#BSUB -w "done(preprocess)"`. How many jobs must complete before `report` starts?

- A) Only the first `preprocess` job submitted, because LSF resolves the name at submission time
- B) Both `preprocess` jobs, because `done(name)` matches all live jobs with that name
- C) Only the most recently submitted `preprocess` job
- D) Neither — using a name instead of a job ID causes LSF to ignore the dependency

**Answer: B**

- A) Incorrect — LSF does not freeze name resolution at submission time. The dependency expression is evaluated dynamically against all active jobs matching the name at the time the condition is checked.
- B) Correct — `done(preprocess)` evaluates to true only when every job currently in the system with the name `preprocess` has reached DONE state. Both submissions must complete. This is why using unique job names (or job IDs) is important in complex pipelines to avoid accidental coupling.
- C) Incorrect — LSF has no "most recent" tie-breaking rule for name-based dependencies. All matching jobs must satisfy the condition.
- D) Incorrect — Name-based dependencies are a first-class feature of LSF and work correctly. The `-w` flag accepts both job names and numeric job IDs.

---

## Q26 — Non-Contiguous Job Array Indices

> **Week reference:** Week 11

**Mental Model:** LSF job arrays support non-contiguous index lists using comma-separated values inside the brackets — this allows you to re-run only specific failed elements without resubmitting the whole array.

Which `#BSUB -J` directive submits a job array that runs only elements 2, 29, 71, and 127?

- A) `#BSUB -J "run[2-127:step=29]"`
- B) `#BSUB -J "run[2,29,71,127]"`
- C) `#BSUB -J "run[2+29+71+127]"`
- D) `#BSUB -J "run[2;29;71;127]"`

**Answer: B**

- A) Incorrect — `[2-127:step=29]` uses stride syntax, which generates indices 2, 31, 60, 89, 118 — not the desired values. Stride syntax is `[start-end:stride]`.
- B) Correct — LSF supports comma-separated index lists inside the array brackets. `run[2,29,71,127]` creates exactly four array elements with `$LSB_JOBINDEX` taking those four values. This is the canonical way to re-run specific failed elements.
- C) Incorrect — `+` is not valid syntax inside LSF array brackets. This would cause a submission parse error.
- D) Incorrect — Semicolons are not valid separators in LSF array index syntax. Only commas (for lists) and hyphens with optional colons (for ranges with stride) are supported.

---

## Q27 — bsub Submission Syntax: < Is Required

> **Week reference:** Week 1

**Mental Model:** LSF reads the job script from stdin — the shell redirection `<` feeds the file to `bsub`; omitting it passes the filename as a positional argument, which `bsub` treats differently and may cause unexpected behavior.

A student types `bsub submit.sh` instead of `bsub < submit.sh`. What is the most likely result?

- A) The job is submitted identically — `bsub` accepts both forms
- B) LSF executes `submit.sh` as a shell command immediately on the login node instead of submitting it
- C) `bsub` treats `submit.sh` as the job command, submitting a job that runs the script directly as a process without parsing any `#BSUB` directives from inside it
- D) LSF throws an error: "file argument not supported"

**Answer: C**

- A) Incorrect — The two forms are not equivalent. `bsub < file` reads BSUB directives from inside the script. `bsub file` passes the filename as an argument, and LSF behavior differs by version — typically it runs the file as the command but ignores embedded `#BSUB` lines, using only command-line flags.
- B) Incorrect — `bsub` does not execute programs directly on the login node. The script is still submitted to the scheduler; the issue is that the embedded `#BSUB` resource directives are silently ignored.
- C) Correct — Without `<`, `bsub` may interpret `submit.sh` as the command to run (as if you had typed `bsub submit.sh` like any other program name), submitting a job with default resources and none of the `#BSUB` directives parsed. The job may run with incorrect memory, wall time, and core counts.
- D) Incorrect — LSF does not produce a clean error in this case on most versions; the misbehavior is silent, making it a dangerous mistake.

---

## Q28 — Memory Charged for Unused Cores

> **Week reference:** Week 1

**Mental Model:** LSF charges memory based on the number of cores **requested** via `-n`, not the number actually used — requesting more cores than the program needs wastes both CPU slots and memory allocation on the cluster.

A job script has `#BSUB -n 16` and `#BSUB -R "rusage[mem=2GB]"`, but the Python program is single-threaded and only ever uses 1 core. How much memory does LSF reserve, and how much of that reservation is actually used?

- A) LSF detects the single-threaded usage and reserves only 2 GB
- B) LSF reserves 32 GB (16 × 2 GB), but the program only uses 2 GB worth — 30 GB is wasted and unavailable to other jobs
- C) LSF reserves 2 GB total because single-threaded programs override the -n value
- D) LSF reserves 16 GB because it uses -n as a divisor on the rusage value

**Answer: B**

- A) Incorrect — LSF has no runtime mechanism to detect thread usage and retroactively reduce memory reservations. Resources are committed at scheduling time based on `-n` and `rusage[mem=X]`.
- B) Correct — LSF always charges `rusage[mem=X] × n_cores` regardless of how many cores the program actually uses. With `-n 16` and `rusage[mem=2GB]`, 32 GB is reserved for the duration of the job. The single-threaded program uses ~2 GB, leaving 30 GB allocated but idle, denying it to other jobs.
- C) Incorrect — LSF has no concept of overriding `-n` based on program behavior. The `-n` value is a reservation, not a runtime observation.
- D) Incorrect — LSF does not divide by `-n`. It multiplies `rusage[mem=X]` by the number of cores to get the total reservation. There is no inverse relationship.

---

## Q29 — USUSP and SSUSP States

> **Week reference:** Week 1

**Mental Model:** LSF has two suspended states beyond PEND/RUN — USUSP (user-suspended via `bstop`) and SSUSP (system-suspended by the scheduler) — a suspended job holds its allocated resources but does not execute.

You run `bjobs -a` and see a job in **USUSP** state. What does this mean, and how do you resume it?

- A) USUSP means the job is waiting for user authentication; run `bauth <jobid>` to authenticate
- B) USUSP means the job was suspended by the user (via `bstop`); run `bresume <jobid>` to resume it
- C) USUSP means the job finished with a user-space error; it cannot be resumed
- D) USUSP is identical to PEND — both mean the job is waiting for resources

**Answer: B**

- A) Incorrect — USUSP has nothing to do with authentication. `bauth` is not a standard LSF command.
- B) Correct — USUSP (user-suspended) means a user explicitly called `bstop <jobid>` to pause the job. The job retains its allocated nodes and cores but stops executing. `bresume <jobid>` releases the suspension and the job continues from where it was paused (for checkpoint-capable jobs) or restarts.
- C) Incorrect — USUSP is a live, reversible state, not a terminal error state. EXIT is the terminal failure state.
- D) Incorrect — PEND means the job has not yet been dispatched to a node. USUSP means it was dispatched, started, and then explicitly paused — a fundamentally different lifecycle stage.

---

## Q30 — Multiple -R Strings: AND Semantics

> **Week reference:** Week 1

**Mental Model:** Multiple `#BSUB -R` lines in a script are combined with AND logic — all constraints must be satisfied simultaneously — this is equivalent to putting all clauses in a single `-R` string separated by spaces.

A job script has these two separate lines:

```
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
```

How does LSF interpret these two `-R` directives?

- A) The second `-R` line overrides the first; only `span[hosts=1]` applies
- B) LSF picks whichever constraint is more restrictive and applies only that one
- C) Both constraints apply simultaneously — LSF allocates all cores on a single node AND reserves 4 GB per core
- D) Using two `-R` lines causes a script parse error; they must be combined into one string

**Answer: C**

- A) Incorrect — LSF does not use last-wins semantics for `-R` lines. Each additional `-R` adds another requirement to the existing set; nothing is overridden.
- B) Incorrect — LSF has no concept of "more restrictive wins." All `-R` constraints form a logical AND and must all be satisfied for a node to be eligible.
- C) Correct — Multiple `#BSUB -R` lines accumulate: LSF requires every `-R` clause to be true. The job gets all cores on one node (`span[hosts=1]`) AND reserves 4 GB per core (`rusage[mem=4GB]`). This is fully equivalent to writing `#BSUB -R "rusage[mem=4GB] span[hosts=1]"` on one line.
- D) Incorrect — Multiple `-R` lines are valid and commonly used in real job scripts (as seen in the course's `submit.sh` examples). There is no parse error.

---

## Q31 — gpu32 Queue vs gpuv100 Queue

> **Week reference:** Week 1

**Mental Model:** DTU HPC has multiple GPU queues for different GPU hardware generations — `gpuv100` targets V100 cards, `gpu32` targets A100/newer 32 GB cards — choosing the wrong queue may put the job on hardware with insufficient VRAM or miss available resources.

A deep learning job requires at least 24 GB of GPU VRAM. The cluster has both `gpuv100` (V100, 16 GB VRAM each) and `gpu32` (A100, 32 GB VRAM each) queues. Which queue should be used?

- A) `gpuv100` — it is always the default GPU queue and handles VRAM requests automatically
- B) `gpu32` — it provides GPUs with 32 GB VRAM, sufficient for the 24 GB requirement
- C) Either queue — LSF automatically selects a GPU with enough VRAM regardless of queue
- D) `hpc` — GPU memory is a software allocation that any queue can satisfy

**Answer: B**

- A) Incorrect — `gpuv100` nodes have V100 GPUs with 16 GB VRAM each, which is less than the 24 GB required. The job would submit successfully but fail at runtime when the model exceeds the GPU's memory capacity.
- B) Correct — `gpu32` nodes have GPUs with 32 GB VRAM (e.g., A100-SXM4-32GB), which exceeds the 24 GB requirement. The correct queue must be chosen based on the hardware's actual VRAM capacity.
- C) Incorrect — LSF queue selection is entirely the user's responsibility. LSF does not inspect VRAM requirements and cannot automatically route jobs to the right GPU hardware based on memory needs.
- D) Incorrect — `hpc` is the standard CPU queue and has no GPU nodes. GPU VRAM is a physical hardware property of the node, not a software-allocated resource.

---

## Q32 — Separating stdout and stderr with -e

> **Week reference:** Week 1

**Mental Model:** By default, LSF merges stdout and stderr into the `-o` file — adding `-e` directs stderr to a separate file, making it much easier to distinguish program output from error messages and tracebacks.

A job script has only `#BSUB -o job_%J.out` and no `-e` flag. The Python script prints normal output to stdout and raises an exception whose traceback goes to stderr. Where does the traceback appear?

- A) It is lost — without `-e`, LSF discards stderr output
- B) It appears in `job_<jobid>.out` alongside the stdout output, since LSF merges them by default
- C) It appears in a file named `job_<jobid>.err` that LSF creates automatically even without `-e`
- D) It is sent by email to the submitting user's account

**Answer: B**

- A) Incorrect — LSF does not discard stderr. Without `-e`, it merges stderr into the stdout file specified by `-o`.
- B) Correct — When no `-e` flag is present, LSF redirects both stdout and stderr into the single `-o` output file. The Python traceback will appear in `job_<jobid>.out`, interleaved with any normal print output. This can make debugging confusing, which is why adding `#BSUB -e job_%J.err` is recommended best practice.
- C) Incorrect — LSF does not automatically create a `.err` file unless `-e` is explicitly specified with a filename. Auto-creation of a separate error file is not standard LSF behavior.
- D) Incorrect — stderr is not emailed. Only explicit `-B`/`-N` notification emails are sent; these carry job status information, not program output.

---
