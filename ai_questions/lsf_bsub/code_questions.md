# LSF / BSUB Job Scripts — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Total Memory Calculation](#q1-total-memory-calculation)
- [Q2 — Incorrect Per-Core Memory](#q2-incorrect-per-core-memory)
- [Q3 — Missing span[hosts=1] for Shared Memory](#q3-missing-spanhosts1-for-shared-memory)
- [Q4 — Converting CPU Script to GPU](#q4-converting-cpu-script-to-gpu)
- [Q5 — Dependency on a Failed Job](#q5-dependency-on-a-failed-job)
- [Q6 — done() vs ended() for Cleanup Jobs](#q6-done-vs-ended-for-cleanup-jobs)
- [Q7 — Job Array Index Off-By-One](#q7-job-array-index-off-by-one)
- [Q8 — Job Array Output File Naming](#q8-job-array-output-file-naming)
- [Q9 — Wall Time Format](#q9-wall-time-format)
- [Q10 — Email Notifications for Large Job Arrays](#q10-email-notifications-for-large-job-arrays)
- [Q11 — Map-Reduce Dependency with Partial Failures](#q11-map-reduce-dependency-with-partial-failures)
- [Q12 — Full Script Resource Summary](#q12-full-script-resource-summary)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q13 — Reading Total Memory from a Script](#q13-reading-total-memory-from-a-script)
- [Q14 — Fixing Over-Allocated Memory](#q14-fixing-over-allocated-memory)
- [Q15 — GPU Script: Spot the Bug](#q15-gpu-script-spot-the-bug)
- [Q16 — Two Scripts: Which Gets a GPU?](#q16-two-scripts-which-gets-a-gpu)
- [Q17 — span[hosts=1] in a Shared Memory Script](#q17-spanhosts1-in-a-shared-memory-script)
- [Q18 — Wall Time: Script Killed Mid-Run](#q18-wall-time-script-killed-mid-run)
- [Q19 — Interpreting bjobs Output](#q19-interpreting-bjobs-output)
- [Q20 — Per-Core Memory in MB vs GB](#q20-per-core-memory-in-mb-vs-gb)
- [Q21 — Combining rusage and span in -R](#q21-combining-rusage-and-span-in-r)
- [Q22 — bkill on a Running vs Pending Job](#q22-bkill-on-a-running-vs-pending-job)
- [Set 3 — Extended Practice](#set-3-extended-practice)
- [Q23 — OMP_NUM_THREADS Not Set: Thread Over-Subscription](#q23--omp_num_threads-not-set-thread-over-subscription)
- [Q24 — Non-Contiguous Array: Which Elements Run?](#q24--non-contiguous-array-which-elements-run)
- [Q25 — Dependency by Numeric Job ID](#q25--dependency-by-numeric-job-id)
- [Q26 — bsub Without Redirection](#q26--bsub-without-redirection)
- [Q27 — Memory Reserved vs Memory Used](#q27--memory-reserved-vs-memory-used)
- [Q28 — Spot the USUSP State](#q28--spot-the-ususp-state)
- [Q29 — Three -R Lines: What Gets Enforced?](#q29--three--r-lines-what-gets-enforced)
- [Q30 — Wrong GPU Queue for VRAM Requirement](#q30--wrong-gpu-queue-for-vram-requirement)
- [Q31 — stderr Missing from Output File](#q31--stderr-missing-from-output-file)
- [Q32 — Wall Time Minutes-Only with Zero Padding](#q32--wall-time-minutes-only-with-zero-padding)

---

> Format: Each question shows a BSUB script or command output to interpret or fix.
> Exam frequency: **Every exam** — second highest priority.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--total-memory-calculation)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — Total Memory Calculation

> **Week reference:** Week 1

You submit the following job script:

```bash
#!/bin/bash
#BSUB -J memjob
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 1:00
#BSUB -o mem_%J.out

python my_script.py
```

How much **total** memory does this job request from the cluster?

- A) 16 GB
- B) 32 GB
- C) 64 GB
- D) 128 GB

**Answer: D**

- A) Incorrect — 16 GB would be the per-core allocation, not the total.
- B) Incorrect — 32 GB would be 16 GB × 2 cores, not 8.
- C) Incorrect — 64 GB would be 16 GB × 4 cores.
- D) Correct — `rusage[mem=XGB]` specifies memory **per core**. With `-n 8`, total = 16 GB × 8 = **128 GB**.

---

## Q2 — Incorrect Per-Core Memory

> **Week reference:** Week 1

A job requires 32 GB of total memory and runs on 4 cores. A student writes:

```bash
#!/bin/bash
#BSUB -J bigmem
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 2:00
#BSUB -o bigmem_%J.out

python memory_heavy.py
```

What is the **correct** value for `rusage[mem=...]` to request exactly 32 GB total?

- A) `rusage[mem=32GB]`
- B) `rusage[mem=16GB]`
- C) `rusage[mem=8GB]`
- D) `rusage[mem=4GB]`

**Answer: C**

- A) Incorrect — This requests 32 GB × 4 cores = 128 GB total, which is 4× too much.
- B) Incorrect — This requests 16 GB × 4 cores = 64 GB total, still 2× too much.
- C) Correct — 8 GB × 4 cores = **32 GB** total. Since `rusage[mem=X]` is per-core, divide total by n_cores: 32 / 4 = 8 GB per core.
- D) Incorrect — This requests 4 GB × 4 cores = 16 GB total, only half the needed memory.

---

## Q3 — Missing span[hosts=1] for Shared Memory

> **Week reference:** Week 1 / Week 6

A script uses Python `multiprocessing` with a shared memory array that all worker processes must access simultaneously. The job script is:

```bash
#!/bin/bash
#BSUB -J shared_mem_job
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o shared_%J.out

python shared_array_processing.py
```

What critical flag is **missing** from this script?

- A) `-R "select[avx2]"` — to require AVX2 CPU instructions
- B) `-R "span[hosts=1]"` — to ensure all cores are on the same physical node
- C) `-R "affinity[core(1)]"` — to pin processes to specific cores
- D) `-P myproject` — to specify the project account

**Answer: B**

- A) Incorrect — AVX2 is for vectorized math, not required for shared memory.
- B) Correct — Without `span[hosts=1]`, LSF may distribute the 8 cores across multiple nodes. Shared memory arrays (via `multiprocessing.shared_memory` or `Array`) only exist within a single node's RAM — processes on different nodes **cannot** access the same shared memory.
- C) Incorrect — Core affinity is an optimization, not required for shared memory correctness.
- D) Incorrect — Project accounting is optional and unrelated to shared memory.

---

## Q4 — Converting CPU Script to GPU

> **Week reference:** Week 1 / Week 9

You have a working CPU job script and want to convert it to run on a GPU node:

```bash
#!/bin/bash
#BSUB -J cpu_training
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 2:00
#BSUB -o train_%J.out

python train_model.py
```

Which of the following is the **correct** modified script for a single GPU job?

- A)
```bash
#BSUB -J gpu_training
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1"
```

- B)
```bash
#BSUB -J gpu_training
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
```

- C)
```bash
#BSUB -J gpu_training
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
```

- D)
```bash
#BSUB -J gpu_training
#BSUB -q gpu
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
```

**Answer: B**

- A) Incorrect — Still uses the CPU queue `hpc`. GPU resources require the `gpuv100` queue.
- B) Correct — Changes the queue to `gpuv100` (the GPU queue on DTU HPC) and adds `-gpu "num=1:mode=exclusive_process"` to request one GPU in exclusive mode, preventing other jobs from sharing the GPU.
- C) Incorrect — Switches to the GPU queue but never requests a GPU resource with `-gpu`, so no GPU will be allocated.
- D) Incorrect — Uses `-q gpu` which is not the correct queue name on DTU HPC (it should be `gpuv100`).

---

## Q5 — Dependency on a Failed Job

> **Week reference:** Week 11

You have submitted a `prepare` job and a dependent `process` job:

```bash
# prepare job: jobID 12345
#!/bin/bash
#BSUB -J prepare
#BSUB -q hpc
#BSUB -n 1
python prepare_data.py

# process job submitted with:
# bsub -w "done(prepare)" < process.sh
```

You check `bjobs -a` and see the following output:

```
JOBID   USER    STAT  QUEUE      FROM_HOST   EXEC_HOST   JOB_NAME
12345   s252786 EXIT  hpc        login1      n-62-12-1   prepare
12346   s252786 PEND  hpc        login1      -           process
```

When will the `process` job start?

- A) Immediately — PEND means it is already queued and will start when resources are available
- B) After a 10-minute retry delay — LSF retries failed dependencies automatically
- C) Never — `done()` requires the dependency to finish in DONE (exit code 0) state; EXIT state means it will never be satisfied
- D) After manual intervention — an admin must reset the dependency flag

**Answer: C**

- A) Incorrect — PEND here means it is waiting on the dependency condition, not waiting for resources.
- B) Incorrect — LSF does not automatically retry jobs with unsatisfied `done()` dependencies.
- C) Correct — `-w "done(jobname)"` requires the named job to reach **DONE** state (successful exit code 0). A job in **EXIT** state (non-zero exit code or killed) will **never** satisfy this condition, so the dependent job stays in PEND forever and must be killed or resubmitted.
- D) Incorrect — No admin action automatically resolves this; the user must `bkill` the stuck job and resubmit.

---

## Q6 — done() vs ended() for Cleanup Jobs

> **Week reference:** Week 11

You have a pipeline where a `cleanup` job must run **regardless of whether the processing jobs succeed or fail** — it needs to remove temporary files either way. The processing job array is named `process`.

Which dependency flag should the cleanup job use?

```bash
#!/bin/bash
#BSUB -J cleanup
#BSUB -q hpc
#BSUB -n 1
# Which -w flag goes here?
python cleanup_tempfiles.py
```

- A) `-w "done(process)"` — ensures cleanup only runs after successful processing
- B) `-w "ended(process)"` — triggers after any termination (DONE or EXIT)
- C) `-w "exit(process)"` — triggers only if processing fails
- D) `-w "started(process)"` — triggers as soon as processing begins

**Answer: B**

- A) Incorrect — `done()` only triggers on **DONE** (success). If any process job exits with an error, the cleanup job would never run, leaving temporary files behind.
- B) Correct — `ended()` triggers when all jobs named `process` have reached **any** terminal state (DONE or EXIT). This guarantees cleanup runs whether processing succeeded or failed.
- C) Incorrect — `exit()` only triggers on failure, so cleanup would not run after a successful processing run.
- D) Incorrect — `started()` triggers when the named jobs begin running, not when they finish. Cleanup would start while processing is still in progress, which is incorrect.

---

## Q7 — Job Array Index Off-By-One

> **Week reference:** Week 11

A student submits a job array to process 5 datasets. The datasets are stored in a Python list at indices 0–4. The job script and Python code are:

```bash
#!/bin/bash
#BSUB -J sim[1-5]
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -o sim_%J_%I.out

python process.py $LSB_JOBINDEX
```

```python
# process.py
import sys

datasets = ["alpha", "beta", "gamma", "delta", "epsilon"]
idx = int(sys.argv[1])
print(f"Processing: {datasets[idx]}")
```

What happens when the **first** array element runs (element index 1)?

- A) It processes `datasets[0]` = "alpha" correctly
- B) It processes `datasets[1]` = "beta" — skipping "alpha" entirely
- C) It raises an `IndexError` because index 5 is out of range
- D) It processes `datasets[-1]` = "epsilon" due to Python negative indexing

**Answer: B**

- A) Incorrect — `$LSB_JOBINDEX` starts at **1**, not 0. The first element passes `1` to the script, which accesses `datasets[1]`.
- B) Correct — `$LSB_JOBINDEX` is **1-based**. The first job passes index 1, accessing "beta". "alpha" (`datasets[0]`) is never processed. The fix is `idx = int(sys.argv[1]) - 1`.
- C) Incorrect — An IndexError would occur for the **last** element if the array goes to 5 (accessing `datasets[5]`), not the first. Though here the last element also has a bug: `datasets[5]` would indeed raise IndexError since valid indices are 0–4.
- D) Incorrect — Python negative indexing would require a negative number; `$LSB_JOBINDEX` only produces positive integers 1–5.

---

## Q8 — Job Array Output File Naming

> **Week reference:** Week 11

A student submits a 10-element job array with this script:

```bash
#!/bin/bash
#BSUB -J compute[1-10]
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 0:30
#BSUB -o output_%J.out

python compute.py $LSB_JOBINDEX
```

What is wrong with the `-o` flag?

- A) Nothing — `%J` correctly captures the job array ID and all output goes to one file
- B) The file will be named `output_compute.out` instead of using a number
- C) All 10 array elements write to the **same** output file because `%J` is the parent job ID; the correct format is `output_%J_%I.out`
- D) The `-o` flag is invalid for array jobs; output must be redirected inside the Python script

**Answer: C**

- A) Incorrect — While it is technically valid, all 10 elements share the same file name, causing their outputs to be interleaved and making debugging nearly impossible.
- B) Incorrect — `%J` expands to the numeric job ID, not the job name.
- C) Correct — `%J` expands to the **parent array job ID** (e.g., `98765`), which is the same for all array elements. All 10 jobs write to `output_98765.out`, interleaving their output. The correct flag uses `%I` for the array index: `-o output_%J_%I.out`, producing `output_98765_1.out` through `output_98765_10.out`.
- D) Incorrect — The `-o` flag works perfectly well for array jobs when `%I` is included.

---

## Q9 — Wall Time Format

> **Week reference:** Week 1

A student submits the following script:

```bash
#!/bin/bash
#BSUB -J walltime_test
#BSUB -q hpc
#BSUB -n 2
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 1:30
#BSUB -o wt_%J.out

python long_computation.py
```

How much wall time is requested by `-W 1:30`?

- A) 1 minute and 30 seconds
- B) 1 hour and 30 minutes
- C) 1.30 hours (1 hour 18 minutes)
- D) 130 minutes (2 hours 10 minutes)

**Answer: B**

- A) Incorrect — The `-W` format is `HH:MM`, not `MM:SS`. Seconds are not specifiable with `-W`.
- B) Correct — The format for `-W` is `[hour:]minute`. The value `1:30` means **1 hour and 30 minutes** (90 minutes total). If only minutes were needed, you would write `-W 90`.
- C) Incorrect — LSF does not interpret `-W` as a decimal number.
- D) Incorrect — 130 minutes would be written as `-W 130` or `-W 2:10`.

---

## Q10 — Email Notifications for Large Job Arrays

> **Week reference:** Week 11

A student submits a 200-element job array with email notifications enabled:

```bash
#!/bin/bash
#BSUB -J bigarray[1-200]
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 0:30
#BSUB -B
#BSUB -N
#BSUB -u student@dtu.dk
#BSUB -o bigarray_%J_%I.out

python simulate.py $LSB_JOBINDEX
```

What is the problem with the email notification setup?

- A) `-B` and `-N` are mutually exclusive flags and will cause a submission error
- B) The `-u` flag requires the full email address with angle brackets: `-u <student@dtu.dk>`
- C) This will send **up to 400 emails** — one per element on start (`-B`) and one per element on end (`-N`) — flooding the inbox
- D) Email notifications do not work for job arrays; the flags are silently ignored

**Answer: C**

- A) Incorrect — `-B` (begin) and `-N` (end) can be used together without conflict.
- B) Incorrect — The `-u` flag accepts a plain email address without angle brackets.
- C) Correct — `-B` sends a notification email when **each** array element starts, and `-N` sends one when **each** element ends. With 200 elements: 200 start emails + 200 end emails = **400 emails**. For large arrays, it is best to omit `-B` and `-N`, or use array-level summary notifications if supported.
- D) Incorrect — The flags do work for array jobs; the problem is the sheer volume of emails generated.

---

## Q11 — Map-Reduce Dependency with Partial Failures

> **Week reference:** Week 11

A map-reduce pipeline uses two jobs: a `map` array and a `reduce` aggregation:

```bash
# map job (submitted first)
#BSUB -J map[1-10]
#BSUB -q hpc
#BSUB -n 1
python map_chunk.py $LSB_JOBINDEX
```

```bash
# reduce job
#BSUB -J reduce
#BSUB -q hpc
#BSUB -n 1
#BSUB -w "done(map)"
python reduce_results.py
```

After submission, you check `bjobs -a` and see that 8 of the 10 `map` jobs are in **DONE** state, but **2 are in EXIT** state. What happens to the `reduce` job?

- A) It starts immediately because the majority (80%) of map jobs succeeded
- B) It starts after a timeout, using only the 8 successful results
- C) It never starts — `done(map)` requires **all** elements of the array to reach DONE; two EXIT jobs prevent this condition from ever being satisfied
- D) It starts, but LSF automatically skips the 2 failed results

**Answer: C**

- A) Incorrect — LSF has no concept of "majority success"; `done()` is all-or-nothing.
- B) Incorrect — There is no automatic timeout or partial-result fallback in LSF job dependencies.
- C) Correct — `-w "done(map)"` requires **every** element of the `map[1-10]` array to be in DONE state. Because 2 elements are in EXIT state, the condition is permanently unsatisfied and `reduce` will remain in PEND indefinitely. To allow reduce to proceed even with partial failures, use `-w "ended(map)"` instead.
- D) Incorrect — LSF does not inspect or filter results; it only checks job completion status codes.

---

## Q12 — Full Script Resource Summary

> **Week reference:** Week 1

You are given the following complete job script:

```bash
#!/bin/bash
#BSUB -J full_job
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 2:00
#BSUB -o full_%J.out
#BSUB -e full_%J.err

python parallel_computation.py
```

How many CPU cores and how much **total** memory does this job request?

- A) 16 cores, 4 GB total
- B) 16 cores, 16 GB total
- C) 16 cores, 64 GB total
- D) 1 core, 64 GB total (span[hosts=1] reduces to 1 core)

**Answer: C**

- A) Incorrect — 4 GB is the per-core allocation, not the total.
- B) Incorrect — 16 GB would be 4 GB × 4 cores; this job uses 16 cores.
- C) Correct — `-n 16` requests **16 CPU cores**. `rusage[mem=4GB]` is **per core**, so total memory = 4 GB × 16 = **64 GB**. `span[hosts=1]` constrains all 16 cores to a single physical node (required for shared memory usage) but does not change the core or memory count. `-W 2:00` sets a 2-hour wall time limit.
- D) Incorrect — `span[hosts=1]` means "all requested cores must be on one host"; it does not reduce the number of cores to 1.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets rusage[mem=X] per-core semantics, GPU queue flags, -n vs span[hosts=1], wall time formats, and bjobs output interpretation

---

## Q13 — Reading Total Memory from a Script

> **Week reference:** Week 1

You receive this job script from a colleague:

```bash
#!/bin/bash
#BSUB -J analysis
#BSUB -q hpc
#BSUB -n 6
#BSUB -R "rusage[mem=3GB]"
#BSUB -W 0:45
#BSUB -o analysis_%J.out

python run_analysis.py
```

How much **total** memory does this script request from the cluster?

- A) 3 GB
- B) 9 GB
- C) 18 GB
- D) 0.5 GB

**Answer: C**

`rusage[mem=X]` is per-core. With `-n 6` cores: 3 GB × 6 = **18 GB** total. A is the trap — reading `rusage[mem=3GB]` as the total rather than per-core. B (9 GB) would result from 3 GB × 3, as if only half the cores were counted. D inverts the math (3 GB ÷ 6), as if dividing rather than multiplying.

---

## Q14 — Fixing Over-Allocated Memory

> **Week reference:** Week 1

A student needs **12 GB** total for their job. They write:

```bash
#!/bin/bash
#BSUB -J overalloc
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=12GB]"
#BSUB -W 1:00
#BSUB -o overalloc_%J.out

python heavy_computation.py
```

How much memory is **actually** reserved, and what is the corrected `rusage` line?

- A) 12 GB is reserved; no change needed
- B) 48 GB is reserved; change to `rusage[mem=3GB]`
- C) 48 GB is reserved; change to `rusage[mem=6GB]`
- D) 3 GB is reserved; change to `rusage[mem=12GB]`

**Answer: B**

`rusage[mem=12GB]` with `-n 4` reserves 12 × 4 = 48 GB — four times more than needed. The correct per-core value is 12 GB ÷ 4 = 3 GB. C applies the wrong divisor (÷2 instead of ÷4). D reads the rusage as per-core and tries to multiply up, but that math also yields 3 × 4 = 12 GB, making it a plausible distractor — however the question states the reservation is wrong, and `rusage[mem=3GB]` is the fix.

---

## Q15 — GPU Script: Spot the Bug

> **Week reference:** Week 1 / Week 9

A student submits the following script to run a CUDA neural network training job:

```bash
#!/bin/bash
#BSUB -J cuda_train
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 4:00
#BSUB -o train_%J.out

python train_cuda.py
```

The job is submitted successfully but immediately fails when `train_cuda.py` tries to initialize CUDA. What is the most likely cause?

- A) The wall time `-W 4:00` is too long for the GPU queue
- B) `-n 4` requests too many cores for a GPU job; GPU jobs must use `-n 1`
- C) The script never requested a GPU with `-gpu "num=1:mode=exclusive_process"`; no GPU device was allocated
- D) `rusage[mem=8GB]` syntax is invalid for the gpuv100 queue

**Answer: C**

Switching to the `gpuv100` queue routes the job to a node that has GPUs, but it does not allocate a GPU to the process. Without `#BSUB -gpu "num=1:mode=exclusive_process"`, CUDA initialization finds zero devices and raises a runtime error. The queue flag and the GPU allocation flag are independent — you need both. Wall time, core count, and memory syntax are all fine.

---

## Q16 — Two Scripts: Which Gets a GPU?

> **Week reference:** Week 1 / Week 9

Compare these two job scripts. Which one will successfully use a GPU?

**Script A:**
```bash
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
python use_gpu.py
```

**Script B:**
```bash
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
python use_gpu.py
```

- A) Script A — it is in the GPU queue so a GPU is allocated automatically
- B) Script B — it includes the `-gpu` flag that explicitly reserves a GPU device
- C) Both — the GPU queue always allocates one GPU per job
- D) Neither — you also need `-R "select[gpu=1]"` for GPU allocation

**Answer: B**

Script B includes `#BSUB -gpu "num=1:mode=exclusive_process"`, which is the directive that actually reserves and allocates the GPU device to the job. Script A lands on a GPU node (correct queue) but receives no GPU allocation — CUDA will see zero devices. The `-gpu` flag is mandatory for GPU access; the queue alone is not sufficient.

---

## Q17 — span[hosts=1] in a Shared Memory Script

> **Week reference:** Week 1 / Week 6

A student writes a job script for a shared-memory parallel program:

```bash
#!/bin/bash
#BSUB -J shmem
#BSUB -q hpc
#BSUB -n 12
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 1:30
#BSUB -o shmem_%J.out

python shared_parallel.py
```

`shared_parallel.py` uses `multiprocessing.shared_memory.SharedMemory` to share a large array across all 12 worker processes. What is the critical problem with this script?

- A) `-n 12` requests too many cores; shared memory only works with up to 8 processes
- B) `rusage[mem=2GB]` is too low; shared memory requires at least 8 GB per core
- C) The script is missing `#BSUB -R "span[hosts=1]"`; without it, LSF may spread the 12 cores across multiple nodes where shared memory is inaccessible
- D) The `-W 1:30` format is incorrect for jobs using shared memory

**Answer: C**

`multiprocessing.shared_memory.SharedMemory` creates a region in a single node's physical RAM. If LSF distributes the 12 cores across two or more nodes (which it may do without `span[hosts=1]`), processes on different nodes cannot access the same shared memory segment. Adding `#BSUB -R "span[hosts=1]"` forces all 12 cores onto a single physical node. Core count limit, memory per core, and wall time format are unrelated to shared memory correctness.

---

## Q18 — Wall Time: Script Killed Mid-Run

> **Week reference:** Week 1

A student submits this job:

```bash
#!/bin/bash
#BSUB -J longrun
#BSUB -q hpc
#BSUB -n 2
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o longrun_%J.out

python simulate.py  # takes about 45 minutes
```

What happens to the job?

- A) LSF extends the wall time automatically when it detects the job is still running
- B) The job is killed after 30 minutes without warning; output written before that point is preserved
- C) The job is paused after 30 minutes and enters USUSP state, resumable later
- D) `-W 0:30` is invalid syntax; the job will be rejected at submission time

**Answer: B**

`-W 0:30` sets a 30-minute hard wall time limit. LSF kills the job when the limit is reached, regardless of progress. Output files written before termination are preserved (they are on the filesystem), but any in-memory results not yet saved are lost. LSF does not extend wall time automatically, pause jobs into USUSP, or reject `HH:MM` format — `0:30` is perfectly valid (0 hours, 30 minutes).

---

## Q19 — Interpreting bjobs Output

> **Week reference:** Week 1

You run `bjobs -a` and see:

```
JOBID   USER    STAT  QUEUE   FROM_HOST   EXEC_HOST   JOB_NAME   SUBMIT_TIME
78901   s252786 RUN   hpc     login1      n-62-18-2   crunch     May 30 14:02
78902   s252786 PEND  hpc     login1      -           crunch2    May 30 14:03
78903   s252786 EXIT  hpc     login1      n-62-10-5   prep       May 30 13:50
78904   s252786 DONE  hpc     login1      n-62-10-5   setup      May 30 13:45
```

Which of the following statements is correct?

- A) `crunch2` (PEND) is currently running on a node — PEND means the node is pending a health check
- B) `prep` (EXIT) succeeded; EXIT means it exited normally from the queue
- C) `crunch` (RUN) is actively executing on node `n-62-18-2`; `crunch2` (PEND) is waiting for resources or a dependency
- D) `setup` (DONE) failed; DONE is a transitional state before final status is assigned

**Answer: C**

RUN means actively executing on the listed EXEC_HOST. PEND means waiting in queue (no EXEC_HOST shown — the dash confirms it has not started). EXIT means the job terminated with a non-zero exit code (failure). DONE means the job completed successfully (exit code 0). These are the four most common statuses and their exact meanings are frequently tested.

---

## Q20 — Per-Core Memory in MB vs GB

> **Week reference:** Week 1

A script specifies `#BSUB -R "rusage[mem=4096]"` with `-n 8`. The cluster interprets `rusage[mem=X]` values without a unit suffix as **MB**. What is the total memory reserved?

- A) 4096 MB (4 GB) total
- B) 32768 MB (32 GB) total
- C) 512 MB (0.5 GB) total
- D) 4096 GB total — LSF assumes GB when no unit is given

**Answer: B**

`rusage[mem=4096]` without a unit suffix is 4096 MB per core. With `-n 8`: 4096 MB × 8 = 32768 MB = 32 GB total. A is the per-core value, not total. C inverts the operation (4096 ÷ 8). D is wrong — the default unit in LSF is MB, not GB; 4096 GB would be an enormous and implausible allocation.

---

## Q21 — Combining rusage and span in -R

> **Week reference:** Week 1

A job script contains:

```bash
#BSUB -n 8
#BSUB -R "rusage[mem=2GB] span[hosts=1]"
```

What does this single `-R` string request?

- A) 2 GB total memory spread across 8 nodes, one node per core
- B) 2 GB per core (16 GB total), all 8 cores on a single node
- C) 2 GB per core only if the node has at least 1 GB free
- D) 8 GB total (2 GB per pair of cores) distributed across available hosts

**Answer: B**

A single `-R` string can combine multiple resource clauses separated by spaces. `rusage[mem=2GB]` specifies 2 GB per core; with `-n 8` that is 16 GB total. `span[hosts=1]` forces all 8 cores onto one physical node. The two clauses are independent and both apply simultaneously. LSF does not split rusage across nodes, pair cores, or interpret span as a node count.

---

## Q22 — bkill on a Running vs Pending Job

> **Week reference:** Week 1

You want to cancel a job that is currently in RUN state (job ID 55001) and another that is in PEND state (job ID 55002). Which command correctly cancels both?

```bash
# Option A
bkill 55001 55002

# Option B
bstop 55001 && bdel 55002

# Option C
bkill -pend 55002 && bkill -run 55001

# Option D
bjobs -kill 55001 55002
```

- A) Option A
- B) Option B
- C) Option C
- D) Option D

**Answer: A**

`bkill` works on jobs in any state — RUN, PEND, USUSP, etc. You can pass multiple job IDs in one command. `bstop` suspends a running job (sets it to USUSP) rather than killing it. `bdel` is not a standard LSF command. `bkill -pend` and `bkill -run` are not valid flag forms. `bjobs -kill` does not exist. Option A is the simplest and correct approach.

---

## Set 3 — Extended Practice

- [Q23 — OMP_NUM_THREADS Not Set: Thread Over-Subscription](#q23--omp_num_threads-not-set-thread-over-subscription)
- [Q24 — Non-Contiguous Array: Which Elements Run?](#q24--non-contiguous-array-which-elements-run)
- [Q25 — Dependency by Numeric Job ID](#q25--dependency-by-numeric-job-id)
- [Q26 — bsub Without Redirection](#q26--bsub-without-redirection)
- [Q27 — Memory Reserved vs Memory Used](#q27--memory-reserved-vs-memory-used)
- [Q28 — Spot the USUSP State](#q28--spot-the-ususp-state)
- [Q29 — Three -R Lines: What Gets Enforced?](#q29--three--r-lines-what-gets-enforced)
- [Q30 — Wrong GPU Queue for VRAM Requirement](#q30--wrong-gpu-queue-for-vram-requirement)
- [Q31 — stderr Missing from Output File](#q31--stderr-missing-from-output-file)
- [Q32 — Wall Time Minutes-Only with Zero Padding](#q32--wall-time-minutes-only-with-zero-padding)

---

## Q23 — OMP_NUM_THREADS Not Set: Thread Over-Subscription

> **Week reference:** Week 1

**Mental Model:** `-n` reserves slots from LSF but does not constrain the threads a program creates at runtime — without `OMP_NUM_THREADS` matching `-n`, a multi-threaded library may use all physical cores on the node regardless of what was requested.

Consider this job script:

```bash
#!/bin/bash
#BSUB -J omp_job
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 0:30
#BSUB -o omp_%J.out

python numpy_heavy.py  # uses scipy.linalg which calls MKL internally
```

The node has 40 physical cores. What is the likely thread count used by MKL during `numpy_heavy.py`, and why is this a problem?

- A) 4 threads — MKL reads LSF's `-n 4` and limits itself accordingly
- B) 40 threads — MKL detects the node's physical core count and spawns a thread per core, over-subscribing the 4 reserved slots by 10×
- C) 1 thread — without `OMP_NUM_THREADS`, MKL defaults to single-threaded mode for safety
- D) 2 threads — MKL uses half the physical cores as a conservative default

**Answer: B**

- A) Incorrect — MKL has no awareness of LSF job slots. It does not read `-n` or any LSF environment variable to learn how many cores were reserved.
- B) Correct — Without `OMP_NUM_THREADS` (or `MKL_NUM_THREADS`) set, MKL queries the hardware and spawns one thread per physical core — 40 threads on a 40-core node. This consumes CPU time from all 40 cores, effectively stealing resources from other users' jobs sharing the same node. The fix is to add `export OMP_NUM_THREADS=4` before the Python call.
- C) Incorrect — Defaulting to 1 thread is not MKL or OpenMP behavior. They default to hardware concurrency, not safe single-threaded operation.
- D) Incorrect — There is no "half cores" default in MKL or OpenMP. Half-core defaults are not documented behavior for any common HPC math library.

---

## Q24 — Non-Contiguous Array: Which Elements Run?

> **Week reference:** Week 11

**Mental Model:** A comma-separated index list in a job array creates exactly the listed indices as `$LSB_JOBINDEX` values — no elements between them are created, and the count is exactly the number of items in the list.

A student submits this script:

```bash
#!/bin/bash
#BSUB -J rerun[2,29,71,127]
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=1GB]"
#BSUB -o rerun_%J_%I.out

echo "Index: $LSB_JOBINDEX"
```

How many jobs are submitted, and what values does `$LSB_JOBINDEX` take?

- A) 126 jobs with indices 2 through 127 (a full contiguous range)
- B) 4 jobs with `$LSB_JOBINDEX` = 2, 29, 71, 127
- C) 4 jobs with `$LSB_JOBINDEX` = 1, 2, 3, 4 (re-mapped to sequential order)
- D) 1 job with `$LSB_JOBINDEX` = "2,29,71,127" as a string

**Answer: B**

- A) Incorrect — Comma syntax does not create a range. To get a range you would use `[2-127]`. The comma list creates exactly the enumerated indices.
- B) Correct — `[2,29,71,127]` creates precisely 4 array elements. Each element's `$LSB_JOBINDEX` is the literal value from the list: the first element has index 2, the second has 29, the third 71, the fourth 127. This is ideal for re-running a specific subset of a failed array.
- C) Incorrect — LSF does not re-map non-contiguous indices to sequential integers. The actual listed values are used directly as `$LSB_JOBINDEX`.
- D) Incorrect — `$LSB_JOBINDEX` is always a single integer per element. It is never a string containing the full list.

---

## Q25 — Dependency by Numeric Job ID

> **Week reference:** Week 11

**Mental Model:** Using a numeric job ID in a dependency expression (`done(12345678)`) creates a point-in-time dependency on exactly one specific submission — unlike name-based dependencies, it is immune to other jobs that happen to share the same name.

You have this dependency setup:

```bash
# Step 1 — submitted first, gets job ID 21241475
#BSUB -J preprocess
...

# Step 2 — uses numeric ID from step 1
#BSUB -J analyze
#BSUB -w "done(21241475)"
...
```

A colleague also submits a job named `preprocess` (job ID 21241600) after your step 1 is running. How does `analyze` behave?

- A) `analyze` waits for both `preprocess` jobs (21241475 and 21241600) because the name matches
- B) `analyze` waits only for job 21241475 to reach DONE, regardless of job 21241600
- C) The dependency becomes invalid because another job with the same name exists
- D) `analyze` starts immediately because the numeric ID is treated as a timestamp, not a job reference

**Answer: B**

- A) Incorrect — The dependency uses the numeric job ID `21241475`, not the name `preprocess`. Numeric ID dependencies are fully independent of job names and unaffected by other jobs.
- B) Correct — `done(21241475)` pins the dependency to exactly one job: the specific submission with that ID. Your colleague's job 21241600 is completely irrelevant. This is the key advantage of ID-based over name-based dependencies in shared cluster environments.
- C) Incorrect — The existence of other jobs with matching names has no effect on numeric-ID-based dependencies.
- D) Incorrect — Job IDs are unique sequential numbers assigned by the LSF scheduler, not timestamps. `done(21241475)` is a legitimate job reference, not a time-based condition.

---

## Q26 — bsub Without Redirection

> **Week reference:** Week 1

**Mental Model:** `bsub < script.sh` reads the script as stdin so LSF parses `#BSUB` directives; `bsub script.sh` passes the filename as a positional argument and `#BSUB` lines inside the file are not parsed as directives.

A student runs:

```bash
bsub submit.sh
```

instead of:

```bash
bsub < submit.sh
```

The `submit.sh` file contains:

```bash
#!/bin/bash
#BSUB -J myjob
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 2:00
python heavy.py
```

What resources does the submitted job actually get?

- A) Exactly the resources specified in the `#BSUB` lines: 8 cores, 32 GB, 2-hour wall time
- B) LSF default resources (typically 1 core, small memory, short wall time) because the `#BSUB` directives inside the file are not parsed
- C) The job is rejected with a "missing queue" error because `-q` was not passed on the command line
- D) The job runs with 8 cores but uses the default wall time because only `-n` is recognized without `<`

**Answer: B**

- A) Incorrect — `#BSUB` directives inside a file are only parsed when the file is fed as stdin via `<`. Passing the filename as a positional argument does not trigger directive parsing.
- B) Correct — Without `<`, LSF treats `submit.sh` as the command to execute and uses all default resource values: typically 1 core, minimal memory, and the queue's default wall time. The `#BSUB` lines are just comments to the shell; they are never seen by the LSF parser in this invocation mode.
- C) Incorrect — LSF does not require `-q` on the command line; it falls back to a default queue. The job submits without error but with wrong resources.
- D) Incorrect — There is no selective parsing; either all `#BSUB` directives are parsed (via `<`) or none of them are (positional argument mode).

---

## Q27 — Memory Reserved vs Memory Used

> **Week reference:** Week 1

**Mental Model:** LSF's memory accounting is based on the requested reservation, not runtime usage — a job that requests excess cores locks up proportionally more memory on the cluster even if the program never touches it.

A job script contains:

```bash
#!/bin/bash
#BSUB -J singlecore
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1:00
#BSUB -o single_%J.out

python single_threaded_analysis.py  # uses exactly 1 core and ~1.5 GB RAM
```

How much memory is charged to this job by LSF, and is the job a good cluster citizen?

- A) 1.5 GB — LSF monitors actual usage and charges accordingly; the job is efficient
- B) 32 GB is reserved (16 × 2 GB); the program uses ~1.5 GB — 30.5 GB is blocked for the job's duration and unavailable to others; the script wastes cluster resources
- C) 2 GB is reserved because `span[hosts=1]` limits memory to the per-core value
- D) 32 GB is reserved, but this is correct because span[hosts=1] requires reserving memory for all 16 potential cores

**Answer: B**

- A) Incorrect — LSF charges based on the static reservation at job submission, not on runtime measurement of actual memory use. Monitoring tools (like `bjobs -l`) can show peak memory, but billing is based on reserved slots.
- B) Correct — With `-n 16` and `rusage[mem=2GB]`, LSF reserves 16 × 2 = 32 GB for the entire wall-time duration. The single-threaded program only uses ~1.5 GB. The remaining ~30.5 GB reservation prevents other jobs from using those memory slots, reducing cluster throughput. The correct script should use `-n 1`.
- C) Incorrect — `span[hosts=1]` constrains node topology, not the number of cores or the total memory reservation. It has no effect on how `rusage[mem=X]` is multiplied.
- D) Incorrect — `span[hosts=1]` means all 16 cores must be on one node; it does not justify requesting 16 cores for a single-threaded program. The `-n` value should match the program's actual parallelism.

---

## Q28 — Spot the USUSP State

> **Week reference:** Week 1

**Mental Model:** `bstop <jobid>` transitions a running job to USUSP; the job holds its nodes but stops executing; `bresume <jobid>` is the only way to restart it — killing it with `bkill` is permanent.

You run `bjobs` and see this output:

```
JOBID   USER    STAT   QUEUE   FROM_HOST   EXEC_HOST   JOB_NAME
66100   s252786 RUN    hpc     login1      n-62-05-3   crunch
66101   s252786 USUSP  hpc     login1      n-62-07-1   longrun
66102   s252786 PEND   hpc     login1      -           next_step
```

Which of the following accurately describes the situation?

- A) `longrun` (USUSP) has crashed and its node n-62-07-1 has been freed for reuse
- B) `longrun` (USUSP) is paused and still occupying node n-62-07-1; it can be resumed with `bresume 66101`
- C) `next_step` (PEND) cannot start because `longrun` is in USUSP — PEND always means waiting for a suspended job to finish
- D) `crunch` (RUN) and `longrun` (USUSP) are sharing node n-62-07-1 because USUSP frees half its cores

**Answer: B**

- A) Incorrect — USUSP does not free the node. The job retains its allocation on n-62-07-1 while suspended. The node's cores and memory remain reserved for `longrun`.
- B) Correct — USUSP means the job is suspended (likely by the user via `bstop 66101`). It holds node n-62-07-1 but is not consuming CPU. `bresume 66101` restarts execution. Note: `next_step` is PEND likely due to a resource wait, not because of `longrun`'s USUSP state specifically.
- C) Incorrect — PEND means waiting for resources or a dependency, but it is not automatically caused by another job being in USUSP. Unless `next_step` has an explicit dependency on `longrun`, its PEND state is independent.
- D) Incorrect — USUSP does not release or share cores. The suspended job retains its full allocation; no resource sharing occurs during suspension.

---

## Q29 — Three -R Lines: What Gets Enforced?

> **Week reference:** Week 1

**Mental Model:** Every `#BSUB -R` line adds a constraint that is ANDed with all others — a node must satisfy all constraints simultaneously to be eligible for the job.

A job script has these directives:

```bash
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6226R]"
```

Which of the following correctly describes what LSF enforces?

- A) Only the last `-R` line (`select[model==XeonGold6226R]`) applies; the others are overridden
- B) All three `-R` constraints apply simultaneously: 8 GB per core (32 GB total), all 4 cores on one node, and only on XeonGold6226R nodes
- C) `rusage` and `span` apply, but `select` is ignored because model selection requires a separate `-select` flag
- D) The three `-R` lines conflict with each other; LSF will reject the submission

**Answer: B**

- A) Incorrect — LSF accumulates all `-R` lines. There is no override behavior; each line adds a requirement.
- B) Correct — All three constraints apply simultaneously. `rusage[mem=8GB]` reserves 8 GB per core (4 cores × 8 = 32 GB total). `span[hosts=1]` forces all 4 cores onto a single physical node. `select[model==XeonGold6226R]` restricts eligible nodes to those with that specific CPU model. A node must satisfy all three to host this job.
- C) Incorrect — `select[]` is a fully supported clause inside a `-R` string. There is no separate `-select` flag; it belongs in `-R "select[...]"` exactly as shown.
- D) Incorrect — Multiple `-R` lines with different clause types do not conflict. LSF is designed to combine them; this pattern appears in the course's own `submit.sh` examples.

---

## Q30 — Wrong GPU Queue for VRAM Requirement

> **Week reference:** Week 1 / Week 9

**Mental Model:** Each DTU HPC GPU queue maps to specific hardware with fixed VRAM — submitting to a queue whose GPUs have less VRAM than the model requires produces a runtime OOM error, not a submission error.

A student submits this script to train a large transformer model that requires ~20 GB of GPU memory:

```bash
#!/bin/bash
#BSUB -J transformer
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -o transformer_%J.out

python train_transformer.py
```

The job submits without error, starts running, then crashes after a few minutes. What is the most likely cause?

- A) `-n 4` is incompatible with `-gpu "num=1"` — GPU jobs must use `-n 1`
- B) The V100 GPU has only 16 GB of VRAM; the model requires ~20 GB and triggers an out-of-memory error during training
- C) `rusage[mem=8GB]` is too low for the CPU memory and causes a CPU OOM kill
- D) The `gpuv100` queue has a maximum wall time of 1 hour; `-W 4:00` causes an immediate kill

**Answer: B**

- A) Incorrect — GPU jobs can use multiple CPU cores (`-n 4` is fine). The CPU cores handle data loading and preprocessing while the GPU handles model computation.
- B) Correct — `gpuv100` nodes have V100 GPUs with **16 GB** of VRAM. A model requiring ~20 GB exceeds this and causes a CUDA out-of-memory error at runtime (e.g., `RuntimeError: CUDA out of memory`). The fix is to use `#BSUB -q gpu32` which provides GPUs with 32 GB VRAM, or to reduce the model size/batch size.
- C) Incorrect — `rusage[mem=8GB]` per core × 4 cores = 32 GB CPU RAM, which is more than adequate for typical transformer training CPU memory overhead.
- D) Incorrect — The `gpuv100` queue's wall time limits are measured in hours, not imposed as a 1-hour cap. A 4-hour request is within normal range.

---

## Q31 — stderr Missing from Output File

> **Week reference:** Week 1

**Mental Model:** Without `-e`, LSF writes both stdout and stderr to the `-o` file; with `-e`, they go to separate files — a Python exception traceback going to stderr will only appear in the `-e` file if one is specified, and will be silently absent from the `-o` file only if `-e` is set to `/dev/null`.

A student runs a job with this script:

```bash
#!/bin/bash
#BSUB -J debug_job
#BSUB -q hpc
#BSUB -n 2
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 0:10
#BSUB -o debug_%J.out
#BSUB -e debug_%J.err

python buggy_script.py
```

`buggy_script.py` crashes immediately with a `ValueError` traceback. The student opens `debug_12345.out` after the job ends and finds it completely empty. Where is the traceback?

- A) It was lost — LSF discards stderr when `-e` is specified
- B) It is in `debug_12345.err` — the `-e` flag redirects stderr to a separate file, and Python tracebacks go to stderr
- C) It is in `debug_12345.out` — Python always writes tracebacks to stdout
- D) It was emailed to the student because the job exited with a non-zero code

**Answer: B**

- A) Incorrect — LSF does not discard stderr when `-e` is given. It redirects it to the specified file. `-e /dev/null` would discard it, but `-e debug_%J.err` saves it.
- B) Correct — Python prints exception tracebacks to `sys.stderr`, not `sys.stdout`. With `#BSUB -e debug_%J.err`, LSF captures stderr into `debug_12345.err`. The `-o` file (`debug_12345.out`) only receives stdout, which is empty if the script crashes before any `print()` call.
- C) Incorrect — Python tracebacks go to stderr by default. Only explicit `print()` statements (or `sys.stdout.write()`) appear in stdout/the `-o` file.
- D) Incorrect — LSF's email notification (`-N`/`-B`) sends job status information, not program output. Tracebacks are never emailed automatically.

---

## Q32 — Wall Time Minutes-Only with Zero Padding

> **Week reference:** Week 1

**Mental Model:** The `#BSUB -W` directive accepts either `HH:MM` or a plain integer (minutes) — zero-padded hours like `00:10` are valid HH:MM format meaning 0 hours 10 minutes, which is exactly 10 minutes.

Consider these two job scripts:

**Script A:**
```bash
#BSUB -W 00:10
python quick_test.py
```

**Script B:**
```bash
#BSUB -W 10
python quick_test.py
```

What wall time does each script request, and are they equivalent?

- A) Script A requests 10 minutes; Script B requests 10 hours — they are not equivalent
- B) Script A requests 0 minutes (the `00` is parsed as zero, discarding `:10`); Script B requests 10 minutes
- C) Both request exactly 10 minutes and are equivalent
- D) Script A requests 1 hour (LSF ignores the `00:` prefix); Script B requests 10 minutes — they differ

**Answer: C**

- A) Incorrect — `HH:MM` format with `00:10` means 0 hours and 10 minutes = 10 minutes. The plain integer `10` also means 10 minutes. Both are 10 minutes.
- B) Incorrect — LSF parses `00:10` as the standard HH:MM format: 00 hours + 10 minutes. It does not discard the minutes portion. Zero-padding in the hours position is valid and common in course job scripts.
- C) Correct — `00:10` in HH:MM format is 0 hours + 10 minutes = 10 minutes. The plain integer `10` is also 10 minutes. The two forms are fully equivalent. This pattern is seen in the course's `job_arrays_1.sh` (`-W 00:10`), confirming zero-padded format is the standard style used at DTU HPC.
- D) Incorrect — LSF has no behavior that ignores the `00:` prefix or interprets it as 1 hour. HH:MM is always parsed as hours:minutes.

---
