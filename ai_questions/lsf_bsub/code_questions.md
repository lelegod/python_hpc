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
