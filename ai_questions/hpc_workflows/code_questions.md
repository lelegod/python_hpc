# HPC Workflows — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — 1-Based LSB_JOBINDEX Off-by-One](#q1--1-based-lsb_jobindex-off-by-one)
- [Q2 — %J vs %I in Output File Names](#q2--j-vs-i-in-output-file-names)
- [Q3 — Background Process Orphan Problem](#q3--background-process-orphan-problem)
- [Q4 — wait After kill for Clean Termination](#q4--wait-after-kill-for-clean-termination)
- [Q5 — Email Notification Count with -N](#q5--email-notification-count-with--n)
- [Q6 — Pool() Uses os.cpu_count() Not LSF Cores](#q6--pool-uses-oscpu_count-not-lsf-cores)
- [Q7 — Step Syntax [1-20:3] Index Values](#q7--step-syntax-1-203-index-values)
- [Q8 — done() Requires All DONE, any EXIT Blocks](#q8--done-requires-all-done-any-exit-blocks)
- [Key Facts Summary](#key-facts-summary)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q9 — Spotting the done() Blocker](#q9-spotting-the-done-blocker)
- [Q10 — Fixing Output File Collision](#q10-fixing-output-file-collision)
- [Q11 — Reading Array Index in Python](#q11-reading-array-index-in-python)
- [Q12 — Concurrent Slot Limit Syntax](#q12-concurrent-slot-limit-syntax)
- [Q13 — Three-Stage Pipeline With done() and ended()](#q13-three-stage-pipeline-with-done-and-ended)
- [Q14 — Identifying the Correct Index in a File List](#q14-identifying-the-correct-index-in-a-file-list)
- [Q15 — done() With Specific Job ID](#q15-done-with-specific-job-id)
- [Q16 — Diagnosing a Stuck Pipeline](#q16-diagnosing-a-stuck-pipeline)
- [Q17 — Constructing the Output Filename](#q17-constructing-the-output-filename)
- [Q18 — Combining Array Index With Python Logic](#q18-combining-array-index-with-python-logic)

---

> Format: Each question shows a job array script or Python HPC code with patterns to analyse.
> Exam frequency: **2024 exam + F25**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#question-1)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — 1-Based LSB_JOBINDEX Off-by-One

A Python job array script processes files in a directory:

```bash
#BSUB -J process[1-10]
#BSUB -n 1
python process_files.py $LSB_JOBINDEX
```

```python
import sys
import os
import numpy as np

idx = int(sys.argv[1])   # BUG: should be int(sys.argv[1]) - 1
files = sorted(os.listdir('/data/'))
process(files[idx])
```

When `$LSB_JOBINDEX=1`, what file does this script access, and what happens when `$LSB_JOBINDEX=10` (assuming exactly 10 files)?

**A)** Accesses the first file (index 0); the last job accesses `files[9]` — no error  
**B)** Accesses the second file (index 1), skipping `files[0]`; the last job tries `files[10]`, raising `IndexError`  
**C)** Accesses `files[1]` and wraps around on the last job  
**D)** `$LSB_JOBINDEX` is 0-based, so this is correct

**Answer: B**

- A) Incorrect — `$LSB_JOBINDEX` is 1-based, so `idx=1` accesses `files[1]`, not `files[0]`
- B) Correct — index 1 skips `files[0]`; index 10 is out of range for a 10-element list, raising `IndexError`
- C) Incorrect — Python lists do not wrap around on out-of-bounds access; they raise `IndexError`
- D) Incorrect — `$LSB_JOBINDEX` is 1-based (starts at 1, not 0)

---

## Q2 — %J vs %I in Output File Names

A job array is submitted with this BSUB header:

```bash
#!/bin/bash
#BSUB -J analysis[1-5]
#BSUB -n 1
#BSUB -o output_%J.out
#BSUB -e error_%J.err

python analysis.py $LSB_JOBINDEX
```

All 5 array elements run simultaneously. What happens to the output files?

**A)** Each element writes to its own file: `output_1.out`, `output_2.out`, ..., `output_5.out`  
**B)** All 5 elements write to the same file `output_<arrayJobID>.out`, causing interleaved/corrupted output  
**C)** LSF automatically creates separate files for array elements when using `%J`  
**D)** Only the first element writes output; the rest are silently discarded

**Answer: B**

- A) Incorrect — `%J` expands to the parent job ID (identical for all elements), not per-element indices
- B) Correct — all 5 jobs share the same `output_<parentJobID>.out` file, producing interleaved output
- C) Incorrect — `%J` gives only the parent job ID; `%I` is needed for per-element separation
- D) Incorrect — all elements write output, but they collide into the same file

---

## Q3 — Background Process Orphan Problem

A job script launches a background monitor and then runs the main computation:

```bash
#!/bin/bash
#BSUB -J heavy_job
#BSUB -n 8
#BSUB -W 2:00

python monitor.py &          # starts monitor in background
python heavy_computation.py  # main job
# script ends here — what happens to monitor.py?
```

What is the problem with this script?

**A)** `monitor.py` is killed immediately when the main script exits; monitoring data is lost  
**B)** The `&` operator is not supported in BSUB job scripts  
**C)** `monitor.py` keeps running after the main job ends, holding the job slot in RUN state until wall time is reached  
**D)** Both processes share stdout and output gets mixed, but there is no scheduling problem

**Answer: C**

- A) Incorrect — background processes are not automatically killed when the parent shell exits in LSF; they become orphans
- B) Incorrect — `&` is valid bash syntax and works in BSUB job scripts
- C) Correct — `monitor.py` runs as an orphan, keeping the LSF job in RUN state until the wall-clock limit is hit, wasting compute hours
- D) Incorrect — the scheduling problem (holding the job slot open) is the primary issue, not just mixed stdout

---

## Q4 — wait After kill for Clean Termination

The script from Q3 is fixed as follows:

```bash
#!/bin/bash
#BSUB -J heavy_job
#BSUB -n 8
#BSUB -W 2:00

python monitor.py &
MONITOR_PID=$!
python heavy_computation.py
kill $MONITOR_PID
wait $MONITOR_PID   # what does this line do?
```

What is the purpose of `wait $MONITOR_PID`?

**A)** It re-launches the monitor if it crashed during execution  
**B)** It blocks until the killed process has fully exited, ensuring clean termination before the script ends  
**C)** It sends a SIGTERM to `$MONITOR_PID` (equivalent to `kill`)  
**D)** It has no effect because `kill` already terminated the process

**Answer: B**

- A) Incorrect — `wait` only waits for a process to end; it does not restart crashed processes
- B) Correct — `kill` sends SIGTERM but returns immediately; `wait` blocks until the process actually exits, ensuring no orphan remains
- C) Incorrect — `wait` does not send any signal; only `kill` does
- D) Incorrect — `kill` sends the signal but the process may not have exited yet; `wait` ensures complete termination

---

## Q5 — Email Notification Count with -N

A job array is submitted with email notification:

```bash
#!/bin/bash
#BSUB -J process[1-50]
#BSUB -n 1
#BSUB -N
#BSUB -u user@dtu.dk
#BSUB -W 0:30

python process.py $LSB_JOBINDEX
```

How many emails will `user@dtu.dk` receive when all jobs complete?

**A)** 1 email summarising all 50 array elements  
**B)** 1 email per unique job state change (e.g., 2: PEND→RUN and RUN→DONE)  
**C)** 50 emails — one per array element on completion  
**D)** 0 emails — `-N` requires `-B` to be effective

**Answer: C**

- A) Incorrect — `-N` sends per-element notifications, not a single summary for the whole array
- B) Incorrect — `-N` triggers on job completion, not on every state transition
- C) Correct — `-N` sends one email per array element on completion, producing 50 separate emails
- D) Incorrect — `-N` is self-sufficient for completion notifications; `-B` is for job-start notifications

---

## Q6 — Pool() Uses os.cpu_count() Not LSF Cores

A job requests 4 cores and sets an environment variable, then launches a Python multiprocessing pool:

```bash
#!/bin/bash
#BSUB -J mp_job
#BSUB -n 4

OMP_NUM_THREADS=4
python script.py
```

```python
from multiprocessing import Pool

def work(x):
    return x ** 2

with Pool() as pool:   # no argument passed to Pool
    results = pool.map(work, range(1000))
```

The job runs on a 32-core node. How many worker processes does `Pool()` create, and is this a problem?

**A)** 4 workers — `Pool()` reads `$OMP_NUM_THREADS` automatically  
**B)** 4 workers — LSF limits the process count to `#BSUB -n`  
**C)** 32 workers — `Pool()` uses `os.cpu_count()` which returns all cores on the node, massively oversubscribing the 4 allocated cores  
**D)** 1 worker — `Pool()` defaults to serial execution on HPC nodes

**Answer: C**

- A) Incorrect — `Pool()` does not read `OMP_NUM_THREADS`; that variable affects OpenMP threads, not Python multiprocessing
- B) Incorrect — LSF does not constrain `os.cpu_count()` or `Pool()` worker count; it only allocates cores
- C) Correct — `Pool()` calls `os.cpu_count()` which sees all 32 node cores, creating 32 workers for only 4 allocated cores
- D) Incorrect — `Pool()` never defaults to serial execution; it always spawns multiple workers

---

## Q7 — Step Syntax [1-20:3] Index Values

A job array uses step syntax:

```bash
#!/bin/bash
#BSUB -J sim[1-20:3]
#BSUB -n 1

python simulate.py $LSB_JOBINDEX
```

Which values of `$LSB_JOBINDEX` will be scheduled?

**A)** 1, 2, 3, 6, 9, 12, 15, 18 (first 3 then every 3rd)  
**B)** 1, 4, 7, 10, 13, 16, 19 (start=1, step=3, all values ≤20)  
**C)** 3, 6, 9, 12, 15, 18 (multiples of 3 up to 20)  
**D)** 1, 20, 3 (only the values listed in the brackets)

**Answer: B**

- A) Incorrect — the syntax `[start-end:step]` does not mean "first N then every N-th"; it means start at 1 and add step each time
- B) Correct — starting at 1 with step 3 gives 1, 4, 7, 10, 13, 16, 19 (7 elements; next would be 22 > 20)
- C) Incorrect — multiples of 3 would require `[3-20:3]`; the range starts at 1
- D) Incorrect — the brackets define a range with step syntax, not an explicit list of values

---

## Q8 — done() Requires All DONE, any EXIT Blocks

A map-reduce HPC workflow uses job dependencies:

```bash
# map.sh
#!/bin/bash
#BSUB -J map[1-100]
#BSUB -n 1
#BSUB -W 0:30
python process.py $LSB_JOBINDEX
```

```bash
# reduce.sh
#!/bin/bash
#BSUB -J reduce
#BSUB -n 4
#BSUB -W 1:00
#BSUB -w "done(map)"
python aggregate.py
```

Both scripts are submitted. During the run, 3 of the 100 map jobs fail and enter EXIT state. When does `reduce` start?

**A)** After the 97 successful map jobs finish; 3 failures are tolerated  
**B)** Never — `done(map)` requires ALL 100 map jobs to reach DONE (success); any EXIT state permanently blocks the condition  
**C)** After all 100 jobs have left RUN state, regardless of exit code  
**D)** Immediately — `reduce` only depends on `map` being submitted, not completed

**Answer: B**

- A) Incorrect — `done()` requires every element to reach DONE; it has no partial-success tolerance
- B) Correct — any EXIT state causes `done(map)` to remain permanently false, leaving `reduce` stuck in PEND
- C) Incorrect — that describes `ended(map)`, not `done(map)`; `done()` strictly requires DONE status
- D) Incorrect — `done()` is a completion dependency, not a submission dependency

---

## Key Facts Summary

| Concept | Detail |
|---------|--------|
| `$LSB_JOBINDEX` is 1-based | Always subtract 1 when indexing Python lists: `idx = int(sys.argv[1]) - 1` |
| Per-element log files | Use `%J_%I.out` not `%J.out`; `%I` expands to `$LSB_JOBINDEX` |
| Orphan background processes | Use `kill $PID; wait $PID` to ensure clean termination |
| `Pool()` uses `os.cpu_count()` | Always pass explicit worker count matching `#BSUB -n` |
| `-N` on large arrays | Sends one email per element — avoid on arrays > ~10 jobs |
| `done()` vs `ended()` | `done()` requires all DONE; `ended()` allows EXIT |
| Step syntax `[1-20:3]` | Produces indices: 1, 4, 7, ... (start + n*step ≤ end) |

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets job arrays with $LSB_JOBINDEX, job dependencies (done vs ended), output file naming, and pipeline construction

---

## Q9 — Spotting the done() Blocker

> **Week reference:** Week 11

Consider this two-stage pipeline:

```bash
# stage1.sh
#!/bin/bash
#BSUB -J "stage1[1-20]"
#BSUB -n 1
#BSUB -W 0:30
python run_stage1.py $LSB_JOBINDEX
```

```bash
# stage2.sh
#!/bin/bash
#BSUB -J "stage2"
#BSUB -n 4
#BSUB -W 1:00
#BSUB -w "done(stage1)"
python aggregate.py
```

During the run, job `stage1[7]` fails with exit code 1. What is the state of `stage2`?

**A)** `stage2` starts after the other 19 elements complete; one failure is tolerated  
**B)** `stage2` remains in PEND forever because `done(stage1)` can never be satisfied  
**C)** `stage2` starts immediately since `stage1[7]` already finished (any exit code counts)  
**D)** LSF resubmits `stage1[7]` automatically until it succeeds

**Answer: B**

- A) Incorrect — `done()` has no partial-success tolerance; every element must reach DONE. One EXIT prevents the condition from ever being true.
- B) Correct — a single EXIT in the array permanently blocks `done(stage1)`. The fix is either to use `ended(stage1)` or to ensure all stage1 jobs succeed.
- C) Incorrect — that describes the behaviour of `ended()`, which accepts any terminal state including EXIT.
- D) Incorrect — LSF never automatically retries failed jobs; resubmission must be done manually or with a wrapper script.

---

## Q10 — Fixing Output File Collision

> **Week reference:** Week 11

A student submits this script:

```bash
#!/bin/bash
#BSUB -J "sim[1-8]"
#BSUB -n 1
#BSUB -o simulation_output.out
#BSUB -W 0:20

python simulate.py $LSB_JOBINDEX
```

All 8 elements run in parallel. What is the problem and what is the correct fix?

**A)** No problem — LSF queues writes so each element's output appears sequentially  
**B)** All 8 elements write to the same `simulation_output.out`, interleaving output; fix by using `#BSUB -o simulation_output_%J_%I.out`  
**C)** The `$LSB_JOBINDEX` in the Python call is invalid inside a BSUB script  
**D)** The output file must be named after the job (`#BSUB -o sim.out`) to avoid the collision

**Answer: B**

- A) Incorrect — LSF does not serialize file writes; all 8 processes write concurrently and output is interleaved without ordering guarantees.
- B) Correct — a fixed filename means all 8 elements race to write to the same file. Using `%J` (job ID) combined with `%I` (array index) creates a unique filename per element per submission.
- C) Incorrect — `$LSB_JOBINDEX` is a valid environment variable inside BSUB job scripts; it is set by LSF before the script body executes.
- D) Incorrect — renaming the output file to a fixed name (`sim.out`) still causes the same collision; the filename must contain `%I` to be unique per element.

---

## Q11 — Reading Array Index in Python

> **Week reference:** Week 11

A job array script passes `$LSB_JOBINDEX` as a command-line argument to Python. The Python script does this:

```python
import sys
import os

files = [f"data_{i:02d}.csv" for i in range(10)]  # data_00.csv ... data_09.csv
job_idx = int(sys.argv[1])
fname = files[job_idx - 1]
print(f"Processing {fname}")
```

The BSUB directive is `#BSUB -J "run[1-10]"`. What does job element 1 process?

**A)** `data_01.csv` (index 1, no subtraction needed)  
**B)** `data_00.csv` (correct: `job_idx=1`, `files[0]` = `data_00.csv`)  
**C)** `IndexError` — `files[0]` is out of range  
**D)** `data_09.csv` — Python reverses the list when indexing from LSF

**Answer: B**

- A) Incorrect — the script subtracts 1 (`job_idx - 1`), so `job_idx=1` gives `files[0]` which is `data_00.csv`, not `data_01.csv`.
- B) Correct — `$LSB_JOBINDEX` for element 1 is `1`; after `- 1` the Python index is `0`; `files[0]` is `"data_00.csv"`.
- C) Incorrect — `files[0]` is a valid access on a 10-element list; no error occurs.
- D) Incorrect — Python lists have no automatic reversal behaviour; they are always 0-indexed from the first element.

---

## Q12 — Concurrent Slot Limit Syntax

> **Week reference:** Week 11

A student wants to submit 200 simulation jobs but limit the cluster to running at most 5 at a time. Which BSUB directive achieves this?

```bash
#!/bin/bash
#BSUB -n 1
#BSUB -W 1:00
python simulate.py $LSB_JOBINDEX
```

Which `-J` line should replace the missing directive?

**A)** `#BSUB -J "sim[1-200:5]"`  
**B)** `#BSUB -J "sim[1-200]" -max 5`  
**C)** `#BSUB -J "sim[1-200]%5"`  
**D)** `#BSUB -J "sim[1-200]" && #BSUB -slots 5`

**Answer: C**

- A) Incorrect — `[1-200:5]` is step syntax; it creates 40 jobs with indices 1, 6, 11, ..., 196. It limits the count, not the concurrency.
- B) Incorrect — `-max` is not a valid LSF directive for controlling array concurrency.
- C) Correct — appending `%N` to the job array range sets the maximum concurrent running elements. `[1-200]%5` submits all 200 jobs but lets at most 5 run simultaneously.
- D) Incorrect — `&&` is a shell operator and cannot be used within a `#BSUB` directive. `#BSUB -slots` is not a valid LSF directive.

---

## Q13 — Three-Stage Pipeline With done() and ended()

> **Week reference:** Week 11

Examine this three-job pipeline submission sequence:

```bash
bsub -J "fetch[1-10]" fetch.sh
bsub -w "done(fetch)" -J "transform[1-10]" transform.sh
bsub -w "ended(transform)" -J "report" report.sh
```

During a run, 2 out of 10 `fetch` jobs fail. What is the final outcome?

**A)** `transform` starts (2 failures tolerated), `report` starts after all `transform` jobs
**B)** `transform` is permanently blocked; `report` never runs
**C)** `transform` is permanently blocked; but `report` still runs because it uses `ended()`
**D)** Both `transform` and `report` start immediately because `ended()` on `report` overrides the `done()` on `transform`

**Answer: B**

- A) Incorrect — `done(fetch)` requires all 10 fetch jobs to reach DONE. With 2 failures, the condition is never satisfied and `transform` never starts.
- B) Correct — `transform` is blocked permanently by the 2 failed `fetch` jobs. Because `transform` never starts (let alone finishes), `ended(transform)` also has nothing to wait for and `report` stays in PEND indefinitely too.
- C) Incorrect — `ended()` triggers when jobs finish, but `transform` jobs were never submitted to run; they sit in PEND. `ended()` on a job that is perpetually PEND is never satisfied.
- D) Incorrect — dependency conditions on `report` are independent of dependency conditions on `transform`; one cannot override the other. Both are evaluated by LSF independently.

---

## Q14 — Identifying the Correct Index in a File List

> **Week reference:** Week 11

A job array with `#BSUB -J "proc[1-5]"` processes a list of CSV files. The Python script receives the index as `sys.argv[1]`. Here are four candidate implementations:

```python
# Option A
files = ["a.csv", "b.csv", "c.csv", "d.csv", "e.csv"]
idx = int(sys.argv[1])
print(files[idx])

# Option B
files = ["a.csv", "b.csv", "c.csv", "d.csv", "e.csv"]
idx = int(sys.argv[1]) - 1
print(files[idx])

# Option C
files = ["a.csv", "b.csv", "c.csv", "d.csv", "e.csv"]
idx = int(sys.argv[1]) + 1
print(files[idx])

# Option D
files = ["a.csv", "b.csv", "c.csv", "d.csv", "e.csv"]
idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1)) - 1
print(files[idx])
```

Which option correctly processes one unique file per array element without error?

**A)** Option A  
**B)** Option B  
**C)** Option C  
**D)** Option D

**Answer: B**

- A) Incorrect — Option A does not subtract 1, so `$LSB_JOBINDEX=1` accesses `files[1]` (b.csv), skipping a.csv. `$LSB_JOBINDEX=5` accesses `files[5]` which is out of range, raising `IndexError`.
- B) Correct — subtracting 1 maps the 1-based LSF index to the 0-based Python list: index 1→files[0], ..., index 5→files[4]. All five files are accessed exactly once with no errors.
- C) Incorrect — adding 1 skips `files[0]` and `files[1]`; `$LSB_JOBINDEX=4` or `5` accesses `files[5]` or `files[6]`, both out of range.
- D) Incorrect — `SLURM_ARRAY_TASK_ID` is the SLURM variable and does not exist in DTU's LSF environment. `os.environ.get()` would return the default `1`, making every array element process `files[0]` (a.csv) instead of its unique file.

---

## Q15 — done() With Specific Job ID

> **Week reference:** Week 11

A user submits a preprocessing array and notes its job ID is 77777:

```bash
bsub -J "prep[1-30]" prep.sh
# LSF prints: Job <77777> is submitted to queue <hpc>.
```

They then submit an analysis job that should only run after ALL preprocessing jobs succeed. Which directive is correct?

**A)** `#BSUB -w "done(prep)"`  
**B)** `#BSUB -w "done(77777)"`  
**C)** `#BSUB -w "ended(prep[1-30])"`  
**D)** Both A and B are correct

**Answer: D**

- A) Correct on its own — `done(prep)` uses the job name and waits for all elements of the array named `prep` to reach DONE.
- B) Correct on its own — `done(77777)` uses the specific job ID and waits for all elements of job 77777 to reach DONE.
- C) Incorrect — `ended()` allows failures; the question requires success-only completion. Also, `[1-30]` subscript in dependency syntax is not standard LSF.
- D) Correct — both A and B are valid ways to express the same dependency in LSF; one uses the job name, the other uses the numeric ID.

---

## Q16 — Diagnosing a Stuck Pipeline

> **Week reference:** Week 11

A user runs `bjobs` and sees:

```
JOBID   USER    STAT  QUEUE    JOB_NAME
88001   user1   DONE  hpc      map[1-50]
88002   user1   PEND  hpc      reduce
```

The `reduce` job was submitted with `#BSUB -w "done(map)"`. Job 88001 shows DONE. Why might `reduce` still be in PEND?

**A)** `reduce` is waiting for resources (cores/memory) to become available on the cluster  
**B)** `bjobs` showing the array as DONE does not mean all individual elements are DONE; one or more elements may be in EXIT state  
**C)** `done(map)` requires the job to be submitted with the same queue as `map`  
**D)** The `reduce` job cannot start until the output files from `map` are closed by the OS

**Answer: B**

- A) Possibly true in general, but the question asks specifically why `done(map)` might not be satisfied. Resource availability would show a different status message and does not relate to the dependency.
- B) Correct — `bjobs` sometimes shows the parent array as DONE even when individual elements are in EXIT. `done()` checks each element individually; any EXIT element blocks the condition. Running `bjobs -A 88001` would reveal per-element status.
- C) Incorrect — `done()` dependencies are not queue-scoped; they match by job name or ID regardless of queue.
- D) Incorrect — LSF dependency conditions are based on job status codes, not filesystem operations. File handle state is irrelevant to when `done()` is satisfied.

---

## Q17 — Constructing the Output Filename

> **Week reference:** Week 11

A job array is submitted as follows:

```bash
#!/bin/bash
#BSUB -J "run[1-25]"
#BSUB -o /scratch/results/out_%J_%I.txt
#BSUB -e /scratch/results/err_%J_%I.txt
#BSUB -n 1
python worker.py $LSB_JOBINDEX
```

The job array receives submission ID 55000. What is the stdout output filename for array element 12?

**A)** `/scratch/results/out_55000.txt`  
**B)** `/scratch/results/out_12.txt`  
**C)** `/scratch/results/out_55000_12.txt`  
**D)** `/scratch/results/out_12_55000.txt`

**Answer: C**

- A) Incorrect — this would be the result of `out_%J.txt` (only `%J`, no `%I`). The `%I` placeholder is also present in the directive, so both are expanded.
- B) Incorrect — this would be the result of `out_%I.txt` (only `%I`, no `%J`). The full directive includes both placeholders.
- C) Correct — `%J` expands to the submission ID (55000) and `%I` expands to the array index (12), in the order they appear in the directive: `out_55000_12.txt`.
- D) Incorrect — the directive is `out_%J_%I.txt`, so `%J` appears before `%I`. Reversing them would require the directive `out_%I_%J.txt`.

---

## Q18 — Combining Array Index With Python Logic

> **Week reference:** Week 11

A job array with `#BSUB -J "batch[1-4]"` processes chunks of a 1000-row dataset, assigning 250 rows to each element. The Python script must compute its start and end row indices:

```python
import os

chunk_size = 250
idx = int(os.environ["LSB_JOBINDEX"])   # 1-based: 1, 2, 3, 4

start = ???
end   = ???
rows  = data[start:end]
```

Which assignment correctly gives each element a non-overlapping 250-row chunk?

**A)** `start = idx * chunk_size; end = start + chunk_size`  
**B)** `start = (idx - 1) * chunk_size; end = start + chunk_size`  
**C)** `start = idx * chunk_size - 1; end = start + chunk_size`  
**D)** `start = (idx - 1) * chunk_size + 1; end = start + chunk_size`

**Answer: B**

- A) Incorrect — `idx=1` gives `start=250`, skipping the first 250 rows entirely. `idx=4` gives `start=1000`, which is out of range for a 1000-row dataset.
- B) Correct — subtracting 1 converts the 1-based index to 0-based chunk numbering. `idx=1` → `start=0, end=250`; `idx=2` → `start=250, end=500`; ...; `idx=4` → `start=750, end=1000`. All 1000 rows are covered exactly once.
- C) Incorrect — `idx=1` gives `start=249`, which is inside the first chunk and causes overlap with the first element.
- D) Incorrect — `idx=1` gives `start=1`, skipping row 0. Each chunk is also offset by 1, introducing gaps between chunks.

---
