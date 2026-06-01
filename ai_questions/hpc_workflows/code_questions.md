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
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q19 — Explicit Index List: Which Elements Run?](#q19--explicit-index-list-which-elements-run)
- [Q20 — Per-Core Memory Trap With -n 4](#q20--per-core-memory-trap-with--n-4)
- [Q21 — Fan-Out Script: Correct Output Filename](#q21--fan-out-script-correct-output-filename)
- [Q22 — done(subhist) Without Quotes: Valid or Not?](#q22--donesubhist-without-quotes-valid-or-not)
- [Q23 — LSB_DJOB_NUMPROC Fix in Pool()](#q23--lsb_djob_numproc-fix-in-pool)
- [Q24 — Element-Level Dependency Script Trace](#q24--element-level-dependency-script-trace)
- [Q25 — Fan-In: Loading All Partial Results](#q25--fan-in-loading-all-partial-results)
- [Q26 — Reading Index With Fallback Default](#q26--reading-index-with-fallback-default)
- [Q27 — ended() With Numeric ID: When Does It Trigger?](#q27--ended-with-numeric-id-when-does-it-trigger)
- [Q28 — Diagnosing Oversubscription From Pool() Output](#q28--diagnosing-oversubscription-from-pool-output)

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

## Set 3 — Extended Practice

> Targets explicit index lists, per-core memory, fan-out/fan-in file naming, LSB_DJOB_NUMPROC, element-level dependencies, glob-based reduce, and ended() with numeric IDs.

---

## Q19 — Explicit Index List: Which Elements Run?

> **Week reference:** Week 11

A developer submits this job array to rerun failed elements:

```bash
#!/bin/bash
#BSUB -J "proc[2,29,71,73,127]"
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -W 00:10
#BSUB -o proc_%J_%I.out

python process.py $LSB_JOBINDEX
```

Which values does `$LSB_JOBINDEX` take across all elements, and how many elements are created?

**A)** 1, 2, 3, 4, 5 — LSF ignores the explicit list and uses sequential 1-based indices; 5 elements  
**B)** 2, 29, 71, 73, 127 — LSF creates exactly those indices; 5 elements  
**C)** 2, 3, 4, ..., 127 — LSF expands the list to the full range; 126 elements  
**D)** 2, 29, 71, 73, 127 — LSF creates those indices but `$LSB_JOBINDEX` is set to 1, 2, 3, 4, 5 internally; 5 elements

**Answer: B**

- A) Incorrect — LSF does not substitute sequential indices for an explicit list. When a comma-separated list is given, `$LSB_JOBINDEX` takes exactly the listed values, preserving their meaning for the script.
- B) Correct — a comma-separated index list creates one element per listed value, with `$LSB_JOBINDEX` set to the exact value for each. The script receives `2`, `29`, `71`, `73`, or `127` directly without any remapping.
- C) Incorrect — LSF does not expand a comma-separated list into a range. Only the five listed values are created; indices in between (3, 4, ..., 28, 30, ...) are not scheduled.
- D) Incorrect — LSF does not renumber the indices internally. The whole point of using an explicit list is that `$LSB_JOBINDEX` matches the listed values exactly, so `process.py` receives the real data indices without needing a separate lookup table.

---

## Q20 — Per-Core Memory Trap With -n 4

> **Week reference:** Week 11

A student writes this job script for a simulation that needs 100 GB RAM total:

```bash
#!/bin/bash
#BSUB -J simulate
#BSUB -q hpc
#BSUB -W 10:00
#BSUB -R "rusage[mem=100GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o sim_%J.out

python simulate.py initconds.npy
```

How much total memory does LSF reserve for this job, and is there a problem?

**A)** 100 GB total — `rusage[mem=100GB]` is the total job memory; no problem  
**B)** 400 GB total — `rusage[mem=X]` is per core; with 4 cores the job reserves 400 GB, massively over-requesting  
**C)** 25 GB total — LSF divides the requested memory by the core count automatically  
**D)** 100 GB total — LSF only allocates what the process actually uses, so the directive is a ceiling, not a reservation

**Answer: B**

- A) Incorrect — `rusage[mem=X]` is a per-slot (per-core) value in LSF. With 4 cores, the total reservation is 4 × 100 GB = 400 GB, not 100 GB.
- B) Correct — the directive `rusage[mem=100GB]` requests 100 GB per core. With `#BSUB -n 4`, LSF reserves 400 GB total. The script may run fine if the node has that much RAM, but it wastes 300 GB of allocation. The correct value to achieve 100 GB total is `rusage[mem=25GB]`.
- C) Incorrect — LSF does not auto-divide your memory request by the core count. It multiplies the per-core request by the number of cores. The student must do the division manually.
- D) Incorrect — `rusage[mem=X]` is a hard reservation in LSF, not a soft ceiling. The scheduler uses it to decide whether a node has enough free memory to accept the job. The actual process may use less, but the full amount is reserved and unavailable to other jobs.

---

## Q21 — Fan-Out Script: Correct Output Filename

> **Week reference:** Week 11

The solution script for the CelebA hue histogram exercise saves partial results like this:

```python
import os
import sys
import numpy as np

if __name__ == '__main__':
    idx = int(sys.argv[1]) - 1   # 0-based index
    # ... load images, compute hist ...
    np.save(f"subhist_{idx}.npy", hist)
```

The job array directive is `#BSUB -J "subhist[1-203]"`. What filename does element 5 save to?

**A)** `subhist_5.npy`  
**B)** `subhist_4.npy`  
**C)** `subhist_05.npy`  
**D)** `subhist_1.npy` — all elements produce the same filename because `idx` is always 1 after subtraction

**Answer: B**

- A) Incorrect — the script subtracts 1 from the LSF index before saving. Element 5 receives `sys.argv[1] = "5"`, so `idx = 5 - 1 = 4`. The filename is `subhist_4.npy`, not `subhist_5.npy`.
- B) Correct — `$LSB_JOBINDEX` for element 5 is `5`. After `int(sys.argv[1]) - 1`, `idx = 4`. The f-string produces `subhist_4.npy`.
- C) Incorrect — the f-string `f"subhist_{idx}.npy"` does not zero-pad integers; it would produce `4`, not `04`. Zero-padding would require `f"subhist_{idx:02d}.npy"`.
- D) Incorrect — `idx` is computed from `sys.argv[1]`, which is the `$LSB_JOBINDEX` value passed from the job script. Each element receives a different index value, so `idx` is different for every element. Only if `sys.argv[1]` were hard-coded would all elements produce the same filename.

---

## Q22 — done(subhist) Without Quotes: Valid or Not?

> **Week reference:** Week 11

A student writes the fan-in (reduce) job script as follows:

```bash
#!/bin/bash
#BSUB -J plothist
#BSUB -q hpc
#BSUB -W 5
#BSUB -n 1
#BSUB -w done(subhist)
#BSUB -R "span[hosts=1]"
#BSUB -o plothist_%J.out

python plothuehist.py
```

The fan-out array was submitted with `#BSUB -J "subhist[1-203]"`. Is the dependency directive correct, and will the job behave as expected?

**A)** No — `done(subhist)` must be quoted: `#BSUB -w "done(subhist)"`, otherwise parentheses are interpreted by the shell  
**B)** Yes — `done(subhist)` without quotes is valid in a BSUB script header because BSUB directives are not parsed by the shell  
**C)** No — the directive must reference the full array: `#BSUB -w "done(subhist[1-203])"` to wait for all elements  
**D)** No — `done()` cannot reference a job array by name; it requires the numeric job ID

**Answer: B**

- A) Incorrect — `#BSUB` directives in a batch script are read by LSF directly from the file, not executed as shell commands. Parentheses in a `#BSUB` directive are not subject to shell expansion or interpretation. Both quoted and unquoted forms work in a script file.
- B) Correct — inside a batch script, `#BSUB` lines are parsed by LSF, not by bash. The parentheses in `done(subhist)` are part of the LSF dependency syntax and are passed directly to the scheduler. Quotes are optional in script files (though conventional for clarity). This is exactly the pattern shown in the official week 11 solution.
- C) Incorrect — `done(subhist)` without a subscript already waits for ALL elements of the named array to reach DONE. Adding `[1-203]` inside the dependency expression is not standard LSF syntax and would likely cause an error.
- D) Incorrect — `done()` accepts both job names and numeric job IDs. Using the job name `subhist` is standard practice and often more readable than using a numeric ID that changes each submission.

---

## Q23 — LSB_DJOB_NUMPROC Fix in Pool()

> **Week reference:** Week 11

A job requests 8 cores. The original buggy script uses `Pool()` with no argument. The fix reads the LSF allocation:

```bash
#!/bin/bash
#BSUB -J mp_job
#BSUB -n 8
#BSUB -R "span[hosts=1]"

python script.py
```

```python
import os
from multiprocessing import Pool

n_workers = int(os.environ.get("LSB_DJOB_NUMPROC", os.cpu_count()))

def work(x):
    return x ** 2

with Pool(processes=n_workers) as pool:
    results = pool.map(work, range(10000))
```

The job runs on a 32-core node. How many worker processes does `Pool()` create?

**A)** 32 — `LSB_DJOB_NUMPROC` is not set in the environment so the fallback `os.cpu_count()` runs  
**B)** 8 — `LSB_DJOB_NUMPROC` is set by LSF to the allocated core count (8), which is used as `n_workers`  
**C)** 1 — `os.environ.get()` with a default returns `None` when the variable is missing, and `int(None)` gives 1  
**D)** 8 — but only by coincidence; `LSB_DJOB_NUMPROC` is not a real LSF variable

**Answer: B**

- A) Incorrect — `LSB_DJOB_NUMPROC` is a genuine LSF environment variable that LSF sets in the job's environment to the number of allocated CPU slots. On a job submitted with `#BSUB -n 8`, it is set to `"8"`. The fallback is not reached.
- B) Correct — LSF sets `LSB_DJOB_NUMPROC=8` in the job environment. `os.environ.get("LSB_DJOB_NUMPROC", os.cpu_count())` returns `"8"`, and `int("8")` gives `n_workers=8`. The pool is created with exactly the allocated count, eliminating oversubscription.
- C) Incorrect — `os.environ.get(key, default)` returns the `default` argument (not `None`) when the key is absent. In this case the default is `os.cpu_count()`, not `None`, so `int(None)` is never called.
- D) Incorrect — `LSB_DJOB_NUMPROC` is a real, documented LSF environment variable. It is not coincidental; LSF explicitly sets it to match the `-n` allocation, making it the canonical way to read the allocated core count from Python.

---

## Q24 — Element-Level Dependency Script Trace

> **Week reference:** Week 11

Two arrays are submitted:

```bash
# First submission (already running):
# Job ID 21241475, array name "array", elements [1-5]

# Second submission:
#!/bin/bash
#BSUB -J "jobname[1-5]"
#BSUB -q hpc
#BSUB -W 5
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -w "done(array[*])"
#BSUB -o jobname_%J.out
```

Element 3 of `array` finishes with exit code 0 (DONE). Elements 1, 2, 4, 5 are still running. What is the state of element 3 of `jobname`?

**A)** Element 3 of `jobname` immediately moves from PEND to RUN, because its matching `array[3]` is DONE  
**B)** All 5 elements of `jobname` remain in PEND until ALL elements of `array` are DONE  
**C)** Element 3 of `jobname` moves to RUN; the other 4 stay in PEND until their matching elements finish  
**D)** Element 3 of `jobname` moves to RUN; the other 4 are cancelled because `done(array[*])` is partially satisfied

**Answer: C**

- A) Incorrect — `done(array[*])` with the `[*]` wildcard is an element-wise dependency, not an all-or-nothing condition. But element 3 of `jobname` waits for `array[3]` specifically, not for all of `array`. A is partially right but misses that the other elements also proceed element-wise, not all at once.
- B) Incorrect — that would be the behaviour of `done(array)` without any subscript. The `[*]` wildcard changes the semantics to element-level chaining: each element waits only for its own counterpart.
- C) Correct — `done(array[*])` creates a per-element dependency. When `array[3]` reaches DONE, `jobname[3]` can start. Elements 1, 2, 4, 5 of `jobname` remain in PEND until their respective counterparts in `array` reach DONE.
- D) Incorrect — LSF never cancels array elements because a sibling's dependency was satisfied. Each element is independent; satisfying one element's dependency has no effect on the scheduling state of the other elements.

---

## Q25 — Fan-In: Loading All Partial Results

> **Week reference:** Week 11

The reduce step of the CelebA hue histogram exercise looks like this:

```python
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    histfiles = glob('subhist_*.npy')
    hist = 0
    for hf in histfiles:
        hist += np.load(hf)
    plt.bar(np.linspace(0, 255, 64), hist, width=4)
    plt.savefig('histogram.png')
```

The fan-out array saved files named `subhist_0.npy` through `subhist_202.npy`. What does `hist` equal after the loop if all 203 files are found?

**A)** A list of 203 NumPy arrays, one per file  
**B)** The element-wise sum of all 203 histogram arrays — a single NumPy array of shape (64,)  
**C)** The integer 0, because `hist = 0` is not overwritten when adding NumPy arrays  
**D)** The last loaded histogram array, because each `+=` overwrites the previous value

**Answer: B**

- A) Incorrect — `+=` on a NumPy array performs element-wise addition in-place, not list appending. `hist` is never converted to a list.
- B) Correct — starting with `hist = 0` and then doing `hist += np.load(hf)` on the first iteration uses NumPy broadcasting: `0 + array` produces a new array. Subsequent iterations add element-wise into that array. The result is the element-wise sum of all 203 arrays, giving a single shape-(64,) array representing the combined histogram across the entire dataset.
- C) Incorrect — Python's `+=` operator on a NumPy array does not no-op when the left side starts as an integer. The first `hist += np.load(hf)` evaluates `0 + array`, which returns a NumPy array. From that point, `hist` is a NumPy array and subsequent `+=` calls accumulate correctly.
- D) Incorrect — `+=` is accumulation, not overwrite. Each iteration adds the new histogram values into `hist` without discarding previous contributions. Overwrite would require `hist = np.load(hf)` (plain assignment).

---

## Q26 — Reading Index With Fallback Default

> **Week reference:** Week 11

A Python script is designed to run both inside an LSF job array and interactively during development. It uses a fallback:

```python
import os

idx = int(os.environ.get("LSB_JOBINDEX", "1"))
folders = sorted(os.listdir("/data/images"))
target = folders[idx - 1]
print(f"Processing folder: {target}")
```

A developer runs this script interactively on their laptop (no LSF environment). There are 10 folders. What does the script print?

**A)** A `KeyError` — `LSB_JOBINDEX` does not exist in the environment and `get()` raises an exception  
**B)** `Processing folder: <second folder>` — the default `"1"` gives `idx=1`, and `idx - 1 = 0` selects the first folder, printing the first folder name  
**C)** `Processing folder: <first folder>` — with `idx=1` and `idx-1=0`, `folders[0]` is the first (alphabetically sorted) folder  
**D)** `Processing folder: <last folder>` — `os.environ.get()` returns `None` when the key is missing, and `int(None)` wraps to the last index

**Answer: C**

- A) Incorrect — `os.environ.get(key, default)` never raises `KeyError`. When the key is absent, it returns the `default` argument (`"1"` here). `KeyError` would only occur with `os.environ[key]` (square bracket access).
- B) Incorrect — the answer correctly computes `idx=1` and `idx-1=0`, then selects `folders[0]`. The label says "second folder" but the index 0 is the first element. This option has the right computation but the wrong description.
- C) Correct — `LSB_JOBINDEX` is absent, so `get()` returns `"1"`. `int("1") = 1`. `idx - 1 = 0`. `folders[0]` is the first element in the sorted list. The script prints the name of the first folder alphabetically.
- D) Incorrect — `os.environ.get(key, "1")` returns the string `"1"`, not `None`. `int("1")` is `1`, not any wrap-around value. The last folder would require `idx = len(folders)`.

---

## Q27 — ended() With Numeric ID: When Does It Trigger?

> **Week reference:** Week 11

A cleanup job depends on a previously submitted array (job ID 21241475):

```bash
#!/bin/bash
#BSUB -J cleanup
#BSUB -q hpc
#BSUB -W 00:10
#BSUB -n 1
#BSUB -w "ended(21241475)"
#BSUB -o cleanup_%J.out

python cleanup.py
```

During the run, 3 out of 5 elements of job 21241475 reach DONE and 2 reach EXIT. When does `cleanup` start?

**A)** Only after all 5 elements reach DONE (success required)  
**B)** After the first element finishes in any state (DONE or EXIT)  
**C)** After all 5 elements have left the RUN state in any terminal state (DONE or EXIT)  
**D)** Never — `ended()` with a numeric ID is not valid LSF syntax; only job names are supported

**Answer: C**

- A) Incorrect — that describes `done(21241475)`. `done()` requires all elements to reach DONE. `ended()` is satisfied by any terminal state including EXIT.
- B) Incorrect — `ended()` waits for all specified jobs to end, not just the first one. A single element finishing does not satisfy the condition; all must reach a terminal state.
- C) Correct — `ended(21241475)` is satisfied when every element of job array 21241475 has left the RUN state in any terminal condition (DONE, EXIT, or even ZOMBI/UNKWN resolved). With 3 DONE and 2 EXIT, all 5 elements are terminal, so `ended()` becomes true and `cleanup` can start.
- D) Incorrect — `ended()` (and `done()`) both accept numeric job IDs as well as job names. Using the numeric ID `21241475` is valid LSF dependency syntax, as shown in the course's `job_dependencies_2.sh` example file.

---

## Q28 — Diagnosing Oversubscription From Pool() Output

> **Week reference:** Week 11

A job is submitted with `#BSUB -n 4` on a 16-core node. The script contains:

```python
import os
from multiprocessing import Pool

print(f"LSF allocation: {os.environ.get('LSB_DJOB_NUMPROC', 'NOT SET')}")
print(f"os.cpu_count(): {os.cpu_count()}")

with Pool() as pool:
    print(f"Pool workers: {pool._processes}")
    results = pool.map(lambda x: x**2, range(100))
```

The job prints:

```
LSF allocation: 4
os.cpu_count(): 16
Pool workers: 16
```

What is the problem and what single-line fix corrects it?

**A)** No problem — `Pool()` uses 16 workers but LSF caps CPU usage to 4 cores automatically  
**B)** The pool spawns 16 workers on a 4-core allocation. Fix: change `Pool()` to `Pool(processes=int(os.environ["LSB_DJOB_NUMPROC"]))`  
**C)** The pool spawns 16 workers. Fix: set `OMP_NUM_THREADS=4` in the job script before calling Python  
**D)** The pool spawns 16 workers. Fix: change `Pool()` to `Pool(processes=4)` hardcoded

**Answer: B**

- A) Incorrect — LSF does not automatically cap CPU usage to the allocated core count. It only reserves the allocation in the scheduler; actual CPU usage enforcement depends on cgroup configuration, and processes can still run on more cores than allocated, causing oversubscription.
- B) Correct — `Pool()` with no argument calls `os.cpu_count()` (16), creating 16 workers for a 4-core job. The fix reads the actual LSF allocation via `LSB_DJOB_NUMPROC` at runtime, making the pool size always match the allocation regardless of node hardware.
- C) Incorrect — `OMP_NUM_THREADS` controls OpenMP thread count and has no effect on Python's `multiprocessing.Pool` worker count. `Pool()` does not read `OMP_NUM_THREADS`.
- D) Incorrect — hard-coding `Pool(processes=4)` fixes this specific job but is brittle. If `#BSUB -n` is later changed to 8, the hard-coded 4 causes underutilization. Reading `LSB_DJOB_NUMPROC` dynamically is the robust solution.

---
