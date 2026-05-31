# HPC Workflows — Code-Based MCQ Practice

> Format: Each question shows a job array script or Python HPC code with patterns to analyse.
> Exam frequency: **2024 exam + F25**.

---

## Question 1

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

## Question 2

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

## Question 3

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

## Question 4

The script from Question 3 is fixed as follows:

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

## Question 5

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

## Question 6

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

## Question 7

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

## Question 8

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
