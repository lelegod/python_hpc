# Week 11 — HPC Workflows Syntax Reference

## Job Array Syntax

```bash
# Basic range: indices 1,2,...,10
#BSUB -J name[1-10]

# Step: indices 1,3,5,...,99
#BSUB -J name[1-100:2]

# Explicit list
#BSUB -J name[1,4,7]

# Mixed
#BSUB -J name[1,4,10-14,20-30:2]

# Per-element output files (ALWAYS do this for arrays)
#BSUB -o name_%J_%I.out      # %J = job ID, %I = array index
#BSUB -e name_%J_%I.err
```

**$LSB_JOBINDEX is 1-based.** Convert in Python:
```python
idx = int(sys.argv[1]) - 1    # 0-based list index
files = sorted(os.listdir('/data/'))
this_file = files[idx]
```

---

## Job Dependency Syntax

```bash
# Wait for ALL jobs named "prepare" to reach DONE (success only)
#BSUB -w "done(prepare)"

# Wait for ALL jobs to finish regardless of success/failure
#BSUB -w "ended(prepare)"

# Wait for specific array element
#BSUB -w "done(array[3])"

# Element-wise: new[i] waits for existing[i]
#BSUB -w "done(array[*])"
```

**`done()` vs `ended()`:**
- `done()`: requires ALL jobs to reach DONE (success). ONE failure → dependent job NEVER starts.
- `ended()`: any termination (DONE or EXIT). Use for cleanup jobs.

---

## Monitoring Commands

```bash
bjobs -A                    # array summary (state counts per array)
bjobs -A -J name            # filter by job name
bpeek JOBID[3]              # live stdout of array element 3
bkill JOBID                 # kill whole array
bkill JOBID[1-5]            # kill elements 1-5
watch -n60 bstat            # poll every 60 seconds (reasonable)
```

**NEVER:** `watch bstat` (every 2s) or `watch -n0.1 bstat` (insane — spams scheduler).

---

## Map-Reduce Pattern

```bash
# map.sh — parallel processing
#!/bin/bash
#BSUB -J map[1-100]
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 00:15
#BSUB -R "rusage[mem=1GB]"
#BSUB -o batch_output/map_%J_%I.out
python process.py $LSB_JOBINDEX

# reduce.sh — aggregation after all map jobs succeed
#!/bin/bash
#BSUB -J reduce
#BSUB -w "done(map)"        # waits for ALL 100 map jobs to reach DONE
python aggregate.py
```

---

## Python Script for Array Jobs

```python
import sys
import os
import numpy as np

def main():
    idx = int(sys.argv[1]) - 1    # 1-based → 0-based
    all_inputs = sorted(os.listdir('/path/to/data'))
    this_input = all_inputs[idx]
    result = process(this_input)
    np.save(f'result_{idx}.npy', result)

if __name__ == '__main__':
    main()
```

---

## Pitfalls

```bash
# DON'T: email with large arrays (generates one email per element)
#BSUB -N
#BSUB -J name[1-200]    # sends 200 emails!

# DO: omit -N for arrays

# DON'T: orphan background processes
python monitor.py &
python main.py
# job stuck until wall time!

# DO: kill background process after main script
python monitor.py &
MONITOR_PID=$!
python main.py
kill $MONITOR_PID
wait $MONITOR_PID
```

---

## Exam Traps

| Trap | Correct |
|---|---|
| `$LSB_JOBINDEX` is 0-based | It's 1-based — subtract 1 in script |
| `-o name_%J.out` for array | Use `name_%J_%I.out` for per-element files |
| `-w done(job)` when any exit is fine | Use `-w ended(job)` |
| ONE job in EXIT state with `done()` | Dependent job NEVER starts |
| Pool() uses LSF -n cores | Uses os.cpu_count() — set thread env vars |
