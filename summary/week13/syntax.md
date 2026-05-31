# Week 13 — HPC Pitfalls Syntax Reference

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Thread Count Environment Variables](#thread-count-environment-variables)
- [Fast I/O Redirection](#fast-io-redirection)
- [Kill Background Processes](#kill-background-processes)
- [Polling the Scheduler](#polling-the-scheduler)
- [File Organisation](#file-organisation)
- [ThreadPool vs Pool for NumPy](#threadpool-vs-pool-for-numpy)
- [Good Job Script Template](#good-job-script-template)
- [Exam Traps](#exam-traps)

---

## Thread Count Environment Variables

**Always set these to match `#BSUB -n` — packages like NumPy use BLAS which ignores LSF allocation and defaults to all hardware cores.**

```bash
# In job script
#BSUB -n 8

NUM_THREADS=8
OMP_NUM_THREADS=$NUM_THREADS
MKL_NUM_THREADS=$NUM_THREADS
OPENBLAS_NUM_THREADS=$NUM_THREADS
MPI_NUM_THREADS=$NUM_THREADS

python script.py
```

**Oversubscription:** if your script uses ThreadPool(8) AND NumPy uses 8 threads internally → 64 threads on 8 cores → slower than either alone.

**Fix:** disable NumPy threading when using your own pool:
```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
python script.py   # now Pool(8) workers each run single-threaded NumPy
```

---

## Fast I/O Redirection

LSF's `-o`/`-e` channels are slow for high-volume output (up to 26× slower than manual redirect).

```bash
# Slow (LSF channel)
#BSUB -o output_%J.out
python -u script.py

# Fast (manual redirect to /work3)
#BSUB -o /work3/02613/dump/lsf_%J.out    # keep for LSF summary only
python -u script.py \
    1> /work3/02613/dump/output_${LSB_JOBID}.txt \
    2> /work3/02613/dump/error_${LSB_JOBID}.txt
```

**Never use `tee`** — it duplicates all output (file + stdout → LSF captures stdout = 2× I/O).

---

## Kill Background Processes

```bash
# WRONG — monitor runs forever after main script ends
python monitor.py &
python main.py
# job stays alive until wall time!

# CORRECT
python monitor.py &
MONITOR_PID=$!           # capture PID immediately after &
python main.py
kill $MONITOR_PID
wait $MONITOR_PID        # wait for kill to complete
```

---

## Polling the Scheduler

```bash
# WRONG — spams scheduler for everyone
watch bstat              # every 2 seconds
watch -n0.1 bstat        # every 0.1 seconds (insane)

# CORRECT
watch -n60 bstat         # once per minute
```

---

## File Organisation

```bash
# BAD: 100,000 files in one directory
ls /data/images/          # takes 2.6 seconds; may overflow shell buffer

# GOOD: organise into subfolders (~1000 files each)
/data/images/000/
/data/images/001/
...

# For Zarr: use nested storage to avoid file explosion
store = zarr.storage.NestedDirectoryStore('data.zarr')
z = zarr.open(store, ...)
```

---

## ThreadPool vs Pool for NumPy

```python
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import numpy as np

# ThreadPool works for NumPy — NumPy releases GIL during BLAS calls
# Avoids copying large arrays (shared memory)
with ThreadPool(8) as pool:
    C = np.concatenate(pool.starmap(np.matmul, zip(A, B)))

# Pool works too but copies arrays to each worker (expensive for large arrays)
with Pool(8) as pool:
    C = np.concatenate(pool.starmap(np.matmul, zip(A, B)))
```

**Why ThreadPool is preferred here:** NumPy's BLAS calls release the GIL → true parallelism without the copy overhead of ProcessPool.

---

## Good Job Script Template

```bash
#!/bin/bash
#BSUB -J myjob
#BSUB -q hpc
#BSUB -W 00:30
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o /work3/02613/dump/myjob_%J.out
#BSUB -e /work3/02613/dump/myjob_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

NUM_THREADS=8
OMP_NUM_THREADS=$NUM_THREADS
MKL_NUM_THREADS=$NUM_THREADS
OPENBLAS_NUM_THREADS=$NUM_THREADS

python -u script.py \
    1> /work3/02613/dump/output_${LSB_JOBID}.txt \
    2> /work3/02613/dump/error_${LSB_JOBID}.txt
```

---

## Exam Traps

| Trap | Correct |
|---|---|
| `Pool()` uses LSF -n cores | Uses `os.cpu_count()` — always set env vars |
| ThreadPool + multi-threaded NumPy = 2× speedup | Results in oversubscription — can be slower |
| LSF `-o` channel is fast for large output | 26× slower than manual redirect |
| `tee` is fine in batch jobs | Duplicates all I/O — never use |
| Background `&` process ends with main script | It keeps running — must `kill $PID` |
