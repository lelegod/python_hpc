# Week 1 — LSF / BSUB Syntax Reference

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [BSUB Directives (inside job script)](#bsub-directives-inside-job-script)
- [LSF Commands](#lsf-commands)
- [Inside Job Script](#inside-job-script)
- [Exam Traps](#exam-traps)

---

## BSUB Directives (inside job script)

```bash
#BSUB -J jobname              # job name shown in bjobs/bstat
#BSUB -q hpc                  # queue: hpc (CPU) | gpuv100 | gpua100 | c02613 (GPU)
#BSUB -n 4                    # number of CPU cores
#BSUB -W 01:30                # wall time HH:MM (or just minutes: -W 90)
#BSUB -R "span[hosts=1]"      # keep all cores on ONE node (required for multiprocessing)
#BSUB -R "rusage[mem=4GB]"    # memory PER CORE — total = 4 * n
#BSUB -R "select[model==XeonGold6226R]"  # pin to specific CPU model
#BSUB -gpu "num=1:mode=exclusive_process"  # GPU jobs only
#BSUB -o output_%J.out        # stdout file (%J = job ID)
#BSUB -e error_%J.err         # stderr file
#BSUB -o output_%J_%I.out     # per-array-element output (%I = array index)
#BSUB -J name[1-10]           # job array with indices 1..10
#BSUB -J name[1-100:2]        # step syntax: 1,3,5,...,99
#BSUB -w "done(jobname)"      # depend on ALL jobs named jobname finishing successfully
#BSUB -w "ended(jobname)"     # depend on ALL jobs named jobname finishing (any exit)
#BSUB -u user@dtu.dk          # email address
#BSUB -N                      # email on job end (DO NOT use with large arrays!)
```

**CRITICAL TRAP:** `rusage[mem=XGB]` is per core, NOT total.
- Program needs 16 GB, using 8 cores → `rusage[mem=2GB]`

---

## LSF Commands

```bash
bsub < submit.sh              # submit job
bstat                         # your jobs (compact)
bjobs                         # your jobs (detailed, shows TIME_LEFT)
bjobs -A                      # array summary (one row per array)
bjobs -p                      # why is job PEND?
bjobs -l JOBID                # full job details
bpeek JOBID                   # live stdout of running job
bpeek JOBID[3]                # live stdout of array element 3
bkill JOBID                   # kill a job
bkill 0                       # kill ALL your jobs
bkill JOBID[1-5]              # kill array elements 1-5
nodestat -f hpc               # see available nodes and CPU models
bhist                         # job history
linuxsh                       # interactive shell on compute node
```

**Job states:** `PEND → RUN → DONE` (success) or `EXIT` (failure/killed)

---

## Inside Job Script

```bash
# Access array index (1-based!)
IDX=${LSB_JOBINDEX}
python script.py ${IDX}       # script converts: idx = int(sys.argv[1]) - 1

# Access job ID
echo $LSB_JOBID

# Initialize conda
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Fast output redirect (26x faster than LSF -o channel)
python -u script.py \
    1> /work3/02613/output_${LSB_JOBID}.txt \
    2> /work3/02613/error_${LSB_JOBID}.txt
```

---

## Exam Traps

| Trap | Correct |
|---|---|
| `rusage[mem=16GB]` with 8 cores when program needs 16 GB total | `rusage[mem=2GB]` |
| `-w done(job)` when cleanup should run even if job failed | `-w ended(job)` |
| `$LSB_JOBINDEX` used as 0-based list index directly | `int(sys.argv[1]) - 1` |
| `-q hpc` for a GPU job | `-q gpuv100` or `-q c02613` |
| Missing `span[hosts=1]` for multiprocessing with shared memory | Add it |
| `-N` flag with 200-element array | Sends 200 emails — omit `-N` |
