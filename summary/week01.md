# Week 1 — Introduction to Python HPC & DTU HPC System

## Overview

Week 1 establishes the foundations for the entire course. The lecture covers what HPC is and why it matters, introduces DTU's specific cluster infrastructure, and gives hands-on practice with the LSF batch job scheduler. The exercises build the muscle memory needed for every subsequent week: SSH into the cluster, transfer files, activate the conda environment, write a job script, and submit it. Everything from Week 2 onward assumes you can do all of this confidently.

**Course structure reminder:**
- Weeks 1-8: CPU-focused HPC
- Weeks 9-10: GPU computing (Numba, CuPy)
- Weeks 11-13: HPC Workflows, mini-project, pitfalls & Q&A
- Mini-project: groups of 3-4, released Week 4, due Sunday 3 May 2026 at 23:59 — must be completed and approved to sit the exam
- Autolab: must complete at least 20 exercises total AND at least 1 from every week to sit the exam (deadline: 11 May at 6 AM)
- Exam: 4 hours, written digital, all aids permitted except internet and generative AI

---

## Theory & Concepts

### What is HPC and Why It Matters

High-Performance Computing (HPC) refers to using aggregated computing power — multiple processors, nodes, and accelerators — to solve problems that are too large, too slow, or too memory-intensive for a single machine.

**Why you need HPC (not just your laptop):**
- Topology optimization, wind turbine aerodynamics, climate modelling, deep learning training — all require more RAM, more cores, or longer runtimes than any laptop provides
- Cloud alternatives (AWS, Azure, Google Cloud) exist but cost money per hour; DTU's cluster is free for students
- Your laptop has a thermal and battery constraint; a cluster runs 24/7

**Course goals (directly from slides):**
1. Identify slow code and improve its performance
2. Understand how hardware affects performance
3. Efficiently utilize single-core, multi-core, and GPU resources
4. Use a modern HPC system, not just your own laptop/PC

### What is an HPC Cluster?

An HPC cluster is a set of networked computers (nodes) coordinated by a **Resource Manager** (also called a scheduler or batch system). DTU uses **LSF 10** (IBM Spectrum LSF) as its resource manager.

Key node types:
- **Login node** (`login.hpc.dtu.dk`): where you SSH in. Used only for editing, file transfer, and submitting jobs. **Never run computationally heavy work here.**
- **Work/compute nodes**: where jobs actually run. Allocated by the scheduler. You cannot SSH directly to them during normal use.

To get an interactive shell on a work node (not the login node), use:
```bash
linuxsh
```

### DTU HPC Cluster Architecture

- Access via SSH: `ssh sXXXXXX@login.hpc.dtu.dk`
- After logging in, call `linuxsh` to move to a work node for interactive use
- Two main queues for this course:
  - `hpc` — CPU jobs (most of the course)
  - `gpuv100` — GPU jobs (Weeks 9-10)
- Nodes in the `hpc` queue have different CPU types (use `nodestat -f hpc` to see them)
- The `hpc` queue has nodes with up to ~32 cores; requesting 64 cores on a single node will fail because no such node exists

### Python's Role in HPC

Python is interpreted, which makes it slower than compiled languages (C, Fortran). However:
- Fast libraries written in C/Fortran are callable from Python: NumPy, SciPy, Pandas
- Python is the "glue" layer — orchestrate fast low-level code from a high-level language
- This course teaches how to write Python that leverages those fast libraries correctly, and how to use the cluster infrastructure to scale beyond a single machine

---

## LSF Job Script Reference

The LSF batch system reads directives from lines beginning with `#BSUB` at the top of a bash script. All directives must come before any commands.

```bash
#!/bin/bash
#BSUB -J jobname           # Job name (shown in bstat/bjobs output)
#BSUB -q hpc               # Queue: 'hpc' (CPU) or 'gpuv100' (GPU)
#BSUB -W HH:MM             # Wall-clock time limit. Job is killed when reached.
                            # Use -W 2 for 2 minutes, -W 1:30 for 1h30m.
#BSUB -n N                 # Number of CPU cores to allocate
#BSUB -R "span[hosts=1]"   # Require all N cores to be on the SAME node
                            # (needed for shared-memory parallelism)
#BSUB -R "rusage[mem=Xmb]" # Memory per core in MB (e.g. rusage[mem=512MB])
                            # Total memory = N * X MB
#BSUB -R "select[model==XeonGold6226R]"  # Request a specific CPU model
                            # Use nodestat -f hpc to see available models
                            # Note: LSF uses underscore in model names;
                            # the CPU variable uses a hyphen
#BSUB -o name_%J.out       # Stdout file. %J is replaced by the numeric job ID.
#BSUB -e name_%J.err       # Stderr file.
#BSUB -u email@address.dk  # Email address for notifications
#BSUB -N                   # Send email notification when job ends
#BSUB -B                   # Send email notification when job begins

# --- Your commands below ---
lscpu          # Print CPU info to stdout (useful for verifying node type)
sleep 60       # The actual work
```

### Annotated Example (from submit.sh in this repo)

```bash
#!/bin/bash
#BSUB -J sleeper              # Job name
#BSUB -q hpc                  # Use the CPU queue
#BSUB -W 2                    # Kill after 2 minutes
#BSUB -R "rusage[mem=512MB]"  # 512 MB per core
#BSUB -n 4                    # 4 cores
#BSUB -R "span[hosts=1]"      # All on one node
#BSUB -R "select[model==XeonGold6226R]"  # Specific CPU type
#BSUB -o sleeper_%J.out       # Stdout to file
#BSUB -e sleeper_%J.err       # Stderr to file

lscpu     # Verify which CPU we got
sleep 60  # Placeholder workload
```

### What Happens When the Wall Time is Exceeded

If your job is still running when the wall-clock limit (`-W`) expires, LSF kills it with status `TERM_RUNLIMIT`. The job summary in the `.out` file will say:
```
TERM_RUNLIMIT: job killed after reaching LSF run time limit.
```
instead of `Successfully completed.`

### Core Count Limits

| Requested cores | Result |
|---|---|
| 1-4 | Runs fine |
| 16 | Runs fine (nodes with 16+ cores exist) |
| 64 | Will NOT start — no node in `hpc` has 64 cores |

When a job cannot start due to insufficient slots: `bjobs -p` shows `Not enough job slot(s): X hosts`.

---

## Key Linux/HPC Commands

### Connecting and File Transfer

```bash
# Log in to DTU HPC (from any network; VPN or SSH keys needed off-campus)
ssh sXXXXXX@login.hpc.dtu.dk

# Move from login node to an interactive work node
linuxsh

# Copy a file TO the cluster (run on your local machine)
scp helloworld.py sXXXXXX@login.hpc.dtu.dk:helloworld.py

# Copy to a specific directory on the cluster
scp helloworld.py sXXXXXX@login.hpc.dtu.dk:Documents/02613/w01/helloworld.py

# Copy a file FROM the cluster back to local machine
scp sXXXXXX@login.hpc.dtu.dk:Documents/02613/w01/content.txt content.txt
```

### Directory Navigation

```bash
mkdir -p Documents/02613/w01    # Create nested directory
mv helloworld.py -t Documents/02613/w01   # Move file into directory
cd Documents/02613/w01          # Navigate to directory
```

### Conda Environment (course-specific)

```bash
# Step 1: activate the shared Anaconda installation
source /dtu/projects/02613_2025/conda/conda_init.sh

# Step 2: activate the course environment
conda activate 02613
```

### Job Submission and Monitoring

```bash
# Submit a job script
bsub < submit.sh

# Check all your jobs (compact view)
bstat

# Check your jobs (detailed view, includes TIME_LEFT column)
bjobs

# Check why a pending job hasn't started
bjobs -p

# Watch a running job's output in real-time
bpeek JOBID

# Kill a job (pending or running)
bkill JOBID

# Kill all your jobs
bkill 0
```

### Node Information

```bash
# See available nodes and CPU types in the hpc queue
nodestat -f hpc

# Print detailed CPU information (run inside a job script or on a work node)
lscpu

# Show system resource usage interactively
htop
```

### Reading bstat / bjobs Output

```
$ bstat
JOBID     USER    QUEUE  JOB_NAME  NALLOC  STAT   START_TIME  ELAPSED
19882525  patmjen hpc    sleeper        0  PEND   -           0:00:00
```
- `STAT = PEND`: job is waiting in the queue
- `STAT = RUN`: job is running; `START_TIME` and `ELAPSED` are populated

```
$ bjobs
JOBID     USER    QUEUE  JOB_NAME  SLOTS  STAT  START_TIME     TIME_LEFT
19882525  patmjen hpc    sleeper       1  RUN   Jan 10 16:39   00:01:53 L
```
- `TIME_LEFT`: how long until the wall-clock limit kills the job (from `bjobs`, not `bstat`)

---

## Timing & Profiling Basics

Week 1 introduces the principle that performance measurement must happen on the actual hardware where the code will run. The login node is a shared machine with variable load — never benchmark there.

**Rule:** always profile and time your code by submitting it as a batch job to a dedicated work node.

Basic timing in Python (covered more deeply in later weeks):

```python
import time

t_start = time.perf_counter()
# ... your code ...
t_end = time.perf_counter()
print(f"Elapsed: {t_end - t_start:.4f} seconds")
```

`time.perf_counter()` is preferred over `time.time()` for short intervals — it uses the highest-resolution clock available and is not affected by system clock adjustments.

---

## Key Code Examples

### content.py (Week 1 Hello World)

```python
def main(text: str = "Hello world"):
    print(text)
    with open("content.txt", "w") as f:
        f.write(text)

if __name__ == "__main__":
    main()
```

Key points:
- Prints to stdout AND writes to a file — the Autolab autograder checks both
- Autograder checks: ran successfully, printed "Hello world", nothing on stderr, created a new file, file content matches printed output

### submit.sh (Week 1 Job Script)

```bash
#!/bin/bash
#BSUB -J sleeper
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=512MB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -o sleeper_%J.out
#BSUB -e sleeper_%J.err

lscpu
sleep 60
```

### Full Workflow (from exercise solutions)

```bash
# 1. On local machine: write and test your script, then transfer
scp helloworld.py s123456@login.hpc.dtu.dk:Documents/02613/w01/helloworld.py

# 2. SSH in
ssh s123456@login.hpc.dtu.dk
linuxsh   # move off login node

# 3. Set up conda
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# 4. Navigate and run interactively
cd Documents/02613/w01
python helloworld.py

# 5. Or submit as a batch job
bsub < submit.sh

# 6. Monitor
bstat
bjobs -p   # if job stuck in PEND

# 7. Copy output back to local
exit   # back to local machine
scp s123456@login.hpc.dtu.dk:Documents/02613/w01/content.txt content.txt
```

---

## Exercise Highlights

**Exercise 1 — Connecting and transferring files:**
1. Write a Python program that prints "Hello world" and writes it to `content.txt` — submit to Autolab
2. Transfer the `.py` file to HPC using `scp` (or sftp/FileZilla)
3. SSH in, run `linuxsh` to leave the login node
4. Create a working directory, activate the 02613 conda environment
5. Run the script, transfer `content.txt` back

**Exercise 2 — Job Scripts (6 sub-exercises, all on Autolab):**
1. Write a minimal job script (use `sleep 60` as the workload), submit it, check status with `bstat`/`bjobs`; observe that `bjobs` shows `TIME_LEFT` but `bstat` does not
2. Add email notifications (`-u`, `-N`, `-B`); test what happens when sleep exceeds the wall time limit (job is killed with `TERM_RUNLIMIT`)
3. Use `nodestat -f hpc` to find CPU types; request a specific model with `select[model==...]`; verify with `lscpu` in the job output
4. Request 1 node, 4 cores (`-n 4` + `span[hosts=1]`) — runs fine
5. Request 1 node, 16 cores — runs fine (nodes with 16+ cores exist)
6. Request 1 node, 64 cores — will NOT start (`Not enough job slot(s)`)
7. Clean up: `bkill JOBID` for any remaining PEND jobs

**Key insight from exercises:** `bjobs -p` is the diagnostic tool when a job is stuck in `PEND`. The `TIME_LEFT` column in `bjobs` (not in `bstat`) shows remaining wall time.

---

## Key Takeaways

- **Never run computation on the login node** — use `linuxsh` for interactive work or submit batch jobs with `bsub`
- **The `#BSUB` directives are the exam** — know every directive: `-J`, `-q`, `-W`, `-n`, `-R "span[hosts=1]"`, `-R "rusage[mem=...]"`, `-R "select[model==...]"`, `-o`, `-e`, `-u`, `-N`, `-B`
- **`span[hosts=1]` is required for shared-memory parallelism** — without it, your N cores may land on different physical nodes and cannot share memory
- **Requesting more cores than any node has causes permanent PEND** — use `bjobs -p` to diagnose and `nodestat -f hpc` to check what is available
- **The job output file (`.out`) contains the job summary** — check it for `Successfully completed.` vs `TERM_RUNLIMIT` and for resource usage (Run time, Max Memory)
- **Activate the conda environment in every job script** if your Python code needs course packages — add the `source` and `conda activate` lines before calling Python
- **Autolab requires at least 1 submission per week AND 20 total** to qualify for the exam — start early, do not leave it to the final deadline
