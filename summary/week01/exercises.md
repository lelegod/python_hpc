# Week 1 Exercises — DTU HPC Intro, Linux Basics, LSF/BSUB

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Section 1: Connecting and Transferring Files to the HPC](#section-1-connecting-and-transferring-files-to-the-hpc)
  - [Exercise 1.1 `[AUTOLAB]`](#exercise-11-autolab)
  - [Exercise 1.2 `[PRACTICE]`](#exercise-12-practice)
  - [Exercise 1.3 `[PRACTICE]`](#exercise-13-practice)
  - [Exercise 1.4 `[PRACTICE]`](#exercise-14-practice)
  - [Exercise 1.5 `[PRACTICE]`](#exercise-15-practice)
  - [Exercise 1.6 `[PRACTICE]`](#exercise-16-practice)
  - [Exercise 1.7 `[PRACTICE]`](#exercise-17-practice)
- [Section 2: Job Scripts](#section-2-job-scripts)
  - [Exercise 2.1 `[AUTOLAB]`](#exercise-21-autolab)
  - [Exercise 2.2 `[AUTOLAB]`](#exercise-22-autolab)
  - [Exercise 2.3 `[PRACTICE]`](#exercise-23-practice)
  - [Exercise 2.4 `[AUTOLAB]`](#exercise-24-autolab)
  - [Exercise 2.5 `[AUTOLAB]`](#exercise-25-autolab)
  - [Exercise 2.6 `[AUTOLAB]`](#exercise-26-autolab)
  - [Exercise 2.7 `[PRACTICE]`](#exercise-27-practice)

---

---

## Section 1: Connecting and Transferring Files to the HPC

The focus of this exercise is to learn how to connect to the HPC nodes, transfer files and run code on the HPC, preferably using the terminal.

---

### Exercise 1.1 `[AUTOLAB]`

Write a Python program that writes "Hello world" to a file (e.g., called 'content.txt'). It should also print the same text to the screen.

> **Solution:**
>
> Student file: `content.py`
>
> ```python
> def main(text: str = "Hello world"):
>     print(text)
>     with open("content.txt", "w") as f:
>         f.write(text)
>
> if __name__ == "__main__":
>     main()
> ```

---

### Exercise 1.2 `[PRACTICE]`

Transfer your Python file to the HPC using scp, sftp, FileZilla, or other.

> **Solution:**
>
> Assuming the script is called `helloworld.py`, transfer it with SCP:
>
> ```bash
> scp helloworld.py s123456@login.hpc.dtu.dk:helloworld.py
> ```
>
> If you already have a target directory on the HPC:
>
> ```bash
> scp helloworld.py s123456@login.hpc.dtu.dk:Documents/02613/w01/helloworld.py
> ```

---

### Exercise 1.3 `[PRACTICE]`

Connect to an HPC terminal. If you are *not on a DTU network*, you may need to use a VPN or set up SSH keys. If you used SSH, make sure you called `linuxsh` so you are *not* on a login node.

> **Solution:**
>
> Log on to the cluster with SSH:
>
> ```bash
> ssh s123456@login.hpc.dtu.dk
> ```
>
> Then type your password. After logging in, run:
>
> ```bash
> linuxsh
> ```
>
> This moves you off the login node to an interactive Linux node where you can run code.

---

### Exercise 1.4 `[PRACTICE]`

Create and/or navigate to your working directory (on the HPC) for this exercise.

> **Solution:**
>
> If the directory does not yet exist, create it and move the uploaded file there:
>
> ```bash
> mkdir -p Documents/02613/w01
> mv helloworld.py -t Documents/02613/w01
> cd Documents/02613/w01
> ```
>
> If you already uploaded directly to the directory:
>
> ```bash
> cd Documents/02613/w01
> ```

---

### Exercise 1.5 `[PRACTICE]`

Initialize the course conda environment.

> **Solution:**
>
> First, activate the shared Anaconda installation:
>
> ```bash
> source /dtu/projects/02613_2025/conda/conda_init.sh
> ```
>
> Then activate the course conda environment:
>
> ```bash
> conda activate 02613
> ```

---

### Exercise 1.6 `[PRACTICE]`

Run your Python program.

> **Solution:**
>
> ```bash
> python helloworld.py
> ```

---

### Exercise 1.7 `[PRACTICE]`

Transfer the generated file (e.g., `content.txt`) back to your local computer.

> **Solution:**
>
> Exit your SSH session (run `exit` until you are back in your own terminal), then from your local machine run:
>
> ```bash
> scp s123456@login.hpc.dtu.dk:Documents/02613/w01/helloworld.py helloworld.py
> ```

---

## Section 2: Job Scripts

For all scripts below, use `/bin/sleep 60` (or a longer period, like 100 or 120 seconds) as the command to run, and use 'hpc' as the queue name (except if stated otherwise).

---

### Exercise 2.1 `[AUTOLAB]`

Write a simple job script, like the one shown in the lecture, and submit it.

> **Solution:**
>
> Student file: `submit1.sh`
>
> ```bash
> #!/bin/bash
> #BSUB -J sleeper
> #BSUB -q hpc
> #BSUB -W 2
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -n 4
> #BSUB -R "span[hosts=1]"
> #BSUB -R "select[model==XeonGold6226R]"
> #BSUB -o sleeper_%J.out
> #BSUB -e sleeper_%J.err
>
> sleep 60
> ```
>
> Submit with:
>
> ```bash
> bsub < submit1.sh
> ```

#### Exercise 2.1a `[PRACTICE]`

Check the status with `bstat` and/or `bjobs`. Use `man bjobs` to get information about the options.

> **Solution:**
>
> Running `bstat` just after submitting will show something like:
>
> ```
> $ bstat
> JOBID      USER    QUEUE    JOB_NAME   NALLOC STAT  START_TIME    ELAPSED
> 19882525   patmjen hpc      sleeper         0 PEND  -             0:00:00
> ```
>
> The STAT column shows `PEND` (pending, not yet started). Once started:
>
> ```
> $ bstat
> JOBID      USER    QUEUE    JOB_NAME   NALLOC STAT  START_TIME    ELAPSED
> 19882525   patmjen hpc      sleeper         1 RUN   Jan 10 16:39  0:00:07
> ```
>
> `bjobs` shows similar information with a `TIME_LEFT` column showing how much walltime remains.

#### Exercise 2.1b `[PRACTICE]`

You can add a walltime limit to the script. Can you see that limit in the `bstat` or the `bjobs` output?

> **Solution:**
>
> The `bstat` output does not include information about the wall time limit. The `bjobs` output includes the `TIME_LEFT` column, which tells how much time is left before hitting the wall time limit set with `#BSUB -W`.

---

### Exercise 2.2 `[AUTOLAB]`

Write a job script that sends you notifications when the job starts and ends — see `man bsub` for the details. Take a look at the job summary: what information can you retrieve from that?

> **Solution:**
>
> Student file: `submit2.sh`
>
> Add `-B` (notify on start) and `-N` (notify on end) flags:
>
> ```bash
> #!/bin/bash
> #BSUB -J sleeper
> #BSUB -q hpc
> #BSUB -W 2
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -n 4
> #BSUB -R "span[hosts=1]"
> #BSUB -R "select[model==XeonGold6226R]"
> #BSUB -o sleeper_%J.out
> #BSUB -e sleeper_%J.err
> #BSUB -B -N
>
> sleep 60
> ```
>
> The job summary email contains:
> - Home and working directories used
> - A copy of the submitted job script
> - A resource usage summary (run time, max memory used, etc.)
> - A completion status — "Successfully completed" if the job finished normally
>
> **Note:** If 'Run time' or 'Max Memory' differs significantly from what was requested, consider adjusting the script if you plan to rerun it.

#### Exercise 2.2a `[PRACTICE]`

To test, increase the period in the sleep command to be longer than the wall time limit, and submit the job again. What happens?

> **Solution:**
>
> Student file: `submit3.sh` (sleep 10 with wall time of 2 minutes — reverse: set sleep longer than wall time to test)
>
> The job will start as before, but after the wall time limit is reached it will be terminated. In the job summary email, instead of "Successfully completed" it will say something like:
>
> ```
> TERM_RUNLIMIT: job killed after reaching LSF run time limit.
> ```

---

### Exercise 2.3 `[PRACTICE]`

The default `hpc` queue has nodes of different type, e.g. with different CPUs. The CPU type can be requested as a feature in a job script.

#### Exercise 2.3a `[AUTOLAB]`

Use the `nodestat` command to check which CPU types are available in the `hpc` queue, and then submit a job script that requests one of the types.

> **Solution:**
>
> To see nodes in the hpc queue with their CPU types:
>
> ```bash
> nodestat -f hpc
> ```
>
> Then request a specific CPU type in the job script using `select[model==...]` (underscores in the LSF model name correspond to hyphens in the `nodestat` output):
>
> ```bash
> #!/bin/bash
> #BSUB -J sleeper
> #BSUB -q hpc
> #BSUB -W 2
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -n 4
> #BSUB -R "span[hosts=1]"
> #BSUB -R "select[model==XeonGold6226R]"
> #BSUB -o sleeper_%J.out
> #BSUB -e sleeper_%J.err
>
> sleep 60
> ```

#### Exercise 2.3b `[PRACTICE]`

Add the necessary commands to your job script to print the CPU type, and check in the job output that your job did indeed run on a node with the requested feature. Note: there is a slight difference in the type request and the type variable — the variable contains a '-', while LSF uses a '_'.

> **Solution:**
>
> Add `lscpu` to the job script. The output will be saved in `sleeper_XXXXXX.out` after the job finishes.
>
> Student file: `submit.sh` includes `lscpu`:
>
> ```bash
> #!/bin/bash
> #BSUB -J sleeper
> #BSUB -q hpc
> #BSUB -W 2
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -n 4
> #BSUB -R "span[hosts=1]"
> #BSUB -R "select[model==XeonGold6226R]"
> #BSUB -o sleeper_%J.out
> #BSUB -e sleeper_%J.err
>
> lscpu
> sleep 60
> ```

---

### Exercise 2.4 `[AUTOLAB]`

Write a job script that requests 1 node and 4 cores.

> **Solution:**
>
> Student file: `submit4.sh`
>
> ```bash
> #!/bin/bash
> #BSUB -J sleeper
> #BSUB -q hpc
> #BSUB -W 2
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -n 4
> #BSUB -R "span[hosts=1]"
> #BSUB -R "select[model==XeonGold6226R]"
> #BSUB -o sleeper_%J.out
> #BSUB -e sleeper_%J.err
>
> sleep 10
> ```
>
> Key directives: `-n 4` requests 4 cores, `-R "span[hosts=1]"` ensures all cores are on a single node.

---

### Exercise 2.5 `[AUTOLAB]`

Write a job script, requesting 1 node and 16 cores. Does it run? If the job doesn't start, use `bjobs -p` to check for the reason.

> **Solution:**
>
> Student file: `submit5.sh`
>
> ```bash
> #!/bin/bash
> #BSUB -J sleeper
> #BSUB -q hpc
> #BSUB -W 2
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -n 16
> #BSUB -R "span[hosts=1]"
> #BSUB -R "select[model==XeonGold6226R]"
> #BSUB -o sleeper_%J.out
> #BSUB -e sleeper_%J.err
>
> sleep 10
> ```
>
> It will start, as the `hpc` queue includes nodes with processors that have more than 16 cores available.

---

### Exercise 2.6 `[AUTOLAB]`

Write a job script, requesting 1 node and 64 cores. Does it run? If the job doesn't start, use the `bjobs -p` command to check for the reason.

> **Solution:**
>
> Student file: `submit6.sh`
>
> ```bash
> #!/bin/bash
> #BSUB -J sleeper
> #BSUB -q hpc
> #BSUB -W 2
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -n 64
> #BSUB -R "span[hosts=1]"
> #BSUB -R "select[model==XeonGold6226R]"
> #BSUB -o sleeper_%J.out
> #BSUB -e sleeper_%J.err
>
> sleep 10
> ```
>
> It will **not** start, as the `hpc` queue does not include nodes with processors that have 64 cores. Running `bjobs -p` will show:
>
> ```
> JOBID      USER    STAT  QUEUE    FROM_HOST  EXEC_HOST  JOB_NAME    SUBMIT_TIME
> 19882618   patmjen PEND  hpc      hpclogin2             sleeper     Jan 10 16:33
> Not enough job slot(s): 10 hosts;
> ```
>
> The message `Not enough job slot(s)` indicates that no node has enough CPU cores available to run the job.

---

### Exercise 2.7 `[PRACTICE]`

Check all your submitted jobs with `bstat` again. If there are any left that still are in status 'PEND', please remove them with `bkill JOBID` (the JOBID is the number in the first column of the `bstat`/`bjobs` output).

> **Solution:**
>
> First check status:
>
> ```bash
> bstat
> ```
>
> Example output showing a pending job:
>
> ```
> JOBID      USER    QUEUE    JOB_NAME   NALLOC STAT  START_TIME    ELAPSED
> 19882618   patmjen hpc      sleeper         0 PEND  -             0:00:00
> ```
>
> Kill any pending jobs by their JOBID:
>
> ```bash
> bkill 19882618
> ```
