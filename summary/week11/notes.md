# Week 11 — HPC Workflows: Job Arrays & Dependencies

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Overview](#overview)
- [Theory & Concepts](#theory-concepts)
  - [HPC as a Compute Cluster](#hpc-as-a-compute-cluster)
  - [Job Arrays](#job-arrays)
  - [Job Dependencies](#job-dependencies)
  - [Job States](#job-states)
  - [Workflow Patterns](#workflow-patterns)
- [Job Array Syntax](#job-array-syntax)
  - [Output file naming for job arrays](#output-file-naming-for-job-arrays)
- [Job Dependency Syntax](#job-dependency-syntax)
  - [Submitting a dependent job from the command line](#submitting-a-dependent-job-from-the-command-line)
- [Monitoring Job Arrays](#monitoring-job-arrays)
- [Key Code Examples](#key-code-examples)
  - [Minimal job array script](#minimal-job-array-script)
  - [Dependent aggregation job](#dependent-aggregation-job)
  - [Element-wise array dependency](#element-wise-array-dependency)
- [Exercise Highlights](#exercise-highlights)
  - [Exercise 1 — Job Script Workflows](#exercise-1-job-script-workflows)
  - [Exercise 2 — CelebA Face Color Histogram (Map-Reduce Workflow)](#exercise-2-celeba-face-color-histogram-map-reduce-workflow)
- [Useful LSF Commands Reference](#useful-lsf-commands-reference)
- [Key Takeaways](#key-takeaways)

---

## Overview

Week 11 shifts perspective from using HPC as a single bigger computer to using it as a **compute cluster**: orchestrating many jobs simultaneously via the LSF batch submission system. The two key tools are **job arrays** (submit many nearly-identical jobs at once) and **job dependencies** (make jobs wait on other jobs). Together these enable map-reduce style workflows where a large dataset is processed in parallel and results are then aggregated by a single downstream job.

---

## Theory & Concepts

### HPC as a Compute Cluster

Previously the course treated HPC as a bigger machine: write one job script, submit it, wait. Week 11 generalises this to running large numbers of coordinated jobs across many work nodes. The login node sits between the user and the cluster; all job submission and monitoring happens from the login node using LSF commands (`bsub`, `bjobs`, `bkill`, `bpeek`, `bstat`).

### Job Arrays

**Definition:** A job array is a group of jobs that share the same executable and resource requirements but operate on different inputs. They share one job ID and are individually identified by an array index.

**Why use them?** Without job arrays you must submit one `bsub` call per input file or parameter value, manually change the script each time, and then manage hundreds of separate job IDs if something goes wrong. Job arrays replace all of that with a single submission.

**Key property:** All subjobs share the same job ID; each is uniquely referenced by its index (e.g. `702576[3]`). The index is exposed inside the running job as the environment variable `$LSB_JOBINDEX`.

**Recommended pattern:** Pass `$LSB_JOBINDEX` to your Python script as a command-line argument; the Python script converts the integer index to a filename or parameter value. This keeps the shell script generic and puts the mapping logic where it is easier to test.

### Job Dependencies

**Definition:** A dependency tells LSF to hold a job in PEND until one or more other jobs reach a specified state.

**Why use them?** Real workflows have stages. For example: process 200 image folders in parallel (job array), then aggregate the results (single job). Without dependencies you would have to manually watch the array and submit the aggregation step by hand. With dependencies the aggregation job is submitted upfront and automatically starts when the array finishes.

**Common real-world cases:**
- GPU training job hits the 24-hour wall time; schedule a resume job that waits for it to exit
- GPU pre-processing followed by long CPU analysis: schedule both at once
- Job array (map) followed by aggregation (reduce): the canonical map-reduce pattern

### Job States

Jobs move through a well-defined state machine:

```
bsub -> PEND -> (suitable host found) -> RUN -> (normal completion) -> DONE
                                              -> (bkill / abnormal exit) -> EXIT
```

The `*SUSP` states (PSUSP, USUSP, SSUSP) exist for suspended jobs but are not used in this course. The distinction between DONE and EXIT matters for dependencies: `done()` triggers only on DONE; `ended()` triggers on both DONE and EXIT.

### Workflow Patterns

- **Map-reduce:** Job array computes per-chunk results; a dependent job aggregates them.
- **Parameter sweeps:** Submit one job array indexed over hyperparameter combinations (e.g. learning rates 1e-3, 1e-4, 1e-5, ...) instead of editing and resubmitting the same script repeatedly.
- **Pipeline stages:** Chain jobs A -> B -> C using dependencies so each stage starts automatically when the previous one finishes.

---

## Job Array Syntax

```bash
# Range (inclusive): indices 1, 2, ..., 10
#BSUB -J name[1-10]

# Explicit list: indices 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
#BSUB -J name[1,2,3,4,5,6,7,8,9,10]

# Range with step: indices 1, 3, 5, 7, 9
#BSUB -J name[1-10:2]

# Mixed: indices 1, 4, 10, 11, 12, 13, 14, 20, 22, 24, 26, 28, 30
#BSUB -J name[1,4,10-14,20-30:2]

# Inside the job script, the current index is:
IDX=${LSB_JOBINDEX}
python script.py ${IDX}
```

### Output file naming for job arrays

```bash
# All subjobs write to the same file (often causes interleaving):
#BSUB -o name_%J.out      # %J = job ID

# Each subjob writes to its own file (recommended):
#BSUB -o name_%J_%I.out   # %J = job ID, %I = array index
#BSUB -e name_%J_%I.err
```

**Warning on email notifications:** Do not use `#BSUB -B` / `#BSUB -N` with job arrays. A 182-element array generates 364 emails; the DTU mail server rejects batches above ~100 in a short interval.

---

## Job Dependency Syntax

Dependencies are specified with `#BSUB -w` (wait condition). The condition uses the job **name** (not the ID), which is the most robust approach when scripts are submitted sequentially.

```bash
# Wait for all jobs named "job1" to finish successfully (state DONE):
#BSUB -w job1

# More explicit form — same effect:
#BSUB -w "done(job1)"

# Wait for a specific element of a job array to finish successfully:
#BSUB -w "done(array[3])"

# Wait for corresponding element of an array (element-wise dependency):
# New array index i waits for existing array element i
#BSUB -w "done(array[*])"

# Wait for any exit (DONE or EXIT) — useful for cleanup jobs:
#BSUB -w "ended(job1)"
```

### Submitting a dependent job from the command line

You can also pass the wait condition at submission time rather than hard-coding it in the script:

```bash
bsub -w "done(job1)" < job2.sh
```

---

## Monitoring Job Arrays

```bash
# Show all your jobs (one row per array element):
bstat

# Show array summary (one row per array, with state counts):
bjobs -A
# Columns: JOBID  ARRAY_SPEC  OWNER  NJOBS  PEND  DONE  RUN  EXIT  SSUSP  USUSP  PSUSP

# Peek at stdout/stderr of a running array element (must specify index):
bpeek 21415299[3]

# Kill all elements of an array:
bkill 21415299

# Kill a subset of elements:
bkill 21415299[1-5]
```

`bjobs -A` is the most useful monitoring command for arrays: it shows at a glance how many jobs are pending, running, done, or exited without listing every individual element.

---

## Key Code Examples

### Minimal job array script

```bash
#!/bin/bash
#BSUB -J subhist[1-203]
#BSUB -q hpc
#BSUB -W 15
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=512MB]"
#BSUB -o batch_output/subhist_%J_%I.out
#BSUB -e batch_output/subhist_%J_%I.err

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

python huedir.py $LSB_JOBINDEX
```

### Dependent aggregation job

```bash
#!/bin/bash
#BSUB -J plothist
#BSUB -q hpc
#BSUB -W 5
#BSUB -n 1
#BSUB -w done(subhist)
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=512MB]"
#BSUB -o batch_output/plothist_%J.out
#BSUB -e batch_output/plothist_%J.err

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

python plothuehist.py
```

The `done(subhist)` condition means: wait until every job named `subhist` (the entire array) reaches DONE state. If any element exits with an error, `plothist` will never start — which is the desired safety behaviour.

### Element-wise array dependency

When you want new_array[i] to wait only for old_array[i] (not the whole array):

```bash
#BSUB -J new_array[1-5]
#BSUB -w "done(old_array[*])"
```

The `[*]` notation makes each element of the new array depend on the corresponding element of the referenced array by index.

---

## Exercise Highlights

### Exercise 1 — Job Script Workflows

Practice writing the LSF header directives for different scenarios:

1. Job array with indices 1–10: `#BSUB -J name[1-10]`
2. Job array with specific non-contiguous indices: `#BSUB -J name[2,29,71,73,127]`
3. Dependency on a specific job ID: `#BSUB -w "done(1234567)"`
4. Element-wise dependency on an already-running array named `array`: `#BSUB -w "done(array[*])"`
5. Wait for all elements of an array to exit for any reason: `#BSUB -w "ended(array)"`

### Exercise 2 — CelebA Face Color Histogram (Map-Reduce Workflow)

**Dataset:** CelebA celebrity image dataset stored in 203 subfolders under `/dtu/projects/02613_2024/data/celeba/images`. Each subfolder holds ~1000 images.

**Goal:** Compute a hue histogram over the entire dataset efficiently using a job array + dependent aggregation job.

**Map step (`huedir.py`):** Each array job receives an index `i` via `$LSB_JOBINDEX`, identifies the `i`th subfolder (sorted), loads every image, computes the hue histogram per image, sums them, and saves `subhist_<i>.npy`.

```python
import os
from os.path import join
import sys
import numpy as np
from PIL import Image

def huehist(image):
    bins = np.linspace(0, 255, 64 + 1)
    hsv_image = np.asarray(Image.fromarray(image).convert('HSV'))
    hue_values = hsv_image[:, :, 0].reshape(-1)
    hue_hist = np.histogram(hue_values, bins)[0]
    return hue_hist

if __name__ == '__main__':
    idx = int(sys.argv[1]) - 1  # Convert 1-indexed LSB_JOBINDEX to 0-indexed
    base_path = '/dtu/projects/02613_2024/data/celeba/images'
    folders = sorted(os.listdir(base_path))
    images = sorted(os.listdir(join(base_path, folders[idx])))

    hist = 0
    for im in images:
        image = Image.open(join(base_path, folders[idx], im))
        image = np.array(image)
        hist += huehist(image)

    np.save(f"subhist_{idx}.npy", hist)
```

**Reduce step (`plothuehist.py`):** Loads all `subhist_*.npy` files, sums them, and plots the combined histogram.

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

**Result:** Red and orange hues dominate the histogram, which makes sense since skin tones fall in that range.

**Important indexing note:** LSF job array indices are 1-based. The Python script subtracts 1 (`idx = int(sys.argv[1]) - 1`) to convert to 0-based list indexing.

---

## Useful LSF Commands Reference

| Command | Purpose |
|---|---|
| `bsub < script.sh` | Submit a job |
| `bstat` | List your active jobs (one row per element) |
| `bjobs -A` | Summarise job arrays (one row per array, state counts) |
| `bpeek JOBID[IDX]` | Peek at live stdout/stderr of an array element |
| `bkill JOBID` | Kill all elements of a job / array |
| `bkill JOBID[1-5]` | Kill a subset of array elements |
| `bkill 0` | Kill all your jobs |
| `bhist` | Job history |
| `bjobs -l JOBID` | Detailed info on a specific job |

---

## Key Takeaways

- **Job arrays** eliminate the need to submit the same script dozens or hundreds of times with minor edits. One `bsub` call, one job ID, automatic parallelism.
- **`$LSB_JOBINDEX`** is the environment variable that distinguishes array elements; convert it to a filename or parameter inside the Python script, not in the shell script, for cleaner code.
- **Index syntax is flexible:** ranges (`[1-100]`), steps (`[1-100:2]`), explicit lists (`[1,4,7]`), and combinations are all supported.
- **`%I` in output filenames** gives each array element its own log file, which is essential for debugging large arrays.
- **Job dependencies** use `#BSUB -w` with conditions like `done(jobname)` (successful exit only) or `ended(jobname)` (any exit). This enables fully automated multi-stage pipelines.
- **`done(array[*])`** creates element-wise dependencies between arrays: new job `i` waits for existing job `i`.
- **Do not use email notifications** (`-B`/`-N`) with large job arrays — you will flood the mail server.
- **Responsibility:** Large job arrays consume shared cluster resources. Always test with a small array (e.g. 3 elements) before submitting hundreds of jobs.
- The map-reduce pattern (array for parallel processing + one dependent job for aggregation) is the fundamental building block for HPC workflows on shared clusters.
