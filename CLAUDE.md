# CLAUDE.md — python_hpc

This file documents the repository for a future Claude Code agent helping Kyle with HPC coursework.

---

## What This Repo Is

Course materials, exercises, solutions, and notes for **02613 Python and High-Performance Computing** at DTU (Technical University of Denmark), Semester 2.

- **Student:** Kyle Nathan Yahya (s252786 on DTU HPC)
- **Instructor:** Matthias Bo Stuart
- **Platform:** DTU's LSF HPC cluster (`login.hpc.dtu.dk`)
- **Textbooks:**
  - "Fast Python" by Tiago Rodrigues Antão
  - "Advanced Python Programming" (free online)

---

## Environment Setup

### Conda Environment

The project uses a conda environment named `python_hpc` with Python 3.12.

```bash
pip install invoke        # install the task runner first
invoke create-env         # creates conda env named python_hpc
invoke install            # installs requirements.txt + requirements_project.txt
```

To activate manually: `conda activate python_hpc`

### Dependencies

`requirements.txt`: invoke, pytest, numpy, pandas, matplotlib, scipy, seaborn, cruft, ipykernel, simple-term-menu, questionary

`requirements_project.txt`: blosc

### Running Python Files Locally

```bash
invoke run-file --folder week7 --args "--flag value"
```

This presents an interactive menu of .py files in the given folder.

---

## HPC Workflow (DTU LSF Cluster)

Kyle's HPC username is `s252786`. The cluster uses LSF/BSUB for job scheduling.

### Transferring Files

```bash
# Download from HPC to local
invoke hpc-get --folder week5 --file-name output.csv

# Upload from local to HPC
invoke hpc-post --folder week5 --file-name script.py
```

These wrap `scp s252786@login.hpc.dtu.dk:Documents/python_hpc/<folder>/<file>`.

### Job Scripts

Each week folder typically has `.sh` job scripts using `#BSUB` directives. Examples:

- `week1/submit.sh`, `submit1.sh` ... `submit6.sh`
- `week2/submit.sh`
- `week5/mandelbrot_job.sh`, `pi.sh`
- `week3/cache.sh`, `blosc.sh`
- `week4/two_loop.sh`

To submit a job on the cluster: `bsub < submit.sh`

---

## Repository Structure

```
python_hpc/
├── CLAUDE.md                   # this file
├── README.md                   # setup instructions
├── tasks.py                    # invoke automation (create-env, install, hpc-get, hpc-post, run-file)
├── requirements.txt            # base Python dependencies
├── requirements_project.txt    # course-specific deps (blosc)
│
├── week1/                      # Intro: DTU HPC system, Linux basics, LSF/BSUB
├── week2/                      # Python basics + NumPy basics (timing, profiling)
├── week3/                      # Cache effects + Blosc compression
├── week4/                      # NumPy broadcasting & vectorization
├── week5/                      # Parallelism Part 1: multiprocessing, Amdahl's law, pi
├── week6/                      # Parallelism Part 2: reductions, shared memory, numactl
├── week7/                      # High-Performance Pandas + Apache Arrow
├── week8/                      # Memory-mapped files (numpy.memmap), shared memory
├── week9/                      # Numba (JIT) + GPU/CUDA computing
├── week10/                     # (check Lecture_week10.pdf — likely MPI or Dask)
├── week11/                     # HPC Workflows: job arrays, job dependencies
├── week12/                     # Course project exercises
├── week13/                     # HPC Pitfalls: common errors HPC users make
│
├── exam/                       # Past exams and answer keys
├── project/                    # Course project materials
├── summary/                    # Weekly summaries (may be empty or populated by agent sessions)
└── notes/                      # LaTeX notes (main.tex → main.pdf)
```

---

## Week Topics Quick Reference

| Week | Topic | Key Files |
|------|-------|-----------|
| 1 | Intro: DTU HPC, Linux, LSF/BSUB | content.py, submit*.sh |
| 2 | Python basics + NumPy (timing, profiling) | basic1-8.py, numpy1-6.py |
| 3 | Cache effects + Blosc compression | cache.py, blosc_ex.py, blosc_quiz.py |
| 4 | NumPy broadcasting & vectorization | bradscasting1-3.py, haversine.py |
| 5 | Parallelism Part 1: multiprocessing, Amdahl's law, pi | pi_serial.py, pi_parallel.py, pi_chunked.py, mandelbrot1-2.py |
| 6 | Parallelism Part 2: reductions, shared memory, numactl | reduction1.py, reduction_full.py, numactl_notes.md |
| 7 | High-Performance Pandas + Apache Arrow | panda_size.py, pyarrow_load.py, reduce_dataframe.py, 2023_01.csv |
| 8 | Memory-mapped files (numpy.memmap), shared memory | mandelbrotref.py |
| 9 | Numba (JIT) + GPU/CUDA computing | (exercises only) |
| 10 | (check Lecture_week10.pdf) | reduce.py |
| 11 | HPC Workflows: job arrays, job dependencies | (exercises only) |
| 12 | Course project exercises | project_exercises.html |
| 13 | HPC Pitfalls: common errors HPC users make | matmul.py |

---

## Navigating the Materials

Each `weekN/` folder typically contains:

- `Lecture_weekNN.pdf` — lecture slides (read in ≤20 page chunks using the `pages` parameter)
- `weekNN_exercises.pdf` and/or `weekNN_exercises.html` — exercise sheet
- `weekNN_solutions.pdf` and/or `weekNN_solutions.html` — official solutions
- `Quiz_weekNN.pdf` — quiz (some weeks)
- `.py` files — Python examples used in lectures or exercises
- `.sh` files — LSF job submission scripts

Some weeks share exercises across two weeks (e.g., week5 and week6 share `Week 5 & 6 Exercises.html`).

Week 10 exercises and solutions are zipped: `week10_exercises.html.zip`, `week10_solutions.html.zip`.

---

## Exam Folder

Located at `/Users/kyleelyk/Documents/DTU/SEM2/python_hpc/exam/`

| File | Description |
|------|-------------|
| `02613_exam_2024.pdf` | 2024 regular exam |
| `02613_exam_2024_answers.pdf` | 2024 regular exam answers |
| `02613_reexam_2024.pdf` | 2024 re-exam |
| `02613_reexam_2024_answers.pdf` | 2024 re-exam answers |
| `Exam_02613_F25.pdf` | 2025 exam (F25 = Spring 2025) |

All are PDFs. Read in ≤20 page chunks using the `pages` parameter.

---

## Notes (LaTeX)

Located at `/Users/kyleelyk/Documents/DTU/SEM2/python_hpc/notes/`

- `main.tex` — master LaTeX source
- `main.pdf` — compiled PDF of all notes

To compile: open `main.tex` in VS Code with the LaTeX Workshop extension; it auto-builds on save. Or run `latexmk -pdf main.tex` from the notes/ directory.

---

## Study Materials

| Location | Description |
|----------|-------------|
| `summary/` | Weekly summaries (may be created by agent sessions; currently empty) |
| `notes/main.pdf` | Compiled LaTeX notes covering course material |
| `exam/` | Past exams with answer keys for practice |
| `project/project_exercises.pdf` | Course project exercise sheet |

---

## Important Agent Notes

### Reading PDFs

All PDFs in this repo must be read using the `pages` parameter in ≤20 page chunks. Do NOT attempt to read an entire PDF in one call — it will fail for large files.

Example: to read pages 1-20 of a lecture:
```
Read file_path="/Users/kyleelyk/Documents/DTU/SEM2/python_hpc/week7/Lecture_week07.pdf" pages="1-20"
```

Then continue with `pages="21-40"`, etc.

### File Paths

Always use absolute paths. The repo root is:
```
/Users/kyleelyk/Documents/DTU/SEM2/python_hpc/
```

### HPC Username

Kyle's DTU HPC username is `s252786`. The login node is `login.hpc.dtu.dk`.

On the cluster, repo files are at `~/Documents/python_hpc/`.

### Conda Environment

The conda environment name is `python_hpc`. When running Python locally, check if the env is active with `echo $CONDA_DEFAULT_ENV`. If not, prefix commands with `conda run -n python_hpc`.

### Week 10 Topic

Week 10's lecture PDF (`Lecture_week10.pdf`) should be checked to confirm the topic. The `reduce.py` file is present, suggesting it may cover MPI or Dask reductions.
