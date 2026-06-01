# Week 2 — Python Bootcamp

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Overview](#overview)
- [Theory & Concepts](#theory-concepts)
  - [Why Python?](#why-python)
  - [The GIL: Global Interpreter Lock](#the-gil-global-interpreter-lock)
  - [HPC Cluster Workflow](#hpc-cluster-workflow)
  - [Computer Terminology](#computer-terminology)
  - [BSub Job Script Structure](#bsub-job-script-structure)
- [Key Python Patterns](#key-python-patterns)
  - [Built-in functions used in exercises](#built-in-functions-used-in-exercises)
  - [Classes](#classes)
  - [Command-line arguments pattern](#command-line-arguments-pattern)
  - [Deduplication via set](#deduplication-via-set)
  - [Sorting tuples by last element](#sorting-tuples-by-last-element)
- [Key NumPy Operations](#key-numpy-operations)
  - [Array creation](#array-creation)
  - [Vector magnitude (norm)](#vector-magnitude-norm)
  - [File I/O](#file-io)
  - [Axis-wise reductions](#axis-wise-reductions)
  - [Matrix power](#matrix-power)
- [Mathematical Content](#mathematical-content)
  - [Vector magnitude formula](#vector-magnitude-formula)
  - [Diagonal matrix](#diagonal-matrix)
  - [Column and row means](#column-and-row-means)
  - [Matrix power](#matrix-power)
- [Timing Code](#timing-code)
- [Key Code Examples](#key-code-examples)
  - [basic1.py — list sum using built-in](#basic1py-list-sum-using-built-in)
  - [basic2.py — deduplication via set conversion](#basic2py-deduplication-via-set-conversion)
  - [basic3.py — sort list of tuples by last element](#basic3py-sort-list-of-tuples-by-last-element)
  - [numpy1.py — vector magnitude (function form)](#numpy1py-vector-magnitude-function-form)
  - [numpy2.py — vector magnitude from command line](#numpy2py-vector-magnitude-from-command-line)
  - [numpy3.py — build and save a diagonal matrix](#numpy3py-build-and-save-a-diagonal-matrix)
  - [numpy4.py — column and row means](#numpy4py-column-and-row-means)
  - [numpy5.py — timed matrix power](#numpy5py-timed-matrix-power)
- [Exercise Highlights](#exercise-highlights)
  - [Section 1: Basic Python (8 exercises)](#section-1-basic-python-8-exercises)
  - [Section 2: Basic NumPy (6 exercises)](#section-2-basic-numpy-6-exercises)
- [Key Takeaways](#key-takeaways)

---

## Overview

Week 2 is a Python bootcamp covering two main topics: core Python programming patterns (data structures, functions, classes, command-line programs) and an introduction to NumPy for numerical computing. The lecture also covers the HPC cluster workflow — login nodes, interactive sessions, and batch jobs — and explains *why* Python is used in HPC despite its performance limitations. Exam date: 1 June 2026.

---

## Theory & Concepts

### Why Python?

Python is slow compared to C/C++ (benchmarks show Python can be 100-180x slower on compute-heavy tasks), and it is not naturally parallel due to the GIL. Yet it is used in HPC because:

- It is the most popular language in scientific computing (TIOBE index, dominant since ~2018)
- It has a world-class ecosystem: NumPy, SciPy, PyTorch, TensorFlow, JAX, scikit-learn, FEniCS
- It offers speed vs. agility — you prototype fast and offload hot paths to C-backed libraries
- Many operations (NumPy, I/O) release the GIL and can run concurrently

### The GIL: Global Interpreter Lock

The GIL is a mutex inside CPython that allows only one thread to execute Python bytecode at a time.

- **Pro:** eliminates race conditions automatically; some operations (I/O, NumPy internals) release it
- **Con:** compute-heavy multi-threading is effectively impossible in pure Python

Race condition example from the lecture: if Thread 1 does `a = a + 2` and Thread 2 does `a = a + 3` concurrently starting from `a = 2`, the result could be 4, 5, or 7 depending on scheduling — a classic race condition. The GIL serialises these operations so only one thread holds it at a time, making the final result deterministic (7), but at the cost of parallelism.

Workarounds: use **multi-processing** (separate processes, no shared GIL) or escape to compiled languages. This is covered in depth in Weeks 6 and 7. Note: PEP 703 (accepted, Python 3.13+) makes the GIL optional via free-threaded CPython.

### HPC Cluster Workflow

The DTU HPC cluster has four login nodes:

```
ssh <username>@login.hpc.dtu.dk
ssh <username>@login2.hpc.dtu.dk
ssh <username>@login.gbar.dtu.dk
ssh <username>@login2.gbar.dtu.dk
```

**Critical rule: login nodes are NOT for computation.** Use them only for editing and submitting jobs.

- `linuxsh` — interactive session for testing/debugging
- `bsub` — submit batch jobs for real computation

The VS Code Run/Debug button executes on the login node, not on a compute node. Always run Python via the terminal after `linuxsh`.

### Computer Terminology

| Component | Role |
|-----------|------|
| CPU | Performs computations; contains multiple independent cores |
| RAM | Temporary working memory — where the CPU "works from" |
| HDD/SSD | Permanent file storage |
| GPU | (covered in Week 9) |

### BSub Job Script Structure

```bash
#!/bin/bash
#BSUB -J sleeper          # Job name
#BSUB -q hpc              # Queue name
#BSUB -W 2                # Wall-clock time limit in minutes (2 = 2 minutes)
#BSUB -R "rusage[mem=512MB]"  # Memory request
#BSUB -n 4                # Number of cores
#BSUB -R "span[hosts=1]"  # Keep cores on one host
#BSUB -o sleeper_%J.out   # stdout file
#BSUB -e sleeper_%J.err   # stderr file

sleep 60
```

---

## Key Python Patterns

### Built-in functions used in exercises

| Pattern | Example |
|---------|---------|
| `sum()` | Sum a list of numbers |
| `set()` | Remove duplicates from a list |
| `sorted(iterable, key=fn)` | Sort by a custom key |
| `filter(fn, iterable)` | Filter elements |
| `sys.argv` | Access command-line arguments |
| List comprehension | `[x**2 for x in lst]` |
| Lambda | `lambda t: t[1]` — sort key on second element |

### Classes

```python
class Student:
    def __init__(self, name, courses):
        self.name = name
        self.courses = courses

    def attends(self, course):
        return course in self.courses
```

Usage:
```python
s = Student('Alice', ['01005', '02613'])
s.attends('02613')  # True
s.attends('02510')  # False
```

### Command-line arguments pattern

```python
import sys

# sys.argv[0] is the script name, sys.argv[1:] are the arguments
grades = list(map(float, sys.argv[1:]))
mean = sum(grades) / len(grades)
print(f"{mean} {'Pass' if mean >= 5 else 'Fail'}")
```

### Deduplication via set

```python
def deduplicate(arr):
    return list(set(arr))
```

Sets have O(1) average lookup and automatically eliminate duplicates. Note: order is not preserved.

### Sorting tuples by last element

```python
def sorttuples(arr):
    return sorted(arr, key=lambda t: t[-1])
# [(2,5),(1,2),(4,4),(2,3),(2,1)] -> [(2,1),(1,2),(2,3),(4,4),(2,5)]
```

---

## Key NumPy Operations

### Array creation

```python
import numpy as np

v = np.array([1, 1, 3, 3, 4])           # from list
e = np.array(sys.argv[1:]).astype(float) # from CLI args, cast to float
D = np.diag(e)                           # diagonal matrix from vector
```

### Vector magnitude (norm)

```python
np.linalg.norm(v)   # Euclidean norm: sqrt(sum(x_i^2))
```

For `v = [1, 1, 3, 3, 4]`: magnitude = sqrt(1+1+9+9+16) = sqrt(36) = 6

### File I/O

```python
np.save("output.npy", array)     # save array to binary .npy file
M = np.load("input.npy")        # load array from .npy file
```

### Axis-wise reductions

```python
np.mean(M, axis=0)   # column means (one value per column)
np.mean(M, axis=1)   # row means (one value per row)
```

For a 3x4 matrix:
- `axis=0` produces an m-dimensional vector (column means)
- `axis=1` produces an n-dimensional vector (row means)

### Matrix power

```python
np.linalg.matrix_power(A, p + 1)   # compute A^(p+1)
```

Example: A = [[1,3],[5,7]], p=2 → A^3 = [[136,216],[360,568]]

---

## Mathematical Content

### Vector magnitude formula

For an n-dimensional vector v = [v_1, v_2, ..., v_n]:

```
||v|| = sqrt(v_1^2 + v_2^2 + ... + v_n^2)
```

Implemented as `np.linalg.norm(v)`.

### Diagonal matrix

Given a vector `e = [e_1, e_2, ..., e_n]`, `np.diag(e)` produces an n×n matrix with `e` on the main diagonal and zeros elsewhere.

### Column and row means

For an n×m matrix M:
- Column mean vector (length m): `np.mean(M, axis=0)`
- Row mean vector (length n): `np.mean(M, axis=1)`

### Matrix power

`np.linalg.matrix_power(A, p)` computes A multiplied by itself p times. This is distinct from element-wise power (`**`).

---

## Timing Code

Use `perf_counter` from the `time` module for high-resolution wall-clock timing:

```python
from time import perf_counter

start = perf_counter()
# ... computation ...
elapsed = perf_counter() - start
print(f"{elapsed}")   # time in seconds
```

`perf_counter` returns a float in seconds with the highest available resolution. It is preferred over `time.time()` for benchmarking because it is not affected by system clock adjustments.

Note: the numpy5.py solution has a bug — it computes `start - end` instead of `end - start`, which will print a negative number. The correct form is `end - start`.

---

## Key Code Examples

### basic1.py — list sum using built-in

```python
def listsum(arr: list[int | float]) -> int | float:
    return sum(arr)
```

Uses Python's built-in `sum()`. The type hint `int | float` (union type, Python 3.10+) documents that the list can contain mixed numeric types.

### basic2.py — deduplication via set conversion

```python
def deduplicate(arr):
    return list(set(arr))
```

### basic3.py — sort list of tuples by last element

```python
def sorttuples(arr):
    return sorted(arr, key=lambda t: t[1])
```

The `key` parameter takes a function applied to each element before comparison. `lambda t: t[1]` extracts the second (index 1) element of each tuple.

### numpy1.py — vector magnitude (function form)

```python
import numpy as np

def magnitude(v):
    return np.linalg.norm(v)
```

### numpy2.py — vector magnitude from command line

```python
import numpy as np
import sys

def magnitude():
    v = np.array(sys.argv[1:]).astype(float)
    return np.linalg.norm(v)

if __name__ == "__main__":
    print(f"{magnitude()}")
```

`sys.argv[1:]` gives all CLI arguments as strings. `.astype(float)` converts the string array to float64 in one vectorised call.

### numpy3.py — build and save a diagonal matrix

```python
import numpy as np
import sys

def save_diag():
    e = np.array(sys.argv[1:]).astype(float)
    np.save("saved.npy", np.diag(e))

if __name__ == "__main__":
    save_diag()
```

`np.diag(e)` when given a 1D array returns a 2D diagonal matrix.

### numpy4.py — column and row means

```python
import numpy as np
import sys

def save_mean():
    M = np.load(sys.argv[1])
    np.save("cols.npy", np.mean(M, axis=0))
    np.save("rows.npy", np.mean(M, axis=1))

if __name__ == "__main__":
    save_mean()
```

### numpy5.py — timed matrix power

```python
import numpy as np
import sys
from time import perf_counter

def save_mean():
    A, p = np.load(sys.argv[1]), int(sys.argv[2])
    start = perf_counter()
    np.save("saved.npy", np.linalg.matrix_power(A, p + 1))
    end = perf_counter()
    print(f"{end - start}")   # note: source has start-end (bug)

if __name__ == "__main__":
    save_mean()
```

---

## Exercise Highlights

### Section 1: Basic Python (8 exercises)

| Exercise | Function | Core technique |
|----------|----------|---------------|
| basic1 | `listsum` | `sum()` builtin |
| basic2 | `deduplicate` | `set()` conversion |
| basic3 | `sorttuples` | `sorted(key=lambda)` |
| basic4 | `squarecubes` | Return tuple of two lists (squares and cubes) |
| basic5 | Grade mean program | `sys.argv`, conditional output (`Pass`/`Fail`) |
| basic6 | Filter even numbers | `filter()` or list comprehension with `% 2 == 0` |
| basic7 | `Student` class | `__init__`, `attends` method, `in` operator |
| basic8 | `coursestudents` | Filter students by course using `attends` method |

### Section 2: Basic NumPy (6 exercises)

| Exercise | Task | Key NumPy call |
|----------|------|---------------|
| numpy1 | Vector magnitude (function) | `np.linalg.norm(v)` |
| numpy2 | Vector magnitude (program, CLI input) | `sys.argv`, `.astype(float)` |
| numpy3 | Build diagonal matrix, save to .npy | `np.diag()`, `np.save()` |
| numpy4 | Column and row means of a matrix | `np.mean(M, axis=0/1)` |
| numpy5 | Matrix power with timing | `np.linalg.matrix_power()`, `perf_counter` |
| numpy6 | Submit numpy5 as a batch job | `bsub` script with conda activation |

Exercise numpy6 is the first batch job submission exercise: wrap the Python script in a BSub script, activate the 02613 conda environment, request 1 core in the `hpc` queue, and submit with `bsub < submit.sh`. The Autolab handin is a zip of both the Python script and the shell script.

---

## Key Takeaways

1. **Python is slow but strategic.** Pure Python is 100x+ slower than C for compute-heavy loops. The answer is not to avoid Python, but to push computation into NumPy (which calls optimised C/Fortran under the hood).

2. **The GIL prevents true multi-threading in Python.** Only one thread executes Python bytecode at a time. For parallelism, use multi-processing or NumPy operations (which release the GIL). This will be revisited in Weeks 6 and 7.

3. **Login nodes are for submission, not computation.** Always use `linuxsh` for interactive testing and `bsub` for real jobs. Running compute on login nodes is bad cluster etiquette.

4. **`sys.argv` is how Python programs accept runtime inputs.** `sys.argv[0]` is the script name; `sys.argv[1:]` are the user-supplied arguments (always strings — cast with `.astype(float)` or `float()`).

5. **NumPy axis semantics:** `axis=0` reduces along rows (gives column-wise result); `axis=1` reduces along columns (gives row-wise result).

6. **`np.diag(v)` is a two-way function:** given a 1D array it produces a diagonal matrix; given a 2D matrix it extracts the diagonal vector.

7. **`perf_counter` is the right tool for timing.** Always capture `start` before and `end` after the operation, then print `end - start`.

8. **BSub scripts must specify:** job name (`-J`), queue (`-q`), wall time (`-W`), memory (`-R rusage`), core count (`-n`), and output files (`-o`, `-e`). Always activate the conda environment inside the script before calling Python.
