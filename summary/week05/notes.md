# Week 5 — Parallelism Part 1: Multiprocessing & the GIL

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Overview](#overview)
- [Theory & Concepts](#theory-concepts)
  - [Why Parallelism?](#why-parallelism)
  - [What Is Parallelism?](#what-is-parallelism)
  - [Amdahl's Law](#amdahls-law)
  - [Types of Parallelism](#types-of-parallelism)
- [Threads vs. Processes](#threads-vs-processes)
  - [Process](#process)
  - [Thread](#thread)
  - [The GIL: Global Interpreter Lock](#the-gil-global-interpreter-lock)
- [Python Multiprocessing](#python-multiprocessing)
  - [Pool API](#pool-api)
- [Key Code Examples](#key-code-examples)
  - [pi_serial.py — Baseline Serial Monte Carlo](#pi_serialpy-baseline-serial-monte-carlo)
  - [pi_parallel.py — Naive Fully Parallel (one task per sample)](#pi_parallelpy-naive-fully-parallel-one-task-per-sample)
  - [pi_chunked.py — Correct Chunked Parallel (one task per process)](#pi_chunkedpy-correct-chunked-parallel-one-task-per-process)
- [Mandelbrot Parallelization](#mandelbrot-parallelization)
  - [mandelbrot1.py — Pool.map with equal distribution](#mandelbrot1py-poolmap-with-equal-distribution)
  - [mandelbrot2.py — Context manager + timing + CLI argument](#mandelbrot2py-context-manager-timing-cli-argument)
- [Exercise Highlights](#exercise-highlights)
  - [Exercise 1 — Amdahl's Law (analytical)](#exercise-1-amdahls-law-analytical)
  - [Exercise 2 — Parallel Pi](#exercise-2-parallel-pi)
  - [Exercise 3 — Mandelbrot Set](#exercise-3-mandelbrot-set)
- [Key Takeaways](#key-takeaways)

---

## Overview

Week 5 introduces parallelism as the primary strategy for squeezing more performance out of modern hardware. Single-core clock speeds plateaued around 2004-2005; processor vendors responded by adding more cores rather than pushing higher frequencies. The lecture covers three interlocking topics: why parallelism matters, how Python's threading model works (and why the GIL blocks naive parallelism), and how to write correct parallel code using `multiprocessing`. The exercises apply these ideas to two classic HPC benchmarks: Monte Carlo pi estimation and the Mandelbrot set.

---

## Theory & Concepts

### Why Parallelism?

From the 50 Years of Microprocessor Trend Data chart (K. Rupp):
- Transistor counts continue growing exponentially (Moore's Law holds)
- Clock frequency (MHz) plateaued around 2004 and has stayed flat
- Single-thread performance is still improving, but slowly
- Number of logical cores has been growing rapidly since ~2005
- There is also a large and growing gap between processor speed and memory speed (the "memory wall")

The moral from the lecture: **parallelism is the name of the game**. Modern servers (shown: 48-core machine) sit largely idle because serial code uses only one core at a time.

### What Is Parallelism?

Parallelism = performing several operations *simultaneously*.

Classic data-parallelism example — multiply every element of an array `x` by 2:
- 1 processor: N time units
- 2 processors: N/2 time units (each handles half the array)
- 4 processors: N/4 time units

This is called **embarrassingly parallel**: every unit of work is independent, so everything can be parallelized without coordination.

**Speedup** is defined using wall-clock time:

```
S(n) = T(1) / T(n)
```

where `T(1)` is the single-process wall-clock time and `T(n)` is the wall-clock time with `n` processors.

### Amdahl's Law

Not all programs are embarrassingly parallel. If a fraction `s` of a program is inherently serial:

```
S(n) = 1 / (s + (1 - s) / n)
```

where:
- `S(n)` = speedup with n processors
- `s` = serial fraction (0 to 1)
- `1 - s` = parallel fraction
- `n` = number of processors

Key insights:
- Adding more processors gives **diminishing returns**
- As n approaches infinity: `S(∞) = 1/s`
- If 10% of a program is serial, max speedup is 10x regardless of core count
- Optimizing the serial part can matter more than buying more processors

**Efficiency** measures how well processors are utilized:

```
E(n) = S(n) / n    (ideal = 1.0, i.e. linear scaling)
```

### Types of Parallelism

- **Task parallelism**: different tasks run on different processors
- **Data parallelism**: the same task runs on different chunks of data (most common in HPC)

---

## Threads vs. Processes

### Process

A process is everything needed to execute a program:
- Its own memory address space
- Handles to OS objects
- At least one thread of execution

### Thread

A thread is an entity *within* a process scheduled for execution:
- Thread-local storage and execution state
- Access to the **shared** process memory
- Multiple threads in one process can run in parallel on multiple cores

Shared memory is powerful but dangerous: two threads reading and writing the same variable simultaneously produce a **race condition** — the result depends on the exact timing of execution. Example from the lecture:

```
# Main sets a = 2
# Thread 1: a = a + 2   (reads 2, writes 4)
# Thread 2: a = a + 3   (reads 2, writes 5 -- reads BEFORE Thread 1 writes)
# Main prints a  -->  could print 4, 5, or 7 depending on timing
```

### The GIL: Global Interpreter Lock

CPython (the standard Python interpreter) has a GIL — a mutex that ensures only **one thread executes Python bytecode at a time**. This serializes all Python-level operations.

- **Pro**: eliminates race conditions for Python objects — convenient, safe
- **Con**: compute-heavy multi-threading is impossible; threads take turns rather than running in parallel

However, some operations manually **release** the GIL while executing C-level code:
- File reading / I/O operations
- **NumPy** array operations (the key exception for HPC)

This means `ThreadPool` with NumPy operations *can* achieve real parallelism — the GIL is released during `np.sum()`, allowing multiple threads to overlap.

**Lecture benchmark — NumPy ThreadPool** (`sums.py`, `np.zeros((1024,1024,1024))`):

| Threads | real time | Speedup |
|---------|-----------|---------|
| 1       | 3.634s    | 1.0x    |
| 2       | 1.873s    | 1.94x   |
| 4       | 1.136s    | 3.20x   |

NumPy releases the GIL, so `ThreadPool` works. But with a pure Python `manual_sum` (Python loop), adding threads gives **no speedup** — the GIL serializes them:

| Threads | real time | Note |
|---------|-----------|------|
| 1       | 1.916s    | baseline |
| 2       | 1.855s    | no improvement |
| 4       | 3.254s    | **slower** (thread management overhead) |

---

## Python Multiprocessing

`multiprocessing` spawns separate **processes** rather than threads. Each process has its own Python interpreter and memory space, so there is no shared GIL. This is the standard way to achieve CPU-bound parallelism in Python.

### Pool API

```python
from multiprocessing import Pool

# pool.map — applies function to each element of iterable
with Pool(n_proc) as pool:
    results = pool.map(func, iterable)

# pool.map with chunksize — controls how many items are sent to each worker at once
with Pool(n_proc) as pool:
    results = pool.map(func, iterable, chunksize=100)

# pool.apply_async — submit individual tasks asynchronously
pool = multiprocessing.Pool(n_proc)
futures = [pool.apply_async(func, (arg,)) for arg in args]
results = [f.get() for f in futures]

# ThreadPool — same API but uses threads instead of processes
from multiprocessing.pool import ThreadPool
with ThreadPool(n_threads) as pool:
    results = pool.map(func, iterable)
```

The `with` statement context manager automatically calls `pool.close()` and `pool.join()`. Manually: call `pool.close()` (no more tasks) then `pool.join()` (wait for completion).

**Important**: the `if __name__ == '__main__':` guard is required on Windows/macOS to prevent infinite spawning when child processes import the script.

---

## Key Code Examples

### pi_serial.py — Baseline Serial Monte Carlo

```python
import random

samples = 1000000
hits = 0

for i in range(samples):
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)
    if x**2 + y**2 <= 1:
        hits += 1

pi = 4.0 * hits/samples
```

### pi_parallel.py — Naive Fully Parallel (one task per sample)

```python
import random
import multiprocessing

def sample():
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)
    if x**2 + y**2 <= 1:
        return 1
    else:
        return 0

if __name__ == '__main__':
    samples = 1000000
    hits = 0

    n_proc = 10
    pool = multiprocessing.Pool(n_proc)
    results_async = [pool.apply_async(sample) for i in range(samples)]
    hits = sum(r.get() for r in results_async)
    pi = 4.0 * hits/samples
```

This is actually **slower** than serial: spawning 1,000,000 tasks means 1,000,000 inter-process messages. Process overhead completely dominates the trivial computation per task.

### pi_chunked.py — Correct Chunked Parallel (one task per process)

```python
import random
import multiprocessing

def sample():
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)
    if x**2 + y**2 <= 1:
        return 1
    else:
        return 0

def sample_multiple(samples_partial):
    return sum(sample() for i in range(samples_partial))

if __name__ == '__main__':
    samples = 1000000
    hits = 0

    n_proc = 10
    chunk_size = samples // n_proc
    pool = multiprocessing.Pool(n_proc)
    results_async = [pool.apply_async(sample_multiple, (chunk_size,))
                    for i in range(n_proc)]
    hits = sum(r.get() for r in results_async)
    pi = 4.0 * hits/samples
```

Each process handles `chunk_size = 100,000` samples. Only 10 inter-process messages are sent. This is the fast version.

**Key lesson**: task granularity matters. Too-fine-grained parallelism has more overhead than benefit. Chunk the work so each task does a meaningful amount.

---

## Mandelbrot Parallelization

The Mandelbrot set is defined by iterating `z_{n+1} = z_n^2 + c` for complex number `c`. A point belongs to the set if the iteration stays bounded; the **escape time** (iterations until |z| > 2) is used to color the image.

Each pixel's escape time is **independent** of all others — embarrassingly parallel.

### mandelbrot1.py — Pool.map with equal distribution

```python
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_escape_time(c):
    z = 0
    for i in range(100):
        z = z**2 + c
        if np.abs(z) > 2.0:
            return i
    return 100

def generate_mandelbrot_set(points, num_processes):
    chunk_size = len(points) // num_processes
    pool = multiprocessing.Pool(num_processes)
    escape_times = pool.map(mandelbrot_escape_time, points, chunksize=chunk_size)
    pool.close()
    pool.join()
    return np.array(escape_times)

if __name__ == "__main__":
    width = 800
    height = 800
    num_proc = 4

    x_values = np.linspace(-2, 2, width)
    y_values = np.linspace(-2, 2, height)
    points = np.array([complex(x, y) for x in x_values for y in y_values])

    mandelbrot_set = generate_mandelbrot_set(points, num_proc)
    mandelbrot_set = mandelbrot_set.reshape((height, width))
    plot_mandelbrot(mandelbrot_set)
```

### mandelbrot2.py — Context manager + timing + CLI argument

```python
import multiprocessing, sys, time
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_escape_time(c):
    z = 0
    for i in range(100):
        z = z**2 + c
        if np.abs(z) > 2.0:
            return i
    return 100

def generate_mandelbrot_set_chunks(points, num_processes):
    chunk_size = 100  # fine-grained chunk for better load balancing
    with multiprocessing.Pool(num_processes) as pool:
        escape_times = pool.map(mandelbrot_escape_time, points, chunksize=chunk_size)
    return np.array(escape_times)

if __name__ == "__main__":
    num_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    points = np.array([complex(x, y)
                       for x in np.linspace(-2, 2, 800)
                       for y in np.linspace(-2, 2, 800)])

    t0 = time.time()
    mandelbrot_set = generate_mandelbrot_set_chunks(points, num_proc)
    elapsed = time.time() - t0

    print(f"{num_proc},{elapsed:.4f}")  # CSV output for speedup plotting
```

Key differences between v1 and v2:
- v2 uses `with Pool(...) as pool:` (preferred, automatic cleanup)
- v2 uses `chunksize=100` instead of `len(points)//num_processes` — smaller chunks improve **load balancing** because Mandelbrot points near the set boundary take more iterations than points far away. Equal-sized chunks by index can leave some processes idle.
- v2 accepts `num_proc` from `sys.argv`, enabling shell loops for speedup measurements
- v2 times the parallel section and prints CSV for plotting

---

## Exercise Highlights

### Exercise 1 — Amdahl's Law (analytical)

Given: serial subtask = 20s, parallel subtask = 100s (total 120s).

- Parallel fraction p = 100/120 = 0.833
- Speedup with 10 processors: S(10) = 1 / (0.167 + 0.833/10) = 1 / (0.167 + 0.083) = 1 / 0.25 = 4.0x
- Maximum theoretical speedup: S(inf) = 1 / 0.167 = 6.0x
- Compare options at 4 processors: reducing serial time to 5s vs. 8 processors — requires computing both runtimes to decide

### Exercise 2 — Parallel Pi

Run all three implementations with `time` on the cluster:
- Implementation 1 (serial): baseline
- Implementation 2 (naive parallel): **slower than serial** — 1M task spawns dominates
- Implementation 3 (chunked): fastest — only 10 inter-process messages

Then vary `n_proc` from 1 to max threads (`lscpu`) and plot speedup vs. n_proc. Fit Amdahl's law to estimate the parallel fraction and theoretical maximum speedup.

Optional: rewrite using `pool.map()` instead of `apply_async`.

### Exercise 3 — Mandelbrot Set

Implement `generate_mandelbrot_set` to distribute the 800x800 = 640,000 points across processes using `pool.map`. Run with varying `num_proc`, measure wall-clock time, produce a speedup plot. Observe that the Mandelbrot set has **load imbalance**: boundary points take more iterations than interior points, so naive equal distribution may leave some workers idle sooner.

---

## Key Takeaways

1. **Single-core performance is stagnant** — to go faster you must go wider (more cores).

2. **Python threads cannot run Python code in parallel** due to the GIL. For CPU-bound work, use `multiprocessing` (separate processes, each with its own GIL).

3. **NumPy releases the GIL**, so `ThreadPool` works for NumPy-heavy computations. Pure Python loops in threads see no benefit and can even slow down.

4. **Granularity is critical**: spawning one task per unit of work (e.g., one task per Monte Carlo sample) is catastrophically slow due to inter-process communication overhead. Chunk the work to match the number of processes.

5. **Amdahl's Law sets a hard ceiling**: if 10% of your code is serial, you can never exceed 10x speedup no matter how many cores you add. Reducing the serial fraction often matters more than adding processors.

6. **Embarrassingly parallel** problems (Mandelbrot, Monte Carlo, image processing) are the ideal case for `multiprocessing` — no inter-process communication during computation, only at the end to collect results.

7. **Measure with wall-clock time** (`time` command, `real` output) for speedup calculations, not CPU time. With parallelism, CPU time can exceed wall-clock time (multiple CPUs working simultaneously).

8. **Load balancing**: if different work items take different amounts of time (like Mandelbrot boundary vs. interior), use smaller `chunksize` in `pool.map` so that fast workers pick up more tasks dynamically rather than waiting for slow workers to finish large equal-sized chunks.
