# Week 5 Exercises — Parallelism Part 1: Multiprocessing, Amdahl's Law, Pi

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Section 1: Amdahl's Law](#section-1-amdahls-law)
- [Exercise 1.1 `[AUTOLAB]`](#exercise-11-autolab)
- [Section 2: Parallel π](#section-2-parallel-π)
- [Exercise 2.1 `[AUTOLAB]`](#exercise-21-autolab)
- [Exercise 2.2 `[AUTOLAB]`](#exercise-22-autolab)
- [Exercise 2.3 `[PRACTICE]`](#exercise-23-practice)
- [Exercise 2.4 `[PRACTICE]`](#exercise-24-practice)
- [Exercise 2.5 `[PRACTICE]` (Optional)](#exercise-25-practice-optional)
- [Section 3: The Mandelbrot Set](#section-3-the-mandelbrot-set)
- [Exercise 3.1 `[AUTOLAB]`](#exercise-31-autolab)
- [Exercise 3.2 `[AUTOLAB]`](#exercise-32-autolab)
- [Exercise 3.3 `[AUTOLAB]`](#exercise-33-autolab)
- [Exercise 3.4 `[PRACTICE]`](#exercise-34-practice)

---

---

## Section 1: Amdahl's Law

## Exercise 1.1 `[AUTOLAB]`

Suppose that a specific task consists of two subtasks. The first subtask retrieves data from a file; it takes 20 seconds to execute on a given system, and it cannot be parallelized. The second subtask processes a large number of data records; it takes 100 seconds to execute sequentially. Each data record can be processed independently, so the second subtask is easily parallelized.

**a.** What is the parallel fraction of the task consisting of the two subtasks?

> **Solution:**
>
> Total time = 20 + 100 = 120 seconds. The parallelizable part is 100 seconds.
>
> Parallel fraction: **p = 100 / 120 = 5/6 ≈ 0.833**

**b.** What is the theoretical speedup for p = 10 processors?

> **Solution:**
>
> Amdahl's Law: S(n) = 1 / ((1 - p) + p/n)
>
> S(10) = 1 / ((1 - 5/6) + (5/6)/10) = 1 / (1/6 + 1/12) = 1 / (3/12) = 1 / (1/4) = **4.0**
>
> Equivalently: S(10) = 1 / (0.1667 + 0.0833) = 1 / 0.25 = 4.0

**c.** What is the theoretical maximum speedup?

> **Solution:**
>
> As n → ∞: S_max = 1 / (1 - p) = 1 / (1/6) = **6.0**

**d.** Imagine we currently have 4 processors available for parallelization. To improve performance, we have two options:
1. Optimize the first subtask so we can retrieve the data from the file in 5 seconds instead of 20.
2. Purchase more processors so we have 8 processors available instead of 4.

Which option will provide the shortest runtime?

> **Solution:**
>
> **Current runtime with 4 processors:**
> S(4) = 1 / (1/6 + (5/6)/4) = 1 / (0.1667 + 0.2083) = 1 / 0.375 = 2.667
> Runtime = 120 / 2.667 ≈ 45 seconds
>
> **Option A — Optimize serial part (5s instead of 20s):**
> New total sequential time = 5 + 100 = 105s. New parallel fraction = 100/105 ≈ 0.952.
> S(4) = 1 / (0.048 + 0.952/4) = 1 / (0.048 + 0.238) = 1 / 0.286 ≈ 3.5
> Runtime = 105 / 3.5 = **30 seconds**
>
> **Option B — 8 processors:**
> S(8) = 1 / (1/6 + (5/6)/8) = 1 / (0.1667 + 0.1042) = 1 / 0.2708 ≈ 3.69
> Runtime = 120 / 3.69 ≈ **32.5 seconds**
>
> **Option A (optimizing the serial subtask) gives the shorter runtime (~30s vs ~32.5s).** This illustrates that reducing the serial fraction is often more impactful than adding more processors.

---

## Section 2: Parallel π

The goal of this exercise is to use parallelization to improve the performance of our code. We will use the Monte Carlo approximation of pi from Advanced Python Programming, Chapter 7. The three implementations are:

**Implementation 1: Fully serial**
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

**Implementation 2: Fully parallel**
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

**Implementation 3: Chunked parallel**
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
    chunk_size = samples//n_proc
    pool = multiprocessing.Pool(n_proc)
    results_async = [pool.apply_async(sample_multiple, (chunk_size,))
                    for i in range(n_proc)]
    hits = sum(r.get() for r in results_async)
    pi = 4.0 * hits/samples
```

## Exercise 2.1 `[AUTOLAB]`

Run the three implementations as a batch job and measure their run time using the `time` command. Remember to select a CPU model for repeatable results and to select multiple cores.

> **Solution:**
>
> Submit each of the three scripts (`pi_serial.py`, `pi_parallel.py`, `pi_chunked.py`) as an LSF batch job using `bsub`. Wrap each execution with the `time` command to capture wall-clock, user, and system time. Example job script snippet:
>
> ```bash
> #BSUB -q hpc
> #BSUB -n 10
> #BSUB -R "span[hosts=1]"
> #BSUB -R "select[model == XeonGold6226R]"
> #BSUB -W 00:10
> source /dtu/sw/dcc/dcc_rc
> conda activate python_hpc
> time python pi_serial.py
> time python pi_parallel.py
> time python pi_chunked.py
> ```
>
> Student files:
>
> `pi_serial.py`:
> ```python
> import random
>
> samples = 1000000
> hits = 0
>
> for i in range(samples):
>     x = random.uniform(-1.0, 1.0)
>     y = random.uniform(-1.0, 1.0)
>     if x**2 + y**2 <= 1:
>         hits += 1
>
> pi = 4.0 * hits/samples
> ```
>
> `pi_parallel.py`:
> ```python
> import random
> import multiprocessing
>
> def sample():
>     x = random.uniform(-1.0, 1.0)
>     y = random.uniform(-1.0, 1.0)
>     if x**2 + y**2 <= 1:
>         return 1
>     else:
>         return 0
>
> if __name__ == '__main__':
>     samples = 1000000
>     hits = 0
>
>     n_proc = 10
>     pool = multiprocessing.Pool(n_proc)
>     results_async = [pool.apply_async(sample) for i in range(samples)]
>     hits = sum(r.get() for r in results_async)
>     pi = 4.0 * hits/samples
> ```
>
> `pi_chunked.py`:
> ```python
> import random
> import multiprocessing
>
> def sample():
>     x = random.uniform(-1.0, 1.0)
>     y = random.uniform(-1.0, 1.0)
>     if x**2 + y**2 <= 1:
>         return 1
>     else:
>         return 0
>
> def sample_multiple(samples_partial):
>     return sum(sample() for i in range(samples_partial))
>
> if __name__ == '__main__':
>     samples = 1000000
>     hits = 0
>
>     n_proc = 10
>     chunk_size = samples//n_proc
>     pool = multiprocessing.Pool(n_proc)
>     results_async = [pool.apply_async(sample_multiple, (chunk_size,))
>                     for i in range(n_proc)]
>     hits = sum(r.get() for r in results_async)
>     pi = 4.0 * hits/samples
> ```

## Exercise 2.2 `[AUTOLAB]`

Which implementation was fastest? Did some perform slower than expected? Why? Can you relate it to the output of the `time` command?

> **Solution:**
>
> - **Implementation 3 (chunked parallel)** is the fastest. It launches only `n_proc` tasks with substantial work each, so inter-process communication overhead is minimized.
> - **Implementation 2 (fully parallel)** is often *slower than serial* despite using 10 processes. It launches 1,000,000 separate tasks — one per sample — causing massive process-pool overhead. The `time` output will show very high `sys` time relative to `user` time, indicating the bottleneck is IPC and scheduling, not computation.
> - **Implementation 1 (serial)** outperforms Implementation 2 because it has zero parallelism overhead.
> - Key insight: process pool overhead dominates when the per-task work is tiny. Chunking aggregates enough work per task that the parallel speedup outweighs the overhead.

## Exercise 2.3 `[PRACTICE]`

Run the fastest parallel implementation for `n_proc` varying from 1 to the maximum number of threads on your CPU. Plot the speedup as a function of the number of processes.

Hint: Use the `lscpu` command to check the number of threads available.

> **Solution:**
>
> Use `lscpu` on the cluster to find the number of available threads (e.g., a Xeon Gold 6226R has 32 threads per node with hyperthreading). Run `pi_chunked.py` for `n_proc` in {1, 2, 4, 8, 16, 32}, record wall time for each, then compute speedup = T(1) / T(n). Plot measured speedup vs. n_proc alongside the ideal linear speedup line.
>
> On a Xeon Gold 6226R, the speedup curve flattens well before the maximum thread count, reflecting that the parallel fraction is less than 1 (Amdahl's Law).
>
> The `plot_speedup.py` helper script can automate the plot from a `timing_results.csv` file:
> ```python
> import numpy as np
> import matplotlib.pyplot as plt
>
> data = np.loadtxt("timing_results.csv", delimiter=",", skiprows=1)
> num_procs = data[:, 0].astype(int)
> times = data[:, 1]
>
> t1 = times[num_procs == 1][0]
> speedup = t1 / times
>
> plt.figure()
> plt.plot(num_procs, speedup, marker='o', label='Measured speedup')
> plt.plot(num_procs, num_procs, linestyle='--', label='Ideal speedup')
> plt.xlabel("Number of processes")
> plt.ylabel("Speedup")
> plt.title("Mandelbrot parallel speedup")
> plt.legend()
> plt.xticks(num_procs)
> plt.tight_layout()
> plt.savefig("speedup.png")
> ```

## Exercise 2.4 `[PRACTICE]`

Estimate the parallel fraction of the program by fitting the theoretical speedup curve from Amdahl's law to your results. Nothing fancy, just play with the numbers until it looks okay. What is the estimated parallel fraction? Given this, what is your theoretical maximum speedup?

> **Solution:**
>
> Fit Amdahl's Law S(n) = 1 / ((1 - p) + p/n) to the measured data by trying different values of p until the theoretical curve matches the measured speedup.
>
> From the official solution (run on Xeon Gold 6226R): a parallel fraction of **p ≈ 0.945** fits well. The theoretical maximum speedup is then:
>
> S_max = 1 / (1 - 0.945) = 1 / 0.055 ≈ **18.2**
>
> This means even with infinite processors, the chunked pi implementation can only achieve ~18x speedup due to its ~5.5% serial overhead (process pool setup, result aggregation, etc.).

## Exercise 2.5 `[PRACTICE]` (Optional)

Rewrite the code using `pool.map()` instead of `pool.apply_async()` to perform the computation of π. What do you observe? What are pros and cons of `pool.map()`?

> **Solution:**
>
> ```python
> import random
> import multiprocessing
> import sys
>
> def sample(_):  # Must take an argument because of map
>     x = random.uniform(-1.0, 1.0)
>     y = random.uniform(-1.0, 1.0)
>     if x**2 + y**2 <= 1:
>         return 1
>     else:
>         return 0
>
> if __name__ == '__main__':
>     samples = 10000000
>     hits = 0
>     n_tasks = int(sys.argv[1])
>     chunk_size = samples // n_tasks
>     pool = multiprocessing.Pool(n_tasks)
>     res = pool.map(sample, range(samples), chunksize=chunk_size)
>     hits = sum(res)
>     pi = 4.0 * hits/samples
>     print(pi)
> ```
>
> **Pros of `pool.map()`:** Much simpler and more readable code — no manual async result collection needed.
>
> **Cons:** Performance is not as good as the manual `apply_async` version for this workload because each iteration does very little computation. `pool.map` has overhead from building and returning the full result list. For tasks with heavier per-item computation, the convenience of `map` is well worth it.

---

## Section 3: The Mandelbrot Set

The Mandelbrot Set is a famous mathematical set of complex numbers defined by iterating the formula:

**z_(n+1) = z_n² + c**

The set contains all complex numbers c for which the iteration remains bounded as n → ∞. In this exercise we compute an image of the set by calculating the *escape time* for each point — the number of iterations until |z| > 2 (or 100 if it never escapes).

Template code:
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
    ########################
    # YOUR CODE HERE #
    ########################
    return escape_times

def plot_mandelbrot(escape_times):
    plt.imshow(escape_times, cmap='hot', extent=(-2, 2, -2, 2))
    plt.axis('off')
    plt.savefig('mandelbrot.png', bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    width = 800
    height = 800
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2
    num_proc = 4

    # Precompute points
    x_values = np.linspace(xmin, xmax, width)
    y_values = np.linspace(ymin, ymax, height)
    points = np.array([complex(x, y) for x in x_values for y in y_values])

    # Compute set
    mandelbrot_set = generate_mandelbrot_set(points, num_proc)

    # Save set as image
    mandelbrot_set = mandelbrot_set.reshape((height, width))
    plot_mandelbrot(mandelbrot_set)
```

## Exercise 3.1 `[AUTOLAB]`

Implement the function `generate_mandelbrot_set` to compute the escape times for each complex number in parallel. The function must return a NumPy array. Distribute the points equally between all the processors. Hint: adapt the chunked parallel implementation from the previous exercise.

> **Solution:**
>
> The key is to use `pool.map` with a `chunksize` equal to `len(points) // num_processes` so that each worker gets an equal share of the points array.
>
> Full implementation (`mandelbrot1.py`):
>
> ```python
> import multiprocessing
> import numpy as np
> import matplotlib.pyplot as plt
>
> def mandelbrot_escape_time(c):
>     z = 0
>     for i in range(100):
>         z = z**2 + c
>         if np.abs(z) > 2.0:
>             return i
>     return 100
>
> def generate_mandelbrot_set(points, num_processes):
>     chunk_size = len(points) // num_processes
>     pool = multiprocessing.Pool(num_processes)
>     escape_times = pool.map(mandelbrot_escape_time, points, chunksize=chunk_size)
>     pool.close()
>     pool.join()
>     return np.array(escape_times)
>
> def plot_mandelbrot(escape_times):
>     plt.imshow(escape_times, cmap='hot', extent=(-2, 2, -2, 2))
>     plt.axis('off')
>     plt.savefig('mandelbrot.png', bbox_inches='tight', pad_inches=0)
>
> if __name__ == "__main__":
>     width = 800
>     height = 800
>     xmin, xmax = -2, 2
>     ymin, ymax = -2, 2
>     num_proc = 4
>
>     x_values = np.linspace(xmin, xmax, width)
>     y_values = np.linspace(ymin, ymax, height)
>     points = np.array([complex(x, y) for x in x_values for y in y_values])
>
>     mandelbrot_set = generate_mandelbrot_set(points, num_proc)
>
>     mandelbrot_set = mandelbrot_set.reshape((height, width))
>     plot_mandelbrot(mandelbrot_set)
> ```

## Exercise 3.2 `[AUTOLAB]`

As before, run the program as a batch job where you vary the number of processes `num_proc`. Make a speedup plot.

> **Solution:**
>
> Submit a job script that loops over `num_proc` values (e.g., 1, 2, 4, 8, 16, 32), records the runtime for each, writes results to a CSV, and then calls `plot_speedup.py` to generate the plot. Example pattern:
>
> ```bash
> for n in 1 2 4 8 16 32; do
>     python mandelbrot1.py $n >> timing_results.csv
> done
> python plot_speedup.py
> ```
>
> On a Xeon Gold 6226R, speedup increases with more cores but flattens before the theoretical maximum. The speedup is not great at high core counts — this motivates the chunked version in Exercise 3.3. The performance still keeps improving as more cores are added, which confirms the computation is parallelizable, but the chunk granularity limits efficiency.

## Exercise 3.3 `[AUTOLAB]`

Implement a new function `generate_mandelbrot_set_chunks`, that distributes the points in smaller chunks to the workers. Make sure there are more chunks than workers. Hint: adapt the chunk parallel implementation by setting `chunk_size` to a fixed number.

> **Solution:**
>
> Use a fixed small `chunk_size` (e.g., 100) instead of dividing evenly by the number of processes. This creates many more chunks than workers, allowing the pool to dynamically assign new chunks to workers that finish early — a form of dynamic load balancing.
>
> Full implementation (`mandelbrot2.py`):
>
> ```python
> import multiprocessing
> import sys
> import time
> import numpy as np
> import matplotlib.pyplot as plt
>
> def mandelbrot_escape_time(c):
>     z = 0
>     for i in range(100):
>         z = z**2 + c
>         if np.abs(z) > 2.0:
>             return i
>     return 100
>
> def generate_mandelbrot_set_chunks(points, num_processes):
>     chunk_size = 100
>     with multiprocessing.Pool(num_processes) as pool:
>         escape_times = pool.map(mandelbrot_escape_time, points, chunksize=chunk_size)
>     return np.array(escape_times)
>
> def plot_mandelbrot(escape_times):
>     plt.imshow(escape_times, cmap='hot', extent=(-2, 2, -2, 2))
>     plt.axis('off')
>     plt.savefig('mandelbrot.png', bbox_inches='tight', pad_inches=0)
>
> if __name__ == "__main__":
>     width = 800
>     height = 800
>     xmin, xmax = -2, 2
>     ymin, ymax = -2, 2
>     num_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
>
>     x_values = np.linspace(xmin, xmax, width)
>     y_values = np.linspace(ymin, ymax, height)
>     points = np.array([complex(x, y) for x in x_values for y in y_values])
>
>     t0 = time.time()
>     mandelbrot_set = generate_mandelbrot_set_chunks(points, num_proc)
>     elapsed = time.time() - t0
>
>     print(f"{num_proc},{elapsed:.4f}")
>
>     mandelbrot_set = mandelbrot_set.reshape((height, width))
>     plot_mandelbrot(mandelbrot_set)
> ```

## Exercise 3.4 `[PRACTICE]`

Run the new function for varying processes, make a speedup plot and estimate the parallel fraction with Amdahl's law. What do you see? Why does the chunked version achieve a higher speedup? Hint: does all parts of the Mandelbrot set require the same amount of computation?

> **Solution:**
>
> The chunked version achieves a much higher speedup. Using the official solution results (Xeon Gold 6226R):
>
> - Estimated parallel fraction: **p ≈ 0.98** (compared to ~0.94 for the equal-split version)
> - Theoretical maximum speedup: S_max = 1 / (1 - 0.98) = **50**
>
> **Why the improvement?** The time required to compute the escape time varies greatly across the complex plane. Points inside the Mandelbrot set always hit the maximum of 100 iterations, while points that escape early (outside the set) finish in very few iterations. When points are divided equally upfront (Exercise 3.1), some processes get unlucky and receive many "slow" interior points, while others finish quickly with exterior points. The fast workers then sit idle while the slow workers finish — this is called **load imbalance**.
>
> With a fixed small chunk_size (100), many more chunks exist than workers. Workers that finish a chunk quickly are immediately assigned the next available chunk. This dynamic load balancing keeps all workers busy and eliminates most of the idle time, resulting in much better utilization and a higher effective speedup.
