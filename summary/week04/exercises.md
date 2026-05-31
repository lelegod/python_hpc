# Week 4 Exercises — NumPy Broadcasting & Vectorization

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Section 1: Broadcasting](#section-1-broadcasting)
- [Exercise 1.1 `[AUTOLAB]`](#exercise-11-autolab)
- [Exercise 1.2 `[AUTOLAB]`](#exercise-12-autolab)
- [Exercise 1.3 `[AUTOLAB]`](#exercise-13-autolab)
- [Section 2: High Performance Haversine](#section-2-high-performance-haversine)
- [Exercise 2.1 `[AUTOLAB]`](#exercise-21-autolab)
- [Exercise 2.2 `[AUTOLAB]`](#exercise-22-autolab)
- [Exercise 2.3 `[AUTOLAB]`](#exercise-23-autolab)
- [Exercise 2.4 `[PRACTICE]`](#exercise-24-practice)
- [Exercise 2.5 `[PRACTICE]`](#exercise-25-practice)
- [Exercise 2.6 `[PRACTICE]`](#exercise-26-practice)
- [Exercise 2.7 `[AUTOLAB]`](#exercise-27-autolab)
- [Exercise 2.8 `[PRACTICE]`](#exercise-28-practice)

---

---

## Section 1: Broadcasting

---

## Exercise 1.1 `[AUTOLAB]`

Write a Python function `standardize_rows` which receives three inputs: `data` with shape n×d, `mean` with shape d, `std` with shape d. It must then subtract `mean` from each row of `data` and then divide by `std`. Finally, it must return the result. Do not use any NumPy functions.

*Input:* Three arrays: `data` with shape n×d, `mean` with shape d, `std` with shape d.

*Output:* a new array `output` where `output[i] = (data[i] - mean) / std`.

*Example:* For inputs

```
data = [[1, 2, 3],
        [4, 5, 6]]
mean = [0.5, 1, 3]
std  = [1,   2, 3]
```

the output should be:

```
[[0.5, 0.5, 0],
 [3.5, 2.0, 1]]
```

> **Solution:**
>
> Broadcasting automatically aligns `mean` (shape `(d,)`) and `std` (shape `(d,)`) against each row of `data` (shape `(n, d)`), so the subtraction and division apply element-wise across all rows simultaneously. No NumPy functions are needed — just arithmetic operators.
>
> ```python
> import numpy as np
>
> def standardize_rows(data, mean, std):
>     return (data - mean) / std
>
> if __name__ == "__main__":
>     data = np.array([[1, 2, 3], [4, 5, 6]])
>     mean = np.array([.5, 1, 3])
>     std = np.array([1, 2, 3])
>     print(standardize_rows(data, mean, std))
> ```

---

## Exercise 1.2 `[AUTOLAB]`

Write a Python function `outer` that receives two vectors as input and returns the [outer product](https://en.wikipedia.org/wiki/Outer_product) of the two. Do not use any NumPy functions.

*Input:* Two vectors as NumPy arrays of length n and m respectively.

*Output:* The n×m matrix giving the outer product.

*Example:* The outer product of x=[1, 2] and y=[3, 4, 5] is:

```
[[ 3,  4,  5],
 [ 6,  8, 10]]
```

> **Solution:**
>
> Reshape `x` to shape `(n, 1)` using `x[:, None]` so that multiplying by `y` (shape `(m,)`) broadcasts across columns, producing an n×m result. No NumPy functions are required — only the `*` operator and indexing.
>
> ```python
> import numpy as np
>
> def outer(x, y):
>     return x[:, None] * y
>
> if __name__ == "__main__":
>     x = np.array([1, 3])
>     y = np.array([1, 2, 3])
>     print(outer(x, y))
> ```

---

## Exercise 1.3 `[AUTOLAB]`

Write a Python function `distmat_1d` that receives two vectors as input and returns the distance matrix between them. Do not use for loops or any NumPy functions except for `abs`.

*Input:* Two vectors `x` and `y` of length n and m respectively.

*Output:* An n×m matrix D where D_ij = |x_i - y_j|.

*Example:* For inputs x=[1, 2] and y=[3, 0.5, 1] the output should be:

```
[[2,   0.5, 0],
 [1,   1.5, 1]]
```

> **Solution:**
>
> Reshape `x` to `(n, 1)` with `x[:, None]` and subtract `y` (shape `(m,)`). Broadcasting produces an n×m difference matrix, and `abs()` takes the absolute value element-wise. No for loops needed.
>
> ```python
> import numpy as np
>
> def distmat_1d(x, y):
>     return abs(x[:, None] - y)
>
> if __name__ == "__main__":
>     x = np.array([1, 3])
>     y = np.array([1, 2, 3])
>     print(distmat_1d(x, y))
> ```

---

## Section 2: High Performance Haversine

The following program loads a set of latitude/longitude coordinates, computes a pairwise Haversine distance matrix, and prints summary statistics. The goal of this exercise is to progressively optimize it using profiling and NumPy.

```python
import sys
import numpy as np

def distance_matrix(p1, p2):
    p1, p2 = np.radians(p1), np.radians(p2)

    D = np.empty((len(p1), len(p2)))
    for i in range(len(p1)):
        for j in range(len(p2)):
            dsin2 = np.sin(0.5 * (p1[i] - p2[j])) ** 2
            cosprod = np.cos(p1[i, 0]) * np.cos(p2[j, 0])
            a = dsin2[0] + cosprod * dsin2[1]
            D[i, j] = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    D *= 6371  # Earth radius in km
    return D


def load_points(fname):
    data = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=(1, 2))
    return data


def distance_stats(D):
    # Extract upper triangular part to avoid duplicate entries
    assert D.shape[0] == D.shape[1], 'D must be square'
    idx = np.triu_indices(D.shape[0], k=1)
    distances = D[idx]
    return {
        'mean': float(distances.mean()),
        'std': float(distances.std()),
        'max': float(distances.max()),
        'min': float(distances.min()),
    }


fname = sys.argv[1]
points = load_points(fname)
D = distance_matrix(points, points)
stats = distance_stats(D)
print(stats)
```

Data is available at `/dtu/projects/02613_2025/data/locations/` on the HPC cluster.

---

## Exercise 2.1 `[AUTOLAB]`

Make a job script that runs the above program on the `hpc` queue. Request a single core and specify a CPU model so the results are repeatable. For the Autolab submission, assume that the input is always a file with path `input.csv`.

> **Solution:**
>
> The job script pins to a specific CPU model (`XeonGold6226R`) for reproducibility. The Python script is called with `input.csv` as required by Autolab.
>
> ```bash
> #!/bin/bash
> #BSUB -J python
> #BSUB -q hpc
> #BSUB -W 1
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -R "select[model==XeonGold6226R]"
> #BSUB -o python%J.out
> #BSUB -e python_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> python haversine.py input.csv
> ```

---

## Exercise 2.2 `[AUTOLAB]`

Modify your job script so it runs the program under the builtin `cProfile` profiler. What do you see? In what functions does the program spend the most time?

Hint: See section 2.1.2 and 2.2.1 in Fast Python. Search for the name of your Python script in the profiler output to focus on the relevant functions. For the second question, use one of the larger datasets.

> **Solution:**
>
> Add `-m cProfile -s cumulative` to the Python invocation to profile and sort by cumulative time. The profiler output will show that `distance_matrix` dominates runtime due to the nested Python for loops calling NumPy functions (`np.sin`, `np.cos`, `np.arctan2`) on scalar values repeatedly — each call has Python-level overhead.
>
> ```bash
> #!/bin/bash
> #BSUB -J python
> #BSUB -q hpc
> #BSUB -W 10
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -R "select[model==XeonGold6226R]"
> #BSUB -o python%J.out
> #BSUB -e python_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> python -m cProfile -s cumulative haversine.py input.csv
> ```

---

## Exercise 2.3 `[AUTOLAB]`

Let us optimize the `distance_matrix` function. Using NumPy array operations and broadcasting, rewrite the function so there is only one loop instead of two nested ones.

> **Solution:**
>
> The inner loop is eliminated by operating on an entire row of `p2` at once. For each `i`, `p1[i]` (shape `(2,)`) is subtracted from `p2` (shape `(m, 2)`) and the result is vectorised over all `j` simultaneously.
>
> ```python
> def distance_matrix(p1, p2):
>     p1, p2 = np.radians(p1), np.radians(p2)
>
>     D = np.empty((len(p1), len(p2)))
>     for i in range(len(p1)):
>         dsin2 = np.sin(0.5 * (p1[i] - p2)) ** 2   # shape (m, 2)
>         cosprod = np.cos(p1[i, 0]) * np.cos(p2[:, 0])  # shape (m,)
>         a = dsin2[:, 0] + cosprod * dsin2[:, 1]    # shape (m,)
>         D[i, :] = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
>
>     D *= 6371
>     return D
> ```

---

## Exercise 2.4 `[PRACTICE]`

Rerun the profiler. What has changed? Did it get faster? How much / little?

> **Solution:**
>
> After replacing the inner loop with NumPy array ops, `distance_matrix` is dramatically faster — roughly an 80x speedup on Intel Xeon Gold 6226 with `locations_500.csv`. Example profiler output (filtered to script functions only):
>
> ```
> 1    0.000    0.000    0.196    0.196 points.py:1(<module>)
> 1    0.026    0.026    0.026    0.026 points.py:21(distance_matrix)
> 1    0.000    0.000    0.008    0.008 points.py:66(load_points)
> 1    0.001    0.001    0.003    0.003 points.py:71(distance_stats)
> ```
>
> The key reason: the old inner loop called NumPy functions on Python scalars (paying Python overhead n×m times). The one-loop version calls NumPy on full arrays (paying overhead only n times), allowing vectorised C-level computation.

---

## Exercise 2.5 `[PRACTICE]`

Let us now zoom in on the `distance_matrix` function using line profiling. Using the `line_profiler`, modify your Python program and job script to run with line profiling enabled on `distance_matrix`. Inspect the results. What do you see? On what lines should we focus?

Hint: See section 2.2.2 in Fast Python.

> **Solution:**
>
> Decorate `distance_matrix` with `@profile` and run with `kernprof -l -v haversine.py input.csv` (or equivalent). The line profiler output shows something like:
>
> ```
> Total time: 0.0329796 s
> File: points.py
> Function: distance_matrix at line 20
>
> Line #      Hits         Time  Per Hit   % Time  Line Contents
> ==============================================================
>     22         1         15.0     15.0      0.0      p1 = np.radians(p1)
>     23         1          4.5      4.5      0.0      p2 = np.radians(p2)
>     25         1          9.6      9.6      0.0      D = np.empty(...)
>     26       500        168.3      0.3      0.5      for i in range(len(p1)):
>     27       499      17275.1     34.6     52.4          dsin2 = ...
>     28       499       6913.0     13.9     21.0          cosprod = ...
>     29       499       1790.0      3.6      5.4          a = ...
>     30       499       4762.1      9.5     14.4          row = np.arctan2(...)
>     31       499       1841.4      3.7      5.6          D[i, :] = row
> ```
>
> Focus should be on the `dsin2` computation (~52%), `cosprod` (~21%), and the `arctan2` call (~14%) — these three lines consume the vast majority of runtime.

---

## Exercise 2.6 `[PRACTICE]`

Optimize `distance_matrix` to improve its performance. There are at least two optimizations you can make:

1. Moving a repeated calculation out of the loop to pre-compute once.
2. The fact that arctan2(sqrt(a), sqrt(1-a)) = arcsin(sqrt(a)). Re-run the profiler. Did it make the code faster? Did it change where time is spent in `distance_matrix`?

> **Solution:**
>
> **Optimization 1 — Pre-compute cosines:** `np.cos(p2[:, 0])` is computed identically on every iteration of `i`. Move it outside the loop to compute once:
>
> ```python
> cos_p2_lat = np.cos(p2[:, 0])   # computed once, reused each iteration
> cos_p1_lat = np.cos(p1[:, 0])   # also pre-compute for p1
> for i in range(len(p1)):
>     dsin2 = np.sin(0.5 * (p1[i] - p2)) ** 2
>     cosprod = cos_p1_lat[i] * cos_p2_lat
>     a = dsin2[:, 0] + cosprod * dsin2[:, 1]
>     D[i, :] = 2 * np.arcsin(np.sqrt(a))   # Optimization 2
> ```
>
> **Optimization 2 — Use arcsin instead of arctan2:** The identity arctan2(sqrt(a), sqrt(1-a)) = arcsin(sqrt(a)) means only one `sqrt` call is needed instead of two, and `arcsin` can be cheaper than `arctan2`.
>
> Combined effect: on Intel Xeon Gold 6226 with the full dataset, these two changes reduce `distance_matrix` runtime from ~12.6 s to ~8.5 s — roughly 25% faster. Not dramatic, but worthwhile.

---

## Exercise 2.7 `[AUTOLAB]`

Finally, let us see what happens if we eliminate for loops completely. Rewrite `distance_matrix` to perform all calculations with NumPy array operations and broadcasting. What happened? Did it get faster?

> **Solution:**
>
> By inserting `None` axes, `p1` and `p2` broadcast against each other to produce n×m intermediate arrays in a single vectorised pass.
>
> ```python
> import sys
> import numpy as np
>
> def distance_matrix(p1, p2):
>     p1, p2 = np.radians(p1), np.radians(p2)
>
>     # No loop: broadcast p1 (n,1,2) against p2 (1,m,2)
>     dsin2 = np.sin(0.5 * (p1[:, None, :] - p2[None, :, :])) ** 2
>     cosprod = np.cos(p1[:, None, 0]) * np.cos(p2[None, :, 0])
>     a = dsin2[:, :, 0] + cosprod * dsin2[:, :, 1]
>     D = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
>
>     D *= 6371  # Earth radius in km
>     return D
>
>
> def load_points(fname):
>     data = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=(1, 2))
>     return data
>
>
> def distance_stats(D):
>     assert D.shape[0] == D.shape[1], 'D must be square'
>     idx = np.triu_indices(D.shape[0], k=1)
>     distances = D[idx]
>     return {
>         'mean': float(distances.mean()),
>         'std': float(distances.std()),
>         'max': float(distances.max()),
>         'min': float(distances.min()),
>     }
>
>
> fname = sys.argv[1]
> points = load_points(fname)
> D = distance_matrix(points, points)
> stats = distance_stats(D)
> print(stats)
> ```
>
> Surprisingly, the no-loop version is **not always faster** than the optimized single-loop version. For large inputs it can be slower, because the n×m intermediate arrays it creates are too large to fit in CPU cache, causing memory bandwidth to become the bottleneck.

---

## Exercise 2.8 `[PRACTICE]`

Measure the performance in MFLOP/S for the optimized single loop and the no-loop version of `distance_matrix`. Use random points as input and vary the number of points from 10^1 to 10^4. Plot the results and explain what you observe.

1. Plot the run time in a loglog plot as a function of the size of the distance matrix in kilobytes. Include CPU cache sizes in your plot. Is one always faster? If yes, which one? If not, when do they switch?

> **Solution:**
>
> A benchmarking script:
>
> ```python
> from time import perf_counter as time
> import numpy as np
>
> def distance_matrix_loop(p1, p2):
>     # One-loop version from Exercise 2.3 (with pre-computed cosines from 2.6)
>     ...
>
> def distance_matrix_noloop(p1, p2):
>     # No-loop version from Exercise 2.7
>     ...
>
> loop_times = []
> noloop_times = []
> ns = np.logspace(1, 4, 30)
> n_repeat = 5
> for n in ns:
>     p1 = np.random.rand(int(n), 2)
>     p2 = np.random.rand(int(n), 2)
>     t = time()
>     for _ in range(n_repeat):
>         distance_matrix_loop(p1, p2)
>     loop_times.append((time() - t) / n_repeat)
>     t = time()
>     for _ in range(n_repeat):
>         distance_matrix_noloop(p1, p2)
>     noloop_times.append((time() - t) / n_repeat)
>
> print('ns =', list(ns))
> print('loop_times =', loop_times)
> print('noloop_times =', noloop_times)
> ```
>
> **Observation:** For small distance matrices (fitting entirely in L1/L2 cache), the no-loop version is significantly faster because all intermediates stay in fast memory. For large matrices (exceeding L2 cache), the no-loop version must allocate and traverse large n×m intermediate arrays that spill to L3 or RAM, making it slower than the single-loop version, which only materialises one row at a time.
>
> **Key takeaway:** You cannot always predict which implementation will be fastest without measuring. The crossover point corresponds roughly to when the distance matrix no longer fits in the L2 cache. This illustrates why memory layout and cache effects matter even for pure NumPy code.
