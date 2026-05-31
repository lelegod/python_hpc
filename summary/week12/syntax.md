# Week 12 — Project Syntax Reference (Wall Heating)

## Jacobi Iteration (reference implementation)

```python
import numpy as np

def jacobi(u, interior):
    u_new = 0.25 * (u[1:-1, :-2]    # left
                  + u[1:-1, 2:]     # right
                  + u[:-2, 1:-1]    # up
                  + u[2:, 1:-1])    # down
    u[1:-1, 1:-1] = np.where(interior, u_new, u[1:-1, 1:-1])
    return u
```

Run until convergence:
- `MAX_ITER = 20000`
- `ABS_TOL = 1e-4` (max change < tolerance → converged)

---

## Loading Building Data

```python
import numpy as np

building_id = '00001'
base = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

domain   = np.load(f'{base}{building_id}_domain.npy')    # temperature grid (514×514)
interior = np.load(f'{base}{building_id}_interior.npy')  # binary mask (1=interior)
```

---

## Computing Statistics

```python
def compute_stats(u, interior):
    temps = u[1:-1, 1:-1][interior.astype(bool)]
    return {
        'mean_temp':    temps.mean(),
        'std_temp':     temps.std(),
        'pct_above_18': (temps > 18).mean() * 100,
        'pct_below_15': (temps < 15).mean() * 100,
    }
```

---

## Numba JIT Jacobi (Task 7)

```python
from numba import njit
import numpy as np

@njit
def jacobi_numba(u, interior):
    rows, cols = u.shape
    max_change = 0.0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if interior[i-1, j-1]:
                new_val = 0.25 * (u[i, j-1] + u[i, j+1] + u[i-1, j] + u[i+1, j])
                max_change = max(max_change, abs(new_val - u[i, j]))
                u[i, j] = new_val
    return u, max_change
```

Cache-efficient ikj loop order matters here — innermost `j` loop accesses memory sequentially.

---

## CUDA Jacobi Kernel (Task 8)

```python
from numba import cuda
import numpy as np

@cuda.jit
def jacobi_kernel(u, interior, u_new):
    i, j = cuda.grid(2)
    if 1 <= i < u.shape[0]-1 and 1 <= j < u.shape[1]-1:
        if interior[i-1, j-1]:
            u_new[i, j] = 0.25 * (u[i, j-1] + u[i, j+1]
                                 + u[i-1, j] + u[i+1, j])
        else:
            u_new[i, j] = u[i, j]

# Launch
tpb = (16, 16)
bpg = ((u.shape[0] + tpb[0] - 1) // tpb[0],
       (u.shape[1] + tpb[1] - 1) // tpb[1])
```

---

## CuPy Jacobi (Task 9)

```python
import cupy as cp

def jacobi_cupy(u, interior):
    u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:]
                  + u[:-2, 1:-1] + u[2:, 1:-1])
    u[1:-1, 1:-1] = cp.where(interior, u_new, u[1:-1, 1:-1])
    return u
```

Only change from NumPy version: `import cupy as cp`, `np.where` → `cp.where`.

---

## Parallelizing Over Floor Plans

```python
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

# Static scheduling (uniform task times)
with Pool(n_workers) as pool:
    results = pool.map(simulate_building, building_ids)

# Dynamic scheduling (variable convergence times)
with Pool(n_workers) as pool:
    results = list(pool.imap_unordered(simulate_building, building_ids))
```

Dynamic is better here — different floor plans take different numbers of iterations to converge.

---

## Profiling Jacobi

```bash
# Line-level profile
kernprof -l -v simulate.py

# GPU profile
nsys profile -o jacobi_profile python simulate_gpu.py
nsys stats jacobi_profile.nsys-rep
```
