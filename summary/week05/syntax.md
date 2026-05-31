# Week 5 — Parallelism & Amdahl's Law Syntax Reference

## Amdahl's Law Formulas

```
S(p) = 1 / ((1 - F) + F/p)    # speedup at p cores
S_max = 1 / (1 - F)            # max speedup (p → ∞)
F = p(1 - 1/S(p)) / (p - 1)   # solve for F from measured speedup
E(p) = S(p) / p                # efficiency (ideal = 1.0)
T(1) = T(p) × S(p)             # recover serial time
```

**Speedup = wall-clock time ratio — never CPU time.**

---

## multiprocessing.Pool

### `Pool(n_proc)`
- **What**: creates a pool of n_proc worker processes
- **Gotcha**: `Pool()` with no argument uses `os.cpu_count()` — NOT your LSF allocation!

```python
from multiprocessing import Pool

# pool.map — static scheduling, ordered results
with Pool(n_proc) as pool:
    results = pool.map(func, iterable)
    results = pool.map(func, iterable, chunksize=100)

# pool.starmap — multiple args per call
with Pool(n_proc) as pool:
    results = pool.starmap(func, [(a1, b1), (a2, b2)])

# pool.imap_unordered — dynamic scheduling (results as completed)
with Pool(n_proc) as pool:
    results = list(pool.imap_unordered(func, iterable))
```

**Always include guard on macOS/Windows:**
```python
if __name__ == '__main__':
    with Pool(n) as pool:
        ...
```

---

## ThreadPool (for GIL-releasing code)

```python
from multiprocessing.pool import ThreadPool

# Works when func releases the GIL (NumPy, Numba nogil=True)
with ThreadPool(n_threads) as pool:
    results = pool.map(np.sum, arrays)
```

---

## Static vs Dynamic Scheduling

| | When to use | How |
|---|---|---|
| **Static** | uniform task times, low overhead | `pool.map(func, items)` |
| **Dynamic** | variable task times, prevents imbalance | `pool.imap_unordered(func, items)` |

High stddev in runtime → dynamic. Low stddev → static.

---

## GIL Rules

| Code type | GIL held? | Use |
|---|---|---|
| Pure Python loops | Yes | `Pool` (multiprocessing) |
| NumPy operations | No (released) | `ThreadPool` works |
| Numba `nogil=True` | No (released) | `ThreadPool` works |
| I/O-bound | Released during I/O | `ThreadPool` works |

---

## Granularity Warning

```python
# BAD: 1M tasks = 1M IPC messages → slower than serial
pool.map(tiny_func, range(1_000_000))

# GOOD: 8 tasks, each does 125k operations
chunk = total // n_proc
pool.map(chunked_func, [chunk] * n_proc)
```

---

## Exam Traps

| Trap | Correct |
|---|---|
| `S_max = 1/(1-F)` and `S(8)` are the same | `S_max` is limit as p→∞; `S(8)` is finite |
| `Pool()` uses LSF allocation | Uses `os.cpu_count()` — set OMP/MKL threads |
| Threading helps CPU-bound pure Python | GIL blocks it — use multiprocessing |
| `real` and `user` both decrease with parallelism | Only `real` decreases; `user` stays same |
