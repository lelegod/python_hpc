# Week 2 — Timing & Profiling Syntax Reference

## Timing

### `time.perf_counter()`
- **What**: highest-resolution CPU clock — measures wall time
- **Returns**: `float` — seconds since some arbitrary reference point
- **Use for**: benchmarking code on CPU
- **Gotcha**: does NOT measure GPU time — need `cuda.synchronize()` first

```python
from time import perf_counter

t = perf_counter()
result = my_function(x)
elapsed = perf_counter() - t          # seconds

# For fast operations, repeat and divide:
t = perf_counter()
for _ in range(1000):
    result = fast_op(x)
elapsed = (perf_counter() - t) / 1000  # seconds per call
```

---

## cProfile — function-level profiling

```bash
# Run from command line
python -m cProfile -s cumulative script.py args
```

### Output columns

| Column | Meaning |
|---|---|
| `ncalls` | number of calls (e.g. `5/1` = 5 total, 1 primitive/non-recursive) |
| `tottime` | time in THIS function only (excludes sub-calls) |
| `percall` | tottime / ncalls |
| `cumtime` | total time including ALL sub-calls |
| `percall` (2nd) | cumtime / ncalls |

**Use `cumtime` to find the overall bottleneck.**
**Use `tottime` to find slow code within a specific function.**

**Scaling to production:**
```
production_time ≈ percall × production_ncalls
```

---

## kernprof / line_profiler — line-level profiling

```bash
# Install: pip install line_profiler
# Decorate the function you want to profile with @profile
kernprof -l -v script.py args
```

### Output columns

| Column | Meaning |
|---|---|
| `Hits` | how many times this line ran = loop iteration count |
| `Time` | total time in **microseconds** |
| `Per Hit` | Time / Hits |
| `% Time` | fraction of total function time |

**`Hits` tells you the input size** — loop body hit 5000 times → n=5000

**FLOP/s calculation:**
```python
FLOP/s = (FLOPs_per_iteration * Hits) / (total_time_seconds)
# total_time_seconds = Time column / 1_000_000
```

---

## `time` shell command — wall vs CPU time

```bash
time python script.py
```

| Output | Meaning | With parallelism |
|---|---|---|
| `real` | wall-clock time | decreases |
| `user` | CPU time summed over all cores | stays same or increases |
| `sys` | OS system call time | roughly same |

**Exam trap:** both `real` and `user` do NOT both decrease — only `real` decreases.

---

## MFLOP/s formula

```
MFLOP/s = num_FLOPs / (time_seconds * 1_000_000)
```

Count FLOPs manually — there is no automatic counter.
Example: `2 * mat[0, :]` on N elements = N FLOPs.
Matrix multiply N×N: N³ FLOPs.
