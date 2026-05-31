# Week 2 — Timing & Profiling Syntax Reference

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Timing](#timing)
  - [`time.perf_counter()`](#timeperf_counter)
- [cProfile — function-level profiling](#cprofile-function-level-profiling)
  - [Output columns](#output-columns)
- [kernprof / line_profiler — line-level profiling](#kernprof-line_profiler-line-level-profiling)
  - [Output columns](#output-columns)
- [`time` shell command — wall vs CPU time](#time-shell-command-wall-vs-cpu-time)
- [MFLOP/s formula](#mflops-formula)

---

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
| `tottime` | time in THIS function's own code only — excludes all sub-calls |
| `percall` (1st) | tottime / ncalls |
| `cumtime` | total wall-clock cost of one call including ALL sub-calls |
| `percall` (2nd) | cumtime / ncalls — **use this for production projections** |

**Use `cumtime` to find the overall bottleneck.**
**Use `tottime` to find slow code inside a specific function (ignores callees).**

**Scaling to production — always use cumtime percall (2nd percall column):**
```
production_time = (cumtime / ncalls) × production_ncalls   ← scaling functions
               + cumtime                                    ← fixed-cost functions (ncalls=1)
```

Example: process_item cumtime=24.5s, ncalls=50, production=5000
→ (24.5/50) × 5000 = 0.490 × 5000 = 2450s

**Trap:** using tottime percall underestimates cost if the function calls expensive helpers.

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
# total_time_seconds = Time column / 1_000_000  (Time is in microseconds!)
```

**What counts as a FLOP?** Count every arithmetic operation on floating-point values:
```python
result += a[i]*b[i] + c[i]
#              ^       ^   ^
#          multiply   add  add (the += is also an addition)
# = 3 FLOPs per iteration
```
`+=` is always 1 FLOP — it is `result = result + (...)`, which is an addition.
Memory loads/stores and index arithmetic do NOT count as FLOPs.

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
