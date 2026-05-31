# Week 12 — Course Project

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Overview](#overview)
- [Project Description](#project-description)
- [Project Requirements](#project-requirements)
- [HPC Concepts Applied](#hpc-concepts-applied)
- [Technical Skills Needed](#technical-skills-needed)
- [Suggested Approach](#suggested-approach)
- [Evaluation Criteria](#evaluation-criteria)
- [Key Takeaways](#key-takeaways)

---

## Overview

Week 12 is the culmination of the course: the mini-project. Rather than isolated exercises, students integrate all techniques learned across weeks 1–11 into a single, realistic scientific computing problem. The project is called **Wall Heating!** — a fictional study of an experimental building heating system. The central goal is to simulate heat distribution across 4571 real building floor plans as fast as possible, applying successive layers of HPC optimization to an initially slow reference implementation.

The project is group work and is submitted as a PDF report plus a zip of all code and job scripts. It must be passed in order to sit the final exam, though the project grade does not affect the final grade.

---

## Project Description

The scenario: instead of radiators, heating elements are embedded in the inside walls of buildings so that the whole wall acts as a radiator. Load-bearing walls are left cold. The question is whether this "Wall Heating" approach actually keeps rooms warm enough.

**Physical model:** Steady-state heat distribution is found by solving Laplace's equation in 2D:

```
d²u/dx² + d²u/dy² = 0
```

with Dirichlet boundary conditions:
- Inside walls fixed at **25°C**
- Load-bearing walls fixed at **5°C**

**Numerical method:** The Jacobi iterative method on a discrete grid. Each interior point (inside a room, not on a wall) is updated as the average of its four neighbors:

```
u[i,j] = 0.25 * (u[i,j-1] + u[i,j+1] + u[i-1,j] + u[i+1,j])
```

Wall points are fixed and never updated. Iteration continues until either a maximum iteration count is reached (`MAX_ITER = 20_000`) or the grid changes by less than a tolerance (`ABS_TOL = 1e-4`).

**Dataset:** The Modified Swiss Dwellings dataset — 4571 real building floor plans, each converted to a **514 x 514 simulation grid**. Per building there are two `.npy` files:
- `{building_id}_domain.npy` — initial temperature grid (load-bearing walls = 5, inside walls = 25, interior = 0)
- `{building_id}_interior.npy` — binary mask (1 = interior point to update, 0 = wall or exterior)

Data location on the cluster: `/dtu/projects/02613_2025/data/modified_swiss_dwellings/`

**Evaluation metrics per building** (computed after convergence):
1. Mean temperature inside rooms
2. Standard deviation of temperature inside rooms
3. Percentage of room area above 18°C (below 18°C risks mold)
4. Percentage of room area below 15°C (below 15°C is too cold for human comfort)

A reference script `simulate.py` is provided. It is correct but too slow to process all 4571 floor plans in reasonable time. The project task is to optimize it progressively.

---

## Project Requirements

Students must complete the following 12 tasks (task 11 is optional):

1. **Data exploration** — Load and visualize input grids for several floor plans.
2. **Baseline timing** — Run and time the reference implementation on 10–20 floor plans via a batch job; extrapolate total runtime for all 4571 buildings.
3. **Result visualization** — Visualize the converged temperature distribution for several floor plans.
4. **Profiling** — Profile the `jacobi` function with `kernprof`; explain each line's contribution to runtime.
5. **Static parallel scheduling** — Parallelize over floor plans using static scheduling (equal work per worker); use up to 100 floor plans for timing experiments.
   - a) Measure and plot speed-up vs. number of workers.
   - b) Estimate the parallel fraction using Amdahl's law.
   - c) Determine theoretical maximum speed-up; compare to achieved speed-up and core count needed.
   - d) Estimate total runtime for all floor plans with the best parallel solution.
6. **Dynamic parallel scheduling** — Repeat the parallelization experiment with dynamic scheduling (accounts for variable convergence time per floor plan).
   - a) Quantify the speed improvement over static scheduling.
   - b) Assess whether speed-up curve improved or worsened.
7. **Numba JIT on CPU** — Rewrite `jacobi` as a Numba JIT-compiled function.
   - a) Time and compare to reference.
   - b) Explain the implementation and how cache-friendly memory access was achieved.
   - c) Estimate total runtime for all floor plans.
8. **Custom CUDA kernel with Numba** — Write a GPU kernel that performs one Jacobi iteration per launch; call it in a loop from a helper function (no early stopping, fixed iteration count).
   - a) Describe the kernel and helper function structure.
   - b) Time and compare to reference.
   - c) Estimate total runtime.
9. **CuPy GPU solution** — Port the reference NumPy implementation to CuPy.
   - a) Time and compare to reference.
   - b) Estimate total runtime.
   - c) Note any surprising performance characteristics.
10. **GPU profiling** — Profile the CuPy solution with `nsys`; identify the main bottleneck (hint: relates to week 10 material) and attempt to fix it.
11. **(Optional) Further optimization** — E.g., parallelize the CPU JIT solution, use SLURM job arrays for multi-job parallelism; report fastest achieved runtime.
12. **Full dataset analysis** — Run one fast implementation over all 4571 floor plans; use Pandas to analyze the CSV output:
    - a) Histogram of mean temperature distribution.
    - b) Average mean temperature across all buildings.
    - c) Average temperature standard deviation.
    - d) Count buildings with >= 50% of area above 18°C.
    - e) Count buildings with >= 50% of area below 15°C.

---

## HPC Concepts Applied

| Week | Concept | Application in Project |
|------|---------|------------------------|
| Week 1–2 | NumPy vectorization and array operations | Jacobi stencil update uses sliced array arithmetic instead of Python loops |
| Week 3 | Profiling with `kernprof` / `line_profiler` | Task 4: line-level profiling of the reference `jacobi` function |
| Week 4 | Parallelism fundamentals, Amdahl's law | Tasks 5–6: measuring parallel fraction and theoretical limits |
| Week 5 | Multiprocessing, static vs. dynamic scheduling | Tasks 5–6: parallelizing over floor plans with both scheduling strategies |
| Week 6 | Numba JIT compilation | Task 7: cache-aware JIT `jacobi` on CPU |
| Week 8 | CUDA kernels with Numba | Task 8: custom GPU kernel for single Jacobi iteration |
| Week 9 | CuPy (GPU NumPy) | Task 9: GPU port using CuPy drop-in replacement |
| Week 10 | GPU profiling with `nsys`, data transfer bottlenecks | Task 10: profiling and fixing the CuPy solution |
| Week 11 | Batch job submission (SLURM), job arrays | All timing tasks require batch jobs; optional task 11 uses job arrays |

---

## Technical Skills Needed

- **NumPy:** array slicing for stencil operations, boolean masking (`interior_mask`), `np.load`, `np.copy`, `np.abs`, `np.empty`
- **Numba CPU JIT:** `@njit` decorator, loop-based Jacobi for cache locality (row-major traversal), avoiding fancy indexing inside JIT
- **Numba CUDA:** `@cuda.jit`, thread/block indexing, kernel launch syntax `kernel[blocks, threads](...)`, understanding that one kernel call = one iteration
- **CuPy:** replacing `numpy` with `cupy` for GPU execution, understanding host-device transfer costs
- **Profiling:** `kernprof -lv simulate.py`, `nsys profile` for GPU timeline analysis
- **Parallel Python:** `multiprocessing.Pool`, `map` (static) vs. `imap_unordered` or `starmap` with chunksize=1 (dynamic)
- **SLURM batch jobs:** writing `.sh` job scripts, timing with `time` or Python `time` module, job arrays (`#SBATCH --array`)
- **Pandas:** reading CSV output, computing aggregates, plotting histograms with Matplotlib
- **Amdahl's law:** computing parallel fraction `p` from measured speed-ups, deriving theoretical maximum `S_max = 1 / (1 - p)`

---

## Suggested Approach

1. **Start with exploration** (tasks 1–3) before touching any optimization. Visualize a few floor plans to build intuition for what the data looks like and what a converged simulation should look like.

2. **Profile before optimizing** (task 4). Use `kernprof` to find where time is actually spent in the reference `jacobi`. The bottleneck is the inner loop running up to 20,000 iterations in interpreted Python — even though each iteration is vectorized NumPy, the loop itself has overhead and the convergence check also takes time.

3. **Layer optimizations progressively.** Work through tasks 5 → 6 → 7 → 8 → 9 in order. Each one introduces a new technique. Compare each new version against the reference to verify numerical correctness before claiming a speed-up.

4. **Always time with batch jobs.** Interactive node timing is noisy. Every timing result should come from a properly submitted SLURM job to get reproducible numbers.

5. **For the Numba CPU kernel (task 7),** write explicit nested loops and traverse in row-major order (i outer, j inner) to maximize cache hits on the C-contiguous NumPy array. Avoid fancy indexing inside the JIT function.

6. **For the CUDA kernel (task 8),** use a 2D thread grid mapping directly onto the simulation grid. Each thread handles one `(i, j)` point; check the interior mask before updating. The kernel runs exactly one Jacobi iteration; the Python helper loops over kernel launches.

7. **For CuPy (task 9),** the code change is minimal — swap `import numpy as np` for `import cupy as cp` in the relevant parts. The challenge is understanding why it may be slower than expected for small N (PCIe transfer overhead per building).

8. **For GPU profiling (task 10),** the likely bottleneck is repeated small data transfers between CPU and GPU for each floor plan. The fix is to batch all floor plans onto the GPU at once and process them together, minimizing transfers.

9. **For the final analysis (task 12),** use your fastest solution to process all 4571 buildings, redirect CSV output to a file, then load with `pd.read_csv` for analysis.

---

## Evaluation Criteria

The project is assessed on an overall judgment of the complete report. Specific expectations:

- **Correctness:** All optimized solutions must produce results that match the reference `simulate.py` output.
- **Coverage:** All 12 mandatory tasks must be addressed; task 11 is optional for extra credit.
- **Analysis quality:** Speed-up plots, Amdahl's law calculations, and runtime estimates should be properly explained, not just numbers.
- **Code quality:** Submitted zip must contain all Python scripts and SLURM job scripts used to produce the results.
- **Report format:** Short PDF report. Code snippets may be included where relevant. Group submission on DTU Learn.

The project must be passed to be eligible for the final exam. Passing is a binary requirement — the project grade does not factor into the final course grade.

Deadline: **3rd of May, 2026, 23:55** (per the HTML version of the project description).

---

## Key Takeaways

- **Real HPC problems require layered optimization.** No single technique dominates. The project forces you to apply profiling, CPU parallelism, JIT compilation, and GPU acceleration — in that order — to build intuition about when each tool helps.
- **Amdahl's law has real teeth.** If even a small fraction of the computation is serial (e.g., data loading, convergence checking), the theoretical speed-up ceiling is finite and often reached quickly.
- **Dynamic scheduling matters when work is uneven.** Different floor plans converge in different numbers of iterations. Static scheduling wastes cores on fast buildings while slow ones become the bottleneck. Dynamic scheduling (task assignment on demand) can provide meaningful additional speed-up at no algorithmic cost.
- **GPU is not always faster out of the box.** The naive CuPy port may underperform expectations for small batches because CPU-GPU data transfer dominates. Batching data onto the GPU and minimizing round-trips is the key fix — the same lesson from week 10's transfer-overhead exercises.
- **Cache locality is a first-class concern.** The Numba JIT exercise (task 7) forces explicit attention to memory access patterns in a way that NumPy's vectorized operations hide. Writing a cache-friendly loop in Numba can match or exceed GPU performance for this problem size.
- **Scientific computing is iterative.** The Jacobi method is a classic PDE solver. Understanding it here generalizes directly to fluid dynamics, electrostatics, and structural mechanics simulations in industry and research.
