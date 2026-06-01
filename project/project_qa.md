# 02613 Mini-Project: Wall Heating — Complete Q&A with Code

All benchmark numbers are from actual DTU HPC runs (queue `hpc`, 8-core node `n-62-30-25`).

---

## Contents

- [Background & Setup](#background--setup)
- [Task 1 — Data Exploration](#task-1--data-exploration)
- [Task 2 — Timing the Reference](#task-2--timing-the-reference)
- [Task 3 — Visualising Results](#task-3--visualising-results)
- [Task 4 — Profiling with cProfile](#task-4--profiling-with-cprofile)
- [Task 5 — Static Scheduling](#task-5--static-scheduling)
- [Task 6 — Dynamic Scheduling](#task-6--dynamic-scheduling)
- [Task 7 — Numba JIT on CPU](#task-7--numba-jit-on-cpu)
- [Task 8 — CUDA Kernel](#task-8--cuda-kernel)
- [Task 9 — CuPy GPU Adaptation](#task-9--cupy-gpu-adaptation)
- [Task 10 — Profiling CuPy with nsys](#task-10--profiling-cupy-with-nsys)
- [Task 11 — Job Array](#task-11--job-array)
- [Task 12 — Final Analysis](#task-12--final-analysis)
- [Speedup Summary](#speedup-summary)

---

## Background & Setup

**What is the problem?**

Steady-state heat distribution in 4571 Swiss building floorplans. The Laplace PDE
`∂²u/∂x² + ∂²u/∂y² = 0` is solved numerically using the **Jacobi method** on a 512×512 grid per building.

**Boundary conditions:**
- Inside walls → 25°C (hot, acting as radiators)
- Load-bearing walls → 5°C (cold, structural)
- Exterior → unchanged (0, never touched)

**Jacobi update rule** (applied to every interior point each iteration):

```
u[i,j] ← 0.25 × (u[i,j−1] + u[i,j+1] + u[i−1,j] + u[i+1,j])
```

Iteration stops when `max(|u_old − u_new|) < atol` or after `max_iter` iterations.

**What are the two input files per building?**

```
{building_id}_domain.npy     # 512×512 float64 — initial temperatures (walls set, interior=0)
{building_id}_interior.npy   # 512×512 bool    — True where Jacobi should update (room interior)
```

**Four output statistics per building:**

| Key | Meaning | Target |
|-----|---------|--------|
| `mean_temp` | Mean interior temperature | High |
| `std_temp` | Std dev of interior temperatures | Low (consistent) |
| `pct_above_18` | % area above 18°C (mold threshold) | High |
| `pct_below_15` | % area below 15°C (comfort threshold) | Low |

---

## Task 1 — Data Exploration

**Q: How is data loaded?**

```python
def load_data(bid):
    SIZE = 512
    # Padded by 1 on each side → (514, 514)
    # The 1-pixel border stays at 0 and is never updated — acts as ghost boundary
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))   # paste 512×512 into centre
    interior_mask = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))  # 512×512 bool
    return u, interior_mask
```

**Why the +2 padding?**
The Jacobi stencil reads `u[i±1, j]` and `u[i, j±1]` for every interior point. If the grid were exactly 512×512, the outermost interior points would read out of bounds. The extra border row/column holds 0 permanently and is never updated — it acts as a zero-temperature ghost zone outside the building.

**Q: How many buildings are there?**

```python
with open(join(LOAD_DIR, "building_ids.txt")) as f:
    building_ids = f.read().splitlines()  # → list of 4571 string IDs
```

4571 buildings total.

**Q: How was the data visualised?**

```python
def task1(building_ids):
    sample_ids = building_ids[:3]
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    for row, bid in enumerate(sample_ids):
        u0, interior_mask = load_data(bid)

        # Left column: initial temperature grid (hot walls visible as bright lines)
        axes[row, 0].imshow(u0, cmap="hot", vmin=0, vmax=25)
        axes[row, 0].set_title(f"Building {bid} — Initial conditions")

        # Right column: interior mask (white = updatable room interior, black = walls/exterior)
        axes[row, 1].imshow(interior_mask, cmap="gray")
        axes[row, 1].set_title(f"Building {bid} — Interior mask")
```

---

## Task 2 — Timing the Reference

**Q: What is the reference implementation?**

```python
def jacobi(u, interior_mask, max_iter=20_000, atol=1e-4):
    u = np.copy(u)                          # work on a copy — don't modify input
    for _ in range(max_iter):
        # Vectorised 4-neighbour average on the entire 514×514 grid at once
        # Slice notation: 1:-1 = rows 1..512, :-2 = cols 0..511, etc.
        u_new = 0.25 * (u[1:-1, :-2]   # left  neighbour
                      + u[1:-1, 2:]    # right neighbour
                      + u[:-2, 1:-1]   # up    neighbour
                      + u[2:,  1:-1])  # down  neighbour

        # Only keep the new values at interior (room) points
        u_new_interior = u_new[interior_mask]

        # Convergence check: max change across all interior points
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()

        # Write updated values back
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:    # early stopping
            break
    return u
```

**Key points about this implementation:**
- `u_new = 0.25 * (...)` creates a full 512×512 temporary array every iteration — expensive
- `u_new[interior_mask]` uses boolean indexing → creates a 1D copy of only the interior values
- `u[1:-1, 1:-1][interior_mask] = ...` is a two-step fancy assignment (also creates a copy internally)
- The `.max()` convergence check is a global reduction — requires NumPy to scan all interior points

**Q: How was task 2 timed?**

```python
def task2(building_ids):
    N = 20
    ids_subset = building_ids[:N]

    start = perf_counter()
    save_time = 0
    for bid in ids_subset:
        u0, mask = load_data(bid)
        u = jacobi(u0, mask)              # the solver

        save_start = perf_counter()
        np.save(join(REPORT_DIR, f"{bid}_result.npy"), u)
        save_time += perf_counter() - save_start   # exclude I/O from timing

    elapsed = perf_counter() - start - save_time
    print(f"Time for {N} buildings:  {elapsed:.2f} s")
    print(f"Time per building:       {elapsed/N:.2f} s")
    print(f"Estimated total (4571):  {elapsed/N * 4571 / 60:.1f} min")
```

**Q: What were the timing results?**

| Measurement | Value |
|-------------|-------|
| 20 buildings | 233.59 s |
| Per building | 11.68 s |
| Estimated for all 4571 | **889.8 min (~14.8 hours)** |

**Q: Why does per-building time vary so much across buildings?**

The number of Jacobi iterations to convergence varies from ~1500 to ~16000 depending on building geometry. Compact buildings with small rooms close to inside walls converge fast; large open rooms take many more iterations. Building 10174 needed 14,788 iterations (7.8 s with Numba), while building 10051 needed only 1,535 (0.75 s). This ~10× variation is the root cause of load imbalance in parallel approaches.

---

## Task 3 — Visualising Results

**Q: How were simulation results visualised?**

```python
def task3(building_ids):
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    for row, bid in enumerate(building_ids[:3]):
        u0, mask = load_data(bid)

        # Load previously computed result (or recompute if missing)
        result_path = join(REPORT_DIR, f"{bid}_result.npy")
        if not exists(result_path):
            u = jacobi(u0, mask)
            np.save(result_path, u)
        else:
            u = np.load(result_path)

        axes[row, 0].imshow(u0, cmap="hot", vmin=0, vmax=25)   # initial
        axes[row, 1].imshow(u,  cmap="hot", vmin=0, vmax=25)   # converged
```

**What does the result show?** Temperature smoothly interpolates from hot inside walls (25°C) outward toward cold load-bearing walls (5°C). Rooms large or far from inside walls stay cool; rooms sandwiched between inside walls heat well.

---

## Task 4 — Profiling with cProfile

**Q: How was profiling done?**

```python
def task4(building_ids, N=1):
    import cProfile, pstats, io

    for bid in building_ids[:N]:
        u0, mask = load_data(bid)
        pr = cProfile.Profile()
        pr.enable()
        jacobi(u0, mask)         # profile only the solver
        pr.disable()

        stream = io.StringIO()
        ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
        ps.print_stats(20)
        # write to report/task4_profile.txt
```

**Q: What did the cProfile output show for building 10000?**

```
10811 function calls in 5.938 seconds

   ncalls  tottime  percall  cumtime  percall  filename
        1    5.861    5.861    5.938    5.938  jacobi()          ← 98.7% of total
     3602    0.068       —    0.077       —   numpy.ufunc.reduce ← .max() calls
        1    0.000       —    0.000       —   np.copy
```

**Q: How do we interpret `tottime` vs `cumtime`?**

- `tottime` = time spent in this function **excluding** time in functions it calls
- `cumtime` = time spent in this function **including** all sub-calls

`jacobi` has `tottime=5.861` and `cumtime=5.938`. The difference (0.077 s) is the time spent in `.max()` calls. Almost all the work is in the tight Jacobi loop itself.

**Q: How many iterations did building 10000 take?**

3602 iterations. The `.max()` is called once per iteration, so `ncalls=3602` confirms this.

**Q: What are the bottleneck parts of the function?**

| Part of `jacobi()` | Estimated time | Why |
|--------------------|---------------|-----|
| `u_new = 0.25 * (...)` | ~5.4 s (91%) | 4 large array slices + multiply each iteration |
| `u[interior][mask] = u_new_interior` | ~0.38 s (6%) | boolean indexing = fancy indexing = copy |
| `.max()` convergence check | 0.077 s (1.3%) | global reduction, called 3602 times |
| `np.copy`, loop overhead | < 0.05 s | negligible |

---

## Task 5 — Static Scheduling

**Q: What is static scheduling?**

Divide the list of N buildings into `n_workers` equal-sized chunks upfront. Each worker gets one chunk and processes it sequentially. No rebalancing happens during execution.

```python
def chunking(bids, n_workers):
    size = len(bids) // n_workers
    # First n_workers-1 chunks get exactly `size` buildings
    chunks = [bids[i * size:(i + 1) * size] for i in range(n_workers - 1)]
    # Last chunk gets the remainder (slightly larger if len(bids) % n_workers != 0)
    chunks.append(bids[(n_workers - 1) * size:])
    return chunks

def process_chunk(bids):
    """Each worker calls this with its assigned chunk."""
    results = []
    for bid in bids:
        u0, mask = load_data(bid)
        u = jacobi(u0, mask)
        results.append((bid, summary_stats(u, mask)))
    return results

def run_parallel(bids, n_workers):
    chunks = chunking(bids, n_workers)
    with Pool(processes=n_workers) as pool:
        nested = pool.map(process_chunk, chunks)   # blocks until all workers done
    return [item for sub in nested for item in sub]
```

**Why `pool.map(process_chunk, chunks)` rather than `pool.map(jacobi, buildings)`?**

Passing one building at a time would require pickling/unpickling one `(u0, interior_mask)` array pair per building (each ~1 MB) — enormous IPC overhead. By passing a whole chunk, we pay the pickling cost once per worker, not once per building.

**Q: What were the speedup results with static scheduling (N=50)?**

| Workers | Time (s) | Speedup |
|---------|----------|---------|
| 1 | 663.04 | 1.000 |
| 2 | 343.77 | 1.929 |
| 4 | 303.20 | 2.187 |
| 8 | 169.46 | **3.913** |

**Q: What is the parallel fraction F according to Amdahl's Law?**

From the code:

```python
best_s = max(speedups)        # 3.913 at 8 workers
best_w = 8
p = (1/best_s - 1) / (1/best_w - 1)   # Amdahl inverse: solve for F
# p = (1/3.913 - 1) / (1/8 - 1) = (-0.7444) / (-0.875) ≈ 0.851
```

F ≈ **0.851** → 85.1% of work is parallelisable.

**Q: Theoretical maximum speedup?**

```
S_max = 1 / (1 − F) = 1 / 0.149 ≈ 6.70×
```

We achieved 3.91× at 8 workers — 58% of the theoretical ceiling.

**Q: Why does the observed speedup fall short of the theoretical maximum?**

1. **Load imbalance** — buildings in a chunk vary 10× in convergence time. The slowest building in each chunk dictates when that worker finishes. Other workers sit idle waiting.
2. **Process fork overhead** — Python `multiprocessing` uses `fork`; spawning 8 workers has startup cost.
3. **Memory bandwidth saturation** — all 8 workers share the node's memory bus; at 8 workers each worker's throughput drops.

**Q: How long to process all 4571 buildings?**

`169.46 s / 50 × 4571 / 60 ≈ **258 min**`

---

## Task 6 — Dynamic Scheduling

**Q: What is dynamic scheduling and how does it differ from static?**

Instead of pre-assigning fixed chunks, `pool.imap_unordered` maintains a task queue. Workers pull one building at a time, process it, and immediately pull the next. Fast workers automatically do more buildings than slow workers.

```python
# Dynamic scheduling with pool.imap_unordered
with Pool(processes=n_workers,
          initializer=init_worker,
          initargs=(load_dir, engine, max_iter, atol)) as pool:
    iterator = pool.imap_unordered(
        worker_solve_building,   # called once per building (not per chunk)
        building_ids,
        chunksize=chunksize,     # how many tasks to send per IPC message
    )
    results = list(iterator)     # consume the lazy iterator to collect all results
```

vs. static:

```python
# Static scheduling with pool.map + chunks
chunks = chunk_building_ids(building_ids, n_workers)   # pre-divide upfront
nested = pool.map(worker_solve_chunk, chunks)           # one task per worker
```

**Q: Did dynamic scheduling improve performance?**

Yes — at 8 workers, NumPy:

| Scheduler | Wall time (s) | Speedup vs 1 worker |
|-----------|--------------|---------------------|
| Static | 169.60 | 3.65× |
| Dynamic (chunksize=1) | 112.61 | 5.36× |

Dynamic is **~34% faster** at 8 workers.

**Q: Why does dynamic scheduling help here?**

Buildings 10174 and 10180 need ~16,000 iterations (8.7 s each). With static scheduling, whichever worker gets assigned those buildings finishes ~5× later than workers that got easy buildings. With dynamic scheduling, workers that finish fast buildings immediately pick up more work — no core goes idle.

---

## Task 7 — Numba JIT on CPU

**Q: What did the Numba JIT implementation look like?**

The `jacobi` function in `heat_cpu.py` (used by task6-7.py) uses `@numba.njit` with explicit loops:

```python
import numba

@numba.njit(cache=True)
def jacobi_numba_core(u, interior_mask, max_iter, atol):
    rows, cols = u.shape
    for iteration in range(max_iter):
        max_delta = 0.0
        for i in range(1, rows - 1):        # skip border rows
            for j in range(1, cols - 1):    # inner loop = contiguous memory axis
                if interior_mask[i-1, j-1]: # interior_mask is (rows-2, cols-2)
                    u_new = 0.25 * (u[i, j-1] + u[i, j+1] + u[i-1, j] + u[i+1, j])
                    delta = abs(u[i, j] - u_new)
                    if delta > max_delta:
                        max_delta = delta
                    u[i, j] = u_new         # in-place update
        if max_delta < atol:
            return iteration + 1
    return max_iter
```

**Q: How does this exploit the CPU cache?**

The array `u` is C-contiguous (row-major). The inner loop increments `j` — the column index — which is the contiguous dimension. Each `u[i, j±1]` access is adjacent in memory to `u[i, j]`. A 64-byte cache line holds 8 float64 values, so a load for `u[i, j]` also pre-fetches `u[i, j+1]` through `u[i, j+7]` — excellent spatial locality with near-zero cache misses in the hot path.

In contrast, `u[i-1, j]` and `u[i+1, j]` access rows 512 elements apart (~4 KB apart in memory), so they will cause cache misses. But this is unavoidable for a 2D stencil — the two vertical neighbours always span rows.

**Q: How did Numba JIT compare to NumPy reference?**

Single-building benchmark (building 10000):

| Engine | Solver time (s) | Speedup |
|--------|----------------|---------|
| NumPy reference | 6.1924 | 1.0× |
| Numba JIT | **1.3081** | **4.73×** |

Both produce identical results to 4 decimal places.

**Q: What were the multi-worker results with Numba?**

(Dynamic scheduling, N=50, chunksize=1)

| Workers | Wall time (s) |
|---------|--------------|
| 1 | 136.79 |
| 2 | 70.20 |
| 4 | 37.42 |
| 8 | **21.02** |

**Q: Why does `init_worker` matter for Numba?**

```python
def init_worker(load_dir, engine, max_iter, atol):
    WORKER_CONFIG.update({"load_dir": load_dir, "engine": engine, ...})
    if engine == "numba":
        warm_up_numba()   # trigger JIT compilation ONCE per worker at startup
```

The first call to a Numba JIT function triggers compilation (~0.5–2 s). If warm-up is skipped, the first building each worker processes is burdened with compilation overhead, skewing results and wasting time. The `initializer` argument to `Pool` runs `init_worker` once in each worker process at startup, before any buildings are assigned.

**Q: What were the chunksize sweep results?**

(8 workers, Numba, dynamic, N=50)

| Chunksize | Wall time (s) |
|-----------|--------------|
| **1** | **20.85** ← best |
| 2 | 25.70 |
| 4 | 29.12 |
| 8 | 40.34 |

Chunksize=1 wins because buildings vary 10× in time. Larger chunksizes batch multiple buildings into one IPC message — reducing message overhead slightly, but reintroducing load imbalance because a worker committed to 4 hard buildings can't hand one back to an idle worker.

**Q: How long to process all 4571 buildings with Numba + 8 workers?**

`21.02 s / 50 buildings × 4571 / 60 ≈ **32 minutes**` — down from 258 min with NumPy static.

---

## Task 8 — CUDA Kernel with Numba

**Q: How was the CUDA kernel structured?**

```python
from numba import cuda
import math

@cuda.jit
def jacobi_kernel(u, u_new, mask):
    # Map each thread to one grid point (i, j) in the 514×514 padded domain
    i, j = cuda.grid(2)
    n, m = u.shape

    if i < n and j < m:                    # boundary check — grid may be larger than array
        if i == 0 or i == n-1 or j == 0 or j == m-1:
            u_new[i, j] = u[i, j]         # border: just copy (never updated)
        else:
            if mask[i-1, j-1]:            # interior point (mask is 512×512, u is 514×514)
                u_new[i, j] = 0.25 * (u[i, j-1] + u[i, j+1] +
                                       u[i-1, j] + u[i+1, j])
            else:
                u_new[i, j] = u[i, j]    # wall/exterior: copy unchanged
```

The helper function that drives repeated iterations:

```python
def jacobi_numba(u, interior_mask, max_iter):
    # Upload input arrays to GPU once
    u_dev     = cuda.to_device(u)
    mask_dev  = cuda.to_device(interior_mask.astype(np.uint8))
    u_new_dev = cuda.device_array_like(u_dev)   # uninitialized output buffer on GPU

    # Launch config: 16×16 thread block, enough blocks to cover 514×514
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(u.shape[0] / threads_per_block[0])  # ceil(514/16) = 33
    blocks_per_grid_y = math.ceil(u.shape[1] / threads_per_block[1])  # ceil(514/16) = 33
    blocks_per_grid   = (blocks_per_grid_x, blocks_per_grid_y)        # 33×33 = 1089 blocks

    for _ in range(max_iter):
        jacobi_kernel[blocks_per_grid, threads_per_block](u_dev, u_new_dev, mask_dev)
        # Swap buffers: u_new becomes the new u, old u becomes the scratch buffer
        u_dev, u_new_dev = u_new_dev, u_dev

    return u_dev.copy_to_host()   # download result back to CPU
```

**Q: Why a 2D grid (16×16 blocks)?**

The grid is 514×514. A 2D thread layout maps directly to (row, col) — each thread handles exactly one point with `i, j = cuda.grid(2)`. 16×16 = 256 threads per block, which is a common sweet spot for occupancy on NVIDIA GPUs (warp size = 32, so 256 threads = 8 warps per block).

**Q: Why skip the early stopping criterion?**

```python
# If we did this every iteration:
delta = float(cp.abs(u_old - u_new).max())   # forces GPU→CPU sync each iteration
if delta < atol:
    break
```

Each `float(...)` call on a CuPy/CUDA array forces the GPU to finish, copies a scalar to the CPU, and stalls the GPU pipeline while Python evaluates `if delta < atol`. At 20,000 iterations per building, this is 20,000 PCIe round-trips (~microseconds each but multiplied by 20,000 = serious overhead). The fix in Task 8: run a **fixed** `max_iter` iterations with no per-iteration CPU sync.

**Q: What is the buffer swap trick?**

```python
u_dev, u_new_dev = u_new_dev, u_dev
```

After each kernel call, `u_new_dev` holds the updated values. Instead of copying `u_new_dev` back into `u_dev` (expensive GPU memcpy), we just swap the Python variable names. The next kernel call writes into what was `u_dev` (now called `u_new_dev`), alternating buffers. Zero extra data movement.

---

## Task 9 — CuPy GPU Adaptation

**Q: What is CuPy and why is it easy to use here?**

CuPy is a NumPy-compatible array library that runs on NVIDIA GPUs. Every function call dispatches a CUDA kernel automatically. For this project, the reference code is **structurally unchanged** — just replace `import numpy as np` with `import cupy as cp`.

**Original NumPy version:**

```python
import numpy as np

def load_data(load_dir, bid):
    u = np.zeros((514, 514))
    u[1:-1, 1:-1] = np.load(f"{bid}_domain.npy")
    interior_mask = np.load(f"{bid}_interior.npy")
    return u, interior_mask

def jacobi(u, interior_mask, max_iter, atol=1e-4):
    u = np.copy(u)
    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u
```

**CuPy drop-in replacement (`cupy_proj.py`):**

```python
import cupy as cp          # ← only change at import level
import numpy as np         # still needed for np.load (disk I/O)

def load_data(load_dir, bid):
    u = cp.zeros((514, 514))
    # np.load reads from disk → CPU array; cp.asarray copies it to GPU
    u[1:-1, 1:-1] = cp.asarray(np.load(f"{bid}_domain.npy"))
    interior_mask = cp.asarray(np.load(f"{bid}_interior.npy"))
    return u, interior_mask     # now both live on GPU

def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = cp.copy(u)              # all cp.* ops dispatch CUDA kernels automatically
    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = cp.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:        # ← this line is the hidden performance problem (see Task 10)
            break
    return u
```

**Q: What is `cp.asarray` doing?**

`np.load` reads from disk into a CPU NumPy array. `cp.asarray` copies it from CPU RAM to GPU VRAM over the PCIe bus. This host→device transfer happens once per building at load time — acceptable cost.

**Q: What does `summary_stats` look like for CuPy?**

```python
def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]   # still on GPU
    mean_temp  = u_interior.mean()
    std_temp   = u_interior.std()
    # .item() copies a scalar from GPU to CPU (1 value, negligible cost)
    pct_above_18 = (cp.sum(u_interior > 18) / u_interior.size * 100).item()
    pct_below_15 = (cp.sum(u_interior < 15) / u_interior.size * 100).item()
    return {'mean_temp': mean_temp, 'std_temp': std_temp,
            'pct_above_18': pct_above_18, 'pct_below_15': pct_below_15}
```

---

## Task 10 — Profiling CuPy with nsys

**Q: What was the main performance issue found?**

The `if delta < atol:` check inside the Jacobi loop forces a **host-device synchronisation every single iteration**:

```python
delta = cp.abs(...).max()   # launches a GPU reduction kernel → result is a CuPy 0-d array on GPU
if delta < atol:            # Python evaluates this → must copy `delta` from GPU to CPU
                            # this implicit D→H copy BLOCKS until the GPU finishes + transfers 8 bytes
```

In the nsys timeline this appears as thousands of tiny `cudaMemcpy Device→Host` calls, each only transferring 8 bytes but each requiring a full synchronisation fence. With 20,000 iterations per building, this is 20,000 GPU→CPU stalls.

**Q: How was it fixed (`cupy_proj_new.py`)?**

```python
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = cp.copy(u)
    for i in range(max_iter):
        # Check convergence only every 100 iterations to amortise sync cost
        if i % 100 == 0:
            u_new_temp = 0.25 * (u[..., 1:-1, :-2] + u[..., 1:-1, 2:] +
                                  u[..., :-2, 1:-1] + u[..., 2:, 1:-1])
            delta = cp.max(cp.abs(u[..., 1:-1, 1:-1] - u_new_temp) * interior_mask)
            if delta < atol:                              # sync only every 100 steps
                u[..., 1:-1, 1:-1] = cp.where(interior_mask, u_new_temp, u[..., 1:-1, 1:-1])
                break

        # Normal update step — no sync, GPU runs freely
        u[..., 1:-1, 1:-1] = jacobi_kernel(
            u[..., 1:-1, 1:-1],
            u[..., 1:-1, :-2],    # left
            u[..., 1:-1, 2:],     # right
            u[..., :-2, 1:-1],    # up
            u[..., 2:, 1:-1],     # down
            interior_mask
        )
    return u
```

The `@cp.fuse()` decorator on `jacobi_kernel` fuses the element-wise operations into a single CUDA kernel, reducing memory traffic:

```python
@cp.fuse()
def jacobi_kernel(u_center, u_left, u_right, u_up, u_down, mask):
    u_new = 0.25 * (u_left + u_right + u_up + u_down)
    return cp.where(mask, u_new, u_center)   # only write interior points
```

**What is `@cp.fuse()`?**

Without fusion, each arithmetic operator (`+`, `*`) and `cp.where` launches a separate CUDA kernel and reads/writes the full array to GPU VRAM each time. With `@cp.fuse()`, CuPy fuses all the operations into one kernel that reads each element once, computes everything, and writes once — reducing VRAM bandwidth by ~5×.

**Q: What are the two improvements in `cupy_proj_new.py`?**

1. **Convergence check every 100 steps** instead of every step → reduces GPU→CPU syncs from ~20,000 to ~200 per building
2. **`@cp.fuse()` kernel** → fuses 5 array ops into 1 kernel, reducing VRAM bandwidth

**Q: What about the batched approach in `cupy_proj_new.py`?**

```python
BATCH_SIZE = 200

for i in range(0, N, BATCH_SIZE):
    batch_bids = building_ids[i:i + BATCH_SIZE]
    current_batch_size = len(batch_bids)

    # Load an entire batch of buildings onto the GPU at once
    batch_u0 = cp.empty((current_batch_size, 514, 514), dtype=cp.float32)
    batch_interior_mask = cp.empty((current_batch_size, 512, 512), dtype='bool')

    for j, bid in enumerate(batch_bids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        batch_u0[j] = u0
        batch_interior_mask[j] = interior_mask

    # Run Jacobi on all buildings in the batch simultaneously
    batch_u = jacobi(batch_u0, batch_interior_mask, MAX_ITER, ABS_TOL)
```

By stacking buildings along a batch dimension, the GPU processes multiple floorplans in parallel within each kernel call. This amortises kernel launch overhead and better utilises GPU parallelism when individual buildings don't saturate all GPU cores alone.

---

## Task 11 — Job Array

**Q: How was the job array structured?**

`#BSUB -J "task11[1-4]"` launches 4 LSF array elements. Each element reads `$LSB_JOBINDEX` (1-based) to determine its shard index. Buildings are sharded with **interleaved stride** so each shard gets an even mix of fast and slow buildings:

```python
def shard_building_ids(building_ids, task_index, task_count):
    # Interleaved: element 0 gets [0, 4, 8, ...], element 1 gets [1, 5, 9, ...], etc.
    return building_ids[task_index::task_count]
```

vs block sharding (worse):

```python
# Block sharding — could put all "hard" buildings in one shard
size = len(building_ids) // task_count
return building_ids[task_index*size:(task_index+1)*size]
```

**Q: How does each element read the array index from LSF?**

```python
env_jobindex = environ.get("LSB_JOBINDEX")    # e.g. "3" for array element [3]
env_jobcount = environ.get("LSB_JOBINDEX_END") # e.g. "4" for [1-4]

# Convert to 0-based Python index
task_index = int(env_jobindex) - 1   # LSB_JOBINDEX is 1-based → convert to 0-based
task_count = int(env_jobcount)       # total number of array elements
```

**Q: What does each job array element do?**

Each element runs its shard of ~1143 buildings using:
- Engine: Numba JIT (fastest CPU solver)
- Scheduler: dynamic (`pool.imap_unordered`)
- Workers: 8
- Chunksize: 1

```python
with Pool(processes=n_workers,
          initializer=init_worker,
          initargs=(load_dir, max_iter, atol)) as pool:
    iterator = pool.imap_unordered(
        worker_solve_building,
        shard_ids,
        chunksize=1,
    )
    results = list(iterator)
```

**Q: How are the 4 shard results merged?**

A separate `task11-merge.py` reads all 4 CSV files and concatenates them:

```python
# Reads: task11_array_001_of_004_results.csv, ..._002_..., ..._003_..., ..._004_...
merged = pd.concat([pd.read_csv(f) for f in shard_csvs])
merged.sort_values("building_id").to_csv("task11_array_merged_results.csv", index=False)
```

**Q: What were the final Task 11 results?**

| Metric | Value |
|--------|-------|
| Total buildings | 4571 |
| Mean solver time per building | 4.77 s |
| Total solver time | 21,791.6 s |
| Total Jacobi iterations | 39,746,490 |

---

## Task 12 — Final Analysis (All 4571 Buildings)

**Q: How was analysis done with Pandas?**

```python
import pandas as pd

df = pd.read_csv("task11_array_merged_results.csv")

# a) Distribution of mean temperatures
df["mean_temp"].hist(bins=40)

# b) Average mean temperature
print(df["mean_temp"].mean())               # → 14.69°C

# c) Average std deviation
print(df["std_temp"].mean())                # → 6.80°C

# d) Buildings with ≥50% area above 18°C (mold-safe)
above_18 = (df["pct_above_18"] >= 50).sum()

# e) Buildings with ≥50% area below 15°C (too cold)
below_15 = (df["pct_below_15"] >= 50).sum()
```

**Q: What were the aggregate statistics?**

| Statistic | Mean | Min | Max |
|-----------|------|-----|-----|
| `mean_temp` (°C) | **14.69** | 6.82 | 21.44 |
| `std_temp` (°C) | 6.80 | 4.00 | 8.81 |
| `pct_above_18` (%) | 38.93 | 4.52 | 80.88 |
| `pct_below_15` (%) | 51.96 | 13.60 | 93.35 |

**Q: Is Wall Heating effective?**

Based on simulation over all 4571 buildings:
- Average room temperature is only **14.69°C** — below the 20°C comfort target
- On average, **~52% of interior area is below 15°C** (too cold for human comfort)
- Only **~39% of interior area is above 18°C** (mold-safe)
- Conclusion: Wall Heating is **insufficient as a standalone solution** for most buildings, especially those with large rooms far from inside walls or high ratios of load-bearing wall to inside wall

---

## Speedup Summary

| Solution | ~Wall time for all 4571 buildings |
|----------|----------------------------------|
| Serial NumPy (1 core) | ~890 min (14.8 hrs) |
| NumPy + static + 8 workers | ~258 min (4.3 hrs) |
| NumPy + dynamic + 8 workers | ~172 min (2.9 hrs) |
| **Numba JIT + dynamic + 8 workers** | **~32 min** |
| Numba JIT + job array (4 × 8 workers) | ~8 min (estimated) |
| CuPy (V100, fused kernel, batch=200) | ~2–5 min (estimated) |
| CUDA custom kernel + job array | ~1–2 min (estimated) |

**Total speedup achieved: ~28× (serial → Numba 8-worker) up to ~500× (serial → GPU)**

---

## Key Concepts Reference

**Why `u_new[interior_mask]` instead of updating in-place directly?**

```python
# This is WRONG for Jacobi — it uses already-updated values in the same iteration:
for i in ...:
    for j in ...:
        u[i,j] = 0.25 * (u[i,j-1] + u[i,j+1] + u[i-1,j] + u[i+1,j])  # ← u[i,j-1] already updated!

# This is correct Jacobi — all updates based on values from the previous iteration:
u_new = 0.25 * (u[1:-1, :-2] + ...)   # read all old values first
u[interior] = u_new[interior]          # then write all new values at once
```

**Why does `atol=1e-4` converge differently from `atol=1e-6`?**

Tighter tolerance → more iterations → more accurate temperature field, but also slower. The reference uses `atol=1e-4`; the CuPy version uses `atol=1e-6` which can require significantly more iterations on complex floorplans.

**What is `MAX_ITER=20_000` for?**

Safety cap. Without it, a pathological building geometry could make Jacobi never converge (e.g., infinite corridor). With `MAX_ITER`, the solver always terminates and returns the best approximation reached so far.
