# 02613 Python HPC — Tips, Tricks & Pitfalls

> **Root files:** [STUDY_GUIDE](STUDY_GUIDE.md) · [Exam Review](exam_review.md) · [Cheat Sheet](master_cheat_sheet.md) · [Tips & Pitfalls](tips_and_tricks.md) · [README](README.md)

---

## Contents

- [Golden Rules of HPC](#golden-rules-of-hpc)
- [Measurement Pitfalls](#measurement-pitfalls)
  - [Never benchmark on the login node](#never-benchmark-on-the-login-node)
  - [Always use `perf_counter`](#always-use-perf_counter-not-timetime)
  - [Use enough repetitions](#use-enough-repetitions-for-fast-operations)
  - [Numba JIT warmup](#numba-jit-warmup-exclude-the-first-call-from-benchmarks)
  - [Separate compute from I/O time](#separate-compute-time-from-io-time-in-benchmarks)
  - [wall time vs CPU time](#wall-time-vs-cpu-time-in-parallel-code)
- [Memory & Cache Pitfalls](#memory--cache-pitfalls)
  - [Column access is slow](#column-access-is-slow-in-row-major-arrays)
  - [Inner loop / smallest stride](#inner-loops-should-iterate-over-the-smallest-stride)
  - [float64 vs float32 memory](#float64-uses-2x-memory-vs-float32)
  - [Fancy indexing returns copies](#fancy-indexing-returns-copies-not-views)
  - [Unnecessary copies](#unnecessary-copies-explode-memory-usage)
  - [NUMA](#numa-the-hidden-performance-wall-on-multi-socket-nodes)
- [Parallelism Pitfalls](#parallelism-pitfalls)
  - [GIL blocks Python threads](#python-threads-cannot-run-python-code-in-parallel-gil)
  - [Too-fine-grained tasks](#too-fine-grained-tasks-communication-dominates-compute)
  - [Oversubscription](#oversubscription-more-threads-than-allocated-cores)
  - [Double parallelism](#double-parallelism-stacking-thread-pool-on-top-of-multi-threaded-numpy)
  - [Missing span[hosts=1]](#not-using-spanhosts1)
  - [Missing `__main__` guard](#forgetting-if-__name__--__main__-guard)
  - [Load imbalance](#load-imbalance-some-workers-finish-much-earlier-than-others)
- [Reduction Pitfalls](#reduction-pitfalls)
  - [Associativity requirement](#the-reduction-operator-must-be-associative-and-commutative)
  - [Float non-associativity](#floating-point-non-associativity-changes-results-after-parallelisation)
  - [Simple reduction scaling](#simple-reduction-does-not-scale-to-many-processors)
- [Job Submission Pitfalls](#job-submission-pitfalls--be-a-good-hpc-citizen)
  - [Running on login node](#running-heavy-computation-on-the-login-node-affects-everyone)
  - [Using `tee`](#using-tee-in-job-scripts-doubles-disk-io)
  - [Slow -o/-e channel](#the--o-e-channel-is-slow-for-high-volume-output)
  - [Spamming bjobs](#spamming-bjobsbstat-hurts-the-scheduler)
  - [No wall time limit](#not-setting-a-wall-time-limit-job-runs-forever-blocks-queue)
  - [Forgetting conda activate](#forgetting-to-activate-conda-environment-in-job-script)
  - [Too many files in one directory](#creating-too-many-files-in-one-directory-filesystem-strain)
  - [Orphan background processes](#leaving-background-processes-running-wastes-walltime)
  - [Requesting GPU but not using it](#requesting-gpu-but-not-using-it)
- [NumPy Gotchas](#numpy-gotchas)
  - [Integer overflow](#integer-overflow-with-small-dtypes)
  - [In-place vs out-of-place](#in-place-vs-out-of-place-operations)
  - [Broadcasting mistakes](#broadcasting-mistakes)
  - [Strides and non-contiguous arrays](#strides-and-non-contiguous-arrays)
- [GPU / Numba Gotchas](#gpu--numba-gotchas)
  - [Transfer dominates runtime](#host-device-data-transfer-dominates-runtime-for-simple-kernels)
  - [JIT warmup](#jit-warmup-time-the-second-call-not-the-first)
  - [Thread divergence](#thread-divergence-kills-gpu-performance)
  - [Shared memory limit](#shared-memory-is-limited-48-kb-per-block)
  - [Bounds check required](#always-bounds-check-in-gpu-kernels)
- [Debugging HPC Code](#debugging-hpc-code)
  - [Step 1: Profile first](#step-1-profile-before-optimising)
  - [Step 2: Scale profiler result](#step-2-scale-the-profiler-result-to-the-actual-workload)
  - [Step 3: Check correctness](#step-3-check-correctness-after-parallelising)
  - [Step 4: Start small](#step-4-start-with-small-data-scale-up-gradually)
  - [Step 5: Diagnose thread count](#step-5-diagnose-thread-count-problems)
  - [Step 6: Diagnose I/O slowness](#step-6-diagnose-io-slowness)
- [Exam-Specific Tips (MCQ)](#exam-specific-tips-mcq-format)
  - [General MCQ Strategy](#general-mcq-strategy)
  - [Amdahl's Law](#topic-amdahls-law-appears-on-every-exam)
  - [LSF/BSUB Scripts](#topic-lsfbsub-job-scripts-appears-on-every-exam)
  - [Reduction Operators](#topic-reduction-operators-commutativity--associativity)
  - [NumPy dtype, reshape, float precision](#topic-numpy-dtype-reshape-float-precision)
  - [Profiler output](#topic-profiler-output-cprofile--line_profiler--nsys)
  - [CUDA threads, blocks, grid](#topic-cuda-threads-blocks-grid)
  - [Cache & memory layout](#topic-cache--memory-layout)
  - [Broadcasting](#topic-broadcasting)
  - [GIL, multiprocessing, threading](#topic-gil-multiprocessing-threading)
  - [Time Management](#time-management)
  - [Formula Reference Card](#formula-reference-card)
- [Complete Good Job Script Template](#complete-good-job-script-template)

---

## Golden Rules of HPC

These are the principles that separate good HPC practitioners from users who waste compute time and frustrate their colleagues.

1. **Profile before you optimise.** You cannot improve what you have not measured. Run a profiler, find the actual bottleneck, and fix that — not the code you think is slow.
2. **Never benchmark on the login node.** It is a shared machine with unpredictable load. Every timing you take there is noise. Submit a batch job and time it on a dedicated compute node.
3. **Treat cluster resources as a shared commons.** Every core you over-subscribe, every job you leave running past its useful life, and every `bstat` call you spam degrades the system for every other user. Behave accordingly.
4. **Parallelism has overhead.** Spawning 1,000,000 tasks for 1,000,000 Monte Carlo samples is slower than serial. Chunk your work so each process does a meaningful amount. The communication cost must be smaller than the compute savings.
5. **Memory layout determines speed as much as algorithm choice.** A column-access loop on a large row-major array can be 12x slower than the equivalent row-access loop — with identical FLOP count. Know your strides.
6. **The serial fraction is the ceiling.** Amdahl's Law is not pessimism — it is a hard physical limit. If 20% of your program cannot be parallelised, you will never exceed 5x speedup no matter how many cores you add. Reduce the serial fraction before adding cores.
7. **Your code may be fine; your dependencies may not be.** NumPy's BLAS backend may spawn 32 threads when you allocated 4 cores. A package you import may use `tee`. Always inspect what your dependencies are doing under the hood.

---

## Measurement Pitfalls

### Never benchmark on the login node

The login node (`login.hpc.dtu.dk`) is a shared machine. Other users are compiling code, transferring files, and running interactive sessions. Any timing you take there reflects noise from those other processes, not your program's actual performance. Always submit a batch job to a dedicated compute node.

```bash
# Wrong: running timing experiments interactively on the login node
ssh s123456@login.hpc.dtu.dk
python -c "import time; t=time.perf_counter(); do_work(); print(time.perf_counter()-t)"

# Right: submit a job with a specific CPU model, get a clean node
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "select[model==XeonGold6226R]"
python benchmark.py
```

### Always use `perf_counter`, not `time.time`

`time.time()` returns wall-clock time but can be affected by NTP adjustments and has lower resolution on some platforms. `time.perf_counter()` uses the highest-resolution monotonic clock available and is the correct choice for measuring elapsed time.

```python
# Wrong
import time
t = time.time()
do_work()
elapsed = time.time() - t

# Right
from time import perf_counter
t = perf_counter()
do_work()
elapsed = perf_counter() - t
```

### Use enough repetitions for fast operations

A single timing measurement of a microsecond-scale operation captures mostly timer overhead and cache warm-up effects, not the actual cost. Repeat the operation many times and divide.

```python
from time import perf_counter
import numpy as np

SIZE = 100
n_repeat = 1000
mat = np.random.rand(SIZE, SIZE)

t = perf_counter()
for _ in range(n_repeat):
    result = mat[0, :] * 1.01
elapsed = (perf_counter() - t) / n_repeat

print(f"Per-operation time: {elapsed * 1e6:.3f} microseconds")
```

### Numba JIT warmup: exclude the first call from benchmarks

The first call to any Numba JIT-compiled function (CPU or GPU) includes compilation time. This is roughly constant regardless of problem size, but can be 50-100x slower than subsequent calls. Always warm up before timing.

```python
from numba import jit
from time import perf_counter

@jit(nopython=True)
def my_func(arr):
    return arr.sum()

arr = np.random.rand(1_000_000)

# Wrong: includes compilation in the timing
t = perf_counter()
result = my_func(arr)
print(perf_counter() - t)  # Measures compilation + execution

# Right: warm up first, then time
my_func(arr)  # Trigger compilation, discard result

t = perf_counter()
result = my_func(arr)
print(perf_counter() - t)  # Measures execution only
```

### Separate compute time from I/O time in benchmarks

When benchmarking I/O (e.g., Blosc vs numpy.save), the OS page cache can make the second write appear much faster because the data is already in memory. Use `os.sync()` to flush the page cache between writes for a fair comparison.

```python
import os
import numpy as np
from time import perf_counter

arr = np.zeros((512, 512, 512), dtype='uint8')

t = perf_counter()
np.save('data.npy', arr)
os.sync()  # Flush OS page cache — required for fair benchmarking
elapsed = perf_counter() - t
print(f"Write: {elapsed:.3f}s")
```

### `wall time` vs `CPU time` in parallel code

The `time` command reports three values: `real` (wall time), `user` (CPU time summed across cores), `sys`. When parallelising:
- `real` goes down as cores are added (this is speedup)
- `user` stays roughly constant or increases (same total CPU work, now distributed)

Do not confuse them. Speedup is always measured using `real` (wall time).

---

## Memory & Cache Pitfalls

### Column access is slow in row-major arrays

NumPy arrays are stored in row-major (C) order by default. `mat[i, :]` reads contiguous memory; `mat[:, j]` reads every N-th element, causing a cache miss on almost every access. The difference can be a 12x slowdown for large matrices.

```python
# Cache-friendly: reads contiguous memory
row = mat[0, :]

# Cache-unfriendly: reads every SIZE-th element in memory
col = mat[:, 0]
```

**Real measured difference (18K x 18K image):**
- Row-wise downsampling: 1.67 seconds
- Column-wise downsampling: 20.15 seconds — 12x slower with identical FLOP count

**Fix:** transpose the data so the axis you iterate over is the last dimension, or restructure your algorithm to access rows.

### Inner loops should iterate over the smallest stride

Given an array with strides, sort the axes by stride and make the smallest-stride axis the innermost loop. This maximises spatial locality.

```python
# Example: images array with strides (600, 40, 8, 200)
# Axis 0: stride 600, Axis 1: stride 40, Axis 2: stride 8, Axis 3: stride 200
# Sort ascending: axis 2 (8) < axis 1 (40) < axis 3 (200) < axis 0 (600)
# Inner to outer loop order: k (axis 2), j (axis 1), l (axis 3), i (axis 0)

for i in range(images.shape[0]):       # outer (stride 600)
    for l in range(images.shape[3]):   # (stride 200)
        for j in range(images.shape[1]):  # (stride 40)
            for k in range(images.shape[2]):  # inner (stride 8, fastest)
                process(images[i, j, k, l])
```

### float64 uses 2x memory vs float32

Check whether you need double precision. float32 gives ~7 significant decimal digits; float64 gives ~15. For most scientific computation, float32 is sufficient and halves memory usage, which means more data fits in cache.

```python
# 512x512x512 array
arr_f64 = np.zeros((512, 512, 512), dtype='float64')  # 1024 MB
arr_f32 = np.zeros((512, 512, 512), dtype='float32')  # 512 MB — fits in cache better
```

### Fancy indexing returns copies, not views

Slicing with `arr[2:5]` returns a view — no copy, same memory. Fancy indexing with `arr[[0, 2, 4]]` returns a copy. This matters both for memory usage and for in-place modification.

```python
# View — modifying this modifies the original
row = arr[0, :]
row[:] = 0  # This changes arr

# Copy — modifying this does NOT modify arr
selection = arr[[0, 2, 4]]
selection[:] = 0  # arr is unchanged
```

### Unnecessary copies explode memory usage

Check whether you need a copy or a view. `arr.T` is a view (no copy, just changed strides). `arr.copy()` allocates new memory. Operations like `arr + 0` create a copy.

```python
# These create views (no copy):
view = arr.T
view = arr.reshape(-1)
view = arr[2:10]

# These create copies (new allocation):
copy = arr.copy()
copy = arr[[0, 1, 2]]    # fancy indexing
copy = arr + 0           # arithmetic always creates a new array
```

### NUMA: the hidden performance wall on multi-socket nodes

DTU HPC nodes (XeonGold6226R) have two physical CPU sockets, each with its own local RAM. Without `numactl`, all memory is allocated on socket 0. Cores on socket 1 must cross the inter-socket link for every memory access. Result: speedup improves up to ~50% of cores, then plateaus or drops.

```bash
# Wrong: memory all on socket 0, socket 1 cores are slow
python -u reduction.py

# Right: interleave memory across both sockets
numactl --interleave=all python -u reduction.py
```

`numactl --interleave=all` distributes memory pages round-robin across all NUMA nodes, giving all cores roughly equal latency.

---

## Parallelism Pitfalls

### Python threads cannot run Python code in parallel (GIL)

The CPython GIL (Global Interpreter Lock) ensures only one thread executes Python bytecode at a time. For CPU-bound pure Python code, adding threads gives zero speedup and adds overhead.

```python
# This looks parallel but is not — GIL serialises the Python loops
from multiprocessing.pool import ThreadPool

def manual_sum(arr):
    total = 0
    for x in arr:     # pure Python loop — GIL-bound
        total += x
    return total

# Adding threads makes this SLOWER, not faster
with ThreadPool(4) as pool:
    results = pool.map(manual_sum, chunks)
```

**Exception:** NumPy releases the GIL during BLAS/LAPACK calls. `ThreadPool` works for NumPy-heavy computation because the GIL is released during the actual number-crunching.

```python
# This works — NumPy releases the GIL during np.sum()
from multiprocessing.pool import ThreadPool
import numpy as np

with ThreadPool(4) as pool:
    results = pool.map(np.sum, array_chunks)  # True parallelism
```

**Rule of thumb:**
- Pure Python loops → use `multiprocessing` (separate processes, each with own GIL)
- NumPy operations → `ThreadPool` works (GIL released)
- Numba with `@jit(nogil=True)` → `ThreadPool` works

### Too-fine-grained tasks: communication dominates compute

Spawning one task per unit of work is catastrophically slow if each unit is trivial. Inter-process communication has fixed overhead per message.

```python
# Wrong: 1,000,000 tasks, each doing 1 operation
# Process overhead >> actual computation time
results = pool.map(sample_one_point, range(1_000_000))

# Right: 10 tasks, each doing 100,000 operations
chunk_size = 1_000_000 // n_proc
results = pool.map(sample_multiple, [chunk_size] * n_proc)
```

Measured result from the course: the naive (one-task-per-sample) Monte Carlo is slower than serial. The chunked version achieves near-linear speedup.

### Oversubscription: more threads than allocated cores

Scientific packages (NumPy, SciPy, scikit-learn) auto-detect hardware core count and spawn threads for all available cores. If your LSF job was allocated 4 cores on a 32-core machine, these packages will spawn 32 threads, each fighting for the 4 allocated cores.

```bash
# Wrong: allocated 4 cores, but NumPy uses all 32 hardware cores
#BSUB -n 4
python -u script.py

# Right: tell packages to use only the allocated cores
#BSUB -n 4
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
MPI_NUM_THREADS=4
python -u script.py
```

**Diagnose this:** check `Max Threads` in the LSF job summary at the end of the `.out` file. If it is much larger than `#BSUB -n`, you have an oversubscription problem.

### Double parallelism: stacking thread pool on top of multi-threaded NumPy

A ThreadPool whose workers call multi-threaded NumPy creates O(n^2) threads.

```python
# Wrong: 8 threads × 8 NumPy internal threads = 64 threads fighting for 8 cores
# This is SLOWER than either strategy alone
with ThreadPool(8) as pool:
    C = np.concatenate(pool.starmap(np.matmul, zip(A, B)))
# Result: ~1.87s — worse than pure NumPy threading (1.28s)
```

**Fix:** when using your own pool, disable package-level threading:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -u matmuls_threadpool.py
# Result: ~1.33s — back to competitive
```

### Not using `span[hosts=1]`

Without `span[hosts=1]`, LSF may distribute your allocated cores across multiple physical nodes. Shared-memory parallelism (`mp.Pool`, `ThreadPool`, `mp.RawArray`) requires all workers to be on the same node. If they are not, workers cannot share memory.

```bash
# Wrong: cores may be on different nodes
#BSUB -n 8

# Right: force all cores onto one node
#BSUB -n 8
#BSUB -R "span[hosts=1]"
```

### Forgetting `if __name__ == '__main__':` guard

On macOS and Windows, multiprocessing uses the `spawn` method: child processes import the parent script. Without the guard, each child process triggers more pool creation, creating an infinite spawn loop.

```python
# Wrong: pool creation runs in every child process
pool = mp.Pool(4)
results = pool.map(my_func, data)

# Right: pool creation only runs in the main process
if __name__ == '__main__':
    pool = mp.Pool(4)
    results = pool.map(my_func, data)
```

### Load imbalance: some workers finish much earlier than others

If different work items take different amounts of time (e.g., Mandelbrot boundary pixels take more iterations than interior pixels), equal-sized static assignments leave fast workers idle while slow workers finish their last chunks.

```python
# Wrong: large chunks assigned statically — boundary pixels take much longer
# Fast workers sit idle while one slow worker finishes a boundary chunk
pool.map(func, items, chunksize=len(items) // n_proc)

# Right: small chunks allow dynamic reassignment to idle workers
pool.map(func, items, chunksize=100)
```

**Rule:** use dynamic scheduling (small `chunksize`) when task duration varies significantly. Use static (large chunks) when tasks are roughly equal in cost.

---

## Reduction Pitfalls

### The reduction operator must be associative AND commutative

For a parallel reduction tree, the operator must be:
1. **Associative**: `(a OP b) OP c == a OP (b OP c)` — required to split into subproblems
2. **Commutative**: `a OP b == b OP a` — required because implementations may reorder inputs

**Valid reduction operators:** sum, product, min, max, logical AND/OR, bitwise XOR, set intersection.

**Invalid:** matrix multiplication (not commutative: AB ≠ BA).

**Tricky case:** `abssum(x, y) = abs(x + y)` is commutative but NOT associative:

```python
# Test associativity with a concrete counterexample:
# (1 + 2) + (-3) → |3| + (-3) → |0| = 0
# 1 + (2 + (-3)) → 1 + |-1| → |1 + 1| = 2
# 0 ≠ 2 — not associative, cannot use in parallel reduction tree

# Fix: do the sum in parallel, then take abs once at the end
total = parallel_sum(data)
result = abs(total)
```

Always test with a concrete counterexample. Commutativity alone is not sufficient.

### Floating-point non-associativity changes results after parallelisation

Floating-point arithmetic is not truly associative due to rounding. A parallel reduction that changes the order of additions will produce a slightly different numerical result than the serial version. This is expected and usually acceptable, but can cause confusion when debugging correctness.

```python
# Serial
result_serial = sum(data)

# Parallel reduction may give a slightly different answer
# due to different summation order — this is normal
result_parallel = parallel_reduce(data)
assert result_serial == result_parallel    # May FAIL due to floating-point rounding
assert abs(result_serial - result_parallel) < 1e-10  # Use tolerance instead
```

### Simple reduction does not scale to many processors

A flat reduction that splits N elements into T chunks, reduces each chunk in parallel, then sums T partial results serially has maximum theoretical speedup of sqrt(N)/2 at T_opt = sqrt(N) tasks. Beyond that, the serial final summation dominates.

Binary tree reduction achieves N/log2(N) speedup — much better for large N.

| N | Simple max speedup | Binary tree max speedup |
|---|---|---|
| 100 | 5x | 15x |
| 100,000 | 158x | ~5,900x |

---

## Job Submission Pitfalls — Be a Good HPC Citizen

### Running heavy computation on the login node (affects everyone!)

The login node is shared by all users. Running CPU or memory-intensive work there degrades the experience for everyone on the cluster. Use `linuxsh` for interactive work on a compute node, or submit a batch job.

```bash
# Wrong: running heavy computation on the login node
ssh s123456@login.hpc.dtu.dk
python -u heavy_computation.py    # This slows down the login node for everyone

# Right: move to a compute node first
linuxsh                           # Interactive compute node session
python -u heavy_computation.py

# Or better: submit as a batch job
bsub < submit.sh
```

### Using `tee` in job scripts (doubles disk I/O)

`tee` writes output to both a file and stdout. In a batch job, LSF captures that stdout into the `-o` file. Every byte gets written twice, doubling filesystem load.

```bash
# Wrong: every byte written twice
python -u script.py | tee output.txt
# → writes to output.txt AND stdout → LSF writes stdout to -o file → 2x I/O

# Right: pick one destination
python -u script.py > /work3/02613/dump/output_${LSB_JOBID}.txt
```

### The `-o`/`-e` channel is slow for high-volume output

LSF's internal channel for capturing stdout/stderr has significant overhead. For 100,000 lines of output: 80 seconds via `-o`/`-e` vs 3 seconds with manual shell redirection — a 26x difference.

```bash
# Slow: all program output flows through LSF's -o channel
#BSUB -o name_%J.out
python -u verbose_script.py

# Fast: redirect program output manually to /work3
#BSUB -o /work3/02613/dump/name_%J.out   # LSF job summary still captured here
python -u verbose_script.py \
    1> /work3/02613/dump/output_${LSB_JOBID}.txt \
    2> /work3/02613/dump/error_${LSB_JOBID}.txt
```

Note: keep `-o`/`-e` for the LSF job summary that appears after completion. Redirect the actual program output separately so the LSF channel carries almost nothing.

### Spamming `bjobs`/`bstat` (hurts the scheduler)

Every call to `bstat` or `bjobs` interrupts the LSF scheduler. The scheduler only updates job data 1-2 times per minute, so polling every 2 seconds is pointless as well as harmful. `watch -n 0.1 bstat` is described in the course lecture as "totally insane."

```bash
# Wrong: interrupts the scheduler every 2 seconds
watch bstat
watch -n 0.1 bstat    # absolutely do not do this

# Right: check manually, or poll at most once per minute
bstat                  # run once, check the status
watch -n60 bstat       # or once per minute if you must watch
```

### Not setting a wall time limit (job runs forever, blocks queue)

If you do not set `-W`, your job may run indefinitely, consuming resources that other users could use. Always set a realistic wall time limit. If the job exceeds it, LSF kills it and you see `TERM_RUNLIMIT` in the output.

```bash
# Wrong: no wall time — job may run forever
#BSUB -q hpc
python -u script.py

# Right: set a realistic wall time
#BSUB -W 02:00    # 2 hours; kills job if it exceeds this
```

### Forgetting to activate conda environment in job script

The conda environment is not activated by default in batch jobs. If you forget, Python will use the system Python, which lacks the course packages.

```bash
#!/bin/bash
#BSUB -q hpc
# ... other BSUB flags ...

# Required: activate the course conda environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Now your script has access to numpy, numba, etc.
python -u my_script.py
```

### Creating too many files in one directory (filesystem strain)

Shared filesystems use metadata servers to track directory contents. Thousands of files in one directory causes slowdowns for all operations. `ls` on a directory with 100,000 files takes 2.6 seconds. `ls *.jpg` with 100,000 files gives "Argument list too long".

```
# Wrong: 100,000 files in one directory
images/
    000001.jpg
    000002.jpg
    ...
    100000.jpg

# Right: at most ~1000 files per directory
images/
    000000/
        000000.jpg ... 000999.jpg
    001000/
        001000.jpg ... 001999.jpg
```

For scientific array data, use `zarr.storage.NestedDirectoryStore` instead of `DirectoryStore`. The nested variant distributes chunk files across a directory hierarchy.

### Leaving background processes running (wastes walltime)

A monitoring script started with `&` in a job script may run an infinite loop. When the main Python script finishes, LSF waits for all child processes to terminate before ending the job. If the monitor runs forever, the job occupies resources until the walltime limit.

```bash
# Wrong: monitor runs forever, job wastes resources until -W limit
mymonitor > monitor_$LSB_JOBID.out &
python -u script.py
# script exits after 1 hour, monitor keeps running for 23 more hours

# Right: kill the monitor when work is done
mymonitor > monitor_$LSB_JOBID.out &
MONITOR_PID=$!
python -u script.py
kill $MONITOR_PID    # terminate monitor immediately when work finishes
wait $MONITOR_PID
```

### Requesting GPU but not using it

If your job script requests a GPU queue or GPU resource but your code runs only on the CPU, you are occupying GPU resources that other users could use. Check that your code actually runs on the GPU, and check the GPU utilisation during the job.

---

## NumPy Gotchas

### Integer overflow with small dtypes

NumPy integer arithmetic wraps silently. Adding 1 to a uint8 array where values are already 255 gives 0, not an error.

```python
arr = np.array([254, 255], dtype='uint8')
print(arr + 1)   # array([255, 0]) — 255 + 1 wraps to 0

# Check before recoding: make sure min/max fit in the target dtype
# uint8: 0 to 255
# int8: -128 to 127
# int16: -32768 to 32767
# uint32: 0 to 4,294,967,295
```

### In-place vs out-of-place operations

`arr *= 2` modifies the array in-place (no new allocation). `arr = arr * 2` creates a new array. This matters for memory usage and for aliased views.

```python
arr = np.zeros((1000, 1000))
view = arr[0, :]

arr *= 2      # In-place: view still refers to same memory, both change
arr = arr * 2 # Creates new array: view still points to the old arr
```

### Broadcasting mistakes

NumPy broadcasts by right-aligning shapes and padding with 1s on the left. A common mistake is getting the axis wrong when subtracting a per-image mean from a stack of images.

```python
images = np.zeros((100, 480, 640, 3))  # N x H x W x C
mean_pixels = np.zeros((100, 3))        # N x C

# Wrong: shapes (100, 480, 640, 3) and (100, 3) do not broadcast as intended
images - mean_pixels

# Right: insert spatial axes explicitly
images - mean_pixels[:, None, None, :]  # (100, 1, 1, 3) broadcasts to (100, 480, 640, 3)
```

Broadcasting rule: right-align shapes, pad shorter shape with 1s on the left, then each dimension must either match or be 1.

### Strides and non-contiguous arrays

Some NumPy operations fail or slow down on non-contiguous arrays. After transposing, slicing, or reshaping, check `arr.flags['C_CONTIGUOUS']`. Use `np.ascontiguousarray()` to force a contiguous copy when needed.

```python
arr = np.random.rand(100, 200)
T = arr.T           # View with swapped strides — not C-contiguous
print(T.flags['C_CONTIGUOUS'])  # False

# Some Numba JIT functions require contiguous input
T_cont = np.ascontiguousarray(T)  # Forces a copy with C-order strides
```

---

## GPU / Numba Gotchas

### Host-device data transfer dominates runtime for simple kernels

The PCI bus between CPU memory and GPU memory is a bottleneck. For simple kernels (vector addition), memory transfers account for ~98% of total runtime. Compute is only ~2%.

```python
# Measured for N=1,000,000 float32 vector addition:
# With host arrays (implicit transfers each call): 5.99 ms
# With pre-transferred device arrays: 0.042 ms
# Conclusion: 98.3% of time was memory transfer

# Wrong: transfer on every call in a loop
for img in images:
    result = kernel[bpg, tpb](img, output)  # Transfers img HtoD + output DtoH each time

# Right: transfer once, compute many times
d_images = cuda.to_device(images)
d_output = cuda.device_array(output_shape)
for i in range(n_images):
    kernel[bpg, tpb](d_images[i], d_output)
result = d_output.copy_to_host()  # One transfer at the end
```

### JIT warmup: time the second call, not the first

Both `@jit` (CPU) and `@cuda.jit` (GPU) compile on first call. The first call timing includes compilation, which is independent of problem size but can be 50-100x slower.

```python
from numba import cuda
from time import perf_counter

@cuda.jit
def my_kernel(x, out):
    i = cuda.grid(1)
    if i < len(x):
        out[i] = x[i] ** 2

# Warm up — discard this timing
my_kernel[bpg, tpb](x, out)
cuda.synchronize()

# Now time subsequent calls
rep = 100
t = perf_counter()
for _ in range(rep):
    my_kernel[bpg, tpb](x, out)
cuda.synchronize()   # Must synchronise before stopping the clock
print((perf_counter() - t) / rep * 1000, 'ms')
```

**Critical:** call `cuda.synchronize()` before stopping the clock. GPU kernels launch asynchronously — the CPU continues while the GPU runs. Without synchronisation, you measure only the launch time, not the execution time.

### Thread divergence kills GPU performance

GPU warps execute one instruction at a time (SIMT — Single Instruction, Multiple Threads). If threads in the same warp take different branches of an `if/else`, both paths execute serially, and the throughput is halved or worse.

```python
@cuda.jit
def bad_kernel(x, out):
    i = cuda.grid(1)
    if i < len(x):
        if x[i] > 0:       # Different threads branch differently
            out[i] = x[i] * 2   # Some threads execute this
        else:
            out[i] = -x[i]  # Others execute this — BOTH paths serialise

# Better: restructure to avoid divergent branches, or ensure uniform branching
```

### Shared memory is limited (~48 KB per block)

GPU shared memory (the per-block fast scratchpad) is typically 48 KB per SM. Requesting more causes the kernel to fail at runtime or reduces occupancy.

```python
from numba import cuda, float32

@cuda.jit
def tiled_matmul(A, B, C):
    # Allocate shared memory: 16x16 float32 = 16*16*4 = 1024 bytes per tile
    tile_A = cuda.shared.array(shape=(16, 16), dtype=float32)  # Fine: 1 KB
    tile_B = cuda.shared.array(shape=(16, 16), dtype=float32)  # Fine: 1 KB
    # Total: 2 KB — well within the 48 KB limit
```

### `@cuda.jit` kernels cannot use all Python features

Inside a `@cuda.jit` kernel you are restricted to a subset of Python:
- No Python objects, classes, or dynamic typing
- No standard library functions (no `print`, `len` outside of simple cases)
- No return values — write results to an output argument
- No calling other kernels from within a kernel
- Can call `@cuda.jit(device=True)` helper functions

```python
@cuda.jit
def my_kernel(x, out):
    i = cuda.grid(1)
    if i < x.shape[0]:     # x.shape[0] works
        out[i] = x[i] * 2
    # return x[i]           # WRONG: kernels have no return value
    # print(x[i])           # Limited support — avoid

@cuda.jit(device=True)
def helper(a, b):           # Helper function callable from within a kernel
    return a + b
```

### Always bounds-check in GPU kernels

The grid may have more threads than output elements (because block count is rounded up). Without a bounds check, out-of-bounds threads access invalid memory.

```python
@cuda.jit
def kernel(x, out):
    i = cuda.grid(1)
    if i < len(x):     # Required bounds check
        out[i] = x[i] * 2

# Grid size rounds up: bpg = (n + tpb - 1) // tpb
# If n=1001 and tpb=256: bpg=4, total threads=1024
# Threads 1001-1023 must not access x[1001]..x[1023] — out of bounds
```

---

## Debugging HPC Code

When something is slow or wrong, work through this sequence before touching the code.

### Step 1: Profile before optimising

Find out where the time actually goes. Do not guess.

```bash
# cProfile: function-level profiling
python -m cProfile -s cumulative script.py

# kernprof (line-level profiling): mark functions with @profile
kernprof -l -v script.py
```

Read the output:
- `cumtime`: total time in a function including all sub-calls — use this to find the bottleneck
- `tottime`: time in a function excluding sub-calls
- `percall`: cost per call; multiply by expected call count to project to production scale
- `Hits` in line profiler: how many times a line ran (use to infer loop count and input size)

### Step 2: Scale the profiler result to the actual workload

Profiling on a small subset means nothing if you do not project the cost to the full workload.

```python
# If process_sample is called 10 times in a subset and takes 0.505s each,
# then at 1000 samples: 1000 × 0.505 = 505s — this is the bottleneck.
# prepare_model (15s, called once) is not the bottleneck despite being large.
```

### Step 3: Check correctness after parallelising

Parallelisation introduces race conditions and numerical reordering. Always verify the output matches the serial version before claiming a speedup.

```python
serial_result = serial_func(data)
parallel_result = parallel_func(data)

# Use tolerance for floating-point results (order of operations changes)
np.testing.assert_allclose(serial_result, parallel_result, rtol=1e-5)
```

### Step 4: Start with small data, scale up gradually

Debug correctness on a 10-element array before running on 10 million elements. A bug that corrupts one element is hard to spot in a 2D plot but obvious in a 4-element printed array.

### Step 5: Diagnose thread count problems

If your job runs slower than expected, check whether packages are spawning unexpected threads:

```bash
# In LSF job output file, check:
# Max Threads: 256  <- this should roughly equal #BSUB -n if threads are controlled

# Or during a running job, SSH to the execution host (shown in bjobs) and run:
top   # look at the THREADS column for your python process
```

### Step 6: Diagnose I/O slowness

```bash
# Check which filesystem your output goes to
df -h /path/to/output    # /work3 is fast; home directories are slow

# Check which BLAS backend NumPy uses (to know which thread variable to set)
python -c "import numpy; numpy.show_config()"
```

---

## Exam-Specific Tips (MCQ Format)

> **THIS YEAR'S EXAM IS MCQ ONLY (~24 questions, 4 options each, 4 hours).** The F25 exam is your best practice resource — same format. Study the distractor patterns below: the wrong options are engineered to catch specific misconceptions.

---

### General MCQ Strategy

**Step 1 — Eliminate obviously wrong options first.**
In almost every question, 1-2 options are clearly wrong. Cross them out mentally before reasoning about the survivors. This turns a 1-in-4 guess into a 1-in-2 if you're stuck.

**Step 2 — Watch for "almost right" distractors.**
The wrong options are not random — they're the answer you'd get if you made one specific mistake:
- Right formula, wrong direction (e.g. total memory instead of per-core)
- Right concept, wrong layer (e.g. confusing thread with block count)
- Off-by-one (ceil vs floor, +1 vs -1)
- Correct for CPU but wrong for GPU (or vice versa)

**Step 3 — For calculation questions: compute first, then look at options.**
Do the arithmetic on paper before reading the options. If you read the options first, the "almost right" numbers will anchor your thinking.

**Step 4 — Flag and skip if stuck, never leave blank.**
All questions are equal weight. A 2-minute question is worth the same as a 10-minute one. If you're stuck after 90 seconds, flag it and move on. Come back with fresh eyes. Never leave blank — no penalty for wrong answers.

**Step 5 — Re-read the question stem on tricky ones.**
Common misreads: "which is NOT correct", "what is the MINIMUM", "what will be printed" (precision/rounding questions).

---

### Topic: Amdahl's Law (appears on every exam)

**The formula and all its directions:**
```
S(p) = 1 / (1 - F + F/p)

Given S(p), solve for F:    F = p(1 - 1/S(p)) / (p - 1)
Max speedup (p → ∞):        S_max = 1 / (1 - F)
Time on 1 core from T(p):   T(1) = T(p) × S(p)
```

**MCQ distractor patterns to watch for:**
- **Trap: F = 0.8 so max speedup = 0.8** — wrong, max speedup = 1/(1-F) = 5, not F itself
- **Trap: S(8) = 8 × F** — wrong, Amdahl's formula has the serial fraction in the denominator
- **Trap: confusing S (speedup) with T (time)** — question asks for speedup, distractor gives time ratio
- **Trap: using p = ∞ when a specific p is given** — compute S(p) not S_max unless asked for max
- **Trap: F derived from one measurement applied to a different p** — re-derive F first, then compute

**MCQ process:**
1. Identify what's given: S(p) measured? F known? T(p) known?
2. Write the formula and substitute — don't do it in your head
3. Pick the option that matches your result exactly

---

### Topic: LSF/BSUB Job Scripts (appears on every exam)

**The flag you MUST know cold:**
```bash
#BSUB -n X                        # X cores requested
#BSUB -R "rusage[mem=XGB]"        # X GB PER CORE (not total)
#BSUB -R "span[hosts=1]"          # all cores on same node
#BSUB -W HH:MM                    # wall-clock limit
#BSUB -q gpuv100                  # GPU queue
#BSUB -gpu "num=1:mode=exclusive_process"  # 1 GPU
#BSUB -J name[1-N]                # job array
#BSUB -w "done(name)"             # dependency: ALL must succeed
#BSUB -w "ended(name)"            # dependency: any termination
```

**MCQ distractor patterns:**
- **Trap: rusage[mem=100GB] for 100 GB total with 4 cores** — wrong, correct is 25GB (total ÷ cores)
  - The 4 wrong options will be: the total (100GB), half (50GB), quarter (25GB ✓), tenth (10GB)
  - Always ask: "is this per core or total?"
- **Trap: span[hosts=1] is optional** — it is REQUIRED for shared-memory multiprocessing
- **Trap: done() vs ended()** — done() blocks forever if any job fails, ended() continues regardless
- **Trap: job array index starts at 0** — it starts at 1 (`$LSB_JOBINDEX` = 1 for first element)

---

### Topic: Reduction Operators (commutativity & associativity)

**The rule:** a parallel reduction requires the operator to be both **associative** and **commutative**.

**MCQ process for "can X be used in a parallel reduction?":**
1. Test associativity: does `(a ⊕ b) ⊕ c == a ⊕ (b ⊕ c)` for all values?
2. Test commutativity: does `a ⊕ b == b ⊕ a`?
3. If both yes → answer is YES. If either fails → answer is NO.

**Test with concrete numbers, not abstract reasoning:**
```
# Is abssum(x,y) = abs(x+y) associative?
abssum(abssum(1, 2), -3) = abssum(3, -3) = abs(0) = 0
abssum(1, abssum(2, -3)) = abssum(1, -1) = abs(0) = 0  ← same, try another
abssum(abssum(3, -1), -3) = abssum(2, -3) = 1
abssum(3, abssum(-1, -3)) = abssum(3, -4) = 1  ← same again... try more
abssum(abssum(1, 2), -3) = 0
abssum(1, abssum(2, -3)) = abs(1 + abs(2-3)) = abs(1+1) = 2  ← 0 ≠ 2, NOT associative
```

**MCQ distractor patterns:**
- **Trap: set intersection is NOT commutative** — it IS commutative (A∩B = B∩A) AND associative → answer YES
- **Trap: matrix multiply IS commutative** — it is NOT → answer NO
- **Trap: "associative but not commutative" option** — subtraction is this; many operators are both or neither
- Distractors will usually offer: Yes / No-not-associative / No-not-commutative / No-neither

---

### Topic: NumPy dtype, reshape, float precision

**MCQ process for reshape questions:**
1. Array is row-major (C order) — elements laid out row by row in memory
2. `reshape(-1)` or `reshape(-1, n)` flattens then cuts — follows memory order
3. Element at index `k` in flattened array = `arr[k // ncols][k % ncols]`

```
# a = [[1,5,43,51,32],[73,2,4,67,37],[9,3,54,8,22]]  (3×5)
# a.reshape(-1) = [1,5,43,51,32,73,2,4,67,37,9,3,54,8,22]
# a.reshape(-1)[8] = element at index 8 = 67
```

**MCQ process for float16 precision questions:**
- float16: resolution ~0.001, max ~65504
- If the true result requires more significant digits than the dtype allows, it rounds
- `10000 + 1` in float16: 10000 needs 5 significant digits, float16 only has ~3 → result stays 10000

**MCQ distractor patterns for dtype:**
- **Trap: int8 can hold 200** — range is -128 to 127, 200 overflows → wraps to -56
- **Trap: float16 result is exact** — check if result needs more precision than the dtype provides
- **Trap: adding 1 to 10000 in float16 gives 10001** — no, it gives 10000 (precision lost)

---

### Topic: Profiler output (cProfile / line_profiler / nsys)

**cProfile columns:**
```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
```
- `tottime` = time IN this function, excluding callees
- `cumtime` = total time INCLUDING all called functions
- **"what takes most time overall?" → look at cumtime**
- **"what is the hotspot inside this function?" → look at tottime**

**MCQ process:**
1. Identify what the question asks: overall time? time excluding callees? bottleneck to optimise?
2. Map to the right column
3. The distractor will be the function with the highest value in the WRONG column

**line_profiler scaling:**
```
# Measured: 10 items, 0.5s per item
# Question: time for 1000 items?
# Answer: 1000 × 0.5 = 500s
# Trap option: 10 × 0.5 = 5s (uses measured count, not target count)
```

**nsys GPU profiler:**
- `gpukernsum` → kernel execution time (compute)
- `gpumemtimesum` → memory transfer time (HtoD + DtoH)
- **"what dominates?" → compare total time in each table**
- **"is this memory-bound or compute-bound?" → memory table >> kernel table → memory-bound**

---

### Topic: CUDA threads, blocks, grid

**MCQ process for "how many blocks needed?"**
```
# Array of N elements, tpb threads per block
blocks = ceil(N / tpb) = (N + tpb - 1) // tpb
```

**MCQ process for "what is thread i's global index?"**
```
i = blockIdx.x * blockDim.x + threadIdx.x
  = (block number) × (threads per block) + (thread position in block)
```

**For 2D:**
```
i, j = cuda.grid(2)
# i = row index in full matrix
# j = col index in full matrix
# Block (bx, by), thread (tx, ty):
# i = bx * tpb_rows + tx
# j = by * tpb_cols + ty
```

**MCQ distractor patterns:**
- **Trap: blocks = N / tpb (floor)** — correct is ceil; floor misses the last partial block
- **Trap: total threads = blocks** — total threads = blocks × tpb
- **Trap: tpb can be any number** — max is 1024; must be multiple of 32 (warp size) for efficiency
- **Trap: bounds check not needed** — it IS needed when N is not divisible by tpb

**Memory transfer counting (Numba automatic transfers):**
```
# kernel(a, b) called N times with plain NumPy arrays
# Each call: a → HtoD, b → HtoD, a → DtoH, b → DtoH = 4 transfers
# Total: 4N transfers

# Optimal (manual device arrays):
# a: 1 HtoD (input, once); b: 1 DtoH (output, once) = 2 transfers total
```

---

### Topic: Cache & memory layout

**CPU rule — sequential access → last index varies fastest:**
```python
# Good (row-major, last index varies in inner loop):
for i in range(rows):
    for j in range(cols):   # j varies fastest → accesses arr[i,0], arr[i,1], ... contiguous
        use(arr[i, j])

# Bad (column access → cache miss every iteration):
for j in range(cols):
    for i in range(rows):
        use(arr[i, j])
```

**GPU rule — Warp Coalescing (Row-Major):**
- Threads in a warp always have consecutive `threadIdx.x`
- To coalesce, the array's **last axis (columns)** must be the one varying across adjacent threads
- **Same conclusion as CPU:** both want last-axis-varying — CPU achieves it through *time* (innermost for-loop), GPU achieves it through *space* (adjacent threads stretch across the last axis)

**The critical trap — standard Numba boilerplate breaks this:**
```python
i, j = cuda.grid(2)   # ← threadIdx.x maps to i (ROW), not j (column) → ruins coalescing
```
`cuda.grid(2)` returns `(x-dim, y-dim)`. The x-dimension (`threadIdx.x`) varies fastest within a warp. Unpacking as `i, j` silently assigns the fast-varying x to `i` (row index), so adjacent threads step through rows — strided access, cache-unfriendly.

**The fix — explicitly map threadIdx.x to the column:**
```python
# Option 1: swap the unpack
j, i = cuda.grid(2)   # threadIdx.x → j (column) → adjacent threads hit adjacent memory ✓

# Option 2: flat block shape — blockDim.x=1 so threadIdx.x=0 for all threads,
# warp varies through threadIdx.y (0..31) → j = blockIdx.y*32 + threadIdx.y varies → coalesced ✓
tpb = (1, 32)
```

**MCQ distractor pattern:**
- **Trap: "column access is faster because columns are contiguous in Fortran order"** — arrays in this course are C order (row-major) unless stated otherwise
- **Trap: GPU and CPU want coalescing for the same reason** — they don't; CPU uses cache lines (temporal/spatial locality), GPU uses warp-level simultaneous access (coalescing). Same physical outcome (last-axis-varying), different mechanisms
- **Trap: assuming `i, j = cuda.grid(2)` gives coalesced access** — it does the opposite; you must swap to `j, i = cuda.grid(2)` or redesign the block shape

---

### Topic: Broadcasting

**Right-align rule:**
```
images:      (N, H, W, 3)
mean_pixels: (N, 3)

Right-align:
  images:      (N, H, W, 3)
  mean_pixels: (      N, 3)   ← 2 dims short, pad left with 1s
  →            (1, 1, N, 3)   ← this is what NumPy treats it as

That would broadcast WRONG. You need (N, 1, 1, 3):
  images - mean_pixels[:, None, None]   ← correct
```

**MCQ process:**
1. Write out both shapes, right-aligned
2. Check each dimension: must be equal OR one must be 1
3. If a dim is neither equal nor 1 → error
4. The output shape = max of each dimension

**MCQ distractor patterns:**
- **Trap: shapes that look compatible but aren't** — (N, 3) subtracted from (N, H, W, 3) fails at dim -2: H vs 3
- **Trap: `mean_pixels[None, None]` vs `mean_pixels[:, None, None]`** — first gives (1,1,N,3), second gives (N,1,1,3)

---

### Topic: GIL, multiprocessing, threading

**The one rule:** Python threads share the GIL → only one thread runs Python at a time → no CPU speedup from threading for Python code.

**MCQ answers:**
- "Use multiprocessing, not threading" → correct for CPU-bound Python code
- "Threading works fine for I/O-bound tasks" → correct (GIL released during I/O)
- "Numba `@jit(nogil=True)` releases the GIL" → correct, enables true thread parallelism for JIT'd code

**MCQ distractor patterns:**
- **Trap: threading gives 4x speedup on 4 cores** — NO, GIL prevents this
- **Trap: multiprocessing.Pool and threading.Pool are interchangeable** — Pool is processes; ThreadPool exists but hits GIL

---

### Time Management

| Topic | Typical difficulty | Strategy |
|---|---|---|
| BSUB memory flag | Easy | 30 seconds: total ÷ cores |
| Amdahl speedup | Easy-medium | 1 min: write formula, substitute |
| cProfile bottleneck | Easy | 30 seconds: find max cumtime |
| Reduction validity | Medium | 1-2 min: test with 3 numbers |
| Broadcasting | Medium | 1-2 min: right-align and check |
| reshape indexing | Medium | 1 min: flatten mentally, count |
| float16 precision | Medium | 1 min: count significant digits |
| CUDA block count | Medium | 1 min: ceil(N/tpb) |
| nsys transfer count | Hard | 2-3 min: count args × calls × directions |
| Cache coalescing | Hard | 2-3 min: think through access pattern |

**Order to attempt:** easy guaranteed marks first (BSUB, Amdahl, cProfile), then medium, flag and return to hard ones.

---

### Formula Reference Card

```
Speedup:             S(p) = T(1) / T(p)
Amdahl:              S(p) = 1 / (1 - F + F/p)
Amdahl max:          S_max = 1 / (1 - F)
Solve for F:         F = p(1 - 1/S(p)) / (p - 1)
Efficiency:          E(p) = S(p) / p

CUDA blocks needed:  bpg = ceil(N / tpb) = (N + tpb - 1) // tpb
CUDA global index:   i = blockIdx.x * blockDim.x + threadIdx.x

Memory per dtype:
  float16 = 2 bytes   float32 = 4 bytes   float64 = 8 bytes
  int8  = 1 byte (-128..127)    uint8  = 1 byte (0..255)
  int16 = 2 bytes               int32  = 4 bytes
  int64 = 8 bytes

Chunk size:          max_rows = available_RAM_bytes / bytes_per_row
Transfer bandwidth:  bytes / time
FLOP/s:              total_FLOPs / elapsed_seconds
matmul FLOPs:        2×N³ for N×N (multiply + add per element pair)

rusage memory:       value = total_GB / n_cores
```

---

## Complete Good Job Script Template

```bash
#!/bin/bash
#BSUB -J jobname
#BSUB -q hpc
#BSUB -W 00:30
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "rusage[mem=4GB]"   # 4 GB per core = 32 GB total
#BSUB -o /work3/02613/dump/jobname_%J.out
#BSUB -e /work3/02613/dump/jobname_%J.err

# Activate course environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Control package thread count to match allocation
NUM_THREADS=8
OMP_NUM_THREADS=$NUM_THREADS
MPI_NUM_THREADS=$NUM_THREADS
MKL_NUM_THREADS=$NUM_THREADS
OPENBLAS_NUM_THREADS=$NUM_THREADS

# Redirect program output to /work3 to bypass the slow -o channel
python -u script.py \
    1> /work3/02613/dump/output_${LSB_JOBID}.txt \
    2> /work3/02613/dump/error_${LSB_JOBID}.txt
```
