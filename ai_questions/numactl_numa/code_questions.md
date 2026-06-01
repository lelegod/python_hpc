# numactl & NUMA Topology — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — What Command Runs Python Under Interleaved Memory](#q1--what-command-runs-python-under-interleaved-memory)
- [Q2 — Identifying the NUMA Plateau from Timing Data](#q2--identifying-the-numa-plateau-from-timing-data)
- [Q3 — Reading numactl --hardware for Node Count](#q3--reading-numactl---hardware-for-node-count)
- [Q4 — Which Flag Restricts Cores to Socket 0](#q4--which-flag-restricts-cores-to-socket-0)
- [Q5 — Where RawArray Pages Land Without numactl](#q5--where-rawarray-pages-land-without-numactl)
- [Q6 — Does span[hosts=1] Apply NUMA Policy?](#q6--does-spanhosts1-apply-numa-policy)
- [Q7 — Predict the Speedup Curve Shape from Code](#q7--predict-the-speedup-curve-shape-from-code)
- [Q8 — What Does init() Do in the Pool Pattern?](#q8--what-does-init-do-in-the-pool-pattern)
- [Q9 — Bug: Wrong numactl Flag for Full Local Binding](#q9--bug-wrong-numactl-flag-for-full-local-binding)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2--generated-practice-questions-exam-day-focus)
- [Q10 — Interleave vs Membind: Choosing for 48-Core Job](#q10--interleave-vs-membind-choosing-for-48-core-job)
- [Q11 — Does numactl --interleave=all Affect CPU Pinning?](#q11--does-numactl---interleaveall-affect-cpu-pinning)
- [Q12 — Reading NUMA Distance from Hardware Output](#q12--reading-numa-distance-from-hardware-output)
- [Q13 — Spotting the NUMA Bottleneck in a Speedup Table](#q13--spotting-the-numa-bottleneck-in-a-speedup-table)
- [Q14 — Effect of numactl on a Single-Process NumPy Sum](#q14--effect-of-numactl-on-a-single-process-numpy-sum)
- [Q15 — Pool.map Reduction: How Many Rounds?](#q15--poolmap-reduction-how-many-rounds)
- [Q16 — Which LSF Flag Keeps Job on One Server?](#q16--which-lsf-flag-keeps-job-on-one-server)
- [Q17 — Memory Policy for Dataset Larger Than One Node](#q17--memory-policy-for-dataset-larger-than-one-node)
- [Q18 — Full numactl + Pool Pattern: What Is Printed?](#q18--full-numactl--pool-pattern-what-is-printed)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q19 — What Does numactl --show Print?](#q19--what-does-numactl---show-print)
- [Q20 — First-Touch in a Worker Process](#q20--first-touch-in-a-worker-process)
- [Q21 — What Does os.sched_getaffinity Print Under --cpunodebind=0?](#q21--what-does-ossched_getaffinity-print-under---cpunodebind0)
- [Q22 — Does --membind=0 Prevent Allocation on Node 1?](#q22--does---membind0-prevent-allocation-on-node-1)
- [Q23 — Predict the Array State After One Reduction Round](#q23--predict-the-array-state-after-one-reduction-round)
- [Q24 — What Is the NUMA Policy After fork()?](#q24--what-is-the-numa-policy-after-fork)
- [Q25 — How Many Active Tasks in Round j of a Binary Reduction?](#q25--how-many-active-tasks-in-round-j-of-a-binary-reduction)
- [Q26 — numactl --interleave=all With --cpunodebind=1: What Happens?](#q26--numactl---interleaveall-with---cpunodebind1-what-happens)
- [Q27 — Identifying the First-Touch Core from Code](#q27--identifying-the-first-touch-core-from-code)
- [Q28 — Compute-Bound vs Memory-Bound: Does numactl Help?](#q28--compute-bound-vs-memory-bound-does-numactl-help)

---

> Format: Each question shows numactl commands, LSF scripts, or Python multiprocessing code with NUMA effects to analyse.
> Exam frequency: **Week 6 topic**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#question-1)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — What Command Runs Python Under Interleaved Memory

```bash
# Option A
python reduction.py /dtu/projects/02613_2025/data/celeba/celeba_100K.npy

# Option B
numactl --interleave=all python reduction.py /dtu/projects/02613_2025/data/celeba/celeba_100K.npy

# Option C
numactl --membind=0 python reduction.py /dtu/projects/02613_2025/data/celeba/celeba_100K.npy

# Option D
numactl --cpunodebind=1 python reduction.py /dtu/projects/02613_2025/data/celeba/celeba_100K.npy
```

A student wants to run the CelebA parallel reduction with memory spread evenly across all NUMA nodes to improve scaling across both CPU sockets. Which command achieves this?

**A)** Option A — no numactl, default first-touch allocation.
**B)** Option B — `--interleave=all` distributes pages round-robin across all NUMA nodes.
**C)** Option C — `--membind=0` concentrates all memory on NUMA node 0.
**D)** Option D — `--cpunodebind=1` distributes memory across both nodes.

**Answer: B**

- A) Without numactl, Linux uses first-touch policy. The main process first-touches all pages when loading the array, placing them on NUMA node 0. Socket 1 workers see slow remote access — the problem the student wants to solve.
- B) Correct — `--interleave=all` distributes successive pages round-robin across all available NUMA nodes. Roughly half the array lands on each socket's DRAM, so all workers get approximately equal average access latency.
- C) Incorrect — `--membind=0` does the opposite: it forces all allocations onto NUMA node 0. This makes the problem worse for socket 1 workers, not better.
- D) Incorrect — `--cpunodebind=1` restricts CPU execution to socket 1 only; it does not spread memory across both nodes. It controls core placement, not memory placement, and placing all cores on socket 1 while data stays on node 0 (default) maximises remote accesses.

---

## Q2 — Identifying the NUMA Plateau from Timing Data

A student benchmarks the CelebA reduction and records these wall-clock times (in seconds) for varying process counts, without numactl:

```
Processes | Time (s) | Speedup
----------|----------|--------
1         | 120.0    | 1.00
4         | 34.0     | 3.53
8         | 18.5     | 6.49
12        | 13.8     | 8.70
16        | 11.4     | 10.53
20        | 11.1     | 10.81
24        | 10.9     | 11.01
28        | 12.2     | 9.84
32        | 14.7     | 8.16
```

Which statement correctly identifies and explains the NUMA plateau?

**A)** The speedup peaks at p = 8 (S ≈ 6.49) because Amdahl's Law limits the maximum speedup.
**B)** The speedup peaks around p = 24 then decreases; this is the NUMA plateau caused by processes on socket 1 making remote memory accesses to data located on NUMA node 0.
**C)** The speedup peaks at p = 16 then decreases because Python's GIL limits parallelism beyond 16 threads.
**D)** The speedup consistently decreases after p = 4, showing a normal Amdahl's Law saturation.

**Answer: B**

- A) Amdahl's Law would produce a monotonically increasing curve that levels off asymptotically — it would not produce a peak followed by a decline. The peak-and-decline pattern is characteristic of NUMA, not Amdahl's serial fraction alone.
- B) Correct — Speedup peaks near p = 24 (approximately one socket's worth of cores) at S ≈ 11, then declines to S ≈ 8.16 at p = 32. This inverted-U shape is the NUMA plateau: the first 24 cores access data locally, but cores 25–32 on socket 1 pay a remote-access penalty that erases their contribution.
- C) The exercise uses `multiprocessing.Pool`, which spawns separate processes, not threads. The GIL is irrelevant for multiprocessing. The plateau at 24 processes (not 16) also refutes this.
- D) A consistent decrease after p = 4 would show the speedup data going 3.53, 3.X, 3.X, ... — but the table shows S increasing from p = 4 to p = 24. This is not a normal Amdahl saturation shape; the data shows increase then decrease.

---

## Q3 — Reading numactl --hardware for Node Count

```bash
$ numactl --hardware
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
node 0 size: 96004 MB
node 0 free: 89321 MB
node 1 cpus: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
node 1 size: 96763 MB
node 1 free: 90102 MB
node distances:
node   0   1
  0:  10  21
  1:  21  10
```

How many physical CPU sockets does this machine have, and what is the relative latency ratio for a core on node 0 accessing memory on node 1?

**A)** 1 socket; latency ratio = 1.0 (uniform access).
**B)** 2 sockets; latency ratio ≈ 2.1 (node-1 access is about 2.1x slower than local).
**C)** 2 sockets; latency ratio = 21.0 (node-1 access is 21x slower than local).
**D)** 4 sockets; latency ratio ≈ 0.48 (node-1 access is faster than local).

**Answer: B**

- A) One socket would show only one NUMA node in `numactl --hardware`. The output clearly shows 2 nodes. A latency ratio of 1.0 would mean uniform access (UMA), contradicted by the asymmetric distance matrix.
- B) Correct — 2 nodes = 2 physical sockets. The NUMA distance matrix shows local (self) distance = 10 and remote distance = 21. The ratio is 21/10 = 2.1, meaning a core on node 0 experiences approximately 2.1x the latency when accessing memory on node 1 compared to accessing its own local DRAM.
- C) The distance values are relative scores, not absolute multipliers. The ratio is 21/10 = 2.1, not 21. A 21x latency penalty would be extreme and is not representative of typical server hardware.
- D) Four sockets would require at least 4 nodes in the output. The output shows exactly 2 nodes. A ratio less than 1 would mean remote memory is faster than local, which contradicts the NUMA model entirely.

---

## Q4 — Which Flag Restricts Cores to Socket 0

```bash
# Which of these commands restricts the process to run only on
# cores 0-15 (socket 0) on a 32-core dual-socket machine?

# Command A
numactl --physcpubind=0 python script.py

# Command B
numactl --cpunodebind=0 python script.py

# Command C
numactl --membind=0 python script.py

# Command D
numactl --interleave=0 python script.py
```

Which command correctly restricts execution to all cores of NUMA node 0 (socket 0) on a 32-core dual-socket machine where cores 0–15 belong to node 0?

**A)** Command A — `--physcpubind=0` pins all socket-0 cores.
**B)** Command B — `--cpunodebind=0` pins the process to all cores belonging to NUMA node 0.
**C)** Command C — `--membind=0` restricts memory and automatically restricts cores.
**D)** Command D — `--interleave=0` interleaves across node 0 cores only.

**Answer: B**

- A) Incorrect — `--physcpubind=0` pins to a single physical CPU ID (core 0 only). To pin to all 16 cores of socket 0 using `--physcpubind`, you would need to list all core IDs: `--physcpubind=0,1,2,...,15`. This command uses only core 0.
- B) Correct — `--cpunodebind=0` uses the NUMA node abstraction: it sets the CPU affinity to the full set of cores belonging to NUMA node 0. On this machine, node 0 owns cores 0–15, so the process can use all 16 without listing them individually.
- C) Incorrect — `--membind` is a memory placement policy. It does not affect CPU affinity at all. The process can still run on any core of either socket with only `--membind=0`.
- D) Incorrect — `--interleave` takes a node list for memory interleaving, not a CPU filter. `--interleave=0` is a memory policy (interleave across just node 0, which is meaningless since there is nothing to interleave with). It has no effect on CPU placement.

---

## Q5 — Where RawArray Pages Land Without numactl

```python
import ctypes
import multiprocessing as mp
import numpy as np

def init(shared_arr_):
    global shared_arr
    shared_arr = shared_arr_

if __name__ == '__main__':
    # Main process loads data and creates shared array
    data = np.load('/dtu/projects/02613_2025/data/celeba/celeba_100K.npy')
    shared_arr = mp.RawArray(ctypes.c_float, data.size)
    arr = np.frombuffer(shared_arr, dtype='float32').reshape(data.shape)
    np.copyto(arr, data)  # <-- first-touch happens here on the main process
    del data

    pool = mp.Pool(32, initializer=init, initargs=(shared_arr,))
    # ... reduction work ...
```

On a dual-socket NUMA machine with no numactl, on which NUMA node do the shared array's memory pages reside after `np.copyto(arr, data)`, and why?

**A)** Evenly split — `mp.RawArray` automatically interleaves pages across both nodes.
**B)** NUMA node 0 — the main process first-touches all pages during `np.copyto`, and the main process runs on socket 0 by default.
**C)** NUMA node 1 — `mp.RawArray` always allocates on the secondary NUMA node to free up primary memory.
**D)** The OS randomly assigns pages to nodes to balance memory usage.

**Answer: B**

- A) `mp.RawArray` does not automatically interleave. It allocates contiguous virtual memory, and physical pages are placed according to the current NUMA policy (default: first-touch). Without numactl, there is no interleaving.
- B) Correct — Linux's first-touch NUMA policy places each physical page on the NUMA node of the core that first writes to it. `np.copyto(arr, data)` is the first write to the shared array's pages, executed in the main process. The main process runs on socket 0 by default, so all pages land on NUMA node 0.
- C) Incorrect — There is no policy that preferentially allocates shared memory to the secondary node. `mp.RawArray` uses standard `mmap`-based shared memory, which follows the process's current NUMA memory policy (first-touch by default).
- D) Incorrect — Linux memory allocation is deterministic, not random. The first-touch policy is predictable: whoever writes first wins the placement.

---

## Q6 — Does span[hosts=1] Apply NUMA Policy?

```bash
#!/bin/bash
#BSUB -J celeba_reduction
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -W 1:00
#BSUB -q hpc
#BSUB -o output_%J.txt

python reduction.py /dtu/projects/02613_2025/data/celeba/celeba_100K.npy
```

A student submits this LSF job. Does `span[hosts=1]` apply a numactl interleave policy, and will the job experience NUMA effects?

**A)** Yes — `span[hosts=1]` automatically applies `--interleave=all` when all cores are on one host.
**B)** No — `span[hosts=1]` only ensures all 32 cores come from the same physical server; NUMA effects between sockets on that server still apply and no memory policy is set.
**C)** Yes — `span[hosts=1]` binds the job to NUMA node 0, preventing any cross-socket access.
**D)** No — `span[hosts=1]` is irrelevant; LSF always splits jobs across multiple hosts regardless.

**Answer: B**

- A) Incorrect — `span[hosts=1]` is a resource placement directive in LSF; it has no integration with numactl and applies no memory policy. NUMA policy must be explicitly set by calling `numactl` in the job script.
- B) Correct — `span[hosts=1]` ensures all 32 requested cores come from one compute node (so shared memory works). Within that single node, there are two NUMA sockets. The Python main process will still first-touch all data on NUMA node 0, and workers on socket 1 will still experience remote access. NUMA effects are fully present; numactl must be called explicitly.
- C) Incorrect — `span[hosts=1]` does not bind to any NUMA node. It is a host-count constraint (one physical server), not a socket constraint (one socket). Both sockets of the server are available to the job.
- D) Incorrect — `span[hosts=1]` is a well-supported LSF directive that does constrain job placement to a single host. Without it, LSF may spread the 32 cores across multiple servers, which would break shared memory.

---

## Q7 — Predict the Speedup Curve Shape from Code

```python
import multiprocessing as mp
import ctypes
import numpy as np
from time import perf_counter

# No numactl — run as: python benchmark.py

if __name__ == '__main__':
    data = np.random.rand(200000, 128, 128, 3).astype('float32')
    shared_arr = mp.RawArray(ctypes.c_float, data.size)
    arr = np.frombuffer(shared_arr, dtype='float32').reshape(data.shape)
    np.copyto(arr, data)
    del data

    results = {}
    for n_proc in [1, 4, 8, 16, 24, 32]:
        # ... parallel reduction with n_proc workers ...
        results[n_proc] = measured_time

    for p, t in results.items():
        print(f"p={p}: S = {results[1] / t:.2f}")
```

On a 32-core dual-socket NUMA node (16 cores per socket) without numactl, which speedup curve shape is expected?

**A)** Monotonically increasing from S=1 to approximately S=32 (near-linear scaling).
**B)** Rises to a peak near p=16, then declines for p=24 and p=32 due to NUMA remote-access penalty.
**C)** Flat at S=1 for all process counts because Python multiprocessing cannot use shared memory.
**D)** Decreases from S=1 at p=1 to S<1 at p=32 because each extra process adds overhead without benefit.

**Answer: B**

- A) Near-linear scaling requires either a UMA machine or numactl interleaving. Without numactl on this NUMA machine, processes on socket 1 pay a remote-access penalty and scaling breaks at the socket boundary.
- B) Correct — Up to p=16, all workers fit on socket 0 and access data locally, giving good scaling. At p=24 and p=32, workers on socket 1 access all data remotely. The remote latency overhead is large enough that adding those workers produces net negative speedup. The characteristic shape is: rise to peak at ~socket size, then decline.
- C) Python multiprocessing can absolutely use shared memory via `mp.RawArray`. The `mp.RawArray` plus initialiser pattern is specifically designed for this purpose and is the pattern used in the exercises. There is no fundamental barrier to shared-memory parallel speedup.
- D) A decrease from S=1 at all core counts is not expected. Even with NUMA effects, the first socket's worth of cores (p=1 to p=16) experience only local accesses and deliver genuine speedup. Overhead only dominates after the socket boundary is crossed.

---

## Q8 — What Does init() Do in the Pool Pattern?

```python
def init(shared_arr_):
    global shared_arr
    shared_arr = shared_arr_

pool = mp.Pool(n_processes, initializer=init, initargs=(shared_arr,))
```

What is the purpose of the `init` function and the `initializer` argument in this multiprocessing pool pattern?

**A)** It copies the entire `shared_arr` into each worker's private memory space to avoid contention.
**B)** It installs `shared_arr` as a global variable in each worker process so that worker functions can access the shared memory segment without receiving it as a pickle-serialised argument on every task.
**C)** It initialises the NumPy random seed for each worker to ensure reproducibility.
**D)** It pins each worker process to a specific CPU core to avoid NUMA effects.

**Answer: B**

- A) Incorrect — The shared array is passed by reference to the shared memory segment, not copied. `mp.RawArray` is designed for zero-copy sharing. If each worker got a private copy, the reduction would not work (writes in one worker would not be visible to others) and memory usage would explode.
- B) Correct — `initializer=init` runs `init(shared_arr_)` once in each worker process after forking. This sets the `shared_arr` global, making the shared memory segment accessible in `reduce_step` via the global name without passing it as a serialised argument (which would be a huge copy) in every `pool.map` call.
- C) Incorrect — The `init` function only assigns the global variable; it performs no random seed initialisation. Random seeds are set with `np.random.seed()`, which is not present here.
- D) Incorrect — The `init` function is pure Python and has no OS-level thread/process affinity control. CPU pinning requires `numactl` at the command line or `os.sched_setaffinity()` inside the worker — neither of which appears here.

---

## Q9 — Bug: Wrong numactl Flag for Full Local Binding

```bash
# Student wants: all computation on socket 0, all memory on socket 0
# (maximum local bandwidth, no remote access)

# Attempt A
numactl --interleave=all python reduction.py data.npy

# Attempt B
numactl --membind=0 python reduction.py data.npy

# Attempt C
numactl --cpunodebind=0 --membind=0 python reduction.py data.npy

# Attempt D
numactl --cpunodebind=0 python reduction.py data.npy
```

The student wants both all computation and all memory allocation confined to NUMA node 0 only. Which attempt correctly achieves this?

**A)** Attempt A — interleaving across all nodes ensures node 0 gets at least half the data.
**B)** Attempt B — `--membind=0` restricts memory to node 0, and cores default to node 0 also.
**C)** Attempt C — `--cpunodebind=0 --membind=0` restricts both cores and memory to NUMA node 0.
**D)** Attempt D — `--cpunodebind=0` restricts cores to node 0 and implicitly restricts memory too.

**Answer: C**

- A) Incorrect — `--interleave=all` explicitly distributes memory across all NUMA nodes. Half the data would be on node 1, meaning ~50% of accesses are remote. This is the opposite of "all memory on NUMA node 0."
- B) Incorrect — `--membind=0` ensures memory is allocated on node 0, but does not restrict which cores run the processes. Workers may be scheduled on socket 1 cores, which would then access their computation data locally on node 1 but the shared array remotely on node 0. Not a complete solution.
- C) Correct — `--cpunodebind=0` sets the CPU affinity to only socket 0's cores; `--membind=0` ensures all allocations use node 0's DRAM. Together, they guarantee that both computation and data are local to NUMA node 0, with zero inter-socket traffic.
- D) Incorrect — `--cpunodebind=0` restricts cores but does not restrict memory. Memory follows the default first-touch policy (which in this case likely puts it on node 0 since the main process runs there), but this is not explicit. A future change (e.g., the process starts on a different core) could break the assumption. The student needs `--membind=0` for an explicit guarantee.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

---

## Q10 — Interleave vs Membind: Choosing for 48-Core Job

```bash
# A student has a 48-core dual-socket node (24 cores per socket).
# They want to run the reduction with all 48 cores.
# The dataset is 50 GB, split between both sockets' local DRAM (25 GB each).

# Option A
numactl --interleave=all python reduction.py data.npy

# Option B
numactl --cpunodebind=0 --membind=0 python reduction.py data.npy

# Option C
python reduction.py data.npy   # no numactl

# Option D
numactl --membind=1 python reduction.py data.npy
```

The student wants to use all 48 cores efficiently with a 50 GB dataset that must use memory from both NUMA nodes. Which option best achieves this?

**A)** Option A — `--interleave=all` spreads the data across both DRAM banks, equalising access latency for all 48 cores.
**B)** Option B — confining to node 0 and its memory is best because it avoids all remote access.
**C)** Option C — no numactl is optimal because the OS handles NUMA placement automatically.
**D)** Option D — placing all memory on node 1 is better because node 1 has more free memory.

**Answer: A**

- A) Correct — With a 50 GB dataset that exceeds one socket's DRAM capacity, memory must span both nodes. `--interleave=all` distributes pages round-robin, so each socket holds approximately 25 GB locally. All 48 cores see ~50% local and ~50% remote accesses on average, giving equal performance regardless of which socket a worker runs on. This is the optimal choice for a dataset requiring both NUMA nodes.
- B) Incorrect — `--cpunodebind=0 --membind=0` confines everything to socket 0 (24 cores, 25 GB DRAM). The 50 GB dataset cannot fit entirely on node 0's 25 GB of DRAM; the allocation will fail or spill unpredictably. This option is not viable for a 50 GB dataset on a node with only 25 GB per socket.
- C) Incorrect — Without numactl, first-touch allocation would attempt to place the 50 GB array entirely on node 0 during the initial `np.copyto`. If node 0 has only 25 GB, allocations would overflow to node 1 unpredictably, and the 24 socket-1 cores would still see remote access for their portion. Explicit interleaving is superior.
- D) Incorrect — Placing all memory on node 1 while cores run on both sockets means 24 socket-0 cores access all data remotely. This is equivalent to the original NUMA problem in reverse, not an improvement.

---

## Q11 — Does numactl --interleave=all Affect CPU Pinning?

```bash
numactl --interleave=all python -c "
import os
print('CPU affinity:', os.sched_getaffinity(0))
"
```

On a 32-core dual-socket machine (cores 0–15 on socket 0, cores 16–31 on socket 1), what does this code print?

**A)** `CPU affinity: {0}` — interleaving restricts to core 0.
**B)** `CPU affinity: {0, 1, 2, ..., 15}` — interleaving restricts to socket 0 cores only.
**C)** `CPU affinity: {0, 1, 2, ..., 31}` — all 32 cores are available; interleaving does not restrict CPU affinity.
**D)** `CPU affinity: {16, 17, ..., 31}` — interleaving restricts to socket 1 cores only.

**Answer: C**

- A) Incorrect — `--interleave` is a memory policy flag; it does not set CPU affinity at all. The process can run on any core unless a `--cpunodebind` or `--physcpubind` flag is also used.
- B) Incorrect — Only `--cpunodebind=0` would restrict the affinity to socket 0's cores. `--interleave=all` leaves CPU affinity unrestricted.
- C) Correct — `numactl --interleave=all` sets only the memory policy (round-robin page placement across all NUMA nodes). CPU affinity remains the default: all 32 cores are available. `os.sched_getaffinity(0)` will return the full set of all CPUs.
- D) Incorrect — Restricting to socket 1 cores would require `--cpunodebind=1`. `--interleave` has no effect on CPU affinity.

---

## Q12 — Reading NUMA Distance from Hardware Output

```
$ numactl --hardware
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
node 0 size: 96004 MB
node 0 free: 89100 MB
node 1 cpus: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
node 1 size: 96763 MB
node 1 free: 90200 MB
node distances:
node   0   1
  0:  10  21
  1:  21  10
```

A process running on core 20 (node 1) reads an array allocated entirely on node 0. Approximately how does the access latency compare to a process on core 4 (node 0) reading the same array?

**A)** Core 20's accesses are approximately 10x slower than core 4's accesses.
**B)** Core 20's accesses are approximately 2.1x slower than core 4's accesses (distance ratio 21/10).
**C)** Core 20's accesses are the same speed as core 4's because the inter-socket bus is transparent.
**D)** Core 20's accesses are faster because node 1 has more free memory (90200 MB vs. 89100 MB).

**Answer: B**

- A) The distance value of 21 does not mean 21x slower. Distance values are relative indices; 10 is the baseline (local) and 21 means the penalty is approximately 21/10 = 2.1x the local latency, not 21x.
- B) Correct — The NUMA distance matrix encodes relative latency as integer scores where 10 is the self (local) reference. Core 20 on node 1 accessing node 0 memory sees a relative distance of 21, so the latency is approximately 21/10 = 2.1x the local latency that core 4 would experience accessing the same data on node 0.
- C) Incorrect — The inter-socket interconnect adds real, measurable latency. It is not transparent in terms of performance. This is precisely why NUMA effects produce the observed speedup plateau.
- D) Incorrect — Free memory on the target NUMA node does not affect access latency. Latency is determined by the electrical path (local DRAM controller vs. inter-socket bus), not by how much memory is in use.

---

## Q13 — Spotting the NUMA Bottleneck in a Speedup Table

A student runs the reduction with numactl and without, and records:

```
Without numactl:
p=1:  S=1.00
p=8:  S=5.20
p=16: S=8.30
p=24: S=8.80   <-- near peak
p=32: S=7.10   <-- decline

With numactl --interleave=all:
p=1:  S=1.00
p=8:  S=5.00
p=16: S=9.10
p=24: S=12.50
p=32: S=15.20  <-- still growing
```

Which statement correctly interprets the difference between the two datasets?

**A)** Without numactl is always faster because all accesses are local for socket 0 workers.
**B)** With numactl is slower at p=8 (S=5.00 vs 5.20) because interleaving adds remote accesses for socket-0 workers, but faster at high core counts because socket-1 workers no longer pay the full remote penalty.
**C)** The two curves are effectively identical; the differences are within measurement noise.
**D)** Without numactl scales better at all core counts because no inter-socket traffic is introduced.

**Answer: B**

- A) Incorrect — For high core counts (p=24 and p=32), numactl is significantly faster (S=12.50 vs S=8.80 and S=15.20 vs S=7.10). Without numactl is only "faster" for small core counts that fit within one socket.
- B) Correct — At p=8 (all workers on socket 0), without numactl has all-local access (fastest), while with numactl has ~50% remote access for socket-0 workers (slightly slower: 5.00 vs 5.20). At p=24 and p=32, with numactl is dramatically faster because socket-1 workers get local access to the interleaved half of the data, whereas without numactl those workers pay full remote latency.
- C) Incorrect — The differences at p=24 (S=8.80 vs S=12.50, a 42% gap) and p=32 (S=7.10 vs S=15.20, a 114% gap) are far beyond measurement noise. These are significant, reproducible NUMA effects.
- D) Incorrect — Without numactl does not scale better at all core counts. It scales better only up to the socket boundary, then collapses. With numactl scales better across all cores beyond the first socket.

---

## Q14 — Effect of numactl on a Single-Process NumPy Sum

```python
import numpy as np
from time import perf_counter

arr = np.random.rand(100_000_000).astype('float32')

t0 = perf_counter()
result = np.sum(arr)
t1 = perf_counter()
print(f"np.sum time: {t1-t0:.3f}s, result: {result:.2f}")
```

This script is run twice on a dual-socket NUMA machine:

```bash
# Run 1 (no numactl):
python single_thread_sum.py

# Run 2 (with interleave):
numactl --interleave=all python single_thread_sum.py
```

Which run is expected to be faster, and why?

**A)** Run 2 (with interleave) is faster because it uses both DRAM controllers simultaneously.
**B)** Run 1 (no numactl) is faster or equal because all array pages land on the local NUMA node (first-touch on socket 0), giving 100% local accesses. Interleaving routes ~50% of accesses to the remote node.
**C)** Both runs are identical because NumPy's BLAS routines bypass NUMA effects.
**D)** Run 2 is faster because the kernel pre-fetches pages from node 1 into node 0's cache.

**Answer: B**

- A) Incorrect — A single-threaded NumPy sum accesses memory sequentially from one core; using both DRAM controllers only helps if two parallel data streams are created. A single thread issues one stream of requests; interleaving sends half of them to the remote node, adding latency without increasing throughput for one thread.
- B) Correct — Without numactl, the main thread (on socket 0) first-touches the array during `np.random.rand()`, placing all pages on NUMA node 0. `np.sum` then reads all pages locally. With interleave, ~50% of pages are on node 1, making ~50% of cache misses remote (higher latency). The single-threaded speed decreases slightly with interleaving — consistent with the lecture note that interleaving solves scaling but not necessarily speed.
- C) Incorrect — NumPy's BLAS or internal loops still issue regular memory load instructions that are subject to NUMA latency. NUMA is a hardware property of the memory subsystem; no software library can fully bypass it.
- D) Incorrect — Interleaving sets the page placement policy at allocation time; it does not trigger cross-node prefetching. Remote pages remain on node 1 and must be fetched over the inter-socket link at access time.

---

## Q15 — Pool.map Reduction: How Many Rounds?

```python
import numpy as np
import math

N = 64  # number of images

# How many rounds of pool.map are needed for a full binary tree reduction?
rounds = int(math.ceil(math.log2(N)))
print(f"Rounds of pool.map needed: {rounds}")
```

What does this code print, and what does each round of `pool.map` correspond to in the parallel reduction?

**A)** Prints `Rounds of pool.map needed: 64`; each round processes one image.
**B)** Prints `Rounds of pool.map needed: 6`; each round is one level of the binary tree where active elements accumulate the value of their neighbour at a doubling stride.
**C)** Prints `Rounds of pool.map needed: 32`; each round halves the problem size by one element.
**D)** Prints `Rounds of pool.map needed: 6`; each round spawns a new pool with twice as many workers.

**Answer: B**

- A) Incorrect — A linear reduction requiring 64 rounds would be a sequential scan, not a parallel reduction. `math.ceil(math.log2(64)) = 6`, not 64.
- B) Correct — `math.ceil(math.log2(64)) = ceil(6.0) = 6`. A binary tree reduction on N=64 elements requires 6 levels (rounds). At round j (stride s = 2^j), active elements at indices 0, 2s, 4s, ... each accumulate the element at index i + s. After 6 rounds, `arr[0]` holds the total sum.
- C) Incorrect — A sequential halving approach (removing one element at a time) would need 63 rounds for N=64, not 32, and would not be a tree reduction. The binary tree reduces N to 1 in log2(N) rounds.
- D) Incorrect — Each round uses the same pool (spawned once before the loop). The pool's worker count does not change between rounds. Only the stride `s` changes each round; no new pool is created.

---

## Q16 — Which LSF Flag Keeps Job on One Server?

```bash
#!/bin/bash
#BSUB -J celeba_numa
#BSUB -n 32
#BSUB -W 0:30
#BSUB -q hpc
#BSUB -o job_%J.out

numactl --interleave=all python reduction.py \
    /dtu/projects/02613_2025/data/celeba/celeba_100K.npy
```

A student notices that `shared_arr = mp.RawArray(...)` sometimes raises an error about worker processes not being able to access the shared memory. What is the most likely missing LSF directive, and what does it do?

**A)** `-R "span[hosts=1]"` — ensures all 32 cores are allocated from a single compute node so the shared memory segment is accessible to all workers.
**B)** `-R "rusage[mem=4096]"` — reserves 4 GB of memory for the job.
**C)** `-R "select[numa]"` — selects a node with NUMA topology enabled.
**D)** `-R "affinity[core(1)]"` — pins each process to a single core for NUMA locality.

**Answer: A**

- A) Correct — Without `span[hosts=1]`, LSF may allocate the 32 cores across multiple physical servers (e.g., 16 from server A and 16 from server B). Each server has its own memory namespace; a `mp.RawArray` created on server A's shared memory is not accessible to processes running on server B. Adding `-R "span[hosts=1]"` forces all 32 cores onto a single server, ensuring all worker processes share the same OS and can access the shared memory segment.
- B) Incorrect — `-R "rusage[mem=4096]"` reserves 4 GB of memory per slot. While important for avoiding out-of-memory errors, it does not affect whether processes can access shared memory. The access error is caused by processes being on different hosts, not by insufficient memory reservation.
- C) Incorrect — `select[numa]` is not a standard LSF host selection expression. LSF selects from available hosts based on the cluster configuration; NUMA is a property of the hardware and is typically always present on modern servers.
- D) Incorrect — `affinity[core(1)]` controls per-slot CPU affinity within a host. It does not address the multi-host shared memory problem and is unrelated to the error described.

---

## Q17 — Memory Policy for Dataset Larger Than One Node

```bash
# Dataset size: 200 GB
# Machine: dual-socket, 128 GB DRAM per socket (256 GB total)
# Goal: use all 48 cores (24 per socket) efficiently

# Option A
numactl --interleave=all python big_reduction.py data.npy

# Option B
numactl --cpunodebind=0 --membind=0 python big_reduction.py data.npy

# Option C
numactl --membind=0 python big_reduction.py data.npy

# Option D
numactl --cpunodebind=1 --membind=0 python big_reduction.py data.npy
```

Which option is correct for a 200 GB dataset on a machine with 128 GB per NUMA node, using all 48 cores?

**A)** Option A — `--interleave=all` distributes the 200 GB across both 128 GB banks, fitting the dataset and equalising access across all 48 cores.
**B)** Option B — `--cpunodebind=0 --membind=0` is best because local access is always faster.
**C)** Option C — `--membind=0` works because socket 0 has 128 GB, which is enough for 200 GB.
**D)** Option D — binding cores to node 1 but memory to node 0 maximises throughput.

**Answer: A**

- A) Correct — The 200 GB dataset is larger than one socket's 128 GB DRAM. `--membind=0` would fail (or use swap) because it cannot fit 200 GB in 128 GB. `--interleave=all` distributes the dataset across both 128 GB banks (100 GB each), fitting it within the total 256 GB while equalising access latency for all 48 cores. This is the correct approach when the dataset requires both NUMA nodes.
- B) Incorrect — `--membind=0` restricts all allocations to node 0, which has only 128 GB. Attempting to allocate 200 GB under `--membind=0` will fail with an out-of-memory error (numactl will not silently spill to node 1 under `--membind`; it will throw an error or use swap).
- C) Incorrect — 128 GB < 200 GB. Option C has the same problem as B: `--membind=0` forces all memory onto node 0's 128 GB, which cannot hold 200 GB.
- D) Incorrect — Running cores on node 1 while allocating all 200 GB on node 0 (128 GB limit — also insufficient) means all 24 socket-1 workers access local memory, but the 24 socket-0 workers access remote memory, and the memory doesn't even fit. This is both wrong and infeasible.

---

## Q18 — Full numactl + Pool Pattern: What Is Printed?

```python
import ctypes
import multiprocessing as mp
import numpy as np
from time import perf_counter

def init(shared_arr_):
    global shared_arr
    shared_arr = shared_arr_

def reduce_step(args):
    b, e, s, shape = args
    arr = np.frombuffer(shared_arr, dtype='float32').reshape((-1,) + shape)
    if b + s < len(arr):
        arr[b] += arr[b + s]

if __name__ == '__main__':
    N = 8
    data = np.arange(N, dtype='float32')[:, None, None, None]
    shape = data.shape[1:]
    shared_arr = mp.RawArray(ctypes.c_float, data.size)
    arr = np.frombuffer(shared_arr, dtype='float32').reshape(data.shape)
    np.copyto(arr, data)
    del data

    pool = mp.Pool(4, initializer=init, initargs=(shared_arr,))

    import math
    for j in range(int(math.ceil(math.log2(N)))):
        s = 2 ** j
        pool.map(reduce_step,
                 [(i, 0, s, shape) for i in range(0, N, 2*s)],
                 chunksize=1)

    print(f"arr[0] = {arr[0, 0, 0, 0]:.1f}")
    print(f"Expected sum = {sum(range(N)):.1f}")
```

The data is `[0, 1, 2, 3, 4, 5, 6, 7]`. What does the code print?

**A)** `arr[0] = 0.0 / Expected sum = 28.0` — the reduction is incorrect and returns 0.
**B)** `arr[0] = 28.0 / Expected sum = 28.0` — the binary tree reduction correctly sums all elements.
**C)** `arr[0] = 7.0 / Expected sum = 28.0` — only the last element is accumulated.
**D)** `arr[0] = 4.0 / Expected sum = 28.0` — only elements from the first half are summed.

**Answer: B**

- A) Incorrect — The reduction logic is correct (same pattern as `reduction_full.py`). Each round at stride s adds `arr[i + s]` into `arr[i]` for all active i. After ceil(log2(8)) = 3 rounds, `arr[0]` holds the total sum.
- B) Correct — The input is [0, 1, 2, 3, 4, 5, 6, 7]. sum(range(8)) = 28. Round 1 (s=1): arr[0]+=arr[1] → 1, arr[2]+=arr[3] → 5, arr[4]+=arr[5] → 9, arr[6]+=arr[7] → 13. Array: [1, 1, 5, 3, 9, 5, 13, 7]. Round 2 (s=2): arr[0]+=arr[2] → 6, arr[4]+=arr[6] → 22. Array: [6, 1, 5, 3, 22, 5, 13, 7]. Round 3 (s=4): arr[0]+=arr[4] → 28. `arr[0] = 28.0`, matching the expected sum.
- C) Incorrect — The final round accumulates arr[4] (which already holds the partial sum 22) into arr[0], not arr[7]. The result is 6 + 22 = 28, not 7.
- D) Incorrect — All 8 elements participate in the reduction. After 3 rounds, arr[0] accumulates contributions from all 8 input values through the tree structure.

---

## Set 3 — Extended Practice

> Targets numactl --show output, first-touch in forked workers, sched_getaffinity under cpunodebind, membind strict failure, reduction state tracing, fork policy inheritance, active task count per round, composing interleave with cpunodebind, identifying first-touch from code structure, and compute-bound vs memory-bound numactl effects.

---

## Q19 — What Does numactl --show Print?

```bash
$ numactl --interleave=all numactl --show
```

On a 32-core dual-socket machine (node 0: cores 0-15, node 1: cores 16-31), what does the inner `numactl --show` print about the memory policy?

**A)** `policy: default` — `--interleave=all` has no effect on what `--show` reports.
**B)** `policy: interleave` with `nodes (0-1)` — the interleave policy is active and `--show` confirms it.
**C)** `policy: bind` with `nodes (0)` — interleaving converts to a bind policy on node 0.
**D)** `policy: preferred` with `nodes (0)` — interleaving sets a preference for node 0.

**Answer: B**

- A) Incorrect — `numactl --interleave=all` sets the interleave memory policy for the launched process. `numactl --show` (run as a child of the outer numactl) inherits that policy and reports it faithfully, not the default.
- B) Correct — `numactl --show` prints the current NUMA memory and CPU policy of the running process. When launched under `numactl --interleave=all`, it reports `policy: interleave` and lists the nodes being interleaved across (in this case nodes 0 and 1). This is the correct way to verify that interleaving is active.
- C) Incorrect — `--interleave=all` sets policy type "interleave", not "bind". `--membind` sets a bind policy. These are different `set_mempolicy()` modes and are reported distinctly by `--show`.
- D) Incorrect — A preferred policy (`--preferred=N`) is set by a different numactl flag. `--interleave=all` sets the interleave mode, not a preference. `--show` would report `policy: preferred` only if `--preferred` had been used.

---

## Q20 — First-Touch in a Worker Process

```python
import ctypes
import multiprocessing as mp
import numpy as np

def worker_fill(args):
    shared_arr, start, end = args
    arr = np.frombuffer(shared_arr, dtype='float32')
    arr[start:end] = 1.0   # <-- first write to these pages

if __name__ == '__main__':
    N = 10_000_000
    shared_arr = mp.RawArray(ctypes.c_float, N)
    # Do NOT write to shared_arr here in the main process

    pool = mp.Pool(4)
    chunk = N // 4
    pool.map(worker_fill,
             [(shared_arr, i*chunk, (i+1)*chunk) for i in range(4)])
    pool.close()
```

On a dual-socket NUMA machine without numactl, with 2 workers on socket 0 (cores 0-1) and 2 workers on socket 1 (cores 16-17), where do the pages of `shared_arr` land?

**A)** All on NUMA node 0 — the main process created `mp.RawArray`, so all pages are first-touched by the main process on node 0.
**B)** Approximately half on node 0 and half on node 1 — each worker first-touches its own chunk, so pages land on the node of the filling worker.
**C)** All on NUMA node 1 — shared memory always allocates on the secondary node.
**D)** Evenly split regardless of worker placement — `mp.RawArray` uses interleaved allocation by default.

**Answer: B**

- A) Incorrect — The main process does NOT write to `shared_arr` after creating it (note the comment "Do NOT write to shared_arr here"). `mp.RawArray` creation reserves the virtual mapping but does not first-touch the pages. The first writes happen in the workers.
- B) Correct — Workers 0 and 1 run on socket 0 (cores 0-1) and first-touch the first and second chunks. Workers 2 and 3 run on socket 1 (cores 16-17) and first-touch the third and fourth chunks. Because Linux first-touch places pages on the touching core's local node, approximately half the array lands on node 0 and half on node 1. This is effectively manual interleaving via worker placement.
- C) Incorrect — There is no policy that puts shared memory on the secondary node. Placement is determined by first-touch, which in this case is split across both nodes based on worker assignment.
- D) Incorrect — `mp.RawArray` uses the default NUMA policy (first-touch), not automatic interleaving. Pages are placed based on which core writes to them first, not by any built-in round-robin mechanism in `RawArray`.

---

## Q21 — What Does os.sched_getaffinity Print Under --cpunodebind=0?

```bash
numactl --cpunodebind=0 python -c "
import os
aff = os.sched_getaffinity(0)
print(f'CPU count: {len(aff)}')
print(f'Min CPU: {min(aff)}, Max CPU: {max(aff)}')
"
```

On a 32-core dual-socket machine where node 0 owns cores 0–15 and node 1 owns cores 16–31, what does this script print?

**A)** `CPU count: 32 / Min CPU: 0, Max CPU: 31` — `--cpunodebind=0` has no effect on affinity.
**B)** `CPU count: 1 / Min CPU: 0, Max CPU: 0` — `--cpunodebind=0` pins to a single core (core 0).
**C)** `CPU count: 16 / Min CPU: 0, Max CPU: 15` — `--cpunodebind=0` restricts to all 16 cores of node 0.
**D)** `CPU count: 16 / Min CPU: 16, Max CPU: 31` — `--cpunodebind=0` moves all cores to node 1.

**Answer: C**

- A) Incorrect — `--cpunodebind=0` explicitly sets the CPU affinity mask to the set of cores belonging to NUMA node 0. `os.sched_getaffinity(0)` reads that affinity mask and will show only the node-0 cores.
- B) Incorrect — `--cpunodebind=0` binds to all cores of NUMA node 0, not to core 0 alone. Binding to a single core requires `--physcpubind=0`. The `cpunodebind` flag operates at the node (socket) granularity, not the individual-core granularity.
- C) Correct — Node 0 has cores 0–15 (16 cores). `--cpunodebind=0` sets the affinity mask to exactly those 16 cores. `os.sched_getaffinity(0)` returns that set, so `len(aff)` is 16, `min(aff)` is 0, and `max(aff)` is 15.
- D) Incorrect — `--cpunodebind=0` binds to node 0's cores (0–15), not node 1's cores (16–31). Swapping to node 1 would require `--cpunodebind=1`.

---

## Q22 — Does --membind=0 Prevent Allocation on Node 1?

```python
import numpy as np
import subprocess, sys

# Run this script under: numactl --membind=0 python script.py
# Node 0 has 8 GB free. This allocation requests 12 GB.

try:
    arr = np.zeros((3_000_000_000,), dtype='float32')   # ~12 GB
    print("Allocation succeeded")
except MemoryError:
    print("MemoryError: allocation failed")
```

On a machine where NUMA node 0 has 8 GB free and node 1 has 16 GB free, running `numactl --membind=0 python script.py`, what is printed?

**A)** `Allocation succeeded` — `--membind=0` tries node 0 first but falls back to node 1 automatically.
**B)** `MemoryError: allocation failed` — `--membind=0` is a strict policy; if node 0 cannot satisfy the allocation, it fails rather than using node 1.
**C)** `Allocation succeeded` — numpy bypasses NUMA policies and allocates wherever memory is available.
**D)** `Allocation succeeded` — the OS uses swap on node 0 instead of allocating on node 1.

**Answer: B**

- A) Incorrect — `--membind=0` (corresponding to `MPOL_BIND` in the kernel) is a strict binding policy, not a preference. It does not silently fall back to node 1. The flag to use for "prefer node 0, fall back if needed" is `--preferred=0`.
- B) Correct — With `--membind=0`, if node 0 cannot satisfy the 12 GB allocation (only 8 GB free), the kernel returns `ENOMEM`. NumPy's allocator propagates this as a Python `MemoryError`. The 16 GB free on node 1 is irrelevant; `--membind` strictly prohibits any allocation on node 1.
- C) Incorrect — NumPy uses standard C `malloc`/`mmap` calls that go through the kernel's memory allocator, which respects the process's NUMA policy. NumPy has no NUMA-bypass mechanism.
- D) Incorrect — Swap is a last resort when physical memory across all nodes is exhausted. With `--membind=0`, the kernel enforces the bind policy before consulting swap. The allocation fails on the policy check, not on a general OOM condition.

---

## Q23 — Predict the Array State After One Reduction Round

```python
import numpy as np

# Initial shared array (simplified to 1D for illustration)
arr = np.array([3.0, 7.0, 2.0, 5.0, 1.0, 4.0, 6.0, 8.0], dtype='float32')

# Round j=0, stride s=1
# Tasks: i in range(0, 8, 2*1) = [0, 2, 4, 6]
# Each task: arr[i] += arr[i + 1]
s = 1
for i in range(0, len(arr), 2*s):
    if i + s < len(arr):
        arr[i] += arr[i + s]

print(arr[:4])
```

What does `print(arr[:4])` output?

**A)** `[10.  2.  6.  6.]`
**B)** `[ 3.  7.  2.  5.]` — the array is unchanged.
**C)** `[10.  7.  7.  5.]`
**D)** `[ 5.  5.  7.  5.]`

**Answer: C**

- A) Incorrect — arr[2] should be 2.0+5.0=7.0 (not 6.0) and arr[3] should remain 5.0 (unchanged, it is not an active index at stride 1). Check: active indices are 0, 2, 4, 6.
- B) Incorrect — The loop does modify the array. At stride s=1, active indices (multiples of 2*1=2) are 0, 2, 4, 6. arr[0]+=arr[1] and arr[2]+=arr[3] both change.
- C) Correct — Active indices at s=1 are {0, 2, 4, 6}. arr[0] += arr[1] → 3+7=10. arr[2] += arr[3] → 2+5=7. arr[4] += arr[5] → 1+4=5 (not printed). arr[6] += arr[7] → 6+8=14 (not printed). arr[1], arr[3] are not active so they remain 7.0 and 5.0. `arr[:4]` is `[10.  7.  7.  5.]`.
- D) Incorrect — arr[0] = 3+7 = 10, not 5. arr[1] = 7 (unchanged). Summing adjacent pairs at even indices gives 10, not 5, for the first element.

---

## Q24 — What Is the NUMA Policy After fork()?

```python
import multiprocessing as mp
import subprocess

def check_policy():
    # Run numactl --show inside the worker
    result = subprocess.run(
        ['numactl', '--show'],
        capture_output=True, text=True
    )
    print(result.stdout.split('\n')[0])  # print first line (policy)

if __name__ == '__main__':
    # Launched as: numactl --interleave=all python policy_check.py
    p = mp.Process(target=check_policy)
    p.start()
    p.join()
```

When `policy_check.py` is run as `numactl --interleave=all python policy_check.py`, what does the spawned `numactl --show` inside the worker print on its first line?

**A)** `policy: default` — the forked process does not inherit the parent's NUMA policy.
**B)** `policy: interleave` — the forked worker inherits the parent's interleave policy via `fork()`.
**C)** `policy: bind` — `fork()` converts interleave policy to a bind policy for safety.
**D)** The command raises an error because `numactl --show` cannot be called from a subprocess.

**Answer: B**

- A) Incorrect — NUMA memory policies set via `set_mempolicy()` are part of the Linux process's memory management context, which is fully copied during `fork()`. The forked worker starts with an identical copy of the parent's NUMA policy.
- B) Correct — `mp.Process` uses `fork()` (on Linux with the default start method). `fork()` creates an exact copy of the parent process, including its NUMA memory policy. Since the parent was launched under `numactl --interleave=all`, the forked child also has `policy: interleave` with nodes (0-1). `numactl --show` inside the worker confirms this.
- C) Incorrect — `fork()` copies the policy type faithfully; it does not transform interleave into bind. The two are distinct kernel policy modes (`MPOL_INTERLEAVE` vs `MPOL_BIND`) and `fork()` preserves the exact mode.
- D) Incorrect — `subprocess.run(['numactl', '--show'], ...)` is a valid way to query the current process's NUMA policy. `numactl --show` reads the policy of the process running it (the subprocess itself), which has inherited the interleave policy.

---

## Q25 — How Many Active Tasks in Round j of a Binary Reduction?

```python
import math

N = 128   # number of images

for j in range(int(math.ceil(math.log2(N)))):
    s = 2 ** j
    active_tasks = list(range(0, N, 2 * s))
    # Filter: only tasks where i + s < N are valid
    valid_tasks = [i for i in active_tasks if i + s < N]
    print(f"Round j={j}, stride s={s}: {len(valid_tasks)} active tasks")
```

What does the code print for `j=0` and `j=6` (the last round)?

**A)** `j=0: 64 tasks, j=6: 0 tasks`
**B)** `j=0: 64 tasks, j=6: 1 task`
**C)** `j=0: 128 tasks, j=6: 1 task`
**D)** `j=0: 64 tasks, j=6: 64 tasks`

**Answer: B**

- A) Incorrect — At j=6 (s=64), `range(0, 128, 128)` produces `[0]`, and since 0+64=64 < 128, the task at i=0 is valid. There is 1 valid task, not 0.
- B) Correct — At j=0 (s=1): `range(0, 128, 2)` produces 64 indices {0,2,4,...,126}. All satisfy i+1<128, so 64 valid tasks. At j=6 (s=64): `range(0, 128, 128)` produces only `[0]`. i=0, i+s=64 < 128, so 1 valid task. The binary tree halves the active task count each round, going from N/2 down to 1 over log2(N) rounds.
- C) Incorrect — At j=0, `range(0, 128, 2*1)` = `range(0, 128, 2)` produces 64 indices, not 128. There are N/2 active tasks in the first round, not N.
- D) Incorrect — The number of active tasks halves each round (the defining property of a binary tree reduction). j=6 has only 1 active task; j=0 has 64. The count cannot stay constant at 64.

---

## Q26 — numactl --interleave=all With --cpunodebind=1: What Happens?

```bash
numactl --interleave=all --cpunodebind=1 python -c "
import os
import numpy as np

# What NUMA policy applies here?
arr = np.zeros(10_000_000, dtype='float32')
aff = os.sched_getaffinity(0)
print(f'Affinity core range: {min(aff)}-{max(aff)}')
print(f'Affinity core count: {len(aff)}')
"
```

On a 32-core dual-socket machine (node 0: cores 0-15, node 1: cores 16-31), what does this script print, and on which NUMA nodes does `arr` reside?

**A)** Affinity core range: 0-31, count 32; arr is fully on node 0 — flags conflict, numactl uses last flag.
**B)** Affinity core range: 16-31, count 16; arr pages are interleaved across nodes 0 and 1 — flags are independent and both apply.
**C)** Affinity core range: 0-15, count 16; arr is fully on node 1 — `--cpunodebind=1` overrides interleave.
**D)** The command fails with an error — `--interleave=all` and `--cpunodebind=1` cannot be used together.

**Answer: B**

- A) Incorrect — numactl does not use "last flag wins" semantics. `--cpunodebind` and `--interleave` are orthogonal flags controlling independent subsystems (CPU affinity vs memory policy). Both are applied simultaneously.
- B) Correct — `--cpunodebind=1` sets CPU affinity to cores 16-31 (node 1's 16 cores). `--interleave=all` sets the memory policy to interleave across all nodes (0 and 1). These flags are independent: one controls where computation runs, the other controls where memory is allocated. The `zeros` call first-touches pages from core(s) on node 1, but under interleave policy the OS places successive pages on alternating nodes, regardless of which core is touching them.
- C) Incorrect — `--cpunodebind=1` restricts cores to node 1 but does not override the memory policy. `--interleave=all` continues to apply, distributing pages across both nodes. CPU affinity and memory policy are orthogonal.
- D) Incorrect — numactl allows combining any memory policy flag (`--interleave`, `--membind`, `--preferred`) with any CPU binding flag (`--cpunodebind`, `--physcpubind`). This is explicitly shown in the lecture slide listing multiple numactl flags that can be used together.

---

## Q27 — Identifying the First-Touch Core from Code

```python
import ctypes
import multiprocessing as mp
import numpy as np

def fill_worker(shared_arr, start, end):
    arr = np.frombuffer(shared_arr, dtype='float32')
    arr[start:end] = 99.0   # first write to pages [start:end]

if __name__ == '__main__':
    N = 8_000_000
    shared_arr = mp.RawArray(ctypes.c_float, N)

    # Main process fills the first half
    arr = np.frombuffer(shared_arr, dtype='float32')
    arr[:N//2] = 1.0    # <-- A: main process writes pages 0 to N//2

    # Worker fills the second half
    p = mp.Process(target=fill_worker, args=(shared_arr, N//2, N))
    p.start()
    p.join()
    # <-- B: worker writes pages N//2 to N
```

On a dual-socket NUMA machine without numactl, where does each half of `shared_arr` reside?

**A)** Both halves on node 0 — the main process created `shared_arr` so all pages are bound to node 0.
**B)** First half on node 0 (first-touched by main process on socket 0), second half on node 1 (first-touched by the worker, which the OS schedules on socket 1 by default).
**C)** First half on node 0 (first-touched by main process), second half on node 0 as well — forked workers inherit the parent's CPU affinity and always run on socket 0.
**D)** The placement is unpredictable — the OS may schedule the main process or the worker on either socket without any pattern.

**Answer: C**

- A) Incorrect — The main process creates `shared_arr` (virtual mapping only), but the actual physical page placement is determined by first-touch writes, not by creation. The main process does write the first half on socket 0, placing those pages on node 0. The second half is written by the worker.
- B) Incorrect — The worker inherits the parent's CPU affinity mask via `fork()`. Without numactl, the parent's affinity is typically the full set of all cores (0-31). The OS scheduler places the worker on any available core; it is not specifically scheduled on socket 1. There is no guarantee the worker ends up on node 1.
- C) Correct — Without any CPU affinity restriction, both the main process and the forked worker can run on any core. In practice on a lightly loaded HPC node, the scheduler tends to keep the worker near the parent (often on socket 0), so the second half is also likely first-touched on socket 0 and lands on node 0. The key insight: `fork()` inherits the parent's unrestricted affinity (all cores), and without numactl there is no mechanism guaranteeing the worker runs on socket 1.
- D) Incorrect — While the OS scheduler does have discretion, on a lightly loaded system with a single forked process, the common outcome is socket 0 placement for both. The answer is not perfectly deterministic, but the most likely outcome is both halves on node 0 — making C the correct answer among the options.

---

## Q28 — Compute-Bound vs Memory-Bound: Does numactl Help?

```python
import numpy as np
from time import perf_counter

N = 10_000_000

# Workload A: memory-bound — sequential read of a large array
arr = np.random.rand(N).astype('float32')
t0 = perf_counter()
result = np.sum(arr)          # memory bandwidth limited
t_mem = perf_counter() - t0

# Workload B: compute-bound — repeated multiply-add on a tiny array
tiny = np.ones(100, dtype='float32')
t0 = perf_counter()
for _ in range(100_000):
    tiny = tiny * 1.0001 + 0.5   # fits entirely in L1 cache
t_cpu = perf_counter() - t0

print(f"Memory-bound time: {t_mem:.4f}s")
print(f"Compute-bound time: {t_cpu:.4f}s")
```

If this script is re-run with `numactl --interleave=all python script.py`, which workload is more likely to change noticeably in runtime, and why?

**A)** Workload B (compute-bound) — interleaving speeds up compute-intensive loops by distributing CPU load.
**B)** Workload A (memory-bound) — interleaving affects DRAM page placement and can alter memory latency and bandwidth, impacting workloads that regularly miss cache.
**C)** Neither — numactl only affects multiprocessing programs; single-threaded programs are unaffected.
**D)** Both equally — numactl changes CPU frequency, which affects both workloads proportionally.

**Answer: B**

- A) Incorrect — `--interleave=all` is a memory policy flag that controls where physical pages are placed. It has no effect on CPU scheduling, core assignment, or the execution of compute-intensive loops that operate entirely within cache. Workload B fits in L1 cache (100 floats = 400 bytes) and never touches DRAM.
- B) Correct — Workload A (`np.sum` on a 10M-element float32 array = 40 MB) exceeds L3 cache and repeatedly fetches data from DRAM. Changing the NUMA memory policy from first-touch (all local to socket 0) to interleave (50% remote) directly alters the latency of each cache miss. Workload A is memory-bandwidth limited and will be affected. Workload B fits in L1 cache and never issues DRAM requests, so interleaving has no measurable effect on it.
- C) Incorrect — numactl applies to any process, including single-threaded ones. Memory policy affects every memory allocation and every cache miss that goes to DRAM, regardless of thread count.
- D) Incorrect — numactl does not modify CPU frequency. It modifies memory placement policy only. CPU frequency is governed by the system's power management (P-states) and is not influenced by numactl.
