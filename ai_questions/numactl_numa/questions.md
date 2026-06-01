# numactl & NUMA Topology — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — What NUMA Stands For](#q1--what-numa-stands-for)
- [Q2 — NUMA Node Count on DTU XeonGold](#q2--numa-node-count-on-dtu-xeongold)
- [Q3 — Where Memory is Allocated by Default](#q3--where-memory-is-allocated-by-default)
- [Q4 — Effect of Default Allocation on Socket 1 Cores](#q4--effect-of-default-allocation-on-socket-1-cores)
- [Q5 — What numactl --interleave=all Controls](#q5--what-numactl---interleaveall-controls)
- [Q6 — The NUMA Plateau in Speedup Curves](#q6--the-numa-plateau-in-speedup-curves)
- [Q7 — NUMA Pros: Double Bandwidth](#q7--numa-pros-double-bandwidth)
- [Q8 — When interleave=all Fixes Scaling](#q8--when-interleaveall-fixes-scaling)
- [Q9 — What --membind Does](#q9--what---membind-does)
- [Q10 — What --cpunodebind Does](#q10--what---cpunodebind-does)
- [Q11 — interleave=all Solves Scaling But Not Speed](#q11--interleaveall-solves-scaling-but-not-speed)
- [Q12 — Why multiprocessing.Pool Suffers NUMA Effects](#q12--why-multiprocessingpool-suffers-numa-effects)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2--generated-practice-questions-exam-day-focus)
- [Q13 — Reading numactl --hardware Output](#q13--reading-numactl---hardware-output)
- [Q14 — When Single-Node Binding is Better Than Interleaving](#q14--when-single-node-binding-is-better-than-interleaving)
- [Q15 — LSF span[hosts=1] and NUMA](#q15--lsf-spanhosts1-and-numa)
- [Q16 — CelebA Reduction: Where the NUMA Plateau Appears](#q16--celeba-reduction-where-the-numa-plateau-appears)
- [Q17 — numactl --physcpubind vs --cpunodebind](#q17--numactl---physcpubind-vs---cpunodebind)
- [Q18 — Single-Threaded Workload and numactl](#q18--single-threaded-workload-and-numactl)
- [Q19 — Interleaving Policy Granularity](#q19--interleaving-policy-granularity)
- [Q20 — NUMA Effect on the Speedup Curve Shape](#q20--numa-effect-on-the-speedup-curve-shape)
- [Q21 — Why the First Socket Sees Fast Access](#q21--why-the-first-socket-sees-fast-access)
- [Q22 — Combining numactl Flags](#q22--combining-numactl-flags)

---

> Topics: NUMA nodes, local vs remote memory, numactl flags, interleave policy, NUMA plateau in speedup.
> Exam frequency: **Week 6 topic**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--what-numa-stands-for)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — What NUMA Stands For

> **Week reference:** Week 6

**Mental Model:** The acronym itself encodes the key fact — memory access time is not the same for all cores; it depends on which physical memory bank holds the data relative to which socket is requesting it.

What does the acronym NUMA stand for?

- A) Non-Uniform Memory Allocation
- B) Networked Uniform Memory Access
- C) Non-Uniform Memory Access
- D) Node-Unified Multi-core Architecture

**Answer: C**

- A) Incorrect — "Allocation" is close but wrong; the "A" in NUMA refers to Access, not Allocation. The key property is that the latency of reading data differs depending on which node the memory bank lives on.
- B) Incorrect — Memory in a NUMA system is not networked; both NUMA nodes are on the same motherboard connected by an inter-socket bus (e.g., Intel QPI or AMD Infinity Fabric). "Networked" implies separate machines.
- C) Correct — NUMA stands for Non-Uniform Memory Access. The core idea is that on a multi-socket server, a CPU on socket 0 accesses its own local DRAM quickly, but accessing DRAM on socket 1 is slower because it crosses the inter-socket interconnect.
- D) Incorrect — This invented phrase has no relation to the actual architecture. NUMA describes a physical topology, not a software abstraction of multi-core systems.

---

## Q2 — NUMA Node Count on DTU XeonGold

> **Week reference:** Week 6

**Mental Model:** On the DTU HPC cluster, each physical compute node is a dual-socket server — two physical CPU packages on one motherboard. Each socket forms one NUMA node, giving exactly two NUMA nodes per compute node.

The DTU HPC cluster uses Xeon Gold 6226R processors. How many NUMA nodes does a single compute node have, and what defines each NUMA node?

- A) 1 NUMA node — the entire node is one flat memory pool.
- B) 2 NUMA nodes — one per physical CPU socket, each with its own local DRAM bank.
- C) 4 NUMA nodes — one per die, as each Xeon Gold is a multi-die chip.
- D) 16 NUMA nodes — one per core.

**Answer: B**

- A) Incorrect — A single flat memory pool (UMA) would mean all memory access times are equal. The whole point of the numactl exercise is that this is not the case; there are two distinct NUMA nodes, each with local DRAM.
- B) Correct — The node has two physical CPU sockets (socket 0 and socket 1), each connected to its own local DRAM. These two domains form NUMA node 0 and NUMA node 1. Cores on socket 0 access node 0 DRAM fast and node 1 DRAM slow.
- C) Incorrect — While some modern processors are chiplet-based, the Xeon Gold 6226R is treated as a single-die processor for NUMA purposes. The OS reports 2 NUMA nodes on such a dual-socket server, not 4.
- D) Incorrect — NUMA nodes correspond to sockets (CPU packages), not individual cores. Cores within the same socket all share local access to the same DRAM bank.

---

## Q3 — Where Memory is Allocated by Default

> **Week reference:** Week 6

**Mental Model:** Linux uses a "first-touch" allocation policy: memory pages are placed on the NUMA node of the core that first writes to them. Since Python starts single-threaded and `np.load()` runs on the main process (on socket 0), the entire array lands on NUMA node 0 by default.

When a Python program calls `np.load()` to load a large array on a dual-socket NUMA system, where is the array's memory allocated by default?

- A) Evenly split between both NUMA nodes in round-robin.
- B) On the NUMA node of the CPU core that first touches (writes) the pages — typically socket 0.
- C) On the NUMA node with the most free memory.
- D) On a NUMA node selected randomly by the OS to balance load.

**Answer: B**

- A) Incorrect — Round-robin distribution is exactly what `numactl --interleave=all` provides, but it is not the default. Without numactl, Linux uses first-touch placement.
- B) Correct — Linux uses first-touch policy: the physical pages backing a memory allocation are placed on the NUMA node of the core that first writes to them. `np.load()` runs on the main thread, which typically runs on socket 0 (NUMA node 0), so the entire array is placed there.
- C) Incorrect — Linux does not dynamically move allocations to the node with most free memory. While it will use remote memory if local memory is exhausted, it does not proactively balance based on free space under normal conditions.
- D) Incorrect — Linux memory allocation is deterministic and policy-driven, not random. The default policy is first-touch, which places pages on the allocating core's local node.

---

## Q4 — Effect of Default Allocation on Socket 1 Cores

> **Week reference:** Week 6

**Mental Model:** Once the array is on NUMA node 0 (socket 0's DRAM), every access from a socket 1 core must cross the inter-socket interconnect. This adds latency (roughly 1.5–2x longer than local access), and since the parallel reduction is memory-bound, this remote penalty directly limits throughput for socket 1 workers.

In the CelebA parallel reduction exercise (without numactl), the shared float32 array is loaded by the main process. What happens to cores on socket 1 when they perform the reduction?

- A) They access memory at the same speed as socket 0 cores because the OS transparently migrates pages to local memory.
- B) They access remote memory on NUMA node 0 over the inter-socket link, which is slower than local access.
- C) They access GPU memory because GPU acceleration is enabled automatically.
- D) They are idle because the OS pins all processes to socket 0.

**Answer: B**

- A) Incorrect — The OS does not transparently migrate hot pages between NUMA nodes during program execution on this cluster. Page migration exists as a feature (autonuma/NUMA balancing) but is not guaranteed to be enabled or fast enough to compensate for the inter-socket latency during a short benchmark.
- B) Correct — The array lives on NUMA node 0. Cores on socket 1 (NUMA node 1) must fetch every cache line across the inter-socket interconnect. This "remote access" is significantly slower, causing the speedup to plateau and even decline when processes spread across both sockets.
- C) Incorrect — The exercise uses `multiprocessing.Pool` on CPU cores only. There is no GPU involved in this exercise.
- D) Incorrect — The OS does not pin all processes to socket 0. With a large pool, workers are assigned to cores across both sockets. The problem is not idleness but the latency penalty paid by socket 1 workers.

---

## Q5 — What numactl --interleave=all Controls

> **Week reference:** Week 6

**Mental Model:** `numactl --interleave=all` is a memory placement policy, not a CPU scheduling policy. It tells the kernel to allocate successive memory pages in round-robin across all NUMA nodes, so no single node holds all the data. It does NOT force threads onto specific cores.

Which of the following correctly describes what `numactl --interleave=all` does?

- A) It distributes threads across CPU sockets in round-robin to balance CPU load.
- B) It distributes memory page allocations across all NUMA nodes in round-robin, so roughly half the data resides on each node.
- C) It pins the process to a single NUMA node and allocates all memory locally.
- D) It doubles the memory bandwidth by enabling dual-channel mode on each socket.

**Answer: B**

- A) Incorrect — `numactl --interleave=all` controls memory placement only. It does not control which cores run the threads; the OS scheduler handles thread placement independently.
- B) Correct — With `--interleave=all`, the kernel allocates consecutive 4 KB pages alternating between NUMA node 0 and NUMA node 1 (round-robin). After interleaving, roughly half the array resides on each socket's local DRAM. Any core accessing the array will find about half the accesses local and half remote, giving all cores a roughly equal average latency.
- C) Incorrect — Pinning to a single node and allocating all memory locally is the behaviour of `numactl --cpunodebind=0 --membind=0`, not `--interleave=all`.
- D) Incorrect — Dual-channel mode is a hardware property of the memory controller and is always enabled; numactl does not control it. The bandwidth benefit of interleaving comes from splitting access pressure between two separate DRAM controllers, not from enabling a hardware feature.

---

## Q6 — The NUMA Plateau in Speedup Curves

> **Week reference:** Week 6

**Mental Model:** The NUMA plateau appears at approximately 50% of the total cores (the size of one socket). Up to that point, all workers fit on one socket and use only fast local memory. Adding workers beyond the first socket's capacity forces remote memory accesses, which counteract any additional parallelism.

In the CelebA reduction speedup plot (without numactl), the speedup improves up to about 16 cores, then stops increasing or decreases. Why does the plateau occur at approximately 50% of cores?

- A) Python's GIL prevents more than one thread from running at a time beyond 16 threads.
- B) The cost of spawning additional processes exceeds the benefit beyond 16 processes.
- C) All cores up to the first socket boundary (16 cores on socket 0) access local memory on NUMA node 0; additional cores land on socket 1 and pay a remote-memory penalty that offsets the parallelism gain.
- D) Amdahl's Law predicts a plateau at 50% of the cores due to the serial fraction of the program.

**Answer: C**

- A) Incorrect — The exercise uses `multiprocessing.Pool`, not multithreading. The GIL does not affect multiprocessing workers; each worker is a separate OS process with its own interpreter.
- B) Incorrect — Process spawning overhead is a one-time cost paid at pool creation, not a per-task overhead that grows with core count. The plateau is not caused by spawning overhead.
- C) Correct — The Xeon Gold 6226R dual-socket node has 16 cores per socket (32 cores total). Workers up to 16 run on socket 0 and access the array in local DRAM on NUMA node 0 at full speed. The 17th and beyond run on socket 1 and must cross the inter-socket link for every array access, paying a heavy latency penalty that erases the speedup from extra parallelism.
- D) Incorrect — Amdahl's Law predicts an asymptotic plateau (S_max = 1/(1-F)) that applies equally at all core counts; it does not specifically produce a plateau at 50% of cores. The observed plateau at the socket boundary is a hardware NUMA effect, not an Amdahl serial-fraction effect.

---

## Q7 — NUMA Pros: Double Bandwidth

> **Week reference:** Week 6

**Mental Model:** A dual-socket NUMA system has two independent DRAM controllers, one per socket. When data is split between them (as with interleaving), both controllers can serve memory requests simultaneously, effectively doubling the aggregate memory bandwidth available to the whole system.

The lecture identifies one advantage of NUMA architecture. Which of the following correctly describes a benefit of having two NUMA nodes?

- A) Data can be migrated instantly between nodes with zero latency.
- B) The aggregate memory bandwidth doubles because both sockets have independent DRAM controllers that can serve requests in parallel.
- C) A single core can access twice the memory capacity without any latency penalty.
- D) NUMA eliminates the need for L3 cache on each socket.

**Answer: B**

- A) Incorrect — There is no zero-latency migration between NUMA nodes. Moving data between nodes requires copying it over the inter-socket interconnect, which is the slow path, not an instant operation.
- B) Correct — Each socket has its own DRAM controller. When both sockets are serving accesses (e.g., after interleaving), the system achieves up to 2x aggregate bandwidth. The lecture explicitly notes "Double the bandwidth" and "Data heavy tasks only stalls one CPU" as pros of NUMA.
- C) Incorrect — A single core can access the full memory space of both nodes, but remote accesses carry a latency penalty. There is no magic that makes remote access as fast as local access.
- D) Incorrect — Each socket retains its own L3 cache regardless of NUMA topology. NUMA describes the memory hierarchy across sockets, not the on-chip cache hierarchy within a socket.

---

## Q8 — When interleave=all Fixes Scaling

> **Week reference:** Week 6

**Mental Model:** `--interleave=all` equalises average memory access latency across all cores, eliminating the asymmetry between socket 0 and socket 1 workers. This restores linear-looking speedup growth because no worker suffers a disproportionate penalty compared to others.

After applying `numactl --interleave=all` to the CelebA reduction, what change in the speedup plot is expected?

- A) Speedup becomes perfectly linear (S = p) because memory latency is eliminated.
- B) Speedup now increases monotonically with every core added, and the plateau or drop beyond the first socket disappears.
- C) Speedup remains the same as without numactl because interleaving does not affect performance.
- D) Speedup decreases because accessing two NUMA nodes introduces overhead versus accessing one.

**Answer: B**

- A) Incorrect — Interleaving equalises average latency but does not eliminate latency. Every memory access is still slower than compute, and Amdahl's serial fraction (pool.map overhead, logging) still applies. Perfect linear speedup is not expected.
- B) Correct — With interleaved memory, all cores — whether on socket 0 or socket 1 — see roughly the same average access time (mix of 50% local, 50% remote). This eliminates the asymmetry that caused the plateau at the socket boundary, so speedup continues to grow as more cores are added.
- C) Incorrect — The numactl notes and exercise solutions explicitly confirm that interleaving does change the speedup curve. The plateau disappears; speedup scales with every added core.
- D) Incorrect — The extra remote accesses introduced by interleaving do not increase overhead relative to the baseline (where all socket 1 accesses were already remote). In fact, interleaving can improve performance for socket 1 workers by giving them local access to half the data.

---

## Q9 — What --membind Does

> **Week reference:** Week 6

**Mental Model:** `--membind=N` restricts all memory allocations to NUMA node N. This guarantees that every allocated page is local to node N, which maximises local memory bandwidth for cores on that node at the cost of not using memory on the other node.

What does the numactl flag `--membind=0` do?

- A) It binds the process's threads to run only on the cores of NUMA node 0.
- B) It restricts all future memory allocations to NUMA node 0's local DRAM bank only.
- C) It interleaves all memory allocations between NUMA node 0 and all others.
- D) It migrates all existing memory pages to NUMA node 0.

**Answer: B**

- A) Incorrect — `--membind` controls memory placement, not CPU scheduling. To bind the process to cores on node 0, use `--cpunodebind=0` instead. These two flags are independent.
- B) Correct — `--membind=0` applies a "bind" memory policy: all mmap/malloc calls will only succeed on NUMA node 0's memory banks. If node 0 runs out of space the allocation will fail rather than spilling to node 1. This ensures all data is local to socket 0.
- C) Incorrect — Interleaving is the behaviour of `--interleave=all`, not `--membind`. Binding concentrates memory onto one node; interleaving spreads it.
- D) Incorrect — `--membind` is a prospective policy that affects future allocations only. It does not migrate pages that have already been allocated. To move existing pages you would need kernel interfaces like `move_pages()`, which is beyond the scope of numactl.

---

## Q10 — What --cpunodebind Does

> **Week reference:** Week 6

**Mental Model:** `--cpunodebind=0` restricts which CPU sockets (NUMA nodes) the process and its children can run on. Combined with `--membind=0`, it ensures both computation and data stay on the same socket, maximising local bandwidth for workloads that fit on one socket.

What does `numactl --cpunodebind=0` do?

- A) It pins every thread to core 0 of the CPU, creating a single-threaded execution.
- B) It restricts the process and all its forked children to run only on cores belonging to NUMA node 0 (socket 0).
- C) It allocates all memory on NUMA node 0 and ignores core placement.
- D) It interleaves CPU time between NUMA node 0 cores and NUMA node 1 cores.

**Answer: B**

- A) Incorrect — `--cpunodebind=0` binds to all cores of NUMA node 0, not to a single core. If NUMA node 0 has 16 cores, the process can use any of those 16 cores. For single-core pinning use `--physcpubind=0`.
- B) Correct — `--cpunodebind=0` (equivalently spelled `--cpubind=0` in some versions) sets the CPU affinity mask to include only the cores that belong to NUMA node 0. The process and any processes it spawns via fork (including multiprocessing workers) are restricted to those cores.
- C) Incorrect — CPU placement and memory placement are independent policies in numactl. `--cpunodebind` affects only core affinity; to also restrict memory use `--membind=0` together.
- D) Incorrect — numactl sets a static affinity mask; it does not dynamically time-share between nodes. Interleaving CPU time between cores on different sockets is the OS scheduler's normal behaviour and is not what `--cpunodebind` controls.

---

## Q11 — interleave=all Solves Scaling But Not Speed

> **Week reference:** Week 6

**Mental Model:** The lecture slide explicitly warns: interleaving "can solve scaling but not necessarily speed." Because interleaving routes roughly half the accesses through the slower inter-socket link, individual access latency may be higher than pure local access, so a single-threaded or low-parallelism workload can actually slow down with interleaving.

The lecture notes that `numactl --interleave=all` "can solve scaling but not necessarily speed." What does this mean in practice?

- A) With interleaving, single-threaded programs run faster because both DRAM banks are used.
- B) With interleaving, the speedup curve scales better across all cores, but the absolute time for small-core-count runs may be similar or slightly worse than without interleaving.
- C) Interleaving always improves speed and scaling simultaneously because total bandwidth doubles.
- D) Interleaving improves speed only if the dataset fits inside the L3 cache of one socket.

**Answer: B**

- A) Incorrect — A single-threaded program running with interleaving will have roughly half its accesses going to the remote NUMA node, adding latency compared to a fully local execution. Single-threaded speed typically does not improve and may worsen slightly.
- B) Correct — Interleaving equalises access latency across all cores (solving scaling) but introduces remote accesses that would otherwise be local for socket 0 workers. The 1-core and small-core-count runtimes may be the same or marginally slower, but the high-core-count runtimes improve dramatically because socket 1 workers no longer pay the full remote penalty.
- C) Incorrect — Doubling aggregate bandwidth benefits the aggregate workload, not individual access latency. An individual access that happens to go to the remote node is still slower than a local access. Speed for a single-threaded workload is not guaranteed to improve.
- D) Incorrect — L3 cache fits only a fraction of the CelebA dataset (hundreds of MB vs. GB of images). The interleaving benefit is not cache-related; it is about equalising DRAM access latency across sockets.

---

## Q12 — Why multiprocessing.Pool Suffers NUMA Effects

> **Week reference:** Week 6

**Mental Model:** `multiprocessing.Pool` creates worker processes via fork(). The parent process (which loaded the data onto NUMA node 0) allocates shared memory using `mp.RawArray`. Workers access that shared array from whatever core they happen to be scheduled on — including cores on socket 1 — but the array's physical pages remain on NUMA node 0 where they were first touched.

Why does a Python `multiprocessing.Pool` reduction over a shared `RawArray` suffer NUMA performance degradation on a dual-socket node?

- A) `multiprocessing.Pool` uses threads internally, and Python's GIL causes contention.
- B) The shared `RawArray` is allocated in the main process, which first-touches all pages on NUMA node 0; worker processes on socket 1 then access those pages remotely over the inter-socket link.
- C) Python's pickle serialisation copies the array to each worker, causing massive data transfer overhead.
- D) `mp.RawArray` uses GPU memory by default, which is slow to access from CPU cores.

**Answer: B**

- A) Incorrect — `multiprocessing.Pool` uses separate OS processes, not threads. The GIL is irrelevant because each process has its own interpreter lock. NUMA effects are a memory-topology issue, not a lock-contention issue.
- B) Correct — `mp.RawArray` is allocated in the main process, which runs on socket 0 and first-touches all pages. With first-touch policy, all pages land on NUMA node 0. Workers on socket 1 access the same shared memory segment but must traverse the inter-socket link for every cache miss, experiencing the full remote-access penalty.
- C) Incorrect — The whole point of using `mp.RawArray` with an initialiser is to avoid pickle serialisation. The array is shared at the OS level (shared memory); no data is copied to each worker. NUMA effects arise not from copying but from where the existing shared pages physically reside.
- D) Incorrect — `mp.RawArray` is ordinary RAM-backed shared memory (mmap/shm). It does not use GPU memory.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets numactl flag identification, reading hardware output, LSF interaction, single-node binding decisions, and the specific CelebA exercise results.

---

## Q13 — Reading numactl --hardware Output

> **Week reference:** Week 6

**Mental Model:** `numactl --hardware` shows the number of NUMA nodes, which CPUs belong to each node, and the distance matrix showing relative latency from each node to every other. A distance of 10 means local (fast); a higher distance (e.g., 21) means remote (slow).

A student runs `numactl --hardware` on the DTU HPC login node and sees:

```
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 ... 15
node 1 cpus: 16 17 18 ... 31
node distances:
node   0   1
  0:  10  21
  1:  21  10
```

Which statement correctly interprets this output?

- A) Accessing memory on node 1 from a core on node 0 is 21x slower than local access.
- B) Accessing memory on node 1 from a core on node 0 is 2.1x slower than local access (ratio 21/10).
- C) Both nodes have equal access speed because the distance matrix is symmetric.
- D) The machine has 2 nodes with 32 total CPUs, and all memory is equidistant from all cores.

**Answer: B**

- A) Incorrect — The distance values are relative indices, not absolute multipliers. A distance of 21 versus a local distance of 10 means remote access is approximately 21/10 = 2.1x slower, not 21x slower.
- B) Correct — NUMA distance values are relative scores where local is 10 by convention. The ratio 21/10 = 2.1 means remote memory access from node 0 to node 1 (or vice versa) incurs approximately 2.1x the latency of local access. This is the standard interpretation used in the numactl documentation.
- C) Incorrect — Symmetry means the same penalty applies in both directions (node 0 accessing node 1 is the same cost as node 1 accessing node 0), which is expected for UMA-like inter-socket links. It does not mean the access speeds are equal to local; the diagonal (10) vs. off-diagonal (21) shows the asymmetry between local and remote.
- D) Incorrect — The output shows distinctly different distances (10 local, 21 remote), which is the definition of non-uniform access. Equidistant memory from all cores would show all 10s and would describe a UMA system.

---

## Q14 — When Single-Node Binding is Better Than Interleaving

> **Week reference:** Week 6

**Mental Model:** If a workload uses few enough processes to fit entirely within one socket, then keeping all computation and memory on that socket yields the lowest possible latency (all local access). Interleaving in this case makes half the accesses remote without any benefit.

A student runs a computation using 8 processes on a 32-core, dual-socket node (16 cores per socket). They want maximum performance. Should they use `numactl --interleave=all` or `numactl --cpunodebind=0 --membind=0`?

- A) `--interleave=all` — always better because it doubles aggregate bandwidth.
- B) `--cpunodebind=0 --membind=0` — better here because 8 processes fit on one socket, so all accesses can be fully local.
- C) `--interleave=all` — better because it prevents any single socket from becoming a bottleneck.
- D) Neither — numactl never helps for fewer than 16 processes.

**Answer: B**

- A) Incorrect — `--interleave=all` doubles aggregate bandwidth for the whole node, but for 8 processes that fit comfortably on socket 0, interleaving would make half the accesses remote (to node 1) without benefit. Local access is always faster than an average of 50% local and 50% remote.
- B) Correct — With 8 processes and 16 cores available on socket 0, `--cpunodebind=0 --membind=0` confines both the threads and the data to NUMA node 0. Every memory access is local (distance 10), achieving the lowest possible latency. Interleaving would needlessly introduce remote accesses.
- C) Incorrect — A single socket bottleneck is only a concern when the workload's memory bandwidth demand exceeds one socket's DRAM controller capacity. For 8 processes, a single socket's bandwidth is more than sufficient, so no bottleneck exists.
- D) Incorrect — numactl is useful for any process count where NUMA topology matters. Even a 2-process job could benefit from pinning if both processes share a large array and fit on one socket.

---

## Q15 — LSF span[hosts=1] and NUMA

> **Week reference:** Week 6

**Mental Model:** The LSF `-R "span[hosts=1]"` directive ensures that all requested cores are allocated from a single physical compute node (server). This is a prerequisite for using shared memory across processes. Without `span[hosts=1]`, LSF might split processes across multiple servers, where shared memory would not work. Within a single multi-socket node, numactl then handles the NUMA topology.

In an LSF job script for the CelebA reduction, a student adds `-R "span[hosts=1]"`. What does this directive ensure?

- A) All processes run on the same physical NUMA node (socket), preventing cross-socket access.
- B) All requested cores are allocated from a single physical compute node (server) so shared memory between processes works correctly.
- C) numactl automatically applies `--interleave=all` to all processes on that host.
- D) The job uses only one CPU core regardless of how many are requested.

**Answer: B**

- A) Incorrect — `span[hosts=1]` means all cores are on one compute node (one server with one or two sockets), not on one NUMA node (one socket). Cross-socket access is still possible within that single server; you need numactl to control intra-server NUMA behaviour.
- B) Correct — Without `span[hosts=1]`, LSF might allocate, e.g., 20 cores from server A and 12 cores from server B. Shared memory (via `mp.RawArray`) only works within a single OS instance (one server). `span[hosts=1]` ensures all processes are co-located on one server so they can share the same memory segment.
- C) Incorrect — `span[hosts=1]` is a resource selection directive for the scheduler. It does not invoke numactl or apply any memory policy. numactl must be called explicitly in the job script.
- D) Incorrect — `span[hosts=1]` has no effect on the number of cores used. The number of cores is controlled by the `-n` directive (e.g., `-n 32`). `span` only restricts which physical hosts those cores can come from.

---

## Q16 — CelebA Reduction: Where the NUMA Plateau Appears

> **Week reference:** Week 6

**Mental Model:** On the XeonGold 6226R dual-socket node, one socket has 16 physical cores (32 with hyperthreading). The NUMA plateau in the CelebA reduction without numactl is expected at approximately the per-socket physical core count, which is where the processes first start spilling over onto socket 1.

In the CelebA mean-face reduction exercise on the DTU HPC cluster (without numactl), a student observes that speedup peaks then decreases. Approximately at how many processes does this turn-around point occur, and why?

- A) At p = 2, because the reduction algorithm requires synchronisation at every step.
- B) At p ≈ 16, around the number of cores on one socket, because adding processes beyond the first socket requires remote NUMA access for all data.
- C) At p = 32, because Python's multiprocessing pool is capped at 32 workers.
- D) At p = 1, because the shared array must be locked by a single process at a time.

**Answer: B**

- A) Incorrect — Binary tree reduction does synchronise at each round, but this adds only logarithmic overhead. It does not cause a peak-and-decline pattern at p = 2; at p = 2 speedup is still growing.
- B) Correct — The Xeon Gold 6226R has 16 physical cores per socket, giving 32 cores total on the dual-socket DTU node. Once all 16 processes on socket 0 are busy, additional workers land on socket 1 and access the array — which is entirely on NUMA node 0 — remotely. The remote latency overhead exceeds the parallelism benefit, causing speedup to plateau and then decline around p = 16.
- C) Incorrect — Python's `multiprocessing.Pool` has no built-in cap at 32 workers. It can spawn as many workers as you request. The plateau is a hardware NUMA effect, not a software limit.
- D) Incorrect — `mp.RawArray` with the initialiser pattern allows concurrent reads and writes to different elements without locking. Each reduction step accesses disjoint elements (stride-based), so there is no global lock causing a bottleneck at p = 1.

---

## Q17 — numactl --physcpubind vs --cpunodebind

> **Week reference:** Week 6

**Mental Model:** `--physcpubind` pins to specific physical CPU IDs (e.g., cores 0, 1, 2), while `--cpunodebind` pins to all cores of a NUMA node. For HPC use cases, `--cpunodebind` is more natural since you think in terms of sockets, not individual core IDs.

A student wants to restrict a Python process to run only on the 16 cores of socket 0 (cores 0–15). Which numactl invocation achieves this?

- A) `numactl --physcpubind=0 python script.py` — pins to core 0 only.
- B) `numactl --cpunodebind=0 python script.py` — pins to all cores belonging to NUMA node 0.
- C) `numactl --membind=0 python script.py` — restricts memory and automatically restricts cores too.
- D) `numactl --interleave=0 python script.py` — interleaves across only node 0 cores.

**Answer: B**

- A) Incorrect — `--physcpubind=0` pins to a single physical CPU with ID 0, giving only one core. The student wants all 16 cores of socket 0, which requires specifying all core IDs (`--physcpubind=0,1,2,...,15`) or using `--cpunodebind=0`.
- B) Correct — `--cpunodebind=0` sets the CPU affinity to the set of all cores that belong to NUMA node 0. If node 0 contains cores 0–15, all 16 are included automatically. This is the idiomatic way to restrict to one socket without manually listing core IDs.
- C) Incorrect — `--membind` controls memory placement, not CPU affinity. A process restricted to `--membind=0` can still run on any core of either socket; it just allocates memory on node 0 only.
- D) Incorrect — `--interleave` takes a list of node IDs across which to interleave memory, not a CPU core filter. `--interleave=0` would interleave across only node 0, which is nonsensical (only one node to interleave over) and has no effect on CPU placement.

---

## Q18 — Single-Threaded Workload and numactl

> **Week reference:** Week 6

**Mental Model:** A single-threaded workload runs on exactly one core. If that core is on socket 0 and the data is also on socket 0 (default first-touch placement), all memory accesses are local. Applying `--interleave=all` to a single-threaded workload adds remote accesses that did not exist before, potentially making it slower.

A student runs a single-threaded NumPy computation on a NUMA machine without numactl. The job runs on a core on socket 0, and the array was allocated on NUMA node 0. What is the expected effect of re-running with `numactl --interleave=all`?

- A) The workload will run significantly faster because both DRAM controllers are now used.
- B) The workload will run at roughly the same speed or slightly slower, because interleaving routes half the accesses through the slower inter-socket link.
- C) The workload will crash because interleaving is only valid with multiple processes.
- D) The workload will run exactly twice as fast because memory bandwidth doubles.

**Answer: B**

- A) Incorrect — Using both DRAM controllers simultaneously only benefits bandwidth if the workload is memory-bandwidth-limited AND the traffic can actually reach both nodes simultaneously. A single-threaded workload accesses memory sequentially from one core; interleaving does not create true concurrent access to both nodes from one thread.
- B) Correct — Without numactl, the single-threaded workload accesses all data locally from socket 0 (100% local accesses, lowest possible latency). With `--interleave=all`, roughly half the pages are on node 1, so about 50% of accesses become remote (higher latency). This causes similar or slightly worse performance — consistent with the lecture warning that interleaving solves scaling but not necessarily speed.
- C) Incorrect — numactl interleaving is valid for any process count, including 1. It simply sets the kernel memory policy for the process.
- D) Incorrect — For a single-threaded workload, doubling the total system memory bandwidth does not double the thread's speed. Bandwidth doubles in aggregate; a single thread's effective bandwidth is limited by how fast it can issue requests, not by total system capacity.

---

## Q19 — Interleaving Policy Granularity

> **Week reference:** Week 6

**Mental Model:** `numactl --interleave=all` distributes memory at the page granularity (typically 4 KB pages). Consecutive pages alternate between NUMA nodes in round-robin. Because pages are 4 KB, a 128×128×3 float32 image (≈196 KB) spans multiple pages and will be split across nodes.

At what granularity does `numactl --interleave=all` distribute memory across NUMA nodes?

- A) At the byte level — individual bytes alternate between NUMA nodes.
- B) At the page level (typically 4 KB) — consecutive pages alternate between NUMA nodes in round-robin.
- C) At the array level — each distinct NumPy array is placed entirely on one or another node, alternating per allocation.
- D) At the process level — each process gets all its memory on one node, and the nodes are assigned alternately per process.

**Answer: B**

- A) Incorrect — Memory is not interleaved byte-by-byte; that would require a different memory controller architecture. The OS allocates physical memory in pages (minimum 4 KB on most systems), and interleaving operates at the page level.
- B) Correct — The Linux NUMA interleave policy distributes consecutive physical pages in round-robin across the listed NUMA nodes. Each 4 KB page is wholly assigned to one node. Large arrays will have approximately equal numbers of pages on each node.
- C) Incorrect — The interleave policy does not operate at the Python object or NumPy array level. It applies per-page as the virtual address range is backed by physical pages. A single large array can (and does) have its pages distributed across nodes.
- D) Incorrect — numactl `--interleave=all` is a per-process policy set at launch. It affects all memory allocations within that process and its forked children, not at the per-process-alternating level. Each process with this policy gets interleaved memory, not one whole node per process.

---

## Q20 — NUMA Effect on the Speedup Curve Shape

> **Week reference:** Week 6

**Mental Model:** Without numactl, the speedup curve for a memory-bound parallel task on a dual-socket NUMA machine has a characteristic shape: rises from 1 to a peak at the first-socket saturation point, then declines or plateaus. With numactl `--interleave=all`, the curve rises monotonically (though still sub-linearly due to Amdahl's Law and reduction overhead).

Which of the following best describes the shape of the CelebA speedup curve without numactl, compared to the shape with `numactl --interleave=all`?

- A) Without numactl: monotonically increasing. With numactl: flat (no speedup gain).
- B) Without numactl: rises to a peak at the first-socket boundary then declines. With numactl: rises monotonically with all added cores.
- C) Without numactl: linear speedup S = p. With numactl: sub-linear speedup.
- D) Both curves are identical because memory placement does not affect Python programs.

**Answer: B**

- A) Incorrect — Without numactl the curve does increase for the first socket's worth of cores; it is not flat. With numactl the curve continues to increase (not flat) because additional cores from socket 1 now contribute useful work without the full remote-penalty overhead.
- B) Correct — This matches the observed and predicted behaviour. Without numactl: good speedup up to ~the socket size, then degradation as socket 1 workers suffer remote access. With numactl `--interleave=all`: the penalty is equalised, so every added core still helps, giving a monotonically increasing (though sub-linear) speedup curve.
- C) Incorrect — Without numactl, speedup is not linear; it is super-linear initially but peaks and falls. With numactl, speedup is sub-linear (Amdahl's serial fraction and pool.map overhead), not linear. The descriptions are swapped and both individually wrong.
- D) Incorrect — Memory placement directly affects Python programs that use shared memory (`mp.RawArray`). The exercises explicitly demonstrate the NUMA effect on Python multiprocessing workloads.

---

## Q21 — Why the First Socket Sees Fast Access

> **Week reference:** Week 6

**Mental Model:** On a dual-socket NUMA system, memory is physically attached to each socket via a dedicated DRAM controller. The first-touch policy places pages on the node of the touching core. The Python main process (running on socket 0, NUMA node 0) loads the data, so the pages are local to that socket's DRAM controller, giving fast access to all cores on socket 0.

In the CelebA reduction without numactl, why do cores on socket 0 access the shared array faster than cores on socket 1?

- A) The OS kernel preferentially schedules socket 0 cores to run at a higher clock frequency.
- B) The shared array's pages are physically located in socket 0's DRAM (NUMA node 0), making access local (fast) for socket 0 cores and remote (slow) for socket 1 cores.
- C) Socket 0 has a larger L3 cache than socket 1, so its cores experience fewer cache misses.
- D) The inter-socket link is one-directional: socket 1 can read from socket 0 but socket 0 cannot read from socket 1.

**Answer: B**

- A) Incorrect — Both sockets run at the same clock frequency (determined by the CPU's power management, not by NUMA topology). NUMA effects are about memory latency, not clock speed.
- B) Correct — The main process first-touches all pages on NUMA node 0 when it loads the data. Socket 0 cores access those pages through their local DRAM controller (short path, low latency). Socket 1 cores must request those pages via the inter-socket link (QPI/UPI), incurring additional latency for every cache miss.
- C) Incorrect — Both physical CPU sockets are the same model (Xeon Gold 6226R) and have identical L3 cache sizes. Cache size is not the differentiating factor; the bottleneck is DRAM access after cache misses.
- D) Incorrect — The inter-socket interconnect is bidirectional. Both sockets can read from and write to both DRAM banks. The asymmetry is not about direction but about latency: accessing remote DRAM always costs more than accessing local DRAM regardless of direction.

---

## Q22 — Combining numactl Flags

> **Week reference:** Week 6

**Mental Model:** numactl flags are composable. `--cpunodebind=0` restricts cores; `--membind=0` restricts memory allocation. Using both together creates a "fully local" binding: the process runs only on socket 0's cores and allocates memory only from socket 0's DRAM. This is the safest strategy for maximising local bandwidth when the workload fits on one socket.

A student wants to ensure that a multiprocessing reduction job runs entirely within NUMA node 0 (both computation and data), without any remote memory access. Which numactl command achieves this?

- A) `numactl --interleave=all python reduction.py`
- B) `numactl --cpunodebind=0 python reduction.py`
- C) `numactl --cpunodebind=0 --membind=0 python reduction.py`
- D) `numactl --membind=1 python reduction.py`

**Answer: C**

- A) Incorrect — `--interleave=all` explicitly distributes memory across both NUMA nodes. About half the data would reside on node 1, causing remote accesses. This is the opposite of "no remote memory access."
- B) Incorrect — `--cpunodebind=0` restricts cores to socket 0 but does not restrict memory placement. Memory will still be allocated using the default first-touch policy. If the main process runs on socket 0, data will be local, but this is not guaranteed without `--membind=0`. More importantly, the student wants an explicit guarantee.
- C) Correct — `--cpunodebind=0` pins all processes to the cores of NUMA node 0, and `--membind=0` ensures all memory is allocated from NUMA node 0's DRAM. Together, both computation and data are confined to node 0, guaranteeing that every memory access is local with no inter-socket traffic.
- D) Incorrect — `--membind=1` allocates all memory on NUMA node 1 while the cores (unbound by this flag) may run on socket 0. This would make all accesses remote for socket 0 cores — the worst possible configuration, not the best.
