# Reduction Quiz

*Autolab: Multiple choice questions about exercise 4.5 from weeks 5 and 6*

---

## Q1: What did you observe in your speedup plot *before* adding `numactl --interleave=all`?

- [ ] Speedup increases with every core added
- [ ] Speedup does not improve until at least 50% of all threads are added
- [x] **Speedup improves until 50% of all threads are added but then decreases**

> Without interleaved memory allocation, all threads on the second CPU socket must fetch memory from socket 0 across the inter-socket link. Once you exceed the cores on socket 0 (~50% of threads), memory bandwidth becomes the bottleneck and speedup degrades.

---

## Q2: What changed *after* adding `numactl --interleave=all`?

- [x] **Speedup now increases with every core added**
- [ ] Speedup still does not improve until at least 50% of all threads are added
- [ ] Speedup still improves until 50% of all threads are added but then decreases

> Interleaved allocation spreads memory pages across both NUMA nodes, so both sockets get local memory access. Speedup now scales properly with every additional core.

---

## Q3: What is the effect of `numactl --interleave=all`?

- [ ] It ensures work is distributed to cores of both CPU sockets
- [x] **It ensures memory allocation is spread to both CPU sockets**
- [ ] It reorders your code so that it runs faster on more CPUs

> `numactl --interleave=all` controls **memory placement** — it round-robins page allocations across all NUMA nodes so both sockets have local memory to read from. It does not affect thread/process scheduling.

---

*Submission history: version 3 = 100.0/100*
