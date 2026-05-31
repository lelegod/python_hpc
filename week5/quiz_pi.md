# Pi Quiz

*Autolab: Multiple choice questions about exercise 2.2 from weeks 5 and 6*

---

## Q1: What implementation was the fastest?

- [ ] Implementation 1: Fully serial
- [ ] Implementation 2: Fully parallel
- [x] **Implementation 3: Chunked parallel**

> The chunked parallel approach avoids per-sample process overhead by batching work into chunks, giving better utilization than spawning a process per sample.

---

## Q2: What implementation was the slowest?

- [ ] Implementation 1: Fully serial
- [x] **Implementation 2: Fully parallel**
- [ ] Implementation 3: Chunked parallel

> Fully parallel spawns a separate process for every single sample, so process creation and IPC overhead completely dominates the actual computation. The serial version is faster than this naive approach.

---

## Q3: Looking at the time output for the fastest parallel implementation, which of the following are true as more cores are added? *(select all that apply)*

- [ ] All times go down
- [x] **The real time goes down**
- [ ] The user time goes down
- [ ] The real time remains mostly unchanged or increases
- [x] **The user time remains mostly unchanged or increases**
- [ ] All times remain mostly unchanged or increase

> With multiprocessing: **real (wall-clock) time decreases** as more cores share the work. **User time** (total CPU time across all cores) stays roughly constant or increases slightly due to coordination overhead — it does not drop with more processes.

---

*Submission history: version 3 = 100.0/100*
