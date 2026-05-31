# Amdahl Quiz

*Autolab: Multiple choice questions about exercise 1.1 from weeks 5 and 6*

---

**Exercise context:** A task consists of two subtasks:
- Subtask 1 (serial): retrieve data from file — takes **20 seconds**, cannot be parallelized
- Subtask 2 (parallel): process data records — takes **100 seconds** sequentially, fully parallelizable
- Total sequential time: **120 seconds**

---

## Q1: What is the parallel fraction of the task consisting of the two subtasks?

**Answer: 0.8333** (= 100/120 = 5/6)

> Parallel fraction f = parallelizable time / total time = 100 / 120 ≈ 0.8333

---

## Q2: What is the theoretical speedup for p = 10 processors?

**Answer: 4.0**

> Amdahl's law: S(p) = 1 / ((1 − f) + f/p) = 1 / (1/6 + (5/6)/10) = 1 / (0.1667 + 0.0833) = 1 / 0.25 = 4.0

---

## Q3: What is the theoretical maximum speedup?

**Answer: 6.0**

> Max speedup = 1 / (1 − f) = 1 / (1/6) = 6.0

---

## Q4: Which option will provide the shortest runtime?

- [x] **Option i: optimize the first subtask**
- [ ] Option ii: purchase more processors

> Optimizing subtask 1 (20s → 5s) changes the serial fraction from 1/6 to 5/105, raising max speedup to 21× and giving shorter runtime at any p. Buying more processors is bounded by the fixed 20s serial overhead (max speedup stays 6×).

---

*Submission history: version 1 = 100.0/100*
