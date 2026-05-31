# Profile Quiz

*Autolab: Multiple choice questions about exercise 2.2 from week 4*

---

## Q1: Which function uses the most cumulative time?

- [x] **distance_matrix**
- [ ] load_points
- [ ] distance_stats

> `distance_matrix` performs the heavy pairwise distance computation across all points — it dominates the runtime.

---

## Q2: What percentage of the total program run time does the slowest function occupy?

- [ ] 0-30 percent
- [ ] 30-60 percent
- [x] **60-100 percent**

> `distance_matrix` accounts for the vast majority (>60%) of total runtime, which is typical of an O(n²) pairwise operation.

---

## Q3: Based on the profiling results, where should we focus our optimization efforts?

- [x] **Only on the slowest function**
- [ ] On both the slowest and next slowest function
- [ ] Equally on all functions

> Amdahl's law: optimize where the time is. If `distance_matrix` takes >60% of runtime, improving it has the biggest impact. Optimizing minor contributors yields negligible overall speedup.

---

*Submission history: version 2 = 100.0/100*
