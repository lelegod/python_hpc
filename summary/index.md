# 02613 Python HPC — Summary Index

> **Root:** [STUDY_GUIDE](../STUDY_GUIDE.md) · [Exam Review](../exam_review.md) · [Cheat Sheet](../master_cheat_sheet.md) · [Tips & Pitfalls](../tips_and_tricks.md)

## Contents

- [Weeks](#weeks)
- [Syntax References](#syntax-references)
- [Exam Quick Reference](#exam-quick-reference)
- [Practice Exams (with code proofs)](#practice-exams-with-code-proofs)

---

Each week has two files:
- `notes.md` — theory, concepts, key takeaways
- `syntax.md` — coding syntax reference with exam traps

---

## Weeks

| Week | Topic | Key APIs / Tools |
|------|-------|-----------------|
| [01](week01/) | [Intro & DTU HPC System](week01/notes.md) | BSUB flags, bsub/bjobs/bkill |
| [02](week02/) | [Python Bootcamp & Profiling](week02/notes.md) | perf_counter, cProfile, kernprof |
| [03](week03/) | [Cache Effects & Blosc](week03/notes.md) | strides, loop ordering, blosc |
| [04](week04/) | [High-Performance NumPy](week04/notes.md) | broadcasting, meshgrid, views |
| [05](week05/) | [Parallelism 1: Multiprocessing & GIL](week05/notes.md) | Pool, ThreadPool, Amdahl |
| [06](week06/) | [Parallelism 2: Reductions & Shared Memory](week06/notes.md) | RawArray, SharedMemory, reduction tree |
| [07](week07/) | [High-Performance Pandas & Arrow](week07/notes.md) | dtype downcast, Parquet, PyArrow |
| [08](week08/) | [Storing Big Data](week08/notes.md) | np.memmap, Zarr, chunked CSV |
| [09](week09/) | [Numba & GPU / CUDA](week09/notes.md) | @jit, @cuda.jit, cuda.grid, transfers |
| [10](week10/) | [CuPy & GPU Profiling](week10/notes.md) | cupy, shared.array, syncthreads, nsys |
| [11](week11/) | [HPC Workflows: Arrays & Dependencies](week11/notes.md) | job arrays, $LSB_JOBINDEX, -w done/ended |
| [12](week12/) | [Course Project (Wall Heating)](week12/notes.md) | Jacobi, multiprocessing, CUDA, CuPy |
| [13](week13/) | [HPC Pitfalls](week13/notes.md) | thread env vars, I/O redirect, orphan procs |

---

## Syntax References

| Week | Syntax File | What's inside |
|------|-------------|---------------|
| [01](week01/syntax.md) | BSUB & LSF | All `#BSUB` directives, bsub/bjobs commands, array syntax, dependency flags |
| [02](week02/syntax.md) | Timing & Profiling | `perf_counter`, cProfile columns, kernprof Hits, FLOP/s formula, `time` command |
| [03](week03/syntax.md) | Cache & Blosc | `arr.strides`, loop ordering rule, cache hierarchy, Blosc API |
| [04](week04/syntax.md) | Broadcasting | Right-align rules, `None` indexing, `meshgrid`, views vs copies |
| [05](week05/syntax.md) | Parallelism | `Pool`, `ThreadPool`, GIL table, static vs dynamic, Amdahl formulas |
| [06](week06/syntax.md) | Reductions | Associativity test, `RawArray`, `SharedMemory`, tree/flat complexity |
| [07](week07/syntax.md) | Pandas & Arrow | dtype ranges, `memory_usage(deep=True)`, Parquet read/write, PyArrow CSV |
| [08](week08/syntax.md) | memmap & Zarr | `np.memmap` modes, chunk shape for access patterns, memory budget formula |
| [09](week09/syntax.md) | CUDA | `@cuda.jit`, thread index vars, grid launch, transfer API, `cuda.synchronize` |
| [10](week10/syntax.md) | CuPy & GPU | `cuda.shared.array`, `cuda.syncthreads`, warp divergence, nsys sections |
| [11](week11/syntax.md) | HPC Workflows | Array syntax, `$LSB_JOBINDEX`, dependency flags, bjobs commands |
| [12](week12/syntax.md) | Project Code | Jacobi stencil, CUDA/CuPy versions, parallelisation patterns |
| [13](week13/syntax.md) | Pitfalls | Thread env vars, I/O redirect template, kill background processes |

---

## Exam Quick Reference

**Highest frequency topics (every exam):**

| Topic | Where to look |
|-------|--------------|
| Amdahl's law — all directions | [week05/syntax.md](week05/syntax.md) |
| LSF/BSUB — rusage per core, done vs ended | [week01/syntax.md](week01/syntax.md) |
| Cache — strides, loop ordering | [week03/syntax.md](week03/syntax.md) |
| GPU — coalescing, 1×256 vs 256×1, CHW vs HWC | [week09/syntax.md](week09/syntax.md) |
| GPU transfers — Numba auto-count, nsys bandwidth | [week09/syntax.md](week09/syntax.md) |
| Profiling — cumtime vs tottime, Hits, scaling | [week02/syntax.md](week02/syntax.md) |
| Broadcasting — right-align, None indexing | [week04/syntax.md](week04/syntax.md) |
| GIL — threading vs multiprocessing | [week05/syntax.md](week05/syntax.md) |
| Pandas dtype — ranges, deep=True | [week07/syntax.md](week07/syntax.md) |
| Parallel reduction — associativity test | [week06/syntax.md](week06/syntax.md) |

---

## Practice Exams (with code proofs)

| File | Format | Notes |
|------|--------|-------|
| [exam_F25_proof.ipynb](../exam/exam_F25_proof.ipynb) | MCQ only | **Primary practice** — same format as real exam |
| [exam_2024_proof.ipynb](../exam/exam_2024_proof.ipynb) | Mixed | Open-ended + MCQ |
| [reexam_2024_proof.ipynb](../exam/reexam_2024_proof.ipynb) | Mixed | Open-ended + MCQ |
