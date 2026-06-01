# AI Practice Questions

Practice MCQs for 02613 Python and High-Performance Computing (DTU F26).
Each topic has two files: **conceptual questions** (theory, definitions, predictions) and **code questions** (read the code, predict output or spot the bug).

---

## By Week

### Week 2–3 — Memory, Cache & Data Types

| Topic | Week | Conceptual | Code-based |
|-------|------|------------|------------|
| Cache & Memory Layout | 3 | [questions](cache_memory_layout/questions.md) | [code questions](cache_memory_layout/code_questions.md) |
| Blosc Compression | 3 | [questions](blosc_compression/questions.md) | [code questions](blosc_compression/code_questions.md) |
| Floating-Point Dtypes | 3–4 | [questions](floating_point_dtypes/questions.md) | [code questions](floating_point_dtypes/code_questions.md) |

### Week 4 — NumPy & Profiling

| Topic | Week | Conceptual | Code-based |
|-------|------|------------|------------|
| NumPy Broadcasting | 4 | [questions](numpy_broadcasting/questions.md) | [code questions](numpy_broadcasting/code_questions.md) |
| Profiling | 4 | [questions](profiling/questions.md) | [code questions](profiling/code_questions.md) |

### Week 5–6 — Parallelism & NUMA

| Topic | Week | Conceptual | Code-based |
|-------|------|------------|------------|
| Amdahl's Law | 5–6 | [questions](amdahls_law/questions.md) | [code questions](amdahls_law/code_questions.md) |
| Parallelism Strategy | 5–6 | [questions](parallelism_strategy/questions.md) | [code questions](parallelism_strategy/code_questions.md) |
| Parallel Reduction | 5–6 | [questions](parallel_reduction/questions.md) | [code questions](parallel_reduction/code_questions.md) |
| numactl & NUMA Topology | 6 | [questions](numactl_numa/questions.md) | [code questions](numactl_numa/code_questions.md) |

### Week 7 — High-Performance Pandas & Arrow

| Topic | Week | Conceptual | Code-based |
|-------|------|------------|------------|
| Pandas Dtype & Chunking | 7 | [questions](pandas_dtype_chunking/questions.md) | [code questions](pandas_dtype_chunking/code_questions.md) |
| PyArrow & Apache Arrow | 7 | [questions](pyarrow_arrow/questions.md) | [code questions](pyarrow_arrow/code_questions.md) |

### Week 8 — Memory-Mapped Files

| Topic | Week | Conceptual | Code-based |
|-------|------|------------|------------|
| Memory-Mapped Files & Zarr | 8 | [questions](memory_mapped_zarr/questions.md) | [code questions](memory_mapped_zarr/code_questions.md) |

### Week 9 — Numba & GPU

| Topic | Week | Conceptual | Code-based |
|-------|------|------------|------------|
| Numba JIT | 9 | [questions](numba_jit/questions.md) | [code questions](numba_jit/code_questions.md) |
| GPU / CUDA Kernels | 9 | [questions](gpu_cuda_kernels/questions.md) | [code questions](gpu_cuda_kernels/code_questions.md) |
| GPU Memory Transfers | 9 | [questions](gpu_memory_transfers/questions.md) | [code questions](gpu_memory_transfers/code_questions.md) |

### Week 11 — HPC Workflows

| Topic | Week | Conceptual | Code-based |
|-------|------|------------|------------|
| HPC Workflows (Job Arrays) | 11 | [questions](hpc_workflows/questions.md) | [code questions](hpc_workflows/code_questions.md) |

### Week 13 — HPC Pitfalls

| Topic | Week | Conceptual | Code-based |
|-------|------|------------|------------|
| HPC Pitfalls | 13 | [questions](hpc_pitfalls/questions.md) | [code questions](hpc_pitfalls/code_questions.md) |

### Cross-Cutting — LSF/BSUB

| Topic | Week | Conceptual | Code-based |
|-------|------|------------|------------|
| LSF / BSUB Job Scripts | 1–13 | [questions](lsf_bsub/questions.md) | [code questions](lsf_bsub/code_questions.md) |

---

## By Type

### Conceptual Questions Only
Quick theory, definitions, and prediction questions — no code to read.

- [Amdahl's Law](amdahls_law/questions.md)
- [Blosc Compression](blosc_compression/questions.md)
- [Cache & Memory Layout](cache_memory_layout/questions.md)
- [Floating-Point Dtypes](floating_point_dtypes/questions.md)
- [GPU / CUDA Kernels](gpu_cuda_kernels/questions.md)
- [GPU Memory Transfers](gpu_memory_transfers/questions.md)
- [HPC Pitfalls](hpc_pitfalls/questions.md)
- [HPC Workflows](hpc_workflows/questions.md)
- [LSF / BSUB](lsf_bsub/questions.md)
- [Memory-Mapped Files & Zarr](memory_mapped_zarr/questions.md)
- [Numba JIT](numba_jit/questions.md)
- [numactl & NUMA](numactl_numa/questions.md)
- [NumPy Broadcasting](numpy_broadcasting/questions.md)
- [Pandas Dtype & Chunking](pandas_dtype_chunking/questions.md)
- [Parallel Reduction](parallel_reduction/questions.md)
- [Parallelism Strategy](parallelism_strategy/questions.md)
- [Profiling](profiling/questions.md)
- [PyArrow & Apache Arrow](pyarrow_arrow/questions.md)

### Code Questions Only
Read the snippet, predict the output, spot the bug, or choose the correct fix.

- [Amdahl's Law](amdahls_law/code_questions.md)
- [Blosc Compression](blosc_compression/code_questions.md)
- [Cache & Memory Layout](cache_memory_layout/code_questions.md)
- [Floating-Point Dtypes](floating_point_dtypes/code_questions.md)
- [GPU / CUDA Kernels](gpu_cuda_kernels/code_questions.md)
- [GPU Memory Transfers](gpu_memory_transfers/code_questions.md)
- [HPC Pitfalls](hpc_pitfalls/code_questions.md)
- [HPC Workflows](hpc_workflows/code_questions.md)
- [LSF / BSUB](lsf_bsub/code_questions.md)
- [Memory-Mapped Files & Zarr](memory_mapped_zarr/code_questions.md)
- [Numba JIT](numba_jit/code_questions.md)
- [numactl & NUMA](numactl_numa/code_questions.md)
- [NumPy Broadcasting](numpy_broadcasting/code_questions.md)
- [Pandas Dtype & Chunking](pandas_dtype_chunking/code_questions.md)
- [Parallel Reduction](parallel_reduction/code_questions.md)
- [Parallelism Strategy](parallelism_strategy/code_questions.md)
- [Profiling](profiling/code_questions.md)
- [PyArrow & Apache Arrow](pyarrow_arrow/code_questions.md)

---

## Exam Coverage

| Exam topic | Topics to review |
|------------|-----------------|
| Cache / memory layout | [Cache](cache_memory_layout/questions.md) · [Cache Code](cache_memory_layout/code_questions.md) · [Blosc](blosc_compression/questions.md) |
| Parallelism & speedup | [Amdahl's Law](amdahls_law/questions.md) · [Parallelism Strategy](parallelism_strategy/questions.md) · [Pi/Reduction](parallel_reduction/questions.md) |
| NUMA & numactl | [numactl](numactl_numa/questions.md) · [numactl Code](numactl_numa/code_questions.md) |
| HPC cluster (LSF) | [LSF/BSUB](lsf_bsub/questions.md) · [HPC Workflows](hpc_workflows/questions.md) · [HPC Pitfalls](hpc_pitfalls/questions.md) |
| Pandas / Arrow | [Pandas](pandas_dtype_chunking/questions.md) · [PyArrow](pyarrow_arrow/questions.md) |
| GPU / CUDA | [CUDA Kernels](gpu_cuda_kernels/questions.md) · [GPU Transfers](gpu_memory_transfers/questions.md) · [Numba](numba_jit/questions.md) |
| Broadcasting | [NumPy Broadcasting](numpy_broadcasting/questions.md) · [Broadcasting Code](numpy_broadcasting/code_questions.md) |
