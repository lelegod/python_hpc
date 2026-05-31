# GPU Memory Transfers — Code-Based MCQ Practice

> Format: Each question shows Python CUDA code or nsys profiler output to interpret.
> Exam frequency: **Every exam**.

---

## Question 1 — Auto-transfer count with implicit transfers

How many total memory transfers (HtoD + DtoH) does Numba perform when running this kernel?

```python
from numba import cuda
import numpy as np

@cuda.jit
def square(y, x):
    i = cuda.grid(1)
    if i < x.shape[0]:
        y[i] = x[i] ** 2

x = np.ones(1000)
y = np.zeros(1000)
square[4, 256](y, x)
```

- A) 1 HtoD, 1 DtoH
- B) 2 HtoD, 1 DtoH
- C) 2 HtoD, 2 DtoH
- D) 0 HtoD, 1 DtoH

**Answer: C**

- A) Incorrect — only counting one direction of transfers misses that Numba copies all args both ways
- B) Incorrect — DtoH also happens for both args, not just one
- C) Correct — Numba auto-transfers every NumPy arg HtoD before launch and DtoH after, so both `x` and `y` go both ways (4 total)
- D) Incorrect — both arrays are transferred HtoD, not zero

---

## Question 2 — Explicit device arrays reduce transfers

How many total memory transfers occur with this version using explicit device arrays?

```python
from numba import cuda
import numpy as np

@cuda.jit
def square(y, x):
    i = cuda.grid(1)
    if i < x.shape[0]:
        y[i] = x[i] ** 2

x = np.ones(1000)
y = np.zeros(1000)

d_x = cuda.to_device(x)
d_y = cuda.device_array(1000)
square[4, 256](d_y, d_x)
result = d_y.copy_to_host()
```

- A) 4 total (same as implicit)
- B) 3 total (2 HtoD, 1 DtoH)
- C) 2 total (1 HtoD, 1 DtoH)
- D) 1 total (0 HtoD, 1 DtoH)

**Answer: C**

- A) Incorrect — explicit device arrays avoid the automatic round-trip copies
- B) Incorrect — `cuda.device_array(1000)` allocates on-device with zero HtoD cost, so only 1 HtoD total
- C) Correct — `cuda.to_device(x)` gives 1 HtoD, `cuda.device_array` gives 0 HtoD, `copy_to_host()` gives 1 DtoH
- D) Incorrect — `x` must still be transferred HtoD via `cuda.to_device`

---

## Question 3 — Transfer count in a loop, then optimal count

Consider this kernel loop processing 100 images with a fixed weights array:

```python
for i in range(100):
    result = process_kernel[bpg, tpb](images[i], weights)
```

Both `images[i]` and `weights` are NumPy arrays. Numba auto-transfers apply.

**(a)** How many total memory transfers occur as written?

**(b)** If you refactor to use explicit device arrays (transferring `weights` once and each `images[i]` once per iteration), what is the minimum total transfer count?

- A) (a) 400 total; (b) 201 total
- B) (a) 200 total; (b) 101 total
- C) (a) 400 total; (b) 100 total
- D) (a) 300 total; (b) 150 total

**Answer: A**

- A) Correct — auto-transfer gives 4 transfers/iteration × 100 = 400; optimal gives 1 HtoD for weights + 100 HtoD for images + 100 DtoH for outputs = 201
- B) Incorrect — auto-transfer is 4 per iteration (2 HtoD + 2 DtoH), not 2
- C) Incorrect — the optimal count is 201, not 100, because image HtoD and output DtoH still occur each iteration
- D) Incorrect — neither figure matches the correct arithmetic

---

## Question 4 — Reading nsys profiler output: what dominates?

Examine this nsys profiler output from a depth-map rendering workload:

```
[7/8] Executing 'gpumemtimesum' stats report

Type       Total Time (ns)   Count
HtoD       26,800,000        52
DtoH          160,000         4

[8/8] Executing 'gpukernsum' stats report

Name                    Total Time (ns)
render_depthmap         13,800,000
```

Which component takes the most time and approximately what fraction of total GPU-related time does it represent?

- A) The kernel (render_depthmap) — roughly 34%
- B) DtoH transfers — roughly 0.4%
- C) HtoD transfers — roughly 66%
- D) HtoD transfers — roughly 34%

**Answer: C**

- A) Incorrect — the kernel is 13.8 ms out of ~40.76 ms total, which is ~34%, not the dominant component
- B) Incorrect — DtoH is only 0.16 ms, a negligible fraction
- C) Correct — HtoD is 26.8 ms out of ~40.76 ms total ≈ 65.7% ≈ 66%, the clear bottleneck
- D) Incorrect — 34% is the kernel's share, not HtoD's

---

## Question 5 — Computing GPU memory bandwidth from nsys output

A profiler reports the following for a simulation kernel:

```
[gpumemsizesum]
Type    Total (MB)
HtoD    500.0
DtoH      2.5

[gpumemtimesum]
Type    Total Time (s)
HtoD    0.050
DtoH    0.001
```

What is the effective HtoD PCIe memory bandwidth measured in GB/s?

- A) 1 GB/s
- B) 5 GB/s
- C) 10 GB/s
- D) 100 GB/s

**Answer: C**

- A) Incorrect — 1 GB/s would imply 500 MB took 0.5 s, not 0.05 s
- B) Incorrect — 5 GB/s would imply 500 MB took 0.1 s
- C) Correct — Bandwidth = 500 MB / 0.050 s = 10,000 MB/s = 10 GB/s, realistic for PCIe 3.0
- D) Incorrect — 100 GB/s far exceeds PCIe 3.0 ×16 theoretical maximum of ~16 GB/s

---

## Question 6 — Missing synchronize before timing

What is wrong with this GPU timing code?

```python
from numba import cuda
from time import perf_counter

d_x = cuda.to_device(x)
d_out = cuda.device_array_like(x)

t_start = perf_counter()
kernel[bpg, tpb](d_out, d_x)
# No cuda.synchronize()
t_end = perf_counter()

print(f"GPU time: {t_end - t_start:.6f} s")
```

- A) `perf_counter()` is not precise enough for GPU timing
- B) The kernel is never actually executed because there is no synchronize
- C) `t_end - t_start` captures only the CPU-side kernel **launch** time, not GPU **execution** time, making it misleadingly short
- D) `d_out` must be copied to host before timing ends

**Answer: C**

- A) Incorrect — `perf_counter()` is sufficiently precise; the problem is synchronization, not resolution
- B) Incorrect — the kernel is queued and will execute asynchronously; `synchronize` is not needed for execution, only for correct timing
- C) Correct — CUDA launches are asynchronous; without `cuda.synchronize()`, `t_end` is recorded before the GPU finishes, capturing only the microsecond-scale launch overhead
- D) Incorrect — copying to host is unrelated to the timing correctness issue

---

## Question 7 — Bytes transferred by device_array_like

How many bytes are copied to the GPU device when this line executes?

```python
import numpy as np
from numba import cuda

x = np.random.rand(1_000_000)   # 8 MB of float64 data
d_out = cuda.device_array_like(x)
```

- A) 8,000,000 bytes (full array copied HtoD)
- B) 4,000,000 bytes (only the shape/dtype metadata is transferred)
- C) 0 bytes
- D) It depends on whether `x` has been modified since last transfer

**Answer: C**

- A) Incorrect — `device_array_like` allocates but does not copy host data; use `cuda.to_device(x)` to copy
- B) Incorrect — no data at all is transferred, not even metadata bytes
- C) Correct — `cuda.device_array_like(x)` allocates an uninitialized GPU buffer matching `x`'s shape and dtype; zero bytes are transferred from host
- D) Incorrect — modification history is irrelevant; this call never copies data regardless

---

## Question 8 — Amortising fixed GPU overhead

A GPU benchmark is run with varying iteration counts. Observed wall-clock times (including data transfer and kernel execution):

| Iterations | Total Time (s) |
|---|---|
| 5 | 0.85 |
| 10 | 1.10 |

What is the **per-iteration GPU cost** and the **fixed overhead** (e.g., driver init, first-transfer warmup)?

- A) Per-iteration = 0.10 s; overhead = 0.35 s
- B) Per-iteration = 0.05 s; overhead = 0.60 s
- C) Per-iteration = 0.17 s; overhead = 0.00 s
- D) Per-iteration = 0.025 s; overhead = 0.725 s

**Answer: B**

- A) Incorrect — subtracting the two equations gives 0.25 = 5c, so c = 0.05, not 0.10
- B) Correct — solving the two-equation linear system: c = (1.10 − 0.85) / (10 − 5) = 0.05 s; overhead = 0.85 − 5 × 0.05 = 0.60 s
- C) Incorrect — assuming zero overhead means 0.17 s/iter but that fails to fit both data points
- D) Incorrect — 0.025 s/iter implies overhead of 0.725 s, which does not satisfy either data point

---

## Question 9 — HtoD count in an image-processing loop

How many HtoD (host-to-device) memory transfers occur when running this code over 50 images?

```python
from numba import cuda

results = []
for img in images:          # 50 NumPy arrays
    d_img = cuda.to_device(img)
    d_out = cuda.device_array_like(img)
    process[bpg, tpb](d_out, d_img)
    results.append(d_out.copy_to_host())
```

- A) 100 HtoD (50 for `img` + 50 for `d_out` allocation)
- B) 50 HtoD
- C) 150 HtoD (50 `img` + 50 `d_out` + 50 `copy_to_host`)
- D) 0 HtoD (device arrays handle everything)

**Answer: B**

- A) Incorrect — `cuda.device_array_like` allocates on-device with 0 HtoD bytes transferred
- B) Correct — only `cuda.to_device(img)` triggers an HtoD transfer, once per image = 50 total
- C) Incorrect — `copy_to_host()` is a DtoH transfer, not HtoD, and `device_array_like` costs 0 HtoD
- D) Incorrect — `cuda.to_device(img)` is an explicit HtoD transfer each iteration

---

## Question 10 — Interpreting cudaapisum in nsys output

An nsys profile reports:

```
[cudaapisum] CUDA API Statistics

Time (%)  Total Time (ns)  Count  Avg (ns)   Name
    45.2        2,000,000    500    4,000     cuLaunchKernel
    32.1        1,420,000     10  142,000     cuMemcpyHtoDAsync
    22.7        1,005,000     10  100,500     cuMemcpyDtoHAsync
```

What does the `cuLaunchKernel` time in `cudaapisum` represent?

- A) The total GPU execution time for all kernels
- B) The CPU-side API overhead to submit kernel launch commands to the CUDA driver
- C) The time spent waiting for the GPU to finish executing the kernel
- D) The PCIe transfer time for kernel arguments

**Answer: B**

- A) Incorrect — actual GPU execution time appears in `gpukernsum`, not `cudaapisum`
- B) Correct — `cudaapisum` records CPU-side time spent inside each CUDA API call; for `cuLaunchKernel` this is the enqueue overhead (~4 µs average), not GPU runtime
- C) Incorrect — waiting/blocking time would show up differently; asynchronous launches return to the CPU immediately
- D) Incorrect — kernel arguments are passed via registers/constant memory, not PCIe transfers tracked here

---

## Key Facts

- **Numba auto-transfers:** ALL NumPy args → HtoD before kernel; ALL → DtoH after kernel (even read-only args)
- **`cuda.device_array()` / `cuda.device_array_like()`:** zero HtoD cost — allocates on GPU only
- **`cuda.synchronize()`** required before timing GPU work with `perf_counter()`
- **nsys sections:** `gpukernsum` = kernel time, `gpumemtimesum` = transfer time, `cudaapisum` = CPU API overhead
- **Bandwidth formula:** Bandwidth (GB/s) = Data (GB) / Time (s)
- **Optimal loop pattern:** transfer invariant data once outside the loop; use `device_array` for outputs
