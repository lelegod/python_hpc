# GPU Memory Transfers — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Question 1 — Auto-transfer count with implicit transfers](#question-1-auto-transfer-count-with-implicit-transfers)
- [Question 2 — Explicit device arrays reduce transfers](#question-2-explicit-device-arrays-reduce-transfers)
- [Question 3 — Transfer count in a loop, then optimal count](#question-3-transfer-count-in-a-loop-then-optimal-count)
- [Question 4 — Reading nsys profiler output: what dominates?](#question-4-reading-nsys-profiler-output-what-dominates)
- [Question 5 — Computing GPU memory bandwidth from nsys output](#question-5-computing-gpu-memory-bandwidth-from-nsys-output)
- [Question 6 — Missing synchronize before timing](#question-6-missing-synchronize-before-timing)
- [Question 7 — Bytes transferred by device_array_like](#question-7-bytes-transferred-by-device_array_like)
- [Question 8 — Amortising fixed GPU overhead](#question-8-amortising-fixed-gpu-overhead)
- [Question 9 — HtoD count in an image-processing loop](#question-9-htod-count-in-an-image-processing-loop)
- [Question 10 — Interpreting cudaapisum in nsys output](#question-10-interpreting-cudaapisum-in-nsys-output)
- [Key Facts](#key-facts)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Question 11 — Total transfers in a weights-plus-images loop](#question-11-total-transfers-in-a-weights-plus-images-loop)
- [Question 12 — Refactoring to minimise transfers in a weights loop](#question-12-refactoring-to-minimise-transfers-in-a-weights-loop)
- [Question 13 — nsys output: compute the HtoD bandwidth](#question-13-nsys-output-compute-the-htod-bandwidth)
- [Question 14 — Identify the unnecessary transfer](#question-14-identify-the-unnecessary-transfer)
- [Question 15 — DtoH bandwidth from nsys](#question-15-dtoh-bandwidth-from-nsys)
- [Question 16 — Transfer count with CuPy](#question-16-transfer-count-with-cupy)
- [Question 17 — Correcting a missing synchronize in a timed loop](#question-17-correcting-a-missing-synchronize-in-a-timed-loop)
- [Question 18 — Pre-allocating output to avoid repeated DtoH in a loop](#question-18-pre-allocating-output-to-avoid-repeated-dtoh-in-a-loop)
- [Question 19 — Identifying transfer type from CuPy code](#question-19-identifying-transfer-type-from-cupy-code)
- [Question 20 — nsys: which transfer is the bottleneck?](#question-20-nsys-which-transfer-is-the-bottleneck)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Question 21 — Host array modified after to_device](#question-21--host-array-modified-after-to_device)
- [Question 22 — Reading from device_array before kernel runs](#question-22--reading-from-device_array-before-kernel-runs)
- [Question 23 — Break-even iteration count from two benchmarks](#question-23--break-even-iteration-count-from-two-benchmarks)
- [Question 24 — Total data moved with three-kernel auto-transfer pipeline](#question-24--total-data-moved-with-three-kernel-auto-transfer-pipeline)
- [Question 25 — Spot the bug: to_device used instead of copy_to_host](#question-25--spot-the-bug-to_device-used-instead-of-copy_to_host)
- [Question 26 — nsys: compute total transfer fraction including kernel](#question-26--nsys-compute-total-transfer-fraction-including-kernel)
- [Question 27 — Transfer count with back-to-back kernel calls on device arrays](#question-27--transfer-count-with-back-to-back-kernel-calls-on-device-arrays)
- [Question 28 — Bandwidth sanity check: spotting the impossible result](#question-28--bandwidth-sanity-check-spotting-the-impossible-result)
- [Question 29 — HtoD transfers when input is already a DeviceNDArray](#question-29--htod-transfers-when-input-is-already-a-devicendarray)
- [Question 30 — Transfer count with conditional copy_to_host in loop](#question-30--transfer-count-with-conditional-copy_to_host-in-loop)

---

> Format: Each question shows Python CUDA code or nsys profiler output to interpret.
> Exam frequency: **Every exam**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#question-1--auto-transfer-count-with-implicit-transfers)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

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

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets nsys bandwidth calculations, transfer counting in loops, pre-allocation benefits, and HtoD vs DtoH asymmetry

---

## Question 11 — Total transfers in a weights-plus-images loop

How many HtoD and DtoH transfers occur in total when running this code over 30 frames?

```python
from numba import cuda
import numpy as np

@cuda.jit
def blend(out, frame, weights):
    i = cuda.grid(1)
    if i < frame.shape[0]:
        out[i] = frame[i] * weights[i]

weights = np.ones(1024, dtype=np.float32)
frames  = [np.random.rand(1024).astype(np.float32) for _ in range(30)]
output  = np.zeros(1024, dtype=np.float32)

for f in frames:
    blend[4, 256](output, f, weights)
```

- A) 30 HtoD, 30 DtoH
- B) 60 HtoD, 30 DtoH
- C) 90 HtoD, 90 DtoH
- D) 60 HtoD, 60 DtoH

**Answer: C**

Numba auto-transfers all three NumPy arguments (`output`, `f`, `weights`) HtoD before each call and DtoH after each call. Per iteration: 3 HtoD + 3 DtoH = 6 transfers. Over 30 iterations: 6 × 30 = 180 transfers total = 90 HtoD + 90 DtoH.

- A) Incorrect — only counting `f` HtoD and `output` DtoH. Numba transfers all three args in both directions every call.
- B) Incorrect — 60 HtoD counts two arrays per direction (misses `weights`), and 30 DtoH counts only one array DtoH. All three args are transferred both ways.
- C) Correct — 3 arrays × 30 iterations = 90 HtoD, and 3 arrays × 30 iterations = 90 DtoH.
- D) Incorrect — 60 HtoD + 60 DtoH = 2 arrays transferred both ways × 30 iterations; the third array (`weights`) is also transferred each iteration.

---

## Question 12 — Refactoring to minimise transfers in a weights loop

What is the minimum number of total transfers after correctly refactoring the loop from Question 11 using explicit device arrays?

```python
from numba import cuda
import numpy as np

@cuda.jit
def blend(out, frame, weights):
    i = cuda.grid(1)
    if i < frame.shape[0]:
        out[i] = frame[i] * weights[i]

weights = np.ones(1024, dtype=np.float32)
frames  = [np.random.rand(1024).astype(np.float32) for _ in range(30)]

d_weights = cuda.to_device(weights)         # (A)
d_out     = cuda.device_array(1024, dtype=np.float32)  # (B)

results = []
for f in frames:
    d_f = cuda.to_device(f)                 # (C)
    blend[4, 256](d_out, d_f, d_weights)
    results.append(d_out.copy_to_host())    # (D)
```

- A) 180 transfers (same as before)
- B) 62 transfers (1 + 30 + 30 + 1)
- C) 61 transfers (1 HtoD for weights + 30 HtoD for frames + 30 DtoH for outputs)
- D) 32 transfers (1 HtoD for weights + 30 HtoD for frames + 1 DtoH at end)

**Answer: C**

Line A: 1 HtoD (`weights`). Line B: 0 transfers (`device_array` allocates on-device only). Loop (×30): line C = 1 HtoD per frame; line D = 1 DtoH per frame. Total = 1 + 0 + 30 + 30 = 61 transfers.

- A) Incorrect — the refactored code uses explicit device arrays and avoids the auto-transfer round-trips. `d_weights` and `d_out` are passed as device arrays to the kernel, so no implicit transfers occur at kernel launch.
- B) Incorrect — 62 counts 2 setup transfers (lines A and B). Line B (`cuda.device_array`) causes zero transfers; only line A is a real HtoD. Correct setup total is 1, not 2.
- C) Correct — 1 (weights HtoD) + 30 (frame HtoD each iteration) + 30 (output DtoH each iteration) = 61.
- D) Incorrect — 32 would require copying output only once after all 30 iterations. The code calls `copy_to_host()` inside the loop, producing 30 separate DtoH transfers, one per frame result.

---

## Question 13 — nsys output: compute the HtoD bandwidth

Given this nsys profiler excerpt, what is the HtoD memory bandwidth in GB/s?

```
[gpumemsizesum]
Type    Total (MB)
HtoD    256.0
DtoH      0.032

[gpumemtimesum]
Type    Total Time (ms)
HtoD    32.0
DtoH     0.004
```

- A) 0.8 GB/s
- B) 8 GB/s
- C) 80 GB/s
- D) 3.2 GB/s

**Answer: B**

Convert time: 32.0 ms = 0.032 s. Bandwidth = 256 MB / 0.032 s = 8,000 MB/s = 8 GB/s. Consistent with PCIe 3.0 x16 real-world performance.

- A) Incorrect — 0.8 GB/s = 800 MB/s implies 256 MB / 0.32 s; the time is 0.032 s, not 0.32 s. Off by 10×.
- B) Correct — 256 MB / 0.032 s = 8,000 MB/s = 8 GB/s.
- C) Incorrect — 80 GB/s = 80,000 MB/s implies 256 MB / 0.0032 s; exceeds PCIe 3.0 maximum by 5× and uses the wrong time.
- D) Incorrect — 3.2 GB/s = 3,200 MB/s implies 256 MB / 0.08 s; that would be 80 ms, not 32 ms.

---

## Question 14 — Identify the unnecessary transfer

Which line in this code causes an unnecessary memory transfer that should be eliminated?

```python
from numba import cuda
import numpy as np

@cuda.jit
def add_one(arr):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] += 1

data = np.zeros(512, dtype=np.float32)
d_data = cuda.to_device(data)          # Line A
d_scratch = cuda.device_array(512, dtype=np.float32)  # Line B
add_one[2, 256](d_data)
result_host = cuda.to_device(data)     # Line C
final = result_host.copy_to_host()     # Line D
```

- A) Line A — `data` is all zeros; there is nothing meaningful to send HtoD.
- B) Line B — `device_array` wastes memory that is never used.
- C) Line C — re-uploading the original host `data` (which has not been updated) instead of copying the modified device array back.
- D) Line D — the result is already on the device; no DtoH is needed.

**Answer: C**

Line C calls `cuda.to_device(data)` on the original, unmodified host array, producing another HtoD transfer of stale data. The correct call should be `d_data.copy_to_host()` to retrieve the GPU's updated values. Line D then has nothing useful to do since `result_host` is a device array, not the result.

- A) Incorrect — transferring the initial zero-filled array HtoD on Line A is correct and necessary: the kernel reads and writes `d_data`, so the device needs the initial values (even if they are all zero).
- B) Incorrect — `d_scratch` is allocated but unused; that wastes device memory but causes zero PCIe transfers. It is inefficient but not a transfer error.
- C) Correct — `cuda.to_device(data)` on Line C re-uploads the original host array (which the kernel never modified) instead of retrieving the device result. The bug discards the kernel's output.
- D) Incorrect — a DtoH call is essential to get results back to the CPU. The problem is not Line D (which is the right idea) but Line C (which performs the wrong operation before it).

---

## Question 15 — DtoH bandwidth from nsys

A simulation moves results from device to host. nsys reports:

```
[gpumemsizesum]
DtoH    50.0 MB

[gpumemtimesum]
DtoH    0.005 s
```

What is the DtoH bandwidth in GB/s?

- A) 1 GB/s
- B) 10 GB/s
- C) 100 GB/s
- D) 0.1 GB/s

**Answer: B**

Bandwidth = 50 MB / 0.005 s = 10,000 MB/s = 10 GB/s.

- A) Incorrect — 1 GB/s = 1,000 MB/s implies 50 MB / 0.05 s; the time is 0.005 s, not 0.05 s. Off by 10×.
- B) Correct — 50 MB / 0.005 s = 10,000 MB/s = 10 GB/s. Realistic PCIe 3.0 bandwidth.
- C) Incorrect — 100 GB/s = 100,000 MB/s would require 50 MB / 0.0005 s; exceeds PCIe hardware limits by roughly 6×.
- D) Incorrect — 0.1 GB/s = 100 MB/s implies a transfer time of 0.5 s for 50 MB; that is 100× slower than the reported 0.005 s.

---

## Question 16 — Transfer count with CuPy

How many host-to-device transfers occur in this CuPy code?

```python
import cupy as cp
import numpy as np

a = np.random.rand(1000)
b = np.random.rand(1000)

d_a = cp.asarray(a)      # Line 1
d_b = cp.asarray(b)      # Line 2
d_c = d_a + d_b          # Line 3
c   = cp.asnumpy(d_c)    # Line 4
```

- A) 0 HtoD — CuPy keeps everything on the GPU
- B) 1 HtoD — only `a` is transferred
- C) 2 HtoD — `a` and `b` are transferred
- D) 3 HtoD — `a`, `b`, and the result are all transferred

**Answer: C**

`cp.asarray(a)` (Line 1) copies `a` from host to device: 1 HtoD. `cp.asarray(b)` (Line 2) copies `b` from host to device: 1 HtoD. Line 3 adds two device arrays on-device: 0 transfers. `cp.asnumpy(d_c)` (Line 4) copies `d_c` from device to host: 1 DtoH. Total HtoD = 2.

- A) Incorrect — `cp.asarray` explicitly copies numpy arrays from host to device. CuPy does not keep host arrays on the GPU automatically.
- B) Incorrect — both `a` and `b` are numpy arrays passed to `cp.asarray`, each triggering 1 HtoD transfer. Only counting `a` misses `b`.
- C) Correct — Lines 1 and 2 each produce 1 HtoD transfer. Lines 3 and 4 produce 0 HtoD (Line 3 is on-device; Line 4 is DtoH).
- D) Incorrect — Line 3 (`d_a + d_b`) operates entirely on device arrays and produces a new device array `d_c`; no HtoD transfer occurs. The result moves DtoH on Line 4, not HtoD.

---

## Question 17 — Correcting a missing synchronize in a timed loop

What is the bug in this timing code, and what is its effect on the measured time?

```python
from numba import cuda
import numpy as np
from time import perf_counter

@cuda.jit
def scale(out, inp, factor):
    i = cuda.grid(1)
    if i < inp.shape[0]:
        out[i] = inp[i] * factor

d_inp = cuda.to_device(np.random.rand(10_000_000).astype(np.float32))
d_out = cuda.device_array(10_000_000, dtype=np.float32)

t0 = perf_counter()
for _ in range(100):
    scale[40000, 256](d_out, d_inp, 2.0)
t1 = perf_counter()
print(f"Time: {t1 - t0:.4f} s")
```

- A) No bug — `perf_counter` is accurate enough for 100 kernel launches.
- B) `d_inp` should be a NumPy array, not a device array; explicit transfers are not needed.
- C) `cuda.synchronize()` is missing after the loop; `t1` is recorded before the GPU finishes executing the 100 kernels, so the measured time is far too short.
- D) The loop should call `d_out.copy_to_host()` each iteration so the GPU flushes its result.

**Answer: C**

CUDA kernel launches are asynchronous. All 100 `scale` kernel calls are enqueued in the CUDA stream and the Python loop completes in microseconds (just the launch overhead). `t1 = perf_counter()` is captured before any kernel has finished executing. Adding `cuda.synchronize()` after the loop forces the CPU to wait until all 100 kernels complete before recording `t1`.

- A) Incorrect — the issue is not precision. Even a nanosecond-accurate clock would give the wrong result if it captures only launch overhead (microseconds) instead of actual GPU execution (milliseconds).
- B) Incorrect — using `DeviceNDArray` arguments to the kernel is correct and efficient. Passing NumPy arrays would trigger auto-transfers (more overhead), not fix the timing bug.
- C) Correct — without `cuda.synchronize()`, `t1 - t0` measures only the CPU time to enqueue 100 kernels, not the GPU time to execute them. The reported time could be 1000× too small.
- D) Incorrect — `copy_to_host()` inside the loop would force synchronisation implicitly (each copy waits for the preceding kernel), but it also adds 100 DtoH transfers that distort the measurement. The clean fix is one `cuda.synchronize()` after the loop with no copy.

---

## Question 18 — Pre-allocating output to avoid repeated DtoH in a loop

A student rewrites a transfer-heavy loop. How many total transfers occur in the refactored version?

```python
from numba import cuda
import numpy as np

@cuda.jit
def compute(out, inp):
    i = cuda.grid(1)
    if i < inp.shape[0]:
        out[i] = inp[i] ** 2 + inp[i]

N = 1_000_000
inp = np.random.rand(N).astype(np.float32)

# Refactored
d_inp = cuda.to_device(inp)                    # (1)
d_out = cuda.device_array(N, dtype=np.float32) # (2)
for _ in range(200):
    compute[4000, 256](d_out, d_inp)           # (3)
result = d_out.copy_to_host()                  # (4)
```

- A) 400 transfers (200 HtoD + 200 DtoH)
- B) 201 transfers (1 HtoD + 200 DtoH)
- C) 2 transfers (1 HtoD + 1 DtoH)
- D) 802 transfers (400 HtoD + 400 DtoH + overhead)

**Answer: C**

(1) `cuda.to_device(inp)` = 1 HtoD. (2) `cuda.device_array(...)` = 0 transfers. (3) Loop 200×: kernel called with device arrays, no auto-transfers = 0 transfers. (4) `copy_to_host()` = 1 DtoH. Total = 2.

- A) Incorrect — 400 transfers would occur in the naive (auto-transfer) version. The refactored code passes device arrays to the kernel, bypassing Numba's implicit transfer mechanism entirely.
- B) Incorrect — 201 transfers would mean `copy_to_host()` is called inside the loop each iteration. It is called once after the loop (line 4), not 200 times.
- C) Correct — data enters the GPU once before the loop (1 HtoD) and leaves once after (1 DtoH). The 200 kernel invocations each operate on already-resident device arrays with no additional transfers.
- D) Incorrect — there is no "overhead" HtoD or DtoH category in CUDA transfers. `cuda.device_array` causes 0 transfers, and kernel launches with device array arguments cause 0 transfers. The total is simply 2.

---

## Question 19 — Identifying transfer type from CuPy code

Classify each marked line as HtoD, DtoH, or no transfer:

```python
import cupy as cp
import numpy as np

host_arr = np.arange(512, dtype=np.float64)

d_arr  = cp.asarray(host_arr)          # Line A
d_sq   = d_arr ** 2                    # Line B
d_sum  = cp.sum(d_sq)                  # Line C
total  = float(d_sum)                  # Line D
```

- A) A=HtoD, B=no transfer, C=no transfer, D=DtoH
- B) A=HtoD, B=HtoD, C=DtoH, D=no transfer
- C) A=no transfer, B=no transfer, C=DtoH, D=DtoH
- D) A=HtoD, B=DtoH, C=HtoD, D=DtoH

**Answer: A**

Line A: `cp.asarray` copies a numpy array to the GPU device — HtoD. Line B: squaring a CuPy array produces a new CuPy array; entirely on-device — no transfer. Line C: `cp.sum` reduces on-device producing a 0-d CuPy array — no transfer. Line D: `float(d_sum)` copies the scalar value from device to host — DtoH.

- A) Correct — the only PCIe movement is A (host numpy → device CuPy) and D (device scalar → host Python float).
- B) Incorrect — Line B is an on-device element-wise operation; no data crosses the PCIe bus. Line C is also on-device; `cp.sum` stays on the GPU.
- C) Incorrect — Line A is an explicit HtoD transfer (`cp.asarray` on a numpy array). "No transfer" for Line A is wrong.
- D) Incorrect — Line B is on-device arithmetic and Line C is on-device reduction; neither causes a transfer. There is only one HtoD (Line A) and one DtoH (Line D).

---

## Question 20 — nsys: which transfer is the bottleneck?

An nsys report for a matrix multiplication pipeline shows:

```
[gpumemtimesum]
Type    Total Time (ms)   Count   Avg (ms)
HtoD    120.0             6       20.0
DtoH     18.0             6        3.0

[gpukernsum]
Name        Total Time (ms)
matmul_v1   42.0
```

The developer wants to optimise the pipeline. What is the single most impactful action?

- A) Optimise the `matmul_v1` kernel to run faster, since it takes 42 ms.
- B) Reduce the amount of data transferred HtoD, since HtoD transfers at 120 ms are the dominant cost.
- C) Fuse the 6 DtoH transfers into 1 to eliminate DtoH overhead.
- D) Add more CUDA streams to overlap HtoD and kernel execution.

**Answer: B**

Total time = 120 + 18 + 42 = 180 ms. HtoD = 120 ms = 67% of total. It is the single largest component. Reducing the volume of data sent to the GPU (e.g., by pre-loading invariant matrices, using lower precision, or restructuring the algorithm to keep data on-device) would have the greatest impact.

- A) Incorrect — the kernel at 42 ms is the second-largest cost (23% of total), but attacking the 120 ms HtoD bottleneck first yields a larger absolute improvement. Halving HtoD saves 60 ms; halving kernel time saves only 21 ms.
- B) Correct — HtoD is 120 ms out of 180 ms (67%). Any reduction in HtoD volume directly and proportionally reduces total runtime. This is the highest-leverage optimisation target.
- C) Incorrect — DtoH is only 18 ms (10% of total). Even eliminating all DtoH transfers entirely saves only 18 ms. Fusing 6 into 1 might save some launch overhead but is a minor win compared to attacking the 120 ms HtoD.
- D) Incorrect — CUDA streams can overlap transfers and kernels, but overlapping only helps if the kernel and transfer are independent. With 6 separate matrix uploads feeding 6 kernel calls, the data dependencies likely prevent full overlap. More importantly, this is an optimisation to utilise existing bandwidth better, not reduce the total amount of data moved — the HtoD volume remains 120 ms worth.

---

## Set 3 — Extended Practice

> Targets copy-vs-view semantics, uninitialized memory traps, break-even calculations, pipeline transfer counts, and subtle bugs not covered in Sets 1–2.

---

## Question 21 — Host array modified after to_device

> **Week reference:** Week 9

What does `d_x[0]` contain after this code runs?

```python
import numpy as np
from numba import cuda

x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
d_x = cuda.to_device(x)   # Transfer happens here
x[0] = 999.0               # Host array modified AFTER transfer
```

- A) 999.0 — `d_x` reflects the latest value of `x` because it is a device view.
- B) 1.0 — `cuda.to_device` copies the data at call time; the host modification does not propagate.
- C) 0.0 — GPU memory is always reset between operations.
- D) Undefined — modifying the host array after `to_device` corrupts the device buffer.

**Answer: B**

`cuda.to_device(x)` performs a full byte-for-byte copy of `x` to GPU VRAM at the moment it is called. The resulting `DeviceNDArray` is an independent allocation; host and device are completely decoupled after the call. Setting `x[0] = 999.0` writes to host DRAM only; that physical address has no connection to the GPU buffer. `d_x[0]` still holds 1.0 (the value at transfer time).

- A) Incorrect — `DeviceNDArray` is not a shared-memory view. PCIe-connected GPU memory cannot observe host writes without an explicit new transfer.
- B) Correct — the copy is a one-time snapshot at call time. Subsequent host mutations are invisible to the device until a new `cuda.to_device` call.
- C) Incorrect — GPU memory is not zeroed between operations. The value from the original transfer (1.0) remains unchanged.
- D) Incorrect — there is no corruption. Host DRAM and GPU VRAM are separate physical memories; writing to one cannot corrupt the other.

---

## Question 22 — Reading from device_array before kernel runs

> **Week reference:** Week 9

What does this code print?

```python
import numpy as np
from numba import cuda

d_buf = cuda.device_array(4, dtype=np.int32)
host = d_buf.copy_to_host()
print(host[0] == 0)
```

No kernel has been called before `copy_to_host()`.

- A) `True` — `cuda.device_array` zero-initialises the allocated buffer.
- B) `False` — `cuda.device_array` allocates without initialising; `host[0]` contains garbage, which is non-zero.
- C) An exception is raised because `copy_to_host()` cannot be called on an uninitialised buffer.
- D) `True` or `False` unpredictably — the result is undefined because the memory is uninitialised.

**Answer: D**

`cuda.device_array` is equivalent to `np.empty`: it allocates GPU memory without writing any initial values. The bytes in the allocation contain whatever residual data was previously stored there. Copying them back and comparing to zero may yield `True` or `False` — the result is truly unpredictable from run to run and machine to machine. Neither A (guaranteed zero) nor B (guaranteed non-zero) is universally correct; D captures the undefined, implementation-dependent nature of the result.

- A) Incorrect — if `device_array` zeroed memory, it would require a `cudaMemset` call, adding overhead. It deliberately omits initialisation, just like `np.empty`.
- B) Incorrect — while uninitialised memory often contains non-zero garbage in practice, it is not guaranteed. The GPU allocator might recycle a previously zeroed page, making `host[0] == 0` return `True` in some runs.
- C) Incorrect — `copy_to_host()` is a valid operation on any `DeviceNDArray` regardless of how it was allocated. There is no initialisation requirement for calling it; it simply copies whatever bytes are present.
- D) Correct — the behaviour is undefined in the sense that the result is entirely determined by the allocator's internal state, which varies across runs, hardware, and driver versions.

---

## Question 23 — Break-even iteration count from two benchmarks

> **Week reference:** Week 9

The following benchmark measures GPU vs CPU performance for different iteration counts:

```
Iterations | GPU total (s) | CPU total (s)
-----------|---------------|---------------
     10    |     1.10      |     0.50
     20    |     1.60      |     1.00
```

Based on the data above, at what minimum number of iterations does the GPU become faster than the CPU?

- A) 25 iterations
- B) 30 iterations
- C) 24 iterations
- D) The GPU never becomes faster — the per-iteration costs are equal and the GPU always carries additional fixed overhead.

**Answer: D**

Extract per-iteration costs using the two data points:
- GPU per-iteration: (1.60 − 1.10) / (20 − 10) = 0.50 / 10 = **0.05 s/iter**
- GPU fixed overhead: 1.10 − 10 × 0.05 = **0.60 s**
- CPU per-iteration: (1.00 − 0.50) / (20 − 10) = 0.50 / 10 = **0.05 s/iter**

The per-iteration costs are identical (both 0.05 s/iter). The break-even equation is 0.60 + 0.05n = 0.05n → 0.60 = 0, which has no solution. The GPU's fixed overhead can never be amortised because the GPU offers zero per-iteration advantage over the CPU.

- A) Incorrect — at 25 iterations: GPU = 0.60 + 25 × 0.05 = 1.85 s; CPU = 25 × 0.05 = 1.25 s. GPU is still 0.60 s slower.
- B) Incorrect — at 30 iterations: GPU = 0.60 + 30 × 0.05 = 2.10 s; CPU = 30 × 0.05 = 1.50 s. GPU is still 0.60 s slower.
- C) Incorrect — at 24 iterations: GPU = 0.60 + 24 × 0.05 = 1.80 s; CPU = 24 × 0.05 = 1.20 s. GPU is still 0.60 s slower.
- D) Correct — the GPU and CPU have identical per-iteration costs (0.05 s/iter). The GPU carries an additional fixed overhead of 0.60 s that cannot be amortised. This benchmark indicates the GPU implementation offers no computational advantage — the GPU is effectively computing at the same rate as the CPU, with all of the GPU overhead and none of the GPU benefit.

---

## Question 24 — Total data moved with two-kernel auto-transfer pipeline

> **Week reference:** Week 9

How many MB of data cross the PCIe bus in total when this code runs?

```python
from numba import cuda
import numpy as np

@cuda.jit
def stage1(out, inp):
    i = cuda.grid(1)
    if i < inp.shape[0]: out[i] = inp[i] * 2.0

@cuda.jit
def stage2(out, inp):
    i = cuda.grid(1)
    if i < inp.shape[0]: out[i] = inp[i] + 1.0

N = 1_000_000  # each array is 4 MB at float32
a = np.ones(N, dtype=np.float32)
b = np.zeros(N, dtype=np.float32)
c = np.zeros(N, dtype=np.float32)

stage1[4000, 256](b, a)   # kernel call 1
stage2[4000, 256](c, b)   # kernel call 2
```

- A) 8 MB (1 HtoD for `a`, 1 DtoH for `c`)
- B) 16 MB (4 arrays × 4 MB, one direction each)
- C) 32 MB (2 kernels × 2 arrays × 4 MB × 2 directions)
- D) 48 MB (3 unique arrays × 4 MB × 4 directions)

**Answer: C**

Each kernel call passes NumPy arrays, so Numba applies auto-transfer: 2 HtoD + 2 DtoH per call. Each array is 4 MB.

Call 1 `stage1(b, a)`: `b` HtoD (4) + `a` HtoD (4) + `b` DtoH (4) + `a` DtoH (4) = 16 MB.
Call 2 `stage2(c, b)`: `c` HtoD (4) + `b` HtoD (4) + `c` DtoH (4) + `b` DtoH (4) = 16 MB.
Total = 16 + 16 = **32 MB**.

Note: `b` is transferred 4 times in total (uploaded twice, downloaded twice) even though its content from kernel 1 is immediately re-uploaded for kernel 2 — a classic avoidable redundancy eliminated with explicit device arrays.

- A) Incorrect — 8 MB assumes optimal explicit transfers. With NumPy arrays, Numba transfers every argument in both directions for every call, not just the "necessary" ones.
- B) Incorrect — 16 MB counts each array once per direction (4 arrays if you count `b` twice, or 3 unique arrays × one direction). Auto-transfer gives 4 transfers per call over 2 calls = 8 transfers × 4 MB = 32 MB.
- C) Correct — 2 kernels × 2 arrays/kernel × 4 MB × 2 directions = 32 MB.
- D) Incorrect — 48 MB would require 3 arrays × 4 directions, but each kernel call only triggers 2 HtoD + 2 DtoH for the 2 arguments it receives, not 4 directions for 3 arrays. The correct unit of counting is per-call, not per-unique-array.

---

## Question 25 — Spot the bug: to_device used instead of copy_to_host

> **Week reference:** Week 9

What is the bug in this code, and what does `final[0]` print?

```python
from numba import cuda
import numpy as np

@cuda.jit
def add_one(arr):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] += 1

data = np.zeros(512, dtype=np.float32)
d_data = cuda.to_device(data)           # (1)
add_one[2, 256](d_data)                 # (2) kernel modifies d_data in-place
final_device = cuda.to_device(data)     # (3)
final = final_device.copy_to_host()     # (4)
print(final[0])
```

- A) The code is correct; `final[0]` prints `1.0`.
- B) Line (3) re-uploads the original unmodified host array `data` (still all zeros) instead of retrieving the kernel's result; `final[0]` prints `0.0`.
- C) Line (3) raises a `ValueError` because `cuda.to_device` cannot be called twice on the same array.
- D) Line (2) is the bug; `add_one` cannot modify a `DeviceNDArray` in-place; `final[0]` prints `0.0`.

**Answer: B**

After line (2), the kernel has correctly incremented every element of `d_data` on the GPU. The host array `data` still contains all zeros — it was never updated because no `copy_to_host()` was called on `d_data`. Line (3) calls `cuda.to_device(data)`, which copies the host array's current state (all zeros) to a new device buffer. Line (4) copies that all-zeros buffer back to `final`. The kernel's result in `d_data` is silently discarded. `final[0]` prints `0.0`. The fix is to replace lines (3) and (4) with `final = d_data.copy_to_host()`.

- A) Incorrect — the kernel's output lives in `d_data`. Line (3) bypasses `d_data` entirely and re-uploads the unmodified host array, so the incremented values are never retrieved.
- B) Correct — this is the classic "re-upload instead of download" bug. `cuda.to_device(data)` always reads from the current host state, which was never updated by the kernel. The kernel's work is silently lost.
- C) Incorrect — `cuda.to_device` can be called any number of times on any NumPy array. Each call creates a new independent device allocation with no restriction on repeat calls.
- D) Incorrect — in-place modification of a `DeviceNDArray` via a kernel is perfectly valid and correct. Line (2) is not the bug; the bug is line (3) re-uploading zeros instead of downloading the result from `d_data`.

---

## Question 26 — nsys: compute total transfer fraction including kernel

> **Week reference:** Week 9

An nsys report shows:

```
[gpumemtimesum]
Type    Total Time (ms)   Count
HtoD    80.0              8
DtoH    20.0              4

[gpukernsum]
Name            Total Time (ms)
filter_kernel   50.0
```

What percentage of total GPU-related time is consumed by memory transfers alone, and what is the average HtoD transfer time per individual transfer?

- A) Transfers = 67%, average HtoD = 10 ms
- B) Transfers = 80%, average HtoD = 10 ms
- C) Transfers = 67%, average HtoD = 80 ms
- D) Transfers = 50%, average HtoD = 10 ms

**Answer: A**

Total time = HtoD + DtoH + kernel = 80 + 20 + 50 = 150 ms.
Transfer fraction = (80 + 20) / 150 = 100 / 150 ≈ 0.667 = **67%**.
Average HtoD per transfer = 80 ms / 8 transfers = **10 ms**.

- A) Correct — 67% transfer fraction and 10 ms average HtoD per transfer are both arithmetically correct.
- B) Incorrect — 80% would require transfers = 0.80 × 150 = 120 ms, but transfers total only 100 ms. 80% is the HtoD share of transfer time (80/100), not the share of total GPU time.
- C) Incorrect — the 67% fraction is correct, but 80 ms is the *total* HtoD time, not the per-transfer average. Dividing by 8 gives 10 ms per transfer.
- D) Incorrect — 50% would require transfers = 75 ms. The actual transfer total is 100 ms (67%). 50% corresponds to no simple combination of the given values.

---

## Question 27 — Transfer count with back-to-back kernel calls on device arrays

> **Week reference:** Week 9

How many PCIe transfers occur in total when running this code?

```python
from numba import cuda
import numpy as np

@cuda.jit
def scale(out, inp, factor):
    i = cuda.grid(1)
    if i < inp.shape[0]:
        out[i] = inp[i] * factor

N = 500_000
inp = np.random.rand(N).astype(np.float32)

d_inp = cuda.to_device(inp)                        # (A)
d_mid = cuda.device_array(N, dtype=np.float32)     # (B)
d_out = cuda.device_array(N, dtype=np.float32)     # (C)

scale[2000, 256](d_mid, d_inp, 2.0)                # (D)
scale[2000, 256](d_out, d_mid, 3.0)                # (E)

result = d_out.copy_to_host()                      # (F)
```

- A) 6 PCIe transfers (A + B + C + D + E + F)
- B) 2 PCIe transfers (A = 1 HtoD, F = 1 DtoH)
- C) 4 PCIe transfers (A, B, C = HtoD; F = DtoH)
- D) 10 PCIe transfers (2 per kernel call + A + F)

**Answer: B**

Line A: `cuda.to_device(inp)` = 1 HtoD.
Line B: `cuda.device_array(...)` = 0 transfers (allocates on-device only).
Line C: `cuda.device_array(...)` = 0 transfers.
Lines D and E: both kernel calls receive `DeviceNDArray` arguments — Numba does NOT apply auto-transfer when arguments are already device arrays = 0 transfers each.
Line F: `copy_to_host()` = 1 DtoH.
Total = 1 + 0 + 0 + 0 + 0 + 1 = **2 PCIe transfers**.

- A) Incorrect — lines B and C cause zero transfers; lines D and E cause zero transfers. Only lines A and F generate PCIe traffic.
- B) Correct — passing `DeviceNDArray` objects to a Numba kernel fully suppresses the auto-transfer mechanism. Only the initial HtoD and final DtoH cross PCIe. All intermediate computation stays in GPU VRAM.
- C) Incorrect — B and C allocate on-device with zero HtoD cost. `cuda.device_array` is not a transfer; counting it as one confuses allocation with data movement.
- D) Incorrect — 10 transfers would apply if all kernel arguments were NumPy arrays (2 args × 2 directions × 2 calls = 8, plus A and F = 10). Since all kernel arguments are `DeviceNDArray`, auto-transfer is bypassed entirely.

---

## Question 28 — Bandwidth sanity check: spotting the impossible result

> **Week reference:** Week 9

A student computes GPU memory bandwidth from nsys reports and gets:

| Run | Data (MB) | Time (ms) | Computed BW (GB/s) |
|-----|-----------|-----------|---------------------|
|  1  |   512     |   40      |  12.8               |
|  2  |   512     |    4      | 128.0               |
|  3  |   512     |   400     |   1.28              |

The system uses a PCIe 3.0 x16 connection (theoretical peak ~16 GB/s per direction). Which run(s) contain a calculation or measurement error?

- A) Run 1 only — 12.8 GB/s slightly exceeds typical PCIe 3.0 real-world performance.
- B) Run 2 only — 128 GB/s far exceeds the PCIe 3.0 x16 theoretical peak of ~16 GB/s.
- C) Runs 2 and 3 — both are outside the plausible range.
- D) All three runs are plausible given PCIe utilisation variance.

**Answer: B**

Run 1: 512 MB / 0.040 s = 12,800 MB/s = 12.8 GB/s. Within PCIe 3.0 x16 limits (peak ~16 GB/s); plausible (real-world is typically 60–85% of peak).
Run 2: 512 MB / 0.004 s = 128,000 MB/s = 128 GB/s. Exceeds PCIe 3.0 x16 peak by 8×. Physically impossible — almost certainly a unit error (treating ms as µs, confusing MiB with GiB, or misreading the time column).
Run 3: 512 MB / 0.400 s = 1,280 MB/s = 1.28 GB/s. Below typical PCIe bandwidth but not physically impossible — a slow slot, high software overhead, or large transfer startup cost could explain it.

- A) Incorrect — 12.8 GB/s is a valid real-world PCIe 3.0 x16 result. Real-world bandwidth commonly reaches 75–85% of the 16 GB/s theoretical peak. Run 1 is plausible.
- B) Correct — 128 GB/s is 8× above the PCIe 3.0 x16 theoretical ceiling. No PCIe 3.0 hardware can deliver this. Run 2 must contain a calculation or measurement error.
- C) Incorrect — Run 3 at 1.28 GB/s is low but not physically impossible. Hardware issues, partial bus width, or large measurement overhead can reduce achieved bandwidth below typical values.
- D) Incorrect — 128 GB/s cannot be attributed to "variance." The theoretical ceiling is a hard physical limit, not a guideline.

---

## Question 29 — HtoD transfers when input is already a DeviceNDArray

> **Week reference:** Week 9

How many HtoD transfers occur when running this code?

```python
from numba import cuda
import numpy as np

@cuda.jit
def double_it(out, inp):
    i = cuda.grid(1)
    if i < inp.shape[0]:
        out[i] = inp[i] * 2.0

x = np.ones(1_000_000, dtype=np.float32)
d_x = cuda.to_device(x)                                  # line A
d_y = cuda.device_array(1_000_000, dtype=np.float32)     # line B

double_it[4000, 256](d_y, d_x)   # line C — first call
double_it[4000, 256](d_y, d_x)   # line D — second call
```

- A) 2 HtoD — one per kernel call (Numba always transfers inputs before each call).
- B) 4 HtoD — `d_x` and `d_y` are transferred HtoD before each of the two kernel calls.
- C) 1 HtoD — only line A transfers data; lines C and D pass device arrays so no auto-transfer occurs.
- D) 0 HtoD — no NumPy arrays are passed to the kernels, so nothing is transferred.

**Answer: C**

Line A: `cuda.to_device(x)` = 1 HtoD.
Line B: `cuda.device_array(...)` = 0 HtoD (device allocation only).
Lines C and D: both arguments (`d_y`, `d_x`) are `DeviceNDArray` objects. Numba's auto-transfer triggers only for plain NumPy array arguments. Passing device arrays suppresses implicit transfers entirely = 0 HtoD per call.
Total HtoD = **1**.

- A) Incorrect — Numba's auto-transfer fires only when a kernel receives a NumPy array. Once data is on the device as a `DeviceNDArray`, repeated kernel calls with that object incur zero additional HtoD transfers.
- B) Incorrect — `d_y` was allocated via `device_array` (never transferred HtoD). Passing it to a kernel does not trigger an HtoD transfer because it is already a device-resident object.
- C) Correct — line A is the only HtoD transfer in the entire code snippet. The two kernel calls operate entirely in GPU VRAM using device arrays.
- D) Incorrect — line A is an explicit `cuda.to_device(x)` call, which IS an HtoD transfer. The total HtoD count is 1, not 0.

---

## Question 30 — Transfer count with conditional copy_to_host in loop

> **Week reference:** Week 9

How many total DtoH transfers occur when running this code?

```python
from numba import cuda
import numpy as np

@cuda.jit
def increment(arr):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] += 1

N = 100_000
d_arr = cuda.to_device(np.zeros(N, dtype=np.int32))

snapshots = []
for step in range(20):
    increment[400, 256](d_arr)        # in-place kernel
    if step % 5 == 0:                 # steps 0, 5, 10, 15
        snapshots.append(d_arr.copy_to_host())
```

- A) 20 DtoH — one per iteration of the loop.
- B) 4 DtoH — only at steps 0, 5, 10, and 15 (every 5th step).
- C) 1 DtoH — `copy_to_host()` returns the same cached object after the first call.
- D) 0 DtoH — `d_arr` is a device array; `copy_to_host()` is a no-op when called on a device array.

**Answer: B**

The kernel is called 20 times (steps 0–19) in-place on `d_arr`. Because `d_arr` is a `DeviceNDArray`, no automatic DtoH transfer is triggered by the kernel call itself. The `copy_to_host()` call is guarded by `if step % 5 == 0`, which is satisfied when step ∈ {0, 5, 10, 15} — exactly 4 times in range(20). Each call creates a new NumPy snapshot. Total DtoH = **4**.

- A) Incorrect — 20 DtoH would occur if `copy_to_host()` were called unconditionally every iteration. The conditional `step % 5 == 0` reduces call frequency by 5×, from 20 to 4.
- B) Correct — `step % 5 == 0` holds for steps 0, 5, 10, 15 (four values in range(20)). Each hit triggers exactly one `copy_to_host()`. The 16 kernel-only iterations contribute zero DtoH transfers.
- C) Incorrect — each `copy_to_host()` call performs a new independent DtoH transfer and allocates a fresh NumPy array. There is no caching; each call copies the current GPU state at that moment in time.
- D) Incorrect — `copy_to_host()` is precisely the method that triggers a DtoH transfer. It is not a no-op; it copies the `DeviceNDArray`'s contents from GPU VRAM to a new host NumPy array.

---
