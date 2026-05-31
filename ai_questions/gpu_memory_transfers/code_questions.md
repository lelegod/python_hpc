# GPU Memory Transfers — Code-Based MCQ Practice

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
