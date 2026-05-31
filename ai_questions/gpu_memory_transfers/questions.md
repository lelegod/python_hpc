# GPU Memory Transfers — MCQ Practice

> Topics: Numba auto-transfers, optimal transfer counts, nsys profiling, bandwidth calculation.
> Exam frequency: **Every exam**.

---

## Q1 — Numba Auto-Transfer Rule

> **Week reference:** Week 9

**Mental Model:** Numba is conservative — it cannot read your mind about which arrays are inputs vs outputs, so it transfers everything both ways. The trap is assuming "output array" means "no HtoD transfer" — Numba does not make that distinction.

When a Numba CUDA kernel is called with NumPy arrays as arguments, what transfer behaviour does Numba apply automatically?

- A) Only input arrays are transferred HtoD before the kernel; no DtoH transfers happen automatically.
- B) All NumPy array arguments are transferred HtoD before the kernel and DtoH after the kernel.
- C) No transfers happen automatically; the programmer must always use `cuda.to_device()` explicitly.
- D) All NumPy array arguments are transferred HtoD before the kernel; DtoH only occurs for scalar return values.

**Answer: B**

- A) Incorrect — Numba does not inspect the kernel's read/write semantics. It has no mechanism to distinguish an input array from an output array at runtime, so it sends all arrays HtoD regardless of whether the kernel actually reads them.
- B) Correct — Numba's implicit transfer policy is: for every NumPy array argument, copy HtoD before the kernel call and copy DtoH after the kernel returns. With k array arguments, this produces exactly 2k transfers per kernel call, regardless of which arrays are read-only or write-only.
- C) Incorrect — Numba does perform implicit transfers when plain NumPy arrays (not `DeviceNDArray` objects) are passed to a kernel. The programmer only needs `cuda.to_device()` when they want to avoid these implicit transfers.
- D) Incorrect — Numba transfers all array arguments DtoH after the kernel, not just scalar return values. Numba CUDA kernels do not have scalar return values at all; outputs are written back through array arguments.

---

## Q2 — Transfer Count for Two-Array Kernel

> **Week reference:** Week 9

**Mental Model:** With k NumPy args → k HtoD + k DtoH = 2k transfers. Two arrays = 4 transfers total. This is the baseline waste you eliminate with explicit transfers.

Given the kernel call `square[bpg, tpb](y, x)` where both `y` and `x` are NumPy arrays, how many memory transfers does Numba perform in total (counting each HtoD and each DtoH separately)?

- A) 1 HtoD + 1 DtoH = 2 transfers
- B) 2 HtoD + 1 DtoH = 3 transfers
- C) 2 HtoD + 2 DtoH = 4 transfers
- D) 1 HtoD + 2 DtoH = 3 transfers

**Answer: C**

- A) Incorrect — with two NumPy arguments, Numba applies the rule independently to each array. One HtoD + one DtoH would only be correct for a single-array kernel, not a two-array kernel.
- B) Incorrect — Numba transfers ALL arguments DtoH after the kernel, not just one. There is no mechanism to select which arrays come back; the policy is symmetric and covers every argument.
- C) Correct — Numba applies: HtoD for `y` + HtoD for `x` = 2 HtoD transfers before the kernel, then DtoH for `y` + DtoH for `x` = 2 DtoH transfers after the kernel. Total = 4 transfers. This is true regardless of the fact that `y` is the output and `x` is the input.
- D) Incorrect — both arrays are sent HtoD before the kernel, not just one. Numba does not peek at the kernel source to determine that `x` is read-only and omit its reverse transfer on the way in.

---

## Q3 — Why Output Arrays Are Transferred HtoD

> **Week reference:** Week 9

**Mental Model:** Numba's transfer policy is "safe and symmetric" — it treats every array as both a potential input and a potential output. Eliminating the unnecessary HtoD for a write-only output requires explicit management via `cuda.device_array()`.

Why does Numba transfer an output array (one that is only written, not read, by the kernel) from host to device (HtoD) before the kernel executes?

- A) Numba pre-reads the output array to detect its dtype before compiling the kernel.
- B) Numba needs the initial values of the output array as part of the kernel computation.
- C) Numba does not distinguish between input and output arrays, so it transfers all arguments both ways as a safe default.
- D) The GPU requires all device buffers to be initialised from host memory before use.

**Answer: C**

- A) Incorrect — dtype detection happens at JIT compile time by inspecting the array's Python object attributes (`dtype`, `ndim`, `shape`). No runtime HtoD transfer is needed for this; the array's metadata is already accessible on the host.
- B) Incorrect — a pure output array's existing values are irrelevant to the computation; the kernel only writes to it. If Numba read them, it would be wasting PCIe bandwidth copying data that will be immediately overwritten.
- C) Correct — Numba's runtime has no knowledge of read/write semantics per argument. Its conservative policy is to always transfer HtoD before and DtoH after, ensuring correctness in all cases. The programmer must opt out of unnecessary transfers by using `cuda.device_array()` to pre-allocate output buffers directly on the device.
- D) Incorrect — `cuda.device_array()` proves this wrong: it allocates uninitialised device memory with zero host-to-device transfers. The GPU has no hardware requirement that buffers be initialised from host memory before writing.

---

## Q4 — Optimal Transfers for Input/Output Kernel

> **Week reference:** Week 9

**Mental Model:** Optimal = only move data when it needs to cross the PCIe bus. Input travels one way (HtoD); output travels the other way (DtoH). Pre-allocate the output on-device with `cuda.device_array()` to avoid the wasteful HtoD for a write-only array.

For the kernel `square[bpg, tpb](y, x)` where `x` is a read-only input and `y` is a write-only output, what is the minimum number of memory transfers required if you manage transfers explicitly?

- A) 2 HtoD + 2 DtoH = 4 transfers
- B) 1 HtoD + 1 DtoH = 2 transfers
- C) 0 HtoD + 1 DtoH = 1 transfer
- D) 1 HtoD + 0 DtoH = 1 transfer

**Answer: B**

- A) Incorrect — this is Numba's auto-transfer behaviour, not the optimal. Explicit transfer management exists precisely to eliminate these 2 redundant transfers (HtoD of `y` and DtoH of `x`), cutting the transfer count in half.
- B) Correct — optimal workflow: `d_x = cuda.to_device(x)` (1 HtoD, sends the input), `d_y = cuda.device_array(n)` (0 transfers, allocates output buffer on-device), run kernel, `result = d_y.copy_to_host()` (1 DtoH, retrieves the output). Total: 1 HtoD + 1 DtoH = 2 transfers.
- C) Incorrect — the input array `x` must still be sent to the device; the kernel reads from it and it lives on the host. Skipping the HtoD for `x` would leave `d_x` uninitialised or missing, causing incorrect results or a segfault.
- D) Incorrect — the output array `y` must be retrieved from the device after the kernel writes results into it. Skipping the DtoH means the computation's results never make it back to the host — the program would complete with the original (unmodified) `y` on the host.

---

## Q5 — Explicit Transfer Functions

> **Week reference:** Week 9

**Mental Model:** Two-step API: `cuda.to_device(x)` pushes data onto the GPU and returns a device array handle; `d_x.copy_to_host()` pulls it back. These are the only two valid explicit transfer calls — all other-sounding names are fabricated.

Which pair of Numba CUDA API calls correctly performs an explicit host-to-device transfer followed by an explicit device-to-host transfer?

- A) `cuda.from_host(x)` then `cuda.to_host(d_x)`
- B) `cuda.to_device(x)` then `d_x.copy_to_host()`
- C) `cuda.device_array(x)` then `d_x.copy_to_host()`
- D) `cuda.to_device(x)` then `cuda.from_device(d_x)`

**Answer: B**

- A) Incorrect — neither `cuda.from_host()` nor `cuda.to_host()` are valid Numba CUDA API calls. These names sound plausible but do not exist in the `numba.cuda` namespace; calling them raises `AttributeError`.
- B) Correct — `cuda.to_device(x)` takes a NumPy array `x`, copies it to the GPU, and returns a `DeviceNDArray` handle `d_x`. Later, `d_x.copy_to_host()` copies the device array back to host memory and returns a new NumPy array with the updated values. This is the canonical explicit transfer pattern.
- C) Incorrect — `cuda.device_array(x)` interprets its argument as a shape/size specification and allocates uninitialised device memory. It does NOT copy any data from `x`. If `x` is a NumPy array passed here, you would get a device array whose size matches `x` but whose contents are undefined — no host data is transferred.
- D) Incorrect — `cuda.from_device()` does not exist in Numba's API. The correct function for DtoH transfer is the method `d_x.copy_to_host()` called on the device array object, not a free function.

---

## Q6 — Zero-Transfer Device Allocation

> **Week reference:** Week 9

**Mental Model:** `cuda.device_array(n)` is the GPU equivalent of `np.empty(n)` — it allocates memory on the device but does not initialise it or touch host memory. Zero PCIe bytes transferred. Use it whenever you just need a scratch buffer for kernel output.

What does `d_out = cuda.device_array(n)` do, and how many host-to-device transfers does it cause?

- A) Allocates a device array by copying `n` elements from host memory; causes 1 HtoD transfer.
- B) Allocates an uninitialised array directly on the device with no host-to-device transfer.
- C) Allocates a zeroed device array by transferring a zero-filled host buffer; causes 1 HtoD transfer.
- D) Allocates a device array and mirrors it with a host array; causes 1 HtoD transfer.

**Answer: B**

- A) Incorrect — `cuda.device_array(n)` takes a shape tuple or integer as its argument (not a host array), and allocates raw device memory. No host data exists to copy; the argument `n` is a size specification, not a data source.
- B) Correct — `cuda.device_array()` is analogous to `np.empty()`: it reserves device memory of the requested shape and dtype without initialising the contents or performing any host-device data movement. This makes it the zero-cost way to create an output buffer for a kernel.
- C) Incorrect — no zeroing transfer occurs; the allocated memory contains whatever was previously in that memory region (undefined/garbage values). To get a zeroed device array, you would use `cuda.to_device(np.zeros(n))`, which does cause 1 HtoD transfer.
- D) Incorrect — `cuda.device_array()` creates a device-only allocation with no corresponding host mirror. There is no synchronised pair; the device array is standalone until you explicitly call `.copy_to_host()` to retrieve results.

---

## Q7 — nsys Profiling Tables

> **Week reference:** Week 9

**Mental Model:** nsys has one table per type of GPU activity: `gpukernsum` = time spent inside kernels, `gpumemtimesum` = time spent on data transfers. Don't confuse `cudaapisum` (CPU-side API call overhead) with actual GPU kernel time.

After profiling a Numba CUDA application with `nsys profile`, which two report sections are most useful for separately examining kernel execution time and memory transfer time?

- A) `gpukernsum` for kernel time; `gpumemtimesum` for transfer time.
- B) `gpumemtimesum` for kernel time; `gpukernsum` for transfer time.
- C) `cudaapisum` for kernel time; `gpukernsum` for transfer time.
- D) `gpukernsum` for kernel time; `cudaapisum` for transfer time.

**Answer: A**

- A) Correct — `gpukernsum` (GPU kernel summary) aggregates the actual on-device execution time for each kernel, including min/max/average duration and call count. `gpumemtimesum` (GPU memory time summary) aggregates the duration of all HtoD and DtoH transfers. These two tables together tell you whether your bottleneck is compute or data movement.
- B) Incorrect — the two table names are swapped. `gpumemtimesum` measures transfer duration, not kernel execution time, and `gpukernsum` measures kernel execution time, not transfer time. Reading them backwards gives the wrong bottleneck diagnosis.
- C) Incorrect — `cudaapisum` reports the CPU-side time spent inside CUDA API calls (e.g., `cudaLaunchKernel`, `cudaMemcpy`), which includes launch latency and driver overhead but NOT the actual time the kernel spent executing on the GPU. Using it for kernel time would underreport compute cost.
- D) Incorrect — while `gpukernsum` correctly gives kernel time, `cudaapisum` is the wrong table for transfer time. `cudaapisum` includes API overhead for all CUDA calls, whereas `gpumemtimesum` specifically and accurately measures the device-side duration of memory transfers.

---

## Q8 — Bandwidth Calculation from nsys

> **Week reference:** Week 9

**Mental Model:** Bandwidth = size / time. Keep units consistent: if size is in MB and time is in seconds, the result is MB/s. 1000 MB/s = 1 GB/s. A quick sanity check: PCIe 4.0 x16 peaks at ~32 GB/s; results above that suggest a measurement error.

An nsys profile reports that 500 MB of data was transferred between host and device in 0.05 seconds total transfer time. What is the achieved memory bandwidth?

- A) 25 MB/s
- B) 250 MB/s
- C) 10,000 MB/s (10 GB/s)
- D) 100,000 MB/s (100 GB/s)

**Answer: C**

- A) Incorrect — this would be 500 / 20 = 25, implying 20 seconds of transfer time. The actual time is 0.05 s, which is 400× faster. Getting 25 MB/s likely means dividing in the wrong order (time / size) instead of (size / time).
- B) Incorrect — this would correspond to 500 MB / 2 s = 250 MB/s, i.e., a transfer time of 2 seconds. The reported time is 0.05 s, which is 40× shorter. Perhaps confused by dividing by 2 instead of 0.05.
- C) Correct — bandwidth = total_MB / total_s = 500 / 0.05 = 10,000 MB/s = 10 GB/s. This is a realistic PCIe 3.0 x16 bandwidth (theoretical peak ~16 GB/s), confirming the answer is plausible.
- D) Incorrect — 100,000 MB/s = 100 GB/s would require 500 / 0.005 = 100,000, i.e., a transfer time of 0.005 s (5 ms), not 0.05 s (50 ms). This exceeds realistic PCIe bandwidth by ~3–6×, which is a red flag even without the arithmetic error.

---

## Q9 — Redundant Transfers in a Loop

> **Week reference:** Week 9

**Mental Model:** Auto-transfer in a loop = 4 transfers × N iterations. Optimal = hoist the HtoD before the loop, allocate output on-device, run N kernels, then do 1 DtoH after. Loop overhead drops from O(N) transfers to O(1).

A program calls `kernel[bpg, tpb](y, x)` 50 times in a loop using the same NumPy arrays `x` and `y` each iteration. How many transfers does Numba's auto-transfer cause, and what is the optimal count using explicit transfers?

- A) Auto: 100 transfers; Optimal: 4 transfers
- B) Auto: 200 transfers; Optimal: 2 transfers
- C) Auto: 100 transfers; Optimal: 2 transfers
- D) Auto: 50 transfers; Optimal: 1 transfer

**Answer: B**

- A) Incorrect — per iteration, Numba performs 2 HtoD + 2 DtoH = 4 transfers. Over 50 iterations that is 4 × 50 = 200, not 100. The optimal is also not 4; it is 2 (one HtoD before the loop + one DtoH after).
- B) Correct — auto-transfer: each of the 50 iterations transfers `y` HtoD, `x` HtoD, `y` DtoH, `x` DtoH = 4 transfers per iteration → 50 × 4 = 200 total. Optimal: `d_x = cuda.to_device(x)` (1 HtoD), `d_y = cuda.device_array(n)` (0 transfers), loop 50× calling `kernel[bpg, tpb](d_y, d_x)` (0 transfers each), then `d_y.copy_to_host()` (1 DtoH) = 2 total. A 100× reduction in transfers.
- C) Incorrect — the auto-transfer count is 200 (4 per iteration × 50 iterations), not 100. Getting 100 suggests counting only HtoD or only DtoH, forgetting that each array is transferred in both directions per call.
- D) Incorrect — 50 iterations × 4 transfers = 200, not 50. And optimal requires at least 2 transfers (1 HtoD to get input data to the device + 1 DtoH to get output results back to the host); you cannot do meaningful computation with 1 or 0 transfers for a non-trivial kernel.

---

## Q10 — Identifying the Bottleneck from nsys

> **Week reference:** Week 9

**Mental Model:** The bottleneck is the largest number. Read the three components from nsys, find the max. Do not assume "computation must dominate" — for data-heavy workloads, PCIe transfer is frequently the bottleneck.

An nsys profile of a GPU program reports: HtoD transfers = 26 ms, kernel execution = 14 ms, DtoH transfers = 0.16 ms. Which component dominates the total runtime?

- A) Kernel execution, because it performs the actual computation.
- B) DtoH transfers, because returning results is always the slowest step.
- C) HtoD transfers, because sending data to the GPU takes the most time.
- D) All three contribute equally.

**Answer: C**

- A) Incorrect — kernel execution at 14 ms is real compute work, but 14 ms < 26 ms. The assumption that "computation must dominate" is a common intuition error. This profile is a textbook transfer-bound case where the time to move data dwarfs the time to process it.
- B) Incorrect — DtoH at 0.16 ms is the smallest component, about 160× smaller than HtoD. Outputs are usually small compared to inputs (e.g., a large array in, a scalar or reduced result out), so DtoH is frequently the cheapest step.
- C) Correct — HtoD = 26 ms is the largest single component, accounting for ~63% of the total ~40 ms runtime. The data movement to the GPU is the bottleneck. The fix would be to reduce how much data is sent (e.g., compress input, use on-device storage, or restructure the algorithm to keep data on-device across multiple kernel calls).
- D) Incorrect — the three times (26 ms, 14 ms, 0.16 ms) differ by up to two orders of magnitude (26 ms vs 0.16 ms is a 160× difference). Equal contribution would require all three to be within ~2–3× of each other.

---

## Q11 — cuda.synchronize() Purpose

> **Week reference:** Week 9

**Mental Model:** Kernel launches are fire-and-forget on the host side — the Python line returns immediately while the GPU is still working. `cuda.synchronize()` is the barrier that says "wait until the GPU is actually done." Without it, any timing or result read is premature.

Why must `cuda.synchronize()` be called before reading kernel results or measuring kernel execution time?

- A) It flushes the GPU's L2 cache so results are visible to the host.
- B) Kernel launches are asynchronous; without synchronisation, timing captures only the launch overhead, not the actual execution.
- C) It triggers the automatic DtoH transfer that Numba would otherwise skip.
- D) It is required to release the GPU device lock so other processes can use the GPU.

**Answer: B**

- A) Incorrect — `cuda.synchronize()` is a CPU-GPU synchronisation barrier, not a cache flush. GPU caches are automatically coherent for results accessed after a kernel completes; the issue is not cache visibility but rather whether the kernel has finished executing at all.
- B) Correct — when Python executes `kernel[bpg, tpb](args)`, the CUDA runtime enqueues the kernel in a command stream and the host-side Python call returns immediately (typically in microseconds). The GPU may still be executing for milliseconds after. Without `cuda.synchronize()`, `time.perf_counter()` after the launch measures only the launch overhead (~50–100 µs), not the actual 10–100 ms of GPU computation.
- C) Incorrect — in Numba's auto-transfer mode, the DtoH transfer happens when the kernel call returns on the host, not when `synchronize()` is called. For explicit transfers, `copy_to_host()` handles the DtoH independently. `synchronize()` does not trigger transfers.
- D) Incorrect — CUDA uses no device-level lock that processes must acquire/release. Multiple processes can use the GPU concurrently via the CUDA multi-process service (MPS). `synchronize()` is a per-process, per-stream ordering guarantee, not a mutex release.

---

## Q12 — GPU vs CPU Break-Even Point

> **Week reference:** Week 9

**Mental Model:** Set `GPU_time(N) = CPU_time(N)` and solve for N. The GPU wins only after enough iterations amortise the fixed transfer overhead. Below break-even, the GPU's overhead makes it slower despite its per-iteration advantage.

A GPU implementation has a fixed overhead of 0.6 s (data transfer + context setup) and then takes 0.05 s per iteration. A CPU implementation takes 0.1 s per iteration. What is the minimum number of iterations at which the GPU is at least as fast as the CPU (the break-even point)?

- A) 6 iterations
- B) 12 iterations
- C) 20 iterations
- D) 24 iterations

**Answer: B**

- A) Incorrect — at N=6: GPU = 0.6 + 6×0.05 = 0.6 + 0.3 = 0.9 s; CPU = 6×0.1 = 0.6 s. GPU is still 0.3 s slower. The fixed overhead has not yet been amortised over enough iterations.
- B) Correct — break-even: `0.6 + N×0.05 = N×0.1` → `0.6 = N×(0.1 − 0.05)` → `0.6 = N×0.05` → `N = 12`. Check: GPU = 0.6 + 12×0.05 = 0.6 + 0.6 = 1.2 s; CPU = 12×0.1 = 1.2 s. Equal at exactly 12 iterations.
- C) Incorrect — at N=20 the GPU wins (GPU = 0.6 + 20×0.05 = 1.6 s; CPU = 20×0.1 = 2.0 s; GPU is 0.4 s faster), but 20 is not the break-even point. The question asks for the minimum N where GPU ≥ CPU in speed, which is 12, not 20.
- D) Incorrect — the algebra gives N = 0.6 / (0.1 − 0.05) = 0.6 / 0.05 = 12, not 24. Getting 24 suggests a factor-of-2 error, perhaps dividing by 0.025 instead of 0.05 or halving the overhead incorrectly.

---
