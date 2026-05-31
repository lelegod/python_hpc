# GPU Memory Transfers — MCQ Practice

> Topics: Numba auto-transfers, optimal transfer counts, nsys profiling, bandwidth calculation.
> Exam frequency: **Every exam**.

---

## Q1 — Numba Auto-Transfer Rule

> **Week reference:** Week 9

When a Numba CUDA kernel is called with NumPy arrays as arguments, what transfer behaviour does Numba apply automatically?

- A) Only input arrays are transferred HtoD before the kernel; no DtoH transfers happen automatically.
- B) All NumPy array arguments are transferred HtoD before the kernel and DtoH after the kernel.
- C) No transfers happen automatically; the programmer must always use `cuda.to_device()` explicitly.
- D) All NumPy array arguments are transferred HtoD before the kernel; DtoH only occurs for scalar return values.

**Answer: B**

- A) Incorrect — Numba does not distinguish inputs from outputs; it transfers all arrays both ways.
- B) Correct — Numba has no way to know which arrays are inputs or outputs, so it sends all arguments HtoD before and DtoH after every kernel call.
- C) Incorrect — Numba does perform implicit transfers when plain NumPy arrays are passed to a kernel.
- D) Incorrect — Numba transfers all array arguments DtoH after the kernel, not just scalar return values.

---

## Q2 — Transfer Count for Two-Array Kernel

> **Week reference:** Week 9

Given the kernel call `square[bpg, tpb](y, x)` where both `y` and `x` are NumPy arrays, how many memory transfers does Numba perform in total (counting each HtoD and each DtoH separately)?

- A) 1 HtoD + 1 DtoH = 2 transfers
- B) 2 HtoD + 1 DtoH = 3 transfers
- C) 2 HtoD + 2 DtoH = 4 transfers
- D) 1 HtoD + 2 DtoH = 3 transfers

**Answer: C**

- A) Incorrect — with two NumPy arguments, each is transferred in both directions, not just one each.
- B) Incorrect — Numba transfers all arguments DtoH after the kernel, not just one.
- C) Correct — Numba transfers every NumPy argument HtoD (2) and then DtoH (2), totalling 4 transfers.
- D) Incorrect — both arrays are sent HtoD before the kernel, not just one.

---

## Q3 — Why Output Arrays Are Transferred HtoD

> **Week reference:** Week 9

Why does Numba transfer an output array (one that is only written, not read, by the kernel) from host to device (HtoD) before the kernel executes?

- A) Numba pre-reads the output array to detect its dtype before compiling the kernel.
- B) Numba needs the initial values of the output array as part of the kernel computation.
- C) Numba does not distinguish between input and output arrays, so it transfers all arguments both ways as a safe default.
- D) The GPU requires all device buffers to be initialised from host memory before use.

**Answer: C**

- A) Incorrect — dtype detection happens at compile time from the array object, not via a runtime HtoD transfer.
- B) Incorrect — a pure output array's initial values are irrelevant to the computation.
- C) Correct — Numba's runtime has no knowledge of read/write semantics; it conservatively transfers every argument HtoD and DtoH.
- D) Incorrect — `cuda.device_array()` can allocate uninitialised device buffers with no HtoD transfer.

---

## Q4 — Optimal Transfers for Input/Output Kernel

> **Week reference:** Week 9

For the kernel `square[bpg, tpb](y, x)` where `x` is a read-only input and `y` is a write-only output, what is the minimum number of memory transfers required if you manage transfers explicitly?

- A) 2 HtoD + 2 DtoH = 4 transfers
- B) 1 HtoD + 1 DtoH = 2 transfers
- C) 0 HtoD + 1 DtoH = 1 transfer
- D) 1 HtoD + 0 DtoH = 1 transfer

**Answer: B**

- A) Incorrect — this is what Numba's auto-transfer does; explicit management can eliminate the unnecessary transfers.
- B) Correct — only `x` needs to go HtoD (it is the input), and only `y` needs to come DtoH (it is the output); pre-allocate `y` on device with `cuda.device_array()`.
- C) Incorrect — the input array `x` must still be sent to the device.
- D) Incorrect — the output array `y` must be retrieved from the device after the kernel.

---

## Q5 — Explicit Transfer Functions

> **Week reference:** Week 9

Which pair of Numba CUDA API calls correctly performs an explicit host-to-device transfer followed by an explicit device-to-host transfer?

- A) `cuda.from_host(x)` then `cuda.to_host(d_x)`
- B) `cuda.to_device(x)` then `d_x.copy_to_host()`
- C) `cuda.device_array(x)` then `d_x.copy_to_host()`
- D) `cuda.to_device(x)` then `cuda.from_device(d_x)`

**Answer: B**

- A) Incorrect — `cuda.from_host()` and `cuda.to_host()` are not valid Numba CUDA API calls.
- B) Correct — `cuda.to_device(x)` copies a NumPy array to the device; `d_x.copy_to_host()` copies it back.
- C) Incorrect — `cuda.device_array(x)` allocates an uninitialised device array; it does not copy host data.
- D) Incorrect — `cuda.from_device()` is not a valid Numba CUDA API call.

---

## Q6 — Zero-Transfer Device Allocation

> **Week reference:** Week 9

What does `d_out = cuda.device_array(n)` do, and how many host-to-device transfers does it cause?

- A) Allocates a device array by copying `n` elements from host memory; causes 1 HtoD transfer.
- B) Allocates an uninitialised array directly on the device with no host-to-device transfer.
- C) Allocates a zeroed device array by transferring a zero-filled host buffer; causes 1 HtoD transfer.
- D) Allocates a device array and mirrors it with a host array; causes 1 HtoD transfer.

**Answer: B**

- A) Incorrect — `cuda.device_array(n)` takes a size (or shape), not a host array, so no host data is copied.
- B) Correct — `cuda.device_array()` allocates raw (uninitialised) device memory with no data transfer from the host.
- C) Incorrect — no zero-fill transfer occurs; the contents are undefined.
- D) Incorrect — there is no mirroring; the allocation is entirely on device.

---

## Q7 — nsys Profiling Tables

> **Week reference:** Week 9

After profiling a Numba CUDA application with `nsys profile`, which two report sections are most useful for separately examining kernel execution time and memory transfer time?

- A) `gpukernsum` for kernel time; `gpumemtimesum` for transfer time.
- B) `gpumemtimesum` for kernel time; `gpukernsum` for transfer time.
- C) `cudaapisum` for kernel time; `gpukernsum` for transfer time.
- D) `gpukernsum` for kernel time; `cudaapisum` for transfer time.

**Answer: A**

- A) Correct — `gpukernsum` summarises GPU kernel durations; `gpumemtimesum` summarises memory transfer durations.
- B) Incorrect — the two tables are swapped; `gpumemtimesum` covers transfers, not kernels.
- C) Incorrect — `cudaapisum` covers CUDA API calls on the CPU side, not GPU kernel execution time.
- D) Incorrect — `cudaapisum` includes both API overhead and launch latency, not pure memory transfer time.

---

## Q8 — Bandwidth Calculation from nsys

> **Week reference:** Week 9

An nsys profile reports that 500 MB of data was transferred between host and device in 0.05 seconds total transfer time. What is the achieved memory bandwidth?

- A) 25 MB/s
- B) 250 MB/s
- C) 10,000 MB/s (10 GB/s)
- D) 100,000 MB/s (100 GB/s)

**Answer: C**

- A) Incorrect — 500/0.05 = 10,000, not 25.
- B) Incorrect — this would correspond to 500/2 seconds, not 0.05 seconds.
- C) Correct — bandwidth = total_MB / total_seconds = 500 / 0.05 = 10,000 MB/s = 10 GB/s.
- D) Incorrect — this would require 0.005 seconds, not 0.05 seconds.

---

## Q9 — Redundant Transfers in a Loop

> **Week reference:** Week 9

A program calls `kernel[bpg, tpb](y, x)` 50 times in a loop using the same NumPy arrays `x` and `y` each iteration. How many transfers does Numba's auto-transfer cause, and what is the optimal count using explicit transfers?

- A) Auto: 100 transfers; Optimal: 4 transfers
- B) Auto: 200 transfers; Optimal: 2 transfers
- C) Auto: 100 transfers; Optimal: 2 transfers
- D) Auto: 50 transfers; Optimal: 1 transfer

**Answer: B**

- A) Incorrect — with two arrays per iteration Numba performs 4 transfers per call (2 HtoD + 2 DtoH), totalling 200 for 50 iterations; and optimal is 2 not 4.
- B) Correct — auto-transfer: 50 × (2 HtoD + 2 DtoH) = 200. Optimal: transfer `x` HtoD once before the loop, pre-allocate `y` on device, run 50 kernel calls, then copy `y` DtoH once = 2 total.
- C) Incorrect — the auto-transfer count should be 200 (4 per iteration × 50), not 100.
- D) Incorrect — 50 iterations × 4 transfers = 200, not 50; and optimal requires at least 2 (1 HtoD + 1 DtoH).

---

## Q10 — Identifying the Bottleneck from nsys

> **Week reference:** Week 9

An nsys profile of a GPU program reports: HtoD transfers = 26 ms, kernel execution = 14 ms, DtoH transfers = 0.16 ms. Which component dominates the total runtime?

- A) Kernel execution, because it performs the actual computation.
- B) DtoH transfers, because returning results is always the slowest step.
- C) HtoD transfers, because sending data to the GPU takes the most time.
- D) All three contribute equally.

**Answer: C**

- A) Incorrect — kernel execution at 14 ms is less than HtoD at 26 ms.
- B) Incorrect — DtoH at 0.16 ms is the smallest component by far.
- C) Correct — HtoD at 26 ms is the largest single component, making data transfer to the GPU the bottleneck.
- D) Incorrect — the three times (26 ms, 14 ms, 0.16 ms) differ by up to two orders of magnitude.

---

## Q11 — cuda.synchronize() Purpose

> **Week reference:** Week 9

Why must `cuda.synchronize()` be called before reading kernel results or measuring kernel execution time?

- A) It flushes the GPU's L2 cache so results are visible to the host.
- B) Kernel launches are asynchronous; without synchronisation, timing captures only the launch overhead, not the actual execution.
- C) It triggers the automatic DtoH transfer that Numba would otherwise skip.
- D) It is required to release the GPU device lock so other processes can use the GPU.

**Answer: B**

- A) Incorrect — `cuda.synchronize()` is a synchronisation barrier, not a cache flush operation.
- B) Correct — CUDA kernel launches return immediately on the host; `cuda.synchronize()` blocks until the device has finished, ensuring correct timing and data availability.
- C) Incorrect — DtoH transfers for auto-transfer mode happen when the kernel call returns, not via `synchronize()`.
- D) Incorrect — `cuda.synchronize()` does not acquire or release a device lock; it is purely a host/device synchronisation point.

---

## Q12 — GPU vs CPU Break-Even Point

> **Week reference:** Week 9

A GPU implementation has a fixed overhead of 0.6 s (data transfer + context setup) and then takes 0.05 s per iteration. A CPU implementation takes 0.1 s per iteration. What is the minimum number of iterations at which the GPU is at least as fast as the CPU (the break-even point)?

- A) 6 iterations
- B) 12 iterations
- C) 20 iterations
- D) 24 iterations

**Answer: B**

- A) Incorrect — at 6 iterations: GPU = 0.6 + 6×0.05 = 0.9 s; CPU = 6×0.1 = 0.6 s; GPU is still slower.
- B) Correct — break-even at N: 0.6 + N×0.05 = N×0.1 → 0.6 = N×0.05 → N = 12 iterations.
- C) Incorrect — at 20 iterations: GPU = 0.6 + 20×0.05 = 1.6 s; CPU = 20×0.1 = 2.0 s; GPU wins but 12 is the exact break-even.
- D) Incorrect — the formula gives N = 0.6 / (0.1 − 0.05) = 12, not 24.

---
