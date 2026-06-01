# GPU Memory Transfers — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Numba Auto-Transfer Rule](#q1-numba-auto-transfer-rule)
- [Q2 — Transfer Count for Two-Array Kernel](#q2-transfer-count-for-two-array-kernel)
- [Q3 — Why Output Arrays Are Transferred HtoD](#q3-why-output-arrays-are-transferred-htod)
- [Q4 — Optimal Transfers for Input/Output Kernel](#q4-optimal-transfers-for-inputoutput-kernel)
- [Q5 — Explicit Transfer Functions](#q5-explicit-transfer-functions)
- [Q6 — Zero-Transfer Device Allocation](#q6-zero-transfer-device-allocation)
- [Q7 — nsys Profiling Tables](#q7-nsys-profiling-tables)
- [Q8 — Bandwidth Calculation from nsys](#q8-bandwidth-calculation-from-nsys)
- [Q9 — Redundant Transfers in a Loop](#q9-redundant-transfers-in-a-loop)
- [Q10 — Identifying the Bottleneck from nsys](#q10-identifying-the-bottleneck-from-nsys)
- [Q11 — cuda.synchronize() Purpose](#q11-cudasynchronize-purpose)
- [Q12 — GPU vs CPU Break-Even Point](#q12-gpu-vs-cpu-break-even-point)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q13 — Bandwidth from nsys: Small Transfer](#q13-bandwidth-from-nsys-small-transfer)
- [Q14 — Transfer Count: Three Arrays, One Loop](#q14-transfer-count-three-arrays-one-loop)
- [Q15 — Pre-allocation Saves Transfers in a Loop](#q15-pre-allocation-saves-transfers-in-a-loop)
- [Q16 — HtoD vs DtoH Asymmetry](#q16-htod-vs-dtoh-asymmetry)
- [Q17 — Counting Transfers with a Conditional Inside a Loop](#q17-counting-transfers-with-a-conditional-inside-a-loop)
- [Q18 — nsys Bandwidth: Converting Units Correctly](#q18-nsys-bandwidth-converting-units-correctly)
- [Q19 — Why DtoH Can Be the Bottleneck Despite Small Size](#q19-why-dtoh-can-be-the-bottleneck-despite-small-size)
- [Q20 — Optimal Transfer Count for a Reduction Kernel](#q20-optimal-transfer-count-for-a-reduction-kernel)
- [Q21 — Comparing Auto-Transfer vs Explicit in One Call](#q21-comparing-auto-transfer-vs-explicit-in-one-call)
- [Q22 — Reading gpumemtimesum to Find Total Transfer Time](#q22-reading-gpumemtimesum-to-find-total-transfer-time)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q23 — to_device Copies Data, Not a View](#q23--to_device-copies-data-not-a-view)
- [Q24 — device_array Contains Uninitialized Memory](#q24--device_array-contains-uninitialized-memory)
- [Q25 — Pinned (Page-Locked) Memory](#q25--pinned-page-locked-memory)
- [Q26 — Keeping Data Resident Across Multiple Kernels](#q26--keeping-data-resident-across-multiple-kernels)
- [Q27 — Forgetting copy_to_host Before CPU Use](#q27--forgetting-copy_to_host-before-cpu-use)
- [Q28 — PCIe Theoretical Peak Bandwidth](#q28--pcie-theoretical-peak-bandwidth)
- [Q29 — When GPU Overhead Dominates: Small-Kernel Trap](#q29--when-gpu-overhead-dominates-small-kernel-trap)
- [Q30 — Unified Memory Semantics](#q30--unified-memory-semantics)
- [Q31 — Streaming to Overlap Transfers and Computation](#q31--streaming-to-overlap-transfers-and-computation)
- [Q32 — Transfer Cost for a Pipeline of Three Kernels](#q32--transfer-cost-for-a-pipeline-of-three-kernels)

---

> Topics: Numba auto-transfers, optimal transfer counts, nsys profiling, bandwidth calculation.
> Exam frequency: **Every exam**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--numba-auto-transfer-rule)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

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

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets nsys bandwidth calculations, transfer counting in loops, pre-allocation benefits, and HtoD vs DtoH asymmetry

---

## Q13 — Bandwidth from nsys: Small Transfer

> **Week reference:** Week 9

An nsys profile shows that a DtoH transfer moved 200 MB in 0.025 seconds. What is the achieved DtoH bandwidth in GB/s?

- A) 0.8 GB/s
- B) 8 GB/s
- C) 80 GB/s
- D) 5 GB/s

**Answer: B**

Bandwidth = 200 MB / 0.025 s = 8,000 MB/s = 8 GB/s. This is consistent with PCIe 3.0 x16 performance (theoretical peak ~16 GB/s; real-world transfers typically achieve 8–12 GB/s).

- A) Incorrect — 0.8 GB/s = 800 MB/s would imply 200 MB / 0.25 s; the denominator is 0.025 s, not 0.25 s. Off by a factor of 10, likely from a decimal point error.
- B) Correct — 200 / 0.025 = 8,000 MB/s = 8 GB/s.
- C) Incorrect — 80 GB/s = 80,000 MB/s would require 200 MB / 0.0025 s; far exceeds PCIe bandwidth limits and uses the wrong time value.
- D) Incorrect — 5 GB/s = 5,000 MB/s would require 200 MB / 0.04 s; the given time is 0.025 s, not 0.04 s.

---

## Q14 — Transfer Count: Three Arrays, One Loop

> **Week reference:** Week 9

A Numba CUDA kernel is called with three NumPy arrays (A, B, C) as arguments, 20 times in a loop, using auto-transfer each iteration. How many total memory transfers does Numba perform?

- A) 60 transfers
- B) 120 transfers
- C) 40 transfers
- D) 180 transfers

**Answer: B**

Per iteration: 3 arrays × 2 directions (HtoD + DtoH) = 6 transfers. Over 20 iterations: 6 × 20 = 120 transfers.

- A) Incorrect — 60 would be 3 transfers per iteration (only one direction), but Numba transfers every array both HtoD before and DtoH after.
- B) Correct — 3 arrays × 2 directions × 20 iterations = 120 total transfers.
- C) Incorrect — 40 would be 2 transfers per iteration, accounting for only one array both ways. All three arrays are transferred.
- D) Incorrect — 180 would be 9 transfers per iteration. There is no basis for 9; the formula is 2 × (number of args) per call.

---

## Q15 — Pre-allocation Saves Transfers in a Loop

> **Week reference:** Week 9

A kernel processes the same fixed input array 100 times, writing results to an output array. With Numba auto-transfer, 400 transfers occur. After refactoring to use explicit `cuda.to_device()` and `cuda.device_array()` with both arrays hoisted outside the loop and a single `copy_to_host()` after, how many transfers occur?

- A) 4 transfers
- B) 2 transfers
- C) 100 transfers
- D) 1 transfer

**Answer: B**

Hoisting: `d_in = cuda.to_device(input_arr)` (1 HtoD) + `d_out = cuda.device_array(n)` (0 transfers) before the loop; loop runs 100 times with zero transfers; `result = d_out.copy_to_host()` (1 DtoH) after. Total = 2.

- A) Incorrect — 4 would correspond to auto-transfer for a single call (2 HtoD + 2 DtoH). After refactoring over 100 iterations, the count drops to 2, not 4.
- B) Correct — 1 HtoD (send input once) + 1 DtoH (retrieve output once) = 2 total, regardless of how many kernel invocations occur.
- C) Incorrect — 100 would mean one transfer per iteration, as if only one direction was optimised. Both HtoD and DtoH are hoisted outside the loop, so the loop itself contributes zero transfers.
- D) Incorrect — you need at least 1 HtoD to get input data onto the device and 1 DtoH to get the result back. A single transfer implies either the input or the output is skipped, which would produce wrong results.

---

## Q16 — HtoD vs DtoH Asymmetry

> **Week reference:** Week 9

An nsys profile reports: HtoD total = 24 ms, DtoH total = 0.2 ms. A student concludes "DtoH must be faster hardware." What is the more likely explanation?

- A) The GPU's memory controller prioritises DtoH transfers on the PCIe bus.
- B) DtoH transfers use NVLink while HtoD transfers use PCIe.
- C) The amount of data transferred DtoH is much smaller than the amount transferred HtoD, so even at similar bandwidth DtoH finishes faster.
- D) DtoH transfers are compressed automatically by CUDA.

**Answer: C**

In typical GPU workloads, large input arrays are sent HtoD, but only a small result (e.g., a scalar, a reduced array, or a single output frame) is returned DtoH. The time difference reflects data volume, not hardware asymmetry.

- A) Incorrect — PCIe is a bidirectional bus and does not prioritise one direction. Both HtoD and DtoH share the same physical link with similar peak bandwidth in each direction.
- B) Incorrect — NVLink connects GPU-to-GPU or GPU-to-CPU in specific server configurations (e.g., IBM Power9 + V100). Standard desktop/datacenter PCIe systems use PCIe for both directions. Unless the system is explicitly described as NVLink-equipped, this assumption is wrong.
- C) Correct — the most common explanation is that far less data travels DtoH. For example, sending a 500 MB dataset HtoD but retrieving only a 4 KB result DtoH explains a 120,000× size difference, which more than accounts for the time difference.
- D) Incorrect — CUDA does not apply transparent compression to memory transfers over PCIe. Data is copied as-is; no automatic compression layer exists in the standard CUDA memory transfer API.

---

## Q17 — Counting Transfers with a Conditional Inside a Loop

> **Week reference:** Week 9

```python
d_weights = cuda.to_device(weights)
for i in range(50):
    if i == 0:
        d_input = cuda.to_device(data)
    output[i] = run_kernel(d_weights, d_input).copy_to_host()
```

`run_kernel` returns a `DeviceNDArray`. How many HtoD transfers occur in total?

- A) 100 HtoD
- B) 51 HtoD
- C) 2 HtoD
- D) 50 HtoD

**Answer: C**

`cuda.to_device(weights)` = 1 HtoD before the loop. `cuda.to_device(data)` inside `if i == 0` = 1 HtoD on the first iteration only, never again. `run_kernel` receives device arrays, so no implicit transfers. Total = 2 HtoD.

- A) Incorrect — 100 HtoD would imply both `weights` and `data` are transferred every iteration. `d_weights` is created once before the loop, and `d_input` is created only when `i == 0`.
- B) Incorrect — 51 would mean one of the two arrays is transferred each iteration plus one setup transfer. Neither matches the code; both `cuda.to_device` calls are guarded to run at most once each.
- C) Correct — exactly 2 HtoD transfers total: one for `weights` (line 1) and one for `data` (first iteration of loop only).
- D) Incorrect — 50 would imply `d_input` is re-transferred each iteration. The `if i == 0` guard means `cuda.to_device(data)` runs exactly once.

---

## Q18 — nsys Bandwidth: Converting Units Correctly

> **Week reference:** Week 9

An nsys report shows an HtoD transfer of 1,024 MB completed in 102.4 ms. What is the bandwidth in GB/s? (Use 1 GB = 1,000 MB for this calculation.)

- A) 100 GB/s
- B) 1 GB/s
- C) 10 GB/s
- D) 0.1 GB/s

**Answer: C**

Convert time: 102.4 ms = 0.1024 s. Bandwidth = 1,024 MB / 0.1024 s = 10,000 MB/s = 10 GB/s.

- A) Incorrect — 100 GB/s = 100,000 MB/s would require 1,024 MB / 0.01024 s; the time is 0.1024 s, not 0.01024 s. Off by a factor of 10.
- B) Incorrect — 1 GB/s = 1,000 MB/s would require 1,024 MB / 1.024 s; the time is 0.1024 s, not 1.024 s. Off by a factor of 10 in the other direction.
- C) Correct — 1,024 MB / 0.1024 s = 10,000 MB/s = 10 GB/s. Plausible PCIe 3.0 x16 bandwidth.
- D) Incorrect — 0.1 GB/s = 100 MB/s would require a transfer time of over 10 seconds for 1,024 MB. That would be slower than a USB 2.0 connection — not a realistic GPU result.

---

## Q19 — Why DtoH Can Be the Bottleneck Despite Small Size

> **Week reference:** Week 9

A GPU application waits on DtoH transfers before each loop iteration can proceed on the CPU. Which scenario best explains why DtoH is a bottleneck even when DtoH data volume is small?

- A) DtoH transfers have lower PCIe bandwidth than HtoD transfers on all hardware.
- B) The CPU is blocked waiting for the DtoH result to arrive before it can start the next iteration, creating a synchronisation stall that dominates latency.
- C) CUDA always transfers DtoH before HtoD in its internal scheduler, causing head-of-line blocking.
- D) The GPU's DMA engine is shared between DtoH and kernel execution, serialising the two.

**Answer: B**

Even a tiny DtoH transfer (e.g., 1 scalar) forces a CPU-GPU synchronisation point. If the CPU must act on that scalar before launching the next kernel, the entire pipeline stalls waiting for the GPU to finish and the transfer to complete — a latency bottleneck, not a bandwidth bottleneck.

- A) Incorrect — PCIe bandwidth is symmetric in both directions; HtoD and DtoH share the same physical link with similar peak rates. DtoH is not hardware-disadvantaged.
- B) Correct — the bottleneck is the synchronisation dependency: the CPU cannot proceed until the DtoH result arrives. Every iteration incurs a full round-trip latency (GPU kernel + DtoH transfer + CPU logic), and the pipeline cannot overlap iterations.
- C) Incorrect — CUDA's internal scheduler uses streams and does not enforce a DtoH-before-HtoD ordering. Multiple streams can interleave HtoD, DtoH, and kernel execution concurrently.
- D) Incorrect — modern GPUs have separate DMA engines for HtoD and DtoH that can operate concurrently with each other and with kernel execution (given separate CUDA streams). They do not share a single DMA engine with the compute units.

---

## Q20 — Optimal Transfer Count for a Reduction Kernel

> **Week reference:** Week 9

A reduction kernel takes a large input array (1 GB) and returns a single float64 scalar result. Using explicit transfers, what is the minimum data movement across PCIe?

- A) 1 GB HtoD + 1 GB DtoH = 2 GB total
- B) 1 GB HtoD + 8 bytes DtoH ≈ 1 GB total
- C) 0 bytes (reduction happens in-place on device)
- D) 2 GB HtoD + 8 bytes DtoH ≈ 2 GB total

**Answer: B**

The input must travel HtoD (1 GB). The scalar result (float64 = 8 bytes) travels DtoH. Total ≈ 1 GB + 8 bytes. No other transfers are needed with explicit management.

- A) Incorrect — copying 1 GB back DtoH would mean returning the entire input array, not a scalar. A reduction compresses N elements into 1 value; the output is a single number, not an array of the same size as the input.
- B) Correct — 1 GB HtoD for the input array + 8 bytes DtoH for the float64 scalar result. The 8-byte DtoH is negligible in practice but still counts as a transfer.
- C) Incorrect — even an in-place reduction still requires the initial data to arrive on the device (1 GB HtoD) and the final scalar to be retrieved (8 bytes DtoH). "In-place" refers to the on-device computation, not to eliminating PCIe transfers.
- D) Incorrect — 2 GB HtoD would imply the input is sent twice. With explicit transfers and a single kernel call, the input is transferred once. A second copy would only occur if the data was erroneously re-uploaded.

---

## Q21 — Comparing Auto-Transfer vs Explicit in One Call

> **Week reference:** Week 9

A single kernel call `process[bpg, tpb](out, inp)` is made once (not in a loop), where `inp` is a 100 MB read-only NumPy array and `out` is a 100 MB write-only NumPy array. Auto-transfer moves _____ MB; optimal explicit transfers move _____ MB.

- A) Auto: 400 MB; Optimal: 200 MB
- B) Auto: 200 MB; Optimal: 100 MB
- C) Auto: 400 MB; Optimal: 100 MB
- D) Auto: 200 MB; Optimal: 200 MB

**Answer: A**

Auto-transfer: `inp` HtoD (100 MB) + `out` HtoD (100 MB) + `inp` DtoH (100 MB) + `out` DtoH (100 MB) = 400 MB. Optimal: `d_inp = cuda.to_device(inp)` (100 MB HtoD) + `d_out = cuda.device_array(...)` (0 MB) + `d_out.copy_to_host()` (100 MB DtoH) = 200 MB.

- A) Correct — auto-transfer moves both arrays in both directions = 4 × 100 MB = 400 MB. Optimal skips the unnecessary HtoD for `out` and DtoH for `inp`, saving 200 MB = 200 MB total.
- B) Incorrect — auto-transfer is 400 MB (4 transfers), not 200 MB (2 transfers). 200 MB is the optimal count, not the auto count.
- C) Incorrect — optimal is 200 MB, not 100 MB. You still need 100 MB HtoD for the input and 100 MB DtoH for the output. 100 MB would skip one of these essential transfers.
- D) Incorrect — auto-transfer and optimal cannot be equal here. Auto-transfer always performs more transfers than optimal for write-only output arrays (it wastes an HtoD for `out` and a DtoH for `inp`).

---

## Q22 — Reading gpumemtimesum to Find Total Transfer Time

> **Week reference:** Week 9

An nsys `gpumemtimesum` report shows:

```
Type    Total Time (ms)   Count
HtoD    40.0              10
DtoH     5.0               5
```

The `gpukernsum` report shows kernel total = 15 ms. What fraction of total GPU-related time is spent on memory transfers (HtoD + DtoH combined)?

- A) 25%
- B) 75%
- C) 45%
- D) 55%

**Answer: B**

Total time = HtoD + DtoH + kernel = 40 + 5 + 15 = 60 ms. Transfer time = 40 + 5 = 45 ms. Fraction = 45 / 60 = 0.75 = 75%.

- A) Incorrect — 25% would be 15 ms out of 60 ms, which is the kernel's share, not the transfers' share.
- B) Correct — (40 + 5) / (40 + 5 + 15) = 45 / 60 = 75% of GPU-related time is memory transfers.
- C) Incorrect — 45% would require transfer time / total = 0.45, i.e., transfer time = 27 ms, which matches neither (40 + 5 = 45 ms) nor any simple arithmetic on the given values.
- D) Incorrect — 55% would require transfer time = 33 ms. The actual transfer time is 45 ms (40 + 5), giving 75%, not 55%.

---

## Set 3 — Extended Practice

> Targets copy-vs-view semantics, uninitialized memory, pinned memory, multi-kernel pipelines, PCIe limits, unified memory, CUDA streams, and common exam traps not covered in Sets 1–2.

---

## Q23 — to_device Copies Data, Not a View

> **Week reference:** Week 9

**Mental Model:** `cuda.to_device(x)` performs a full copy of `x` from host to device at the moment it is called. The resulting `DeviceNDArray` is independent of the original NumPy array — subsequent changes to the host array are not reflected on the device, and vice versa, until an explicit transfer is made.

After calling `d_x = cuda.to_device(x)`, you modify `x[0] = 999` on the host. What is the value stored in `d_x[0]` on the device?

- A) 999, because `d_x` is a view into `x` and reflects the change immediately.
- B) The original value of `x[0]` before the `cuda.to_device` call, because the transfer is a copy not a view.
- C) Undefined — modifying a host array after `to_device` corrupts the device buffer.
- D) 0, because `cuda.to_device` always zero-initialises the device copy.

**Answer: B**

- A) Incorrect — `cuda.to_device(x)` is a full data copy over PCIe, not a shared-memory view. There is no mechanism for host memory writes to appear on the GPU without an explicit new transfer. If `d_x` were a live view, every host write would silently invalidate GPU-side data, which is not how PCIe-connected devices work.
- B) Correct — at the moment `cuda.to_device(x)` is called, the current values of `x` are copied byte-for-byte to the GPU. The host array `x` and the device array `d_x` are then completely independent. Changing `x[0]` on the host has no effect on `d_x`; to propagate the change you would need to call `cuda.to_device(x)` again or use a targeted device-side write.
- C) Incorrect — there is no corruption. Host and device memory are separate address spaces. Writing to host array `x` after a `to_device` call is perfectly legal and does not affect the device buffer in any way.
- D) Incorrect — `cuda.to_device(x)` copies the actual data from `x`, not zeros. Zero-initialisation would defeat the entire purpose of the transfer. The only function that allocates without copying is `cuda.device_array()`.

---

## Q24 — device_array Contains Uninitialized Memory

> **Week reference:** Week 9

**Mental Model:** `cuda.device_array(n)` is the GPU analogue of `np.empty(n)` — it reserves device memory of the requested shape but does NOT initialise the contents. Reading from it before the kernel writes to it yields garbage values. This is a correctness trap when programmers assume zero-initialisation.

A student writes:
```
d_out = cuda.device_array(1000, dtype=np.float32)
result = d_out.copy_to_host()
print(result[0])
```
No kernel has been called. What does `result[0]` contain?

- A) 0.0, because `cuda.device_array` always zero-initialises the allocated buffer.
- B) An unpredictable (garbage) value, because `cuda.device_array` allocates without initialising.
- C) `nan`, because uninitialised GPU memory always contains IEEE NaN.
- D) The value from the previous kernel that used the same memory region, which is always recoverable.

**Answer: B**

- A) Incorrect — if `cuda.device_array` zero-initialised, it would need to perform a kernel or memset operation on the GPU, adding overhead. By design it does not; it behaves like `np.empty`. Assuming zero initialisation is the exact mistake that leads to subtle, hard-to-debug correctness errors.
- B) Correct — `cuda.device_array(n)` is analogous to `np.empty(n)`: it reserves GPU memory from the allocator's pool without writing any initial values. The bytes in that region contain whatever was previously there — unrelated data from past allocations, OS memory, or truly random bit patterns. Reading before writing is undefined behaviour from a correctness standpoint.
- C) Incorrect — uninitialised GPU memory can contain any bit pattern. NaN is a specific bit pattern (exponent all-ones, non-zero mantissa) and there is no guarantee uninitialised memory will happen to hold it. Some runs may produce NaN by chance; others will not.
- D) Incorrect — while the memory may contain bits from a previous allocation, this "value" is not reliably recoverable or predictable across runs, allocation orders, or GPU drivers. It cannot be treated as meaningful data from a prior kernel.

---

## Q25 — Pinned (Page-Locked) Memory

> **Week reference:** Week 9

**Mental Model:** Standard (pageable) host memory can be moved by the OS virtual memory system. Before a DMA transfer, CUDA must first stage the data into a temporary pinned buffer — adding a CPU copy step. Pinned (page-locked) memory bypasses this staging copy, allowing the DMA engine to read directly from the host allocation and achieving close to the full PCIe theoretical bandwidth.

Why does allocating host memory as pinned (page-locked) improve PCIe transfer speed compared to using standard pageable memory?

- A) Pinned memory is stored in the GPU's L2 cache, so it is faster to read.
- B) Pinned memory prevents the OS from paging that region to disk, allowing the GPU's DMA engine to access it directly without an intermediate staging copy.
- C) Pinned memory uses a wider PCIe lane allocation than pageable memory.
- D) Pinned memory compresses data before sending it over PCIe, reducing transfer time.

**Answer: B**

- A) Incorrect — pinned memory lives in the host's RAM, not in any GPU cache. GPU L2 cache is an on-chip resource used for caching device memory accesses during kernel execution; it has no role in host-to-device transfers.
- B) Correct — standard pageable memory can be swapped out by the OS at any time, so the GPU's DMA engine cannot safely read it directly (the physical address could change). CUDA handles pageable transfers by first copying data to an internal pinned staging buffer, then DMA-ing from that buffer — a two-step process. With pinned memory, the physical address is guaranteed stable, so the DMA engine reads directly from the user's buffer in one step. This eliminates the CPU copy, reduces latency, and achieves bandwidths close to the PCIe theoretical maximum.
- C) Incorrect — PCIe lane allocation is a hardware property of the motherboard slot and is the same for all host memory transfers regardless of pinning status. The number of lanes used is fixed.
- D) Incorrect — CUDA does not transparently compress data during PCIe transfers. Data is sent as raw bytes in both cases. Compression would require decompression on the other side and would only be beneficial for compressible data, neither of which applies here.

---

## Q26 — Keeping Data Resident Across Multiple Kernels

> **Week reference:** Week 9

**Mental Model:** The key motivation for explicit device array management is to keep data on the GPU across multiple kernel calls. If the output of kernel A feeds into kernel B, and both are invoked with the same device arrays, zero PCIe transfers are needed between the two kernel calls — all data movement is on-chip between the GPU's compute units and its own VRAM.

A pipeline calls three successive kernels: `preprocess`, `compute`, and `postprocess`. Each kernel's output is the next kernel's input. What is the minimum number of PCIe transfers required for the entire pipeline?

- A) 6 transfers (2 per kernel: 1 HtoD + 1 DtoH each)
- B) 3 transfers (1 HtoD per kernel, outputs stay on device)
- C) 2 transfers (1 HtoD for the initial input + 1 DtoH for the final output)
- D) 0 transfers (kernels communicate through shared GPU memory)

**Answer: C**

- A) Incorrect — this is the auto-transfer cost if each kernel is called with plain NumPy arrays. With explicit device arrays, the intermediate results never leave the GPU, eliminating the 4 unnecessary inter-kernel transfers.
- B) Incorrect — with explicit transfers, the initial input is uploaded once (1 HtoD). The intermediate data (`preprocess` output → `compute` input, `compute` output → `postprocess` input) travels only through GPU VRAM, never over PCIe. Only 1 HtoD is needed, not 3.
- C) Correct — optimal pattern: upload the raw input data once (1 HtoD), chain all three kernels using device arrays so intermediate results stay in GPU VRAM, then download the final result once (1 DtoH). Total = 2 PCIe transfers regardless of how many kernels are in the chain.
- D) Incorrect — while intermediate results stay in GPU VRAM (not PCIe transfers), the original input data must still arrive from the host (1 HtoD) and the final output must return to the host (1 DtoH). Zero transfers is only possible if all data is already on the device, which is not the case for a fresh computation.

---

## Q27 — Forgetting copy_to_host Before CPU Use

> **Week reference:** Week 9

**Mental Model:** A `DeviceNDArray` returned by `cuda.to_device()` or produced by a kernel lives in GPU VRAM. It is not a NumPy array and cannot be used in CPU-side Python operations (indexing, printing, passing to NumPy functions) without first calling `.copy_to_host()`. The exam trap is code that "works" at runtime but silently operates on the device handle rather than the actual data.

After running a kernel with explicit device arrays, a student writes `print(d_result.mean())`. What actually happens?

- A) Python prints the mean of the kernel's output data, just like on a NumPy array.
- B) Python raises an `AttributeError` because `DeviceNDArray` does not have a `.mean()` method.
- C) Python raises a `CudaDriverError` because `.mean()` triggers an illegal device memory access.
- D) Python silently prints 0.0 because device arrays default to zero.

**Answer: B**

- A) Incorrect — `DeviceNDArray` is not a NumPy array. It does not implement the full NumPy array API. Calling `.mean()` on it does not trigger a DtoH transfer followed by a NumPy mean; the method simply does not exist on the device array object.
- B) Correct — `DeviceNDArray` is a Numba CUDA device array object. It has methods for device-level operations (like `.copy_to_host()`) but it does not implement NumPy ufuncs or reduction methods such as `.mean()`. Calling `.mean()` raises `AttributeError: 'DeviceNDArray' object has no attribute 'mean'`. The correct pattern is `d_result.copy_to_host().mean()`.
- C) Incorrect — no illegal device memory access occurs because `.mean()` is never dispatched to the GPU at all. The error is a pure Python attribute lookup failure, caught before any CUDA driver interaction.
- D) Incorrect — `DeviceNDArray` does not silently return zero for unknown method calls. Python's attribute system raises `AttributeError` immediately on a nonexistent attribute lookup.

---

## Q28 — PCIe Theoretical Peak Bandwidth

> **Week reference:** Week 9

**Mental Model:** PCIe bandwidth is a fixed hardware constraint — a sanity-check ceiling for any nsys bandwidth calculation. PCIe 3.0 x16 peaks at ~16 GB/s per direction; PCIe 4.0 x16 doubles to ~32 GB/s. Real-world achieved bandwidth is typically 60–80% of theoretical peak due to protocol overhead. Any nsys result claiming more than the PCIe generation's theoretical peak signals a calculation error.

An nsys profile of a CUDA program reports an HtoD bandwidth of 64 GB/s on a system using a PCIe 4.0 x16 slot. Which conclusion is most appropriate?

- A) The result is correct — PCIe 4.0 x16 supports up to 64 GB/s per direction.
- B) The result likely contains a measurement or unit-conversion error — PCIe 4.0 x16 peaks at ~32 GB/s per direction.
- C) The result is correct if pinned memory was used, which doubles the PCIe bandwidth.
- D) The result is correct — the GPU used NVLink to bypass the PCIe bottleneck.

**Answer: B**

- A) Incorrect — PCIe 4.0 x16 has a raw bit rate of approximately 32 GT/s × 16 lanes × 2 bits/transfer (NRZ encoding) ÷ 8 bits/byte, giving ~64 GB/s total bidirectional bandwidth. However, that 64 GB/s is *bidirectional* (32 GB/s HtoD + 32 GB/s DtoH simultaneously). The per-direction peak is ~32 GB/s. A single HtoD measurement of 64 GB/s is impossible on PCIe 4.0 x16.
- B) Correct — PCIe 4.0 x16 provides approximately 32 GB/s of bandwidth in a single direction (HtoD or DtoH). An HtoD measurement of 64 GB/s is physically impossible on this hardware. The most likely cause is a unit-conversion mistake (e.g., dividing MB by ms giving MB/ms, then forgetting to multiply by 1000 to reach MB/s, or confusing MiB with MB).
- C) Incorrect — pinned memory improves bandwidth utilisation by eliminating staging copies and can push achieved bandwidth closer to the theoretical peak (~32 GB/s for PCIe 4.0 x16), but it cannot exceed the hardware's physical limit. Pinned memory does not "double" PCIe bandwidth.
- D) Incorrect — NVLink is a GPU-to-GPU (or GPU-to-CPU in specific server configurations) interconnect, not a replacement for PCIe for host memory transfers on standard workstations. Even if NVLink were present, the question specifies an HtoD transfer from host RAM, which uses PCIe.

---

## Q29 — When GPU Overhead Dominates: Small-Kernel Trap

> **Week reference:** Week 9

**Mental Model:** Every GPU invocation carries a fixed overhead: kernel launch latency (~5–50 µs), transfer setup, and driver API calls. For small arrays or fast-completing kernels, this fixed overhead can exceed the actual compute benefit. The GPU wins only when the speedup-per-iteration is large enough and/or the number of iterations is large enough to amortise the fixed overhead.

A benchmark times a GPU kernel on a 1 KB array (256 float32 values). It measures: kernel execution = 0.01 ms, HtoD + DtoH transfers = 0.15 ms, CUDA context/launch overhead = 0.20 ms. A CPU implementation takes 0.02 ms. Which statement is correct?

- A) The GPU is faster because its kernel executes 2× faster than the CPU.
- B) The GPU total time is 0.36 ms, which is 18× slower than the CPU.
- C) The GPU is faster because CUDA parallelism is always advantageous for array operations.
- D) The GPU total time equals the CPU time because transfers and kernel time cancel out.

**Answer: B**

- A) Incorrect — comparing only kernel execution time (0.01 ms) to CPU time (0.02 ms) ignores the unavoidable overheads. The relevant comparison is total wall-clock time, not just the compute phase. Selecting the GPU based on kernel time alone is the classic benchmarking error from exam question Q23.
- B) Correct — total GPU time = kernel (0.01 ms) + transfers (0.15 ms) + launch overhead (0.20 ms) = 0.36 ms. CPU time = 0.02 ms. Ratio = 0.36 / 0.02 = 18×. For very small arrays, the fixed GPU overhead completely dominates the computation, making the GPU vastly slower than the CPU. The break-even point requires a much larger array or many more repeated calls.
- C) Incorrect — GPU parallelism is beneficial only when the amount of work is large enough to saturate the GPU and the overhead can be amortised. For tiny arrays (1 KB), the GPU spends far more time on overhead than on useful work. "Always advantageous" is false.
- D) Incorrect — there is no cancellation. The overheads add up: 0.01 + 0.15 + 0.20 = 0.36 ms total, which is far above the CPU's 0.02 ms. The times do not cancel.

---

## Q30 — Unified Memory Semantics

> **Week reference:** Week 9

**Mental Model:** Unified memory (`cuda.managed_array()` in Numba, or `cudaMallocManaged` in C++) creates a single allocation accessible from both CPU and GPU. The CUDA runtime migrates pages automatically on demand — when the CPU accesses a page, it migrates to host; when the GPU accesses a page, it migrates to device. This simplifies code but does not eliminate PCIe traffic; it just makes transfers implicit and demand-driven.

Which statement about Numba's `cuda.managed_array()` (unified memory) is most accurate?

- A) Unified memory eliminates all PCIe transfers — data is accessible from both CPU and GPU with zero data movement.
- B) Unified memory migrates pages automatically between host and device on demand, making transfers implicit rather than eliminating them.
- C) Unified memory is always faster than explicit `cuda.to_device()` and `copy_to_host()` calls.
- D) Unified memory requires the GPU and CPU to share the same physical DRAM chip.

**Answer: B**

- A) Incorrect — unified memory does not eliminate PCIe transfers. It makes them implicit: when the GPU accesses a page that currently resides on the host, the CUDA runtime migrates that page over PCIe. The transfer still happens; the programmer just does not call `to_device()` explicitly. For large datasets accessed many times, unified memory can actually perform worse than explicit transfers due to page-fault overhead.
- B) Correct — unified memory provides a single virtual address space visible to both CPU and GPU. The hardware and CUDA runtime manage page migration transparently: when either processor faults on a page not locally resident, the page migrates over PCIe. This simplifies programming but does not change the fundamental PCIe bandwidth constraint or eliminate data movement.
- C) Incorrect — unified memory can be slower than explicit transfers in performance-critical code because page-fault-driven migration has higher latency and less predictable behaviour than a single, batched `to_device` call. Explicit transfers allow the programmer to overlap transfers with computation using CUDA streams; unified memory's demand-driven migration is harder to overlap.
- D) Incorrect — in a standard discrete GPU setup, the GPU has its own GDDR/HBM VRAM and the CPU uses system DRAM — two separate physical memories connected by PCIe. Unified memory is a software/hardware abstraction over this physically separate memory topology, not a requirement for shared physical DRAM. (Integrated GPUs that share physical RAM are a different hardware class.)

---

## Q31 — Streaming to Overlap Transfers and Computation

> **Week reference:** Week 9

**Mental Model:** CUDA streams allow HtoD transfers, kernel execution, and DtoH transfers to be pipelined for independent data chunks. While chunk i is being processed by the kernel, chunk i+1 is being uploaded via HtoD and chunk i-1 is being downloaded via DtoH — all concurrently on separate hardware engines. This is only possible when the operations are on independent data and use different streams.

Which condition must hold for CUDA stream-based pipelining to successfully overlap an HtoD transfer with kernel execution?

- A) The kernel and the HtoD transfer must operate on the same data, so the GPU can prefetch while computing.
- B) The kernel and the HtoD transfer must operate on independent data (different buffers), so neither has to wait for the other to complete.
- C) Stream pipelining only works with pinned memory for the kernel input; standard pageable memory cannot be streamed.
- D) The HtoD transfer and the kernel must both be submitted to the same CUDA stream in order to overlap.

**Answer: B**

- A) Incorrect — if the kernel reads from the same buffer that the HtoD transfer is writing, the GPU must wait for the transfer to complete before the kernel can begin. Overlapping requires independence: the kernel processes already-resident data while new data arrives in a separate buffer.
- B) Correct — the fundamental requirement for transfer/compute overlap is that the kernel and the concurrent HtoD transfer touch different buffers. A typical double-buffering pattern: buffer A is currently on device and being processed by the kernel; buffer B (the next chunk) is being uploaded via HtoD in a separate stream. Once the kernel finishes with A, the DtoH for A and HtoD for C start, while the kernel begins on B. No operation depends on another in-flight operation.
- C) Incorrect — while pinned memory is required for asynchronous (non-blocking) transfers (`cudaMemcpyAsync`), it is not an additional condition for overlap beyond what the stream API already requires. The assertion that "standard pageable memory cannot be streamed" conflates two separate issues. Overlap does require pinned memory for the async copy path, but the fundamental condition for overlap is data independence (option B), not the memory type per se.
- D) Incorrect — placing both operations in the same stream serialises them (all operations in a stream execute in order). To overlap a transfer and a kernel, they must be in different streams so the CUDA runtime can schedule them concurrently on separate hardware engines (copy engine vs. SM compute engine).

---

## Q32 — Transfer Cost for a Pipeline of Three Kernels

> **Week reference:** Week 9

**Mental Model:** Each kernel in a pipeline shares data with its neighbours. With auto-transfer (NumPy args), every kernel call incurs a full 2k-transfer round-trip even for data that was just computed on the GPU in the previous call. With explicit device arrays, intermediate results travel only through GPU VRAM — they never cross PCIe. The total PCIe cost for a k-kernel pipeline with explicit transfers is always 2 transfers (1 HtoD + 1 DtoH), regardless of the number of kernels.

A pipeline has 3 kernels: `K1(inp, mid1)`, `K2(mid1, mid2)`, `K3(mid2, out)` where `inp` is the initial input and `out` is the final output. All arrays are 100 MB. With Numba auto-transfer (all NumPy arrays), how many total PCIe transfers and how many MB of data cross the PCIe bus?

- A) 3 transfers, 300 MB total
- B) 6 transfers, 600 MB total
- C) 12 transfers, 1,200 MB total
- D) 18 transfers, 1,800 MB total

**Answer: C**

- A) Incorrect — auto-transfer moves every argument of every kernel call both HtoD before and DtoH after. There are 3 kernels with 2 arguments each → 6 transfers per kernel direction. Counting only 3 (one per kernel) ignores both the DtoH direction and the second argument.
- B) Incorrect — 6 transfers would correspond to 1 HtoD + 1 DtoH per kernel (only one argument counted). Each kernel has 2 array arguments, each transferred in both directions: 4 transfers per kernel × 3 kernels = 12 transfers.
- C) Correct — each kernel call with 2 NumPy arguments triggers 2 HtoD + 2 DtoH = 4 transfers per call. Over 3 kernel calls: 4 × 3 = 12 transfers. Each transfer moves 100 MB, so total data moved = 12 × 100 MB = 1,200 MB. With explicit device arrays, this collapses to 2 transfers and 200 MB total — a 6× improvement.
- D) Incorrect — 18 transfers would imply 6 transfers per kernel (3 arguments in both directions). Each kernel has exactly 2 arguments, not 3. Counting 18 means either adding a ghost argument or mistakenly counting 3 arguments per kernel.

---
