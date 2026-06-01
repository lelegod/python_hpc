# GPU / CUDA Kernels — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Question 1 — Thread Index Calculation](#question-1-thread-index-calculation)
- [Question 2 — Grid Launch Bug (Off-by-One)](#question-2-grid-launch-bug-off-by-one)
- [Question 3 — 2D Grid Coverage Bug](#question-3-2d-grid-coverage-bug)
- [Question 4 — Memory Coalescing in 3D Arrays](#question-4-memory-coalescing-in-3d-arrays)
- [Question 5 — Missing Bounds Check](#question-5-missing-bounds-check)
- [Question 6 — Correct Block Count Formula](#question-6-correct-block-count-formula)
- [Question 7 — Coalescing in a CHW Image Kernel](#question-7-coalescing-in-a-chw-image-kernel)
- [Question 7b — Coalescing in a CHW Kernel (Standard Block Rules)](#question-7b-coalescing-in-a-chw-kernel-standard-block-rules)
- [Question 8 — Device Function Callable Scope](#question-8-device-function-callable-scope)
- [Question 9 — Block Dimension Orientation and Coalescing](#question-9-block-dimension-orientation-and-coalescing)
- [Question 10 — Thread Block Size and Warp Efficiency](#question-10-thread-block-size-and-warp-efficiency)
- [Key Facts](#key-facts)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Question 11 — syncthreads Inside an If-Branch](#question-11-syncthreads-inside-an-if-branch)
- [Question 12 — Transfer Count in an Explicit Loop](#question-12-transfer-count-in-an-explicit-loop)
- [Question 13 — Naive vs Optimised Transfer Loop](#question-13-naive-vs-optimised-transfer-loop)
- [Question 14 — Block Reduction: Step Trace](#question-14-block-reduction-step-trace)
- [Question 15 — Warp Divergence Code Analysis](#question-15-warp-divergence-code-analysis)
- [Question 16 — nsys Bandwidth Calculation](#question-16-nsys-bandwidth-calculation)
- [Question 17 — 3D Kernel: Choose Block Shape for Coalescing](#question-17-3d-kernel-choose-block-shape-for-coalescing)
- [Question 18 — Transfer Count: Pre-allocated Device Output](#question-18-transfer-count-pre-allocated-device-output)
- [Question 19 — Mixed Arguments: Automatic vs Explicit Transfers](#question-19-mixed-arguments-automatic-vs-explicit-transfers)
- [Question 20 — Shared Memory Reduction: Missing syncthreads](#question-20-shared-memory-reduction-missing-syncthreads)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Question 21 — Shared Array with Variable Size: Compilation Trap](#question-21--shared-array-with-variable-size-compilation-trap)
- [Question 22 — Atomic Add vs Non-Atomic Race Condition](#question-22--atomic-add-vs-non-atomic-race-condition)
- [Question 23 — Timing Without `cuda.synchronize()`](#question-23--timing-without-cudasynchronize)
- [Question 24 — Pinned Memory Transfer Code Reading](#question-24--pinned-memory-transfer-code-reading)
- [Question 25 — In-Place Kernel: How Many DtoH Transfers?](#question-25--in-place-kernel-how-many-dtoh-transfers)
- [Question 26 — Grid-Stride Loop: Which Indices Does Thread 5 Process?](#question-26--grid-stride-loop-which-indices-does-thread-5-process)
- [Question 27 — `blockDim.x` vs `gridDim.x` Inside a Kernel](#question-27--blockdimx-vs-griddimx-inside-a-kernel)
- [Question 28 — 2D Grid Index Verification](#question-28--2d-grid-index-verification)
- [Question 29 — Shared Memory Padding for Bank Conflicts](#question-29--shared-memory-padding-for-bank-conflicts)
- [Question 30 — What Does the Kernel Call Return?](#question-30--what-does-the-kernel-call-return)

---

> Format: Each question shows a Numba CUDA kernel or grid setup code to analyse.
> Exam frequency: **Every exam** — 4+ questions per exam.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions (Q1–Q10)](#question-1--thread-index-calculation)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice (Q11–Q20)](#set-2--generated-practice-questions-exam-day-focus)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 3 — Extended Practice (Q21–Q30)](#set-3--extended-practice)

---

## Question 1 — Thread Index Calculation

Given the following kernel:

```python
@cuda.jit
def kernel(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] + B[row, col]
```

The kernel is launched with block dimensions `(16, 16)`. A thread has `blockIdx = (1, 2)` and `threadIdx = (3, 5)`. What are the values of `row` and `col`?

- A) row = 3, col = 5
- B) row = 16, col = 32
- C) row = 19, col = 37
- D) row = 37, col = 19

**Answer: C**

- A) Incorrect — these are just `threadIdx` values; the block offset (`blockIdx * blockDim`) must be added
- B) Incorrect — these are the block-start offsets (`blockIdx * blockDim`) without adding `threadIdx`
- C) Correct — `cuda.grid(2)` computes `(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)`: row = 1×16 + 3 = 19, col = 2×16 + 5 = 37
- D) Incorrect — `row` and `col` are swapped here; `cuda.grid(2)` returns (x-dim, y-dim) = (row, col), not (col, row)

---

## Question 2 — Grid Launch Bug (Off-by-One)

The following code launches a 1D kernel over an array of N=500 elements:

```python
@cuda.jit
def kernel(arr):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] = arr[i] * 2.0

N = 500
tpb = 32
bpg = N // tpb   # integer division
kernel[bpg, tpb](arr)
```

What is the bug, and how many elements are left unprocessed?

- A) No bug — integer division always works for GPU grids
- B) `bpg = 15`, so only 480 elements are processed; the last 20 are never touched
- C) `bpg = 16`, so 512 elements are processed and 12 are processed twice
- D) The kernel will crash because 500 is not divisible by 32

**Answer: B**

- A) Incorrect — integer division truncates, so when N is not a multiple of tpb some elements at the end are never covered by any thread
- B) Correct — `500 // 32 = 15`, launching 15 × 32 = 480 threads; indices 480–499 are never processed; fix with `(N + tpb - 1) // tpb = 16`
- C) Incorrect — `bpg` is 15, not 16; 16 would be the correct ceiling-division result, not what the code produces
- D) Incorrect — CUDA does not crash for non-divisible sizes; the bounds check `if i < arr.shape[0]` handles extra threads safely

---

## Question 3 — 2D Grid Coverage Bug

The following kernel processes a 200×200 output matrix with block dimensions (16, 16):

```python
@cuda.jit
def kernel(out):
    row, col = cuda.grid(2)
    if row < out.shape[0] and col < out.shape[1]:
        out[row, col] = 1.0

tpb = (16, 16)
bpg = (200 // 16, 200 // 16)   # BUG: integer division in both dims
kernel[bpg, tpb](out)
```

How many output elements are left unprocessed?

- A) 0 — all 40000 elements are covered
- B) 400 — only the last row and column are missed
- C) 3136 — the grid covers only 192×192 elements
- D) 784 — only the corner block is missed

**Answer: C**

- A) Incorrect — `200 // 16 = 12`, not 13; ceiling division is required to cover all elements
- B) Incorrect — `200 - 192 = 8` rows and 8 columns are missed, not just 1 row and column; 8×200 + 8×200 − 64 = 3136 elements
- C) Correct — `200 // 16 = 12` blocks per dim; 12 × 16 = 192 elements covered per dim; missed = 40000 − 36864 = 3136
- D) Incorrect — the missed region is an L-shaped band (rightmost 8 cols + bottom 8 rows), not a single corner block

---

## Question 4 — Memory Coalescing in 3D Arrays

A volumetric dataset `vol` has shape `(I, J, K)` and is stored in row-major (C-contiguous) order, so element `vol[i, j, k]` is at memory address `base + i*J*K + j*K + k`.

Two kernels are proposed:

```python
# Kernel A — threads vary in the i dimension
@cuda.jit
def kernel_a(vol):
    i, j, k = cuda.grid(3)
    vol[i, j, k] = vol[i, j, k] + 1.0

# Block dims for A: (32, 1, 1)  — 32 threads differ in i

# Kernel B — threads vary in the k dimension
@cuda.jit
def kernel_b(vol):
    i, j, k = cuda.grid(3)
    vol[i, j, k] = vol[i, j, k] + 1.0

# Block dims for B: (1, 1, 32)  — 32 threads differ in k
```

Which kernel gives better memory coalescing?

- A) Kernel A, because row-major means rows are contiguous
- B) Kernel B, because varying k (the last/fastest axis) gives consecutive memory addresses for adjacent threads
- C) Both are equally coalesced — row-major does not affect GPU memory access
- D) Neither is coalesced; a 2D block is always required

**Answer: B**

- A) Incorrect — in row-major order, varying `i` strides `J*K` elements between adjacent threads, which is far from consecutive and highly uncoalesced
- B) Correct — `k` is the fastest-changing (last) index; adjacent warp threads differ by 1 in `k`, giving consecutive memory addresses and fully coalesced access
- C) Incorrect — row-major layout directly determines which index varies fastest; it absolutely affects coalescing
- D) Incorrect — a 1D block varying in the last index achieves full coalescing; a 2D block is not required

---

## Question 5 — Missing Bounds Check

```python
@cuda.jit
def add(x, y, out):
    i = cuda.grid(1)
    out[i] = x[i] + y[i]   # No bounds check!

N = 900
add[4, 256](x, y, out)   # 4 * 256 = 1024 threads launched
```

The arrays `x`, `y`, and `out` each have length 900. What happens for threads where `i` is in the range 900 to 1023?

- A) Nothing — CUDA automatically ignores threads beyond the array size
- B) Those threads silently write zeros to `out[900]` through `out[1023]`
- C) Out-of-bounds memory access occurs, which can cause incorrect results or a device error
- D) The kernel launch raises a Python IndexError before execution starts

**Answer: C**

- A) Incorrect — CUDA provides no automatic bounds checking; it is the programmer's responsibility to add the guard
- B) Incorrect — CUDA does not write zeros; it accesses whatever memory lies beyond the array, producing undefined behaviour
- C) Correct — without `if i < out.shape[0]:`, threads 900–1023 read and write beyond the allocated buffer, causing undefined behaviour or a CUDA error
- D) Incorrect — kernel launches are asynchronous; Python raises no error at launch time; the problem occurs silently at runtime on the GPU

---

## Question 6 — Correct Block Count Formula

A kernel is launched with the following grid calculation:

```python
@cuda.jit
def process(data):
    i = cuda.grid(1)
    if i < data.shape[0]:
        data[i] = data[i] ** 2

N = 1000
tpb = 256
bpg = (N + tpb - 1) // tpb
process[bpg, tpb](data)
```

How many thread blocks are launched?

- A) 3
- B) 4
- C) 5
- D) 16

**Answer: B**

- A) Incorrect — 3 blocks × 256 = 768 threads, covering only indices 0–767 and leaving 232 elements unprocessed
- B) Correct — `(1000 + 255) // 256 = 1255 // 256 = 4`; 4 blocks × 256 = 1024 threads; the bounds check skips threads 1000–1023
- C) Incorrect — 5 blocks would work but wastes a full block; 4 is the minimum required to cover all 1000 elements
- D) Incorrect — 16 comes from `1000 // 64` with a different tpb; with tpb=256 the answer is 4

---

## Question 7 — Coalescing in a CHW Image Kernel

The following kernel processes a channels-first image `img` with shape `(C, H, W)` stored in row-major order:

```python
@cuda.jit
def process(img):   # img shape: (C, H, W)
    row, col = cuda.grid(2)
    if row < img.shape[1] and col < img.shape[2]:
        for c in range(img.shape[0]):
            img[c, row, col] *= 2.0
```

The kernel is launched with block dims `(16, 16)`. With `row, col = cuda.grid(2)` and blockDim.x=16: Thread ID = threadIdx.x + threadIdx.y*16. Warp threads 0–15 have threadIdx.x=0..15 (row varies), warp threads 16–31 have threadIdx.x=0..15, threadIdx.y=1 (row varies, col changes once). Does this kernel achieve coalesced memory access for `img[c, row, col]`?

- A) No — iterating over channels in a loop always prevents coalescing
- B) No — with (16,16) block and `row,col=cuda.grid(2)`, row varies in warp → strided access regardless of layout
- C) No — with (16,16) block and `row,col=cuda.grid(2)`, row (x-dim) varies in warp, not col; accesses go down rows → non-coalesced
- D) Yes — but only if you use block (1, 256) so col varies in the warp

**Answer: B**

- A) Incorrect — the channel loop is not the issue. The problem is that with (16,16) block and `row,col=cuda.grid(2)`, threadIdx.x=0..15 varies in the warp → row varies. Accessing `img[c, row, col]` with row varying hits different rows (stride=W elements) — non-coalesced.
- B) Correct — With (16,16): Thread ID = threadIdx.x + threadIdx.y*16. Warp threads 0–15: threadIdx.x=0..15 (row varies), col=C+0 (same). Warp threads 16–31: row varies again, col=C+1. Row accesses are N elements apart in memory → non-coalesced. The DTU lecture calls this "Column-wise layout."
- C) Incorrect — col does NOT vary across warp threads with a (16,16) block; only row does (threadIdx.x cycles first). Col changes only between the two half-warps (C and C+1), giving partial at best.
- D) Incorrect — "only for C=1" is not the limiting factor. The coalescing problem is entirely about block shape, not channel count. Fix: use block (1, 256) so blockDim.x=1 → threadIdx.x=0 always → col varies via threadIdx.y → coalesced.

---

## Question 7b — Coalescing in a CHW Kernel (Standard Block Rules)

Same kernel as Q7, but now apply standard Numba block behaviour with no special rules given:

```python
@cuda.jit
def process(img):   # img shape: (C, H, W)
    row, col = cuda.grid(2)
    if row < img.shape[1] and col < img.shape[2]:
        for c in range(img.shape[0]):
            img[c, row, col] *= 2.0
```

The kernel is launched with block `(16, 16)`. Using `row, col = cuda.grid(2)`, and the standard formula `Thread ID = threadIdx.x + threadIdx.y * blockDim.x`, what do warp threads access?

- A) Coalesced — col varies across warp threads, so img[c, row, col] accesses consecutive memory
- B) Not coalesced — row varies across warp threads, so img[c, row, col] accesses different rows (large stride)
- C) Coalesced — CHW layout always gives coalescing regardless of block shape
- D) Not coalesced — the channel loop breaks coalescing for any block shape

**Answer: B**

- A) Incorrect — with block (16,16) and `row = blockIdx.x*16 + threadIdx.x`, threadIdx.x varies across warp threads (tid 0–15 have threadIdx.x=0..15, same threadIdx.y=0). So row varies, col stays the same. Warp threads access `img[c, R+0, col]`, `img[c, R+1, col]`... — different rows, stride = W elements apart.
- B) Correct — Thread ID = threadIdx.x + threadIdx.y*16. For warp threads 0–31: threadIdx.x cycles 0..15 twice while threadIdx.y is 0 then 1. Row = blockIdx.x*16 + threadIdx.x varies; col = blockIdx.y*16 + threadIdx.y stays the same within the first 16 threads. Warp threads access the same column in different rows — strided access, not coalesced.
- C) Incorrect — CHW layout makes W (last axis) contiguous in memory, but coalescing requires threads to vary in col (W), not row. Block shape determines which varies in the warp.
- D) Incorrect — the channel loop is not the issue; each loop iteration is still a full warp instruction. The problem is row varying instead of col.

**Fix:** use block `(1, 256)` so threadIdx.x=0 always, col varies via threadIdx.y — coalesced ✅

---

## Question 8 — Device Function Callable Scope

Consider the following Numba code:

```python
from numba import cuda

@cuda.jit(device=True)
def clamp(x, lo, hi):
    return min(max(x, lo), hi)

@cuda.jit
def normalize(arr):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] = clamp(arr[i], 0.0, 1.0)
```

Can `clamp` be called directly from host Python code, for example as `result = clamp(0.5, 0.0, 1.0)`?

- A) Yes — `device=True` only affects how the function runs on the GPU; it is still callable from Python
- B) Yes — but only if the arguments are numpy scalars, not plain Python floats
- C) No — `device=True` means the function can only be called from within another CUDA kernel or device function, not from the host
- D) No — device functions cannot accept scalar arguments; only arrays are supported

**Answer: C**

- A) Incorrect — `@cuda.jit(device=True)` compiles the function to PTX for GPU execution only; calling it from the host raises a `TypingError`
- B) Incorrect — neither numpy scalars nor plain floats can invoke a device function from the host; the function lives exclusively on the GPU
- C) Correct — device functions are GPU-only helpers callable only from `@cuda.jit` kernels or other device functions; the host cannot invoke them directly
- D) Incorrect — device functions do accept scalar arguments (as shown by `lo` and `hi`); the restriction is about call site (GPU only), not argument type

---

## Question 9 — Block Dimension Orientation and Coalescing

A kernel processes a 2D image `img` with shape `(H, W)` in row-major order:

```python
import math

@cuda.jit
def scale(img):
    row, col = cuda.grid(2)
    if row < img.shape[0] and col < img.shape[1]:
        img[row, col] *= 2.0

H, W = 512, 512
tpb = (256, 1)
bpg = (math.ceil(H / 256), math.ceil(W / 1))
scale[bpg, tpb](img)
```

What memory access pattern do threads in the same warp have for `img[row, col]`?

- A) Coalesced — threads access 256 consecutive rows, which are stored contiguously
- B) Strided — threads in the same warp have the same `col` but differ in `row`, so accesses are W elements apart in memory
- C) Coalesced — row-major means all rows are laid out in memory, making row-wise access efficient
- D) Random — the access pattern depends on the GPU hardware, not the block layout

**Answer: B**

- A) Incorrect — consecutive rows are not contiguous; `img[row, col]` and `img[row+1, col]` are W elements apart (stride = W × sizeof(dtype) bytes)
- B) Correct — with `tpb = (256, 1)`, warp threads share the same `col` and differ in `row`; in row-major layout each step in `row` jumps W elements, giving highly strided uncoalesced access
- C) Incorrect — "row-major" means columns within a row are contiguous, not rows themselves; varying `row` at fixed `col` is the strided direction
- D) Incorrect — the access pattern is determined by the index mapping and memory layout, both of which are known statically; fix is `tpb = (1, 256)` so threads vary in `col`

---

## Question 10 — Thread Block Size and Warp Efficiency

The following kernel launch uses `TPB = 64`:

```python
from numba import cuda
import math

@cuda.jit
def compute(data):
    i = cuda.grid(1)
    if i < data.shape[0]:
        data[i] = data[i] * 3.14

N = 8192
TPB = 64
bpg = (N + TPB - 1) // TPB
compute[bpg, TPB](data)
```

Is `TPB = 64` a good choice for threads per block? Choose the best answer.

- A) No — TPB must always be 256 or 512; other values waste hardware resources
- B) No — 64 is less than the warp size of 32, so warps cannot be filled
- C) Yes — 64 is a multiple of the warp size (32), so no partial warps are created and all threads in every block are used efficiently
- D) No — TPB must be a power of 2 greater than 128 to enable instruction-level parallelism

**Answer: C**

- A) Incorrect — 256 and 512 are common but not mandatory; any multiple of 32 in the range [32, 1024] is valid and efficient
- B) Incorrect — 64 is greater than the warp size of 32 (64 = 2 × 32); this option confuses "less than" with "not a multiple of"
- C) Correct — 64 = 2 × 32 creates exactly 2 full warps per block with zero padding; all threads do useful work and the scheduler incurs no partial-warp overhead
- D) Incorrect — instruction-level parallelism is not gated on a minimum block size of 128; smaller multiples of 32 such as 64 are perfectly valid

---

## Key Facts

- `cuda.grid(2)` returns `(row, col)` where row=x-dim, col=y-dim. Adjacent threads differ by 1 in **row** (x-dim, threadIdx.x). For coalesced access of `x[row, col]`, you need **col** to vary → requires blockDim.x=1 (block shape (1, N)). DTU lecture: `i, j = cuda.grid(2)` with (32,32) = "Column-wise layout" (bad). Fix: `j, i = cuda.grid(2)` with swapped grid (good).
- Grid dims: always use `(N + tpb - 1) // tpb` (ceiling division) to avoid missing elements.
- Always bounds check: `if i < arr.shape[0]:` — CUDA will not do it for you.
- **Warp size = 32**; use multiples of 32 for `tpb` to avoid partial-warp waste.
- **Coalescing rule (row-major):** threads in a warp should vary in the **last (fastest-changing) index** so their memory accesses are consecutive.
- `@cuda.jit(device=True)` functions are GPU-only; they cannot be called from host Python.
- `@cuda.jit` kernels return `None` and are launched with `kernel[bpg, tpb](args)` from the host.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets exam patterns not in Set 1: syncthreads correctness, transfer counting, warp divergence, 3D coalescing, shared memory reduction, bandwidth calculation.

---

## Question 11 — syncthreads Inside an If-Branch

```python
@cuda.jit
def bad_reduce(data, out, n):
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    s = 1
    while s < cuda.blockDim.x:
        if tid % (2 * s) == 0:
            if i + s < n:
                data[i] += data[i + s]
            cuda.syncthreads()   # Bug here
        s *= 2
    if tid == 0:
        out[cuda.blockIdx.x] = data[i]
```

What is the bug?

- A) `data[i] += data[i + s]` should use `cuda.atomic.add`
- B) `cuda.syncthreads()` inside the `if` means threads that fail `tid % (2*s) == 0` never reach the barrier → deadlock
- C) `s *= 2` must appear before the `if` check
- D) `cuda.syncthreads()` is missing after the while loop

**Answer: B**

- A) Incorrect — `atomic.add` handles concurrent writes to the same address; the issue here is barrier synchronisation, not concurrent writes
- B) Correct — `cuda.syncthreads()` requires ALL threads in the block to call it; threads where `tid % (2*s) ≠ 0` skip the `if` body entirely and never reach the barrier → those threads wait at the next `syncthreads()` that never comes → deadlock; fix: move `syncthreads()` outside the `if`
- C) Incorrect — the ordering of `s *= 2` relative to the `if` affects the reduction pattern but is not the cause of the deadlock
- D) Incorrect — the missing sync after the loop is not the primary bug; the deadlock from the misplaced sync inside the `if` is

---

## Question 12 — Transfer Count in an Explicit Loop

```python
from numba import cuda
import numpy as np

@cuda.jit
def square(x, out):
    i = cuda.grid(1)
    if i < out.shape[0]:
        out[i] = x[i] ** 2

results = []
for arr in arrays:           # 25 NumPy arrays
    d_arr = cuda.to_device(arr)
    d_out = cuda.device_array_like(arr)
    square[bpg, tpb](d_arr, d_out)
    results.append(d_out.copy_to_host())
```

How many HtoD and DtoH transfers occur in total?

- A) 25 HtoD + 25 DtoH
- B) 50 HtoD + 50 DtoH
- C) 25 HtoD + 0 DtoH
- D) 0 HtoD + 25 DtoH

**Answer: A**

- A) Correct — per iteration: `cuda.to_device(arr)` = 1 HtoD; `device_array_like` = 0; `square` receives device arrays → no auto-transfer; `copy_to_host()` = 1 DtoH. Over 25 iterations: 25 HtoD + 25 DtoH.
- B) Incorrect — `d_arr` and `d_out` are device arrays passed to the kernel; Numba does not auto-transfer device array arguments
- C) Incorrect — `copy_to_host()` is an explicit DtoH; it fires 25 times
- D) Incorrect — `cuda.to_device(arr)` is an explicit HtoD; it fires 25 times

---

## Question 13 — Naive vs Optimised Transfer Loop

Two implementations process 100 images, each a NumPy array:

```python
# Naive — passes NumPy arrays directly
for img in images:
    d_out_np = np.zeros_like(img)
    process[bpg, tpb](img, d_out_np)   # 2 NumPy args → auto HtoD + DtoH each

# Optimised — pre-allocates device output
d_out = cuda.device_array(img_shape, dtype=np.float32)
for img in images:
    d_img = cuda.to_device(img)
    process[bpg, tpb](d_img, d_out)
final = d_out.copy_to_host()
```

What are the transfer counts for each?

- A) Naive: 100 HtoD + 100 DtoH. Optimised: 100 HtoD + 1 DtoH.
- B) Naive: 200 HtoD + 200 DtoH. Optimised: 100 HtoD + 1 DtoH.
- C) Naive: 100 HtoD + 100 DtoH. Optimised: 0 HtoD + 1 DtoH.
- D) Both: 200 HtoD + 200 DtoH.

**Answer: B**

- A) Incorrect — naive passes 2 NumPy arrays per call → 2 HtoD + 2 DtoH per call × 100 = 200 HtoD + 200 DtoH; counting one array per call misses the output auto-transfer
- B) Correct — Naive: 2 NumPy args × 100 calls = 200 HtoD + 200 DtoH. Optimised: `cuda.to_device(img)` × 100 = 100 HtoD; `d_out` is a device array → 0 auto DtoH; 1 explicit `copy_to_host()` = 1 DtoH. Total optimised: 100 HtoD + 1 DtoH.
- C) Incorrect — `cuda.to_device(img)` is an HtoD; it is called 100 times in the optimised loop
- D) Incorrect — the optimised version eliminates repeated DtoH for the output; device arrays don't auto-transfer back

---

## Question 14 — Block Reduction: Step Trace

```python
@cuda.jit
def block_reduce(data, out, n):
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    s = 1
    while s < cuda.blockDim.x:
        if tid % (2 * s) == 0 and i + s < n:
            data[i] = data[i] + data[i + s]
        s *= 2
        cuda.syncthreads()
    if tid == 0:
        out[cuda.blockIdx.x] = data[i]
```

With `blockDim.x = 8` and block-local `data = [1, 2, 3, 4, 5, 6, 7, 8]`, what does `data[0]` contain after the loop?

- A) 1
- B) 10 (sum of first four elements)
- C) 36 (sum of all eight elements)
- D) 8 (last element)

**Answer: C**

- A) Incorrect — thread 0 participates in every step and accumulates values from all elements
- B) Incorrect — 10 = 1+2+3+4; the reduction continues to completion (all 8 elements)
- C) Correct — Step s=1: data[0]+=data[1]→3, data[2]+=data[3]→7, data[4]+=data[5]→11, data[6]+=data[7]→15. Step s=2: data[0]+=data[2]→10, data[4]+=data[6]→26. Step s=4: data[0]+=data[4]→36. Final = 1+2+3+4+5+6+7+8 = 36.
- D) Incorrect — 8 is just the last element; the reduction funnels ALL elements' sum into data[0]

---

## Question 15 — Warp Divergence Code Analysis

```python
@cuda.jit
def process(data):
    i = cuda.grid(1)
    if i < data.shape[0]:
        if i % 32 == 0:
            data[i] = data[i] * 100.0
        else:
            data[i] = data[i] + 1.0
```

Launched with `tpb=32`. Does warp divergence occur, and what is the impact?

- A) Yes — within each warp, 1 thread takes `*100` and 31 take `+1`; SM serialises both paths → ~2× slower for this branch
- B) No — the condition `i % 32 == 0` always matches exactly one thread at the warp boundary; the compiler fuses the paths
- C) No — arithmetic instructions are never divergent on CUDA
- D) Yes — the outer `if i < data.shape[0]` is the source of divergence, not the inner `if`

**Answer: A**

- A) Correct — with tpb=32, each warp of 32 threads covers a range [n×32..(n+1)×32-1]; exactly thread 0 of each warp has `i % 32 == 0`, the other 31 don't; the SM must execute both branches with masking → two passes → throughput approximately halved for these instructions
- B) Incorrect — predictability of which thread diverges does not prevent divergence; the SM hardware still serialises both paths when any warp-internal split exists
- C) Incorrect — arithmetic instructions are not special; any `if/else` with intra-warp divergence causes serialised passes
- D) Incorrect — the last-block `i < data.shape[0]` guard is a minor issue; the primary divergence is from the inner `if i % 32 == 0` which fires every warp

---

## Question 16 — nsys Bandwidth Calculation

An nsys run reports:

```
gpumemsizesum: HtoD total = 800 MB
gpumemtimesum: HtoD total = 40,000,000 ns  (= 40 ms)
```

What is the effective HtoD transfer bandwidth?

- A) 2 GB/s
- B) 200 MB/s
- C) 20 GB/s
- D) 0.02 GB/s

**Answer: C**

- A) Incorrect — 2 GB/s would require 0.4 s for 800 MB, not 40 ms
- B) Incorrect — 200 MB/s would require 4 s for 800 MB; that is USB 2.0 speed, not PCIe
- C) Correct — 800 MB ÷ 0.04 s = 20,000 MB/s = 20 GB/s; typical PCIe 4.0 ×16 bandwidth is ~16–32 GB/s
- D) Incorrect — 0.02 GB/s = 20 MB/s; GPU buses operate three orders of magnitude faster

---

## Question 17 — 3D Kernel: Choose Block Shape for Coalescing

```python
@cuda.jit
def vol_process(vol):   # shape (D, H, W), row-major
    i, j, k = cuda.grid(3)
    if i < vol.shape[0] and j < vol.shape[1] and k < vol.shape[2]:
        vol[i, j, k] = vol[i, j, k] ** 2
```

Which block shape gives the best memory coalescing?

- A) `(32, 1, 1)` — 32 threads in i
- B) `(4, 8, 1)` — spread across i and j
- C) `(1, 1, 32)` — 32 threads in k
- D) `(16, 2, 1)` — 32 threads across i and j

**Answer: C**

- A) Incorrect — 32 threads varying in i (x-dim, first return of `cuda.grid(3)`) → stride = H×W between adjacent warp threads → worst coalescing
- B) Incorrect — threads spread across i and j (both slow axes) → strided in two dimensions
- C) Correct — blockDim=(1,1,32): 32 threads differ in k (z-dim, last axis, stride=1) → consecutive addresses → fully coalesced
- D) Incorrect — i and j are slow axes; spreading 32 threads across them produces strided access in both dimensions

---

## Question 18 — Transfer Count: Pre-allocated Device Output

```python
d_data = cuda.to_device(input_data)       # shape (10000,)
d_result = cuda.device_array(10000, dtype=np.float64)

for step in range(200):
    update[bpg, tpb](d_data, d_result, step)

final = d_result.copy_to_host()
```

How many DtoH transfers occur in total?

- A) 0 — device arrays never transfer automatically
- B) 1 — only the explicit `copy_to_host()` after the loop
- C) 200 — one per loop iteration
- D) 201 — one per iteration plus the final copy

**Answer: B**

- A) Incorrect — `copy_to_host()` IS an explicit DtoH; it fires once
- B) Correct — `d_data` and `d_result` are device arrays; passing them to `update` triggers no automatic DtoH; the sole DtoH is the explicit `copy_to_host()` after the loop
- C) Incorrect — there is no `copy_to_host()` inside the loop; this would require it to be there
- D) Incorrect — `copy_to_host()` appears only once (after the loop)

---

## Question 19 — Mixed Arguments: Automatic vs Explicit Transfers

```python
@cuda.jit
def blend(src_np, d_mask, d_out):
    i = cuda.grid(1)
    if i < d_out.shape[0]:
        d_out[i] = src_np[i] * d_mask[i]

src = np.ones(512, dtype=np.float32)
mask_dev = cuda.to_device(np.ones(512, dtype=np.float32))
out_dev = cuda.device_array(512, dtype=np.float32)

blend[16, 32](src, mask_dev, out_dev)
result = out_dev.copy_to_host()
```

How many HtoD and DtoH transfers occur in total?

- A) 3 HtoD + 1 DtoH
- B) 2 HtoD + 1 DtoH
- C) 0 HtoD + 1 DtoH
- D) 2 HtoD + 2 DtoH (1 explicit pre-load of mask + 1 auto for src; 1 auto copy-back of src + 1 explicit copy_to_host)

**Answer: D**

- A) Incorrect — only `src` triggers an automatic HtoD (1 auto); `mask_dev` was pre-loaded with 1 explicit HtoD; `out_dev` is device-only (0 HtoD); total HtoD = 2, not 3
- B) Incorrect — misses the automatic DtoH for `src`; Numba's automatic transfer for NumPy args is a round-trip (HtoD before kernel + DtoH after), so `src` triggers both directions
- C) Incorrect — `src` is a NumPy array passed to a `@cuda.jit` kernel → automatic HtoD occurs
- D) Correct — `cuda.to_device(np.ones(...))` = 1 HtoD (explicit); `blend(src, ...)` with `src` as NumPy = 1 auto HtoD + 1 auto DtoH (Numba copies it back after the kernel in case it was modified); `copy_to_host()` = 1 DtoH (explicit); total = 2 HtoD + 2 DtoH

---

## Question 20 — Shared Memory Reduction: Missing syncthreads

```python
@cuda.jit
def reduce_max(data, out):
    shared = cuda.shared.array(shape=256, dtype=float64)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    shared[tid] = data[i] if i < data.shape[0] else -1e300
    # ← no syncthreads here

    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            if shared[tid + s] > shared[tid]:
                shared[tid] = shared[tid + s]
        s //= 2
        cuda.syncthreads()

    if tid == 0:
        out[cuda.blockIdx.x] = shared[0]
```

What is the bug?

- A) `cuda.shared.array` cannot store `float64`; use `float32`
- B) `s //= 2` should come before `cuda.syncthreads()` inside the loop
- C) A `cuda.syncthreads()` is needed between the initialisation and the while loop to ensure all threads have written `shared` before any thread reads `shared[tid + s]`
- D) `tid < s` should be `tid <= s`

**Answer: C**

- A) Incorrect — `cuda.shared.array` supports `float64` (imported from `numba`)
- B) Incorrect — `s //= 2` only updates the loop variable; whether it precedes or follows `syncthreads()` within the same iteration does not affect correctness, as the barrier still runs before the next iteration's reads
- C) Correct — after `shared[tid] = data[i]`, all threads must finish writing before any thread reads `shared[tid + s]` in the first loop iteration; without a `syncthreads()` between init and the loop, a fast warp can start the first reduction step while slow warps are still writing their initial values → incorrect maximum
- D) Incorrect — `tid < s` is correct; `tid = s` would access `shared[2s]`, which is outside the active range for that step

---

## Set 3 — Extended Practice

> Targets concepts not in Sets 1 & 2: shared-memory compile-time size, atomic operations, synchronize() timing, pinned memory, in-place kernels, grid-stride loops, blockDim/gridDim inspection, 2D index mapping, bank conflict padding, kernel return values.

---

## Question 21 — Shared Array with Variable Size: Compilation Trap

```python
from numba import cuda
from numba import float32

@cuda.jit
def kernel(data, tpb):
    shared = cuda.shared.array(tpb, dtype=float32)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    if i < data.shape[0]:
        shared[tid] = data[i]
    cuda.syncthreads()
    if tid == 0:
        data[cuda.blockIdx.x] = shared[0]
```

What happens when this kernel is compiled and run?

- A) It compiles and runs correctly; `tpb` is passed at launch time so Numba treats it as a constant
- B) It raises a compilation error because `cuda.shared.array` requires a compile-time integer literal, not a runtime kernel argument
- C) It compiles but `shared` silently has size 1 regardless of `tpb`
- D) It compiles and the shared array size is determined from `cuda.blockDim.x` at runtime

**Answer: B**

- A) Incorrect — `tpb` is a kernel argument received at runtime; even though the caller knows its value, the Numba compiler cannot evaluate it when generating PTX; the compiler sees an unknown symbol and raises a `TypingError`
- B) Correct — `cuda.shared.array` allocates memory at PTX compile time; the shape argument must be a Python integer literal embedded in the source (e.g., `cuda.shared.array(256, dtype=float32)`); passing a variable causes a `numba.core.errors.TypingError` stating that the shape must be a constant
- C) Incorrect — Numba does not silently default to size 1; it fails loudly with a compilation error before any execution occurs
- D) Incorrect — shared memory allocation happens at compile time, not at kernel launch; `cuda.blockDim.x` is a runtime value and cannot influence the compile-time PTX allocation

---

## Question 22 — Atomic Add vs Non-Atomic Race Condition

```python
from numba import cuda
import numpy as np

# Version A — non-atomic
@cuda.jit
def count_nonzero_bad(data, counter):
    i = cuda.grid(1)
    if i < data.shape[0]:
        if data[i] != 0:
            counter[0] += 1   # Race condition!

# Version B — atomic
@cuda.jit
def count_nonzero_good(data, counter):
    i = cuda.grid(1)
    if i < data.shape[0]:
        if data[i] != 0:
            cuda.atomic.add(counter, 0, 1)

data = np.array([1, 0, 3, 0, 5, 0, 7, 0], dtype=np.int32)
# Expected: 4 non-zero elements
```

What result does `count_nonzero_bad` produce in practice, and why?

- A) Always 4 — the GPU serialises all threads automatically for writes to the same location
- B) An unpredictable value between 1 and 4 — multiple threads read `counter[0]` before any writes it back, so some increments are lost to race conditions
- C) Always 0 — the non-atomic write is silently discarded for safety
- D) A value greater than 4 — threads can increment `counter[0]` more than once each

**Answer: B**

- A) Incorrect — the GPU does not automatically serialise conflicting writes; that is exactly what `cuda.atomic.add` is for; without it, concurrent read-modify-write operations on the same address produce undefined behaviour
- B) Correct — `counter[0] += 1` compiles to a load-add-store sequence; if threads A and B both load the current value (say 2) before either stores the result, both compute 3 and write 3 — one increment is lost; with many concurrent threads this can drop the final count well below 4; the exact result is non-deterministic
- C) Incorrect — CUDA does not discard writes for safety; it executes every write; the problem is that the writes overwrite each other rather than accumulating
- D) Incorrect — the race condition causes under-counting (lost increments), not over-counting; each thread increments by exactly 1 and cannot exceed its intended contribution

---

## Question 23 — Timing Without `cuda.synchronize()`

```python
from numba import cuda
from time import perf_counter
import numpy as np

@cuda.jit
def scale(data, factor):
    i = cuda.grid(1)
    if i < data.shape[0]:
        data[i] *= factor

N = 10_000_000
d_data = cuda.to_device(np.ones(N, dtype=np.float32))

t0 = perf_counter()
scale[N // 256 + 1, 256](d_data, 2.0)
t1 = perf_counter()
print(f"Time: {(t1 - t0)*1000:.3f} ms")
```

The printed time is consistently 0.04 ms. The actual kernel takes about 2 ms on this GPU. What is wrong?

- A) The kernel is too fast for `perf_counter()` to measure accurately
- B) `d_data` is a device array, so the kernel is skipped
- C) The kernel launch is asynchronous; `t1` is recorded after the launch call returns, not after GPU execution completes; insert `cuda.synchronize()` before `t1 = perf_counter()`
- D) The `factor` argument is a Python float, causing an implicit type conversion that makes timing unreliable

**Answer: C**

- A) Incorrect — `perf_counter()` resolution is sub-microsecond; 2 ms is easily measurable; the problem is not resolution but that the clock stops before the GPU has actually finished
- B) Incorrect — device array arguments are handled correctly by Numba; the kernel executes normally; the issue is purely in the timing methodology
- C) Correct — CUDA kernel launches are fire-and-forget from the host perspective; the `scale[...]()` call enqueues the kernel on the GPU command queue and returns immediately (in ~0.04 ms); to measure true kernel time, add `cuda.synchronize()` between the launch and `t1 = perf_counter()` to block until all GPU work completes
- D) Incorrect — Numba handles Python float → float32 conversion automatically without observable timing side effects; this is not the cause of the 50× timing error

---

## Question 24 — Pinned Memory Transfer Code Reading

```python
from numba import cuda
import numpy as np

N = 1_000_000

# Approach 1: regular NumPy
x_regular = np.random.rand(N).astype(np.float32)

# Approach 2: pinned NumPy
x_pinned = cuda.pinned_array(N, dtype=np.float32)
x_pinned[:] = np.random.rand(N).astype(np.float32)

d1 = cuda.to_device(x_regular)
d2 = cuda.to_device(x_pinned)
```

Both `d1` and `d2` contain the same data on the GPU. Which statement about the two transfers is correct?

- A) Both transfers are functionally identical with the same speed; `pinned_array` only affects GPU-side memory layout
- B) `d2` (from pinned memory) achieves higher and more consistent HtoD bandwidth because the DMA engine transfers directly without an intermediate staging buffer
- C) `d1` is faster because the OS can optimise pageable memory transfers using its own caching
- D) `d2` requires an additional copy during the transfer, making it slower than `d1`

**Answer: B**

- A) Incorrect — `pinned_array` specifically affects host-side memory allocation to be page-locked; this directly impacts PCIe transfer performance, not GPU-side layout
- B) Correct — page-locked (pinned) host memory has a fixed physical address; the GPU's DMA engine can initiate a direct PCIe transfer without first staging data to a locked intermediate buffer (which pageable transfers require); this eliminates one CPU-side memory copy and results in higher sustainable bandwidth, often approaching PCIe theoretical limits
- C) Incorrect — OS caching of pageable memory does not speed up GPU transfers; if anything, the OS needing to lock pages before DMA adds latency; pageable transfers are slower than pinned
- D) Incorrect — it is the pageable case (`d1`) that involves an extra staging copy; pinned memory eliminates this extra copy, making `d2` faster

---

## Question 25 — In-Place Kernel: How Many DtoH Transfers?

```python
from numba import cuda
import numpy as np

@cuda.jit
def normalise_inplace(data, mean, std):
    i = cuda.grid(1)
    if i < data.shape[0]:
        data[i] = (data[i] - mean) / std

N = 500_000
raw = np.random.randn(N).astype(np.float32)
mean_val = float(raw.mean())
std_val = float(raw.std())

d_data = cuda.to_device(raw)
bpg = (N + 255) // 256

normalise_inplace[bpg, 256](d_data, mean_val, std_val)

result = d_data.copy_to_host()
```

How many HtoD and DtoH transfers occur in total?

- A) 1 HtoD + 1 DtoH
- B) 2 HtoD + 2 DtoH (mean and std also transfer)
- C) 1 HtoD + 0 DtoH (result stays on device)
- D) 0 HtoD + 1 DtoH (device arrays need no HtoD)

**Answer: A**

- A) Correct — `cuda.to_device(raw)` = 1 HtoD; `mean_val` and `std_val` are Python scalars passed by value as kernel arguments — they are not arrays and do not trigger memory transfers; `d_data` is modified in-place on the GPU; `d_data.copy_to_host()` = 1 DtoH; total = 1 HtoD + 1 DtoH
- B) Incorrect — scalar kernel arguments (`mean_val`, `std_val`) are passed in GPU registers, not via PCIe memory transfers; only array arguments can trigger HtoD transfers
- C) Incorrect — `copy_to_host()` is an explicit DtoH transfer; it always fires when called
- D) Incorrect — `cuda.to_device(raw)` is an explicit HtoD transfer; it fires once at the start

---

## Question 26 — Grid-Stride Loop: Which Indices Does Thread 5 Process?

```python
from numba import cuda
import numpy as np

@cuda.jit
def double(data, n):
    i = cuda.grid(1)
    stride = cuda.blockDim.x * cuda.gridDim.x
    while i < n:
        data[i] *= 2
        i += stride

N = 100
tpb = 16
bpg = 2  # total threads = 32
double[bpg, tpb](d_data, N)
```

With `tpb=16` and `bpg=2`, what indices does thread with global index 5 process?

- A) Only index 5
- B) Indices 5, 37, 69 (stride = 32, stopping before 100)
- C) Indices 5, 21, 37, 53, 69, 85 (stride = 16)
- D) Indices 5, 36, 67, 98 (stride = 31)

**Answer: B**

- A) Incorrect — without a grid-stride loop, thread 5 would only process index 5; but the loop adds the stride each iteration, so thread 5 processes multiple indices
- B) Correct — total threads = bpg × tpb = 2 × 16 = 32; stride = 32; thread 5 starts at i=5 and processes: 5 (< 100 ✓), 5+32=37 (✓), 37+32=69 (✓), 69+32=101 (≥ 100, exits); indices processed = {5, 37, 69}
- C) Incorrect — stride = 16 would be just `blockDim.x`, not the full grid stride; the full grid stride is `blockDim.x * gridDim.x = 16 * 2 = 32`
- D) Incorrect — stride 31 would indicate an off-by-one error in computing `blockDim.x * gridDim.x - 1`; the correct stride is 32

---

## Question 27 — `blockDim.x` vs `gridDim.x` Inside a Kernel

```python
from numba import cuda
import numpy as np

@cuda.jit
def inspect(out):
    tid = cuda.threadIdx.x
    if tid == 0 and cuda.blockIdx.x == 0:
        out[0] = cuda.blockDim.x
        out[1] = cuda.gridDim.x

d_out = cuda.device_array(2, dtype=np.int32)
inspect[5, 64](d_out)
result = d_out.copy_to_host()
print(result)
```

What does `print(result)` output?

- A) `[5, 64]`
- B) `[64, 5]`
- C) `[320, 1]` (total threads, 1 grid)
- D) `[64, 320]`

**Answer: B**

- A) Incorrect — `blockDim.x` is the threads-per-block value (64), not the blocks-per-grid value (5); the launch syntax `kernel[bpg, tpb]` puts blocks first, threads second
- B) Correct — `inspect[5, 64]` launches with 5 blocks per grid and 64 threads per block; inside the kernel: `cuda.blockDim.x = 64` (threads per block) and `cuda.gridDim.x = 5` (blocks per grid); `out[0] = 64`, `out[1] = 5` → `result = [64, 5]`
- C) Incorrect — `blockDim.x` is not the total thread count (320); it is per-block thread count (64); `gridDim.x` is 5, not 1
- D) Incorrect — `blockDim.x = 64` (correct) but `gridDim.x = 5`, not 320; 320 would be `blockDim.x * gridDim.x` (total threads), which is not what either variable holds

---

## Question 28 — 2D Grid Index Verification

```python
from numba import cuda
import numpy as np

@cuda.jit
def fill_indices(out):
    row, col = cuda.grid(2)
    if row < out.shape[0] and col < out.shape[1]:
        out[row, col] = row * 100 + col

H, W = 32, 32
d_out = cuda.device_array((H, W), dtype=np.int32)
fill_indices[(2, 2), (16, 16)](d_out)
result = d_out.copy_to_host()
```

A thread has `blockIdx = (1, 0)` and `threadIdx = (3, 7)`. What value does it write to `out`?

- A) `out[3, 7] = 307`
- B) `out[19, 7] = 1907`
- C) `out[7, 19] = 719`
- D) `out[3, 23] = 323`

**Answer: B**

- A) Incorrect — `row` and `col` include the block offset; `blockIdx.x * blockDim.x = 1 * 16 = 16` must be added to `threadIdx.x = 3`, giving row = 19, not 3
- B) Correct — `row = blockIdx.x * blockDim.x + threadIdx.x = 1*16 + 3 = 19`; `col = blockIdx.y * blockDim.y + threadIdx.y = 0*16 + 7 = 7`; value = `19 * 100 + 7 = 1907`; `out[19, 7] = 1907`
- C) Incorrect — row and col are swapped; `cuda.grid(2)` returns `(x_index, y_index)` = `(row, col)`; blockIdx.x maps to row, blockIdx.y maps to col
- D) Incorrect — `col = blockIdx.y * blockDim.y + threadIdx.y = 0*16 + 7 = 7`, not 23; 23 would come from incorrectly using `blockIdx.x` for col

---

## Question 29 — Shared Memory Padding for Bank Conflicts

```python
from numba import cuda, float32

TILE = 32

@cuda.jit
def transpose_padded(src, dst):
    # Pad by 1 to avoid bank conflicts
    shared = cuda.shared.array((TILE, TILE + 1), dtype=float32)
    x = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    y = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    if x < src.shape[1] and y < src.shape[0]:
        shared[cuda.threadIdx.y, cuda.threadIdx.x] = src[y, x]
    cuda.syncthreads()
    # Write transposed
    x2 = cuda.blockIdx.y * TILE + cuda.threadIdx.x
    y2 = cuda.blockIdx.x * TILE + cuda.threadIdx.y
    if x2 < dst.shape[1] and y2 < dst.shape[0]:
        dst[y2, x2] = shared[cuda.threadIdx.x, cuda.threadIdx.y]
```

Why is the shared array declared as `(TILE, TILE + 1)` instead of `(TILE, TILE)`?

- A) To ensure the array fits within the 48 KB shared memory limit
- B) The extra column pads each row so that threads reading the same column after transposition map to different shared memory banks, eliminating 32-way bank conflicts
- C) CUDA requires shared arrays to have dimensions that are multiples of 33
- D) The `+ 1` creates space for a sentinel value used by `cuda.syncthreads()`

**Answer: B**

- A) Incorrect — `32 * 32 * 4 = 4096` bytes is well within the 48 KB limit; `32 * 33 * 4 = 4224` bytes is also fine; memory size is not the motivation
- B) Correct — shared memory has 32 banks (each 4 bytes wide); with a 32-column array, column `k` of every row maps to bank `k`; when reading the transposed tile, 32 threads all access column 0 (then column 1, etc.) of different rows — all land in bank 0 simultaneously (32-way conflict); padding to 33 columns shifts each row by one element, scattering accesses across different banks and eliminating the conflict
- C) Incorrect — there is no CUDA requirement that shared array dimensions be multiples of 33; this is a performance optimisation pattern, not a hardware constraint
- D) Incorrect — `cuda.syncthreads()` uses no sentinel values; it is a hardware barrier instruction with no associated data storage requirement

---

## Question 30 — What Does the Kernel Call Return?

```python
from numba import cuda
import numpy as np

@cuda.jit
def square_sum(data, result):
    shared = cuda.shared.array(256, dtype=np.float64)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    shared[tid] = data[i] ** 2 if i < data.shape[0] else 0.0
    cuda.syncthreads()
    if tid == 0:
        s = 0.0
        for k in range(cuda.blockDim.x):
            s += shared[k]
        cuda.atomic.add(result, 0, s)

data = np.arange(256, dtype=np.float64)
d_data = cuda.to_device(data)
d_result = cuda.device_array(1, dtype=np.float64)

x = square_sum[1, 256](d_data, d_result)
print(x)
print(d_result.copy_to_host())
```

What do the two `print` statements output?

- A) `None` then `[5559680.0]` — the kernel returns None; the result (sum of squares 0²+…+255²) is read from the device array
- B) `5559680.0` then `[5559680.0]` — the kernel returns the computed sum
- C) `None` then `[0.0]` — `cuda.atomic.add` requires both arguments to be NumPy arrays
- D) `None` then `[5559679.5]` — floating-point rounding causes a slight error

**Answer: A**

- A) Correct — all `@cuda.jit` kernel calls return `None` to the host, so `x = square_sum[...](...) ` assigns `None` to `x`; `print(x)` prints `None`; the kernel writes its result via `cuda.atomic.add(result, 0, s)` into `d_result`; `d_result.copy_to_host()` retrieves `[5559680.]`; the exact value is `sum(k² for k in 0..255) = 255 × 256 × 511 / 6 = 5,559,680` — an integer value, representable exactly in float64
- B) Incorrect — `@cuda.jit` kernels never return a value to Python; attempting to use the return value of a kernel call always gives `None`; there is no mechanism for a GPU kernel to pass a scalar back to the Python caller via `return`
- C) Incorrect — `cuda.atomic.add(result, 0, s)` is valid: `result` is a device array, `0` is the index, `s` is a scalar float; this correctly accumulates the block sum into `result[0]`
- D) Incorrect — the calculation uses float64 throughout (`dtype=np.float64`); for sums of integers up to 255², float64 has more than enough precision to give an exact integer result; rounding error is not the issue here

---

