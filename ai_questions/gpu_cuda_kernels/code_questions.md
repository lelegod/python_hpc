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

---

> Format: Each question shows a Numba CUDA kernel or grid setup code to analyse.
> Exam frequency: **Every exam** — 4+ questions per exam.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions (Q1–Q10)](#question-1--thread-index-calculation)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice (Q11–Q20)](#set-2--generated-practice-questions-exam-day-focus)

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
- B) 2 HtoD + 1 DtoH (1 explicit pre-load of mask + 1 auto for src; 1 explicit copy_to_host)
- C) 0 HtoD + 1 DtoH
- D) 1 HtoD + 0 DtoH

**Answer: B**

- A) Incorrect — only `src` triggers an automatic HtoD (1 auto); `mask_dev` was pre-loaded with 1 explicit HtoD; `out_dev` is device-only (0 HtoD); total HtoD = 2, not 3
- B) Correct — `cuda.to_device(np.ones(...))` = 1 HtoD (explicit, before call); `blend(src, ...)` with `src` as NumPy = 1 auto HtoD; `mask_dev` and `out_dev` are device arrays = 0 auto; `copy_to_host()` = 1 DtoH; total = 2 HtoD + 1 DtoH
- C) Incorrect — `src` is a NumPy array passed to a `@cuda.jit` kernel → automatic HtoD occurs
- D) Incorrect — `copy_to_host()` is an explicit DtoH; it fires once

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
