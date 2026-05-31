# GPU / CUDA Kernels — Code-Based MCQ Practice

> Format: Each question shows a Numba CUDA kernel or grid setup code to analyse.
> Exam frequency: **Every exam** — 4+ questions per exam.

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

**Explanation:** `cuda.grid(2)` computes `(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)`. With block dims `(16, 16)`:
- `row = 1 * 16 + 3 = 19`
- `col = 2 * 16 + 5 = 37`

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

**Explanation:** `500 // 32 = 15`, which launches 15 × 32 = 480 threads. Elements at indices 480–499 are never processed. The correct formula is `bpg = (N + tpb - 1) // tpb = (500 + 31) // 32 = 16`, which covers all 500 elements with 512 threads (the bounds check handles the extra 12).

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

**Explanation:** `200 // 16 = 12` in each dimension. The grid covers 12 × 16 = 192 elements per dimension, so 192 × 192 = 36864 elements are processed. The total output has 200 × 200 = 40000 elements. The shortfall is 40000 − 36864 = **3136 elements** (the rightmost 8 columns, bottom 8 rows, and their intersection are all missed). The fix is `bpg = (math.ceil(200/16), math.ceil(200/16))` or equivalently `((200 + 15) // 16, (200 + 15) // 16) = (13, 13)`.

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

**Explanation:** In row-major order, the last index `k` varies fastest in memory. When threads in a warp all differ by 1 in `k` (Kernel B), they access consecutive memory addresses — this is the definition of coalesced access. In Kernel A, adjacent threads differ in `i`, which strides `J*K` elements apart in memory, causing highly strided (uncoalesced) access. Always ensure warp threads vary in the **last** (fastest-changing) index for row-major arrays.

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

**Explanation:** Numba CUDA kernels do not perform automatic bounds checking. Threads with `i >= 900` will attempt to read `x[i]` and `y[i]` and write `out[i]` beyond the allocated array bounds, leading to undefined behaviour (corrupted data or a CUDA error). The fix is to add `if i < out.shape[0]:` before the computation.

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

**Explanation:** `bpg = (1000 + 256 - 1) // 256 = 1255 // 256 = 4`. Four blocks of 256 threads = 1024 threads total; threads 1000–1023 are harmlessly skipped by the bounds check. Using 3 blocks would only cover 768 elements — not enough. 5 blocks would work but 4 is the minimum needed.

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

The kernel is launched with block dims `(16, 16)` where the first dimension maps to `row` and the second to `col`. Adjacent threads in the second dimension differ by 1 in `col`. Does this kernel achieve coalesced memory access?

- A) No — iterating over channels in a loop always prevents coalescing
- B) No — CHW layout means channels are contiguous, not columns
- C) Yes — adjacent threads differ by 1 in `col` (the last/W dimension), so accesses to `img[c, row, col]` are consecutive in memory
- D) Yes — but only for C=1; for multi-channel images coalescing breaks

**Answer: C**

**Explanation:** In CHW row-major layout, the memory stride for `col` is 1 (the W dimension is the fastest axis). Adjacent threads with `col, col+1, col+2, ...` access `img[c, row, col]`, `img[c, row, col+1]`, `img[c, row, col+2]`, which are consecutive in memory — this is coalesced. The channel loop processes each channel sequentially but does not break the coalescing pattern within each iteration.

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

**Explanation:** The `@cuda.jit(device=True)` decorator marks a function as a **device function**: it compiles to PTX and runs on the GPU, callable only from other `@cuda.jit` kernels or device functions. Calling it from host Python raises a `TypingError` or similar error. To use the same logic on the host, define a separate plain Python function. Compare with a regular `@cuda.jit` kernel, which is callable from the host as a kernel launch (but returns nothing).

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

**Explanation:** With `tpb = (256, 1)`, threads in a warp all have the **same** `col` but consecutive `row` values. In row-major layout, `img[row, col]` and `img[row+1, col]` are `W` elements apart (stride = `W * sizeof(dtype)` bytes). For W=512 with float32, that is a stride of 2048 bytes — far from coalesced. The fix is to swap the block dimensions: `tpb = (1, 256)` so threads vary in `col`, accessing `img[row, col]`, `img[row, col+1]`, ... which are consecutive in memory.

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

**Explanation:** CUDA hardware schedules threads in units of 32 called **warps**. If `TPB` is not a multiple of 32, the last warp of each block is only partially filled (padding threads still consume scheduler resources but do no useful work). `TPB = 64 = 2 × 32` creates exactly 2 full warps per block — zero waste. Any multiple of 32 in the range [32, 1024] is a valid and efficient choice. Common values are 128, 256, and 512, but 64 is perfectly valid.

---

## Key Facts

- `cuda.grid(2)` returns `(row, col)`. Adjacent threads in the same warp differ by 1 in the **last** dimension (`col`).
- Grid dims: always use `(N + tpb - 1) // tpb` (ceiling division) to avoid missing elements.
- Always bounds check: `if i < arr.shape[0]:` — CUDA will not do it for you.
- **Warp size = 32**; use multiples of 32 for `tpb` to avoid partial-warp waste.
- **Coalescing rule (row-major):** threads in a warp should vary in the **last (fastest-changing) index** so their memory accesses are consecutive.
- `@cuda.jit(device=True)` functions are GPU-only; they cannot be called from host Python.
- `@cuda.jit` kernels return `None` and are launched with `kernel[bpg, tpb](args)` from the host.
