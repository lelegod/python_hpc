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
