# CUDA Grid & Warp Coalescing — Visual Reference

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md) · [CUDA Grid Visual](cuda_grid_visual.md)

## Contents

- [The Root Cause of All Confusion](#the-root-cause-of-all-confusion)
- [`row, col = cuda.grid(2)` — What They Map To](#row-col-cudagrid2-what-they-map-to)
- [Core Formula (memorise this)](#core-formula-memorise-this)
- [Full Warp Rundown: Block (1, 256) ✅ BEST](#full-warp-rundown-block-1-256-best)
- [Full Warp Rundown: Block (256, 1) ❌ WORST](#full-warp-rundown-block-256-1-worst)
- [Full Warp Rundown: Block (16, 16) ⚠️ PARTIAL](#full-warp-rundown-block-16-16-partial)
- [How to Work Out Any Block Shape in 4 Steps](#how-to-work-out-any-block-shape-in-4-steps)
- [The Fix: Swapped Convention `j, i = cuda.grid(2)`](#the-fix-swapped-convention-j-i-cudagrid2)
- [Full Warp Rundown: Block (32, 16) ❌ SAME AS (256,1)](#full-warp-rundown-block-32-16-same-as-2561)
- [Full Warp Rundown: Block (16, 32) ⚠️ SAME AS (16,16)](#full-warp-rundown-block-16-32-same-as-1616)
- [Summary Table](#summary-table)
- [One-Sentence Rule for the Exam](#one-sentence-rule-for-the-exam)
- [How to Find a Thread's Global (row, col) — Visual](#how-to-find-a-threads-global-row-col-visual)

---

## The Root Cause of All Confusion

**NumPy / CPU (row-major):** rightmost index is fastest in memory
```
x[row, col] → col is adjacent in memory (col+1 is 8 bytes away)
              row is far apart (row+1 is N*8 bytes away)
```

**CUDA blocks:** leftmost dimension (x) is fastest within a warp
```
Thread ID = threadIdx.x + threadIdx.y * blockDim.x
→ threadIdx.x increments fastest
→ row (mapped to x-dim) varies fastest across warp threads
```

**They pull in opposite directions.** NumPy wants the rightmost index to vary.
CUDA warps naturally vary the leftmost index. You must force them to agree.

---

## `row, col = cuda.grid(2)` — What They Map To

```python
row = blockIdx.x * blockDim.x + threadIdx.x   # x-dim — FIRST return value — varies fastest in warp
col = blockIdx.y * blockDim.y + threadIdx.y   # y-dim — SECOND return value — varies slower in warp
```

**Problem:** row varies in warp (bad — row access is strided in row-major).
**Solution:** make blockDim.x = 1 so row can't vary, forcing col to take over.

---

## Core Formula (memorise this)

```
Thread ID = threadIdx.x + threadIdx.y * blockDim.x
```

To find any thread's indices:
```
threadIdx.x = ID % blockDim.x
threadIdx.y = ID // blockDim.x
```

---

## Full Warp Rundown: Block (1, 256) ✅ BEST

```
blockDim.x=1, blockDim.y=256
Thread ID = threadIdx.x + threadIdx.y * 1 = threadIdx.y
```

```
tid |  .x  |  .y  | row | col
----|------|------|-----|------
 0  |   0  |   0  |  R  |  C+0
 1  |   0  |   1  |  R  |  C+1
 2  |   0  |   2  |  R  |  C+2
 3  |   0  |   3  |  R  |  C+3
... |   0  |  ...  |  R  |  ...
31  |   0  |  31  |  R  |  C+31
```

- row = **R for every thread** (threadIdx.x stuck at 0)
- col = **C, C+1, ..., C+31** (threadIdx.y = ID)

Memory: `x[R,C]`, `x[R,C+1]`, ... `x[R,C+31]` → **32 consecutive bytes → 1 cache transaction** ✅

---

## Full Warp Rundown: Block (256, 1) ❌ WORST

```
blockDim.x=256, blockDim.y=1
Thread ID = threadIdx.x + threadIdx.y * 256 = threadIdx.x
```

```
tid |  .x  |  .y  |  row  | col
----|------|------|-------|-----
 0  |   0  |   0  |  R+0  |  C
 1  |   1  |   0  |  R+1  |  C
 2  |   2  |   0  |  R+2  |  C
 3  |   3  |   0  |  R+3  |  C
... |  ...  |   0  |  ...  |  C
31  |  31  |   0  |  R+31 |  C
```

- row = **R, R+1, ..., R+31** (each thread = different row)
- col = **C for every thread**

Memory: `x[R,C]`, `x[R+1,C]`, ... `x[R+31,C]` → **each access N*8 bytes apart → 32 cache transactions** ❌

---

## Full Warp Rundown: Block (16, 16) ⚠️ PARTIAL

```
blockDim.x=16, blockDim.y=16
Thread ID = threadIdx.x + threadIdx.y * 16
```

```
tid |  .x  |  .y  |  row   | col    ← note: threadIdx.y resets every 16
----|------|------|--------|--------
 0  |   0  |   0  |  R+0   |  C+0
 1  |   1  |   0  |  R+1   |  C+0
 2  |   2  |   0  |  R+2   |  C+0
... |  ...  |   0  |  ...   |  C+0
15  |  15  |   0  |  R+15  |  C+0   ← threadIdx.x wraps at 16
16  |   0  |   1  |  R+0   |  C+1   ← threadIdx.y increments
17  |   1  |   1  |  R+1   |  C+1
18  |   2  |   1  |  R+2   |  C+1
... |  ...  |   1  |  ...   |  C+1
31  |  15  |   1  |  R+15  |  C+1
```

- Threads 0–15: **16 different rows, same col C**
- Threads 16–31: **16 different rows, same col C+1**
- col has only **2 distinct values** across the whole warp

Memory: two columns accessed, each in 16 strided rows — not contiguous ⚠️

---

## How to Work Out Any Block Shape in 4 Steps

1. Write `threadIdx.x = ID % blockDim.x` and `threadIdx.y = ID // blockDim.x`
2. Compute `row = R + threadIdx.x`, `col = C + threadIdx.y`
3. Count distinct col values across IDs 0..31
4. More distinct col values = better coalescing

**Number of distinct col values = min(32, ceil(32 / blockDim.x))**

| blockDim.x | distinct col values | coalescing |
|---|---|---|
| 1 | 32 | ✅ perfect |
| 2 | 16 | ⚠️ partial |
| 4 | 8 | ⚠️ partial |
| 8 | 4 | ⚠️ poor |
| 16 | 2 | ⚠️ poor |
| 32 | 1 | ❌ none |
| 64+ | 1 | ❌ none |

**Smaller blockDim.x = more col variation = better coalescing.**

---

## The Fix: Swapped Convention `j, i = cuda.grid(2)`

Some kernels swap names so the column gets the x-dim:
```python
j, i = cuda.grid(2)
# j = x-dim → varies fastest in warp
# i = y-dim → same in warp
y[i, j] = 2 * x[i, j]   # j varies → col varies → coalesced ✅
```

This works even with large blockDim.x because now j (column) = x-dim = naturally varies in warp.
The grid launch must also flip: `bpg = (cols // tpb[0], rows // tpb[1])`.

---

## Full Warp Rundown: Block (32, 16) ❌ SAME AS (256,1)

```
blockDim.x=32, blockDim.y=16
threadIdx.x = ID % 32
threadIdx.y = ID // 32
```

```
tid |  .x  |  .y  |  row   | col
----|------|------|--------|-----
 0  |   0  |   0  |  R+0   |  C
 1  |   1  |   0  |  R+1   |  C
 2  |   2  |   0  |  R+2   |  C
... |  ...  |   0  |  ...   |  C
31  |  31  |   0  |  R+31  |  C   ← threadIdx.y never left 0
```

- row = R+0 to R+31 (32 different rows)
- col = **C for every thread** — threadIdx.y never reaches 1 within 32 threads

Memory: 32 different rows all in the same column → **strided, same as (N,1)** ❌

---

## Full Warp Rundown: Block (16, 32) ⚠️ SAME AS (16,16)

```
blockDim.x=16, blockDim.y=32
threadIdx.x = ID % 16
threadIdx.y = ID // 16
```

```
tid |  .x  |  .y  |  row   | col
----|------|------|--------|------
 0  |   0  |   0  |  R+0   |  C+0
 1  |   1  |   0  |  R+1   |  C+0
...
15  |  15  |   0  |  R+15  |  C+0   ← threadIdx.x wraps at 16
16  |   0  |   1  |  R+0   |  C+1   ← threadIdx.y increments
17  |   1  |   1  |  R+1   |  C+1
...
31  |  15  |   1  |  R+15  |  C+1
```

- col has only **2 values** (C+0 and C+1) — identical to (16,16)
- blockDim.y=32 instead of 16 makes no difference to warp 0

**Only blockDim.x matters for coalescing. blockDim.y is irrelevant.**

---

## Summary Table

| Block (Dx, Dy) | row in warp | col in warp | Coalescing |
|---|---|---|---|
| **(1, N)** | same | varies 0..31 | ✅ Best |
| **(16, 16)** | varies 0..15 (×2) | 2 values | ⚠️ Partial |
| **(16, 32)** | varies 0..15 (×2) | 2 values | ⚠️ Same as (16,16) |
| **(32, 16)** | varies 0..31 | same | ❌ Same as (N,1) |
| **(32, 32)** | varies 0..31 | same | ❌ Bad |
| **(N, 1)** | varies 0..31 | same | ❌ Worst |

**Key insight: only blockDim.x determines coalescing quality. blockDim.y is irrelevant.**

---

## One-Sentence Rule for the Exam

> **NumPy is right-most fastest. CUDA warps are left-most fastest. To coalesce, force them to agree: use block (1, N) so the left-most dimension is locked at 1, making the right-most (col) vary across the warp.**

---

## How to Find a Thread's Global (row, col) — Visual

**Example:** block size (16,16), blockIdx=(1,2), threadIdx=(3,5)

**Step 1 — The block grid** (each square = one 16×16 block):

```
           blockIdx.y →
        0        1        2        3
      +--------+--------+--------+--------+
   0  | (0,0)  | (0,1)  | (0,2)  | (0,3)  |
      +--------+--------+--------+--------+
↑  1  | (1,0)  | (1,1)  |■(1,2)■ | (1,3)  |   ← you are in this block
bIdx.x+--------+--------+--------+--------+
   2  | (2,0)  | (2,1)  | (2,2)  | (2,3)  |
      +--------+--------+--------+--------+
```

**Step 2 — Zoom into block (1,2)** — showing global (row, col) inside each cell:

```
              threadIdx.y →          (= col offset within block)
         0        1        2        3        4        5        6   ...
      +--------+--------+--------+--------+--------+--------+--------+
   0  | (16,32)|(16,33) |(16,34) |(16,35) |(16,36) |(16,37) |(16,38) |
↑     +--------+--------+--------+--------+--------+--------+--------+
tid.x 1  | (17,32)|(17,33) |(17,34) |(17,35) |(17,36) |(17,37) |(17,38) |
      +--------+--------+--------+--------+--------+--------+--------+
   2  | (18,32)|(18,33) |(18,34) |(18,35) |(18,36) |(18,37) |(18,38) |
      +--------+--------+--------+--------+--------+--------+--------+
   3  | (19,32)|(19,33) |(19,34) |(19,35) |(19,36) |  ★    |(19,38) |
      +--------+--------+--------+--------+--------+--------+--------+
   4  | (20,32)|(20,33) |(20,34) |(20,35) |(20,36) |(20,37) |(20,38) |
      +--------+--------+--------+--------+--------+--------+--------+

★ = threadIdx=(3,5) → global (row=19, col=37)
```

**Step 3 — Compute global position:**

```
Block (1,2) starts at:
  row_start = blockIdx.x × blockDim.x = 1 × 16 = 16
  col_start = blockIdx.y × blockDim.y = 2 × 16 = 32

Thread ★ at offset (3,5) within the block:
  row = 16 + 3 = 19
  col = 32 + 5 = 37
```

Thread ★ computes `C[19, 37]`.

**General formula:**
```
row = blockIdx.x * blockDim.x + threadIdx.x
col = blockIdx.y * blockDim.y + threadIdx.y
```
