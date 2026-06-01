# NumPy Broadcasting — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Shape (4,1,3) + (1,5,3)](#q1--shape-413--153)
- [Q2 — Shape (100,1,6,3) + (100,1,3)](#q2--shape-100163--10013)
- [Q3 — Image Batch Mean Subtraction (N,H,W,C) - (3,)](#q3--image-batch-mean-subtraction-nhwc---3)
- [Q4 — Per-Image Mean Shape Error (32,3)](#q4--per-image-mean-shape-error-323)
- [Q5 — Outer Product via [:, None]](#q5--outer-product-via--none)
- [Q6 — Feature Normalization Shape](#q6--feature-normalization-shape)
- [Q7 — Row Vector vs Matrix Error (3,4) + (3,)](#q7--row-vector-vs-matrix-error-34--3)
- [Q8 — Pairwise Difference Shape](#q8--pairwise-difference-shape)
- [Q9 — Spatial Mask Broadcasting](#q9--spatial-mask-broadcasting)
- [Q10 — Per-Sample Mean Subtraction Error](#q10--per-sample-mean-subtraction-error)
- [Broadcasting Rules — Quick Reference](#broadcasting-rules-quick-reference)
  - [Common Pitfalls](#common-pitfalls)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q11 — Shape (5,3,4) + (3,) Right-Alignment Trap](#q11--shape-534--3-right-alignment-trap)
- [Q12 — Outer Subtraction via [:, None]](#q12--outer-subtraction-via--none)
- [Q13 — Pairwise Lat/Lon Difference Shape](#q13--pairwise-latlon-difference-shape)
- [Q14 — Canonical Outer-Product Shape (3,1) + (1,4)](#q14--canonical-outer-product-shape-31--14)
- [Q15 — ImageNet Normalize Broadcasting](#q15--imagenet-normalize-broadcasting)
- [Q16 — Row Vector Times Matrix](#q16--row-vector-times-matrix)
- [Q17 — Shape (2,3,4,5) + (4,1)](#q17--shape-2345--41)
- [Q18 — Pairwise Distance Matrix Shape](#q18--pairwise-distance-matrix-shape)
- [Q19 — Row-Wise Normalization Shape](#q19--row-wise-normalization-shape)
- [Q20 — Fix (3,4) + (3,) with [:, None]](#q20--fix-34--3-with--none)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q21 — np.broadcast_to Shape and Writability](#q21--npbroadcast_to-shape-and-writability)
- [Q22 — In-Place Add with Shape Expansion](#q22--in-place-add-with-shape-expansion)
- [Q23 — Haversine dsin2 Intermediate Shape](#q23--haversine-dsin2-intermediate-shape)
- [Q24 — distmat_1d Output Shape](#q24--distmat_1d-output-shape)
- [Q25 — 0-D Array Broadcasting Shape](#q25--0-d-array-broadcasting-shape)
- [Q26 — np.broadcast_shapes ValueError](#q26--npbroadcast_shapes-valueerror)
- [Q27 — Stacked newaxis Reshape](#q27--stacked-newaxis-reshape)
- [Q28 — In-Place Op on Size-1 Leading Dim](#q28--in-place-op-on-size-1-leading-dim)
- [Q29 — Three-Way Add with 0-D](#q29--three-way-add-with-0-d)
- [Q30 — np.outer vs Broadcasting Outer Product](#q30--npouter-vs-broadcasting-outer-product)

---

> Format: Each question shows NumPy code to evaluate — predict output shape or error.
> Exam frequency: **2024 exam + re-exam**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — Shape (4,1,3) + (1,5,3)

```python
import numpy as np
a = np.ones((4, 1, 3))
b = np.ones((1, 5, 3))
print((a + b).shape)
```

**What is the output?**

- A) `(4, 5, 3)`
- B) `(4, 1, 3)`
- C) `ValueError`
- D) `(1, 5, 3)`

**Answer: A**

- A) Correct — right-align: 4 vs 1 → 4, 1 vs 5 → 5, 3 vs 3 → 3, giving `(4, 5, 3)`
- B) Incorrect — dim 1 expands from 1 to 5 due to broadcasting
- C) Incorrect — all dim pairs are compatible (equal or one is 1)
- D) Incorrect — dim 0 expands from 1 to 4 due to broadcasting

---

## Q2 — Shape (100,1,6,3) + (100,1,3)

```python
import numpy as np
a = np.ones((100, 1, 6, 3))
b = np.ones((100, 1, 3))
print((a + b).shape)
```

**What is the output?**

- A) `ValueError`
- B) `(100, 1, 6, 3)`
- C) `(100, 100, 6, 3)`
- D) `(100, 6, 3)`

**Answer: C**

- A) Incorrect — all dim pairs are compatible after left-padding
- B) Incorrect — after padding `b` becomes `(1, 100, 1, 3)`, so dim 1 expands to 100
- C) Correct — `b` pads left to `(1, 100, 1, 3)`; aligned dims: 100 vs 1 → 100, 1 vs 100 → 100, 6 vs 1 → 6, 3 vs 3 → 3
- D) Incorrect — result has 4 dims, not 3

---

## Q3 — Image Batch Mean Subtraction (N,H,W,C) - (3,)

```python
import numpy as np
images = np.zeros((32, 224, 224, 3))   # N, H, W, C
mean = np.array([0.485, 0.456, 0.406]) # shape (3,)
result = images - mean
print(result.shape)
```

**What is the output?**

- A) `ValueError`
- B) `(32, 224, 224, 3)`
- C) `(32, 224, 224)`
- D) `(3,)`

**Answer: B**

- A) Incorrect — `mean` pads left to `(1, 1, 1, 3)`, which is fully compatible
- B) Correct — `mean` shape `(3,)` pads to `(1, 1, 1, 3)` and broadcasts over all N, H, W dimensions
- C) Incorrect — the channel dimension (3) is preserved, not dropped
- D) Incorrect — the result takes the shape of the larger operand

---

## Q4 — Per-Image Mean Shape Error (32,3)

```python
import numpy as np
images = np.zeros((32, 224, 224, 3))
mean = np.zeros((32, 3))  # per-image channel mean
result = images - mean    # Will this work?
```

**What happens?**

- A) `result.shape` is `(32, 224, 224, 3)`
- B) `result.shape` is `(32, 32, 224, 3)`
- C) `ValueError` is raised
- D) `result.shape` is `(32, 224, 3)`

**Answer: C**

- A) Incorrect — `mean` pads to `(1, 1, 32, 3)`, making dim 2 (224 vs 32) incompatible
- B) Incorrect — broadcasting never duplicates axes this way; the shapes are incompatible
- C) Correct — `mean` pads left to `(1, 1, 32, 3)`; dim 2: 224 vs 32, neither is 1, so a ValueError is raised
- D) Incorrect — no implicit axis reduction occurs during broadcasting

---

## Q5 — Outer Product via [:, None]

```python
import numpy as np
x = np.arange(5)        # shape (5,)
y = np.arange(3)        # shape (3,)
outer = x[:, None] * y  # outer product
print(outer.shape)
```

**What is the output?**

- A) `(5,)`
- B) `(3,)`
- C) `(5, 3)`
- D) `ValueError`

**Answer: C**

- A) Incorrect — the result is 2D after the `[:, None]` reshape
- B) Incorrect — the result has 5 rows, not 3
- C) Correct — `x[:, None]` reshapes to `(5, 1)`, which broadcasts with `y` shape `(1, 3)` → `(5, 3)`
- D) Incorrect — `(5, 1)` and `(1, 3)` are fully compatible via broadcasting

---

## Q6 — Feature Normalization Shape

```python
import numpy as np
data = np.random.rand(100, 10)  # 100 samples, 10 features
mean = data.mean(axis=0)        # shape (10,)
std  = data.std(axis=0)         # shape (10,)
normalized = (data - mean) / std
print(normalized.shape)
```

**What is the output?**

- A) `(10,)`
- B) `(100,)`
- C) `ValueError`
- D) `(100, 10)`

**Answer: D**

- A) Incorrect — the result retains all 100 rows
- B) Incorrect — the result retains all 10 feature columns
- C) Incorrect — `mean` and `std` both pad to `(1, 10)`, which is fully compatible with `(100, 10)`
- D) Correct — `mean` shape `(10,)` pads to `(1, 10)` and broadcasts across all 100 samples, preserving shape `(100, 10)`

---

## Q7 — Row Vector vs Matrix Error (3,4) + (3,)

```python
import numpy as np
a = np.ones((3, 4))
b = np.ones((3,))
c = a + b   # Will this work?
```

**What happens?**

- A) `c.shape` is `(3, 4)`
- B) `c.shape` is `(3, 3)`
- C) `ValueError` is raised
- D) `c.shape` is `(4, 3)`

**Answer: C**

- A) Incorrect — `b` pads to `(1, 3)`, not `(1, 4)`, so the shapes are incompatible
- B) Incorrect — broadcasting never produces this shape from these inputs
- C) Correct — `b` shape `(3,)` pads left to `(1, 3)`; dim 1: 4 vs 3, neither is 1, so a ValueError is raised
- D) Incorrect — broadcasting does not transpose or reorder axes

---

## Q8 — Pairwise Difference Shape

```python
import numpy as np
p1 = np.random.rand(50, 2)   # 50 points in 2D
p2 = np.random.rand(80, 2)   # 80 points in 2D
diff = p1[:, None, :] - p2[None, :, :]
print(diff.shape)
```

**What is the output?**

- A) `(50, 2)`
- B) `(80, 2)`
- C) `(50, 80)`
- D) `(50, 80, 2)`

**Answer: D**

- A) Incorrect — the result is 3D, not 2D
- B) Incorrect — the result is 3D and includes all 50 points from p1
- C) Incorrect — the coordinate dimension (2) is preserved in the result
- D) Correct — `p1[:, None, :]` is `(50, 1, 2)` and `p2[None, :, :]` is `(1, 80, 2)`; broadcasting gives `(50, 80, 2)`

---

## Q9 — Spatial Mask Broadcasting

```python
import numpy as np
mask = np.zeros((64, 64))              # H, W
images = np.zeros((10, 64, 64, 3))    # N, H, W, C
result = images - mask[None, :, :, None]
print(result.shape)
```

**What is the output?**

- A) `ValueError`
- B) `(10, 64, 64)`
- C) `(10, 64, 64, 3)`
- D) `(64, 64, 3)`

**Answer: C**

- A) Incorrect — `mask[None, :, :, None]` reshapes to `(1, 64, 64, 1)`, which is fully compatible
- B) Incorrect — the result is 4D and includes the channel dimension
- C) Correct — `mask[None, :, :, None]` is `(1, 64, 64, 1)`; dims 0 and 3 broadcast to 10 and 3 respectively
- D) Incorrect — the result retains all 10 images from the batch dimension

---

## Q10 — Per-Sample Mean Subtraction Error

```python
import numpy as np
X  = np.random.rand(100, 50)  # 100 samples, 50 features
mu = X.mean(axis=1)           # shape (100,) — per-sample mean
X_centered = X - mu           # Will this work?
```

**What happens?**

- A) `X_centered.shape` is `(100, 50)`
- B) `X_centered.shape` is `(100, 100)`
- C) `ValueError` is raised
- D) `X_centered.shape` is `(50,)`

**Answer: C**

- A) Incorrect — `mu` pads to `(1, 100)`, making dim 1 (50 vs 100) incompatible
- B) Incorrect — broadcasting never produces this shape from these inputs
- C) Correct — `mu` shape `(100,)` pads left to `(1, 100)`; dim 1: 50 vs 100, neither is 1, so a ValueError is raised
- D) Incorrect — no axis reduction occurs; the fix is `mu[:, None]` to get shape `(100, 1)`

---

## Broadcasting Rules — Quick Reference

| Step | Rule |
|------|------|
| 1 | **Right-align** shapes, pad shorter shape with 1s on the left |
| 2 | **Compare** each dimension pair |
| 3 | Dims are **compatible** if equal, or one of them is 1 |
| 4 | **Output** dim = `max(dim_a, dim_b)` for each position |
| 5 | If any dim pair fails compatibility → **ValueError** |

### Common Pitfalls

- `(3,)` vs `(3, 4)`: pads to `(1, 3)` → dim 1: **3 vs 4 — ERROR**. Use `arr[:, None]` to get `(3, 1)`.
- `(N,)` per-sample means: must use `mu[:, None]` to broadcast over features.
- `(N, C)` per-image means: must use `mean[:, None, None, :]` to broadcast over H, W.
- When unsure, print `.shape` at each step and trace the alignment manually.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets broadcasting shape compatibility rules, output shape prediction, common errors, and vectorized distance/haversine patterns

---

## Q11 — Shape (5,3,4) + (3,) Right-Alignment Trap

```python
import numpy as np
a = np.ones((5, 3, 4))
b = np.ones((3,))
print((a + b).shape)
```

**What is the output?**

- A) `(5, 3, 4)`
- B) `(5, 3, 3)`
- C) `ValueError`
- D) `(5, 3)`

**Answer: C**

- A) Incorrect — `b` shape `(3,)` pads left to `(1, 1, 3)`. The last dim is 4 vs 3, neither is 1, so broadcasting fails. The 3 in `b` does NOT match the 3 in axis 1; right-alignment puts it against axis 2 (size 4).
- B) Incorrect — no broadcasting rule produces a `(5,3,3)` result from these shapes. The trailing dim mismatch causes an error before any shape is produced.
- C) Correct — `(3,)` pads to `(1,1,3)`. Dim 2: 4 vs 3, neither is 1 → `ValueError`. This is the classic right-alignment trap.
- D) Incorrect — broadcasting never reduces rank or drops dimensions. Even a compatible result would be 3D.

---

## Q12 — Outer Subtraction via [:, None]

```python
import numpy as np
x = np.arange(4)           # shape (4,)
y = np.arange(4)           # shape (4,)
outer = x[:, None] - y     # outer subtraction
print(outer.shape)
```

**What is the output?**

- A) `(4,)`
- B) `(4, 4)`
- C) `ValueError`
- D) `(1, 4)`

**Answer: B**

- A) Incorrect — `x[:,None]` makes the result 2D. A 1D output would require both operands to be 1D and compatible, but the explicit `[:,None]` prevents that.
- B) Correct — `x[:,None]` is `(4,1)`. `y` is `(4,)` which pads to `(1,4)`. Broadcasting: 4 vs 1 → 4, 1 vs 4 → 4. Result: `(4,4)`. Entry `[i,j]` = `x[i] - y[j]`.
- C) Incorrect — `(4,1)` and `(1,4)` are fully compatible. Every dim pair has at least one 1.
- D) Incorrect — `(1,4)` would require the first dim to be 1; here it is 4 because of `x[:,None]`.

---

## Q13 — Pairwise Lat/Lon Difference Shape

```python
import numpy as np
lats = np.random.rand(200)   # shape (200,)
lons = np.random.rand(200)   # shape (200,)
dlat = lats[:, None] - lats[None, :]
dlon = lons[:, None] - lons[None, :]
print(dlat.shape, dlon.shape)
```

**What is the output?**

- A) `(200,) (200,)`
- B) `(200, 200) (200, 200)`
- C) `ValueError`
- D) `(1, 200) (1, 200)`

**Answer: B**

- A) Incorrect — the `[:,None]` and `[None,:]` indexing creates 2D arrays. Element-wise subtraction of two `(200,200)` arrays produces a `(200,200)` result, not 1D.
- B) Correct — `lats[:,None]` is `(200,1)` and `lats[None,:]` is `(1,200)`. Broadcasting: (200,1) vs (1,200) → `(200,200)`. Same for `dlon`. This is the vectorized pairwise difference pattern used in haversine distance computation.
- C) Incorrect — `(200,1)` and `(1,200)` are perfectly compatible: every dim has at least one value of 1.
- D) Incorrect — `(1,200)` would only arise if the first array had a leading 1. `lats[:,None]` has shape `(200,1)`, not `(1,200)`.

---

## Q14 — Canonical Outer-Product Shape (3,1) + (1,4)

```python
import numpy as np
a = np.ones((3, 1))
b = np.ones((1, 4))
c = a + b
print(c.shape)
```

**What is the output?**

- A) `(3, 4)`
- B) `(1, 1)`
- C) `(3, 1)`
- D) `ValueError`

**Answer: A**

- A) Correct — `(3,1)` vs `(1,4)`. Dim 0: 3 vs 1 → 3. Dim 1: 1 vs 4 → 4. Result: `(3,4)`. This is the canonical outer-product shape: both 1-dims stretch independently.
- B) Incorrect — 1-dims always stretch to match their partner; they never stay at 1 in the output when the other dim is greater than 1.
- C) Incorrect — `(3,1)` would require dim 1 to stay 1, but `b` has dim 1 = 4, so it stretches.
- D) Incorrect — both dimension pairs have exactly one 1: (3,1) and (1,4). This is the ideal scenario for broadcasting; no error occurs.

---

## Q15 — ImageNet Normalize Broadcasting

```python
import numpy as np
images = np.zeros((16, 128, 128, 3))  # N, H, W, C
mean = np.array([0.485, 0.456, 0.406])  # shape (3,)
std  = np.array([0.229, 0.224, 0.225])  # shape (3,)
normalized = (images - mean) / std
print(normalized.shape)
```

**What is the output?**

- A) `(3,)`
- B) `ValueError`
- C) `(16, 128, 128, 3)`
- D) `(16, 3)`

**Answer: C**

- A) Incorrect — the result takes the shape of the broadcast-expanded operands, not the smaller one alone.
- B) Incorrect — `mean` shape `(3,)` pads left to `(1,1,1,3)`. Last dim: 3 vs 3 (equal), then 1 vs 128, 1 vs 128, 1 vs 16 — all compatible. No error.
- C) Correct — `mean` and `std` both pad to `(1,1,1,3)`, broadcasting over all N, H, W dimensions. The subtraction and division each produce shape `(16,128,128,3)`.
- D) Incorrect — broadcasting expands 1-dims outward; no axes are merged or dropped.

---

## Q16 — Row Vector Times Matrix

```python
import numpy as np
w = np.ones((7,))   # shape (7,)
M = np.ones((3, 7)) # shape (3, 7)
result = M * w
print(result.shape)
```

**What is the output?**

- A) `(3,)`
- B) `(7,)`
- C) `(3, 7)`
- D) `ValueError`

**Answer: C**

- A) Incorrect — the result must preserve both dimensions; no dimension is reduced.
- B) Incorrect — multiplying a 2D matrix by a 1D row vector does not reduce the result to 1D.
- C) Correct — `w` shape `(7,)` pads to `(1,7)`. Against `(3,7)`: dim 0: 3 vs 1 → 3, dim 1: 7 vs 7 → 7. Result: `(3,7)`. Each row of M is element-wise multiplied by `w`.
- D) Incorrect — the trailing dims match (7 vs 7), so no ValueError is raised.

---

## Q17 — Shape (2,3,4,5) + (4,1)

```python
import numpy as np
a = np.ones((2, 3, 4, 5))
b = np.ones((4, 1))
print((a + b).shape)
```

**What is the output?**

- A) `(2, 3, 4, 5)`
- B) `ValueError`
- C) `(2, 3, 4, 1)`
- D) `(2, 3, 4, 4)`

**Answer: A**

- A) Correct — `b` shape `(4,1)` pads left to `(1,1,4,1)`. Against `(2,3,4,5)`: dim 0: 2 vs 1 → 2, dim 1: 3 vs 1 → 3, dim 2: 4 vs 4 → 4, dim 3: 5 vs 1 → 5. Result: `(2,3,4,5)`.
- B) Incorrect — all dim pairs are compatible: every 1 stretches, the 4s match.
- C) Incorrect — the last dim of `b` is 1, which stretches to match the 5 from `a`. The output last dim is 5, not 1.
- D) Incorrect — dim 3 is 5 (from `a`) vs 1 (from `b`). The 1 stretches to 5; there is no way to get 4 in the last dim.

---

## Q18 — Pairwise Distance Matrix Shape

```python
import numpy as np
x = np.random.rand(50, 1, 3)
y = np.random.rand(1, 40, 3)
diff = x - y
dist = np.sqrt((diff ** 2).sum(axis=2))
print(dist.shape)
```

**What is the output?**

- A) `(50, 40, 3)`
- B) `(50, 40)`
- C) `(50, 1, 3)`
- D) `ValueError`

**Answer: B**

- A) Incorrect — `dist` has `sum(axis=2)` applied, which reduces the last dimension (size 3) to a scalar per entry, dropping axis 2 entirely.
- B) Correct — `diff = x - y`: `(50,1,3)` vs `(1,40,3)` → `(50,40,3)`. Then `(diff**2).sum(axis=2)` sums over the coordinate axis (size 3), giving `(50,40)`. `np.sqrt` preserves shape. Result: `(50,40)`, the pairwise distance matrix.
- C) Incorrect — the broadcasting of x and y expands the second dim to 40, not preserves x's shape.
- D) Incorrect — `(50,1,3)` and `(1,40,3)` are fully compatible. The 1s stretch to 50 and 40 respectively.

---

## Q19 — Row-Wise Normalization Shape

```python
import numpy as np
arr = np.ones((6, 4))
v   = np.ones((6, 1))
result = arr / v
print(result.shape)
```

**What is the output?**

- A) `ValueError`
- B) `(6, 4)`
- C) `(6, 1)`
- D) `(1, 4)`

**Answer: B**

- A) Incorrect — `(6,4)` and `(6,1)` are compatible: dim 0 is 6 vs 6 (equal), dim 1 is 4 vs 1 → 4 (broadcasts).
- B) Correct — `v` shape `(6,1)` broadcasts along dim 1: 1 stretches to 4. Result: `(6,4)`. Each element `arr[i,j]` is divided by `v[i,0]` — this is the pattern for row-wise normalization.
- C) Incorrect — dim 1 of `v` (size 1) stretches to match `arr`'s dim 1 (size 4). The output cannot be `(6,1)`.
- D) Incorrect — dim 0 is 6 vs 6 (equal, not 1), so no stretching in dim 0 occurs. The output has 6 rows.

---

## Q20 — Fix (3,4) + (3,) with [:, None]

```python
import numpy as np
a = np.ones((3, 4))
b = np.ones((3,))
try:
    c = a + b
    print("Shape:", c.shape)
except ValueError as e:
    print("Error")

b2 = b[:, None]  # fix
c2 = a + b2
print("Fixed shape:", c2.shape)
```

**What is the output?**

- A) `Shape: (3, 4)` then `Fixed shape: (3, 4)`
- B) `Error` then `Fixed shape: (3, 4)`
- C) `Error` then `Fixed shape: (3, 1)`
- D) `Shape: (3, 3)` then `Fixed shape: (3, 4)`

**Answer: B**

- A) Incorrect — `a + b` where `b` shape `(3,)` pads to `(1,3)`. Last dims 4 vs 3 — mismatch. The first operation raises a ValueError, not produces a shape.
- B) Correct — `a + b` fails (ValueError caught, prints "Error"). Then `b2 = b[:,None]` gives shape `(3,1)`. `a + b2`: `(3,4)` vs `(3,1)` → dim 0: 3 vs 3, dim 1: 4 vs 1 → 4. Result `(3,4)`. Prints "Fixed shape: (3, 4)".
- C) Incorrect — `b[:,None]` is `(3,1)` and the fixed result is `(3,4)`, not `(3,1)`. The 1 stretches to 4.
- D) Incorrect — `(3,)` cannot produce a `(3,3)` result with `(3,4)` since 4 ≠ 3 and neither is 1. The first operation always fails.

---

## Set 3 — Extended Practice

> Targets np.broadcast_to, in-place restrictions, haversine internals, 0-D arrays, np.broadcast_shapes, and advanced newaxis patterns.

---

## Q21 — np.broadcast_to Shape and Writability

```python
import numpy as np
a = np.ones((1, 4))
b = np.broadcast_to(a, (3, 4))
print(b.shape)
b[0, 0] = 99  # attempt to write
```

**What is the output?**

- A) `(3, 4)` then `b[0,0]` becomes `99`
- B) `(1, 4)` — shape is not updated by broadcast_to
- C) `(3, 4)` then `ValueError` is raised on the assignment
- D) `(3, 4)` then a `UserWarning` is emitted and the write is silently ignored

**Answer: C**

- A) Incorrect — `np.broadcast_to` returns a read-only view. The shape is correctly reported as `(3, 4)`, but any write attempt raises a `ValueError`, not silently succeeds.
- B) Incorrect — `np.broadcast_to(a, (3, 4))` does update the reported `.shape` to `(3, 4)`. The strides change (broadcast dim gets stride 0), but the shape attribute reflects the target shape.
- C) Correct — `b.shape` is `(3, 4)`. However, `b.flags.writeable` is `False`. Writing to any element raises `ValueError: assignment destination is read-only`. To get a writable copy call `b.copy()`.
- D) Incorrect — NumPy does not emit warnings for attempted writes to read-only arrays; it raises a hard `ValueError` immediately.

---

## Q22 — In-Place Add with Shape Expansion

```python
import numpy as np
a = np.ones((2, 3))
b = np.ones((4, 3))
try:
    a += b
    print("OK", a.shape)
except ValueError:
    print("Error")
```

**What is the output?**

- A) `OK (4, 3)`
- B) `OK (2, 3)`
- C) `Error`
- D) `OK (2, 4, 3)`

**Answer: C**

- A) Incorrect — in-place operations cannot change the shape of `a`. Even if NumPy could broadcast `(2,3)` and `(4,3)`, the result would need to fit back into `a`, which has only 2 rows.
- B) Incorrect — `(2,3)` and `(4,3)` are not broadcast-compatible at all: dim 0 is 2 vs 4, neither is 1. This is incompatible even for a regular (non-in-place) addition.
- C) Correct — `(2,3)` and `(4,3)` cannot be broadcast: dim 0 is 2 vs 4, both > 1 and unequal. NumPy raises `ValueError`. Even if they were compatible, an in-place op requires the output shape to equal `a`'s shape `(2,3)`.
- D) Incorrect — broadcasting never adds a dimension to the output unless the inputs have different ranks. Here both inputs are rank 2.

---

## Q23 — Haversine dsin2 Intermediate Shape

```python
import numpy as np

def dsin2_shape(p1, p2):
    p1 = np.radians(p1)
    p2 = np.radians(p2)
    dsin2 = np.sin(0.5 * (p1[:, None, :] - p2[None, :, :])) ** 2
    return dsin2.shape

p1 = np.random.rand(10, 2)
p2 = np.random.rand(15, 2)
print(dsin2_shape(p1, p2))
```

**What is the output?**

- A) `(10, 15)`
- B) `(10, 15, 2)`
- C) `(10, 2, 15)`
- D) `(15, 10, 2)`

**Answer: B**

- A) Incorrect — the coordinate axis (size 2) is not reduced at this stage. The subtraction is on the full `(lat, lon)` pair. The axis-2 reduction happens later when computing `a = dsin2[:,:,0] + cosprod * dsin2[:,:,1]`.
- B) Correct — `p1[:, None, :]` is `(10, 1, 2)` and `p2[None, :, :]` is `(1, 15, 2)`. Broadcasting: dim 0: 10 vs 1 → 10, dim 1: 1 vs 15 → 15, dim 2: 2 vs 2 → 2. `np.sin` is a ufunc that preserves shape. `** 2` also preserves shape. Result: `(10, 15, 2)`.
- C) Incorrect — the coordinate axis (size 2) is always the last axis in this indexing pattern. The `[:, None, :]` places the new axis in the middle, not at the end.
- D) Incorrect — the index order `p1[:, None, :]` makes p1's N the first axis and p2's N the second axis. The result is `(N_p1, N_p2, 2)` = `(10, 15, 2)`, not the transposed `(15, 10, 2)`.

---

## Q24 — distmat_1d Output Shape

```python
import numpy as np

def distmat_1d(x, y):
    return abs(x[:, None] - y)

x = np.array([1, 3])
y = np.array([1, 2, 3])
result = distmat_1d(x, y)
print(result.shape)
print(result[1, 2])
```

**What is the output?**

- A) `(2, 3)` then `0`
- B) `(2, 3)` then `2`
- C) `(3, 2)` then `0`
- D) `ValueError`

**Answer: A**

- A) Correct — `x[:,None]` is `(2,1)` and `y` is `(3,)` which pads to `(1,3)`. Broadcasting: `(2,1) - (1,3)` → `(2,3)`. `result[1, 2]` = `abs(x[1] - y[2])` = `abs(3 - 3)` = `0`.
- B) Incorrect — `result[1, 2]` = `abs(x[1] - y[2])` = `abs(3 - 3)` = 0, not 2. `result[0, 2]` would be `abs(1 - 3)` = 2.
- C) Incorrect — `x[:,None]` puts x's elements along axis 0 (rows) and y's elements along axis 1 (columns), giving shape `(len(x), len(y))` = `(2, 3)`. The transposed `(3,2)` would require `y[:,None] - x`.
- D) Incorrect — `(2,1)` and `(1,3)` are perfectly compatible for broadcasting.

---

## Q25 — 0-D Array Broadcasting Shape

```python
import numpy as np
scalar = np.float64(3.0)     # 0-D array, shape ()
arr    = np.ones((4, 5, 2))
result = arr * scalar
print(result.shape)
print(scalar.ndim)
```

**What is the output?**

- A) `(4, 5, 2)` then `0`
- B) `(1,)` then `1`
- C) `(4, 5, 2)` then `1`
- D) ValueError

**Answer: A**

- A) Correct — `np.float64(3.0)` is a 0-D NumPy scalar with shape `()` and `ndim == 0`. It broadcasts against any shape. The result shape equals `arr`'s shape: `(4, 5, 2)`. `scalar.ndim` is `0`.
- B) Incorrect — `np.float64(3.0)` has shape `()`, not `(1,)`. A 0-D array has no axes; `ndim` is 0, not 1. `np.array([3.0])` would have shape `(1,)` and `ndim == 1`.
- C) Incorrect — the result shape is correct at `(4,5,2)`, but `scalar.ndim` is 0, not 1. A 0-D array truly has zero dimensions.
- D) Incorrect — 0-D arrays are unconditionally compatible for broadcasting with any array shape. No error is raised.

---

## Q26 — np.broadcast_shapes ValueError

```python
import numpy as np
try:
    s = np.broadcast_shapes((4, 3), (3, 4))
    print(s)
except ValueError:
    print("Error")
```

**What is the output?**

- A) `(4, 4)`
- B) `(4, 3, 4)`
- C) `Error`
- D) `(3, 4)`

**Answer: C**

- A) Incorrect — for a `(4,4)` result both dim 0 and dim 1 would need to resolve to 4. Dim 0: 4 vs 3 — neither is 1, they are unequal. Incompatible.
- B) Incorrect — `np.broadcast_shapes` never increases the rank beyond the maximum of its inputs. Both inputs are rank 2 so the result (if valid) would also be rank 2.
- C) Correct — right-align (same rank): dim 0 is 4 vs 3 (incompatible), dim 1 is 3 vs 4 (incompatible). Both pairs fail the broadcast rule. `np.broadcast_shapes` raises `ValueError`. The trap: the shapes contain the same numbers (3 and 4) but in opposite orders, so every dimension pair is a mismatch.
- D) Incorrect — `(3,4)` would be the second input's shape. `np.broadcast_shapes` does not simply return one of the inputs; it raises an error when shapes are incompatible.

---

## Q27 — Stacked newaxis Reshape

```python
import numpy as np
x = np.arange(12).reshape(3, 4)
y = x[:, None, :, None]
print(y.shape)
print(y[2, 0, 3, 0])
```

**What is the output?**

- A) `(3, 4, 1, 1)` then `11`
- B) `(3, 1, 4, 1)` then `11`
- C) `(3, 1, 4, 1)` then `3`
- D) `(1, 3, 1, 4)` then `11`

**Answer: B**

- A) Incorrect — the `None`s are interleaved between the two slice positions, not appended. The pattern `[:, None, :, None]` inserts a size-1 axis after axis 0 and after axis 1 of the original array.
- B) Correct — `x` has shape `(3, 4)`. Indexing `[:, None, :, None]`: `:` → axis 0 (size 3), `None` → new axis (size 1), `:` → axis 1 (size 4), `None` → new axis (size 1). Shape: `(3, 1, 4, 1)`. `y[2, 0, 3, 0]` = `x[2, 3]` = element at row 2, col 3 of a 0-indexed (3,4) range array = `2*4 + 3` = `11`.
- C) Incorrect — the shape is correct at `(3,1,4,1)` but `y[2,0,3,0]` = `x[2,3]` = `11`, not `3`. `x[0,3]` would be `3`.
- D) Incorrect — `(1,3,1,4)` would require both `None`s to come before the slices: `[None, :, None, :]`. The given pattern interleaves `None` after each slice.

---

## Q28 — In-Place Op on Size-1 Leading Dim

```python
import numpy as np
a = np.ones((1, 5))
b = np.ones((3, 5))
try:
    a += b
    print("OK", a.shape)
except ValueError:
    print("Error")
```

**What is the output?**

- A) `OK (1, 5)` — in-place ops broadcast silently
- B) `OK (3, 5)` — `a` is resized to fit the broadcast result
- C) `Error`
- D) `OK (1, 5)` — only one row of `b` is used

**Answer: C**

- A) Incorrect — broadcasting `(1,5)` and `(3,5)` would give `(3,5)`. But `a` has shape `(1,5)` and an in-place op cannot resize it. NumPy raises a ValueError rather than silently truncating.
- B) Incorrect — NumPy never reallocates or resizes an array during in-place operations. The left-hand operand's shape is fixed; the result must fit exactly.
- C) Correct — `(1,5)` and `(3,5)` broadcast to `(3,5)`. Since `(3,5)` ≠ `(1,5)`, the in-place `+=` would need to expand `a`, which is not allowed. NumPy raises `ValueError: non-broadcastable output operand with shape (1,5) doesn't match the broadcast shape (3,5)`.
- D) Incorrect — NumPy does not partially apply broadcasts in in-place ops. The mismatch is detected and an error is raised immediately.

---

## Q29 — Three-Way Add with 0-D

```python
import numpy as np
a = np.ones((2, 3))
b = np.ones((3,))
c = np.float64(10.0)
result = a + b + c
print(result.shape)
print(result[0, 0])
```

**What is the output?**

- A) `(2, 3)` then `12.0`
- B) `(3,)` then `12.0`
- C) `(2, 3)` then `11.0`
- D) ValueError

**Answer: A**

- A) Correct — evaluate left-to-right. First `a + b`: `(2,3)` and `(3,)` pads to `(1,3)` → `(2,3)`. All elements are 2.0. Then `(2,3) + c` where `c` is 0-D: 0-D broadcasts to `(2,3)`. All elements become `2.0 + 10.0 = 12.0`. Shape: `(2,3)`.
- B) Incorrect — the result shape is `(2,3)`, not `(3,)`. The addition with `a` (shape `(2,3)`) expands the result to `(2,3)`.
- C) Incorrect — the value is 12.0, not 11.0. `a` has all ones, `b` has all ones, and `c` is 10.0. `1 + 1 + 10 = 12`.
- D) Incorrect — all three operands are mutually broadcast-compatible. No ValueError occurs at any step.

---

## Q30 — np.outer vs Broadcasting Outer Product

```python
import numpy as np
u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0])

r1 = np.outer(u, v)
r2 = u[:, None] * v[None, :]

print(r1.shape, r2.shape)
print(np.array_equal(r1, r2))
```

**What is the output?**

- A) `(3, 2) (3, 2)` then `True`
- B) `(3, 2) (3, 2)` then `False`
- C) `(6,) (3, 2)` then `False`
- D) `(2, 3) (3, 2)` then `False`

**Answer: A**

- A) Correct — `np.outer(u, v)` computes the outer product, giving shape `(3, 2)` with `r1[i,j] = u[i] * v[j]`. `u[:,None]` is `(3,1)` and `v[None,:]` is `(1,2)`. Broadcasting: `(3,1) * (1,2)` → `(3,2)` with the same element formula. Both shapes are `(3,2)` and the values are identical, so `np.array_equal` returns `True`.
- B) Incorrect — `np.outer` and the broadcasting outer product produce numerically identical results. `np.array_equal` returns `True`.
- C) Incorrect — `np.outer` always returns a 2D array of shape `(len(u), len(v))`. It does not flatten to 1D. `(6,)` would require flattening, which `np.outer` does not do to the output.
- D) Incorrect — `np.outer(u, v)` has shape `(len(u), len(v))` = `(3, 2)`, not the transposed `(2, 3)`. The first argument's length is always the number of rows.

---
