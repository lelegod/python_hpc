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
