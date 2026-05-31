# NumPy Broadcasting — Code-Based MCQ Practice

> Format: Each question shows NumPy code to evaluate — predict output shape or error.
> Exam frequency: **2024 exam + re-exam**.

---

## Q1

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

## Q2

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

## Q3

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

## Q4

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

## Q5

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

## Q6

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

## Q7

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

## Q8

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

## Q9

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

## Q10

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
