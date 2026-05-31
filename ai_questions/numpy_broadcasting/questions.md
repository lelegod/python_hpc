# NumPy Broadcasting — MCQ Practice

> Topics: Shape alignment rules, None/newaxis insertion, output shape derivation, common patterns.
> Exam frequency: **2024 exam + re-exam** — tested multiple times.

---

## Q1 — Output Shape: (3,) + (2,3)
> **Week reference:** Week 4

What is the output shape when you add arrays with shapes `(3,)` and `(2,3)`?

- A) ValueError — shapes are incompatible
- B) `(2,3)`
- C) `(3,3)`
- D) `(2,)`

**Answer: B**

- A) Incorrect — right-aligning `(3,)` as `(1,3)` and comparing with `(2,3)` gives dims `(2,3)`, which is valid.
- B) Correct — `(3,)` is left-padded to `(1,3)`; dims 2 vs 1 → 2, dims 3 vs 3 → 3; result is `(2,3)`.
- C) Incorrect — output shape is the element-wise max of each dimension pair, not a product or sum.
- D) Incorrect — the second dimension (3) is preserved; the output cannot lose that axis.

---

## Q2 — Error Case: (2,3) + (2,)
> **Week reference:** Week 4

What happens when you add arrays with shapes `(2,3)` and `(2,)`?

- A) Output shape is `(2,3)` — the first dimension matches.
- B) Output shape is `(2,2)`.
- C) ValueError — the shapes are incompatible after alignment.
- D) Output shape is `(2,)`.

**Answer: C**

- A) Incorrect — broadcasting aligns from the right, not the left; right-aligning gives dimension mismatch 3 vs 2.
- B) Incorrect — no valid broadcast produces a `(2,2)` result here.
- C) Correct — right-aligned: `(2,3)` vs `(1,2)`; last dims 3 vs 2, neither is 1 → ValueError.
- D) Incorrect — you cannot reduce to a smaller shape through broadcasting addition.

---

## Q3 — Output Shape: (5,1,3) + (4,3)
> **Week reference:** Week 4

What is the output shape when you add arrays with shapes `(5,1,3)` and `(4,3)`?

- A) `(5,4,3)`
- B) `(5,1,3)`
- C) ValueError — incompatible shapes
- D) `(5,4,6)`

**Answer: A**

- A) Correct — `(4,3)` is left-padded to `(1,4,3)`; dims: 5 vs 1 → 5, 1 vs 4 → 4, 3 vs 3 → 3; result `(5,4,3)`.
- B) Incorrect — the middle axis of size 1 expands to 4, not stays at 1.
- C) Incorrect — all dimension pairs are compatible (each pair is equal or one is 1).
- D) Incorrect — output shape takes the max per dimension, not the sum; the last dim stays 3.

---

## Q4 — Image Batch Mean Subtraction
> **Week reference:** Week 4

You have an array `images` with shape `(N, H, W, 3)` and per-image mean pixel values `mean_pixels` with shape `(N, 3)`. Which expression correctly subtracts the per-image mean from every spatial location?

- A) `images - mean_pixels`
- B) `images - mean_pixels[None, :]`
- C) `images - mean_pixels[:, None, None, :]`
- D) `images - mean_pixels[:, None, :]`

**Answer: C**

- A) Incorrect — right-aligning `(N,3)` with `(N,H,W,3)` gives dims `W vs N` mismatched at position 2.
- B) Incorrect — `mean_pixels[None,:]` gives shape `(1,N,3)`, which does not align with `(N,H,W,3)`.
- C) Correct — `mean_pixels[:,None,None,:]` gives shape `(N,1,1,3)`, which broadcasts over H and W correctly.
- D) Incorrect — `mean_pixels[:,None,:]` gives shape `(N,1,3)`, missing one spatial dimension.

---

## Q5 — None/newaxis Effect on Shape
> **Week reference:** Week 4

Given `x` with shape `(n,)`, what are the shapes of `x[:, None]` and `x[None, :]` respectively?

- A) Both remain `(n,)`
- B) `(n, 1)` and `(1, n)`
- C) `(1, n)` and `(n, 1)`
- D) `(n, n)` and `(1, 1)`

**Answer: B**

- A) Incorrect — inserting `None` adds a new axis of size 1, changing the shape.
- B) Correct — `x[:,None]` inserts a new axis after the first, giving `(n,1)`; `x[None,:]` inserts before, giving `(1,n)`.
- C) Incorrect — these are swapped; `None` after `:` appends the axis, `None` before prepends it.
- D) Incorrect — `None` inserts a size-1 axis; it does not tile or repeat values.

---

## Q6 — Outer Product via Broadcasting
> **Week reference:** Week 4

You have `x` with shape `(5,)` and `y` with shape `(3,)`. What is the shape of `x[:, None] * y`?

- A) `(5,)`
- B) `(3,)`
- C) `(5, 3)`
- D) ValueError — incompatible shapes

**Answer: C**

- A) Incorrect — the result must contain contributions from both dimensions.
- B) Incorrect — the result must contain contributions from both dimensions.
- C) Correct — `x[:,None]` has shape `(5,1)`, `y` is treated as `(1,3)`; broadcasting gives `(5,3)`, the outer product.
- D) Incorrect — `(5,1)` and `(1,3)` are fully compatible; every dim pair has at least one 1.

---

## Q7 — Four-Dimensional Broadcasting
> **Week reference:** Week 4

What is the output shape when adding arrays with shapes `(100, 1, 6, 3)` and `(100, 1, 3)`?

- A) `(100, 100, 6, 3)`
- B) `(100, 1, 6, 3)`
- C) `(100, 6, 3)`
- D) ValueError — incompatible shapes

**Answer: A**

- A) Correct — `(100,1,3)` is left-padded to `(1,100,1,3)`; dims: 100 vs 1 → 100, 1 vs 100 → 100, 6 vs 1 → 6, 3 vs 3 → 3; result `(100,100,6,3)`.
- B) Incorrect — the second axis of the left array is 1 and the second axis of the right (after padding) is 100, so it expands.
- C) Incorrect — left-padding adds a leading 1 to `(100,1,3)`, creating a new axis mismatch that broadcasts rather than collapsing.
- D) Incorrect — all dimension pairs are compatible after left-padding.

---

## Q8 — Subtracting a Minimum Image
> **Week reference:** Week 4

You have `images` with shape `(N, H, W, 3)` and `mim` (a minimum image) with shape `(H, W)`. You want to subtract `mim` from every color channel of every image. Which reshape/indexing gives the correct broadcast?

- A) `mim[None, :, :]` — shape `(1, H, W)`
- B) `mim[None, :, None]` — shape `(1, H, 1)`
- C) `mim[None, :, :, None]` — shape `(1, H, W, 1)`
- D) `mim` as-is — shape `(H, W)`

**Answer: C**

- A) Incorrect — `(1,H,W)` right-aligns as `(1,1,H,W)` against `(N,H,W,3)`; last dim H vs W (different) and W vs 3 — incompatible unless H==W==3.
- B) Incorrect — `(1,H,1)` right-aligns as `(1,1,H,1)` against `(N,H,W,3)`; second-to-last dim H vs W is incompatible (H≠W in general).
- C) Correct — `mim[None,:,:,None]` gives `(1,H,W,1)`; broadcasting against `(N,H,W,3)`: 1→N, H→H, W→W, 1→3 — all dims compatible.
- D) Incorrect — `(H,W)` right-aligns as `(1,1,H,W)` against `(N,H,W,3)`; last dim W vs 3 is incompatible (unless W==3 by coincidence).

---

## Q9 — Scalar Broadcasting
> **Week reference:** Week 4

You have `arr` with shape `(4, 7, 2)`. What is the shape of `arr + 5`?

- A) `(1,)`
- B) `(4, 7, 2)` — the scalar broadcasts to every element
- C) ValueError — a scalar cannot be broadcast with a 3-D array
- D) `(4, 7, 5)`

**Answer: B**

- A) Incorrect — a scalar has effective shape `()`, which broadcasts to any shape without reducing it.
- B) Correct — a scalar is treated as a 0-D array and broadcasts to match any array shape; output shape equals the array shape.
- C) Incorrect — scalar broadcasting is always valid regardless of array dimensionality.
- D) Incorrect — the scalar value 5 does not affect the shape; it is added element-wise.

---

## Q10 — Three-Dimensional Middle-Axis Broadcast
> **Week reference:** Week 4

What is the output shape when you add arrays with shapes `(4, 1, 3)` and `(1, 5, 3)`?

- A) `(4, 5, 3)`
- B) `(4, 1, 3)`
- C) `(1, 5, 3)`
- D) `(4, 5, 6)`

**Answer: A**

- A) Correct — dims: 4 vs 1 → 4, 1 vs 5 → 5, 3 vs 3 → 3; result `(4,5,3)`.
- B) Incorrect — the middle axis of size 1 expands to match the other array's middle axis of size 5.
- C) Incorrect — the first axis of size 1 expands to match the other array's first axis of size 4.
- D) Incorrect — the last axis stays 3 (equal in both); it does not sum to 6.

---

## Q11 — Subtracting Row Means from a Matrix
> **Week reference:** Week 4

You have a data matrix `data` with shape `(n, d)` and its column-wise mean `mean` with shape `(d,)`. Which expression correctly subtracts the mean from every row?

- A) `data - mean[:, None]`
- B) `data - mean`
- C) `data - mean[None, :]`
- D) Both B and C are correct

**Answer: D**

- A) Incorrect — `mean[:,None]` gives shape `(d,1)`; right-aligned against `(n,d)` gives dims n vs d (likely mismatched) and d vs 1.
- B) Correct — `(n,d) - (d,)`: right-align pads `(d,)` to `(1,d)`, then dims n vs 1 → n, d vs d → d; result `(n,d)`.
- C) Correct — `mean[None,:]` explicitly gives shape `(1,d)`, which broadcasts over the n rows identically to option B.
- D) Correct — B and C produce identical results; both are valid and commonly used idioms.

---

## Q12 — Right-Align Error vs. Valid Case
> **Week reference:** Week 4

Which of the following broadcasting operations raises a `ValueError`?

- A) `np.ones((3,4)) + np.ones((4,))`
- B) `np.ones((3,4)) + np.ones((3,))`
- C) `np.ones((3,4)) + np.ones((1,4))`
- D) `np.ones((3,4)) + np.ones((3,1))`

**Answer: B**

- A) Incorrect (valid) — `(4,)` right-aligns as `(1,4)` against `(3,4)`; dims 3 vs 1 → 3, 4 vs 4 → 4; result `(3,4)`.
- B) Correct (raises ValueError) — `(3,)` right-aligns as `(1,3)` against `(3,4)`; last dims 4 vs 3, neither is 1 → incompatible.
- C) Incorrect (valid) — `(1,4)` vs `(3,4)`; dims 3 vs 1 → 3, 4 vs 4 → 4; result `(3,4)`.
- D) Incorrect (valid) — `(3,1)` vs `(3,4)`; dims 3 vs 3 → 3, 1 vs 4 → 4; result `(3,4)`.

---
