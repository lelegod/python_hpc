# NumPy Broadcasting — MCQ Practice

> Topics: Shape alignment rules, None/newaxis insertion, output shape derivation, common patterns.
> Exam frequency: **2024 exam + re-exam** — tested multiple times.

---

## Q1 — Output Shape: (3,) + (2,3)
> **Week reference:** Week 4

**Mental Model:** Right-align shapes and left-pad with 1s. Then apply the rule: each dimension pair must be (equal) OR (one is 1). The output dim = max of each pair. Never add or multiply dims — just take the max.

What is the output shape when you add arrays with shapes `(3,)` and `(2,3)`?

- A) ValueError — shapes are incompatible
- B) `(2,3)`
- C) `(3,3)`
- D) `(2,)`

**Answer: B**

- A) Incorrect — right-aligning `(3,)` as `(1,3)` and comparing with `(2,3)` gives pairs (1, 2) and (3, 3). Both pairs satisfy the rule: one is 1 for the first pair, equal for the second. No error occurs.
- B) Correct — after left-padding: `(1,3)` vs `(2,3)`. Dim 0: max(1,2) = 2 (the 1 broadcasts). Dim 1: max(3,3) = 3 (equal, no stretch needed). Result: `(2,3)`. The `(3,)` array is conceptually tiled into 2 rows.
- C) Incorrect — the output shape takes the element-wise max of each dimension pair, not a product or sum. There is no way to get a 3 in the first dimension when the inputs only provide 1 and 2 for that axis.
- D) Incorrect — broadcasting never reduces a dimension. The result must cover all elements of both inputs; collapsing the second dimension (3) to nothing would lose data.

---

## Q2 — Error Case: (2,3) + (2,)
> **Week reference:** Week 4

**Mental Model:** Broadcasting aligns from the RIGHT, not the left. `(2,)` right-aligns to `(1,2)`, making the last dims 3 vs 2 — neither is 1, so it fails. The trap is assuming "the 2s match each other" (that's a left-alignment intuition, which is wrong).

What happens when you add arrays with shapes `(2,3)` and `(2,)`?

- A) Output shape is `(2,3)` — the first dimension matches.
- B) Output shape is `(2,2)`.
- C) ValueError — the shapes are incompatible after alignment.
- D) Output shape is `(2,)`.

**Answer: C**

- A) Incorrect — this reasoning uses left-alignment ("the leading 2s match"), which is NOT how NumPy works. NumPy always aligns from the right. After right-alignment: `(2,3)` vs `(1,2)`. Last dims: 3 vs 2 — neither is 1 → incompatible.
- B) Incorrect — there is no broadcasting rule that produces a `(2,2)` result from these inputs. The last-dim mismatch (3 vs 2) is a hard error, not a shape-morphing operation.
- C) Correct — right-align: `(2,3)` vs `(1,2)`. Compare last dims: 3 vs 2. Neither is 1, and they are not equal → NumPy raises `ValueError: operands could not be broadcast together with shapes (2,3) (2,)`. This is silent only if you mistakenly think of left-alignment.
- D) Incorrect — broadcasting never reduces shapes; the output can only be equal to or larger than each input in every dimension. You cannot broadcast down to `(2,)` from a `(2,3)` input.

---

## Q3 — Output Shape: (5,1,3) + (4,3)
> **Week reference:** Week 4

**Mental Model:** Left-pad the shorter shape to match rank, then go dimension by dimension. `(4,3)` → `(1,4,3)`. Then: 5 vs 1 → 5, 1 vs 4 → 4, 3 vs 3 → 3. Result: `(5,4,3)`. The 1s in the original arrays are the "stretch points."

What is the output shape when you add arrays with shapes `(5,1,3)` and `(4,3)`?

- A) `(5,4,3)`
- B) `(5,1,3)`
- C) ValueError — incompatible shapes
- D) `(5,4,6)`

**Answer: A**

- A) Correct — `(4,3)` is left-padded to `(1,4,3)`. Now compare: dim 0: 5 vs 1 → 5 (the 1 stretches to 5). Dim 1: 1 vs 4 → 4 (the 1 in the left array stretches to 4). Dim 2: 3 vs 3 → 3 (equal). Output: `(5,4,3)`. Both arrays needed a dimension stretched: `(5,1,3)` stretches in dim 1, `(1,4,3)` stretches in dim 0.
- B) Incorrect — the middle axis of size 1 does not stay 1; it expands to match the 4 in the right array. A size-1 axis always stretches to match the other array's size for that dimension.
- C) Incorrect — all three dimension pairs are compatible: (5,1), (1,4), and (3,3). Each pair either has a 1 or is equal. No ValueError is raised.
- D) Incorrect — the last dimension stays 3 because both arrays have exactly 3 there (3 vs 3 = 3). Broadcasting outputs the max per dim, not the sum. Getting 6 would require one array to have shape 3 and the other shape 6 with a 1 somewhere else.

---

## Q4 — Image Batch Mean Subtraction
> **Week reference:** Week 4

**Mental Model:** You need `mean_pixels` to have shape `(N,1,1,3)` to broadcast against `(N,H,W,3)`. Insert two `None`s to create the two size-1 spatial axes. One `None` → one missing dim → still incompatible.

You have an array `images` with shape `(N, H, W, 3)` and per-image mean pixel values `mean_pixels` with shape `(N, 3)`. Which expression correctly subtracts the per-image mean from every spatial location?

- A) `images - mean_pixels`
- B) `images - mean_pixels[None, :]`
- C) `images - mean_pixels[:, None, None, :]`
- D) `images - mean_pixels[:, None, :]`

**Answer: C**

- A) Incorrect — right-aligning `(N,3)` against `(N,H,W,3)`: the last dim pairs are 3 vs 3 (ok), W vs N (mismatch unless W==N by coincidence), H vs — (missing). NumPy raises a ValueError for the W vs N conflict.
- B) Incorrect — `mean_pixels[None,:]` inserts a leading axis, giving shape `(1,N,3)`. Right-aligned against `(N,H,W,3)`: dims are 3 vs 3 (ok), N vs W (mismatch), 1 vs H (ok), — vs N (missing). Still incompatible.
- C) Correct — `mean_pixels[:,None,None,:]` gives shape `(N,1,1,3)`. Right-aligned against `(N,H,W,3)`: 3 vs 3 (equal), 1 vs W → W (stretches), 1 vs H → H (stretches), N vs N (equal). Result: `(N,H,W,3)`. Every pixel in image n gets the mean for image n subtracted, correctly.
- D) Incorrect — `mean_pixels[:,None,:]` gives shape `(N,1,3)`. Right-aligned against `(N,H,W,3)`: 3 vs 3 (ok), 1 vs W → W (ok), N vs H (mismatch unless N==H). Missing one `None` for the width axis.

---

## Q5 — None/newaxis Effect on Shape
> **Week reference:** Week 4

**Mental Model:** `None` (alias `np.newaxis`) inserts a size-1 axis at exactly the position where you write it. After the colon = appended axis. Before the colon = prepended axis. It never copies data — it just reinterprets the shape.

Given `x` with shape `(n,)`, what are the shapes of `x[:, None]` and `x[None, :]` respectively?

- A) Both remain `(n,)`
- B) `(n, 1)` and `(1, n)`
- C) `(1, n)` and `(n, 1)`
- D) `(n, n)` and `(1, 1)`

**Answer: B**

- A) Incorrect — `None` always adds a new axis of size 1, changing the shape. If shape remained unchanged, `None` would be useless. The number of dimensions increases by 1 for each `None` inserted.
- B) Correct — `x[:, None]` selects all elements of the existing axis (`:`) and then inserts a new axis of size 1, giving `(n, 1)`. `x[None, :]` inserts a new axis before the existing axis, giving `(1, n)`. These are the column-vector and row-vector representations of a 1D array, enabling outer-product-style broadcasting.
- C) Incorrect — these are the swapped versions. The position of `None` in the index expression directly determines where the new axis appears: `None` before `:` means the new axis is first (prepended); `None` after `:` means the new axis is last (appended).
- D) Incorrect — `None` inserts a size-1 axis; it does not tile or replicate values. The result has exactly n elements in the original axis and 1 in the new axis — not n×n total elements.

---

## Q6 — Outer Product via Broadcasting
> **Week reference:** Week 4

**Mental Model:** To compute an outer product of shapes `(m,)` and `(n,)`, reshape one to `(m,1)` and let the other broadcast as `(1,n)`. The `(m,1)` × `(1,n)` = `(m,n)` pattern is the canonical broadcasting outer product.

You have `x` with shape `(5,)` and `y` with shape `(3,)`. What is the shape of `x[:, None] * y`?

- A) `(5,)`
- B) `(3,)`
- C) `(5, 3)`
- D) ValueError — incompatible shapes

**Answer: C**

- A) Incorrect — the result must incorporate contributions from both `x` (5 elements) and `y` (3 elements). A shape of `(5,)` would only reflect `x` and ignore the 3-element axis of `y`, which cannot happen through broadcasting multiplication.
- B) Incorrect — similarly, `(3,)` only reflects `y` and discards the 5-element axis of `x`. Broadcasting expands dimensions, it never silently collapses them.
- C) Correct — `x[:,None]` has shape `(5,1)`. `y` is implicitly treated as `(1,3)` (left-padded with a 1). Broadcasting: dim 0: 5 vs 1 → 5; dim 1: 1 vs 3 → 3. Output: `(5,3)`. This computes the outer product: `result[i, j] = x[i] * y[j]` for all i,j.
- D) Incorrect — `(5,1)` and `(1,3)` are fully compatible: every dimension pair has at least one 1 (dim 0: 5 vs 1, dim 1: 1 vs 3). The rule is: incompatible only when both sizes are >1 AND unequal.

---

## Q7 — Four-Dimensional Broadcasting
> **Week reference:** Week 4

**Mental Model:** Left-pad `(100,1,3)` to `(1,100,1,3)`. Then go dim by dim: 100 vs 1 → 100, 1 vs 100 → 100, 6 vs 1 → 6, 3 vs 3 → 3. The trap here is forgetting that left-padding adds a NEW leading axis that pairs with the existing leading axis of the other array.

What is the output shape when adding arrays with shapes `(100, 1, 6, 3)` and `(100, 1, 3)`?

- A) `(100, 100, 6, 3)`
- B) `(100, 1, 6, 3)`
- C) `(100, 6, 3)`
- D) ValueError — incompatible shapes

**Answer: A**

- A) Correct — `(100,1,3)` is left-padded to `(1,100,1,3)`. Now compare with `(100,1,6,3)`: dim 0: 100 vs 1 → 100; dim 1: 1 vs 100 → 100; dim 2: 6 vs 1 → 6; dim 3: 3 vs 3 → 3. Result: `(100,100,6,3)`. The seemingly innocuous padding of a leading 1 causes the second array to broadcast across the first dimension, creating a 100× expansion in that axis.
- B) Incorrect — the second axis of `(100,1,6,3)` is 1, and the second axis of the padded array `(1,100,1,3)` is 100. A 1 always expands to match its partner, so axis 1 becomes 100, not stays 1.
- C) Incorrect — you cannot arrive at `(100,6,3)` because left-padding `(100,1,3)` adds a new leading dimension, not merges with existing ones. Broadcasting adds dimensions but never reduces rank.
- D) Incorrect — every dimension pair after padding is compatible: (100,1), (1,100), (6,1), (3,3). Each pair satisfies the rule (equal or one is 1). No error is raised.

---

## Q8 — Subtracting a Minimum Image
> **Week reference:** Week 4

**Mental Model:** Target shape is `(N,H,W,3)`. You need to add axes to `(H,W)` to reach rank 4, and the added axes must be 1 so they broadcast. The axes for N (batch) and 3 (channels) need to be 1 → `(1,H,W,1)`.

You have `images` with shape `(N, H, W, 3)` and `mim` (a minimum image) with shape `(H, W)`. You want to subtract `mim` from every color channel of every image. Which reshape/indexing gives the correct broadcast?

- A) `mim[None, :, :]` — shape `(1, H, W)`
- B) `mim[None, :, None]` — shape `(1, H, 1)`
- C) `mim[None, :, :, None]` — shape `(1, H, W, 1)`
- D) `mim` as-is — shape `(H, W)`

**Answer: C**

- A) Incorrect — `(1,H,W)` is left-padded to `(1,1,H,W)` when compared to `(N,H,W,3)`. Pairing: 1 vs N → N (ok), 1 vs H → H (ok), H vs W (mismatch unless H==W), W vs 3 (mismatch unless W==3). Two potential mismatches; this fails for non-square, non-RGB images.
- B) Incorrect — `(1,H,1)` is left-padded to `(1,1,H,1)`. Pairing: 1 vs N → N (ok), 1 vs H → H (ok), H vs W (mismatch unless H==W), 1 vs 3 → 3 (ok). Still fails on the H vs W mismatch for non-square images.
- C) Correct — `mim[None,:,:,None]` gives shape `(1,H,W,1)`. Pairing against `(N,H,W,3)`: 1 vs N → N (broadcasts over batch); H vs H → H (exact match); W vs W → W (exact match); 1 vs 3 → 3 (broadcasts over channels). Result: `(N,H,W,3)`. Every pixel `(n,h,w,c)` has `mim[h,w]` subtracted, achieving the intended per-spatial-location minimum subtraction across all images and channels.
- D) Incorrect — `(H,W)` is left-padded to `(1,1,H,W)`. Pairing: 1 vs N → N (ok), 1 vs H → H (ok), H vs W (mismatch for non-square images), W vs 3 (mismatch for images where W ≠ 3). Fails in the general case.

---

## Q9 — Scalar Broadcasting
> **Week reference:** Week 4

**Mental Model:** A Python scalar or 0-D array has shape `()` — it broadcasts to any shape. Think of it as a single value that gets stamped onto every element. No reshape needed; it always works.

You have `arr` with shape `(4, 7, 2)`. What is the shape of `arr + 5`?

- A) `(1,)`
- B) `(4, 7, 2)` — the scalar broadcasts to every element
- C) ValueError — a scalar cannot be broadcast with a 3-D array
- D) `(4, 7, 5)`

**Answer: B**

- A) Incorrect — a scalar broadcasts outward to the array's shape, it does not reduce the array to a single element. Adding 5 to every element of a `(4,7,2)` array produces a `(4,7,2)` result, not a `(1,)` scalar.
- B) Correct — a scalar is treated as a 0-D array with shape `()`. NumPy left-pads `()` with as many 1s as needed to match the array's rank: effectively `(1,1,1)` here. Each size-1 dim broadcasts to match `(4,7,2)`. Result shape = `(4,7,2)`, with 5 added to all 56 elements.
- C) Incorrect — scalar broadcasting is unconditionally valid for any array shape, regardless of dimensionality. There is no rank limit that prevents scalar operations.
- D) Incorrect — the scalar value 5 is added to each element but has absolutely no effect on the shape. The shape is determined by the array operand; the `5` in `arr + 5` is a value, not a dimension specification.

---

## Q10 — Three-Dimensional Middle-Axis Broadcast
> **Week reference:** Week 4

**Mental Model:** Same rank, so no left-padding needed. Just go dim by dim: 4 vs 1 → 4, 1 vs 5 → 5, 3 vs 3 → 3. Both arrays contribute one "stretchy" dimension each; they stretch independently.

What is the output shape when you add arrays with shapes `(4, 1, 3)` and `(1, 5, 3)`?

- A) `(4, 5, 3)`
- B) `(4, 1, 3)`
- C) `(1, 5, 3)`
- D) `(4, 5, 6)`

**Answer: A**

- A) Correct — both arrays have rank 3, so no padding needed. Dim 0: 4 vs 1 → 4 (the 1 in `(1,5,3)` stretches to 4). Dim 1: 1 vs 5 → 5 (the 1 in `(4,1,3)` stretches to 5). Dim 2: 3 vs 3 → 3 (equal, no stretch). Result: `(4,5,3)`, containing all 60 combinations.
- B) Incorrect — the middle axis of size 1 in `(4,1,3)` expands to match the 5 in `(1,5,3)`. Size-1 axes are always the ones that stretch to meet their partner; they never stay at 1 in the output.
- C) Incorrect — the first axis of size 1 in `(1,5,3)` expands to match the 4 in `(4,1,3)`. Same logic: both 1s stretch, both non-1s are preserved.
- D) Incorrect — the last dimension stays at 3 because both arrays have exactly 3 there. Broadcasting outputs the maximum per dim; since max(3,3) = 3, there is no reason for the last dim to become 6.

---

## Q11 — Subtracting Row Means from a Matrix
> **Week reference:** Week 4

**Mental Model:** `(n,d) - (d,)`: right-align → `(n,d)` vs `(1,d)` → valid, result `(n,d)`. The `(d,)` broadcasts over the n rows. Adding `[None,:]` makes the `(1,d)` explicit but produces the same result. Both idioms are correct.

You have a data matrix `data` with shape `(n, d)` and its column-wise mean `mean` with shape `(d,)`. Which expression correctly subtracts the mean from every row?

- A) `data - mean[:, None]`
- B) `data - mean`
- C) `data - mean[None, :]`
- D) Both B and C are correct

**Answer: D**

- A) Incorrect — `mean[:,None]` reshapes `(d,)` to `(d,1)`. Right-aligned against `(n,d)`: dim 0 is d vs n (incompatible unless d==n), dim 1 is 1 vs d → d. This subtracts a column vector from a matrix, which is the wrong operation and raises ValueError in the general case (n ≠ d).
- B) Correct — `(n,d) - (d,)`: NumPy right-pads `(d,)` to `(1,d)`. Pairing: dim 0: n vs 1 → n (mean row broadcasts over all n rows); dim 1: d vs d → d (direct subtraction). Each row `data[i,:]` has the same `mean` subtracted. Result: `(n,d)`.
- C) Correct — `mean[None,:]` explicitly creates shape `(1,d)`. The broadcasting arithmetic is identical to option B: dim 0 stretches from 1 to n, dim 1 matches at d. Both B and C subtract the same mean from every row; they differ only in explicitness.
- D) Correct — B and C produce identical numerical results. Option B relies on implicit left-padding (common in practice), while C makes the broadcast intent explicit (preferred for readability). Both are standard idioms.

---

## Q12 — Right-Align Error vs. Valid Case
> **Week reference:** Week 4

**Mental Model:** The trap question: `(3,4) + (3,)` looks like "the 3s match" but right-alignment puts `(3,)` → `(1,3)`, making the last-dim comparison 4 vs 3 — fail. The dangerous intuition is left-to-left matching; always right-align first.

Which of the following broadcasting operations raises a `ValueError`?

- A) `np.ones((3,4)) + np.ones((4,))`
- B) `np.ones((3,4)) + np.ones((3,))`
- C) `np.ones((3,4)) + np.ones((1,4))`
- D) `np.ones((3,4)) + np.ones((3,1))`

**Answer: B**

- A) Incorrect (valid) — `(4,)` right-aligns as `(1,4)` against `(3,4)`. Dim 0: 3 vs 1 → 3 (broadcasts). Dim 1: 4 vs 4 → 4 (equal). Result: `(3,4)`. This works because the 4 aligns perfectly with the last dimension of `(3,4)`.
- B) Correct (raises ValueError) — `(3,)` right-aligns as `(1,3)` against `(3,4)`. Last dims: 4 vs 3 — neither is 1 and they are unequal → `ValueError: operands could not be broadcast together with shapes (3,4) (3,)`. This is the classic trap: the 3 in `(3,)` appears to "match" the 3 in `(3,4)`, but right-alignment puts it against the 4.
- C) Incorrect (valid) — `(1,4)` vs `(3,4)`. Dim 0: 3 vs 1 → 3 (broadcasts). Dim 1: 4 vs 4 → 4 (equal). Result: `(3,4)`. Explicitly adding the leading 1 makes the broadcast intention clear.
- D) Incorrect (valid) — `(3,1)` vs `(3,4)`. Dim 0: 3 vs 3 → 3 (equal). Dim 1: 1 vs 4 → 4 (broadcasts). Result: `(3,4)`. This is the column-vector-times-matrix pattern, stretching the single column across all 4 columns.

---
