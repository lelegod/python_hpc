# NumPy Broadcasting — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Output Shape: (3,) + (2,3)](#q1-output-shape-3-23)
- [Q2 — Error Case: (2,3) + (2,)](#q2-error-case-23-2)
- [Q3 — Output Shape: (5,1,3) + (4,3)](#q3-output-shape-513-43)
- [Q4 — Image Batch Mean Subtraction](#q4-image-batch-mean-subtraction)
- [Q5 — None/newaxis Effect on Shape](#q5-nonenewaxis-effect-on-shape)
- [Q6 — Outer Product via Broadcasting](#q6-outer-product-via-broadcasting)
- [Q7 — Four-Dimensional Broadcasting](#q7-four-dimensional-broadcasting)
- [Q8 — Subtracting a Minimum Image](#q8-subtracting-a-minimum-image)
- [Q9 — Scalar Broadcasting](#q9-scalar-broadcasting)
- [Q10 — Three-Dimensional Middle-Axis Broadcast](#q10-three-dimensional-middle-axis-broadcast)
- [Q11 — Subtracting Row Means from a Matrix](#q11-subtracting-row-means-from-a-matrix)
- [Q12 — Right-Align Error vs. Valid Case](#q12-right-align-error-vs-valid-case)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q13 — Shape Compatibility: (5,3,4) + (3,)](#q13-shape-compatibility-534-3)
- [Q14 — newaxis to Fix a Column Subtraction](#q14-newaxis-to-fix-a-column-subtraction)
- [Q15 — Pairwise Euclidean Distance Shape](#q15-pairwise-euclidean-distance-shape)
- [Q16 — Broadcasting Copies Data?](#q16-broadcasting-copies-data)
- [Q17 — np.broadcast_shapes Result](#q17-npbroadcast_shapes-result)
- [Q18 — Which Shape Pair is NOT Compatible?](#q18-which-shape-pair-is-not-compatible)
- [Q19 — np.expand_dims Equivalence](#q19-npexpand_dims-equivalence)
- [Q20 — Haversine All-Pairs Pattern](#q20-haversine-all-pairs-pattern)
- [Q21 — Three-Way Broadcasting](#q21-three-way-broadcasting)
- [Q22 — Fixing a Row-Mean Subtraction Bug](#q22-fixing-a-row-mean-subtraction-bug)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q23 — In-Place Broadcasting Restriction](#q23--in-place-broadcasting-restriction)
- [Q24 — np.broadcast_to Read-Only View](#q24--npbroadcast_to-read-only-view)
- [Q25 — Zero-Dimensional Array Broadcasting](#q25--zero-dimensional-array-broadcasting)
- [Q26 — Which newaxis Pattern Computes a 2D Outer Product?](#q26--which-newaxis-pattern-computes-a-2d-outer-product)
- [Q27 — Haversine cosprod Intermediate Shape](#q27--haversine-cosprod-intermediate-shape)
- [Q28 — ufunc Broadcasting vs Explicit Loop](#q28--ufunc-broadcasting-vs-explicit-loop)
- [Q29 — np.broadcast_shapes with Incompatible Inputs](#q29--npbroadcast_shapes-with-incompatible-inputs)
- [Q30 — Stacking newaxis at Different Positions](#q30--stacking-newaxis-at-different-positions)
- [Q31 — In-Place Op Output Shape Constraint](#q31--in-place-op-output-shape-constraint)
- [Q32 — Rank-0 vs Rank-1 Shape Difference](#q32--rank-0-vs-rank-1-shape-difference)

---

> Topics: Shape alignment rules, None/newaxis insertion, output shape derivation, common patterns.
> Exam frequency: **2024 exam + re-exam** — tested multiple times.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--output-shape-3--23)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

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

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets broadcasting shape compatibility rules, output shape prediction, common errors, and vectorized distance/haversine patterns

---

## Q13 — Shape Compatibility: (5,3,4) + (3,)

> **Week reference:** Week 4

Which of the following correctly describes the result of adding arrays with shapes `(5, 3, 4)` and `(3,)`?

- A) Output shape is `(5, 3, 4)` — `(3,)` broadcasts over the last axis
- B) Output shape is `(5, 3, 3)`
- C) ValueError — the trailing dims 4 and 3 are incompatible
- D) Output shape is `(5, 3)`

**Answer: C**

Right-align `(3,)` as `(1, 1, 3)` against `(5, 3, 4)`. Dimension pairs: 5 vs 1 → ok, 3 vs 1 → ok, 4 vs 3 → neither is 1 and they are unequal → ValueError. The trap is assuming "the 3 in `(3,)` matches the 3 in the middle of `(5,3,4)`" — but right-alignment puts `(3,)` against the last dimension, which is 4. Always right-align before comparing.

- A) Incorrect — `(3,)` right-aligns to `(1,1,3)`, so its value compares with the last dimension (4), not the middle one (3). Since 4 ≠ 3 and neither is 1, this is an error.
- B) Incorrect — no broadcasting rule produces a `(5,3,3)` result from these shapes. The last dims mismatch.
- C) Correct — right-alignment places `(3,)` against the trailing dimension of 4. Since 4 ≠ 3 and neither is 1, NumPy raises a ValueError.
- D) Incorrect — broadcasting never reduces rank or drops dimensions; the output can only be the same rank or higher than the inputs.

---

## Q14 — newaxis to Fix a Column Subtraction

> **Week reference:** Week 4

You have a matrix `M` of shape `(N, M)` and a 1D array `v` of shape `(N,)` holding per-row values. You want to subtract `v` from every column. Which expression is correct?

- A) `M - v`
- B) `M - v[None, :]`
- C) `M - v[:, None]`
- D) `M - v[:, :, None]`

**Answer: C**

`v[:, None]` converts shape `(N,)` to `(N, 1)`. Right-aligned against `(N, M)`: dim 0 is N vs N (equal), dim 1 is 1 vs M → M (broadcasts). Result: `(N, M)` with each row having its corresponding `v[i]` subtracted from all M columns.

- A) Incorrect — `v` shape `(N,)` pads left to `(1, N)`. Dim 1: M vs N — incompatible unless M == N. In the general case this raises a ValueError.
- B) Incorrect — `v[None,:]` gives shape `(1, N)`. Same issue: right-aligned, the last dim is N vs M, which fails unless N == M.
- C) Correct — `v[:,None]` gives shape `(N, 1)`. Dim 0 matches exactly (N vs N), dim 1 broadcasts (1 vs M → M).
- D) Incorrect — `v[:,:,None]` would fail because `v` is 1D; indexing with two slices on a 1D array raises an IndexError.

---

## Q15 — Pairwise Euclidean Distance Shape

> **Week reference:** Week 4

You have `A` with shape `(N, 2)` and `B` with shape `(M, 2)` representing 2D points. Which expression computes `diff` such that `diff[i, j, :]` holds the coordinate differences between point `A[i]` and point `B[j]`, and what is `diff`'s shape?

- A) `diff = A - B`, shape `(N, 2)` (requires N == M)
- B) `diff = A[:, None, :] - B[None, :, :]`, shape `(N, M, 2)`
- C) `diff = A[None, :, :] - B[:, None, :]`, shape `(M, N, 2)`
- D) `diff = A[:, :, None] - B[None, :, :]`, shape `(N, 2, M)`

**Answer: B**

`A[:, None, :]` has shape `(N, 1, 2)` and `B[None, :, :]` has shape `(1, M, 2)`. Broadcasting: dim 0: N vs 1 → N, dim 1: 1 vs M → M, dim 2: 2 vs 2 → 2. Result: `(N, M, 2)`. Entry `[i, j, :]` is `A[i] - B[j]`, which is exactly the per-pair coordinate difference needed to compute pairwise distances.

- A) Incorrect — direct subtraction of `(N,2)` and `(M,2)` requires N == M and gives element-wise differences between corresponding pairs, not all pairs.
- B) Correct — the `[:, None, :]` and `[None, :, :]` expansions create compatible `(N,1,2)` and `(1,M,2)` shapes, yielding all N×M differences with shape `(N, M, 2)`.
- C) Incorrect — this swaps the roles of A and B, giving shape `(M, N, 2)` where entry `[j, i, :]` is `B[j] - A[i]`. The index order is transposed relative to the stated requirement.
- D) Incorrect — `A[:, :, None]` is `(N, 2, 1)` and `B[None, :, :]` is `(1, M, 2)`. Dim 1: 2 vs M and dim 2: 1 vs 2 — both must be checked and this does not produce the intended `(N, M, 2)` layout.

---

## Q16 — Broadcasting Copies Data?

> **Week reference:** Week 4

When NumPy broadcasts a `(1, 4)` array against a `(3, 4)` array, what happens internally?

- A) NumPy creates a new `(3, 4)` array by physically copying the row 3 times before computing
- B) NumPy sets the stride of the broadcast dimension to 0, so the same memory is read repeatedly without copying
- C) NumPy raises a warning and falls back to a loop
- D) NumPy temporarily converts both arrays to lists, then broadcasts

**Answer: B**

Broadcasting uses stride tricks: the stretched dimension gets stride = 0 in the broadcast view. Reading along that axis repeatedly returns the same memory address — no copy is made. This is why broadcasting is both memory-efficient and fast. Physical copies only happen if you explicitly call `np.broadcast_to(...).copy()` or trigger a write to the broadcast result.

- A) Incorrect — no physical copy is made during broadcasting. The data is read via a zero stride, which means the same bytes are addressed on each pass.
- B) Correct — NumPy's broadcast mechanism sets the memory stride to 0 along any stretched dimension. The data appears tiled but occupies its original footprint in memory.
- C) Incorrect — NumPy never falls back to a Python loop internally for broadcasting; the operation is fully implemented in C with stride manipulation.
- D) Incorrect — NumPy never converts arrays to Python lists for arithmetic. All operations remain at the C/BLAS level for performance.

---

## Q17 — np.broadcast_shapes Result

> **Week reference:** Week 4

What does `np.broadcast_shapes((5, 1, 4), (3, 4))` return?

- A) `(5, 3, 4)`
- B) ValueError
- C) `(5, 1, 4)`
- D) `(3, 4)`

**Answer: A**

Left-pad `(3, 4)` to `(1, 3, 4)`. Compare dim by dim: 5 vs 1 → 5, 1 vs 3 → 3, 4 vs 4 → 4. Result: `(5, 3, 4)`. The middle dimension of 1 in the first shape stretches to 3, and the leading 1 from padding stretches to 5.

- A) Correct — after padding `(3,4)` becomes `(1,3,4)`. Pairs: (5,1)→5, (1,3)→3, (4,4)→4. Output: `(5,3,4)`.
- B) Incorrect — all dimension pairs are compatible (equal or one is 1). No error is raised.
- C) Incorrect — the middle axis of 1 in `(5,1,4)` stretches to match the 3 from the second shape. It does not remain 1 in the output.
- D) Incorrect — the result must accommodate all dimensions of both inputs. Returning only `(3,4)` would discard the leading 5.

---

## Q18 — Which Shape Pair is NOT Compatible?

> **Week reference:** Week 4

Which pair of shapes raises a `ValueError` when added?

- A) `(5, 3, 4)` and `(5, 1, 1)`
- B) `(5, 3, 4)` and `(1, 3, 1)`
- C) `(5, 3, 4)` and `(4,)`
- D) `(5, 3, 4)` and `(3,)`

**Answer: D**

Right-align and check each dim pair:

- A) `(5,1,1)` vs `(5,3,4)`: dims (5,5)→5, (1,3)→3, (1,4)→4 → valid, result `(5,3,4)`.
- B) `(1,3,1)` vs `(5,3,4)`: dims (1,5)→5, (3,3)→3, (1,4)→4 → valid, result `(5,3,4)`.
- C) `(4,)` pads to `(1,1,4)`. Dims (1,5)→5, (1,3)→3, (4,4)→4 → valid, result `(5,3,4)`.
- D) `(3,)` pads to `(1,1,3)`. Dim 2: 4 vs 3, neither is 1 → ValueError.

- A) Incorrect (valid) — `(5,1,1)` vs `(5,3,4)`: dims (5,5)→5, (1,3)→3, (1,4)→4. Result `(5,3,4)`. Both 1s stretch.
- B) Incorrect (valid) — `(1,3,1)` vs `(5,3,4)`: dims (1,5)→5, (3,3)→3, (1,4)→4. Result `(5,3,4)`. The two 1s stretch.
- C) Incorrect (valid) — `(4,)` pads to `(1,1,4)`. Dim 2: 4 vs 4 (equal). All dims compatible. Result `(5,3,4)`.
- D) Correct — `(3,)` pads to `(1,1,3)`. Dim 2: 4 vs 3, neither is 1 → ValueError. The classic exam trap: the 3 in `(3,)` looks like it should match the 3 in `(5,3,4)`, but right-alignment places it against the 4.

---

## Q19 — np.expand_dims Equivalence

> **Week reference:** Week 4

Given `a` with shape `(6, 4)`, which expression produces a shape of `(6, 1, 4)`?

- A) `np.expand_dims(a, axis=0)`
- B) `np.expand_dims(a, axis=1)`
- C) `np.expand_dims(a, axis=-1)`
- D) `a[None, :, :]`

**Answer: B**

`np.expand_dims(a, axis=1)` inserts a new axis at position 1, turning `(6, 4)` into `(6, 1, 4)`. Axis 0 would give `(1, 6, 4)`, axis -1 (last) would give `(6, 4, 1)`, and `a[None, :, :]` inserts a leading axis giving `(1, 6, 4)`.

- A) Incorrect — `axis=0` inserts at the front, giving `(1, 6, 4)`.
- B) Correct — `axis=1` inserts in the middle, giving `(6, 1, 4)`. This is equivalent to `a[:, None, :]`.
- C) Incorrect — `axis=-1` inserts at the end (after the last existing axis), giving `(6, 4, 1)`.
- D) Incorrect — `a[None, :, :]` inserts a leading axis, giving `(1, 6, 4)`, same as `np.expand_dims(a, axis=0)`.

---

## Q20 — Haversine All-Pairs Pattern

> **Week reference:** Week 4

You have `lats` and `lons` each with shape `(N,)` representing N GPS points. To compute all-pairs great-circle distances, you first need the coordinate differences `dlat[i,j] = lats[i] - lats[j]` for all i, j. Which code correctly computes `dlat` with shape `(N, N)`?

- A) `dlat = lats - lats`
- B) `dlat = lats[:, None] - lats[None, :]`
- C) `dlat = lats[None, :] - lats[:, None]`
- D) `dlat = lats[:, None] - lats[:, None]`

**Answer: B**

`lats[:, None]` has shape `(N, 1)` and `lats[None, :]` has shape `(1, N)`. Broadcasting: `(N,1)` vs `(1,N)` → `(N,N)`. Entry `[i,j]` is `lats[i] - lats[j]` — exactly the desired all-pairs difference. This is the outer-subtraction pattern used in vectorized haversine calculations.

- A) Incorrect — `lats - lats` is element-wise subtraction of the array with itself, giving an `(N,)` array of zeros. No pairwise differences are computed.
- B) Correct — `(N,1)` minus `(1,N)` broadcasts to `(N,N)`. Entry `[i,j]` = `lats[i] - lats[j]`, the per-pair latitude difference.
- C) Incorrect — this gives `(N,N)` too, but entry `[i,j]` = `lats[j] - lats[i]` (transposed). The i and j roles are swapped, so the sign is flipped.
- D) Incorrect — `lats[:,None] - lats[:,None]` subtracts two identical `(N,1)` arrays, giving an `(N,1)` array of zeros — not the pairwise matrix.

---

## Q21 — Three-Way Broadcasting

> **Week reference:** Week 4

What is the output shape of `np.ones((8, 1, 6)) + np.ones((1, 5, 1)) + np.ones((6,))`?

- A) `(8, 5, 6)`
- B) ValueError
- C) `(8, 5, 6)` — but only after the first two operands are summed first
- D) `(8, 1, 6)`

**Answer: A**

NumPy evaluates left-to-right. First, `(8,1,6) + (1,5,1)`: pairs (8,1)→8, (1,5)→5, (6,1)→6 → intermediate `(8,5,6)`. Then `(8,5,6) + (6,)`: `(6,)` pads to `(1,1,6)` → pairs (8,1)→8, (5,1)→5, (6,6)→6 → `(8,5,6)`. Final shape: `(8,5,6)`.

- A) Correct — the three-way addition resolves to `(8,5,6)`. Each 1-dim in the first two arrays stretches, and the 1D `(6,)` aligns with the last axis.
- B) Incorrect — all pairwise dimension checks pass. No ValueError occurs.
- C) Incorrect — option C is numerically equivalent to A and describes the same process. The answer is `(8,5,6)` regardless of how you frame intermediate steps; this option is a duplicate of A and is therefore not a distinct alternative.
- D) Incorrect — the second operand `(1,5,1)` causes the middle axis to expand from 1 to 5. The output cannot remain `(8,1,6)`.

---

## Q22 — Fixing a Row-Mean Subtraction Bug

> **Week reference:** Week 4

A student writes:

```python
X = np.random.rand(100, 50)
row_mean = X.mean(axis=1)   # shape (100,)
X_centered = X - row_mean   # BUG
```

The code raises a `ValueError`. Which fix is correct, and why?

- A) `X_centered = X - row_mean[None, :]` — adds a leading axis
- B) `X_centered = X - row_mean[:, None]` — converts to column vector `(100, 1)`
- C) `X_centered = X.T - row_mean` — transposes first
- D) `X_centered = X - row_mean.reshape(1, 100)` — reshapes to row vector

**Answer: B**

`row_mean` has shape `(100,)` which pads left to `(1, 100)`. Against `(100, 50)`: last dims 50 vs 100 — mismatch. The fix is `row_mean[:, None]`, giving shape `(100, 1)`. Against `(100, 50)`: dim 0 is 100 vs 100 (equal), dim 1 is 1 vs 50 → 50 (broadcasts). Each row of X has its own mean subtracted from all 50 features.

- A) Incorrect — `row_mean[None,:]` gives `(1, 100)`. Against `(100, 50)`: dim 1 is 100 vs 50 — still a mismatch.
- B) Correct — `row_mean[:,None]` gives `(100, 1)`. Dim 0 matches exactly (100), dim 1 broadcasts (1 → 50). Each row gets its row mean subtracted.
- C) Incorrect — `X.T` has shape `(50, 100)`. Subtracting `row_mean` of shape `(100,)` pads to `(1, 100)`. Dim 0: 50 vs 1 → 50, dim 1: 100 vs 100 → 100. This gives a transposed `(50, 100)` result — not the desired `(100, 50)` row-centered matrix.
- D) Incorrect — `row_mean.reshape(1, 100)` gives shape `(1, 100)`. Against `(100, 50)`: dim 1 is 100 vs 50 — still incompatible, same as option A.

---

## Set 3 — Extended Practice

> Targets in-place restrictions, np.broadcast_to read-only semantics, 0-D arrays, ufunc broadcasting, haversine internals, and advanced shape-manipulation traps.

---

## Q23 — In-Place Broadcasting Restriction

> **Week reference:** Week 4

**Mental Model:** An in-place operation (`+=`, `-=`, `*=`, etc.) is only legal when the broadcast output shape exactly equals the shape of the left-hand operand. You cannot use in-place ops to silently enlarge an array — NumPy raises a ValueError if the broadcast result would be bigger than the target.

Which of the following in-place operations raises a `ValueError`?

- A) `a = np.ones((3, 4)); a += np.ones((4,))`
- B) `a = np.ones((3, 4)); a += np.ones((1, 4))`
- C) `a = np.ones((3, 4)); a += np.ones((3, 1))`
- D) `a = np.ones((3, 4)); a += np.ones((3,))`

**Answer: D**

- A) Incorrect (valid) — `(4,)` pads to `(1, 4)`. Broadcasting `(3,4)` and `(1,4)` → `(3,4)`, which matches `a`'s shape. In-place is allowed.
- B) Incorrect (valid) — `(1, 4)` broadcasts against `(3, 4)` → `(3, 4)`, exactly `a`'s shape. In-place is allowed.
- C) Incorrect (valid) — `(3, 1)` broadcasts against `(3, 4)` → `(3, 4)`, exactly `a`'s shape. In-place is allowed; this is the column-vector pattern.
- D) Correct (raises ValueError) — `(3,)` pads to `(1, 3)`. Last dims: 4 vs 3 — incompatible. NumPy raises `ValueError: operands could not be broadcast together with shapes (3,4) (3,)`. This is the right-alignment trap applied to in-place ops; the 3 in `(3,)` aligns against the 4, not the 3 in `a`.

---

## Q24 — np.broadcast_to Read-Only View

> **Week reference:** Week 4

**Mental Model:** `np.broadcast_to(arr, shape)` returns a read-only view with stride 0 in all expanded dimensions. No data is copied. Attempting to write to the result raises a `ValueError` because the underlying storage is shared and writing would corrupt the source array.

What is true about `b = np.broadcast_to(np.ones((1, 4)), (3, 4))`?

- A) `b` is a new `(3, 4)` array with the row physically copied three times
- B) `b` is a read-only view; `b[0, 0] = 5` raises a `ValueError`
- C) `b` is writable because broadcast_to always returns a copy
- D) `b.shape` is `(1, 4)` — broadcast_to only changes strides, not the reported shape

**Answer: B**

- A) Incorrect — `np.broadcast_to` never copies data. The entire point of the function is to produce a view where repeated dimensions are read by setting their stride to 0. Memory usage stays at the size of the original `(1, 4)` array.
- B) Correct — `np.broadcast_to` always returns a read-only view (`b.flags.writeable == False`). Any attempt to assign to an element raises `ValueError: assignment destination is read-only`. To get a writable copy, call `.copy()` on the result.
- C) Incorrect — `np.broadcast_to` explicitly guarantees it returns a view, not a copy. If a copy were always made, the memory-efficiency benefit of broadcasting would be lost.
- D) Incorrect — `np.broadcast_to` does update the reported `.shape` to the target shape. The strides in the broadcast dimensions are 0, but `.shape` correctly reports the new shape.

---

## Q25 — Zero-Dimensional Array Broadcasting

> **Week reference:** Week 4

**Mental Model:** A 0-D (scalar) array has shape `()`. When broadcast, it is treated as if it has as many leading 1-dims as needed, then expanded to match every dimension of the other array. A 0-D array broadcasts against any shape without error — it is the ultimate "stretch to fit" operand.

What is the shape of `np.array(7.0) + np.ones((2, 3, 5))`?

- A) ValueError — a 0-D array has no dimensions to broadcast
- B) `(1,)`
- C) `(2, 3, 5)`
- D) `()`

**Answer: C**

- A) Incorrect — a 0-D array is the most flexible broadcasting operand. It has no fixed dimension constraints and expands to match any shape unconditionally.
- B) Incorrect — `(1,)` would imply the scalar was first reshaped to a 1-element 1-D array, but broadcasting does not reshape the 0-D array to `(1,)` — it directly expands to match the full target shape.
- C) Correct — a 0-D array is treated as shape `()`, which left-pads to `(1,1,1)` and then broadcasts to `(2,3,5)`. The output shape equals the shape of the non-scalar operand: `(2,3,5)`.
- D) Incorrect — `()` is the shape of the 0-D array itself, not the broadcast result. Broadcasting always produces the shape of the larger operand when one operand can expand to match the other.

---

## Q26 — Which newaxis Pattern Computes a 2D Outer Product?

> **Week reference:** Week 4

**Mental Model:** For an outer product of `u` (shape `(m,)`) and `v` (shape `(n,)`), you need a `(m,1)` times `(1,n)` arrangement. The `[:, None]` on `u` gives `(m,1)`, and the bare `v` is implicitly treated as `(1,n)`. Alternatively `u[None,:]` would give `(1,m)`, flipping the role.

`u` has shape `(5,)` and `v` has shape `(4,)`. Which expression produces a `(5, 4)` outer product where `result[i, j] = u[i] * v[j]`?

- A) `u[None, :] * v[:, None]`
- B) `u[:, None] * v[None, :]`
- C) `u * v[:, None]`
- D) `u[None, :] * v`

**Answer: B**

- A) Incorrect — `u[None,:]` is `(1,5)` and `v[:,None]` is `(4,1)`. Broadcasting: dim 0: 4 vs 1 → 4, dim 1: 1 vs 5 → 5. Result shape is `(4,5)`, not `(5,4)`. Entry `[i,j]` is `u[j] * v[i]` — the roles of i and j are transposed.
- B) Correct — `u[:,None]` is `(5,1)` and `v[None,:]` is `(1,4)`. Broadcasting: dim 0: 5 vs 1 → 5, dim 1: 1 vs 4 → 4. Result shape is `(5,4)`. Entry `[i,j]` = `u[i] * v[j]`, which is exactly the outer product definition.
- C) Incorrect — `u` is `(5,)` and `v[:,None]` is `(4,1)`. Right-aligned: `(1,5)` vs `(4,1)`. Dim 0: 4 vs 1 → 4, dim 1: 1 vs 5 → 5. Result is `(4,5)` — wrong shape and wrong index ordering.
- D) Incorrect — `u[None,:]` is `(1,5)` and `v` is `(4,)` which pads to `(1,4)`. Dim 1: 5 vs 4 — incompatible, raises ValueError.

---

## Q27 — Haversine cosprod Intermediate Shape

> **Week reference:** Week 4

**Mental Model:** In the vectorized haversine, `p1` and `p2` both have shape `(N, 2)` (lat, lon pairs). The expression `p1[:, None, 0]` selects column 0 (latitudes) and inserts a middle axis, giving shape `(N, 1)`. Multiplied with `p2[None, :, 0]` of shape `(1, N)`, the result is the `(N, N)` cosine-product matrix needed for the haversine formula.

In the vectorized haversine, `p1` and `p2` both have shape `(N, 2)`. What is the shape of `np.cos(p1[:, None, 0]) * np.cos(p2[None, :, 0])`?

- A) `(N, N, 2)`
- B) `(N, 2)`
- C) `(N, N)`
- D) `(N,)`

**Answer: C**

- A) Incorrect — the index `[:, None, 0]` selects column 0 (a scalar per row) and inserts a middle axis. The final integer index 0 collapses the last dimension entirely, leaving no size-2 axis in the output.
- B) Incorrect — after applying `[:, None, 0]`, the result has shape `(N, 1)`, not `(N, 2)`. The `0` index removes the coordinate dimension.
- C) Correct — `p1[:, None, 0]` selects latitude values and inserts a middle axis: shape `(N, 1)`. `p2[None, :, 0]` selects latitude values and inserts a leading axis: shape `(1, N)`. Broadcasting `(N,1) * (1,N)` → `(N,N)`. This is the all-pairs cosine product used in the haversine formula.
- D) Incorrect — a `(N,)` result would require the broadcasting to collapse both extra axes, which does not happen. The `[:, None]` and `[None, :]` patterns are specifically designed to produce a 2D result.

---

## Q28 — ufunc Broadcasting vs Explicit Loop

> **Week reference:** Week 4

**Mental Model:** NumPy universal functions (ufuncs) like `np.sin`, `np.cos`, `np.sqrt` apply broadcasting rules identically to arithmetic operators. They operate element-wise on arrays of any shape and broadcast inputs when needed. There is no need for explicit loops; the ufunc handles all dimensions.

Which statement about NumPy ufuncs and broadcasting is correct?

- A) Ufuncs like `np.sin` only operate on 1-D arrays; higher-dimensional arrays must be looped over manually
- B) Ufuncs apply broadcasting rules to their inputs, so `np.sin(a) + np.cos(b)` follows the same shape rules as `a + b`
- C) Ufuncs always return an array with the same shape as their first argument, regardless of broadcasting
- D) Ufuncs bypass broadcasting and always operate element-wise on flattened arrays

**Answer: B**

- A) Incorrect — ufuncs operate on N-dimensional arrays natively. One of their core advantages is eliminating manual loops over dimensions. `np.sin(arr)` applies the sine function to every element of `arr` regardless of its shape.
- B) Correct — ufuncs fully support broadcasting. `np.sin(a)` produces an array with the same shape as `a`, and binary ufuncs like `np.add(a, b)` (equivalently `a + b`) apply the broadcasting rules to their input shapes to determine the output shape. This is what makes vectorized code like the haversine formula possible.
- C) Incorrect — binary ufuncs return an array with the broadcast result shape, which may differ from the first argument's shape. For example, `np.add(np.ones((3,1)), np.ones((1,4)))` returns shape `(3,4)`, not `(3,1)`.
- D) Incorrect — ufuncs never flatten inputs. Flattening would destroy the shape information that broadcasting relies on. The operation is always multi-dimensional and shape-preserving.

---

## Q29 — np.broadcast_shapes with Incompatible Inputs

> **Week reference:** Week 4

**Mental Model:** `np.broadcast_shapes` applies the same compatibility rules as actual broadcasting, but returns the result shape without creating any arrays. If any dimension pair is incompatible (both > 1 and unequal), it raises a `ValueError`. It is useful for validating shapes before doing expensive computations.

What does `np.broadcast_shapes((4, 3), (3, 4))` return?

- A) `(4, 4)`
- B) `(4, 3, 4)`
- C) ValueError
- D) `(4, 4, 3)`

**Answer: C**

- A) Incorrect — `(4,4)` would require both inputs to have shapes compatible with a `(4,4)` output. For `(4,3)` and `(3,4)`: dim 0 is 4 vs 3 (incompatible), dim 1 is 3 vs 4 (incompatible). There is no valid broadcast output.
- B) Incorrect — `np.broadcast_shapes` never increases the rank beyond the maximum rank of its inputs. Both inputs are rank 2, so the output must also be rank 2 (if valid).
- C) Correct — right-align (same rank): dim 0 is 4 vs 3, neither is 1 and they are unequal — incompatible. `np.broadcast_shapes` raises `ValueError`. This is a common exam-style trap: the numbers 3 and 4 appear in both shapes but in different positions, making both dims incompatible simultaneously.
- D) Incorrect — same reason as B: rank cannot increase beyond max input rank.

---

## Q30 — Stacking newaxis at Different Positions

> **Week reference:** Week 4

**Mental Model:** Each `None`/`np.newaxis` in an index expression inserts exactly one size-1 axis at that position. Multiple `None`s insert multiple axes. The position of each `None` relative to `:` (or other indices) determines exactly where the new axis lands in the resulting shape.

`x` has shape `(6, 4)`. What is the shape of `x[:, None, :, None]`?

- A) `(6, 4, 1, 1)`
- B) `(6, 1, 4, 1)`
- C) `(1, 6, 1, 4)`
- D) `(6, 4)`

**Answer: B**

- A) Incorrect — the `None`s are inserted between and after the existing axes, not appended together at the end. The position of each `None` in the index expression directly controls where each new axis appears.
- B) Correct — parse left to right: `:` selects axis 0 (size 6), `None` inserts a new axis (size 1), `:` selects axis 1 (size 4), `None` inserts another new axis (size 1). Final shape: `(6, 1, 4, 1)`. This pattern is common for broadcasting a `(6, 4)` matrix against a `(1, C, 1, D)` tensor.
- C) Incorrect — `(1, 6, 1, 4)` would result from `None, :, None, :`, i.e., both `None`s placed before their corresponding slice, not interleaved.
- D) Incorrect — two `None`s are inserted, increasing the rank by 2. The result must be rank 4, not the original rank 2.

---

## Q31 — In-Place Op Output Shape Constraint

> **Week reference:** Week 4

**Mental Model:** For `a += b`, the broadcast result of `a` and `b` must have the same shape as `a`. If broadcasting would produce a larger shape than `a`, NumPy refuses the in-place operation. This rule prevents accidental in-place expansion of an array's shape, which would require reallocating memory.

`a = np.ones((1, 5))`. Which operation raises a `ValueError`?

- A) `a += np.ones((3, 5))`
- B) `a += np.ones((1, 1))`
- C) `a += np.ones((5,))`
- D) `a += np.ones((1, 5))`

**Answer: A**

- A) Correct — `(1,5)` and `(3,5)` broadcast to `(3,5)`. But the output shape `(3,5)` does not match `a`'s shape `(1,5)` — in-place ops cannot expand the array. NumPy raises `ValueError: non-broadcastable output operand with shape (1,5) doesn't match the broadcast shape (3,5)`.
- B) Incorrect (valid) — `(1,5)` and `(1,1)` broadcast to `(1,5)`, which matches `a`'s shape. In-place is allowed; the single value in `(1,1)` is added to all 5 elements.
- C) Incorrect (valid) — `(5,)` pads to `(1,5)`. Broadcasting `(1,5)` and `(1,5)` → `(1,5)`, which matches `a`'s shape. In-place is allowed.
- D) Incorrect (valid) — same shape `(1,5)` and `(1,5)`. Trivially compatible and the output shape matches.

---

## Q32 — Rank-0 vs Rank-1 Shape Difference

> **Week reference:** Week 4

**Mental Model:** `np.array(5)` is a 0-D array with shape `()` and 0 axes. `np.array([5])` is a 1-D array with shape `(1,)` and 1 axis. Although both hold a single value, their broadcasting behaviour differs: `()` broadcasts against any shape, while `(1,)` has an explicit axis that must align from the right.

`a = np.ones((3,))`. What are the shapes of `a + np.array(5)` and `a + np.array([5])` respectively?

- A) Both `(3,)` — scalars always broadcast to any shape
- B) `(3,)` and `(3,)` — but for different reasons
- C) `(3,)` and ValueError
- D) ValueError and `(3,)`

**Answer: B**

- A) Partially correct but misleading — both results are `(3,)`, but `np.array([5])` is NOT a scalar (it is a 1-D array of shape `(1,)`). The explanation "scalars always broadcast" only applies to the 0-D case.
- B) Correct — `np.array(5)` has shape `()`. It broadcasts to `(3,)` because a 0-D array left-pads with 1s as needed, giving `(1,)` → `(3,)`. `np.array([5])` has shape `(1,)`. Right-aligned against `(3,)`: 1 vs 3 → 3 (broadcasts). Both produce `(3,)` — but the first is 0-D broadcasting and the second is 1-D size-1 broadcasting. Understanding the difference matters for in-place op rules.
- C) Incorrect — `np.array([5])` has shape `(1,)` which is fully compatible with `(3,)` via the size-1 stretch rule. No error occurs.
- D) Incorrect — `np.array(5)` is the universally compatible 0-D case; it never raises a ValueError in broadcasting.

---
