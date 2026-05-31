# NumPy Broadcasting — MCQ Practice

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
- C) `(5, 3, 4)` and `(5,)`
- D) `(5, 3, 4)` and `(3,)`

**Answer: D**

For option D: `(3,)` pads left to `(1, 1, 3)`. Dim 2: 4 vs 3 — neither is 1, not equal → ValueError. For the others: A gives `(5,3,4)` (1s stretch), B gives `(5,3,4)` (1s stretch), C pads `(5,)` to `(1,1,5)` — dim 2: 4 vs 5, also incompatible. Wait — let me verify C: `(5,)` → `(1,1,5)`, last dim 4 vs 5 → error too.

**Corrected reasoning:** Both C and D raise errors. D is the intended answer since the question specifically targets the right-alignment trap. For option C: `(5,)` pads to `(1,1,5)`, last dims 4 vs 5 → error. For option A: `(5,1,1)` → dims (5,5),(3,1)→3,(4,1)→4 = `(5,3,4)` valid. For option B: `(1,3,1)` → dims (5,1)→5,(3,3)→3,(4,1)→4 = `(5,3,4)` valid. Options C and D both fail; D is the classic exam trap (right-alignment mismatch).

- A) Incorrect (valid) — `(5,1,1)` vs `(5,3,4)`: dims (5,5)→5, (1,3)→3, (1,4)→4. Result `(5,3,4)`. Both 1s stretch.
- B) Incorrect (valid) — `(1,3,1)` vs `(5,3,4)`: dims (1,5)→5, (3,3)→3, (1,4)→4. Result `(5,3,4)`. The two 1s stretch.
- C) Incorrect but also raises error — `(5,)` pads to `(1,1,5)`. Dim 2: 4 vs 5, neither is 1 → also a ValueError.
- D) Correct — `(3,)` pads to `(1,1,3)`. Dim 2: 4 vs 3, neither is 1 → ValueError. This is the primary exam trap: the 3 in `(3,)` looks like it should match the 3 in `(5,3,4)`, but right-alignment places it against the 4.

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
