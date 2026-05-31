# Week 4 — NumPy Broadcasting & Vectorization Syntax Reference

## Broadcasting Rules

1. Right-align shapes
2. Left-pad shorter shape with `1`s
3. Each dim: must be equal OR one is `1`
4. Output shape = max of each dim
5. Mismatch → `ValueError`

```python
(3,)   + (2, 3) → pad to (1, 3) → output (2, 3)    ✓
(2, 3) + (2,)   → pad to (1, 2) → dims 3 vs 2       ✗ ValueError
(4, 1, 3) + (1, 5, 3)           → output (4, 5, 3)  ✓
```

---

## Key Broadcasting Patterns

```python
import numpy as np

# Add a new axis
x[:, None]          # shape (n,) → (n, 1)
x[None, :]          # shape (n,) → (1, n)
x[:, None, None]    # shape (n,) → (n, 1, 1)

# Outer product: (n,) × (m,) → (n, m)
x[:, None] * y      # (n,1) * (m,) → (n, m)

# Pairwise distances: (n,) vs (m,) → (n, m)
abs(x[:, None] - y)

# Image mean subtraction: images (N,H,W,3), mean (N,3) → (N,H,W,3)
images - mean[:, None, None]   # mean becomes (N,1,1,3)

# Haversine all-pairs: (N,2) vs (M,2) → (N,M,2)
p1[:, None, :] - p2[None, :, :]   # (N,1,2) - (1,M,2) = (N,M,2)

# Standardize rows: data (n,d), mean (d,) → (n,d)
(data - mean) / std    # mean/std broadcast as (1,d)
```

---

## np.meshgrid

### `np.meshgrid(x, y)`
- **What**: creates 2D coordinate grids from 1D arrays
- **Returns**: two 2D arrays — `xpts` (varies along columns), `ypts` (varies along rows)
- **Gotcha**: `xpts` shape is `(len(y), len(x))` — rows × cols

```python
x = np.linspace(-2, 2, 5)    # (5,)
y = np.linspace(-2, 2, 5)    # (5,)
xpts, ypts = np.meshgrid(x, y)
# xpts.shape = (5, 5), ypts.shape = (5, 5)
# xpts[i, :] is constant for all i — x varies along columns
# ypts[:, j] is constant for all j — y varies along rows
```

---

## Views vs Copies

| Operation | Result |
|---|---|
| `a.T` | View (strides reversed) |
| `a[1, :]` | View |
| `a.reshape(...)` on C-order | View if possible |
| `a.reshape(...)` on F-order | Copy |
| `a[[0, 2], :]` | Always copy (fancy indexing) |
| `np.shares_memory(a, b)` | True if same buffer |

---

## Haversine Distance

```python
def distance_matrix_noloop(p1, p2):
    p1, p2 = np.radians(p1), np.radians(p2)
    dsin2 = np.sin(0.5 * (p1[:, None, :] - p2[None, :, :])) ** 2
    cosprod = np.cos(p1[:, None, 0]) * np.cos(p2[None, :, 0])
    a = dsin2[:, :, 0] + cosprod * dsin2[:, :, 1]
    D = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    D *= 6371   # km
    return D    # shape (N, M)
```

---

## Exam Traps

| Trap | Correct |
|---|---|
| `(2,3) + (2,)` works | `ValueError` — right-align gives 3 vs 2 |
| Output shape = sum of dims | Output shape = max of each dim |
| `a.T` changes memory layout | Only reverses strides, same buffer |
| Broadcasting adds dims on the right | Adds dims on the LEFT (right-align first) |
