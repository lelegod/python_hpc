# Floating Point & dtype Arithmetic — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — float16 precision loss at large values](#q1-float16-precision-loss-at-large-values)
- [Q2 — float16 rounding near 100](#q2-float16-rounding-near-100)
- [Q3 — int8 signed overflow](#q3-int8-signed-overflow)
- [Q4 — uint8 underflow (wrap-around)](#q4-uint8-underflow-wrap-around)
- [Q5 — float16 overflow to inf](#q5-float16-overflow-to-inf)
- [Q6 — int16 signed overflow at boundary](#q6-int16-signed-overflow-at-boundary)
- [Q7 — dtype downcast safety check](#q7-dtype-downcast-safety-check)
- [Q8 — float32 exact integer representation limit](#q8-float32-exact-integer-representation-limit)
- [Summary Table](#summary-table)
- [Key Facts Reference](#key-facts-reference)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q9 — Array Memory Footprint](#q9-array-memory-footprint)
- [Q10 — Dtype Memory Comparison](#q10-dtype-memory-comparison)
- [Q11 — int8 Value Wrapping on Array Construction](#q11-int8-value-wrapping-on-array-construction)
- [Q12 — Pandas astype Memory Reduction](#q12-pandas-astype-memory-reduction)
- [Q13 — Range Check for uint8](#q13-range-check-for-uint8)
- [Q14 — float16 Memory for Large Array](#q14-float16-memory-for-large-array)
- [Q15 — Detecting Unsafe Downcast](#q15-detecting-unsafe-downcast)
- [Q16 — Categorical vs Object Memory](#q16-categorical-vs-object-memory)
- [Q17 — float16 NaN vs inf](#q17-float16-nan-vs-inf)
- [Q18 — int32 vs int64 Memory Tradeoff](#q18-int32-vs-int64-memory-tradeoff)

---

> Format: Each question shows NumPy dtype code — predict the output.
> Exam frequency: **F25 exam Q2**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--float16-precision-loss-at-large-values)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — float16 precision loss at large values

```python
import numpy as np
print(np.float16(10000) + np.float16(1))
```

**A)** `10001.0`
**B)** `10000.0`
**C)** `inf`
**D)** `RuntimeWarning: overflow`

**Answer: B**

- A) Incorrect — adding 1 is below the ULP threshold at 10000, so no increment occurs
- B) Correct — float16 ULP at 10000 ≈ 10; adding 1 is far below half-ULP, so it rounds away and the result stays 10000.0
- C) Incorrect — 10001 is nowhere near the float16 max of 65504
- D) Incorrect — NumPy does not raise a RuntimeWarning for sub-ULP additions; the result simply stays the same

---

## Q2 — float16 rounding near 100

```python
import numpy as np
print(np.float16(100) + np.float16(0.05))
```

**A)** `100.05`
**B)** `100.0`
**C)** `100.06`
**D)** `100.0625`

**Answer: C**

- A) Incorrect — 100.05 is not exactly representable in float16; the stored value is 100.0625
- B) Incorrect — 0.05 exceeds the half-ULP threshold (0.03125), so it rounds up rather than being discarded
- C) Correct — float16 ULP at 100 = 0.0625; 0.05 > half-ULP so it rounds up to 0.0625, and float16 repr prints 100.0625 as `100.06`
- D) Incorrect — the stored value is 100.0625, but float16's `__repr__` rounds it to `100.06` when printing

---

## Q3 — int8 signed overflow

```python
import numpy as np
x = np.int8(120)
y = np.int8(10)
print(x + y)
```

**A)** `130`
**B)** `2`
**C)** `-126`
**D)** `OverflowError`

**Answer: C**

- A) Incorrect — 130 exceeds int8 max of 127 and cannot be stored
- B) Incorrect — 2 would imply a different wrap point; two's complement gives 130 - 256 = -126
- C) Correct — int8 max is 127; the true sum 130 wraps silently via two's complement: 130 - 256 = -126
- D) Incorrect — NumPy integer overflow is silent; no exception is raised

---

## Q4 — uint8 underflow (wrap-around)

```python
import numpy as np
arr = np.array([200], dtype=np.uint8)
print(arr - np.uint8(210))
```

**A)** `[-10]`
**B)** `[246]`
**C)** `[0]`
**D)** `RuntimeWarning: invalid value`

**Answer: B**

- A) Incorrect — -10 cannot be represented in uint8; the result wraps modulo 256
- B) Correct — uint8 underflow wraps mod 256: 200 - 210 = -10; -10 + 256 = 246
- C) Incorrect — uint8 does not clamp to zero; it wraps around
- D) Incorrect — NumPy wraps silently without raising a RuntimeWarning for integer underflow

---

## Q5 — float16 overflow to inf

```python
import numpy as np
x = np.float16(65000)
y = np.float16(1000)
print(x + y)
```

**A)** `66000.0`
**B)** `65504.0`
**C)** `nan`
**D)** `inf`

**Answer: D**

- A) Incorrect — 66000 exceeds float16 max ≈ 65504 and cannot be stored as a finite value
- B) Incorrect — the result is not clamped to the max; IEEE 754 produces infinity instead
- C) Incorrect — nan results from invalid operations (e.g., 0/0 or inf-inf), not overflow
- D) Correct — float16 max ≈ 65504; the sum 66000 exceeds this, so IEEE 754 produces `inf`

---

## Q6 — int16 signed overflow at boundary

```python
import numpy as np
val = np.int16(32767)
print(val + np.int16(1))
```

**A)** `32768`
**B)** `0`
**C)** `-32768`
**D)** `OverflowError`

**Answer: C**

- A) Incorrect — 32768 exceeds int16 max of 32767 and cannot be stored
- B) Incorrect — 0 would imply a different wrap; two's complement at 16-bit boundary gives -32768
- C) Correct — int16 max is 32767; adding 1 wraps silently to the minimum value -32768 via two's complement
- D) Incorrect — NumPy integer overflow is silent; no exception is raised

---

## Q7 — dtype downcast safety check

```python
col_min = -5
col_max = 250

# Can we safely cast this column to int8?
print(col_min >= -128 and col_max <= 127)
```

**A)** `True` — both values fit in int8
**B)** `False` — the column cannot be stored in int8
**C)** `True` — negative values make int8 the right choice
**D)** `False` — int8 cannot hold negative values

**Answer: B**

- A) Incorrect — 250 > 127, so col_max does not fit in int8 and the expression evaluates to False
- B) Correct — `250 <= 127` is False, so the whole expression is False; int8 max is 127 which cannot hold 250
- C) Incorrect — the presence of negative values does not make int8 correct when the max exceeds 127
- D) Incorrect — int8 can hold negative values (range -128 to 127); the problem is the upper bound, not the lower

---

## Q8 — float32 exact integer representation limit

```python
import numpy as np
a = np.float32(16_777_217)   # 2^24 + 1
print(a + np.float32(1) == a)
```

**A)** `False` — float32 is precise enough here
**B)** `True` — 16,777,217 cannot be represented exactly in float32
**C)** `True` — adding 1 to any float32 is a no-op
**D)** `False` — float32 handles integers exactly up to 2^32

**Answer: B**

- A) Incorrect — float32 only has a 23-bit mantissa; exact integer representation tops out at 2^24 = 16,777,216
- B) Correct — 16,777,217 = 2^24 + 1 rounds down to 16,777,216 in float32; adding 1 again still yields 16,777,216, so the comparison is True
- C) Incorrect — adding 1 is only a no-op beyond the exact integer limit; at small values float32 can represent consecutive integers fine
- D) Incorrect — float32 exact integer range ends at 2^24 = 16,777,216, not 2^32

---

## Summary Table

| Q | dtype | Phenomenon | Result |
|---|-------|------------|--------|
| 1 | float16 | ULP at 10000 ≈ 10; +1 is below resolution | `10000.0` |
| 2 | float16 | ULP at 100 = 0.0625; 0.05 rounds up | `100.06` |
| 3 | int8 | Signed overflow wraps (two's complement) | `-126` |
| 4 | uint8 | Unsigned underflow wraps mod 256 | `[246]` |
| 5 | float16 | Exceeds max ≈ 65504 → IEEE 754 infinity | `inf` |
| 6 | int16 | Signed overflow at boundary wraps to min | `-32768` |
| 7 | int8 | Range check: 250 > 127, use int16 instead | `False` |
| 8 | float32 | Exact integer limit = 2^24; beyond rounds | `True` |

## Key Facts Reference

- **float16:** ULP at 100 = 0.0625; ULP at 10000 ≈ 10; max ≈ 65504; overflow → `inf`
- **float32:** ~7 significant decimal digits; exact integers up to 2^24 = 16,777,216
- **int8:** -128 to 127; signed overflow wraps silently (no exception)
- **uint8:** 0 to 255; underflow wraps mod 256
- **int16:** -32768 to 32767; signed overflow wraps silently
- **uint16:** 0 to 65535
- NumPy integer overflow is **silent** (no `OverflowError`), unlike Python `int`
- NumPy float overflow follows **IEEE 754**: produces `inf`, not a wrapped integer

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets dtype memory footprint calculations, value range constraints, and pandas/numpy memory reduction strategies

---

## Q9 — Array Memory Footprint

> **Week reference:** Week 2

```python
import numpy as np
arr = np.zeros(1_000_000, dtype=np.float32)
print(arr.nbytes)
```

**A)** `1000000`
**B)** `2000000`
**C)** `4000000`
**D)** `8000000`

**Answer: C**

- A) Incorrect — 1,000,000 bytes = 1 byte per element, which corresponds to int8 or uint8, not float32
- B) Incorrect — 2,000,000 bytes = 2 bytes per element, which corresponds to float16 or int16, not float32
- C) Correct — float32 uses 4 bytes per element; 1,000,000 × 4 = 4,000,000 bytes. `arr.nbytes` returns total byte count
- D) Incorrect — 8,000,000 bytes = 8 bytes per element, which corresponds to float64, not float32

---

## Q10 — Dtype Memory Comparison

> **Week reference:** Week 2

```python
import numpy as np
a = np.ones(500_000, dtype=np.float64)
b = np.ones(500_000, dtype=np.float32)
print(a.nbytes // b.nbytes)
```

**A)** `1`
**B)** `2`
**C)** `4`
**D)** `8`

**Answer: B**

- A) Incorrect — float64 and float32 differ in size; they are not equal. float64 = 8 bytes, float32 = 4 bytes
- B) Correct — a.nbytes = 500,000 × 8 = 4,000,000; b.nbytes = 500,000 × 4 = 2,000,000; 4,000,000 // 2,000,000 = 2
- C) Incorrect — 4 would be the ratio of float64 to float16 (8 bytes vs 2 bytes), not float64 to float32
- D) Incorrect — 8 would imply a float64 to int8/bool ratio (8 bytes vs 1 byte), not a float32 comparison

---

## Q11 — int8 Value Wrapping on Array Construction

> **Week reference:** Week 2

```python
import numpy as np
arr = np.array([100, 150, 200], dtype=np.int8)
print(arr)
```

**A)** `[100 127 127]`
**B)** `[100 150 200]`
**C)** `[100 -106  -56]`
**D)** `OverflowError`

**Answer: C**

- A) Incorrect — NumPy does not clamp/saturate values at int8 max (127); it wraps using two's complement
- B) Incorrect — int8 max is 127; 150 and 200 both exceed this and cannot be stored as-is
- C) Correct — two's complement wrap: 150 − 256 = −106; 200 − 256 = −56; 100 fits in int8 unchanged
- D) Incorrect — NumPy integer overflow is silent; no exception is raised during array construction

---

## Q12 — Pandas astype Memory Reduction

> **Week reference:** Week 7

```python
import pandas as pd
import numpy as np

s = pd.Series(np.random.randn(1_000_000), dtype='float64')
s32 = s.astype('float32')
print(s.nbytes // s32.nbytes)
```

**A)** `1`
**B)** `2`
**C)** `4`
**D)** `8`

**Answer: B**

- A) Incorrect — astype('float32') does reduce memory; float64 (8 bytes) and float32 (4 bytes) are different sizes
- B) Correct — s.nbytes = 1,000,000 × 8 = 8,000,000; s32.nbytes = 1,000,000 × 4 = 4,000,000; 8,000,000 // 4,000,000 = 2
- C) Incorrect — 4× reduction comes from float64→float16 (8 bytes → 2 bytes), not float64→float32
- D) Incorrect — 8× reduction would require float64→bool or float64→int8 (8 bytes → 1 byte), not float32

---

## Q13 — Range Check for uint8

> **Week reference:** Week 2

```python
import numpy as np
values = [0, 50, 127, 200, 255]
arr = np.array(values, dtype=np.uint8)
print(list(arr))
```

**A)** `[0, 50, 127, 200, 255]`
**B)** `[0, 50, 127, -56, -1]`
**C)** `[0, 50, 127, 200, 127]`
**D)** `OverflowError`

**Answer: A**

- A) Correct — uint8 range is 0 to 255; all five values (0, 50, 127, 200, 255) are within this range and are stored exactly
- B) Incorrect — negative wrapping occurs with signed int8, not unsigned uint8. uint8 has no negative values; 200 and 255 are valid uint8 values
- C) Incorrect — uint8 does not clamp at 127; 127 is the max of int8 (signed), not uint8 (unsigned). uint8 max is 255
- D) Incorrect — NumPy integer overflow is always silent; no OverflowError is raised, and all values here are in range anyway

---

## Q14 — float16 Memory for Large Array

> **Week reference:** Week 2

```python
import numpy as np
arr = np.zeros(2_000_000, dtype=np.float16)
print(arr.nbytes / 1e6)
```

**A)** `1.0`
**B)** `2.0`
**C)** `4.0`
**D)** `16.0`

**Answer: C**

- A) Incorrect — 1.0 MB would be 2,000,000 × 0.5 bytes per element, which is not a valid dtype size
- B) Incorrect — 2.0 MB = 2,000,000 × 1 byte per element, corresponding to int8 or bool dtype
- C) Correct — float16 uses 2 bytes per element; 2,000,000 × 2 = 4,000,000 bytes = 4.0 MB
- D) Incorrect — 16.0 MB = 2,000,000 × 8 bytes per element, corresponding to float64

---

## Q15 — Detecting Unsafe Downcast

> **Week reference:** Week 2

```python
import numpy as np
data = np.array([10, 50, 130, 200], dtype=np.int32)
can_use_int8 = (data.min() >= -128) and (data.max() <= 127)
print(can_use_int8)
```

**A)** `True` — all values are positive and fit in int8
**B)** `False` — int8 cannot hold any positive values
**C)** `False` — 130 and 200 exceed int8 max of 127
**D)** `True` — int8 range is 0 to 255 for positive values

**Answer: C**

- A) Incorrect — "all positive" does not guarantee int8 safety; int8 max is 127, so 130 and 200 fail the `<= 127` check
- B) Incorrect — int8 holds positive values 0 to 127 perfectly; the issue is the upper bound, not the sign
- C) Correct — `data.max()` is 200; `200 <= 127` is False, so `can_use_int8` evaluates to False. The check correctly identifies the unsafe downcast
- D) Incorrect — int8 range is −128 to 127 (signed). The 0–255 range belongs to uint8. int8 cannot hold 130 or 200

---

## Q16 — Categorical vs Object Memory

> **Week reference:** Week 7

```python
import pandas as pd
s_obj = pd.Series(['yes', 'no', 'yes', 'no'] * 250_000)  # 1M elements, 2 unique
s_cat = s_obj.astype('category')
print(s_obj.nbytes > s_cat.nbytes)
```

**A)** `False` — Categorical always uses more memory due to the lookup table
**B)** `True` — Categorical uses integer codes instead of repeated string objects
**C)** `False` — object and category Series always have identical memory usage
**D)** `True` — Categorical compresses using gzip

**Answer: B**

- A) Incorrect — the lookup table overhead (2 short strings) is negligible compared to storing 1,000,000 Python string objects. Categorical is much smaller for low-cardinality data
- B) Correct — the object Series stores 1,000,000 Python string pointers plus heap-allocated string objects (~50+ bytes each). Categorical stores 1,000,000 int8 codes (1 byte each) plus 2 strings in the lookup. Memory drops dramatically
- C) Incorrect — object dtype tracks individual Python objects; category dtype uses integer codes. They have very different memory footprints for repeated string data
- D) Incorrect — Categorical uses dictionary encoding (integer codes), not gzip byte compression. It is a structural/algorithmic optimization, not a compression codec

---

## Q17 — float16 NaN vs inf

> **Week reference:** Week 2

```python
import numpy as np
a = np.float16(0.0)
b = np.float16(0.0)
print(a / b)
```

**A)** `0.0`
**B)** `inf`
**C)** `nan`
**D)** `ZeroDivisionError`

**Answer: C**

- A) Incorrect — 0.0 / 0.0 is an indeterminate form in IEEE 754; the result is NaN, not 0.0
- B) Incorrect — `inf` results from a nonzero value divided by zero (e.g., `1.0 / 0.0`). Zero divided by zero is undefined, producing NaN
- C) Correct — IEEE 754 defines 0.0 / 0.0 as NaN (Not a Number) because it is an indeterminate form. NumPy will also issue a RuntimeWarning: invalid value encountered in divide
- D) Incorrect — NumPy floating-point division by zero does not raise a Python exception; it follows IEEE 754 and produces NaN or inf with a RuntimeWarning

---

## Q18 — int32 vs int64 Memory Tradeoff

> **Week reference:** Week 2

```python
import numpy as np
a = np.zeros(1_000_000, dtype=np.int64)
b = np.zeros(1_000_000, dtype=np.int32)
print(a.nbytes - b.nbytes)
```

**A)** `0`
**B)** `1000000`
**C)** `4000000`
**D)** `8000000`

**Answer: C**

- A) Incorrect — int64 (8 bytes) and int32 (4 bytes) have different sizes; their nbytes values are not equal
- B) Incorrect — 1,000,000 byte difference would imply a 1 byte per element difference, corresponding to int8 vs no-dtype or similar. int64 vs int32 differs by 4 bytes per element
- C) Correct — a.nbytes = 1,000,000 × 8 = 8,000,000; b.nbytes = 1,000,000 × 4 = 4,000,000; difference = 4,000,000 bytes
- D) Incorrect — 8,000,000 would be the full size of the int64 array (a.nbytes), not the difference between the two arrays

---
