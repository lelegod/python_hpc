# Floating Point & dtype Arithmetic — Code-Based MCQ Practice

> Format: Each question shows NumPy dtype code — predict the output.
> Exam frequency: **F25 exam Q2**.

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
