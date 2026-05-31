# Floating Point & dtype Arithmetic — MCQ Practice

> Topics: float16 relative resolution, at-value precision, dtype ranges, downcasting safety.
> Exam frequency: **F25 exam Q2** — less frequent but distinctly tricky.

---

## Q1 — float16 Addition Near 10000

> **Week reference:** Week 2

What does the following print?

```python
import numpy as np
print(np.float16(10000) + np.float16(1))
```

- A) 10001.0
- B) 10000.0
- C) 10002.0
- D) inf

**Answer: B**

- A) Incorrect — float16 cannot represent 10001; the increment needed is ~10 at this magnitude, so 1 is rounded to 0.
- B) Correct — float16 relative resolution is ~0.001, so at value 10000 the smallest representable increment is ~10; adding 1 has no effect.
- C) Incorrect — 10002 is also not representable; the result stays at 10000.
- D) Incorrect — 10000 is well within float16's max of ~65504, so no overflow occurs.

---

## Q2 — float16 Addition Near 100

> **Week reference:** Week 2

What does the following print?

```python
import numpy as np
print(np.float16(100) + np.float16(0.05))
```

- A) 100.0
- B) 100.06
- C) 100.1
- D) 0.05

**Answer: B**

- A) Incorrect — the ULP (unit in last place) at value 100 in float16 is 0.0625; since 0.05 > 0.03125 (half ULP), it rounds UP, not to 0.
- B) Correct — at value 100, float16 exponent gives ULP = 0.0625; 0.05 > 0.03125 (half ULP) so it rounds up to 0.0625, giving 100.0625, printed as 100.06.
- C) Incorrect — 0.1 would be two ULPs above 100; 0.05 is less than one ULP (0.0625) so it cannot reach 100.1.
- D) Incorrect — the base value 100.0 dominates; only the small addend gets rounded, not lost entirely.

---

## Q3 — float16 vs int16 for Exact Integer Representation

> **Week reference:** Week 2

You need to store the integer value 10001 exactly. Which dtype succeeds?

- A) `np.float16` — it represents all integers up to 65504
- B) `np.int16` — it stores integers exactly within its range, and 10001 < 32767
- C) Both `np.float16` and `np.int16` store 10001 exactly
- D) Neither — 10001 requires at least 32-bit storage

**Answer: B**

- A) Incorrect — float16 has only ~10 bits of mantissa, so large integers like 10001 are rounded (stored as 10000).
- B) Correct — int16 uses exact integer arithmetic; 10001 is within its range of −32768 to 32767 and is stored without rounding.
- C) Incorrect — float16 rounds 10001 to 10000 due to limited mantissa precision.
- D) Incorrect — int16 (a 16-bit signed integer) can hold 10001 exactly.

---

## Q4 — int8 Overflow

> **Week reference:** Week 2

What does the following print?

```python
import numpy as np
print(np.int8(120) + np.int8(10))
```

- A) 130
- B) 127
- C) -126
- D) 0

**Answer: C**

- A) Incorrect — int8 max is 127; 130 is out of range and NumPy wraps on overflow.
- B) Incorrect — NumPy does not saturate at the maximum; it wraps using two's complement.
- C) Correct — 130 overflows int8 (max 127); two's complement wrap: 130 − 256 = −126.
- D) Incorrect — overflow does not yield 0; it wraps to the two's complement value.

---

## Q5 — uint8 Negative Wrapping

> **Week reference:** Week 2

What does the following print?

```python
import numpy as np
print(np.int8(-1).view(np.uint8))
```

- A) -1
- B) 0
- C) 127
- D) 255

**Answer: D**

- A) Incorrect — uint8 is unsigned and cannot represent negative values; it wraps.
- B) Incorrect — 0 would be the result of wrapping from uint8(256), not uint8(−1).
- C) Incorrect — 127 is the max of int8, not uint8; there is no saturation here.
- D) Correct — −1 in two's complement (mod 256) wraps to 255, the maximum uint8 value.

---

## Q6 — Downcasting Safety Check

> **Week reference:** Week 2

A DataFrame column contains integer values ranging from −1 to 5730. Which of the following dtypes is safe to downcast to?

- A) `int8` — small enough to be efficient
- B) `uint16` — max 65535 easily covers 5730
- C) `int16` — signed, max 32767 > 5730, and handles negative values
- D) `uint8` — max 255 is sufficient for most values

**Answer: C**

- A) Incorrect — int8 max is 127, which is far below 5730; overflow would occur.
- B) Incorrect — uint16 cannot represent −1 (unsigned types reject negatives; wrapping gives 65535).
- C) Correct — int16 is signed (handles −1) and has max 32767, which safely covers the range −1 to 5730.
- D) Incorrect — uint8 max is 255, far below 5730, and it cannot hold −1.

---

## Q7 — float16 Overflow to Infinity

> **Week reference:** Week 2

What does the following print?

```python
import numpy as np
print(np.float16(70000))
```

- A) 70000.0
- B) 65504.0
- C) nan
- D) inf

**Answer: D**

- A) Incorrect — 70000 exceeds float16's maximum representable value of ~65504.
- B) Incorrect — float16 does not saturate at its maximum; it overflows to infinity.
- C) Incorrect — overflow produces inf, not nan; nan arises from undefined operations (e.g., 0/0).
- D) Correct — float16 max is ~65504; values beyond this overflow to positive infinity.

---

## Q8 — float32 Precision Limits

> **Week reference:** Week 2

What is the most likely stored value when you compute `np.float32(1234567.89)`?

- A) 1234567.89 — float32 has enough precision
- B) 1234568.0 — float32 has ~7 significant decimal digits, so the fractional part is lost
- C) 1234567.0 — float32 truncates to the nearest integer
- D) 1234560.0 — float32 rounds to the nearest ten

**Answer: B**

- A) Incorrect — float32 provides only ~7 significant decimal digits; 1234567.89 has 9 significant digits, so the last two are lost.
- B) Correct — float32 can represent about 7 significant digits; 1234567.89 rounds to approximately 1234568.0.
- C) Incorrect — float32 does not truncate to integers; it rounds to the nearest representable float.
- D) Incorrect — the rounding granularity at this magnitude is roughly 1 unit, not 10.

---

## Q9 — Relative vs Absolute Resolution of float16

> **Week reference:** Week 2

The machine epsilon for float16 is approximately 0.001. Which statement correctly describes what this means?

- A) The absolute precision of float16 is 0.001 everywhere — any two values differing by more than 0.001 can always be distinguished.
- B) The relative precision is ~0.001 — the smallest distinguishable increment at value V is approximately V × 0.001.
- C) float16 can only represent numbers up to 1/0.001 = 1000 before losing all precision.
- D) float16 and float32 have the same relative precision; float64 alone uses 0.001 epsilon.

**Answer: B**

- A) Incorrect — 0.001 is the *relative* resolution, not absolute; at large values (e.g., 10000), the smallest increment is ~10, not 0.001.
- B) Correct — machine epsilon defines *relative* precision; the gap between consecutive representable values scales linearly with the magnitude of the value.
- C) Incorrect — float16 can represent values up to ~65504; the epsilon does not set an upper limit on representable values.
- D) Incorrect — float32 epsilon is ~1.2×10⁻⁷ and float64 epsilon is ~2.2×10⁻¹⁶; they differ significantly from float16's ~0.001.

---

## Q10 — int16 Range Confusion with uint16

> **Week reference:** Week 2

A dataset column has non-negative integer values with a maximum of 40000. A colleague suggests casting it to `int16` to save memory. What is the problem?

- A) There is no problem — int16 max is 65535, which is greater than 40000.
- B) int16 is a signed type with max 32767; 40000 overflows and wraps to a negative value.
- C) int16 cannot hold non-negative values at all.
- D) 40000 exceeds float16 precision, so int16 also cannot hold it exactly.

**Answer: B**

- A) Incorrect — 65535 is the max of *uint16*, not int16; int16 max is only 32767.
- B) Correct — int16 is signed, with range −32768 to 32767; 40000 overflows and wraps to a negative two's complement value (~−25536).
- C) Incorrect — int16 can hold non-negative values (0 to 32767); the issue is the maximum, not the sign.
- D) Incorrect — int16 is an integer type; its precision is unrelated to float16's epsilon; the problem is purely the range limit.

---
