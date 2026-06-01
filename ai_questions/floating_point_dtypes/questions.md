# Floating Point & dtype Arithmetic — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — float16 Addition Near 10000](#q1-float16-addition-near-10000)
- [Q2 — float16 Addition Near 100](#q2-float16-addition-near-100)
- [Q3 — float16 vs int16 for Exact Integer Representation](#q3-float16-vs-int16-for-exact-integer-representation)
- [Q4 — int8 Overflow](#q4-int8-overflow)
- [Q5 — uint8 Negative Wrapping](#q5-uint8-negative-wrapping)
- [Q6 — Downcasting Safety Check](#q6-downcasting-safety-check)
- [Q7 — float16 Overflow to Infinity](#q7-float16-overflow-to-infinity)
- [Q8 — float32 Precision Limits](#q8-float32-precision-limits)
- [Q9 — Relative vs Absolute Resolution of float16](#q9-relative-vs-absolute-resolution-of-float16)
- [Q10 — int16 Range Confusion with uint16](#q10-int16-range-confusion-with-uint16)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q11 — Memory Footprint of a float64 Array](#q11-memory-footprint-of-a-float64-array)
- [Q12 — Memory Savings: float64 to float32 Downcast](#q12-memory-savings-float64-to-float32-downcast)
- [Q13 — Smallest Safe Integer Dtype for Range 0–200](#q13-smallest-safe-integer-dtype-for-range-0200)
- [Q14 — Pandas memory_usage(deep=True)](#q14-pandas-memory_usagedeeptrue)
- [Q15 — pd.Categorical for Low-Cardinality Strings](#q15-pdcategorical-for-low-cardinality-strings)
- [Q16 — float64 to float16 Memory Reduction](#q16-float64-to-float16-memory-reduction)
- [Q17 — Total DataFrame Memory Calculation](#q17-total-dataframe-memory-calculation)
- [Q18 — pd.to_numeric with downcast='float'](#q18-pdto_numeric-with-downcastfloat)
- [Q19 — int8 Range: Values Including 200](#q19-int8-range-values-including-200)
- [Q20 — float32 vs float64 for ML Features](#q20-float32-vs-float64-for-ml-features)
- [Set 3 — Extended Practice](#set-3-extended-practice)
- [Q21 — astype() Copy vs View Semantics](#q21-astype-copy-vs-view-semantics)
- [Q22 — dtype Promotion When Mixing float32 and float64](#q22-dtype-promotion-when-mixing-float32-and-float64)
- [Q23 — np.finfo() Fields: eps vs resolution](#q23-npfinfo-fields-eps-vs-resolution)
- [Q24 — float64 Exact Integer Representation Limit](#q24-float64-exact-integer-representation-limit)
- [Q25 — NaN Propagation Rules](#q25-nan-propagation-rules)
- [Q26 — complex64 Memory Layout](#q26-complex64-memory-layout)
- [Q27 — Mixed-dtype Array Construction Promotion](#q27-mixed-dtype-array-construction-promotion)
- [Q28 — uint16 Overflow Wrapping](#q28-uint16-overflow-wrapping)
- [Q29 — np.iinfo() for Safe Downcast Decisions](#q29-npiinfo-for-safe-downcast-decisions)
- [Q30 — inf Arithmetic and NaN-Producing Operations](#q30-inf-arithmetic-and-nan-producing-operations)

---

> Topics: float16 relative resolution, at-value precision, dtype ranges, downcasting safety.
> Exam frequency: **F25 exam Q2** — less frequent but distinctly tricky.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--float16-addition-near-10000)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — float16 Addition Near 10000

> **Week reference:** Week 2

**Mental Model:** float16 values in [8192, 16384) are spaced 8 apart (ULP = 8 at that exponent range). 10000 falls in this range, so ULP = 8 at 10000. Adding 1, which is less than half-ULP (4), rounds to zero change. Trap: treating the ~0.1% relative spacing as an absolute threshold.

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

- A) Incorrect — float16 cannot represent 10001. Float16 values near 10000 are spaced 8 apart (ULP = 8). Adding 1, which is less than half-ULP (4), rounds to zero change, leaving the result at 10000.
- B) Correct — 10000 is exactly representable in float16 (it falls in the [8192, 16384) exponent range where adjacent values are 8 apart). Since 1 < half-ULP (4), it rounds down to 0 increment, leaving the result at 10000.0.
- C) Incorrect — 10002 is not a representable float16 value near 10000. The next representable value above 10000 is 10008. Adding 1 does not reach 10008 either.
- D) Incorrect — 10000 is well within float16's maximum of ~65504, so no overflow occurs. Infinity only appears when the value exceeds ~65504.

---

## Q2 — float16 Addition Near 100

> **Week reference:** Week 2

**Mental Model:** 100 falls in the float16 range [64, 128) = [2^6, 2^7). ULP = 2^(6−10) = 2^(−4) = 0.0625. Half-ULP = 0.03125. Since 0.05 > 0.03125, it rounds UP to one full ULP = 0.0625. So 100 + 0.05 → 100.0625, printed as 100.06.

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

- A) Incorrect — the ULP at value 100 in float16 is 0.0625; 0.05 is larger than half-ULP (0.03125), so it rounds UP to one full ULP. The increment is not lost; it rounds to 0.0625.
- B) Correct — at value 100, the float16 exponent gives ULP = 0.0625. Since 0.05 > 0.03125 (half ULP), round-to-nearest gives 0.0625. Result = 100.0625, which numpy prints as approximately 100.06.
- C) Incorrect — reaching 100.1 would require adding approximately 0.0625 × 2 = 0.125 (two ULPs above 100). Since 0.05 < 0.0625, it rounds to at most one ULP, giving 100.0625, not 100.1.
- D) Incorrect — 0.05 is not the output; the base value 100.0 completely dominates in a floating-point addition. The question is only whether the small addend rounds to 0 or to 0.0625 — not whether 100 disappears.

---

## Q3 — float16 vs int16 for Exact Integer Representation

> **Week reference:** Week 2

**Mental Model:** float16 has a 10-bit mantissa, giving ~3 decimal digits of precision. Large integers like 10001 require distinguishing consecutive integers — at this magnitude that needs more than 10 bits of mantissa. int16 stores integers exactly with no rounding. They are different tools: float16 for approximate real numbers, int16 for exact integers.

You need to store the integer value 10001 exactly. Which dtype succeeds?

- A) `np.float16` — it represents all integers up to 65504
- B) `np.int16` — it stores integers exactly within its range, and 10001 < 32767
- C) Both `np.float16` and `np.int16` store 10001 exactly
- D) Neither — 10001 requires at least 32-bit storage

**Answer: B**

- A) Incorrect — float16 can represent values up to ~65504, but NOT all integers in that range exactly. The 10-bit mantissa means consecutive integers near 10000 are spaced ~10 apart; 10001 rounds to 10000. "Max value" and "exact integer range" are different things.
- B) Correct — int16 uses exact integer arithmetic (two's complement) with no rounding. Its range is −32768 to 32767; since 10001 < 32767, it fits and is stored without any rounding or precision loss.
- C) Incorrect — float16 rounds 10001 to 10000 due to its limited 10-bit mantissa. The ULP at 10000 is ~10, so 10001 and 10000 are indistinguishable in float16.
- D) Incorrect — int16 (a 16-bit signed integer) can hold 10001 exactly. 32-bit storage is only required if the value exceeds 32767 (for signed) or 65535 (for unsigned 16-bit).

---

## Q4 — int8 Overflow

> **Week reference:** Week 2

**Mental Model:** int8 range is −128 to 127. When you exceed 127, NumPy wraps using two's complement arithmetic: subtract 256 from the mathematical result. 120 + 10 = 130 → 130 − 256 = −126. No exception is raised; the wrap is silent.

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

- A) Incorrect — int8 max is 127; 130 is out of range. NumPy scalar arithmetic wraps on overflow rather than raising an exception or returning the clipped maximum.
- B) Incorrect — NumPy does not saturate at the maximum value (127) like some embedded/DSP systems. It wraps using two's complement arithmetic, which is the standard C integer overflow behaviour.
- C) Correct — 120 + 10 = 130 mathematically. Since 130 > 127 (int8 max), two's complement wrap gives 130 − 256 = −126. This is the bit pattern `0b10000010` interpreted as a signed 8-bit integer.
- D) Incorrect — overflow does not yield 0; it wraps to the two's complement value. For example, `np.int8(128)` wraps to −128 (since 128 − 256 = −128), not 0. The value 0 would arise from adding values that cancel (e.g., 1 + (−1)), not from overflow.

---

## Q5 — uint8 Negative Wrapping

> **Week reference:** Week 2

**Mental Model:** −1 in two's complement (any width) is "all bits set." In 8-bit: `11111111` = 255 unsigned. `.view(np.uint8)` reinterprets the same bits without conversion — so the bit pattern of int8(−1) (which is `0xFF`) becomes uint8(255).

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

- A) Incorrect — uint8 is unsigned and cannot represent negative values; the range is 0 to 255. `.view()` reinterprets the bit pattern without any sign extension or conversion.
- B) Incorrect — 0 would be the uint8 view of int8(0), not int8(−1). The bit pattern of −1 is `0xFF` (255), not `0x00`.
- C) Incorrect — 127 is the maximum value of int8 (bit pattern `0x7F`), not the view of −1. The bit pattern of int8(−1) is `0xFF`, which is 255 as uint8.
- D) Correct — int8(−1) has the two's complement bit pattern `0xFF` = `11111111`. `.view(np.uint8)` reinterprets those same 8 bits as an unsigned integer: `11111111` = 255.

---

## Q6 — Downcasting Safety Check

> **Week reference:** Week 2

**Mental Model:** Check three things: (1) signed vs unsigned (negative values require signed), (2) max value fits in range, (3) min value fits in range. Here: −1 requires signed → eliminate uint types. Range 5730 < 32767 (int16 max) → int16 is safe.

A DataFrame column contains integer values ranging from −1 to 5730. Which of the following dtypes is safe to downcast to?

- A) `int8` — small enough to be efficient
- B) `uint16` — max 65535 easily covers 5730
- C) `int16` — signed, max 32767 > 5730, and handles negative values
- D) `uint8` — max 255 is sufficient for most values

**Answer: C**

- A) Incorrect — int8 max is 127, which is far below 5730. Values from 128 to 5730 would silently wrap to incorrect negative values. The "small enough" logic is backwards: you need the dtype to be large enough, not small.
- B) Incorrect — uint16 cannot represent −1. Unsigned types have no negative range. Casting −1 to uint16 wraps to 65535 (0xFFFF), corrupting the data silently. The maximum headroom is irrelevant if the minimum value is negative.
- C) Correct — int16 is signed (handles −1 through −32768) and has a maximum of 32767, which comfortably covers 5730. Both the minimum (−1 > −32768) and maximum (5730 < 32767) fit within int16's range.
- D) Incorrect — uint8 max is 255, far below 5730 — values above 255 would wrap. Additionally, uint8 cannot represent −1 at all. Both the range and the signedness requirements are violated.

---

## Q7 — float16 Overflow to Infinity

> **Week reference:** Week 2

**Mental Model:** float16 max finite value ≈ 65504. Exceeding this overflows to `inf` (not to the max value, not to NaN). Saturation would be safer but IEEE 754 mandates overflow → infinity. NaN only arises from undefined operations like 0/0 or inf−inf.

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

- A) Incorrect — 70000 exceeds float16's maximum finite representable value of approximately 65504. There is no way to encode 70000 in 16-bit IEEE 754 format.
- B) Incorrect — float16 does not saturate (clamp) at its maximum value. IEEE 754 specifies that overflow produces infinity, not the maximum finite value. Saturation semantics exist in some special integer formats but not in standard IEEE floats.
- C) Incorrect — overflow produces `inf`, not `nan`. NaN arises from undefined operations such as `0.0/0.0`, `inf - inf`, or `sqrt(-1)`. Overflow from a finite input always gives ±inf.
- D) Correct — float16 max is ~65504; values beyond this overflow to positive infinity (`inf`) per IEEE 754 standard. `np.float16(70000)` evaluates to `inf` at construction time.

---

## Q8 — float32 Precision Limits

> **Week reference:** Week 2

**Mental Model:** float32 has a 23-bit mantissa ≈ 7 significant decimal digits. 1234567.89 has 9 significant digits, so the last two are lost to rounding. The fractional part `.89` is rounded away, leaving approximately 1234568.0.

What is the most likely stored value when you compute `np.float32(1234567.89)`?

- A) 1234567.89 — float32 has enough precision
- B) 1234568.0 — float32 has ~7 significant decimal digits, so the fractional part is lost
- C) 1234567.0 — float32 truncates to the nearest integer
- D) 1234560.0 — float32 rounds to the nearest ten

**Answer: B**

- A) Incorrect — float32 provides only ~7 significant decimal digits (23-bit mantissa × log₁₀(2) ≈ 6.92 digits). 1234567.89 has 9 significant digits, so the last ~2 digits cannot be represented and are rounded away.
- B) Correct — float32 can represent approximately 7 significant digits. The value 1234567.89 rounds to the nearest representable float32, which is approximately 1234568.0. The `.89` fractional part is lost because the integer part already uses all 7 significant digits.
- C) Incorrect — float32 does not truncate to integers; it rounds to the nearest representable floating-point value. Truncation would give 1234567.0, but round-to-nearest gives 1234568.0 (since .89 > .5).
- D) Incorrect — the rounding granularity (ULP) at 1234567 in float32 is approximately 0.0625 to 1 unit, not 10. Rounding to the nearest ten would require float16, which has ULP ≈ 10 at this magnitude.

---

## Q9 — Relative vs Absolute Resolution of float16

> **Week reference:** Week 2

**Mental Model:** Machine epsilon is a RELATIVE quantity — it scales with the value being represented. At V = 10000, the absolute gap between consecutive floats is 8 (exact: 2^3, since 10000 is in [8192, 16384)). At V = 0.001, the absolute gap is ~0.000001. The ~0.001 epsilon only equals ~0.001 in absolute terms near V = 1.

The machine epsilon for float16 is approximately 0.001. Which statement correctly describes what this means?

- A) The absolute precision of float16 is 0.001 everywhere — any two values differing by more than 0.001 can always be distinguished.
- B) The relative precision is ~0.001 — the smallest distinguishable increment at value V is approximately V × 0.001.
- C) float16 can only represent numbers up to 1/0.001 = 1000 before losing all precision.
- D) float16 and float32 have the same relative precision; float64 alone uses 0.001 epsilon.

**Answer: B**

- A) Incorrect — 0.001 is the *relative* resolution, not absolute. At value 10000, the absolute gap between consecutive float16 values is ~10, not 0.001. Two values differing by 5 (> 0.001 absolute) cannot be distinguished at that magnitude. This is the central trap of this question.
- B) Correct — machine epsilon defines *relative* precision. The absolute gap between consecutive representable values (ULP) at value V scales with V. At V = 1 the gap is ~0.001; at V = 10000 the gap is 8 (exact, since 10000 is in the [8192,16384) exponent range); at V = 0.01 the gap is ~0.000001.
- C) Incorrect — float16 can represent values up to ~65504 (limited by the 5-bit exponent, not the mantissa). Machine epsilon does not set an upper representable limit; it only describes the *density* of representable values relative to their magnitude.
- D) Incorrect — float16 epsilon ≈ 0.001, float32 epsilon ≈ 1.2×10⁻⁷, float64 epsilon ≈ 2.2×10⁻¹⁶. These differ by orders of magnitude. float16 has the worst (largest) relative precision among standard IEEE 754 types.

---

## Q10 — int16 Range Confusion with uint16

> **Week reference:** Week 2

**Mental Model:** int16 is signed: range −32768 to 32767. uint16 is unsigned: range 0 to 65535. The trap is confusing 65535 (uint16 max) with int16 max. When a colleague says "int16 max is 65535," they are wrong — that's uint16. Values above 32767 overflow int16 silently.

A dataset column has non-negative integer values with a maximum of 40000. A colleague suggests casting it to `int16` to save memory. What is the problem?

- A) There is no problem — int16 max is 65535, which is greater than 40000.
- B) int16 is a signed type with max 32767; 40000 overflows and wraps to a negative value.
- C) int16 cannot hold non-negative values at all.
- D) 40000 exceeds float16 precision, so int16 also cannot hold it exactly.

**Answer: B**

- A) Incorrect — 65535 is the maximum of *uint16* (unsigned 16-bit integer), not int16. int16 is a signed type using one bit for the sign, leaving 15 bits for magnitude: max = 2¹⁵ − 1 = 32767. This confusion between signed and unsigned ranges is the most common int16 mistake.
- B) Correct — int16 is signed with range −32768 to 32767. The value 40000 exceeds 32767, so it overflows. Two's complement wrap: 40000 − 65536 = −25536. This negative value silently corrupts the data with no warning.
- C) Incorrect — int16 can hold non-negative values (0 to 32767) perfectly well. The issue is not that it rejects non-negatives, but that its positive maximum (32767) is too small for 40000.
- D) Incorrect — int16 is an integer type and its precision is entirely unrelated to float16's machine epsilon. Integer types represent their range exactly with no floating-point rounding; the problem here is purely the range limit, not floating-point precision.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets dtype memory footprint calculations, value range constraints, and pandas/numpy memory reduction strategies

---

## Q11 — Memory Footprint of a float64 Array

> **Week reference:** Week 2

A NumPy array contains 500,000 elements of dtype `float64`. How many megabytes of memory does it occupy?

- A) 2 MB
- B) 4 MB
- C) 8 MB
- D) 16 MB

**Answer: B**

- A) Incorrect — 2 MB would correspond to float32 (4 bytes × 500,000 = 2,000,000 bytes = 2 MB) or float16 for 1 million elements. float64 is 8 bytes per element.
- B) Correct — float64 uses 8 bytes per element. 500,000 × 8 = 4,000,000 bytes = 4 MB. Memory = n_elements × bytes_per_dtype is the fundamental formula.
- C) Incorrect — 8 MB would be 1,000,000 × 8 bytes (1 million float64 elements), not 500,000. Off by a factor of 2 in element count.
- D) Incorrect — 16 MB would require either 2 million float64 elements or 2 million float32 pairs. This is double the correct answer.

---

## Q12 — Memory Savings: float64 to float32 Downcast

> **Week reference:** Week 2

A DataFrame column of 2,000,000 float64 values is downcast to float32. What fraction of memory is saved?

- A) 25% saved (75% remains)
- B) 50% saved (50% remains)
- C) 75% saved (25% remains)
- D) No saving — float32 and float64 use the same memory

**Answer: B**

- A) Incorrect — 25% savings would mean going from 4 bytes to 3 bytes per element, which corresponds to no standard dtype conversion. float64→float32 is exactly a 2× reduction.
- B) Correct — float64 is 8 bytes per element; float32 is 4 bytes. The ratio is 2×, so 50% of memory is saved. Before: 2,000,000 × 8 = 16 MB; after: 2,000,000 × 4 = 8 MB; savings = 8 MB = 50%.
- C) Incorrect — 75% savings would come from float64→float16 (8 bytes → 2 bytes = 4× reduction). float32 is only a 2× reduction from float64, not 4×.
- D) Incorrect — float32 uses exactly half the memory of float64. They differ by a factor of 2 in bytes per element (4 vs 8).

---

## Q13 — Smallest Safe Integer Dtype for Range 0–200

> **Week reference:** Week 2

A column contains non-negative integer values from 0 to 200 inclusive. What is the smallest NumPy dtype that can hold all values safely?

- A) `int8` — its max of 127 covers the common case
- B) `uint8` — its range of 0 to 255 covers 0 to 200
- C) `int16` — the only safe signed option
- D) `uint16` — needed because 200 is close to int8's limit

**Answer: B**

- A) Incorrect — int8 max is 127 (signed), which is less than 200. Values 128–200 would silently wrap to negative values. This is one of the most common dtype mistakes on the exam.
- B) Correct — uint8 is unsigned with range 0 to 255. Since all values are non-negative and the max is 200 < 255, uint8 fits perfectly and uses only 1 byte per element.
- C) Incorrect — int16 would work but it uses 2 bytes per element, not 1. It is not the smallest safe dtype; uint8 (1 byte) is sufficient and smaller.
- D) Incorrect — uint16 (0 to 65535) would also work but wastes 2 bytes per element when uint8 (0 to 255, 1 byte) is sufficient for values up to 200.

---

## Q14 — Pandas memory_usage(deep=True)

> **Week reference:** Week 7

Why does `df.memory_usage(deep=True)` often return a larger number than `df.memory_usage(deep=False)` for DataFrames with string columns?

- A) `deep=True` counts index memory twice to account for duplication
- B) `deep=True` resolves Python object overhead for variable-length strings, including each string's actual character data
- C) `deep=True` converts all dtypes to float64 before measuring
- D) `deep=True` and `deep=False` always return the same value; the parameter is deprecated

**Answer: B**

- A) Incorrect — `deep=True` does not double-count the index. Index memory is always counted once; the `deep` parameter only affects object-dtype columns.
- B) Correct — object-dtype string columns store pointers (8 bytes each) in the array, but the actual string data lives in separate Python objects on the heap. `deep=True` follows those pointers and adds the actual string memory, which can be much larger than just the pointer array.
- C) Incorrect — `memory_usage` never converts dtypes; it only measures existing memory layout. Conversion would change the DataFrame, which a reporting function never does.
- D) Incorrect — `deep=True` is actively used and can return substantially larger values for DataFrames with object or string columns. The parameter is not deprecated.

---

## Q15 — pd.Categorical for Low-Cardinality Strings

> **Week reference:** Week 7

A DataFrame column `df['city']` has dtype `object`, contains 5,000,000 rows, but only 50 unique city names averaging 10 characters each. Converting it to `pd.Categorical` saves memory because:

- A) Categorical stores each string compressed with gzip
- B) Categorical stores integer codes (one small int per row) plus a fixed lookup of the 50 unique strings, instead of 5,000,000 separate string objects
- C) Categorical automatically downcasts all numeric columns too
- D) Categorical removes duplicate rows, reducing the row count

**Answer: B**

- A) Incorrect — Categorical does not apply gzip or any byte-level compression. It uses integer encoding (dictionary encoding), which is a structural change, not a compression algorithm.
- B) Correct — Categorical replaces 5,000,000 Python string objects with 5,000,000 small integers (e.g., int8 or int16 codes) plus a one-time lookup array of 50 unique strings. Memory drops from ~5M × (50+ bytes per object) to ~5M × 1–2 bytes for codes.
- C) Incorrect — `pd.Categorical` only affects the column it is applied to; it does not touch other columns in the DataFrame.
- D) Incorrect — Categorical preserves all rows and their order; it never removes duplicates. The 5,000,000 rows remain; only their internal representation changes.

---

## Q16 — float64 to float16 Memory Reduction

> **Week reference:** Week 2

An array of 1,000,000 float64 values is converted to float16. What is the new memory usage in MB?

- A) 8 MB
- B) 4 MB
- C) 2 MB
- D) 1 MB

**Answer: C**

- A) Incorrect — 8 MB is the original float64 size (1,000,000 × 8 bytes). After conversion to float16 the memory should be smaller, not the same.
- B) Incorrect — 4 MB corresponds to float32 (1,000,000 × 4 bytes). float16 is half the size of float32, giving 2 MB.
- C) Correct — float16 uses 2 bytes per element. 1,000,000 × 2 = 2,000,000 bytes = 2 MB. This is a 4× reduction from the original 8 MB float64 array.
- D) Incorrect — 1 MB would require 1 byte per element (e.g., int8 or uint8). float16 uses 2 bytes, not 1.

---

## Q17 — Total DataFrame Memory Calculation

> **Week reference:** Week 7

A DataFrame has 100,000 rows and 4 columns: two float64 columns, one int32 column, and one int8 column. What is the approximate total memory usage?

- A) 1,700,000 bytes (1.7 MB)
- B) 2,100,000 bytes (2.1 MB)
- C) 2,900,000 bytes (2.9 MB)
- D) 3,200,000 bytes (3.2 MB)

**Answer: B**

Verification: 2 × float64 = 2 × (100,000 × 8) = 1,600,000 bytes; int32 = 100,000 × 4 = 400,000 bytes; int8 = 100,000 × 1 = 100,000 bytes. Total = 2,100,000 bytes = 2.1 MB.

- A) Incorrect — 1.7 MB does not correspond to any combination of these dtypes. The two float64 columns alone contribute 1,600,000 bytes (1.6 MB), so the total must exceed 1.6 MB.
- B) Correct — Memory = (2 × 100,000 × 8) + (100,000 × 4) + (100,000 × 1) = 1,600,000 + 400,000 + 100,000 = 2,100,000 bytes = 2.1 MB. Apply n_elements × bytes_per_dtype for each column and sum.
- C) Incorrect — 2.9 MB would require approximately three float64 columns plus one float32 column. The int32 (4 bytes) and int8 (1 byte) columns are much smaller than float64, so the total falls well below 2.9 MB.
- D) Incorrect — 3.2 MB would result if all four columns were float64 (4 × 100,000 × 8 = 3,200,000). The int32 and int8 columns use far less memory than float64, bringing the total down to 2.1 MB.

---

## Q18 — pd.to_numeric with downcast='float'

> **Week reference:** Week 7

What does `pd.to_numeric(series, downcast='float')` do to a Series currently stored as float64?

- A) Converts to integer dtype if all values are whole numbers
- B) Converts to the smallest float dtype that can represent all values without loss (often float32)
- C) Converts to float16 always, regardless of value range
- D) Has no effect — float64 is already a float dtype so no downcast is possible

**Answer: B**

- A) Incorrect — `downcast='float'` only considers float dtypes, not integer dtypes. To downcast to integers, you would use `downcast='integer'` or `downcast='signed'`.
- B) Correct — `pd.to_numeric(downcast='float')` tries float32 first; if all values fit without precision loss relative to the stored float64 values, it uses float32. This is the standard pandas memory optimization pattern.
- C) Incorrect — pandas does not blindly convert to float16 because float16 has very limited precision (~3 digits) and would corrupt most real-world data. It tries the smallest dtype that preserves values.
- D) Incorrect — `downcast='float'` actively attempts a smaller float dtype. float64 is not the minimum; float32 (and in rare cases float16) are smaller alternatives that pandas will try.

---

## Q19 — int8 Range: Values Including 200

> **Week reference:** Week 2

A NumPy array is created with `np.array([50, 100, 150, 200], dtype=np.int8)`. What are the stored values?

- A) `[50, 100, 127, 127]` — values above 127 are clipped to the max
- B) `[50, 100, 150, 200]` — int8 can hold these values
- C) `[50, 100, -106, -56]` — values above 127 wrap via two's complement
- D) `OverflowError` is raised at construction

**Answer: C**

- A) Incorrect — NumPy does not clip (saturate) on integer overflow. It wraps using two's complement. Saturation semantics exist in some SIMD/DSP contexts but not in standard NumPy.
- B) Incorrect — int8 max is 127. Both 150 and 200 exceed this limit and cannot be stored without wrapping. "int8 can hold these values" is false.
- C) Correct — Two's complement wrapping: 150 − 256 = −106; 200 − 256 = −56. So the array stores `[50, 100, -106, -56]`. No warning is raised.
- D) Incorrect — NumPy integer overflow is always silent. No `OverflowError` is raised; the values wrap silently. This is a key difference from Python's arbitrary-precision `int`.

---

## Q20 — float32 vs float64 for ML Features

> **Week reference:** Week 2

A machine learning pipeline stores a feature matrix of 10,000,000 elements as float64. A colleague suggests switching to float32. Which statement best justifies this change?

- A) float32 cannot be used in NumPy; only float64 is supported for arithmetic
- B) float32 halves memory usage (4 bytes vs 8 bytes per element) with ~7 significant digits of precision, which is sufficient for most ML feature values
- C) float32 is always more accurate than float64 for values between 0 and 1
- D) float32 reduces memory by 75% compared to float64

**Answer: B**

- A) Incorrect — NumPy fully supports float32 arithmetic. Many ML frameworks (PyTorch, TensorFlow) actually prefer float32 as their default dtype for training.
- B) Correct — float32 uses 4 bytes vs float64's 8 bytes, cutting memory in half. Its ~7 significant digits of precision is adequate for normalized ML features (e.g., values scaled to [0, 1] or standardized). Before: 10M × 8 = 80 MB; after: 10M × 4 = 40 MB.
- C) Incorrect — float64 is strictly more precise than float32 for all value ranges, including [0, 1]. float32 gives ~7 digits; float64 gives ~15 digits. More precision is always "more accurate," though the extra precision is often unnecessary for ML.
- D) Incorrect — 75% reduction would require float64→float16 (8 bytes → 2 bytes = 4× reduction). float64→float32 is a 2× reduction = 50% savings, not 75%.

---

## Set 3 — Extended Practice

> Targets astype() semantics, dtype promotion, np.finfo/iinfo, NaN/inf rules, complex dtypes, and uint overflow traps not covered in Sets 1 and 2.

---

## Q21 — astype() Copy vs View Semantics

> **Week reference:** Week 2

**Mental Model:** `astype()` always returns a copy by default, even if the source and target dtypes are identical. This is because dtype conversion semantics are defined as producing a new allocation. Contrast with `.view()`, which reinterprets the existing memory buffer without copying.

Which statement about `arr.astype(np.float32)` is correct when `arr` is a float64 array?

- A) It returns a view of `arr` reinterpreted as float32 — no memory is allocated
- B) It returns a new array with a copied and converted buffer; modifying the result does not affect `arr`
- C) It modifies `arr` in-place and changes its dtype to float32
- D) It raises a `TypeError` because float64 and float32 are incompatible dtypes

**Answer: B**

- A) Incorrect — `astype()` always allocates a new array (copies data) by default. A view reinterpretation with `.view(np.float32)` would be possible but would misinterpret the raw bytes, not perform a proper value conversion.
- B) Correct — `astype()` allocates a new buffer, converts each element from float64 to float32, and returns the new array. The original `arr` is untouched. This is the default behavior (`copy=True`).
- C) Incorrect — NumPy arrays have a fixed dtype after creation; `astype()` never modifies the dtype of the original array in-place. The result is always a separate object.
- D) Incorrect — float64 and float32 are fully compatible numeric dtypes. `astype()` supports conversion between any standard NumPy numeric dtypes, including all float, int, and complex types.

---

## Q22 — dtype Promotion When Mixing float32 and float64

> **Week reference:** Week 2

**Mental Model:** NumPy follows type promotion rules: when operands have different dtypes, the result is promoted to the "higher" dtype to avoid loss of information. float64 > float32 in this hierarchy, so mixing them always produces float64. This is a silent precision preservation, but it also silently undoes any memory savings from downcasting.

A NumPy array `a` has dtype `float32` and array `b` has dtype `float64`. What is the dtype of `a + b`?

- A) `float32` — the lower-precision operand determines the output dtype
- B) `float64` — NumPy promotes to the higher-precision dtype to avoid data loss
- C) `float128` — NumPy always promotes to a higher type than either input
- D) `float32` — NumPy always returns the dtype of the left operand

**Answer: B**

- A) Incorrect — NumPy never demotes precision silently. Returning float32 when one operand is float64 would discard precision, which violates NumPy's type promotion rules.
- B) Correct — NumPy's promotion rules elevate the result to the higher dtype. float64 has more precision than float32, so the result of any arithmetic mixing them is float64. This can silently undo memory savings from downcasting one array.
- C) Incorrect — NumPy does not invent a higher dtype than the inputs; it promotes to the highest dtype present among the operands. float128 would only appear if one of the operands was already float128 (on platforms that support it).
- D) Incorrect — the result dtype depends on both operands' dtypes, not just the left one. `float32_array + float64_array` and `float64_array + float32_array` both produce float64.

---

## Q23 — np.finfo() Fields: eps vs resolution

> **Week reference:** Week 2

**Mental Model:** `np.finfo(dtype).eps` is machine epsilon — the smallest value such that `1.0 + eps != 1.0`. `np.finfo(dtype).resolution` is the approximate decimal resolution (~10× larger than eps, representing decimal digits). These are often confused: eps is the raw ULP at 1.0, resolution is the human-readable decimal precision (~1e-6 for float32, ~1e-15 for float64).

What does `np.finfo(np.float32).eps` represent?

- A) The smallest positive float32 value (subnormal minimum)
- B) The smallest value such that `np.float32(1.0) + eps` is distinguishable from `np.float32(1.0)` — i.e., the ULP at 1.0
- C) The number of decimal digits of precision in float32 (approximately 7)
- D) The maximum representable float32 value (~3.4 × 10³⁸)

**Answer: B**

- A) Incorrect — the smallest positive float32 value (including subnormals) is `np.finfo(np.float32).tiny` or `np.finfo(np.float32).smallest_subnormal`. Machine epsilon is not about the minimum representable value.
- B) Correct — machine epsilon (`eps`) is defined as the smallest positive number such that `1.0 + eps != 1.0`. For float32, this is approximately 1.19 × 10⁻⁷. It equals one ULP (unit in the last place) at value 1.0.
- C) Incorrect — the approximate number of significant decimal digits is given by `np.finfo(np.float32).precision` (which returns 6 for float32). The `eps` attribute is a floating-point value, not an integer digit count.
- D) Incorrect — the maximum representable float32 value is `np.finfo(np.float32).max` ≈ 3.4 × 10³⁸. This is entirely separate from machine epsilon, which is a precision measure near 1.0.

---

## Q24 — float64 Exact Integer Representation Limit

> **Week reference:** Week 2

**Mental Model:** float64 has a 52-bit mantissa, so it can represent integers exactly up to 2^53 = 9,007,199,254,740,992. Beyond this, consecutive integers are no longer distinguishable. This is much larger than float32's limit of 2^24 = 16,777,216, but it still catches users who assume float64 is exact for all integers.

Up to what integer value can `float64` represent all integers exactly?

- A) Up to 2^32 − 1 = 4,294,967,295 (32-bit unsigned integer max)
- B) Up to 2^52 − 1 = 4,503,599,627,370,495
- C) Up to 2^53 = 9,007,199,254,740,992
- D) float64 represents all integers exactly with no upper limit

**Answer: C**

- A) Incorrect — 2^32 − 1 is the limit of uint32, not float64. float64's exact integer range is much larger: its 52-bit mantissa (plus the implicit leading 1) gives exact representation up to 2^53.
- B) Incorrect — the limit is 2^53, not 2^52. The float64 mantissa has 52 explicit bits, but the implicit leading 1 bit gives an effective 53-bit significand, allowing exact integers up to 2^53.
- C) Correct — float64 has a 52-bit explicit mantissa plus an implicit leading 1 bit, giving 53 bits of significand. All integers from −2^53 to 2^53 are representable exactly. Beyond 2^53, consecutive integers merge.
- D) Incorrect — float64 does have an exact integer limit at 2^53. For example, `float64(2**53 + 1)` rounds to `float64(2**53)` — the two values are indistinguishable in float64.

---

## Q25 — NaN Propagation Rules

> **Week reference:** Week 2

**Mental Model:** NaN is "contagious" in IEEE 754 arithmetic: any arithmetic operation involving NaN produces NaN. This means a single NaN in a large array can silently corrupt aggregations (sum, mean, max). Comparisons with NaN always return False, including `NaN == NaN`. NumPy provides `np.nansum`, `np.nanmean` etc. to skip NaN values.

An array contains one `np.nan` value among 999 valid float64 values. Which statement is correct?

- A) `np.sum(arr)` returns the sum of the 999 valid values, ignoring `nan`
- B) `np.sum(arr)` returns `nan` because NaN propagates through arithmetic
- C) `np.sum(arr)` raises a `ValueError` because NaN is not a valid float
- D) `np.sum(arr)` returns `inf` because NaN indicates overflow

**Answer: B**

- A) Incorrect — `np.sum()` does not skip NaN values. Summing NaN with any finite value produces NaN. To skip NaN, use `np.nansum()`.
- B) Correct — IEEE 754 mandates that any arithmetic operation involving NaN returns NaN. A single NaN in the array propagates through the entire sum, returning NaN regardless of the other 999 values.
- C) Incorrect — NaN is a valid IEEE 754 floating-point bit pattern. NumPy never raises a ValueError for NaN in arithmetic. NaN is represented as a float with all exponent bits set and a non-zero mantissa.
- D) Incorrect — NaN and inf are distinct IEEE 754 special values. NaN arises from undefined operations (0/0, inf−inf, sqrt(−1)); inf arises from overflow or non-zero divided by zero. NaN does not become inf through summation.

---

## Q26 — complex64 Memory Layout

> **Week reference:** Week 2

**Mental Model:** NumPy's `complex64` stores a complex number as two consecutive `float32` values (real part then imaginary part). This means each element occupies 8 bytes total — the same as `float64`. `complex128` uses two `float64` values = 16 bytes per element. This surprises users who expect complex64 to be "half the size" of complex128.

How many bytes does each element of a `complex64` NumPy array occupy?

- A) 4 bytes — the same as float32
- B) 8 bytes — two float32 components (real + imaginary)
- C) 16 bytes — two float64 components (real + imaginary)
- D) 6 bytes — three float16 components for real, imaginary, and magnitude

**Answer: B**

- A) Incorrect — `complex64` contains both a real and an imaginary component, each stored as float32 (4 bytes). The total is 4 + 4 = 8 bytes, not 4. A single float32 alone is 4 bytes.
- B) Correct — `complex64` stores two consecutive float32 values: the real part (4 bytes) followed by the imaginary part (4 bytes). Total: 8 bytes per element. Verify with `np.dtype(np.complex64).itemsize == 8`.
- C) Incorrect — 16 bytes per element is `complex128`, which stores two float64 values. `complex64` uses two float32 values = 8 bytes.
- D) Incorrect — complex numbers have exactly two components (real and imaginary). There is no "magnitude" component stored separately. The magnitude is computed from real and imaginary when needed.

---

## Q27 — Mixed-dtype Array Construction Promotion

> **Week reference:** Week 2

**Mental Model:** When `np.array()` constructs an array from a Python list containing mixed numeric types (int and float), it promotes all elements to a single dtype — the minimum dtype that can represent all values without loss. A Python float literal defaults to float64, so mixing Python ints with floats gives float64. Mixing NumPy scalars follows NumPy promotion rules.

What dtype does `np.array([1, 2, 3.0])` have?

- A) `int64` — the majority of values are integers
- B) `object` — mixed types cannot be represented in a single dtype
- C) `float64` — the Python float `3.0` promotes the whole array to float64
- D) `float32` — NumPy defaults to the smallest float dtype

**Answer: C**

- A) Incorrect — `np.array()` does not use majority-voting for dtype selection. A single float literal forces promotion to a float type to avoid discarding the fractional capability, even if that particular value has no fractional part.
- B) Incorrect — `object` dtype is used when values cannot be cast to a common numeric type (e.g., mixing strings with numbers). Integers and floats are fully compatible numeric types and promote to float64.
- C) Correct — Python's `3.0` is a float64 scalar. `np.array()` promotes all elements to the minimum common dtype, which is float64. The integers `1` and `2` are losslessly converted to float64.
- D) Incorrect — NumPy does not default to float32 for Python floats; Python floats are 64-bit (`float64`). float32 would only appear if the elements were explicitly `np.float32` scalars or the `dtype=` argument specified it.

---

## Q28 — uint16 Overflow Wrapping

> **Week reference:** Week 2

**Mental Model:** uint16 max is 65535 = 2^16 − 1. Adding 1 to 65535 in uint16 wraps to 0 (modulo 2^16 arithmetic). This is the unsigned equivalent of the int16 signed overflow trap. Confusion arises because 65535 is often mistaken for int16's max (which is only 32767).

What is the result of `np.uint16(65535) + np.uint16(1)`?

- A) 65536
- B) 32767
- C) -1
- D) 0

**Answer: D**

- A) Incorrect — 65536 exceeds the uint16 range of 0 to 65535. Arithmetic wraps modulo 2^16: 65535 + 1 = 65536 mod 65536 = 0.
- B) Incorrect — 32767 is the maximum value of the signed `int16` type, not uint16. The uint16 max is 65535 (all 16 bits set).
- C) Incorrect — negative values are impossible in uint16 (unsigned). The bit pattern that int16 would interpret as −1 (`0xFFFF`) is interpreted as 65535 in uint16. Overflow wraps to 0, not −1.
- D) Correct — uint16 arithmetic is modulo 2^16 = 65536. `65535 + 1 = 65536`, and `65536 mod 65536 = 0`. This is the unsigned equivalent of int16's boundary wrap from 32767 to −32768.

---

## Q29 — np.iinfo() for Safe Downcast Decisions

> **Week reference:** Week 2

**Mental Model:** `np.iinfo(dtype)` returns an object with `.min` and `.max` attributes giving the exact integer bounds for that dtype. This is the canonical way to check downcast safety programmatically, rather than hard-coding magic numbers like 127 or 32767. Exam questions often ask which attributes to use or what values they return.

What does `np.iinfo(np.int16).max` return?

- A) 255
- B) 32767
- C) 65535
- D) 2147483647

**Answer: B**

- A) Incorrect — 255 is the maximum of `uint8` (unsigned 8-bit), not int16. `np.iinfo(np.uint8).max == 255`.
- B) Correct — int16 is a signed 16-bit integer: range = −2^15 to 2^15 − 1 = −32768 to 32767. `np.iinfo(np.int16).max` returns 32767. This is the value to check when deciding if a column's max fits in int16.
- C) Incorrect — 65535 = 2^16 − 1 is the maximum of `uint16` (unsigned 16-bit). `np.iinfo(np.uint16).max == 65535`. Confusing int16 max (32767) with uint16 max (65535) is the most common int16 exam trap.
- D) Incorrect — 2,147,483,647 = 2^31 − 1 is the maximum of `int32`. `np.iinfo(np.int32).max == 2147483647`. int16 is only 16 bits wide, giving a much smaller range.

---

## Q30 — inf Arithmetic and NaN-Producing Operations

> **Week reference:** Week 2

**Mental Model:** IEEE 754 defines several operations that produce NaN from inf: `inf - inf`, `inf * 0`, `inf / inf`, and `0 / 0`. Operations like `inf + finite`, `inf * positive`, `1 / 0` all produce inf, not NaN. This is a common exam trap: students assume any inf-involving arithmetic produces NaN.

Which of the following IEEE 754 floating-point operations produces `nan` (not `inf`)?

- A) `np.float64(1.0) / np.float64(0.0)`
- B) `np.float64('inf') + np.float64(1e300)`
- C) `np.float64('inf') - np.float64('inf')`
- D) `np.float64('inf') * np.float64(2.0)`

**Answer: C**

- A) Incorrect — dividing a non-zero finite value by zero produces `inf` (positive infinity), not NaN. IEEE 754 defines `1.0 / 0.0 = +inf`. NaN arises from indeterminate forms like `0.0 / 0.0`.
- B) Incorrect — adding a finite value (even a very large one like 1e300) to infinity still gives infinity: `inf + finite = inf`. The result does not become NaN because the operation is not indeterminate.
- C) Correct — `inf - inf` is an indeterminate form (like ∞ − ∞ in mathematics), so IEEE 754 defines it as NaN. This is one of the standard NaN-producing operations alongside `0/0`, `0*inf`, and `inf/inf`.
- D) Incorrect — multiplying infinity by a positive finite value produces infinity: `inf * 2.0 = inf`. NaN only arises when multiplying `inf * 0` (indeterminate form `0 × ∞`), not `inf * positive_nonzero`.

---
