# Pandas Dtype Optimization & Chunking — MCQ Practice

> Topics: dtype downcast decisions, object→datetime/category, memory budget calculation, chunked reading.
> Exam frequency: **2024 exam + F25**.

---

## Q1 — int64 mach_id Downcast

> **Week reference:** Week 7
> **Mental Model:** Work through a checklist: is the range negative? (eliminates uint*). What's the max value? Match it against int8 (127), int16 (32,767), int32 (~2.1B). Pick the smallest that covers both ends of the actual range.

A DataFrame has a column `mach_id` stored as `int64` with observed values ranging from -1 to 5730. Which dtype minimises memory usage while safely holding all values?

- A) `uint8`
- B) `int8`
- C) `int16`
- D) `int32`

**Answer: C**

- A) Incorrect — uint8 holds 0 to 255; the value -1 is negative and would silently overflow to 255 (uint8 wraps around), corrupting the data. Any negative value eliminates all uint* types.
- B) Incorrect — int8 holds -128 to 127; the maximum observed value of 5730 far exceeds 127 and would overflow. int8 can hold -1, but not 5730.
- C) Correct — int16 holds -32,768 to 32,767, which covers the full range -1 to 5730. Using int16 saves 75% memory vs int64 (2 bytes vs 8 bytes per element), a 4× reduction per row.
- D) Incorrect — int32 holds ±2,147,483,647, which also fits but uses 4 bytes per element (only 2× reduction vs int64). int16 at 2 bytes is the smallest type that safely holds 5730, so int32 wastes 2 bytes per row unnecessarily.

---

## Q2 — int64 version Column Downcast

> **Week reference:** Week 7
> **Mental Model:** When all values are non-negative, consider uint* types — they double the positive range vs int* at the same byte width. uint8 (0–255) vs int8 (-128–127): same 1 byte, but uint8 covers 0–42 more semantically clearly for a non-negative column.

A `version` column is stored as `int64` with all values in the range 0 to 42. Which dtype is the smallest that can safely represent all values?

- A) `int8`
- B) `uint8`
- C) `int16`
- D) `float32`

**Answer: B**

- A) Incorrect — int8 holds -128 to 127, which also fits 0 to 42. Both int8 and uint8 are 1 byte, so memory is identical. However, int8 wastes the negative range on a column that is never negative; uint8 is semantically correct and is the standard answer when data is non-negative and fits in 0–255.
- B) Correct — uint8 holds 0 to 255, perfectly covering 0–42, and uses only 1 byte per element (8× reduction vs int64). It is the smallest possible dtype that can hold these values without wasting range.
- C) Incorrect — int16 (2 bytes) also fits 0–42, but uint8 (1 byte) is smaller and preferred here. Using int16 wastes a byte per element with no benefit.
- D) Incorrect — converting integers to float32 risks precision loss for integers above 2^24 (~16.7M) and wastes bytes. float32 has ~7 significant decimal digits; for small integers it works, but int8/uint8 is always preferable for integer data.

---

## Q3 — Low-Cardinality Object Column

> **Week reference:** Week 7
> **Mental Model:** category stores a compact lookup table of unique strings + an integer code per row. With 8 unique values across 500,000 rows, the table has 8 entries; the codes use int8 (1 byte). vs object: 8 bytes (pointer) + heap string per row. Savings are dramatic at low cardinality.

A `location` column is stored as `object` (strings) and has exactly 8 unique values out of 500,000 rows. What is the most memory-efficient dtype to convert it to?

- A) `int8`
- B) `datetime64`
- C) `category`
- D) Keep as `object`; no savings possible

**Answer: C**

- A) Incorrect — int8 is a numeric type; string data cannot be meaningfully encoded as int8 without manual label mapping. Even if you manually mapped strings to integers, the result would lose the original string labels and require external decoding logic.
- B) Incorrect — datetime64 is for timestamps, not arbitrary string categories. Converting location names like "Copenhagen" to datetime64 is semantically invalid and would fail with a parse error.
- C) Correct — category stores a lookup table of 8 unique strings plus an integer code per row. With 500,000 rows: object uses ~500,000 × (8 byte pointer + heap string) ≈ 30+ MB; category uses ~8 unique strings + 500,000 × 1 byte codes ≈ 500 KB. Savings are roughly 60×.
- D) Incorrect — object columns store a Python object reference (8-byte pointer) per row, plus the heap-allocated Python str object for each unique string referenced. category dramatically reduces this by storing unique strings once and using small integer codes for each row.

---

## Q4 — High-Cardinality Object Column with Timestamps

> **Week reference:** Week 7
> **Mental Model:** category breaks down when cardinality is high. The lookup table grows linearly with unique values. datetime64 is the correct type for date strings regardless of cardinality — 8 bytes/element like float64, but enables vectorized date arithmetic and sorted-index binary search.

A `date` column is stored as `object` and contains 70,079 unique string values formatted as `"2023-01-15"`. Which conversion gives the best memory and performance outcome?

- A) Convert to `category` — many unique values still benefit from the lookup table
- B) Convert to `datetime64` — timestamp parsing enables efficient time-based indexing
- C) Keep as `object` — with this many unique values no conversion helps
- D) Convert to `float64` to store as Unix timestamps

**Answer: B**

- A) Incorrect — category overhead outweighs savings when cardinality is high. With ~70,000 unique values, the lookup table stores 70,000 strings. The category codes would need int16 or int32 (not int8), using 2-4 bytes/row. Compared to datetime64's 8 bytes/row, category may save nothing while adding complexity and slowing date arithmetic.
- B) Correct — datetime64 uses 8 bytes/element as a proper 64-bit integer timestamp (nanoseconds since epoch). It enables vectorized date arithmetic (e.g., `df['date'].dt.month`), allows sorted-index binary search for O(log n) lookups, and is Pandas' native representation for time series data.
- C) Incorrect — object wastes memory (8-byte pointer + heap string object per row) and makes date arithmetic slow (requires string parsing on each operation). Converting to datetime64 always helps for date strings.
- D) Incorrect — float64 also uses 8 bytes, equal in size to datetime64, but loses the entire pandas datetime API (`.dt.year`, `.dt.month`, `resample`, `to_period`, etc.) and makes the column human-unreadable. No benefit over datetime64.

---

## Q5 — int64 units Column Exceeding int16

> **Week reference:** Week 7
> **Mental Model:** The classic exam trap: confusing int16 (signed, max 32,767 = 2^15 − 1) with uint16 (unsigned, max 65,535 = 2^16 − 1). Always verify: signed int16 max = 32,767. If max value > 32,767, int16 overflows even if the value looks "close to 65,535."

A `units` column is `int64` with observed values from 932 to 68,837. A teammate suggests downcasting to `int16`. Is this correct?

- A) Yes — int16 holds up to 65,535 which covers 68,837
- B) No — int16 only holds up to 32,767; use int32 instead
- C) No — int16 only holds up to 32,767; use int8 instead (smallest integer type)
- D) Yes — int16 holds up to 68,837 because values are non-negative

**Answer: B**

- A) Incorrect — 65,535 is the maximum for *uint16* (unsigned 16-bit), not int16. int16 is signed and uses one bit for the sign, giving a range of -32,768 to +32,767 (= 2^15 − 1). Since 68,837 > 32,767, int16 would overflow and corrupt the data.
- B) Correct — int16 max is 32,767 < 68,837, so int16 overflows. int32 has range -2,147,483,648 to +2,147,483,647, which safely holds 68,837 using 4 bytes per element (2× savings vs int64). Since values are non-negative, uint16 (max 65,535) is also too small for 68,837; uint32 would be another valid option.
- C) Incorrect — int8 max is only 127, which is far below 68,837 and even below the minimum 932. Using int8 would cause catastrophic data corruption. This is wrong in two ways: the reasoning about int8 being applicable, and suggesting a type that's even smaller.
- D) Incorrect — int16 range is determined by the signed bit representation, not by whether stored values happen to be non-negative. The range is ±32,767 regardless. Non-negative values would need uint16 (max 65,535), which also doesn't cover 68,837. The signedness of the data doesn't change the type's maximum.

---

## Q6 — Integer-to-Float Precision Trap

> **Week reference:** Week 7
> **Mental Model:** float32 has 24 mantissa bits → ~7 significant decimal digits. Any integer larger than 2^24 = 16,777,216 cannot be represented exactly as float32. For order IDs > 16M, adjacent integers round to the same float32 value, causing silent equality collisions.

A colleague proposes storing a column of order IDs (integer values up to 10,000,000) as `float32` to "save space over int64". What is the main risk?

- A) float32 uses more memory than int32 for the same data
- B) float32 has only ~7 decimal digits of precision, so large integers may be rounded
- C) Pandas does not support float32 columns
- D) float32 cannot represent numbers above 65,535

**Answer: B**

- A) Incorrect — float32 and int32 both use 4 bytes per element, so memory is equal, not larger. The colleague's proposal would save memory vs int64 (4 bytes vs 8 bytes), but the better alternative is int32, not float32.
- B) Correct — float32 has a 24-bit mantissa, giving ~7 significant decimal digits. 10,000,000 is 8 digits, right at the precision limit. `float32(10_000_001)` may round to exactly `float32(10_000_000)`, making two different order IDs appear identical. Use int32 instead: same 4 bytes, exact representation, no precision loss.
- C) Incorrect — Pandas fully supports float32 columns. `df['col'].astype('float32')` works without issue. The problem is precision, not support.
- D) Incorrect — float32 can represent very large numbers (up to ~3.4×10^38 in magnitude); the IEEE 754 exponent field handles this. The issue is precision (significant digits), not range. float32 can represent 10,000,000 as a number, but may not distinguish it from 10,000,001.

---

## Q7 — Memory Budget Calculation (Mixed Dtypes)

> **Week reference:** Week 7
> **Mental Model:** bytes_per_row = sum of each column's dtype bytes. Then max_rows = budget_bytes / bytes_per_row. Always compute bytes_per_row first, then divide. Common dtype sizes: float64=8, float32=4, int64=8, int32=4, int16=2, int8=1, uint8=1.

A DataFrame has three columns: `price` (float64), `count` (int32), and `flag` (uint8). You have a memory budget of 130 MB (130 × 10^6 bytes). Approximately how many rows can the DataFrame hold within budget?

- A) 1,000,000
- B) 5,000,000
- C) 10,000,000
- D) 16,250,000

**Answer: C**

- A) Incorrect — 1,000,000 rows × 13 bytes/row = 13,000,000 bytes = 13 MB, far below the 130 MB budget. You could fit 10× more rows.
- B) Incorrect — 5,000,000 rows × 13 bytes/row = 65,000,000 bytes = 65 MB, still only half the budget. You could fit 2× more rows.
- C) Correct — bytes per row = float64(8) + int32(4) + uint8(1) = 13 bytes. max_rows = 130,000,000 / 13 ≈ 10,000,000 rows. This is also a useful way to plan chunked reading: if budget = 130 MB, read in chunks of 10M rows.
- D) Incorrect — 16,250,000 rows × 13 bytes/row = 211,250,000 bytes ≈ 211 MB, which exceeds the 130 MB budget by 62%. This would cause memory pressure or an OOM error.

---

## Q8 — Memory Budget Calculation (All int64)

> **Week reference:** Week 7
> **Mental Model:** bytes_per_row = ncols × 8 for all-int64. With 3 columns: 24 bytes/row. Then max_rows = budget / 24. This pattern appears directly in chunked-reading code: chunksize = available_memory_bytes / bytes_per_row.

A DataFrame has three columns all stored as `int64`. You need to load it into 24 MB (24 × 10^6 bytes) of RAM. What is the maximum number of rows you can load at once?

- A) 333,333
- B) 500,000
- C) 1,000,000
- D) 3,000,000

**Answer: C**

- A) Incorrect — 333,333 rows × 24 bytes/row = 8,000,000 bytes = 8 MB. This would use only 1/3 of the available budget, leaving 16 MB unused. The formula would give 3 bytes/row, not 24 bytes/row.
- B) Incorrect — 500,000 rows × 24 bytes/row = 12,000,000 bytes = 12 MB. This is half the budget, corresponding to 16 bytes/row (2 int64 columns), not 3 columns.
- C) Correct — bytes per row = 3 × int64(8) = 24 bytes. max_rows = 24,000,000 / 24 = 1,000,000 rows. This is the exact budget utilization — 1,000,000 rows × 24 bytes = 24,000,000 bytes = 24 MB.
- D) Incorrect — 3,000,000 rows × 24 bytes/row = 72,000,000 bytes = 72 MB, which is 3× the 24 MB budget. Loading this many rows would require 72 MB, causing an OOM error on a 24 MB budget.

---

## Q9 — Chunked Reading Limitations

> **Week reference:** Week 7
> **Mental Model:** A chunked reader is a forward-only iterator (like a file handle). You can iterate through it once from start to finish. You cannot rewind, jump, or index into it. To re-read, call pd.read_csv again. len() and indexing both raise errors.

You open a CSV using `pd.read_csv("data.csv", chunksize=100_000)` and iterate over the resulting object. Which statement is TRUE?

- A) `len(reader)` returns the total number of chunks
- B) You can index individual chunks with `reader[2]` to get the third chunk
- C) Each iteration yields a DataFrame of up to 100,000 rows, but you cannot random-access chunks
- D) After the first full iteration, the second `for chunk in reader:` loop automatically re-reads from the start

**Answer: C**

- A) Incorrect — the TextFileReader object does not support `len()`. Calling `len(reader)` raises a TypeError. The total chunk count is unknown until the entire file is read (you'd need to count rows first, then divide by chunksize).
- B) Incorrect — chunked readers are forward-only iterators, not indexable sequences. `reader[2]` raises a TypeError. To get a specific chunk, you must iterate up to it sequentially, which defeats random access.
- C) Correct — the reader is a forward-only iterator. Each `next()` call reads the next chunksize rows and returns a DataFrame. The last chunk may have fewer than 100,000 rows. You cannot skip, rewind, or jump to arbitrary positions.
- D) Incorrect — once the iterator is exhausted (all chunks read), it is spent. A second `for chunk in reader:` loop immediately exits with zero iterations — no data is re-read. You must call `pd.read_csv(..., chunksize=100_000)` again to get a fresh iterator.

---

## Q10 — memory_usage deep=True vs deep=False

> **Week reference:** Week 7
> **Mental Model:** Without deep=True, Pandas reports only the size of the pointer array (8 bytes × nrows) for object columns — it doesn't follow the pointers into the heap. This massively understates memory for string columns. Always use deep=True when auditing actual memory usage.

A DataFrame has an `object` dtype column containing variable-length strings. You call `df.memory_usage()` (without `deep=True`). What does the reported size for that column represent?

- A) The total number of bytes used by all strings on the heap
- B) Only the 8-byte object pointer per row, not the actual string content
- C) The average string length times the number of rows
- D) An exact count including Python object headers for each string

**Answer: B**

- A) Incorrect — without `deep=True`, Pandas does not traverse into the heap-allocated Python string objects. It only counts the pointer array. To get the actual heap string bytes, you must pass `deep=True`, which traverses each object via `sys.getsizeof`.
- B) Correct — each object column entry is stored as a Python object reference (an 8-byte pointer on 64-bit systems). The shallow estimate reports nrows × 8 bytes for an object column, regardless of string lengths. For a column with 1M rows of 50-character strings, the true memory is ~100MB, but shallow reports only ~8MB.
- C) Incorrect — Pandas does not compute or use average string length for the shallow memory estimate. It simply counts the pointer array size, which is always exactly nrows × 8 bytes for object columns.
- D) Incorrect — full Python object header traversal (including the 56-byte str header + character data) requires `deep=True`. The default shallow estimate intentionally skips this for performance reasons, at the cost of accuracy for object columns.

---

## Q11 — Sorted Index for Repeated Date Lookups

> **Week reference:** Week 7
> **Mental Model:** Sorted DatetimeIndex → O(log n) binary search per lookup. Unsorted column scan → O(n) per lookup. For thousands of queries on millions of rows: O(n) scan is prohibitive; sorted index is essential. set_index + sort_index is the standard pattern.

You have a DataFrame with a `date` column and need to perform thousands of point-lookups by date. Which transformation gives the best lookup performance?

- A) Convert `date` to `category` dtype and use `.loc[]`
- B) Call `df.set_index('date').sort_index()` to enable binary search
- C) Keep `date` as `object` and use `df[df['date'] == target]`
- D) Store dates as `float64` Unix timestamps for faster comparison

**Answer: B**

- A) Incorrect — category dtype speeds up groupby/value_counts operations (replacing hash lookups with integer comparisons) but does not convert O(n) scans to O(log n) for `.loc[]` lookups. Without a sorted index, each lookup still scans all rows.
- B) Correct — `set_index('date').sort_index()` creates a sorted DatetimeIndex, enabling Pandas to use O(log n) binary search (via `searchsorted`) instead of O(n) linear scan. For 1M rows and 10,000 queries: 10,000 × log₂(1M) ≈ 200,000 operations vs 10,000 × 1,000,000 = 10 billion with linear scan.
- C) Incorrect — boolean masking (`df[df['date'] == target]`) scans all n rows for every query. For 10,000 queries on 1M rows, this is 10 billion comparisons — roughly 5 orders of magnitude slower than the sorted index approach.
- D) Incorrect — float64 comparison is faster element-wise than string comparison, but it still requires an O(n) linear scan without a sorted index. The bottleneck is the scan, not the comparison operator.

---

## Q12 — High-Cardinality String Column Decision

> **Week reference:** Week 7
> **Mental Model:** category efficiency = (unique_count / total_rows). When this ratio approaches 1 (near-unique), the lookup table is nearly as large as the data itself, and codes need wider integers (int16/int32). The crossover point is roughly unique_count > 1,000–5,000 unique values; beyond that, category rarely helps.

A `product_sku` column is stored as `object` with 50,000 unique string values across 60,000 rows. A teammate wants to convert it to `category`. What is the best advice?

- A) Convert to `category` — any reduction in unique values helps
- B) Convert to `datetime64` if the SKUs look like timestamps; otherwise keep as object
- C) Do not convert to `category`; at near-unique cardinality the overhead can exceed the savings
- D) Convert to `int64` by hashing the strings for maximum compression

**Answer: C**

- A) Incorrect — category is only memory-efficient at low cardinality (typically < 1,000–5,000 unique values per 60,000 rows). At 50,000 unique SKUs, the category lookup table stores 50,000 strings plus int16 or int32 codes for 60,000 rows. The table alone may exceed the savings from replacing pointer-per-row storage.
- B) Incorrect — SKU codes are product identifiers, not timestamps. Converting to datetime64 is semantically wrong and would fail with a parse error on strings like "SKU-00043A". This answer conflates the dtype rule for timestamps with a different question.
- C) Correct — with 50,000 unique values across 60,000 rows (83% uniqueness), the category overhead is severe. The lookup table holds 50,000 strings; the codes require int16 or int32 (2–4 bytes/row) since codes won't fit in int8. Compare: object = 8 bytes/row (pointer) + heap; category ≈ 4 bytes/row (int32 code) + 50,000-string table. Savings are marginal at best, negative at worst.
- D) Incorrect — hashing destroys the original string values (hash collisions could map different SKUs to the same integer, and you cannot recover the original string from a hash). This is not a standard Pandas optimisation pattern and would corrupt the data semantically.

---
