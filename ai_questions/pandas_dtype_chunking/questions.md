# Pandas Dtype Optimization & Chunking — MCQ Practice

> Topics: dtype downcast decisions, object→datetime/category, memory budget calculation, chunked reading.
> Exam frequency: **2024 exam + F25**.

---

## Q1 — int64 mach_id Downcast

> **Week reference:** Week 7

A DataFrame has a column `mach_id` stored as `int64` with observed values ranging from -1 to 5730. Which dtype minimises memory usage while safely holding all values?

- A) `uint8`
- B) `int8`
- C) `int16`
- D) `int32`

**Answer: C**

- A) Incorrect — uint8 holds 0–255; the value -1 is negative and would overflow.
- B) Incorrect — int8 holds -128 to 127; 5730 exceeds 127 and would overflow.
- C) Correct — int16 holds -32,768 to 32,767, which covers -1 to 5730, saving 75% vs int64.
- D) Incorrect — int32 would also work but is larger than necessary (4 bytes vs 2 bytes for int16).

---

## Q2 — int64 version Column Downcast

> **Week reference:** Week 7

A `version` column is stored as `int64` with all values in the range 0 to 42. Which dtype is the smallest that can safely represent all values?

- A) `int8`
- B) `uint8`
- C) `int16`
- D) `float32`

**Answer: B**

- A) Incorrect — int8 also holds 0–42 (max 127) and is the same 1-byte size as uint8; however int8 wastes the negative range on non-negative data, making uint8 the more semantically correct choice.
- B) Correct — uint8 holds 0 to 255, perfectly covering 0–42, and uses only 1 byte per element.
- C) Incorrect — int16 (2 bytes) also fits, but uint8 (1 byte) is smaller and preferred here.
- D) Incorrect — converting integers to float32 risks precision loss for large integers and wastes bytes unnecessarily.

---

## Q3 — Low-Cardinality Object Column

> **Week reference:** Week 7

A `location` column is stored as `object` (strings) and has exactly 8 unique values out of 500,000 rows. What is the most memory-efficient dtype to convert it to?

- A) `int8`
- B) `datetime64`
- C) `category`
- D) Keep as `object`; no savings possible

**Answer: C**

- A) Incorrect — int8 is a numeric type; string data cannot be meaningfully encoded as int8 without manual label mapping.
- B) Incorrect — datetime64 is for timestamps, not arbitrary string categories.
- C) Correct — category stores a lookup table of 8 unique strings plus an integer code per row, giving huge memory savings over storing the full string pointer for each of 500,000 rows.
- D) Incorrect — object columns store a Python string object pointer (8 bytes) per row plus heap allocation; category dramatically reduces this.

---

## Q4 — High-Cardinality Object Column with Timestamps

> **Week reference:** Week 7

A `date` column is stored as `object` and contains 70,079 unique string values formatted as `"2023-01-15"`. Which conversion gives the best memory and performance outcome?

- A) Convert to `category` — many unique values still benefit from the lookup table
- B) Convert to `datetime64` — timestamp parsing enables efficient time-based indexing
- C) Keep as `object` — with this many unique values no conversion helps
- D) Convert to `float64` to store as Unix timestamps

**Answer: B**

- A) Incorrect — category overhead outweighs savings when cardinality is high (~70,000 unique values); the lookup table itself becomes large.
- B) Correct — datetime64 uses 8 bytes/element as a proper numeric timestamp, enables vectorised date arithmetic, and allows sorted-index binary search.
- C) Incorrect — object wastes memory and makes date arithmetic slow; conversion always helps here.
- D) Incorrect — float64 also uses 8 bytes but loses the pandas datetime API and readability.

---

## Q5 — int64 units Column Exceeding int16

> **Week reference:** Week 7

A `units` column is `int64` with observed values from 932 to 68,837. A teammate suggests downcasting to `int16`. Is this correct?

- A) Yes — int16 holds up to 65,535 which covers 68,837
- B) No — int16 only holds up to 32,767; use int32 instead
- C) No — int16 only holds up to 32,767; use int8 instead (smallest integer type)
- D) Yes — int16 holds up to 68,837 because values are non-negative

**Answer: B**

- A) Incorrect — 65,535 is the max for *uint16*, not int16; int16 max is 32,767 (= 2^15 − 1).
- B) Correct — int16 max is 32,767 < 68,837; int32 (range −2,147,483,648 to +2,147,483,647) safely holds the full range with 4 bytes per element.
- C) Incorrect — int8 max is only 127, which is far below 68,837; this would cause data corruption.
- D) Incorrect — int16 range is ±32,767 regardless of the sign of the values stored.

---

## Q6 — Integer-to-Float Precision Trap

> **Week reference:** Week 7

A colleague proposes storing a column of order IDs (integer values up to 10,000,000) as `float32` to "save space over int64". What is the main risk?

- A) float32 uses more memory than int32 for the same data
- B) float32 has only ~7 decimal digits of precision, so large integers may be rounded
- C) Pandas does not support float32 columns
- D) float32 cannot represent numbers above 65,535

**Answer: B**

- A) Incorrect — float32 and int32 both use 4 bytes, so memory is equal, not larger.
- B) Correct — float32 has ~7 significant decimal digits; `float32(10_000_000) + float32(1)` may evaluate to 10,000,000.0 due to rounding, corrupting integer data.
- C) Incorrect — Pandas fully supports float32 columns.
- D) Incorrect — float32 can represent very large numbers (up to ~3.4×10^38); the issue is precision, not range.

---

## Q7 — Memory Budget Calculation (Mixed Dtypes)

> **Week reference:** Week 7

A DataFrame has three columns: `price` (float64), `count` (int32), and `flag` (uint8). You have a memory budget of 130 MB (130 × 10^6 bytes). Approximately how many rows can the DataFrame hold within budget?

- A) 1,000,000
- B) 5,000,000
- C) 10,000,000
- D) 16,250,000

**Answer: C**

- A) Incorrect — 1,000,000 rows would use only 13 MB, far below the budget.
- B) Incorrect — 5,000,000 rows would use 65 MB, still below budget.
- C) Correct — bytes per row = float64(8) + int32(4) + uint8(1) = 13 bytes; 130,000,000 / 13 ≈ 10,000,000 rows.
- D) Incorrect — 16,250,000 × 13 = 211 MB, which exceeds the 130 MB budget.

---

## Q8 — Memory Budget Calculation (All int64)

> **Week reference:** Week 7

A DataFrame has three columns all stored as `int64`. You need to load it into 24 MB (24 × 10^6 bytes) of RAM. What is the maximum number of rows you can load at once?

- A) 333,333
- B) 500,000
- C) 1,000,000
- D) 3,000,000

**Answer: C**

- A) Incorrect — that would be for 3 bytes/row; int64 is 8 bytes each.
- B) Incorrect — that would be for 16 bytes/row (2 int64 columns).
- C) Correct — bytes per row = 3 × int64(8) = 24 bytes; 24,000,000 / 24 = 1,000,000 rows.
- D) Incorrect — 3,000,000 × 24 = 72 MB, which exceeds the budget.

---

## Q9 — Chunked Reading Limitations

> **Week reference:** Week 7

You open a CSV using `pd.read_csv("data.csv", chunksize=100_000)` and iterate over the resulting object. Which statement is TRUE?

- A) `len(reader)` returns the total number of chunks
- B) You can index individual chunks with `reader[2]` to get the third chunk
- C) Each iteration yields a DataFrame of up to 100,000 rows, but you cannot random-access chunks
- D) After the first full iteration, the second `for chunk in reader:` loop automatically re-reads from the start

**Answer: C**

- A) Incorrect — the TextFileReader object does not support `len()`; you cannot know the total chunk count without reading.
- B) Incorrect — chunked readers are iterators, not indexable sequences; `reader[2]` raises a TypeError.
- C) Correct — the reader is a forward-only iterator; you get chunks sequentially but cannot index or re-iterate without reopening the file.
- D) Incorrect — once exhausted, the iterator is spent; re-iterating yields nothing without calling `pd.read_csv` again.

---

## Q10 — memory_usage deep=True vs deep=False

> **Week reference:** Week 7

A DataFrame has an `object` dtype column containing variable-length strings. You call `df.memory_usage()` (without `deep=True`). What does the reported size for that column represent?

- A) The total number of bytes used by all strings on the heap
- B) Only the 8-byte object pointer per row, not the actual string content
- C) The average string length times the number of rows
- D) An exact count including Python object headers for each string

**Answer: B**

- A) Incorrect — without `deep=True`, Pandas does not traverse into the heap-allocated string objects.
- B) Correct — each object column entry is a Python object reference (pointer), which is 8 bytes; the actual string content is not counted unless `deep=True` is passed.
- C) Incorrect — Pandas does not compute average string length for the shallow estimate.
- D) Incorrect — full object header traversal requires `deep=True`.

---

## Q11 — Sorted Index for Repeated Date Lookups

> **Week reference:** Week 7

You have a DataFrame with a `date` column and need to perform thousands of point-lookups by date. Which transformation gives the best lookup performance?

- A) Convert `date` to `category` dtype and use `.loc[]`
- B) Call `df.set_index('date').sort_index()` to enable binary search
- C) Keep `date` as `object` and use `df[df['date'] == target]`
- D) Store dates as `float64` Unix timestamps for faster comparison

**Answer: B**

- A) Incorrect — category dtype speeds up groupby/value_counts but does not convert O(n) scans to O(log n) for `.loc[]` lookups.
- B) Correct — a sorted DatetimeIndex allows Pandas to use O(log n) binary search instead of O(n) linear scan, crucial when thousands of queries are issued.
- C) Incorrect — boolean masking on an unsorted column is O(n) per query, which is slow for many repeated lookups.
- D) Incorrect — float64 comparison is faster element-wise but still requires an O(n) scan without a sorted index.

---

## Q12 — High-Cardinality String Column Decision

> **Week reference:** Week 7

A `product_sku` column is stored as `object` with 50,000 unique string values across 60,000 rows. A teammate wants to convert it to `category`. What is the best advice?

- A) Convert to `category` — any reduction in unique values helps
- B) Convert to `datetime64` if the SKUs look like timestamps; otherwise keep as object
- C) Do not convert to `category`; at near-unique cardinality the overhead can exceed the savings
- D) Convert to `int64` by hashing the strings for maximum compression

**Answer: C**

- A) Incorrect — category is only efficient at low cardinality (typically < 1,000 unique values); the lookup table itself consumes significant memory at 50,000 unique strings.
- B) Incorrect — SKU codes are not timestamps; converting to datetime64 is semantically wrong.
- C) Correct — with 50,000 unique values in 60,000 rows, the category lookup table is almost as large as the data itself, providing little or no memory saving and adding overhead.
- D) Incorrect — hashing destroys the original string data and is not a standard Pandas optimisation pattern.

---
