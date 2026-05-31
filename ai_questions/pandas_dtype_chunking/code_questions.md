# Pandas Dtype Optimization & Chunking — Code-Based MCQ Practice

> Format: Each question shows a DataFrame summary, code snippet, or memory_usage output to interpret.
> Exam frequency: **2024 exam + F25**.

---

## Question 1

You run `df.info()` and get the following summary for a DataFrame with 500,000 rows:

```
Column        Dtype   Non-Null   Unique   Min        Max
timestamp     object  500000     500000   2023-01-01 2023-12-31
station_id    int64   500000     8        1          8
temperature   float64 500000     -        -15.3      42.8
reading_code  object  500000     6        -          -
```

Which single column would benefit most from conversion to `category` dtype?

- A) `timestamp`
- B) `station_id`
- C) `temperature`
- D) `reading_code`

**Answer: D) `reading_code`**

**Explanation:** `reading_code` has only 6 unique values across 500,000 rows — a perfect cardinality ratio for `category`. Pandas `category` dtype stores an integer code per row plus a lookup table of unique labels, giving massive memory savings when unique values are far fewer than total rows. `station_id` also has low cardinality (8 unique) but is already numeric; converting it to a smaller integer type (e.g. `uint8`) is more appropriate. `timestamp` should become `datetime64`. `temperature` is a continuous float — `category` would be counterproductive.

---

## Question 2

Using the same DataFrame summary from Question 1:

```
Column        Dtype   Non-Null   Unique   Min        Max
timestamp     object  500000     500000   2023-01-01 2023-12-31
station_id    int64   500000     8        1          8
temperature   float64 500000     -        -15.3      42.8
reading_code  object  500000     6        -          -
```

Which column should be converted to `datetime64[ns]`?

- A) `timestamp` — it has 500,000 unique date strings
- B) `station_id` — it is a time-series index
- C) `temperature` — it contains time-varying measurements
- D) `reading_code` — it encodes temporal events

**Answer: A) `timestamp`**

**Explanation:** `timestamp` is currently an `object` column containing date strings (e.g. `"2023-01-01"`). Converting it to `datetime64[ns]` with `pd.to_datetime()` enables efficient time-based operations (`.dt` accessor, `.resample()`, time-aware indexing) and reduces memory from ~50 bytes/element (Python string heap) to exactly 8 bytes/element (int64 nanoseconds under the hood). The 500,000 unique values confirm these are individual timestamps, not a categorical-style column.

---

## Question 3

You inspect a column in a DataFrame with 500,000 rows:

```python
df['station_id'].dtype          # int64
df['station_id'].min()          # 1
df['station_id'].max()          # 8
df['station_id'].memory_usage() # 4,000,000 bytes
```

After running:

```python
df['station_id'] = df['station_id'].astype('uint8')
```

How many bytes does `station_id` now consume?

- A) 4,000,000 bytes (no change)
- B) 2,000,000 bytes (2× reduction)
- C) 500,000 bytes (8× reduction)
- D) 125,000 bytes (32× reduction)

**Answer: C) 500,000 bytes (8× reduction)**

**Explanation:** `int64` uses 8 bytes per element → 500,000 × 8 = 4,000,000 bytes. `uint8` uses 1 byte per element → 500,000 × 1 = 500,000 bytes. The range 1–8 fits comfortably within `uint8` (0–255), so no data is lost. This is an 8× memory reduction. The original `memory_usage()` call returns 4,000,000 confirming 500,000 rows × 8 bytes.

---

## Question 4

A DataFrame has the following dtype layout, and you want to compute the maximum number of rows you can load given a 140 MB RAM budget:

```python
# DataFrame columns:
# col_a: uint32   → 4 bytes per row
# col_b: float64  → 8 bytes per row
# col_c: int16    → 2 bytes per row
# Available RAM: 140 MB = 140,000,000 bytes

bytes_per_row = 4 + 8 + 2  # = 14
max_rows = 140_000_000 // 14
```

What is `max_rows`?

- A) 1,000,000
- B) 5,000,000
- C) 10,000,000
- D) 14,000,000

**Answer: C) 10,000,000**

**Explanation:** Each row consumes 4 + 8 + 2 = 14 bytes. Integer floor division: 140,000,000 ÷ 14 = 10,000,000 rows. This is the exact calculation used on the DTU 02613 exams to determine a safe `chunksize` or maximum in-memory DataFrame size given a RAM constraint. In practice you should leave a safety margin (e.g. use 80% of available RAM) to account for Python overhead and intermediate arrays.

---

## Question 5

A student writes the following code to count chunks in a CSV file:

```python
chunks = pd.read_csv('data.csv', chunksize=100_000)
print(len(chunks))   # intended to count number of chunks
```

What actually happens when this code runs?

- A) Prints the correct number of chunks (e.g. `42`)
- B) Prints `0` because the file has not been read yet
- C) Raises `TypeError` because `len()` is not supported on a `TextFileReader`
- D) Reads the entire file into memory to compute the length

**Answer: C) Raises `TypeError` because `len()` is not supported on a `TextFileReader`**

**Explanation:** `pd.read_csv(..., chunksize=N)` returns a `TextFileReader` object, which is a **lazy iterator** — it reads one chunk at a time only when iterated. It does not implement `__len__`, so calling `len()` on it raises `TypeError: object of type 'TextFileReader' has no len()`. To count chunks you must iterate: `sum(1 for _ in pd.read_csv('data.csv', chunksize=100_000))`. Note this exhausts the iterator — you must reopen the file to iterate again.

---

## Question 6

A column `version` in a 1,000,000-row DataFrame is currently stored as `int64`. You observe:

```python
df['version'].min()   # 0
df['version'].max()   # 42
```

A colleague suggests:

```python
df['version'] = df['version'].astype('int8')
```

Is this conversion safe?

- A) No — `int8` can only hold values 0 to 127, so values above 127 would be silently truncated
- B) Yes — `int8` holds -128 to 127, and the range 0–42 fits without loss
- C) No — integer columns cannot be cast to smaller integer types in pandas
- D) Yes — but only if all values are non-negative (requires `uint8` otherwise)

**Answer: B) Yes — `int8` holds -128 to 127, and the range 0–42 fits without loss**

**Explanation:** `int8` (signed 8-bit integer) stores values from -128 to 127. The column range 0–42 is fully contained within this interval, so the cast is lossless. Memory drops from 8 bytes/element (int64) to 1 byte/element (int8) — an 8× reduction, saving 7 MB for a 1M-row column. `uint8` (0–255) would also work here, but `int8` is valid and the question asks specifically about `int8`. Always verify `min()` and `max()` before downcasting.

---

## Question 7

A student downcasts an integer column to `float32`:

```python
df['count'] = df['count'].astype('float32')
# df['count'] is int64, range: 0 to 100,000
```

What is the primary concern with this conversion?

- A) `float32` cannot represent integers at all — all values become `NaN`
- B) `float32` uses 4 bytes vs `int64`'s 8 bytes, so there is no memory benefit
- C) Integers up to 16,777,216 are exactly representable in `float32`, so 0–100,000 is numerically safe, but converting integers to float introduces semantic ambiguity and prevents future use of integer-specific operations
- D) `float32` rounds all values to the nearest even number, corrupting the data

**Answer: C) Integers up to 16,777,216 are exactly representable in `float32`, so 0–100,000 is numerically safe, but converting integers to float introduces semantic ambiguity and prevents future use of integer-specific operations**

**Explanation:** `float32` has a 23-bit mantissa, giving it exact integer representation up to 2²⁴ = 16,777,216. Since 100,000 < 16,777,216, no rounding occurs. However, if you need to do integer arithmetic (counting, indexing, aggregations that assume whole numbers), floating-point semantics can introduce bugs. Better choices: `uint32` (0–4,294,967,295, 4 bytes) or `uint16` (0–65,535, 2 bytes) — but 100,000 exceeds `uint16` max (65,535), so `uint32` is the correct integer downcast here.

---

## Question 8

You call `df.memory_usage()` twice on the same DataFrame:

```python
df.memory_usage()
# Index      128
# station    4000000    ← 8 bytes × 500,000 rows

df.memory_usage(deep=True)
# Index      128
# station    22500000   ← ~45 bytes × 500,000 rows
```

What does the large discrepancy for `station` indicate?

- A) `station` is a `float64` column, and `deep=True` includes NaN overhead
- B) `station` is an `object` (string) column; without `deep=True` only pointer sizes are counted, while `deep=True` measures actual heap memory of the Python string objects
- C) `deep=True` double-counts memory by including both the column and its index
- D) The DataFrame has a MultiIndex that `deep=True` includes in the `station` entry

**Answer: B) `station` is an `object` (string) column; without `deep=True` only pointer sizes are counted, while `deep=True` measures actual heap memory of the Python string objects**

**Explanation:** For `object`-dtype columns, each element is a Python object pointer (8 bytes on 64-bit systems). `memory_usage()` without `deep=True` reports only these pointer sizes: 500,000 × 8 = 4,000,000 bytes. With `deep=True`, pandas traverses the heap and measures the actual size of each Python string object — here averaging ~45 bytes per string (fixed overhead ~49 bytes + string length). This is why `object` columns are memory-expensive and should be converted to `category` or `string[pyarrow]` when possible. Always use `deep=True` for accurate memory profiling.

---

## Question 9

Consider two approaches to filter a large time-indexed DataFrame for a specific date:

```python
# Approach A
df_indexed = df.set_index('date').sort_index()
result_a = df_indexed.loc['2023-06-15']['temperature'].mean()

# Approach B
result_b = df[df['date'] == '2023-06-15']['temperature'].mean()
```

Both return the same numeric result. Why is Approach A faster for **repeated queries** on different dates?

- A) `set_index` loads the column into L1 cache, making comparisons faster
- B) A sorted index enables binary search O(log n) for label-based lookup; Approach B requires a full O(n) boolean scan of the entire column each time
- C) `loc` is implemented in C while boolean indexing is pure Python
- D) Approach A reads only the filtered rows from disk; Approach B reads all rows

**Answer: B) A sorted index enables binary search O(log n) for label-based lookup; Approach B requires a full O(n) boolean scan of the entire column each time**

**Explanation:** After `set_index('date').sort_index()`, pandas can use binary search on the sorted index to locate matching rows in O(log n) time — similar to a B-tree lookup in a database. For 10 million rows, that's ~23 comparisons vs 10,000,000. Approach B (`df['date'] == '2023-06-15'`) creates a boolean array by scanning every row — always O(n) regardless of the query. For a single query, the overhead of `set_index`/`sort_index` may not pay off; for many repeated queries on a stable DataFrame, the sorted index gives a large advantage.

---

## Question 10

A student writes a two-pass aggregation over a chunked CSV file:

```python
dfc = pd.read_csv('big_data.csv', chunksize=50_000)

# First pass: compute total sum
total = 0
for chunk in dfc:
    total += chunk['value'].sum()

print(f"Total: {total}")

# Second pass: inspect shapes
for chunk in dfc:        # <-- second iteration
    print(chunk.shape)
```

What happens during the second `for` loop?

- A) Raises `StopIteration` immediately and crashes the program
- B) Iterates the file again from the beginning, printing shapes as expected
- C) Prints no output — the `TextFileReader` is exhausted after the first pass and the second loop body never executes
- D) Raises `FileNotFoundError` because the file handle was closed after the first pass

**Answer: C) Prints no output — the `TextFileReader` is exhausted after the first pass and the second loop body never executes**

**Explanation:** A `TextFileReader` (returned by `pd.read_csv` with `chunksize`) is a **forward-only, single-use iterator**. After the first `for` loop advances it to the end of the file, the internal file pointer is at EOF. The second `for` loop calls `__next__()` immediately, which returns nothing (the iterator is "exhausted"), so the loop body never executes and no output is printed — no exception, just silence. To make a second pass you must reopen the reader: `dfc = pd.read_csv('big_data.csv', chunksize=50_000)`. This is a common exam trap and a real-world bug.

---

## Key Facts Reference

| Topic | Detail |
|-------|--------|
| `uint8` range | 0 to 255 |
| `int8` range | -128 to 127 |
| `int16` range | -32,768 to 32,767 |
| `uint16` range | 0 to 65,535 |
| `int32` / `uint32` range | ±2.1 × 10⁹ / 0 to 4.3 × 10⁹ |
| `float32` exact integers | Up to 2²⁴ = 16,777,216 |
| `object` column memory | ~8 bytes pointer; use `deep=True` for real size |
| `category` best for | Low-cardinality string/int columns (< 0.5% unique ratio) |
| Chunked reader | Forward-only iterator; must reopen file for second pass |
| Sorted index lookup | O(log n) binary search vs O(n) boolean mask |
| `datetime64[ns]` | 8 bytes/element; enables `.dt` accessor and time-aware ops |
