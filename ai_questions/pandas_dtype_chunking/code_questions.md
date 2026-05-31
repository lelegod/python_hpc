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

**Answer: D**

- A) Incorrect — `timestamp` has 500,000 unique values (one per row); `category` needs low cardinality to save memory, and this column should become `datetime64[ns]` instead
- B) Incorrect — `station_id` has low cardinality (8 unique) but is already numeric; the right optimisation is downcasting to a smaller integer type like `uint8`, not `category`
- C) Incorrect — `temperature` is a continuous float with near-infinite unique values; `category` would be counterproductive and wasteful
- D) Correct — `reading_code` has only 6 unique values across 500,000 rows; `category` stores an integer code per row plus a tiny lookup table, giving massive memory savings at low cardinality

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

**Answer: A**

- A) Correct — `timestamp` is an `object` column of date strings; `pd.to_datetime()` converts it to `datetime64[ns]`, dropping memory from ~50 bytes/element to 8 bytes/element and enabling `.dt` accessor and `.resample()`
- B) Incorrect — `station_id` contains integer station identifiers, not dates or times; it should be downcast to `uint8`
- C) Incorrect — `temperature` is a continuous float measurement, not a timestamp; it belongs in `float32` or `float64`
- D) Incorrect — `reading_code` is a low-cardinality string category with only 6 unique values; it should become `category`, not `datetime64`

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

**Answer: C**

- A) Incorrect — `int64` uses 8 bytes/element; `uint8` uses 1 byte/element, so memory does change (8× reduction)
- B) Incorrect — 2,000,000 bytes would imply a 4-byte type like `int32` or `float32`, not `uint8`
- C) Correct — `uint8` is 1 byte/element; 500,000 × 1 = 500,000 bytes, an 8× reduction from 4,000,000; range 1–8 fits comfortably within `uint8` (0–255)
- D) Incorrect — 125,000 bytes would require a sub-byte representation, which does not exist in numpy/pandas

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

**Answer: C**

- A) Incorrect — 1,000,000 rows × 14 bytes = 14 MB, far below the 140 MB budget; this would come from dividing by 140 instead of 14
- B) Incorrect — 5,000,000 × 14 = 70 MB; this uses only half the budget and would come from dividing 70,000,000 by 14
- C) Correct — 140,000,000 ÷ 14 = 10,000,000 rows exactly; each row is uint32 (4) + float64 (8) + int16 (2) = 14 bytes
- D) Incorrect — 14,000,000 × 14 = 196 MB, which exceeds the 140 MB budget; this incorrectly treats the budget as equal to the row count

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

**Answer: C**

- A) Incorrect — `TextFileReader` does not implement `__len__`; calling `len()` raises `TypeError`, it does not return a count
- B) Incorrect — the error occurs before any reading happens, but the error is a `TypeError`, not a zero return value
- C) Correct — `pd.read_csv(..., chunksize=N)` returns a lazy `TextFileReader` iterator without `__len__`; calling `len()` raises `TypeError: object of type 'TextFileReader' has no len()`
- D) Incorrect — `TextFileReader` is explicitly lazy and never reads the full file into memory; it reads one chunk at a time only when iterated

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

**Answer: B**

- A) Incorrect — this describes `uint8` (0–255) incorrectly; `int8` range is -128 to 127, so values up to 127 are safe and 42 is well within range
- B) Correct — `int8` (signed 8-bit integer) stores -128 to 127; range 0–42 is fully contained, so the cast is lossless and saves 7 bytes per element (8→1 bytes)
- C) Incorrect — pandas supports downcasting integers to smaller integer types with `.astype()`; this is a standard memory optimisation
- D) Incorrect — `int8` is valid here regardless; negative values are not present in this column, but `int8` handles them fine; `uint8` would also work but is not required

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

**Answer: C**

- A) Incorrect — `float32` can represent integers exactly up to 2²⁴ = 16,777,216; no values become `NaN`
- B) Incorrect — `float32` (4 bytes) vs `int64` (8 bytes) is a 2× memory reduction; there is a memory benefit, though better integer alternatives exist
- C) Correct — `float32` exactly represents integers up to 16,777,216, so 0–100,000 is lossless, but using float for a count column introduces semantic ambiguity; `uint32` or `uint16` are more appropriate integer downcasts (100,000 exceeds `uint16` max of 65,535, so `uint32` is correct)
- D) Incorrect — IEEE 754 round-to-nearest-even only applies when a value exceeds the representable integer range; for values ≤ 16,777,216 integers are stored exactly

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

**Answer: B**

- A) Incorrect — `float64` stores values inline at exactly 8 bytes/element with no heap indirection; `deep=True` would show no discrepancy for a float column
- B) Correct — `object` dtype stores 8-byte pointers to Python heap objects; without `deep=True` only the pointer array is measured (500,000 × 8 = 4,000,000 bytes); `deep=True` traverses the heap and measures actual string sizes (~45 bytes/string on average)
- C) Incorrect — `deep=True` does not double-count; it simply includes the actual memory of referenced Python objects that shallow counting ignores
- D) Incorrect — the `station` entry in `memory_usage()` reflects only the `station` column's own memory, not index structures

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

**Answer: B**

- A) Incorrect — `set_index` reorganises the DataFrame structure; it does not specifically load data into L1 cache
- B) Correct — a sorted index allows O(log n) binary search (~23 comparisons for 10M rows); Approach B always scans all n rows to build a boolean mask regardless of query, giving O(n) per query
- C) Incorrect — both `.loc` and boolean indexing are implemented at the C/Cython level in pandas; neither is pure Python
- D) Incorrect — both approaches operate on an in-memory DataFrame; neither reads from disk at query time

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

**Answer: C**

- A) Incorrect — `StopIteration` is caught internally by the `for` loop machinery; it does not crash the program, the loop simply exits immediately with no iterations
- B) Incorrect — `TextFileReader` is a forward-only, single-use iterator; it does not reset to the beginning after being exhausted
- C) Correct — after the first `for` loop advances the `TextFileReader` to EOF, the internal file pointer stays at EOF; the second loop's first `__next__()` call finds nothing, so the loop body never executes and no output is printed
- D) Incorrect — the file handle remains open; no `FileNotFoundError` is raised; the iterator is simply exhausted at EOF

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
