# Pandas Dtype Optimization & Chunking — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Choosing category Column from Info Summary](#q1--choosing-category-column-from-info-summary)
- [Q2 — Choosing datetime64 Column from Info Summary](#q2--choosing-datetime64-column-from-info-summary)
- [Q3 — Memory After uint8 Downcast](#q3--memory-after-uint8-downcast)
- [Q4 — Max Rows for RAM Budget](#q4--max-rows-for-ram-budget)
- [Q5 — len() on TextFileReader Raises TypeError](#q5--len-on-textfilereader-raises-typeerror)
- [Q6 — Safe int8 Downcast for Range 0-42](#q6--safe-int8-downcast-for-range-0-42)
- [Q7 — float32 for Integer Count Column](#q7--float32-for-integer-count-column)
- [Q8 — memory_usage deep=True for Object Column](#q8--memory_usage-deeptrue-for-object-column)
- [Q9 — Sorted Index vs Boolean Mask Lookup](#q9--sorted-index-vs-boolean-mask-lookup)
- [Q10 — TextFileReader Exhausted After One Pass](#q10--textfilereader-exhausted-after-one-pass)
- [Key Facts Reference](#key-facts-reference)
- [Set 2 — Generated Practice Questions (Exam-Day Focus)](#set-2-generated-practice-questions-exam-day-focus)
- [Q11 — float32 Column Memory Calculation](#q11--float32-column-memory-calculation)
- [Q12 — Why Second groupby().sum() Is Needed](#q12--why-second-groupbysum-is-needed)
- [Q13 — Memory After dtype Optimization](#q13--memory-after-dtype-optimization)
- [Q14 — first_chunk.shape After next(reader)](#q14--first_chunkshape-after-nextreader)
- [Q15 — cat.codes.dtype for 3-Value Category](#q15--catcodesdtype-for-3-value-category)
- [Q16 — Shallow vs Deep Object Column Memory](#q16--shallow-vs-deep-object-column-memory)
- [Q17 — Chunked Per-Day Aggregation Pattern](#q17--chunked-per-day-aggregation-pattern)
- [Q18 — memory_usage().sum() Calculation](#q18--memory_usagesum-calculation)
- [Q19 — category vs Manual uint8 Mapping](#q19--category-vs-manual-uint8-mapping)
- [Q20 — PyArrow dictionary_encode() Type and Memory](#q20--pyarrow-dictionary_encode-type-and-memory)
- [Set 3 — Extended Practice](#set-3--extended-practice)
- [Q21 — to_numeric downcast Output Dtype](#q21--to_numeric-downcast-output-dtype)
- [Q22 — usecols Peak Memory Comparison](#q22--usecols-peak-memory-comparison)
- [Q23 — astype int32 on Float Column with NaN](#q23--astype-int32-on-float-column-with-nan)
- [Q24 — cat.codes for Unseen Category Value](#q24--catcodes-for-unseen-category-value)
- [Q25 — convert_dtypes on int64 Column](#q25--convert_dtypes-on-int64-column)
- [Q26 — select_dtypes Conversion Pattern](#q26--select_dtypes-conversion-pattern)
- [Q27 — nrows vs chunksize Shape Output](#q27--nrows-vs-chunksize-shape-output)
- [Q28 — reduce_dmi_df parameterId Column Result](#q28--reduce_dmi_df-parameterid-column-result)
- [Q29 — CategoricalDtype with Unseen Value](#q29--categoricaldtype-with-unseen-value)
- [Q30 — memory_usage index=False vs Sum](#q30--memory_usage-indexfalse-vs-sum)

---

> Format: Each question shows a DataFrame summary, code snippet, or memory_usage output to interpret.
> Exam frequency: **2024 exam + F25**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--choosing-category-column-from-info-summary)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — Choosing category Column from Info Summary

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

## Q2 — Choosing datetime64 Column from Info Summary

Using the same DataFrame summary from Q1:

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

## Q3 — Memory After uint8 Downcast

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

## Q4 — Max Rows for RAM Budget

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

## Q5 — len() on TextFileReader Raises TypeError

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

## Q6 — Safe int8 Downcast for Range 0-42

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

## Q7 — float32 for Integer Count Column

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

## Q8 — memory_usage deep=True for Object Column

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

## Q9 — Sorted Index vs Boolean Mask Lookup

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

## Q10 — TextFileReader Exhausted After One Pass

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

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

> Targets read_csv dtype and chunksize parameters, category dtype savings, memory_usage calculations, and chunk processing patterns

---

## Q11 — float32 Column Memory Calculation

You run the following code on a DataFrame with 800,000 rows:

```python
df = pd.read_csv('records.csv', dtype={'status': 'category', 'score': 'float32'})
print(df['status'].dtype)   # category
print(df['score'].dtype)    # float32
print(df['score'].memory_usage())
```

The `score` column has no index-related overhead here. What value does `df['score'].memory_usage()` print?

- A) 6,400,000
- B) 3,200,000
- C) 800,000
- D) 1,600,000

**Answer: B**

`float32` uses 4 bytes per element. 800,000 rows × 4 bytes = 3,200,000 bytes. Note: `df['score'].memory_usage()` on a Series returns the bytes for that column only (not including the Index by default for a Series call; the Index is separate). Option A (6,400,000) corresponds to float64 (8 bytes × 800,000). Option C (800,000) corresponds to a 1-byte type like int8 or uint8. Option D (1,600,000) corresponds to int16 or float16 (2 bytes × 800,000). The dtype `float32` is 4 bytes — always half of float64.

---

## Q12 — Why Second groupby().sum() Is Needed

Consider this chunk-based groupby pattern:

```python
results = []
for chunk in pd.read_csv('events.csv', chunksize=50_000):
    results.append(chunk.groupby('event_type')['count'].sum())

final = pd.concat(results).groupby(level=0).sum()
```

Why is the second `.groupby(...).sum()` call on `final` necessary?

- A) The first groupby inside the loop produces incorrect results due to chunk boundaries splitting groups
- B) Each chunk's groupby only sees rows in that chunk; the same `event_type` may appear in multiple chunks and must be re-aggregated after concatenation
- C) `pd.concat` scrambles the index ordering, requiring a second groupby to re-sort
- D) The second groupby converts the result from a Series to a DataFrame

**Answer: B**

Each chunk is an independent DataFrame slice of the file. If `event_type = "click"` appears in chunk 1, chunk 3, and chunk 7, the `results` list will contain three separate `click` entries after the loop. `pd.concat(results)` stacks them into one Series with duplicate index entries. The second `.groupby(level=0).sum()` collapses duplicate index keys, producing the true global sum per event_type. Option A is wrong — the per-chunk groupby is correct within its chunk; the issue is cross-chunk aggregation, not corruption. Option C is wrong — `pd.concat` preserves index values; ordering may vary but is corrected by groupby, not a problem in itself. Option D is wrong — the result of groupby+sum on a Series is still a Series, not a DataFrame.

---

## Q13 — Memory After dtype Optimization

A student runs:

```python
df = pd.read_csv('data.csv')
print(df.memory_usage(deep=True).sum() / 1024**2, "MB")
```

Then optimises by specifying dtypes:

```python
df2 = pd.read_csv('data.csv', dtype={
    'sensor_id': 'int16',
    'reading':   'float32',
    'active':    'bool'
})
print(df2.memory_usage(deep=True).sum() / 1024**2, "MB")
```

The original `df` had `sensor_id` as int64, `reading` as float64, `active` as object (strings "True"/"False"). The DataFrame has 1,000,000 rows. How much memory does `df2` use for just these three columns?

- A) ~7.6 MB
- B) ~19 MB
- C) ~24 MB
- D) ~3.8 MB

**Answer: D**

Per row for the three columns: int16 (2) + float32 (4) + bool (1) = 7 bytes. Total: 1,000,000 × 7 = 7,000,000 bytes = 7 MB — but we need `deep=True` for the bool column since it started as object. After conversion to `bool` dtype, it stores 1 byte/element with no heap strings, so `deep=True` makes no difference for bool. Total ≈ 7 MB. Closest answer is D (~3.8 MB is actually the float32 column alone at ~3.8 MB). Wait — let me recompute: float32 alone: 1M × 4 = 4 MB; int16: 1M × 2 = 2 MB; bool: 1M × 1 = 1 MB; total = 7 MB. The closest answer is A (~7.6 MB, which includes a small Index contribution). Option B (~19 MB) is roughly the original `df` if `active` object columns add ~12 MB with deep=True. Option C (~24 MB) is the original int64+float64+object without optimisation (8+8+8 pointer = 24 MB shallow). Option D is a distractor; A is correct at ~7–7.6 MB including the Index.

**Answer: A** *(correction: 7 bytes/row × 1M rows = 7 MB data + ~0.6 MB Index = ~7.6 MB)*

---

## Q14 — first_chunk.shape After next(reader)

You have this code:

```python
reader = pd.read_csv('transactions.csv', chunksize=200_000)
first_chunk = next(reader)
print(first_chunk.shape)
```

The file has 750,000 rows and 4 columns. What does `first_chunk.shape` print?

- A) `(750000, 4)`
- B) `(200000, 4)`
- C) `(187500, 4)`
- D) `(4, 200000)`

**Answer: B**

`next(reader)` advances the TextFileReader by one chunk and returns a DataFrame of up to `chunksize` rows. Since chunksize=200,000 and the file has 750,000 rows, the first chunk contains exactly 200,000 rows and all 4 columns — shape (200000, 4). Option A (750000, 4) would result from loading the entire file without chunking. Option C (187500, 4) is 750,000/4 rows — there is no division by number of chunks in chunksize logic. Option D (4, 200000) has axes transposed — pandas DataFrames are (rows, cols), never (cols, rows) in `.shape`.

---

## Q15 — cat.codes.dtype for 3-Value Category

A DataFrame column `sensor_type` is converted to category dtype. You then inspect it:

```python
df['sensor_type'] = df['sensor_type'].astype('category')
print(df['sensor_type'].cat.categories)
print(df['sensor_type'].cat.codes.dtype)
```

The column has 3 unique values: `"temp"`, `"humidity"`, `"pressure"`. What does `cat.codes.dtype` print?

- A) `object`
- B) `int64`
- C) `int8`
- D) `uint8`

**Answer: C**

Pandas assigns category codes as signed integers, choosing the smallest signed integer type that fits the number of unique values. With 3 unique values (codes 0, 1, 2 plus -1 reserved for NaN), int8 (range -128 to 127) is used. The signed int is required because pandas uses -1 to represent NaN/missing codes. Option A (object) is wrong — codes are always integer dtype, not object. Option B (int64) is wrong — pandas does not use the default int64 for category codes; it uses the smallest sufficient signed integer. Option D (uint8) is wrong — pandas uses signed integers for codes specifically to allow -1 as a sentinel for missing values; uint8 cannot represent -1.

---

## Q16 — Shallow vs Deep Object Column Memory

Consider this memory audit snippet:

```python
import pandas as pd
df = pd.DataFrame({
    'a': pd.array([1, 2, 3] * 100_000, dtype='int64'),
    'b': pd.array(['x', 'y'] * 150_000, dtype='object')
})

shallow = df.memory_usage()
deep    = df.memory_usage(deep=True)

print(shallow['b'], deep['b'])
```

The DataFrame has 300,000 rows. Which output is correct?

- A) `2400000  2400000`
- B) `2400000  ~15000000`
- C) `300000   300000`
- D) `300000   ~15000000`

**Answer: B**

Column `b` has dtype `object`. Shallow (`deep=False`): 300,000 rows × 8 bytes/pointer = 2,400,000 bytes. Deep (`deep=True`): each Python `str` object (`'x'` or `'y'`) has overhead of ~49–50 bytes (CPython str header + 1 character). With 300,000 rows: ~300,000 × 50 bytes ≈ 15,000,000 bytes. So shallow=2,400,000 and deep≈15,000,000. Option A claims both are equal — wrong, that's only true for numeric dtypes. Option C shows 300,000 for shallow, implying 1 byte/element — that would be a 1-byte numeric dtype like bool or int8, not object. Option D shows 300,000 shallow — same error as C.

---

## Q17 — Chunked Per-Day Aggregation Pattern

A student wants to process a 5 GB CSV and collect per-day totals. They write:

```python
daily_totals = {}
for chunk in pd.read_csv('big.csv', chunksize=100_000,
                          parse_dates=['date'],
                          dtype={'amount': 'float32'}):
    for date, group in chunk.groupby('date'):
        daily_totals[date] = daily_totals.get(date, 0) + group['amount'].sum()
```

Which statement about this code is TRUE?

- A) It fails because `parse_dates` and `dtype` cannot be used together in the same `read_csv` call
- B) It correctly accumulates per-day totals without loading the entire 5 GB file into memory
- C) It produces incorrect totals because `float32` precision loss accumulates across chunks
- D) It fails because `groupby` is not available on chunk DataFrames

**Answer: B**

This is a valid chunked accumulation pattern. Each chunk is read with `parse_dates=['date']` (so `date` becomes datetime64) and `dtype={'amount': 'float32'}` (reducing memory). The `groupby('date')` within each chunk groups rows for that chunk only; the outer dict accumulates across chunks. Since every chunk contributes its portion and the dict covers all dates seen, the final `daily_totals` is correct. Option A is false — `parse_dates` and `dtype` can coexist in `read_csv`; they affect different columns. Option C is an overstatement — float32 accumulation error for monetary sums is typically negligible in practice (< 1 part per million per addition); the exam does not penalise this pattern. Option D is false — `groupby` is fully available on any DataFrame, including chunk DataFrames.

---

## Q18 — memory_usage().sum() Calculation

You call `df.memory_usage()` on a five-column DataFrame with 1,000,000 rows and receive:

```
Index      8000000
col_a      8000000   # float64
col_b      4000000   # float32
col_c      2000000   # int16
col_d      1000000   # uint8
col_e      8000000   # object (shallow)
dtype: int64
```

What is the total reported by `df.memory_usage().sum()`?

- A) 23,000,000
- B) 31,000,000
- C) 27,000,000
- D) 15,000,000

**Answer: B**

Sum all entries: Index (8,000,000) + col_a (8,000,000) + col_b (4,000,000) + col_c (2,000,000) + col_d (1,000,000) + col_e (8,000,000) = 31,000,000. Option A (23,000,000) omits the Index. Option C (27,000,000) omits the Index and undercounts col_e. Option D (15,000,000) sums only col_a through col_d without Index or col_e. The Index row is always included in `memory_usage()` output and must be included in the sum.

---

## Q19 — category vs Manual uint8 Mapping

A colleague proposes this optimisation for a column with 12 unique string values across 2,000,000 rows:

```python
# Original
df['status'] = df['status'].astype('category')

# Colleague's alternative
mapping = {v: i for i, v in enumerate(df['status'].unique())}
df['status'] = df['status'].map(mapping).astype('uint8')
```

Compared to `category`, what does the colleague's `uint8` approach lose?

- A) Memory efficiency — `uint8` uses more memory than category codes
- B) The original string labels — `category` preserves labels via `.cat.categories`; the manual `uint8` column stores only integers with no built-in reverse mapping
- C) The ability to use groupby on the column
- D) Compatibility with `pd.read_csv`

**Answer: B**

Both `category` and the manual `uint8` mapping achieve the same per-row memory (1 byte for up to 127/255 unique values). The critical difference is that `category` dtype retains the original string labels in `.cat.categories`, allowing transparent decoding, human-readable groupby output, and correct behaviour in merge/join operations. The manual `uint8` column stores raw integers; without the external `mapping` dict, you cannot recover what string "3" or "7" meant. Option A is wrong — both use 1 byte per row for 12 unique values. Option C is wrong — `groupby` works on uint8 columns (grouping by integer code), but the output labels are integers not strings. Option D is wrong — both dtypes are compatible with `pd.read_csv` via `dtype=`.

---

## Q20 — PyArrow dictionary_encode() Type and Memory

Examine this PyArrow snippet:

```python
import pyarrow as pa
import pyarrow.csv as pa_csv

table = pa_csv.read_csv('data.csv')
col = table.column('region')
dict_col = col.dictionary_encode()
print(dict_col.type)
```

The `region` column has 5 unique string values across 1,000,000 rows. What does `dict_col.type` print, and approximately how does memory compare to the original?

- A) `string`; memory is unchanged because dictionary encoding is lossless
- B) `dictionary<values=string, indices=int8>`; memory drops from ~50 MB (string) to ~1 MB (codes) + small dictionary
- C) `int64`; strings are hashed to 64-bit integers
- D) `large_string`; Arrow uses a wider string type after encoding

**Answer: B**

`dictionary_encode()` converts an Arrow `string` column to a `dictionary<values=string, indices=int8>` type (Arrow uses int8 for the index when there are ≤127 unique values). With 5 unique values, int8 codes (1 byte each) replace full string storage. Original `string` column: ~50 bytes per string × 1,000,000 ≈ 50 MB. After encoding: 1,000,000 × 1 byte codes = 1 MB + 5 strings × ~50 bytes dictionary = ~1 MB total. Option A is wrong — memory changes dramatically; lossless only means no data is lost, not that memory is unchanged. Option C is wrong — `dictionary_encode()` preserves original string values in the dictionary; it does not hash them. Option D is wrong — `large_string` in Arrow is for strings > 2 GB total; it is not what `dictionary_encode()` produces.

---

## Set 3 — Extended Practice

> Targets to_numeric downcast, usecols, NaN/nullable integer trap, cat.codes sentinel, convert_dtypes, select_dtypes, nrows vs chunksize, reduce_dmi_df pattern, CategoricalDtype with unseen values, memory_usage index=False.

---

## Q21 — to_numeric downcast Output Dtype

> **Week reference:** Week 7

```python
import pandas as pd

s = pd.Series(['100', '200', '50', '175'], dtype='object')
result = pd.to_numeric(s, downcast='integer')
print(result.dtype)
```

The values are all in the range 0–200. What does `result.dtype` print?

- A) `int64`
- B) `int8`
- C) `int16`
- D) `uint8`

**Answer: D**

- A) Incorrect — `downcast='integer'` explicitly instructs pandas to find the smallest integer type. Without `downcast`, the result would be `int64`. The whole point of `downcast='integer'` is to avoid this.
- B) Incorrect — `int8` covers -128 to 127. The value 200 exceeds int8's maximum of 127, so int8 would overflow. Pandas checks the actual range and picks a type that fits without overflow.
- C) Incorrect — `int16` (range -32,768 to 32,767) would safely hold 0–200, but it is not the smallest option. Since all values are non-negative and fit within 0–255, pandas selects `uint8` first when using `downcast='integer'`.
- D) Correct — `pd.to_numeric(..., downcast='integer')` tries types in order: int8 (fails: 200 > 127), then uint8 (succeeds: 200 ≤ 255). With all values in 0–200, `uint8` is the smallest type that fits, so `result.dtype` is `uint8`.

---

## Q22 — usecols Peak Memory Comparison

> **Week reference:** Week 7

```python
import pandas as pd

# Approach A
df_a = pd.read_csv('sensors.csv')
df_a = df_a[['stationId', 'value']]

# Approach B
df_b = pd.read_csv('sensors.csv', usecols=['stationId', 'value'])
```

The CSV has 8 float64 columns and 1,000,000 rows. `stationId` is int64 and `value` is float64. Which statement about peak RAM usage during execution is TRUE?

- A) Both approaches use the same peak RAM because Python's allocator frees unused columns immediately
- B) Approach A uses more peak RAM — it allocates all 8 columns before dropping 6; Approach B allocates only 2 columns
- C) Approach B uses more peak RAM because `usecols` requires pandas to scan all columns to find the two named ones
- D) Approach A uses less peak RAM because dropping columns in-place frees memory before the assignment completes

**Answer: B**

- A) Incorrect — Python's garbage collector frees objects when reference counts drop to zero, but the peak memory during `pd.read_csv('sensors.csv')` without `usecols` still allocated all 8 columns simultaneously. Peak RAM is the maximum at any point, not the final state.
- B) Correct — Approach A: `pd.read_csv` parses all 8 columns into memory (8 × 1,000,000 × 8 = 64 MB peak) before the column selection drops 6. Approach B: `usecols` tells the parser to skip 6 columns entirely, so only 2 × 1,000,000 × 8 = 16 MB is ever allocated. Peak RAM for B is 4× lower.
- C) Incorrect — `usecols` does not require allocating all columns to find the named ones. The CSV parser reads column headers first, then skips bytes for unneeded columns as it parses each row. No allocation occurs for skipped columns.
- D) Incorrect — `df_a = df_a[['stationId', 'value']]` creates a new DataFrame with 2 columns and reassigns `df_a`, allowing the 8-column DataFrame to be garbage collected. But the 64 MB peak already occurred during the read call, before the selection.

---

## Q23 — astype int32 on Float Column with NaN

> **Week reference:** Week 7

```python
import pandas as pd
import numpy as np

s = pd.Series([1.0, 2.0, np.nan, 4.0])
print(s.astype('int32'))
```

What happens when this code runs?

- A) Prints `[1, 2, 0, 4]` — NaN is silently converted to 0
- B) Prints `[1, 2, -2147483648, 4]` — NaN wraps to int32 minimum
- C) Raises `ValueError: Cannot convert non-finite values (NA or inf) to integer`
- D) Prints `[1, 2, 2, 4]` — NaN is forward-filled from the previous value

**Answer: C**

- A) Incorrect — pandas does not silently coerce NaN to 0 during `.astype('int32')`. Such silent mutation would be dangerous. You must explicitly call `.fillna(0).astype('int32')` if zero-filling is intended.
- B) Incorrect — numpy integer overflow wraps under specific conditions, but NaN-to-integer conversion does not produce INT_MIN. The operation fails with a ValueError before any value is stored.
- C) Correct — `int32` is a numpy dtype with no representation for NaN or infinity. Calling `.astype('int32')` on a float Series containing NaN raises `ValueError: Cannot convert non-finite values (NA or inf) to integer`. To safely downcast with NaN present, use `pd.Int32Dtype()` (capital-I nullable integer) which stores NaN as `pd.NA`.
- D) Incorrect — `.astype()` does not forward-fill values. Forward-filling is an explicit operation performed by `.ffill()` or `.fillna(method='ffill')`.

---

## Q24 — cat.codes for Unseen Category Value

> **Week reference:** Week 7

```python
import pandas as pd

known = pd.CategoricalDtype(categories=['A', 'B', 'C'], ordered=False)
s = pd.Series(['A', 'B', 'D', 'C'], dtype=known)
print(s.cat.codes.tolist())
```

The value `'D'` is not in the predefined category list. What does `s.cat.codes.tolist()` print?

- A) `[0, 1, 2, 3]`
- B) `[0, 1, 3, 2]`
- C) `[0, 1, -1, 2]`
- D) Raises `ValueError: 'D' is not a valid category`

**Answer: C**

- A) Incorrect — this would imply `'D'` was added as category index 2 and `'C'` moved to index 3. But `CategoricalDtype` with a fixed category list does not expand to accommodate unseen values; `'D'` becomes NaN.
- B) Incorrect — same issue as A; codes are determined by the position in the fixed `categories` list (`A`=0, `B`=1, `C`=2). `'D'` is not in the list and becomes the NaN sentinel, not code 3.
- C) Correct — when a value is not in the predefined `categories`, pandas stores it as NaN in the categorical, represented by the code `-1`. The codes list is `[0, 1, -1, 2]` for `['A', 'B', 'D', 'C']` respectively: A→0, B→1, D→-1 (NaN sentinel), C→2.
- D) Incorrect — pandas does not raise a `ValueError` for unseen values when creating a categorical Series. Instead, the unseen value is silently converted to NaN. A `ValueError` would only occur if you explicitly passed `errors='raise'` to a validation function.

---

## Q25 — convert_dtypes on int64 Column

> **Week reference:** Week 7

```python
import pandas as pd

s = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype='int64')
s2 = s.convert_dtypes()
print(s2.dtype)
print(s2.memory_usage())
```

The Series has 9 elements. What does `s2.dtype` print, and is `s2` smaller in memory than `s`?

- A) `int8` — yes, memory decreases because the values fit in 1 byte
- B) `Int64` — no, memory is the same or larger (nullable Int64 adds a boolean mask)
- C) `uint8` — yes, memory decreases because all values are non-negative
- D) `float64` — no, convert_dtypes promotes integers to float for NA safety

**Answer: B**

- A) Incorrect — `convert_dtypes()` does not downcast to smaller integer types. It converts to the pandas nullable extension type for the same bit-width. Values 0–8 fitting in `int8` is irrelevant; `convert_dtypes` maps `int64` → `Int64` unconditionally.
- B) Correct — `convert_dtypes()` converts `int64` to `Int64` (pandas nullable integer, capital I). `Int64` uses the same 8-byte storage per element as `int64` plus a companion boolean null mask (1 bit per element, stored efficiently). Memory usage is equal to or slightly greater than `int64`, not less.
- C) Incorrect — `convert_dtypes()` never produces `uint8`. It maps each numpy dtype to its nullable extension equivalent: `int64` → `Int64`, `float64` → `Float64`, etc. No downcasting occurs.
- D) Incorrect — `convert_dtypes()` preserves integer semantics. An `int64` column does not become `float64`; it becomes `Int64`. Float conversion would change the semantic meaning of integer data.

---

## Q26 — select_dtypes Conversion Pattern

> **Week reference:** Week 7

```python
import pandas as pd

df = pd.DataFrame({
    'a': [1.0, 2.0, 3.0],       # float64
    'b': [4.0, 5.0, 6.0],       # float64
    'c': ['x', 'y', 'z'],       # object
    'd': [True, False, True],    # bool
})

float_cols = df.select_dtypes(include='float64').columns
df[float_cols] = df[float_cols].astype('float32')
print(df.dtypes.tolist())
```

What does `df.dtypes.tolist()` print after the conversion?

- A) `[dtype('float32'), dtype('float32'), dtype('float32'), dtype('float32')]`
- B) `[dtype('float32'), dtype('float32'), dtype('O'), dtype('bool')]`
- C) `[dtype('float64'), dtype('float64'), dtype('O'), dtype('bool')]`
- D) Raises `TypeError` because `select_dtypes` cannot be used with assignment

**Answer: B**

- A) Incorrect — `select_dtypes(include='float64')` selects only columns with dtype float64. Columns `c` (object) and `d` (bool) are not float64 and are therefore not selected or converted. They retain their original dtypes.
- B) Correct — `select_dtypes(include='float64').columns` returns `['a', 'b']`. The assignment converts only `a` and `b` to float32. `c` remains `object` (shown as `dtype('O')`) and `d` remains `bool`. Result: `[float32, float32, object, bool]`.
- C) Incorrect — this would mean no conversion occurred. The assignment `df[float_cols] = df[float_cols].astype('float32')` does execute and does change `a` and `b` from float64 to float32.
- D) Incorrect — using `df[float_cols] = ...` for column assignment is standard pandas syntax. `select_dtypes` returns a DataFrame, `.columns` gives an Index of column names, and column assignment with a list of column names is fully supported.

---

## Q27 — nrows vs chunksize Shape Output

> **Week reference:** Week 7

```python
import pandas as pd

# The file 'data.csv' has 500,000 rows and 3 columns.

result_a = pd.read_csv('data.csv', nrows=10_000)
result_b = pd.read_csv('data.csv', chunksize=10_000)

print(type(result_a).__name__)
print(type(result_b).__name__)
```

What do the two `print` statements output?

- A) `DataFrame` / `DataFrame`
- B) `TextFileReader` / `TextFileReader`
- C) `DataFrame` / `TextFileReader`
- D) `list` / `TextFileReader`

**Answer: C**

- A) Incorrect — `chunksize=10_000` does not return a DataFrame. It returns a lazy `TextFileReader` iterator. No rows are read until you iterate it.
- B) Incorrect — `nrows=10_000` returns a fully-loaded, in-memory `DataFrame` of the first 10,000 rows. It does not return a `TextFileReader`.
- C) Correct — `nrows` is a limit parameter: it reads exactly N rows and returns a complete `DataFrame`. `chunksize` is an iteration parameter: it returns a `TextFileReader` iterator that yields chunks on demand. The type names printed are `DataFrame` and `TextFileReader`.
- D) Incorrect — neither call returns a `list`. `nrows` returns a `DataFrame`; `chunksize` returns a `TextFileReader`. Lists are only produced if you explicitly call `list(reader)` to materialise all chunks.

---

## Q28 — reduce_dmi_df parameterId Column Result

> **Week reference:** Week 7

The `reduce_dmi_df` function from week 7's `reduce_dataframe.py` applies these rules per column:

```python
def reduce_dmi_df(df):
    for column in df.columns:
        if column in ("created", "observed"):
            df[column] = pd.to_datetime(df[column], format="ISO8601")
        elif pd.api.types.is_integer_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], downcast="integer")
        elif pd.api.types.is_float_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], downcast="float")
        elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].astype("category")
    return df
```

The `2023_01.csv` file has a `parameterId` column loaded as `object` dtype containing strings like `"precip_past1min"`. There are 6 unique values across the entire dataset. What dtype does `parameterId` have after `reduce_dmi_df` is applied?

- A) `int8` — it is downcast via `pd.to_numeric`
- B) `datetime64[ns]` — it matches the ISO8601 date format check
- C) `float32` — object columns are downcast to float
- D) `category` — it is an object dtype column with low cardinality

**Answer: D**

- A) Incorrect — `pd.api.types.is_integer_dtype(df['parameterId'])` returns `False` for an `object` column of strings. The integer downcast branch is never reached. `pd.to_numeric` would raise or return NaN on non-numeric strings like `"precip_past1min"`.
- B) Incorrect — `"parameterId"` is not in the tuple `("created", "observed")`, so the `to_datetime` branch is skipped entirely. Attempting to parse `"precip_past1min"` as an ISO8601 date would fail anyway.
- C) Incorrect — `pd.api.types.is_float_dtype(df['parameterId'])` returns `False` for an `object` column. The float downcast branch is skipped. Object columns do not pass the float check.
- D) Correct — `parameterId` is an `object` dtype column, so `pd.api.types.is_object_dtype(df['parameterId'])` returns `True`. The function converts it to `category` via `.astype("category")`. With only 6 unique values across hundreds of thousands of rows, this is also the right optimisation for memory savings.

---

## Q29 — CategoricalDtype with Unseen Value

> **Week reference:** Week 7

```python
import pandas as pd

dtype = pd.CategoricalDtype(categories=['low', 'mid', 'high'], ordered=True)
chunk = pd.Series(['low', 'high', 'extreme', 'mid'], dtype=dtype)
print(chunk.isna().sum())
print(chunk.cat.categories.tolist())
```

What do the two `print` statements output?

- A) `0` / `['low', 'mid', 'high', 'extreme']`
- B) `1` / `['low', 'mid', 'high']`
- C) Raises `ValueError: 'extreme' is not in categories`
- D) `0` / `['extreme', 'high', 'low', 'mid']`

**Answer: B**

- A) Incorrect — if `'extreme'` were added to the categories automatically, `isna().sum()` would be 0 and the category list would include `'extreme'`. But `CategoricalDtype` with a fixed `categories=` list does not auto-expand; unseen values become NaN.
- B) Correct — `'extreme'` is not in `['low', 'mid', 'high']`, so it is stored as NaN (code -1). `isna().sum()` returns 1. `cat.categories` reflects the originally defined list `['low', 'mid', 'high']`, unchanged. The category set is fixed at dtype definition time.
- C) Incorrect — pandas does not raise an error when a Series is constructed from a `CategoricalDtype` and some values are not in the category list. Those values are silently converted to NaN. Only certain strict validation functions raise on unseen categories.
- D) Incorrect — `isna().sum()` is 0 only if all values are found in the categories. Since `'extreme'` is not, the count is 1. Also, `cat.categories` does not include `'extreme'` and does not get sorted alphabetically to produce `['extreme', 'high', 'low', 'mid']`.

---

## Q30 — memory_usage index=False vs Sum

> **Week reference:** Week 7

```python
import pandas as pd

df = pd.DataFrame({
    'x': pd.array(range(1_000_000), dtype='int32'),
    'y': pd.array(range(1_000_000), dtype='float32'),
})

total_with_index    = df.memory_usage().sum()
total_without_index = df.memory_usage(index=False).sum()

print(total_with_index - total_without_index)
```

What does the final `print` statement output?

- A) `0`
- B) `128`
- C) `8000000`
- D) `4000000`

**Answer: B**

- A) Incorrect — `memory_usage()` and `memory_usage(index=False)` differ precisely by the Index entry. For a default `RangeIndex`, the Index entry is 128 bytes, so the difference is 128, not 0.
- B) Correct — `df.memory_usage()` includes an `Index` row of 128 bytes for a default `RangeIndex` (regardless of the number of rows — RangeIndex is stored as start/stop/step, not as an array). `memory_usage(index=False)` excludes this entry. The difference is exactly 128 bytes.
- C) Incorrect — 8,000,000 is the memory for one of the 4-byte columns (4 × 1,000,000). The Index entry is not the size of a data column; it is the compact RangeIndex descriptor (128 bytes).
- D) Incorrect — 4,000,000 would be the memory for half of one float32 column (2 bytes × 1,000,000), which has no relevance here. The Index overhead for a RangeIndex is 128 bytes, independent of row count.

---
