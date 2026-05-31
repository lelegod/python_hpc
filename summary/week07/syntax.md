# Week 7 — Pandas & PyArrow Syntax Reference

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Pandas Memory Inspection](#pandas-memory-inspection)
- [dtype Downcasting](#dtype-downcasting)
  - [dtype reference table](#dtype-reference-table)
- [Pandas Indexing for Speed](#pandas-indexing-for-speed)
- [PyArrow CSV Loading](#pyarrow-csv-loading)
- [Parquet](#parquet)
- [File Format Comparison](#file-format-comparison)
- [Exam Traps](#exam-traps)

---

## Pandas Memory Inspection

```python
import pandas as pd

df.info(memory_usage='deep')              # summary with total memory
df.memory_usage(deep=True)                # per-column bytes
df.memory_usage(deep=True).sum()          # total bytes
# deep=True REQUIRED for object columns — without it shows only pointer size (8 bytes)
```

---

## dtype Downcasting

```python
# Convert timestamp strings → datetime64
df['col'] = pd.to_datetime(df['col'], format='ISO8601')

# Convert low-cardinality strings → category
df['col'] = df['col'].astype('category')

# Downcast numeric types
df['col'] = pd.to_numeric(df['col'], downcast='integer')  # int64 → smallest int
df['col'] = pd.to_numeric(df['col'], downcast='float')    # float64 → float32

# Manual cast
df['col'] = df['col'].astype('int16')
df['col'] = df['col'].astype('uint8')
df['col'] = df['col'].astype('float32')
```

### dtype reference table

| dtype | Bytes | Min | Max | Notes |
|---|---|---|---|---|
| `int8` | 1 | -128 | 127 | |
| `uint8` | 1 | 0 | 255 | |
| `int16` | 2 | -32,768 | 32,767 | |
| `uint16` | 2 | 0 | 65,535 | |
| `int32` | 4 | -2,147,483,648 | 2,147,483,647 | |
| `uint32` | 4 | 0 | 4,294,967,295 | |
| `int64` | 8 | -9.2×10¹⁸ | 9.2×10¹⁸ | Python default int |
| `float16` | 2 | -65,504 | 65,504 | ~3 sig. digits; ULP at V ≈ V×0.001 |
| `float32` | 4 | ~-3.4×10³⁸ | ~3.4×10³⁸ | ~7 sig. digits; exact integers up to 2²⁴ = 16,777,216 |
| `float64` | 8 | ~-1.8×10³⁰⁸ | ~1.8×10³⁰⁸ | ~15 sig. digits; exact integers up to 2⁵³ |

**float16 precision trap:**
- Resolution ≈ 0.001 is **relative** (like machine epsilon), NOT absolute
- At value V: smallest representable increment ≈ V × 0.001
- `float16(10000) + 1 = 10000` — increment needed is ~10, so 1 is lost
- `float16(100) + 0.05 = 100.06` — ULP at 100 is 0.0625, so 0.05 rounds up

**float32 exact integer limit:**
- Integers up to **2²⁴ = 16,777,216** are exact in float32
- Above that: consecutive integers are no longer distinguishable
- `float32(16_777_217) == float32(16_777_216)` → True

---

## Pandas Indexing for Speed

```python
# Slow: O(n) boolean scan
df[df['date'] == '2023-06-15']['value'].sum()

# Fast: O(log n) binary search after sorting index
df_idx = df.set_index('date').sort_index()
df_idx.loc['2023-06-15']['value'].sum()   # ~15000x faster for repeated queries
```

---

## PyArrow CSV Loading

```python
from pyarrow import csv
import pyarrow as pa

# Fast multi-threaded CSV load (~3.7x faster than pd.read_csv)
table = csv.read_csv('file.csv')
df = table.to_pandas()

# With type optimization
convert_options = csv.ConvertOptions(
    column_types={
        'value': pa.float32(),
        'parameterId': pa.dictionary(pa.int32(), pa.string()),  # category
    }
)
table = csv.read_csv('file.csv', convert_options=convert_options)
```

---

## Parquet

```python
import pyarrow.parquet as pq
import pyarrow as pa

# Write
table = pa.Table.from_pandas(df)
pq.write_table(table, 'file.parquet')

# Read (PyArrow — faster ~400ms vs Pandas ~1s)
table = pq.read_table('file.parquet')
table = pq.read_table('file.parquet', columns=['parameterId', 'value'])  # column pruning

# Chunked Parquet read
pf = pq.ParquetFile('file.parquet')
pf.num_row_groups                                    # number of chunks
group = pf.read_row_group(i)                         # read chunk i
group = pf.read_row_group(i, columns=['a', 'b'])     # with column pruning
df = group.to_pandas()

# CSV → chunked Parquet conversion
writer = None
for chunk in pd.read_csv('big.csv.zip', chunksize=100_000):
    table = pa.Table.from_pandas(chunk)
    if writer is None:
        writer = pq.ParquetWriter('out.parquet', schema=table.schema)
    writer.write_table(table)
writer.close()
```

---

## File Format Comparison

| Format | Size | Read speed | Use when |
|---|---|---|---|
| CSV | ~1.2 GB | slow ~12s | human readable |
| CSV.zip | compressed | medium ~11s | save disk |
| Parquet | ~86 MB | fast ~400ms | repeated analysis |

Parquet ~14x smaller than CSV, ~10-30x faster to read.

---

## Exam Traps

| Trap | Correct |
|---|---|
| `memory_usage()` for object columns | Need `deep=True` |
| Low-cardinality → datetime | Low-cardinality → `category`; timestamps → `datetime` |
| int16 max = 65535 | int16 max = 32,767 (uint16 max = 65,535) |
| Convert integers to float32 for smaller size | Risk of precision loss — avoid |
| Can re-iterate `pd.read_csv(chunksize=...)` | Exhausted after one pass — reopen file |
