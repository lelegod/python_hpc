# Week 7 — Pandas & PyArrow Syntax Reference

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

### dtype ranges

| dtype | Range |
|---|---|
| int8 | -128 to 127 |
| uint8 | 0 to 255 |
| int16 | -32,768 to 32,767 |
| int32 | ±2,147,483,647 |
| float32 | ~7 decimal digits |
| float64 | ~15 decimal digits |

### dtype byte sizes: uint8=1, int16=2, int32=4, float32=4, int64=8, float64=8

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
