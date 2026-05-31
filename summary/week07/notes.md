# Week 7 — High-Performance Pandas & Apache Arrow

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Overview](#overview)
- [Theory & Concepts](#theory-concepts)
  - [From Array Data to Tabular Data](#from-array-data-to-tabular-data)
  - [Pandas Memory Optimization](#pandas-memory-optimization)
  - [Apache Arrow](#apache-arrow)
  - [Parquet File Format](#parquet-file-format)
  - [DMI Weather Data (course example)](#dmi-weather-data-course-example)
- [Mathematical Content](#mathematical-content)
- [Key Code Examples](#key-code-examples)
  - [Loading and inspecting memory](#loading-and-inspecting-memory)
  - [Loading with PyArrow (returns Arrow Table)](#loading-with-pyarrow-returns-arrow-table)
  - [Loading with PyArrow and converting to Pandas](#loading-with-pyarrow-and-converting-to-pandas)
  - [PyArrow with ConvertOptions for memory optimization on load](#pyarrow-with-convertoptions-for-memory-optimization-on-load)
  - [Parquet read/write](#parquet-readwrite)
- [Reducing DataFrame Memory](#reducing-dataframe-memory)
- [File Format Comparison](#file-format-comparison)
- [Exercise Highlights](#exercise-highlights)
  - [Exercise 1 — Storage and Reading Files with Pandas](#exercise-1-storage-and-reading-files-with-pandas)
  - [Exercise 2 — Reading Files with Arrow](#exercise-2-reading-files-with-arrow)
  - [Exercise 3 — Parquet Files](#exercise-3-parquet-files)
  - [Exercise 4 — Pandas Fast Operations (total precipitation)](#exercise-4-pandas-fast-operations-total-precipitation)
- [Key Takeaways](#key-takeaways)

---

## Overview

This week transitions from pure array/NumPy data to tabular (heterogeneous) data. The core question is: how do you work with large DataFrames efficiently, both in terms of memory and speed? The lecture covers the three main bottlenecks in a typical Pandas workflow — **load speed**, **memory use**, and **operation speed** — and introduces Apache Arrow and Parquet as tools to address all three. The DMI weather dataset (January 2023, ~8 million rows) is the running example throughout all exercises.

Typical Pandas workflow and where bottlenecks appear:

| Step | Bottleneck |
|------|------------|
| 1. Load data | Load speed |
| 2. Explore data | Memory use + speed |
| 3. Settle on workflow | Memory use + speed |

---

## Theory & Concepts

### From Array Data to Tabular Data

- Arrays (NumPy): **homogeneous** dtype, single contiguous memory block, fast
- DataFrames (Pandas): **heterogeneous** — each column is a `Series` with its own dtype (often backed by a NumPy array), plus row labels (index) and column labels
- A `Series` = index (row labels) + data (NumPy array)
- Operations on Series align by **index**, not by position — mismatched indices produce `NaN`

Key implication: arithmetic on two Series with different indices aligns on the shared index, promotes dtype to float64 to accommodate NaN, and can silently produce unexpected results.

Example from quiz:
```python
a = pd.Series([1,2,3], index=[1,2,3])
b = pd.Series([1,2,3], index=[1,3,2])
a + b  # => index 1: 1+1=2, index 2: 1+3=5 (b[2]=3), index 3: 2+2=...
# Result: [2, 5, 5]  (aligns by label, not position)

a = pd.Series([1,2,3])          # default index 0,1,2
a + a[::-1]                     # reversed has same index values => [4,4,4]
```

### Pandas Memory Optimization

Three main strategies, in order of impact on the DMI dataset:

1. **Date/datetime columns**: Pandas reads timestamps as `object` (Python strings). Each string is a separate heap allocation — very expensive. Convert with `pd.to_datetime()` to get `datetime64[ns]` — a fixed 8 bytes per value.

   - Example: `tpep_pickup_datetime` as `object` = 464 MB; as `datetime64[ns]` = 48 MB (10x reduction)

2. **Categorical variables**: Columns with a small number of unique string values (e.g., `parameterId` with 47 unique values, `store_and_fwd_flag` with 3) stored as `object` carry full Python string overhead per row. Convert to `category` dtype — Pandas stores a dictionary of unique values plus an integer code per row.

   - Example: `store_and_fwd_flag` as `object` = 352 MB; as `category` = 6 MB (58x reduction)
   - Manual recoding is equivalent: `fillna(" ").apply(ord).apply(lambda x: [32,78,89].index(x)-1).astype(np.int8)` = 6 MB

3. **Smaller numeric types**: Default Pandas reads integers as `int64` (8 bytes) and floats as `float64` (8 bytes). Downcast when range allows:
   - `int64` → `int16` or `int32` using `pd.to_numeric(col, downcast='integer')`
   - `float64` → `float32` using `pd.to_numeric(col, downcast='float')` (check for overflow first)
   - `stationId` has 247 unique values but min=4203, max=34339, so must use `int16` (not `int8`)
   - `float16` overflows for `value` column; `float32` works (error < 6e-5)

Inspecting memory use:

```python
df.info(memory_usage='deep')        # summary with total
df.memory_usage(deep=True)          # per-column bytes
df.memory_usage(deep=True).sum()    # total bytes
```

Why `deep=True` matters: without it, `object` columns report only the pointer array size, not the actual string heap memory.

DMI dataset memory profile (before/after optimization):

| Column | Before (MB) | After (MB) | Change |
|--------|------------|------------|--------|
| coordsx | 62 | small (category) | large |
| coordsy | 62 | small (category) | large |
| created | 652 | ~48 (datetime64) | ~14x |
| observed | 597 | ~48 (datetime64) | ~12x |
| parameterId | 546 | ~6 (category) | ~90x |
| stationId | 62 | ~16 (int16) | ~4x |
| value | 62 | ~31 (float32) | ~2x |
| **Total** | **~2045 MB** | **~200 MB** | **~10x** |

### Apache Arrow

Apache Arrow is a cross-language, columnar in-memory format designed for analytics:

- **Columnar layout**: all values of one column are stored contiguously, enabling cache-efficient scans
- **Zero-copy reads**: different tools (Pandas, Spark, DuckDB, etc.) can share Arrow memory without copying
- **Efficient string storage**: strings are stored in a single large byte buffer; the column stores offset+length per entry — far less overhead than Python objects
- **Fast CSV parser**: PyArrow's CSV reader is multi-threaded and substantially faster than Pandas

Speed comparison on the DMI CSV (unzipped):

| Method | Time |
|--------|------|
| `pd.read_csv` | ~12.7 s |
| `pd.read_csv` (from zip) | ~11.1 s |
| `pyarrow.csv.read_csv` | ~3 s (~3.7x faster) |
| PyArrow + `.to_pandas()` | ~3.7 s (still faster than pure Pandas) |

Memory comparison:

| Representation | Size |
|----------------|------|
| Pandas default | ~2045 MB |
| PyArrow Table (default) | ~507 MB |
| Pandas after Arrow conversion | ~919 MB |
| PyArrow with ConvertOptions | ~312 MB |

Arrow's string representation is much more compact than Pandas `object` (Python strings), which is why the PyArrow table is smaller than the Pandas DataFrame created from it.

### Parquet File Format

Parquet is a binary columnar file format designed for efficient storage and retrieval:

- **Binary encoding**: numbers stored as binary, not text — far more compact than CSV
- **Column-level compression**: each column compressed independently (often Snappy or Zstd)
- **Schema embedded**: types preserved on disk — no re-parsing needed on load
- **Predicate pushdown**: can skip entire row groups without reading them

DMI data Parquet benchmark:

| Format | File size | Read time (Pandas) | Read time (PyArrow) | Write time |
|--------|-----------|--------------------|---------------------|------------|
| CSV (unzipped) | ~1.2 GB | ~12.7 s | ~3 s | slow |
| CSV.zip | compressed | ~11.1 s | N/A | — |
| Parquet | **86 MB** | **~1 s** | **~400 ms** | ~1.6 s |

Parquet is the preferred format for storing and reloading data between sessions.

### DMI Weather Data (course example)

The dataset is the Danish Meteorological Institute (DMI) weather observations for January 2023, sourced from `/dtu/projects/02613_2025/data/dmi/2023_01.csv.zip`. Columns:

- `coordsx`, `coordsy` — station coordinates (float, low cardinality — good candidates for category)
- `created` — timestamp data entered in DB (object → datetime64)
- `observed` — timestamp of measurement (object → datetime64)
- `parameterId` — parameter type, e.g. `precip_past10min`, `wind_dir_past1h` (object, 47 unique → category)
- `stationId` — integer station ID, 247 unique values (int64 → int16)
- `value` — measured value (float64 → float32)

---

## Mathematical Content

**Memory footprint formula** for a numeric column of N rows with dtype of k bytes:
```
memory = N * k bytes
```

For the DMI dataset with ~8 million rows:
- `float64` column: 8M * 8 = 64 MB
- `float32` column: 8M * 4 = 32 MB
- `int64` column: 8M * 8 = 64 MB
- `int16` column: 8M * 2 = 16 MB
- `object` column: 64 MB pointer array + heap memory per string (depends on string length and count)

**Speed-up calculation**: if raw loop processes 100K rows in 5.5 s, vectorized in 270 ms:
```
speedup = 5500 ms / 270 ms ≈ 20x
```

For the full dataset (8M rows):
- `apply()`: ~46 s
- Vectorized boolean indexing: ~400 ms → 115x speedup
- Indexed lookup (`set_index` + `.loc`): ~3 ms (excluding index build) → 15,000x vs raw loop

---

## Key Code Examples

### Loading and inspecting memory

```python
import pandas as pd

def df_memsize(df: pd.DataFrame):
    return df.memory_usage(deep=True).sum()

if __name__ == '__main__':
    df = pd.read_csv('week7/2023_01.csv.zip')
    print(df_memsize(df))
```

### Loading with PyArrow (returns Arrow Table)

```python
from pyarrow import csv

def pyarrow_load(path):
    return csv.read_csv(path)

if __name__ == "__main__":
    df = pyarrow_load('week7/2023_01.csv')
    print(df)
```

### Loading with PyArrow and converting to Pandas

```python
from pyarrow import csv

def pyarrow_load(path):
    pyarrow_df = csv.read_csv(path)
    return pyarrow_df.to_pandas()

if __name__ == "__main__":
    df = pyarrow_load('week7/2023_01.csv')
    print(df)
```

### PyArrow with ConvertOptions for memory optimization on load

```python
import pyarrow as pa
from pyarrow import csv

convert_options = csv.ConvertOptions(
    column_types={
        'value': pa.float32(),
        'parameterId': pa.dictionary(pa.int32(), pa.string()),
        'coordsx': pa.dictionary(pa.int32(), pa.float64()),
        'coordsy': pa.dictionary(pa.int32(), pa.float64()),
    }
)
table = csv.read_csv('2023_01.csv', convert_options=convert_options)
# Reduces table from 507 MB to 312 MB
```

### Parquet read/write

```python
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Write
df = pd.read_csv('2023_01.csv')
table = pa.Table.from_pandas(df)
pq.write_table(table, '2023_01.parquet')

# Read with Pandas
df = pd.read_parquet('2023_01.parquet')

# Read with PyArrow (faster)
table = pq.read_table('2023_01.parquet')
```

---

## Reducing DataFrame Memory

`reduce_dataframe.py` implements an automatic dtype optimization pass over the DMI DataFrame:

```python
import pandas as pd

def summarize_columns(df):
    print(pd.DataFrame([
        (
            c,
            df[c].dtype,
            len(df[c].unique()),
            df[c].memory_usage(deep=True) // (1024**2)
        ) for c in df.columns
    ], columns=['name', 'dtype', 'unique', 'size (MB)']))
    print('Total size:', df.memory_usage(deep=True).sum() / 1024**2, 'MB')

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

if __name__ == '__main__':
    df = reduce_dmi_df(pd.read_csv('week7/2023_01.csv.zip'))
    print(summarize_columns(df))
```

The logic:
- `created` and `observed`: convert to `datetime64` using ISO8601 format string
- Integer columns: `pd.to_numeric(downcast="integer")` finds the smallest int type that fits
- Float columns: `pd.to_numeric(downcast="float")` tries `float32` first
- Object/string columns: `.astype("category")` — stores as integer codes + a lookup table

---

## File Format Comparison

| Format | Read Speed | Write Speed | Compression | File Size | Use When |
|--------|-----------|------------|-------------|-----------|----------|
| CSV | slow (~12 s) | slow | no | ~1.2 GB | sharing with non-Python tools, human-readable |
| CSV.zip | medium (~11 s) | slow | yes (zip) | compressed | saving disk when CSV is required |
| Parquet | fast (~1 s Pandas, ~400 ms Arrow) | medium (~1.6 s) | yes (columnar) | **86 MB** | storing for repeated analysis |
| Arrow/Feather | fastest | fast | optional | small | in-process / same-session transfers |

Key insight: Parquet is ~14x smaller than raw CSV and ~10-30x faster to read back. Once you have processed and saved as Parquet, subsequent loads are much cheaper.

---

## Exercise Highlights

### Exercise 1 — Storage and Reading Files with Pandas

- Comparing `unzip` + `read_csv` vs. direct `pd.read_csv('file.csv.zip')`:
  - Unzip (4.7 s) + read CSV (12.7 s) = **17.4 s total**
  - Direct zip read: **11.1 s** — faster, less disk space
- Default DataFrame size: **~2045 MB** for the DMI January 2023 data
- Memory reduction strategy (columns to change):
  - `created`, `observed` → `datetime64` (string to 8-byte int)
  - `parameterId` → `category` (47 unique values)
  - `coordsx`, `coordsy` → `category` (224/219 unique station coords)
  - `stationId` → `int16` (range 4203–34339 fits in uint16)
  - `value` → `float32` (float16 overflows, float32 error < 6e-5)

### Exercise 2 — Reading Files with Arrow

- PyArrow `csv.read_csv` is **~3.7x faster** than `pd.read_csv` on the same unzipped file
- Converting Arrow table → Pandas adds ~700 ms; total still faster than pure Pandas
- PyArrow table: **507 MB** vs Pandas default: **2045 MB** — Arrow's string storage is much more efficient
- With `ConvertOptions` (dictionary encoding + float32): **312 MB**

### Exercise 3 — Parquet Files

- CSV file → Parquet: from ~1.2 GB to **86 MB** (14x smaller)
- Write time: ~1.6 s; Read time: ~1 s (Pandas), ~400 ms (PyArrow)
- Both dramatically faster than reading CSV

### Exercise 4 — Pandas Fast Operations (total precipitation)

Goal: sum all `value` where `parameterId == 'precip_past10min'` — answer is **12548.63**.

| Implementation | 100K rows | 8M rows | Speedup (full) |
|----------------|-----------|---------|----------------|
| Raw Python loop (`iloc`) | ~5.5 s | ~440 s est. | 1x |
| `apply()` with lambda | ~780 ms | ~46 s | 10x |
| Vectorized boolean indexing | ~270 ms | ~400 ms | **1150x** |
| `set_index` + `.loc` (excl. build) | — | ~3 ms | **15,000x** |

Vectorized solution:
```python
def total_precip(df):
    return df[df['parameterId'] == 'precip_past10min']['value'].sum()
```

Indexed solution (good when many queries on the same index):
```python
df_pid = df.set_index('parameterId').sort_index()
precip = df_pid.loc['precip_past10min']['value'].sum()  # 3 ms
# Building the index takes ~6.3 s — only worth it for multiple queries
```

---

## Key Takeaways

1. **Data types dominate memory cost.** The difference between `object` and `category` or `datetime64` can be 10-90x for string columns. Always run `df.info(memory_usage='deep')` or `summarize_columns` before doing anything else with a new dataset.

2. **`object` dtype is a performance trap.** An `object` column stores Python object pointers — each string is a separate heap allocation. This is slow to scan, slow to compare, and uses more memory than necessary. Convert strings to `category` or `datetime64` as early as possible.

3. **PyArrow is faster for CSV loading.** It is multi-threaded and uses a more efficient string representation. Even if you need a Pandas DataFrame in the end, loading with Arrow and converting is faster than loading with Pandas directly.

4. **Parquet is the right format for analysis.** Once data is processed, save as Parquet. It is 14x smaller than raw CSV, 10x faster to reload, and preserves dtypes — no re-parsing or re-typing on load.

5. **Vectorized operations over loops.** Raw Python `for` loops over DataFrame rows (`iloc`) are extremely slow. `apply()` gives a modest speedup. Vectorized boolean indexing (`df[mask]['col'].sum()`) is 1000x+ faster and should be the default approach.

6. **Indexing is powerful but has setup cost.** `df.set_index('col').sort_index()` enables O(log n) lookups via `.loc`, which can be 15,000x faster than a loop. But building the index takes several seconds — only worth it when performing many repeated queries on the same column.

7. **Index alignment is a Pandas gotcha.** Series operations align by index label, not position. Two Series with the same values but different index assignments will produce `NaN` where indices don't overlap. If you want positional arithmetic, call `.to_numpy()` first.
