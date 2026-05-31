# Week 7 Exercises — High-Performance Pandas + Apache Arrow

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Section 1 — Storage and Reading Files with Pandas](#section-1-storage-and-reading-files-with-pandas)
  - [Exercise 1.1 `[PRACTICE]`](#exercise-11-practice)
  - [Exercise 1.2 `[AUTOLAB]`](#exercise-12-autolab)
  - [Exercise 1.3 `[PRACTICE]`](#exercise-13-practice)
  - [Exercise 1.4 `[AUTOLAB]`](#exercise-14-autolab)
- [Section 2 — Reading Files with Arrow](#section-2-reading-files-with-arrow)
  - [Exercise 2.1 `[AUTOLAB]`](#exercise-21-autolab)
  - [Exercise 2.2 `[AUTOLAB]`](#exercise-22-autolab)
  - [Exercise 2.3 `[PRACTICE]`](#exercise-23-practice)
  - [Exercise 2.4 `[PRACTICE]`](#exercise-24-practice)
- [Section 3 — Parquet Files](#section-3-parquet-files)
  - [Exercise 3.1 `[AUTOLAB]`](#exercise-31-autolab)
  - [Exercise 3.2 `[PRACTICE]`](#exercise-32-practice)
- [Section 4 — Pandas Fast Operations](#section-4-pandas-fast-operations)
  - [Exercise 4.1 `[PRACTICE]`](#exercise-41-practice)
  - [Exercise 4.2 `[PRACTICE]`](#exercise-42-practice)
  - [Exercise 4.3 `[PRACTICE]`](#exercise-43-practice)
  - [Exercise 4.4 `[AUTOLAB]`](#exercise-44-autolab)
  - [Exercise 4.5 `[PRACTICE]`](#exercise-45-practice)

---

**Dataset:** Weather data from DMI (Danmarks Meteorologiske Institut), January 2023.
Located at `/dtu/projects/02613_2025/data/dmi/2023_01.csv.zip`.

Each row is a single parameter measurement from one of DMI's stations. Columns:
1. `coordsx` — first coordinate of measurement station
2. `coordsy` — second coordinate of measurement station
3. `created` — time measurement was entered in database
4. `observed` — time measurement was observed
5. `parameterId` — measured parameter
6. `stationId` — ID of measurement station
7. `value` — measured value

---

## Section 1 — Storage and Reading Files with Pandas

### Exercise 1.1 `[PRACTICE]`

The CSV file is stored as a zip file. Compare the time it takes for the following two approaches:

1. First unpacking the zip file with the `unzip` command and then reading the CSV file with `read_csv`.
2. Read the zip directly with `read_csv`.

Which one is faster?

> **Solution:**
>
> Reading directly from the zip is faster. Unpacking the zip first (~4.7 s) plus reading the CSV (~12.7 s) is slower than reading the zipped CSV directly with Pandas (~11.1 s). Reading directly from zip also saves disk space since no intermediate uncompressed file is needed.

---

### Exercise 1.2 `[AUTOLAB]`

Load the data to a pandas dataframe (use `pd.read_csv()`). How much memory does the data frame occupy? Create a function `df_memsize` that takes a Pandas DataFrame as input and returns its size in bytes.

> **Solution:**
>
> The dataframe occupies around 2045 MB (~2 GB) of memory.
>
> ```python
> import pandas as pd
>
> def df_memsize(df: pd.DataFrame):
>     return df.memory_usage(deep=True).sum()
>
> if __name__ == '__main__':
>     df = pd.read_csv('week7/2023_01.csv.zip')
>     print(df_memsize(df))
> ```

---

### Exercise 1.3 `[PRACTICE]`

Check the columns and list all the changes you can do to save memory. Hint: check sections 7.1.2 and 7.1.3 on Fast Python for inspiration. You can use the following function to summarize the dataframe:

```python
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
```

> **Solution:**
>
> Initial column summary:
>
> ```
>         name    dtype   unique  size (MB)
> 0      coordsx  float64      224         62
> 1      coordsy  float64      219         62
> 2      created   object  8142495        652
> 3     observed   object    44640        597
> 4  parameterId   object       47        546
> 5    stationId    int64      247         62
> 6        value  float64    11532         62
> Total size: 2045.0958003997803 MB
> ```
>
> Recommended reductions:
> - **`created` and `observed`**: convert from `object` to `datetime` — stores timestamps as integers instead of Python string objects.
> - **`parameterId`**: only 47 unique values — encode as `category` (stores indices into a small dictionary instead of repeated strings).
> - **`stationId`**: 247 unique values, range 4203–34339 — fits in `int16` (max 65535), saving space vs `int64`.
> - **`value`**: check if `float32` is sufficient — `float16` overflows but `float32` is accurate within ~6e-5.
> - **`coordsx` / `coordsy`**: only 224/219 unique values (one per station) — encode as `category`.

---

### Exercise 1.4 `[AUTOLAB]`

Perform the changes to reduce the memory usage. What is the new size of the dataframe? Create a function `reduce_dmi_df` that takes a Pandas DataFrame as input and returns a transformed DataFrame where at least 3 columns have had their memory use reduced.

> **Solution:**
>
> ```python
> import pandas as pd
>
> def summarize_columns(df):
>     print(pd.DataFrame([
>         (
>             c,
>             df[c].dtype,
>             len(df[c].unique()),
>             df[c].memory_usage(deep=True) // (1024**2)
>         ) for c in df.columns
>     ], columns=['name', 'dtype', 'unique', 'size (MB)']))
>     print('Total size:', df.memory_usage(deep=True).sum() / 1024**2, 'MB')
>
> def reduce_dmi_df(df):
>     for column in df.columns:
>         if column in ("created", "observed"):
>             df[column] = pd.to_datetime(df[column], format="ISO8601")
>         elif pd.api.types.is_integer_dtype(df[column]):
>             df[column] = pd.to_numeric(df[column], downcast="integer")
>         elif pd.api.types.is_float_dtype(df[column]):
>             df[column] = pd.to_numeric(df[column], downcast="float")
>         elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
>             df[column] = df[column].astype("category")
>     return df
>
> if __name__ == '__main__':
>     df = reduce_dmi_df(pd.read_csv('week7/2023_01.csv.zip'))
>     print(summarize_columns(df))
> ```

---

## Section 2 — Reading Files with Arrow

### Exercise 2.1 `[AUTOLAB]`

Load the DMI data from January 2023, as we did in the last exercise, but this time using PyArrow. Compute the time it took to read the CSV, what was the speed up when comparing with pandas? Create a function `pyarrow_load` that takes a path to a CSV file as input, loads the file with PyArrow and returns the PyArrow table.

> **Solution:**
>
> PyArrow reads the CSV in ~3 seconds vs ~11.1 seconds for Pandas — roughly **3.7x faster**.
>
> ```python
> from pyarrow import csv
>
> def pyarrow_load(path):
>     return csv.read_csv(path)
>
> if __name__ == "__main__":
>     df = pyarrow_load('week7/2023_01.csv')
>     print(df)
> ```

---

### Exercise 2.2 `[AUTOLAB]`

Now convert the PyArrow table to a pandas data frame, how long does it take? Is the total time (time to load the data with PyArrow + the time to convert it to a pandas data frame) faster or slower than pure Pandas? Modify your previous function to return a Pandas DataFrame instead of a PyArrow table.

> **Solution:**
>
> Converting the PyArrow table to Pandas adds ~700 ms, but the combined load + convert time (~3.7 s) is still faster than loading with Pandas directly (~11.1 s).
>
> ```python
> from pyarrow import csv
>
> def pyarrow_load(path):
>     pyarrow_df = csv.read_csv(path)
>     return pyarrow_df.to_pandas()
>
> if __name__ == "__main__":
>     df = pyarrow_load('week7/2023_01.csv')
>     print(df)
> ```

---

### Exercise 2.3 `[PRACTICE]`

What was the size of the PyArrow table you got after loading the data? What was the size of the data frame after the conversion? Why are they different?

> **Solution:**
>
> - PyArrow table size: **507 MB**
> - Pandas DataFrame after conversion: **919 MB**
>
> They differ because `parameterId` is stored differently. In PyArrow, all strings are stored in a single contiguous buffer with each column entry storing only an offset and length — very compact. In Pandas, each string becomes a Python object with its own heap allocation, adding substantial overhead. The `created`/`observed` columns and `stationId` are already converted to efficient types by PyArrow, so those are smaller than the original Pandas load (2045 MB), but `parameterId` is the main remaining source of bloat in the Pandas representation.

---

### Exercise 2.4 `[PRACTICE]`

Modify your loading code in order to perform (as many as possible of) the memory reducing operations you did in exercise 1.4. What is the new size of the PyArrow table?

> **Solution:**
>
> Use `csv.ConvertOptions` to apply type conversions at load time:
>
> ```python
> import pyarrow as pa
> from pyarrow import csv
>
> convert_options = csv.ConvertOptions(
>     column_types={
>         'value': pa.float32(),
>         'parameterId': pa.dictionary(pa.int32(), pa.string()),
>         'coordsx': pa.dictionary(pa.int32(), pa.float64()),
>         'coordsy': pa.dictionary(pa.int32(), pa.float64()),
>     }
> )
>
> table = csv.read_csv('2023_01.csv', convert_options=convert_options)
> ```
>
> The PyArrow table size drops from 507 MB to **312 MB**.

---

## Section 3 — Parquet Files

### Exercise 3.1 `[AUTOLAB]`

Make a Python program that receives the path to a CSV file as a command line argument, loads the CSV file and finally saves it again as a Parquet file. What is the difference in size between the original CSV file and the Parquet file?

> **Solution:**
>
> The resulting Parquet file is **86 MB** vs the original uncompressed CSV (~500+ MB) — a large reduction due to binary columnar storage and built-in compression.
>
> ```python
> import sys
> import pandas as pd
>
> if __name__ == '__main__':
>     csv_path = sys.argv[1]
>     parquet_path = csv_path.replace('.csv', '.parquet')
>     df = pd.read_csv(csv_path)
>     df.to_parquet(parquet_path)
>     print(f"Saved to {parquet_path}")
> ```

---

### Exercise 3.2 `[PRACTICE]`

Measure the times to read and write the DMI data to parquet files and compare it to reading and writing CSV.

> **Solution:**
>
> | Operation | Time |
> |---|---|
> | Write Parquet | ~1.6 s |
> | Read Parquet (Pandas `read_parquet`) | ~1.0 s |
> | Read Parquet (PyArrow `read_table`) | ~0.4 s |
> | Read CSV (Pandas) | ~11.1 s |
> | Read CSV (PyArrow) | ~3.0 s |
>
> Both Parquet readers are significantly faster than reading CSV. The binary columnar format avoids text parsing overhead and supports predicate pushdown for even faster partial reads.

---

## Section 4 — Pandas Fast Operations

**Goal:** Compute the total precipitation across all measurement stations by summing all `value` records where `parameterId == 'precip_past10min'`.

Reference raw Python implementation:

```python
def total_precip(df):
    total = 0.0
    for i in range(len(df)):
        row = df.iloc[i]
        if row['parameterId'] == 'precip_past10min':
            total += row['value']
    return total
```

---

### Exercise 4.1 `[PRACTICE]`

Run the raw python implementation and time it. *Note:* Use the Pandas DataFrame method `sample` to subsample a smaller DataFrame, to avoid excessive wait times. How long did it take to execute?

> **Solution:**
>
> On a 100K-row sample (~1.2% of full data), the raw loop takes approximately **5.5 seconds**. The bottleneck is `df.iloc[i]` which is a Python-level row access — every iteration incurs full Python object overhead.

---

### Exercise 4.2 `[PRACTICE]`

Build a pandas-based implementation using the `apply()` method to perform the same operation and time it. What was the speed up?

> **Solution:**
>
> ```python
> def total_precip(df):
>     total = df.apply(
>         lambda row: (
>             row['value'] if row['parameterId'] == 'precip_past10min' else 0.0
>         ),
>         axis=1
>     ).sum()
>     return total
> ```
>
> On the 100K subset: **~780 ms** (~7x faster than the raw loop). On the full dataset: ~46 seconds. Result: 12548.63.
>
> `apply()` is still row-by-row Python iteration under the hood, so the speedup is modest.

---

### Exercise 4.3 `[PRACTICE]`

Now write a pure Pandas version using a vectorized approach. How long did it take to execute? What was the speed up in comparison with the two previous implementations?

> **Solution:**
>
> ```python
> def total_precip(df):
>     total = df[df['parameterId'] == 'precip_past10min']['value'].sum()
>     return total
> ```
>
> - 100K subset: **~270 ms** — 21x faster than raw loop, 3x faster than `apply`
> - Full dataset: **~400 ms** — 112x faster than `apply`
>
> Vectorization works because the boolean mask and `.sum()` are both executed in compiled NumPy/Pandas C code rather than Python loops.

---

### Exercise 4.4 `[AUTOLAB]`

Turn your previous implementation into a Python program that receives the path to a CSV file as a command line argument, loads the CSV, computes the total precipitation and finally prints the result (and only the result).

> **Solution:**
>
> ```python
> import sys
> import pandas as pd
>
> def total_precip(df):
>     return df[df['parameterId'] == 'precip_past10min']['value'].sum()
>
> if __name__ == '__main__':
>     csv_path = sys.argv[1]
>     df = pd.read_csv(csv_path)
>     print(total_precip(df))
> ```

---

### Exercise 4.5 `[PRACTICE]`

A significant portion of the work in the vectorized version is doing the indexing to extract the relevant rows. Rewrite the vectorized version to first use the `parameterId` column as an index. Excluding the time it takes to build the index, how long does it take to compute the total precipitation now? Including the time to build the index? When would one prefer one to the other? Hint: see section 7.2.1 in Fast Python.

> **Solution:**
>
> ```python
> df_pid = df.set_index('parameterId').sort_index()
> precip = df_pid.loc['precip_past10min']['value'].sum()
> ```
>
> - **Excluding** index build time: **~3 ms** — 130x faster than vectorized boolean mask, 1500x faster than `apply`
> - **Including** index build time: **~6.6 s** — much slower overall for a single query
>
> **When to use an index:** Only worth it when performing many queries over different `parameterId` values. The index build cost is amortized across multiple lookups. For a single one-off query, the plain vectorized boolean mask is more efficient.
