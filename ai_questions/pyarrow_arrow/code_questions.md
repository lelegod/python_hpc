# PyArrow & Apache Arrow — Code-Based MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Return Type of pyarrow.csv.read_csv()](#q1--return-type-of-pyarrowcsvread_csv)
- [Q2 — Column Access on a pyarrow.Table](#q2--column-access-on-a-pyarrowtable)
- [Q3 — Converting Arrow to Pandas](#q3--converting-arrow-to-pandas)
- [Q4 — ConvertOptions for Type Override](#q4--convertoptions-for-type-override)
- [Q5 — Memory Size Comparison: Arrow vs Pandas](#q5--memory-size-comparison-arrow-vs-pandas)
- [Q6 — Parquet Column Projection](#q6--parquet-column-projection)
- [Q7 — ChunkedArray vs Array](#q7--chunkedarray-vs-array)
- [Q8 — from_pandas() and Zero-Copy](#q8--from_pandas-and-zero-copy)
- [Q9 — Writing and Reading Parquet](#q9--writing-and-reading-parquet)
- [Q10 — Dictionary Encoding at Load Time](#q10--dictionary-encoding-at-load-time)
- [Q11 — Arrow compute Function vs Pandas](#q11--arrow-compute-function-vs-pandas)
- [Q12 — Schema Inspection](#q12--schema-inspection)
- [Q13 — Parquet Column Projection with pyarrow.parquet](#q13--parquet-column-projection-with-pyarrowparquet)
- [Q14 — Arrow Slice Semantics](#q14--arrow-slice-semantics)
- [Q15 — ConvertOptions column_types vs dtype](#q15--convertoptions-column_types-vs-dtype)
- [Q16 — to_pandas() String Memory Inflation](#q16--to_pandas-string-memory-inflation)
- [Q17 — Parquet Row Group Count](#q17--parquet-row-group-count)
- [Q18 — Total Pipeline: Fastest Load for Single Aggregation](#q18--total-pipeline-fastest-load-for-single-aggregation)

---

> Format: Each question shows PyArrow/Parquet Python code with behaviour to predict or fix.
> Exam frequency: **Week 7 topic**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#question-1)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — Return Type of pyarrow.csv.read_csv()

```python
import pyarrow.csv as pa_csv

result = pa_csv.read_csv('2023_01.csv')
print(type(result))
```

What does this print?

**A)** `<class 'pandas.core.frame.DataFrame'>`
**B)** `<class 'pyarrow.lib.Table'>`
**C)** `<class 'pyarrow.lib.RecordBatch'>`
**D)** `<class 'list'>`

**Answer: B**

- A) Incorrect — `pyarrow.csv.read_csv()` never returns a Pandas DataFrame; that requires an explicit `.to_pandas()` call.
- B) Correct — The return type is `pyarrow.lib.Table` (displayed as `pyarrow.Table`). This is Arrow's primary in-memory table structure, holding all columns as Arrow arrays.
- C) Incorrect — A `RecordBatch` is a single chunk of rows; `read_csv()` assembles all batches into a complete `Table` before returning.
- D) Incorrect — `read_csv()` never returns a plain Python list.

---

## Q2 — Column Access on a pyarrow.Table

```python
import pyarrow.csv as pa_csv

table = pa_csv.read_csv('2023_01.csv')
col = table['value']
print(type(col))
```

What is the type of `col`?

**A)** `pandas.Series`
**B)** `numpy.ndarray`
**C)** `pyarrow.lib.ChunkedArray`
**D)** `pyarrow.lib.Array`

**Answer: C**

- A) Incorrect — `pyarrow.Table` does not return Pandas objects; `.to_pandas()` is required for that.
- B) Incorrect — Arrow and NumPy are distinct; column access on a `pyarrow.Table` returns an Arrow type, not a NumPy array. Call `.to_numpy()` or `.to_pydict()` to get NumPy/Python types.
- C) Correct — Accessing a column on a `pyarrow.Table` via `table['column_name']` returns a `pyarrow.ChunkedArray`, which is one or more contiguous Arrow `Array` objects treated as a single logical column.
- D) Incorrect — The return from `table[col]` is specifically a `ChunkedArray` (even if it contains only one chunk internally), not a plain `Array`.

---

## Q3 — Converting Arrow to Pandas

```python
import pyarrow.csv as pa_csv

table = pa_csv.read_csv('2023_01.csv')
df = table.to_pandas()
print(type(df))
```

What does this code produce, and which statement about the conversion is accurate?

**A)** Returns a `pyarrow.Table`; `.to_pandas()` is an alias for `.copy()`.
**B)** Returns a `pandas.DataFrame`; for object/string columns the conversion must copy and encode data into Python `str` objects.
**C)** Returns a `pandas.DataFrame`; the conversion is always zero-copy because Arrow and NumPy share a common memory standard.
**D)** Raises `AttributeError` because `pyarrow.Table` has no `.to_pandas()` method.

**Answer: B**

- A) Incorrect — `.to_pandas()` returns a `pandas.DataFrame`, not a `pyarrow.Table`; it is not an alias for `.copy()`.
- B) Correct — The result is a `pandas.DataFrame`. For numeric (NumPy-compatible) columns, Arrow may share the underlying buffer; for string columns, Arrow must allocate Python `str` objects from its contiguous UTF-8 buffer, which requires a copy and is why the DataFrame is larger than the Arrow table.
- C) Incorrect — Zero-copy is possible for numeric columns, but string columns always require copying and allocation of Python objects.
- D) Incorrect — `pyarrow.Table` does have a `.to_pandas()` method; it is a core part of the PyArrow API.

---

## Q4 — ConvertOptions for Type Override

```python
import pyarrow as pa
import pyarrow.csv as pa_csv

convert_options = pa_csv.ConvertOptions(
    column_types={
        'value': pa.float32(),
        'parameterId': pa.dictionary(pa.int32(), pa.string()),
    }
)
table = pa_csv.read_csv('2023_01.csv', convert_options=convert_options)
print(table.schema)
```

Which of the following will appear in the printed schema for `value` and `parameterId`?

**A)** `value: double, parameterId: string`
**B)** `value: float, parameterId: dictionary<values=string, indices=int32, ordered=0>`
**C)** `value: float32, parameterId: category`
**D)** `value: float, parameterId: string` — `ConvertOptions` is silently ignored for CSV files.

**Answer: B**

- A) Incorrect — Without `ConvertOptions`, `value` would default to `double` (float64). The code explicitly overrides it to `float32` (displayed as `float` in Arrow schema notation).
- B) Correct — Arrow schema prints `float` for `pa.float32()` and the full dictionary type descriptor for `pa.dictionary(pa.int32(), pa.string())`.
- C) Incorrect — `category` is a Pandas dtype concept, not an Arrow schema type. Arrow uses `dictionary<...>` notation.
- D) Incorrect — `ConvertOptions` is the correct mechanism for type overrides and is fully supported; it is not silently ignored.

---

## Q5 — Memory Size Comparison: Arrow vs Pandas

```python
import pandas as pd
import pyarrow.csv as pa_csv

df_pandas = pd.read_csv('2023_01.csv')
table_arrow = pa_csv.read_csv('2023_01.csv')

size_pandas = df_pandas.memory_usage(deep=True).sum()
size_arrow = table_arrow.nbytes

print(size_pandas > size_arrow)
```

What does this print, and why?

**A)** `False` — Pandas is more memory-efficient because it uses NumPy arrays.
**B)** `True` — Pandas stores string columns as Python `str` objects with per-object overhead; Arrow stores strings in a contiguous buffer with an offsets array.
**C)** `False` — Arrow must store both the data buffers and validity bitmaps, making it larger.
**D)** The code raises `AttributeError` because `pyarrow.Table` has no `.nbytes` property.

**Answer: B**

- A) Incorrect — Pandas is less memory-efficient for datasets with string columns due to Python object overhead. The DMI dataset's Arrow size (~507 MB) is roughly 4x smaller than the Pandas size (~2045 MB).
- B) Correct — The `created`, `observed`, and `parameterId` columns are string types. In Pandas each string becomes a Python `str` object (~49 bytes overhead). In Arrow all strings are packed into a single UTF-8 buffer with a compact integer offsets array, resulting in far lower total memory usage.
- C) Incorrect — The validity bitmap is a single bit per row (about 1 MB for 8M rows), negligible compared to the savings from compact string storage.
- D) Incorrect — `pyarrow.Table` does have an `.nbytes` property that returns the total size of all column buffers.

---

## Q6 — Parquet Column Projection

```python
import pandas as pd

df = pd.read_parquet('2023_01.parquet', columns=['parameterId', 'value'])
print(df.columns.tolist())
```

What does `df.columns.tolist()` print, and what is the key benefit of using `columns=` here?

**A)** `['coordsx', 'coordsy', 'created', 'observed', 'parameterId', 'stationId', 'value']` — `columns=` is a hint only; all columns are still read.
**B)** `['parameterId', 'value']` — only these two columns are read from disk; the other five column byte ranges in the Parquet file are skipped entirely.
**C)** `['parameterId', 'value']` — but all columns are read into memory first and then the others are dropped before returning.
**D)** Raises `ValueError` because `columns=` is not a valid parameter for `read_parquet`.

**Answer: B**

- A) Incorrect — `columns=` is not merely a hint; the Parquet reader uses it to skip non-selected column byte ranges at the file level.
- B) Correct — Parquet's columnar layout stores each column's data at a separate byte offset. The `columns=` parameter causes the reader to seek to and read only those specific byte ranges, leaving the rest of the file untouched. This is column projection.
- C) Incorrect — A correct Parquet reader does not read and then drop columns; it skips their byte ranges before any I/O or deserialisation occurs.
- D) Incorrect — `columns=` is a valid and standard parameter for both `pandas.read_parquet()` and `pyarrow.parquet.read_table()`.

---

## Q7 — ChunkedArray vs Array

```python
import pyarrow.csv as pa_csv

table = pa_csv.read_csv('2023_01.csv')
col = table['value']

print(type(col))
print(len(col.chunks))
```

Which of the following is a correct interpretation of this code?

**A)** `col` is a `pyarrow.Array` and `col.chunks` raises `AttributeError` because plain arrays have no chunks.
**B)** `col` is a `pyarrow.ChunkedArray`; `col.chunks` returns a list of the underlying `Array` objects. The number of chunks depends on how many batches the reader used internally.
**C)** `col` is a `pyarrow.ChunkedArray`; `col.chunks` is always a list of exactly one element for CSV files.
**D)** `col` is a `pyarrow.ChunkedArray`; `col.chunks` always equals the number of CPU cores used during parsing.

**Answer: B**

- A) Incorrect — `table['value']` returns a `ChunkedArray`, not a plain `Array`. `ChunkedArray` does have a `.chunks` attribute.
- B) Correct — `ChunkedArray.chunks` is a Python list of `pyarrow.Array` objects. The number of chunks is determined by how many record batches the CSV reader produced internally (related to batch size and file size), not a fixed number.
- C) Incorrect — For a large file the CSV reader may produce multiple batches, resulting in multiple chunks. One chunk is not guaranteed.
- D) Incorrect — The number of chunks is not tied to CPU core count; it reflects the reader's internal batching strategy.

---

## Q8 — from_pandas() and Zero-Copy

```python
import pandas as pd
import pyarrow as pa

df = pd.DataFrame({'x': [1.0, 2.0, 3.0], 'label': ['a', 'b', 'c']})
table = pa.Table.from_pandas(df)

# Modify the original numpy array
import numpy as np
df['x'].values[0] = 99.0

print(table['x'][0].as_py())
```

What does `table['x'][0].as_py()` print?

**A)** `99.0` — Arrow shares the NumPy buffer by default for aligned numeric columns, so the modification is reflected in the table.
**B)** `1.0` — Arrow always copies numeric data from Pandas, so the table is unaffected.
**C)** Raises `ArrowInvalid` because Arrow tables are immutable and `from_pandas` rejects mutable input.
**D)** The result is implementation-defined and depends on the PyArrow version.

**Answer: A**

- A) Correct — `pa.Table.from_pandas()` performs zero-copy sharing of the underlying NumPy buffer for aligned, compatible numeric columns by default. The Arrow array points into the same memory as the Pandas column. After `df['x'].values[0] = 99.0` mutates the buffer, `table['x'][0].as_py()` reflects that mutation and prints `99.0`.
- B) Incorrect — PyArrow does NOT always copy numeric data. For C-contiguous, aligned NumPy arrays, `from_pandas()` shares the buffer (zero-copy) unless `copy=True` is explicitly passed. Claiming it "always copies" is wrong.
- C) Incorrect — `from_pandas` does not raise an error for mutable input; Arrow's immutability is a convention, not enforced at the memory level for shared buffers.
- D) Incorrect — The behaviour is consistent for modern PyArrow (≥ 0.17): zero-copy is the default for compatible numeric arrays. It is not undefined.

---

## Q9 — Writing and Reading Parquet

```python
import pandas as pd
import pyarrow.parquet as pq

df = pd.read_csv('2023_01.csv')
df.to_parquet('2023_01.parquet')

table = pq.read_table('2023_01.parquet')
print(type(table))
```

What does this print?

**A)** `<class 'pandas.core.frame.DataFrame'>`
**B)** `<class 'pyarrow.lib.Table'>`
**C)** `<class 'pyarrow.lib.ParquetFile'>`
**D)** `<class 'pyarrow.lib.RecordBatch'>`

**Answer: B**

- A) Incorrect — `pyarrow.parquet.read_table()` returns a `pyarrow.Table`, not a Pandas DataFrame. Use `pd.read_parquet()` or `.to_pandas()` if a DataFrame is needed.
- B) Correct — `pyarrow.parquet.read_table()` returns a `pyarrow.Table`, just as `pyarrow.csv.read_csv()` does. Parquet is simply a more efficient on-disk format; both produce the same in-memory Arrow representation.
- C) Incorrect — `pyarrow.parquet.ParquetFile` is a handle for inspecting Parquet file metadata; `read_table()` returns a fully loaded `Table`.
- D) Incorrect — `read_table()` assembles all row groups into a complete `Table`, not individual `RecordBatch` objects.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

---

## Q10 — Dictionary Encoding at Load Time

```python
import pyarrow as pa
import pyarrow.csv as pa_csv

convert_options = pa_csv.ConvertOptions(
    column_types={
        'parameterId': pa.dictionary(pa.int8(), pa.string()),
        'coordsx':     pa.dictionary(pa.int16(), pa.float64()),
    }
)
table = pa_csv.read_csv('2023_01.csv', convert_options=convert_options)
```

The original CSV has `parameterId` with 47 unique values and `coordsx` with 224 unique values. Which index type is too small and would likely cause an error or data loss?

**A)** `pa.int8()` for `parameterId` — `int8` holds -128 to 127, so 47 unique values fit fine.
**B)** `pa.int8()` for `parameterId` — `int8` is unsigned and cannot hold negative indices.
**C)** `pa.int16()` for `coordsx` — `int16` holds up to 32767, but 224 values exceeds this.
**D)** Neither index type is too small; both fit their respective unique value counts.

**Answer: D**

- A) Correct reasoning but wrong answer label — 47 values fit in `int8` (range 0–127 for unsigned, or treated as 0-indexed). However, the correct overall answer is D because both types are sufficient.
- B) Incorrect — Arrow's dictionary indices are treated as non-negative integer codes; `int8` stores values 0–127 (more than enough for 47 unique values).
- C) Incorrect — `int16` holds up to 32767; 224 unique values fit easily.
- D) Correct — `int8` supports up to 128 unique values (indices 0–127), more than enough for 47. `int16` supports up to 32767, more than enough for 224. Both choices are appropriate.

---

## Q11 — Arrow compute Function vs Pandas

```python
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv

table = pa_csv.read_csv('2023_01.csv')
mask = pc.equal(table['parameterId'], 'precip_past10min')
filtered = table.filter(mask)
total = pc.sum(filtered['value'])
print(total.as_py())
```

How does this compare to the equivalent Pandas vectorised approach `df[df['parameterId'] == 'precip_past10min']['value'].sum()`?

**A)** The PyArrow version is slower because `pyarrow.compute` functions are interpreted Python code.
**B)** The PyArrow version avoids creating a Pandas DataFrame entirely, operating on Arrow's contiguous buffers, and is generally faster while also using less memory.
**C)** Both approaches are identical in speed because both use C-compiled kernels underneath.
**D)** The PyArrow version raises `TypeError` because `pc.equal` cannot compare a `ChunkedArray` to a Python string.

**Answer: B**

- A) Incorrect — `pyarrow.compute` functions are compiled C++ kernels, not Python loops; they are not interpreted.
- B) Correct — Operating on `pyarrow.ChunkedArray` buffers avoids the Python object overhead that Pandas incurs for object-type columns like `parameterId`. The Arrow compute path also skips the `.to_pandas()` conversion step entirely.
- C) Incorrect — Pandas uses NumPy/Cython kernels which are fast, but Arrow compute avoids the Python string object overhead for `parameterId`, giving an additional advantage when the column is stored as a plain string type.
- D) Incorrect — `pc.equal(chunked_array, scalar)` is valid; PyArrow automatically broadcasts the scalar to compare against each element.

---

## Q12 — Schema Inspection

```python
import pyarrow.csv as pa_csv

table = pa_csv.read_csv('2023_01.csv')
print(table.schema)
```

What does `table.schema` represent, and which of the following is true?

**A)** `table.schema` is a Python `dict` mapping column names to Python type strings like `'int64'` or `'object'`.
**B)** `table.schema` is a `pyarrow.Schema` object listing each column's name and Arrow data type; it is inferred from the first batch of data read from the CSV.
**C)** `table.schema` raises `AttributeError` because schema information is only available after calling `.to_pandas()`.
**D)** `table.schema` is always `None` for CSV-loaded tables because CSV files contain no embedded schema.

**Answer: B**

- A) Incorrect — `table.schema` is a `pyarrow.Schema` object, not a plain Python `dict`. It uses Arrow type objects like `pa.int64()` and `pa.string()`, not Python type strings.
- B) Correct — `pyarrow.Schema` is Arrow's type descriptor; it holds a list of `pyarrow.Field` objects (name + type). For CSV files, Arrow infers the schema from the first batch and applies it consistently to all subsequent batches.
- C) Incorrect — Schema information is a core attribute of every `pyarrow.Table`; it is always present without needing a Pandas conversion.
- D) Incorrect — While CSV files themselves have no embedded schema, Arrow infers and stores a schema for the resulting `Table`. It is never `None`.

---

## Q13 — Parquet Column Projection with pyarrow.parquet

```python
import pyarrow.parquet as pq

parquet_file = pq.ParquetFile('2023_01.parquet')
table = parquet_file.read(columns=['stationId', 'value'])
```

Compared to `pq.read_table('2023_01.parquet')`, what is the key difference in I/O behaviour?

**A)** `parquet_file.read(columns=...)` loads all columns into memory first and then filters by name; `read_table` only reads the specified columns.
**B)** Both behave identically; `columns=` is a cosmetic parameter with no I/O effect.
**C)** `parquet_file.read(columns=['stationId', 'value'])` reads only the byte ranges for those two columns from the Parquet file; the other five columns are never read from disk.
**D)** `ParquetFile.read()` is deprecated; only `pq.read_table()` supports column projection.

**Answer: C**

- A) Incorrect — This reverses the truth. It is `parquet_file.read(columns=...)` that is more I/O-efficient; it does not load all columns first.
- B) Incorrect — `columns=` is not cosmetic; it directly controls which byte ranges are read from the file.
- C) Correct — The Parquet format stores each column's data at a separate byte offset within each row group. Specifying `columns=` causes the reader to seek to and deserialise only those offsets, skipping all others. This is the key performance advantage of columnar file formats for projection-heavy workloads.
- D) Incorrect — `ParquetFile.read()` is not deprecated; both APIs are current and both support column projection.

---

## Q14 — Arrow Slice Semantics

```python
import pyarrow.csv as pa_csv

table = pa_csv.read_csv('2023_01.csv')
col = table['value']
# Combine chunks so we can inspect the underlying buffer
col_arr    = col.combine_chunks()        # pyarrow.Array
sliced_arr = col_arr[100:200]            # sliced pyarrow.Array

original_buf = col_arr.buffers()[1]
sliced_buf   = sliced_arr.buffers()[1]

print(original_buf == sliced_buf)
```

What does this code demonstrate about Arrow slice semantics?

**A)** `True` — slicing returns a view over the same buffer; no data is copied.
**B)** `False` — slicing always allocates a new buffer containing only the sliced elements.
**C)** Raises `AttributeError` — `Array` has no `.buffers()` method.
**D)** `True` — but only because `col_arr` and `sliced_arr` point to the same Python object.

**Answer: A**

- A) Correct — Arrow slicing is zero-copy. `col_arr[100:200]` creates a new `Array` object that records a different `offset` and `length` into the original buffer. The underlying data buffer is shared, not copied. Both arrays point to the same memory, so `original_buf == sliced_buf` evaluates to `True`.
- B) Incorrect — Arrow explicitly avoids copying on slice for performance reasons; this is one of Arrow's core design goals.
- C) Incorrect — `pyarrow.Array` does have a `.buffers()` method that returns the list of underlying memory buffers. (Note: `ChunkedArray` does not have `.buffers()` directly; that is why `combine_chunks()` is called first to obtain a single `Array`.)
- D) Incorrect — `col_arr` and `sliced_arr` are distinct Python objects with different offsets/lengths, but they share the same underlying data buffer. They are not the same object.

---

## Q15 — ConvertOptions column_types vs dtype

```python
import pyarrow.csv as pa_csv
import pyarrow as pa

# Attempt 1
table1 = pa_csv.read_csv('2023_01.csv', dtype={'value': 'float32'})

# Attempt 2
table2 = pa_csv.read_csv(
    '2023_01.csv',
    convert_options=pa_csv.ConvertOptions(column_types={'value': pa.float32()})
)
```

Which attempt succeeds, and which raises a `TypeError`?

**A)** Attempt 1 succeeds; Attempt 2 raises `TypeError` because `ConvertOptions` does not accept Arrow types.
**B)** Both succeed; `dtype=` and `ConvertOptions` are equivalent aliases in PyArrow.
**C)** Attempt 1 raises `TypeError` because `pyarrow.csv.read_csv` has no `dtype=` parameter; Attempt 2 succeeds.
**D)** Both raise `TypeError`; type overrides are only supported via `schema=` in `ReadOptions`.

**Answer: C**

- A) Incorrect — `dtype=` is a Pandas convention; `pyarrow.csv.read_csv()` does not accept it, so Attempt 1 raises `TypeError`.
- B) Incorrect — They are not equivalent aliases; `dtype=` is not a valid keyword argument for `pyarrow.csv.read_csv()`.
- C) Correct — PyArrow's CSV reader uses `convert_options=csv.ConvertOptions(column_types={...})` for type overrides. Passing `dtype=` as a keyword argument raises `TypeError: read_csv() got an unexpected keyword argument 'dtype'`.
- D) Incorrect — `ReadOptions` controls lower-level behaviour like block size and column names; type coercion belongs to `ConvertOptions`. Attempt 2 is the correct pattern.

---

## Q16 — to_pandas() String Memory Inflation

```python
import pyarrow.csv as pa_csv

table = pa_csv.read_csv('2023_01.csv')
df = table.to_pandas()

arrow_bytes = table.nbytes
pandas_bytes = df.memory_usage(deep=True).sum()

print(f"Arrow: {arrow_bytes / 1e6:.0f} MB")
print(f"Pandas: {pandas_bytes / 1e6:.0f} MB")
```

Based on the Week 7 DMI dataset results, which output is most accurate?

**A)** `Arrow: 2045 MB` / `Pandas: 507 MB`
**B)** `Arrow: 507 MB` / `Pandas: 919 MB`
**C)** `Arrow: 507 MB` / `Pandas: 507 MB`
**D)** `Arrow: 86 MB` / `Pandas: 507 MB`

**Answer: B**

- A) Incorrect — The figures are swapped; Arrow is more compact than Pandas for this dataset.
- B) Correct — Arrow stores the table in ~507 MB because strings are in contiguous buffers. Converting to Pandas inflates the size to ~919 MB as Python `str` objects are allocated for string columns (`created`, `observed`, `parameterId`). Note: the full Pandas `read_csv()` load is ~2045 MB because all string columns default to `object` without any inference; PyArrow infers efficient types during reading.
- C) Incorrect — Conversion to Pandas inflates memory, especially for string columns; the sizes are not equal.
- D) Incorrect — 86 MB is the on-disk Parquet file size, not the in-memory Arrow table size.

---

## Q17 — Parquet Row Group Count

```python
import pyarrow.parquet as pq

pf = pq.ParquetFile('2023_01.parquet')
print(pf.metadata.num_row_groups)
print(pf.metadata.num_rows)
```

What does `num_row_groups` tell you, and why does it matter for performance?

**A)** The number of columns in the Parquet file — more columns means more row groups.
**B)** The number of horizontal partitions in the file — each row group stores its columns independently, and row group statistics enable the reader to skip groups that cannot satisfy a filter predicate.
**C)** The number of compression blocks applied to the file — each row group uses a different codec.
**D)** The number of threads used when writing the Parquet file — more row groups means more parallelism.

**Answer: B**

- A) Incorrect — `num_row_groups` counts horizontal (row) partitions, not columns. Column count is `pf.metadata.num_columns`.
- B) Correct — A row group is a horizontal slice of the table. Each row group stores per-column min/max statistics. When a filter predicate is applied, the Parquet reader checks these statistics and skips entire row groups that provably contain no matching rows, without deserialising them.
- C) Incorrect — Compression codecs are configured per column chunk within a row group; row groups and compression codecs are orthogonal concepts.
- D) Incorrect — Row groups are a storage layout concept; the number of row groups is set by the writer's row group size parameter, not the thread count.

---

## Q18 — Total Pipeline: Fastest Load for Single Aggregation

```python
import pyarrow.csv as pa_csv
import pyarrow.compute as pc

# Option A: PyArrow end-to-end
table = pa_csv.read_csv('2023_01.csv')
mask = pc.equal(table['parameterId'], 'precip_past10min')
result_a = pc.sum(table.filter(mask)['value']).as_py()

# Option B: Arrow load, Pandas compute
import pandas as pd
table2 = pa_csv.read_csv('2023_01.csv')
df = table2.to_pandas()
result_b = df[df['parameterId'] == 'precip_past10min']['value'].sum()

# Option C: Pandas end-to-end
df2 = pd.read_csv('2023_01.csv')
result_c = df2[df2['parameterId'] == 'precip_past10min']['value'].sum()
```

Assuming all three produce the same numerical result, which ordering of total wall-clock time (fastest to slowest) is correct?

**A)** Option C < Option B < Option A
**B)** Option A < Option B < Option C
**C)** Option B < Option A < Option C
**D)** All three options take the same time because the computation step dominates.

**Answer: B**

- A) Incorrect — Option C uses `pandas.read_csv()` (~11 s load) which is the slowest loading path; it cannot be fastest overall.
- B) Correct — Option A avoids `pandas.read_csv()` (~11 s) and avoids `.to_pandas()` conversion (~700 ms), operating purely on Arrow buffers. Option B pays the conversion cost (~700 ms) on top of the Arrow load (~3 s) but is still faster than Option C (~11 s load). So: A (fastest) < B < C (slowest).
- C) Incorrect — Option B requires `.to_pandas()` conversion in addition to Arrow loading, so it is slower than Option A which skips the conversion entirely.
- D) Incorrect — The load time dominates: ~3 s (Arrow) vs ~11 s (Pandas CSV). The computation step (~ms) is negligible compared to I/O differences.

---
