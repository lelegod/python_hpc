# PyArrow & Apache Arrow — MCQ Practice

> [← Topics](../README.md) · [Questions](questions.md) · [Code Questions](code_questions.md)

## Contents

- [Q1 — Columnar vs Row-Based Storage](#q1--columnar-vs-row-based-storage)
- [Q2 — PyArrow CSV Read Return Type](#q2--pyarrow-csv-read-return-type)
- [Q3 — Arrow vs Pandas Load Speed](#q3--arrow-vs-pandas-load-speed)
- [Q4 — Zero-Copy Semantics](#q4--zero-copy-semantics)
- [Q5 — Arrow String Memory Layout](#q5--arrow-string-memory-layout)
- [Q6 — Arrow Table to Pandas Conversion Cost](#q6--arrow-table-to-pandas-conversion-cost)
- [Q7 — Chunked Arrays](#q7--chunked-arrays)
- [Q8 — Parquet File Advantages](#q8--parquet-file-advantages)
- [Q9 — Parquet Row Groups](#q9--parquet-row-groups)
- [Q10 — Column Projection in Parquet](#q10--column-projection-in-parquet)
- [Q11 — Arrow Dictionary Encoding](#q11--arrow-dictionary-encoding)
- [Q12 — Schema Inference](#q12--schema-inference)
- [Q13 — Memory Size: Arrow vs Pandas (Same Data)](#q13--memory-size-arrow-vs-pandas-same-data)
- [Q14 — Arrow IPC Format](#q14--arrow-ipc-format)
- [Q15 — ConvertOptions at Load Time](#q15--convertoptions-at-load-time)
- [Q16 — Why Columnar Beats Row-Based for Analytics](#q16--why-columnar-beats-row-based-for-analytics)
- [Q17 — Parquet Read Speed vs CSV](#q17--parquet-read-speed-vs-csv)
- [Q18 — from_pandas() Behaviour](#q18--from_pandas-behaviour)
- [Q19 — Arrow Column Access Pattern](#q19--arrow-column-access-pattern)
- [Q20 — Parquet vs CSV File Size](#q20--parquet-vs-csv-file-size)
- [Q21 — Arrow and Cache Efficiency](#q21--arrow-and-cache-efficiency)
- [Q22 — When Arrow Conversion Cost is Worth It](#q22--when-arrow-conversion-cost-is-worth-it)

---

> Topics: Columnar storage, zero-copy, PyArrow tables, Parquet, Arrow↔Pandas conversion, memory layout.
> Exam frequency: **Week 7 topic**.

**Navigate:** &nbsp;[▶ Set 1 — Original Questions](#q1--columnar-vs-row-based-storage)&nbsp;&nbsp;|&nbsp;&nbsp;[▶ Set 2 — New Practice](#set-2--generated-practice-questions-exam-day-focus)

---

## Q1 — Columnar vs Row-Based Storage

> **Week reference:** Week 7

**Mental Model:** In a row-based format every row's fields are stored together; in a columnar format all values of each column are stored together. Analytics that touch only a few columns benefit enormously from columnar storage because irrelevant data is never read from disk.

Which of the following best describes the key difference between row-based and columnar storage for analytical workloads?

- A) Row-based storage uses less disk space because metadata overhead is lower per record.
- B) Columnar storage is faster for row-level inserts because each row is stored contiguously.
- C) Columnar storage is more efficient for analytics that only read a subset of columns because only the relevant columns are loaded from disk.
- D) Row-based storage allows better compression because related data types are grouped together.

**Answer: C**

- A) Incorrect — Row-based formats generally store more metadata per record and achieve lower compression ratios than columnar formats.
- B) Incorrect — Row-based storage is better for inserts since a full row is written in one seek; columnar requires writing to multiple column buffers.
- C) Correct — Because each column is stored as a contiguous block, a query that reads only two of seven columns only reads those two blocks from disk, skipping the rest entirely.
- D) Incorrect — Columnar storage achieves better compression because all values within a column share the same data type and often have correlated values or low cardinality.

---

## Q2 — PyArrow CSV Read Return Type

> **Week reference:** Week 7

**Mental Model:** `pyarrow.csv.read_csv()` is a distinct function from `pandas.read_csv()` and returns an Arrow-native data structure, not a Pandas DataFrame.

What does `pyarrow.csv.read_csv(path)` return?

- A) A `pandas.DataFrame`
- B) A `pyarrow.Table`
- C) A `pyarrow.RecordBatch`
- D) A Python `list` of dictionaries

**Answer: B**

- A) Incorrect — `pyarrow.csv.read_csv()` returns an Arrow-native `pyarrow.Table`, not a Pandas DataFrame. Call `.to_pandas()` explicitly to get a DataFrame.
- B) Correct — The function returns a `pyarrow.Table`, which holds all columns as Arrow arrays stored contiguously in Arrow's binary format.
- C) Incorrect — A `RecordBatch` is a single chunk of rows. `read_csv()` returns a complete `Table`, which may internally consist of one or more `RecordBatch` objects.
- D) Incorrect — That would be the result of `csv.DictReader` from Python's standard library, not PyArrow.

---

## Q3 — Arrow vs Pandas Load Speed

> **Week reference:** Week 7

**Mental Model:** PyArrow's CSV reader is heavily parallelised and uses a more efficient parsing pipeline than Pandas. On the DMI 2023 January weather dataset (~8M rows), Arrow was roughly 3.7x faster than Pandas.

In the Week 7 exercises, reading the DMI January 2023 CSV with `pyarrow.csv.read_csv()` took approximately 3 seconds, while `pandas.read_csv()` took approximately 11 seconds. Which explanation best accounts for this speedup?

- A) PyArrow skips parsing most rows because it samples the file to build a schema.
- B) PyArrow's CSV reader uses parallelised, zero-copy parsing and avoids creating Python objects for each value.
- C) PyArrow reads only the columns needed, whereas Pandas reads all columns by default.
- D) Pandas performs schema validation on every row which slows it down significantly.

**Answer: B**

- A) Incorrect — PyArrow reads every row; it does not sample to infer schema (it infers from the first batch but reads all data).
- B) Correct — PyArrow's CSV reader is multi-threaded, parses data into Arrow's binary columnar format directly, and avoids the Python object allocation overhead that Pandas incurs for string and object columns.
- C) Incorrect — Both readers load all columns by default; column selection is an opt-in option, not the default behaviour that causes the speedup.
- D) Incorrect — Pandas does not perform row-by-row schema validation; the slowness comes from single-threaded parsing and Python object creation for non-numeric types.

---

## Q4 — Zero-Copy Semantics

> **Week reference:** Week 7

**Mental Model:** Zero-copy means sharing memory between processes or libraries without duplicating the underlying data buffer. Arrow's standard memory layout makes zero-copy slicing and sharing possible across language boundaries.

Which of the following operations on a `pyarrow.Table` is zero-copy?

- A) Converting the entire table to a Pandas DataFrame with `.to_pandas()`
- B) Slicing a column to get a sub-array with `column[10:100]`
- C) Casting a column from `int64` to `float32`
- D) Adding a new computed column derived from two existing columns

**Answer: B**

- A) Incorrect — `.to_pandas()` generally copies data to create Python/NumPy objects, though for numeric types Arrow may be able to share the buffer. For object columns (strings) a copy is unavoidable.
- B) Correct — Slicing an Arrow array creates a new array object that simply records a different offset and length into the same underlying memory buffer — no data is copied.
- C) Incorrect — A type cast produces a new buffer with values reinterpreted or converted; this is never zero-copy.
- D) Incorrect — Computing a new column reads existing buffers and writes results into a freshly allocated buffer.

---

## Q5 — Arrow String Memory Layout

> **Week reference:** Week 7

**Mental Model:** Arrow stores string columns as two contiguous buffers: one large buffer holding all character data back-to-back, and an offsets buffer recording where each string starts and ends. Pandas stores each string as a separate Python `str` object on the heap.

Why does a PyArrow table with a string column occupy significantly less memory than the equivalent Pandas DataFrame?

- A) PyArrow uses LZ4 compression on all string columns automatically.
- B) Arrow stores all string data in one contiguous buffer with an offsets array, whereas Pandas creates a Python `str` object per value with its own heap allocation.
- C) PyArrow drops duplicate strings automatically, keeping only one copy in memory.
- D) Pandas stores strings as Unicode-4 (4 bytes per character) while Arrow uses UTF-8 (1 byte per ASCII character).

**Answer: B**

- A) Incorrect — PyArrow does not apply transparent compression to in-memory tables; compression is a Parquet/IPC file feature.
- B) Correct — Arrow's string (or large_string) type packs all characters into a single buffer and uses an integer offsets array to find individual strings. Each Python `str` object in Pandas carries ~50 bytes of interpreter overhead on top of the character data.
- C) Incorrect — Arrow does not deduplicate strings in a plain `string` column; that is what dictionary encoding does (a separate feature).
- D) Incorrect — Both Arrow and Pandas use UTF-8 encoding for string data; the memory difference is due to object representation, not character encoding width.

---

## Q6 — Arrow Table to Pandas Conversion Cost

> **Week reference:** Week 7

**Mental Model:** Even though PyArrow loads CSV much faster than Pandas, the `.to_pandas()` conversion adds overhead (roughly 700 ms on the DMI dataset). The combined time is still faster than loading with Pandas directly.

When comparing the three approaches for the DMI dataset, which ordering of total wall-clock time (fastest to slowest) is correct?

- A) `pandas.read_csv()` < `pyarrow.csv.read_csv()` < `pyarrow.csv.read_csv()` + `.to_pandas()`
- B) `pyarrow.csv.read_csv()` < `pyarrow.csv.read_csv()` + `.to_pandas()` < `pandas.read_csv()`
- C) `pyarrow.csv.read_csv()` + `.to_pandas()` < `pandas.read_csv()` < `pyarrow.csv.read_csv()`
- D) All three approaches take approximately the same time.

**Answer: B**

- A) Incorrect — `pandas.read_csv()` at ~11 s is the slowest of the three; it cannot beat PyArrow-only loading.
- B) Correct — Loading with PyArrow alone (~3 s) is fastest; adding `.to_pandas()` (~3.7 s total) is still faster than `pandas.read_csv()` (~11 s).
- C) Incorrect — `pyarrow.csv.read_csv()` alone is faster than adding the conversion step; it cannot be slower.
- D) Incorrect — There is a roughly 3–4x difference between the PyArrow-only path and the Pandas path.

---

## Q7 — Chunked Arrays

> **Week reference:** Week 7

**Mental Model:** A `pyarrow.ChunkedArray` is a logical array composed of one or more contiguous `Array` chunks. This allows Arrow to represent datasets that are too large to fit in a single contiguous allocation, or that were built by concatenating independently-read batches.

What is a `pyarrow.ChunkedArray`?

- A) A compressed array that stores data in fixed-size blocks for better cache performance.
- B) A logical sequence of values backed by one or more contiguous Arrow `Array` chunks, allowing large or incrementally-built datasets without requiring a single contiguous allocation.
- C) An array stored on GPU memory, partitioned into CUDA thread blocks.
- D) A sparse array that only stores non-zero values to save memory.

**Answer: B**

- A) Incorrect — A `ChunkedArray` is not compressed; compression is a file-format concept. The "chunks" refer to separate contiguous memory allocations, not cache blocks.
- B) Correct — A `ChunkedArray` wraps a list of regular `Array` objects. When `pyarrow.csv.read_csv()` reads a very large file in multiple passes, it may produce columns backed by multiple chunks.
- C) Incorrect — Arrow is a CPU-side, in-memory format; `ChunkedArray` has no GPU semantics.
- D) Incorrect — `ChunkedArray` is a general-purpose array; sparse storage is a separate concept not related to chunking.

---

## Q8 — Parquet File Advantages

> **Week reference:** Week 7

**Mental Model:** Parquet is a binary columnar file format that combines Arrow-like columnar layout on disk with built-in compression and rich type metadata. It is far more compact and faster to read than CSV.

Which of the following is NOT an advantage of the Parquet format over CSV?

- A) Parquet files are significantly smaller due to binary encoding and columnar compression.
- B) Parquet files can be read faster because text parsing is avoided.
- C) Parquet files are human-readable and can be opened directly in a text editor.
- D) Parquet files store schema and type information, eliminating the need for type inference at read time.

**Answer: C**

- A) Incorrect as a "not an advantage" — Parquet files are indeed much smaller; the DMI dataset dropped from a large CSV to ~86 MB as Parquet.
- B) Incorrect as a "not an advantage" — Parquet reads are much faster (e.g., ~0.4 s with PyArrow vs ~3 s for the equivalent CSV).
- C) Correct — Parquet is a binary format and cannot be read with a text editor; this is genuinely not an advantage (though it is an acceptable trade-off).
- D) Incorrect as a "not an advantage" — Parquet embeds a schema, so readers know column types without guessing, which is a clear advantage over CSV.

---

## Q9 — Parquet Row Groups

> **Week reference:** Week 7

**Mental Model:** Parquet divides a table into horizontal partitions called row groups. Each row group stores its columns independently with separate statistics (min/max). This allows readers to skip entire row groups that cannot satisfy a filter predicate without reading their data — a technique called predicate pushdown.

What is a Parquet row group and why does it matter for performance?

- A) A row group is a set of columns that are stored together; it allows partial column reads without reading other columns.
- B) A row group is a horizontal partition of the table; each row group stores its columns independently, enabling readers to skip row groups that cannot satisfy a filter predicate.
- C) A row group is a compression block; larger row groups achieve better compression ratios.
- D) A row group is a replica of the full table stored at a different offset in the file, enabling fault tolerance.

**Answer: B**

- A) Incorrect — That describes column projection (reading only specific columns), not row groups. Row groups are horizontal (row) partitions, not vertical (column) ones.
- B) Correct — Each row group is an independent horizontal slice of the table. Row group statistics (min/max per column) allow the Parquet reader to skip entire groups when the filter predicate cannot be satisfied — this is predicate pushdown.
- C) Incorrect — Compression is applied within each column chunk inside a row group; row group size affects compression, but the primary performance benefit is predicate pushdown, not compression ratio alone.
- D) Incorrect — Parquet is not a replicated format; row groups are not replicas.

---

## Q10 — Column Projection in Parquet

> **Week reference:** Week 7

**Mental Model:** Because Parquet is columnar, you can read only the specific columns you need. Columns not listed in the projection are never read from disk, which is impossible with CSV (where the entire row must be parsed to reach any field).

You have a 7-column Parquet file but only need 2 columns for your analysis. Which approach is most efficient?

- A) Read all 7 columns and then drop the 5 unwanted columns with `df.drop()`.
- B) Read the file with `columns=['col1', 'col2']` to load only the needed columns from disk.
- C) Read the file normally; Parquet automatically detects which columns are accessed and only loads those.
- D) Convert the Parquet file back to CSV first, then use `usecols=` in `pandas.read_csv()`.

**Answer: B**

- A) Incorrect — Reading all 7 columns then dropping 5 wastes I/O bandwidth and memory; the 5 unwanted columns are fully deserialized even though they are discarded immediately.
- B) Correct — The `columns=` parameter is passed to `pyarrow.parquet.read_table()` or `pandas.read_parquet()` and instructs the Parquet reader to skip the other columns' byte ranges entirely at the file level.
- C) Incorrect — Parquet readers do not automatically perform lazy column loading; you must explicitly specify the desired columns.
- D) Incorrect — Converting to CSV first throws away every advantage of Parquet (columnar layout, binary encoding, column skipping) and makes the problem significantly slower.

---

## Q11 — Arrow Dictionary Encoding

> **Week reference:** Week 7

**Mental Model:** Dictionary encoding in Arrow is equivalent to Pandas `category` dtype: a small integer index replaces each repeated string value, with a lookup table mapping integers to the actual strings. This dramatically reduces memory for low-cardinality string columns.

In Exercise 2.4, `parameterId` was loaded with `pa.dictionary(pa.int32(), pa.string())`. What does this achieve?

- A) It sorts the `parameterId` column alphabetically for faster lookups.
- B) It compresses the column using the LZ4 algorithm.
- C) It stores only the 47 unique parameter IDs as strings and represents each row's value as a small integer index, reducing memory for the column.
- D) It creates a Python `dict` mapping parameter names to their row indices for O(1) access.

**Answer: C**

- A) Incorrect — Dictionary encoding does not imply sorting; it maps unique values to integer codes regardless of order.
- B) Incorrect — LZ4 is a compression algorithm used in Blosc/IPC file formats; dictionary encoding is a logical representation, not a compression codec.
- C) Correct — With 47 unique values and ~8M rows, storing a 4-byte `int32` index per row instead of a full string per row saves substantial memory. The 47 strings are stored once in the dictionary.
- D) Incorrect — Arrow's dictionary type is a memory layout, not a Python `dict`; it does not expose a Python-level dictionary interface.

---

## Q12 — Schema Inference

> **Week reference:** Week 7

**Mental Model:** Both `pandas.read_csv()` and `pyarrow.csv.read_csv()` infer column types automatically from the data. Arrow's inference is performed on the first batch of rows and is generally faster; both can be overridden with explicit type specifications.

How can you force PyArrow to load a column with a specific type instead of the inferred type?

- A) Pass a `dtype=` dictionary directly to `pyarrow.csv.read_csv()`, mirroring the Pandas API.
- B) Use `csv.ConvertOptions(column_types={...})` to specify the desired Arrow types before reading.
- C) Read the file first with default types, then cast each column afterwards using `table.cast()`.
- D) Set `schema_inference=False` in `csv.ReadOptions` to disable inference and use default Arrow types.

**Answer: B**

- A) Incorrect — PyArrow's CSV reader does not accept a `dtype=` keyword; that is a Pandas convention. The PyArrow equivalent is `csv.ConvertOptions`.
- B) Correct — `csv.ConvertOptions(column_types={'col': pa.float32()})` is the correct mechanism. It is passed as the `convert_options=` argument to `csv.read_csv()`.
- C) Incorrect — While post-hoc casting is possible with `column.cast()`, it requires reading the data first with the inferred type, then allocating a new buffer for the cast result — less efficient than specifying the type at load time.
- D) Incorrect — There is no `schema_inference=False` option in PyArrow's `ReadOptions`; inference is always performed unless overridden via `ConvertOptions`.

---

## Set 2 — Generated Practice Questions (Exam-Day Focus)

---

## Q13 — Memory Size: Arrow vs Pandas (Same Data)

> **Week reference:** Week 7

**Mental Model:** Loading the same CSV file produces very different in-memory sizes: Arrow at ~507 MB vs Pandas at ~2045 MB for the DMI dataset. The difference comes almost entirely from how each library represents Python objects, especially strings.

The DMI January 2023 CSV loaded with `pandas.read_csv()` occupies ~2045 MB in memory. The same file loaded with `pyarrow.csv.read_csv()` occupies ~507 MB. What is the primary driver of this ~4x difference?

- A) Pandas stores all numeric columns in float64 by default, while Arrow uses float32.
- B) Pandas creates Python `str` objects with ~50 bytes of interpreter overhead per string value, while Arrow stores all strings in a single contiguous UTF-8 buffer with an offsets array.
- C) Arrow only loads every 4th row when reading CSVs to reduce memory usage.
- D) Pandas duplicates each column in memory for safety, while Arrow uses shared references.

**Answer: B**

- A) Incorrect — Both Pandas and Arrow default to float64 for numeric columns; that is not the source of the large memory gap.
- B) Correct — The `created`, `observed`, and `parameterId` columns are string/object in Pandas. Each Python `str` object has ~49 bytes of overhead plus the character data. Arrow stores all characters contiguously with a compact offsets array, avoiding per-string Python object overhead.
- C) Incorrect — Arrow reads every row; it does not subsample data at load time.
- D) Incorrect — Pandas does not duplicate column buffers; the bloat comes from Python object overhead, not duplication.

---

## Q14 — Arrow IPC Format

> **Week reference:** Week 7

**Mental Model:** Arrow IPC (Inter-Process Communication) is a binary streaming or file format that serialises Arrow tables into byte streams, preserving the exact in-memory layout. This enables zero-copy sharing between processes and language runtimes that support Arrow.

What is the Arrow IPC format primarily used for?

- A) Transmitting Arrow tables between processes or systems with minimal serialisation overhead, preserving the columnar memory layout.
- B) Storing Arrow tables on disk with strong compression for long-term archival.
- C) Providing a text-based exchange format compatible with JSON parsers.
- D) Encoding Arrow tables as Parquet files for compatibility with Spark and Hive.

**Answer: A**

- A) Correct — Arrow IPC (the "feather" file or streaming format) serialises Arrow buffers directly. Because the on-wire format matches the in-memory layout, deserialisation on the receiving side can map the bytes directly without copying or transformation.
- B) Incorrect — Arrow IPC can write to disk but does not apply heavy compression like Parquet's row-group-level codecs. Parquet is the preferred format for long-term, compressed archival.
- C) Incorrect — Arrow IPC is a binary format, not text-based; it is not compatible with JSON parsers.
- D) Incorrect — Arrow IPC and Parquet are distinct formats. IPC does not produce Parquet files.

---

## Q15 — ConvertOptions at Load Time

> **Week reference:** Week 7

**Mental Model:** Specifying types at load time via `csv.ConvertOptions` is more efficient than loading with default types and casting afterwards, because Arrow writes directly into the target-type buffer without an intermediate allocation.

Which of the following correctly demonstrates loading a CSV with PyArrow and forcing the `value` column to `float32`?

- A) `pyarrow.csv.read_csv(path, dtype={'value': 'float32'})`
- B) `pyarrow.csv.read_csv(path, convert_options=csv.ConvertOptions(column_types={'value': pa.float32()}))`
- C) `pyarrow.csv.read_csv(path, schema=pa.schema([('value', pa.float32())]))`
- D) `pyarrow.csv.read_csv(path).cast({'value': pa.float32()})`

**Answer: B**

- A) Incorrect — PyArrow's `read_csv` does not accept a `dtype=` keyword argument; that pattern is from Pandas.
- B) Correct — `csv.ConvertOptions(column_types={...})` is the PyArrow mechanism for specifying per-column types at read time.
- C) Incorrect — While `csv.ReadOptions` can receive a `column_names` argument, passing a schema directly is not the correct API for type coercion at load time.
- D) Incorrect — `pyarrow.Table` has no `.cast(dict)` method; casting is done per-column via `table.column('col').cast(pa.float32())`, and this would still require reading the data first.

---

## Q16 — Why Columnar Beats Row-Based for Analytics

> **Week reference:** Week 7

**Mental Model:** An analytical query like `SUM(value WHERE parameterId = 'precip_past10min')` needs to scan the `parameterId` and `value` columns but touches zero data from the other five columns. Columnar storage makes this possible at the hardware level.

A query needs to sum the `value` column for all rows where `parameterId == 'precip_past10min'`. The dataset has 7 columns and 8 million rows. Why is columnar storage particularly advantageous here?

- A) Columnar storage automatically pushes the filter into the OS page cache, reducing system calls.
- B) Only the `parameterId` and `value` columns need to be read from disk/memory; the 5 other columns are never touched, reducing I/O and improving CPU cache utilisation.
- C) Columnar storage allows the database engine to use SIMD on row groups without any code changes.
- D) Columnar storage compresses the `parameterId` and `value` columns together because they are accessed together.

**Answer: B**

- A) Incorrect — Predicate pushdown is a feature of Parquet readers, not a property of the OS page cache; and columnar storage itself is what enables skipping column data, not system call reduction.
- B) Correct — With columnar layout, the 5 unneeded columns occupy completely separate memory regions. The CPU only loads cache lines from the two relevant columns, so both disk I/O (for Parquet) and L1/L2 cache pollution are minimised.
- C) Incorrect — SIMD usage is an implementation detail of the execution engine, not an automatic consequence of columnar storage layout.
- D) Incorrect — Columnar formats store each column independently; `parameterId` and `value` are in separate buffers even if they are both accessed by one query.

---

## Q17 — Parquet Read Speed vs CSV

> **Week reference:** Week 7

**Mental Model:** Parquet avoids all text parsing (the main bottleneck in CSV reading), stores data in binary columnar form, and supports column skipping. These factors combine to make Parquet reads substantially faster than CSV reads.

In the Week 7 exercises, reading the DMI dataset from Parquet with PyArrow took ~0.4 s, compared to ~3 s for the same data from CSV with PyArrow. What accounts for this ~7.5x speedup?

- A) The Parquet file is stored on a faster storage tier (SSD) while the CSV is on a spinning disk.
- B) Parquet stores data in binary columnar format, eliminating text parsing overhead, and the Parquet reader can use stored min/max statistics to skip data.
- C) The Parquet reader uses 16 threads while the CSV reader is single-threaded.
- D) The Parquet file uses in-memory caching, so repeated reads are served from RAM.

**Answer: B**

- A) Incorrect — The exercises use the same underlying storage for both files; the speedup is due to format, not storage hardware.
- B) Correct — Binary encoding means numbers are stored as raw bytes, not text digits, so no number-parsing is required. Column statistics enable row group skipping. Column projection avoids reading unneeded columns. Together these eliminate the dominant costs of CSV reading.
- C) Incorrect — While PyArrow's readers are multi-threaded, the speedup differential between Parquet and CSV is primarily about format efficiency, not thread count.
- D) Incorrect — Parquet has no built-in in-memory caching layer; reads go through the OS page cache just like CSV reads.

---

## Q18 — from_pandas() Behaviour

> **Week reference:** Week 7

**Mental Model:** `pyarrow.Table.from_pandas(df)` converts a Pandas DataFrame into an Arrow Table. For numeric columns backed by NumPy arrays, Arrow may share the same buffer (zero-copy). For object columns, Arrow must allocate new buffers and encode the Python objects.

Which statement about `pyarrow.Table.from_pandas(df)` is most accurate?

- A) It always performs a full deep copy of all data to ensure Arrow's immutability guarantee.
- B) For numeric columns, it may share the underlying NumPy buffer (zero-copy); for object/string columns, it must copy and encode data into Arrow's contiguous string buffers.
- C) It converts the DataFrame into a Parquet file and reads it back as an Arrow table.
- D) It fails if the DataFrame contains `NaN` values because Arrow does not support null values.

**Answer: B**

- A) Incorrect — Arrow explicitly supports zero-copy for well-aligned numeric NumPy arrays; a full deep copy is not always performed.
- B) Correct — For `int64`, `float64`, and similar numpy-backed columns, Arrow wraps the existing buffer without copying. For `object` (Python string) columns, the strings must be serialised into Arrow's contiguous UTF-8 buffer, which requires a copy and new allocation.
- C) Incorrect — `from_pandas()` converts in memory; it does not involve any file I/O or Parquet format.
- D) Incorrect — Arrow has first-class support for nulls via a separate validity bitmap; `NaN` values in numeric columns are handled (either preserved as NaN or converted to null depending on the option passed).

---

## Q19 — Arrow Column Access Pattern

> **Week reference:** Week 7

**Mental Model:** A `pyarrow.Table` exposes columns by name with `table['column_name']` or `table.column('column_name')`. The result is a `ChunkedArray`. To get a regular Python or NumPy array, you call `.to_pylist()` or `.to_numpy()` on the result.

Given `table = pyarrow.csv.read_csv('data.csv')`, how do you correctly access the `value` column?

- A) `table.value`
- B) `table['value']`
- C) `table.loc[:, 'value']`
- D) `table.get_column('value')`

**Answer: B**

- A) Incorrect — `pyarrow.Table` does not support attribute-style column access; that is a Pandas convention.
- B) Correct — `table['value']` returns the `value` column as a `pyarrow.ChunkedArray`. This is the standard and idiomatic way to access a column on a `pyarrow.Table`.
- C) Incorrect — `.loc[]` is a Pandas DataFrame accessor; `pyarrow.Table` does not have a `.loc` attribute.
- D) Incorrect — The correct method name is `table.column('value')`, not `table.get_column('value')`. Both `table['value']` and `table.column('value')` are valid; `get_column` is not.

---

## Q20 — Parquet vs CSV File Size

> **Week reference:** Week 7

**Mental Model:** The DMI January 2023 CSV (uncompressed) is several hundred MB; the equivalent Parquet file is ~86 MB. The reduction comes from binary encoding of numbers (more compact than ASCII digits), columnar layout enabling better compression, and built-in compression codecs (Snappy by default).

In the Week 7 exercises, saving the DMI dataset as Parquet produced an 86 MB file, much smaller than the original uncompressed CSV. Which combination of factors produces this size reduction?

- A) Parquet removes duplicate rows and applies delta encoding to sorted columns.
- B) Parquet uses binary encoding for all values (numbers stored as raw bytes, not text digits) and applies columnar compression codecs such as Snappy or Zstandard.
- C) Parquet stores only a representative sample of the data, discarding rows at the tail of each row group.
- D) Parquet converts all strings to integer codes globally, reducing the file to almost pure integer data.

**Answer: B**

- A) Incorrect — Parquet does not remove duplicate rows. Delta encoding is one available encoding for sorted integer columns, but it is not the primary factor in the overall size reduction.
- B) Correct — Storing a 64-bit float as 8 bytes rather than up to 22 ASCII characters eliminates significant bloat. Columnar layout groups same-typed values together, making compression (Snappy, Gzip, Zstandard) far more effective than row-interleaved data.
- C) Incorrect — Parquet preserves every row; it is a lossless format.
- D) Incorrect — Parquet uses dictionary encoding for certain columns, but this is applied per-column within row groups, not globally across all strings in the file.

---

## Q21 — Arrow and Cache Efficiency

> **Week reference:** Week 7

**Mental Model:** Because a column in Arrow is stored as a single contiguous buffer, a computation that scans the entire column accesses memory sequentially. The CPU hardware prefetcher can predict sequential access patterns and load the next cache line before it is needed, dramatically reducing effective memory latency.

Why does Apache Arrow's columnar memory layout improve CPU cache efficiency for column-scan workloads?

- A) Arrow allocates columns in NUMA-local memory, ensuring each CPU core accesses only its nearest RAM bank.
- B) Arrow stores each column as a contiguous memory buffer, enabling sequential (stride-1) access patterns that the CPU hardware prefetcher can exploit.
- C) Arrow pins column buffers in the L3 cache to prevent eviction by other processes.
- D) Arrow aligns column buffers to SIMD register widths (256 or 512 bits), guaranteeing SIMD instructions can be used without masking.

**Answer: B**

- A) Incorrect — Arrow does not manage NUMA placement; that is a concern for `numactl` or the OS memory allocator.
- B) Correct — A contiguous column buffer produces stride-1 memory accesses as each element is processed in order. The hardware prefetcher detects this pattern and preloads the next cache line, minimising cache misses.
- C) Incorrect — User-space libraries cannot pin memory in the L3 cache; cache pinning is a hardware or OS privilege operation.
- D) Incorrect — Arrow does align buffers to 64-byte boundaries (matching cache line size) by default, but alignment to SIMD width does not guarantee SIMD use; that depends on the computation engine.

---

## Q22 — When Arrow Conversion Cost is Worth It

> **Week reference:** Week 7

**Mental Model:** If you need to do Pandas-style analytics (`.groupby()`, `.merge()`, index-based access) you must convert to Pandas. But if you are loading data just to immediately process it with Arrow-native operations or pass it to another Arrow-aware library, skipping the conversion saves time and memory.

You need to load a large CSV file and immediately compute the sum of one column using Pandas' vectorised operations. Which workflow is most efficient?

- A) Load with `pandas.read_csv()` directly, since the final step is a Pandas operation and conversion overhead would be wasted.
- B) Load with `pyarrow.csv.read_csv()`, convert with `.to_pandas()`, then use Pandas operations — the total time is still lower than loading with Pandas directly.
- C) Load with `pyarrow.csv.read_csv()` and use Arrow compute functions (`pyarrow.compute.sum()`) without converting to Pandas at all.
- D) Both B and C are correct and would produce equivalent performance.

**Answer: C**

- A) Incorrect — `pandas.read_csv()` is ~3.7x slower than `pyarrow.csv.read_csv()` for this dataset, so loading with Pandas is the slowest option even when the result is used in Pandas.
- B) Incorrect — Option B is faster than A, but it still pays the `.to_pandas()` conversion cost (~700 ms). When only a single aggregation is needed, there is no reason to convert to Pandas at all.
- C) Correct — `pyarrow.compute.sum(table['value'])` computes directly on the Arrow buffer without allocating any Pandas objects. This is the fastest and most memory-efficient path for a simple aggregation.
- D) Incorrect — C avoids the `.to_pandas()` conversion cost entirely, so it is measurably faster than B for a single aggregation.

---
