# Exam Prep — 02613 Python HPC
*Based on the 2024 exam, 2024 re-exam, and all AI practice question sets.*

---

## Contents

1. [Amdahl's Law](#1-amdahls-law)
2. [LSF / BSUB](#2-lsf--bsub)
3. [Job Arrays and Dependencies](#3-job-arrays-and-dependencies)
4. [Cache and Memory Layout](#4-cache-and-memory-layout)
5. [NumPy Broadcasting](#5-numpy-broadcasting)
6. [Profiling](#6-profiling)
7. [Parallelism Strategy](#7-parallelism-strategy)
8. [GPU / CUDA Kernels](#8-gpu--cuda-kernels)
9. [GPU Memory Transfers](#9-gpu-memory-transfers)
10. [Parallel Reduction](#10-parallel-reduction)
11. [Pandas Dtype Optimization](#11-pandas-dtype-optimization)
12. [Zarr and Memory-Mapped Files](#12-zarr-and-memory-mapped-files)
13. [NUMA and numactl](#13-numa-and-numactl)
14. [HPC Pitfalls](#14-hpc-pitfalls)
15. [Blosc Compression](#15-blosc-compression)
16. [Floating-Point Dtypes](#16-floating-point-dtypes)
17. [Numba JIT](#17-numba-jit)
18. [PyArrow and Apache Arrow](#18-pyarrow-and-apache-arrow)
19. [Quick Reference Card](#19-quick-reference-card)
20. [Exam Traps](#20-exam-traps)

---

## 1. Amdahl's Law

**Formula:** `S(p) = 1 / ((1-F) + F/p)`

**Max speedup:** `S_max = 1 / (1-F)`

**Reverse — find T(1) from T(p):**
```
T(p) = T(1) * ((1-F) + F/p)
T(1) = T(p) / ((1-F) + F/p)
```

**Reading a speedup plot:**
- Curve flattens asymptotically at `S_max` → read the plateau → `F = 1 - 1/S_max`
- Speedup rises then **drops** → NUMA effect (not Amdahl)

**Worked example:** T(4)=10 min, F=0.8 → T(1) = 10 / (0.2 + 0.8/4) = 10/0.4 = 25 min

**Optimizing serial vs buying cores:**
- Optimizing serial: raises S_max dramatically (e.g. 20s→5s serial changes S_max from 6 to 21)
- Buying cores: bounded by fixed serial fraction — wins only if serial fraction is tiny

---

## 2. LSF / BSUB

**Core directives:**
```bash
#BSUB -J jobname
#BSUB -q hpc                        # CPU queue
#BSUB -q gpuv100                    # GPU queue
#BSUB -n 8                          # number of cores
#BSUB -R "span[hosts=1]"            # all cores on ONE node (required for shared memory)
#BSUB -R "rusage[mem=2GB]"          # memory PER CORE (not total)
#BSUB -R "rusage[ngpus_excl_p=1]"   # 1 exclusive GPU
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 00:15                      # wall time hh:mm
#BSUB -o out_%J.out                 # %J = job ID
#BSUB -e err_%J.err
```

**Memory is per core:** 16GB total with 8 cores → `rusage[mem=2GB]`

**GPU job:** change queue to `gpuv100` and add `rusage[ngpus_excl_p=1]`

**Thread environment variables must be exported:**
```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
```
Without `export`, Python never sees them — the single most common pitfall.

---

## 3. Job Arrays and Dependencies

**Array syntax:**
```bash
#BSUB -J job[1-10]         # 10 jobs, indices 1..10
#BSUB -J job[1-20:3]       # step=3 → 1,4,7,10,13,16,19 (7 jobs)
#BSUB -J job[1-100]%5      # submit 100, max 5 running at once
```

**$LSB_JOBINDEX is 1-based — always subtract 1:**
```python
idx = int(sys.argv[1]) - 1    # pass $LSB_JOBINDEX as arg
files[idx]                    # correct 0-based access
```

**Per-element log files — use %I:**
```bash
#BSUB -o results_%J_%I.out   # %I = array index; without %I all elements share one file
```

**Dependencies:**

| Directive | Behaviour |
|-----------|-----------|
| `#BSUB -w "done(job)"` | ALL elements must reach DONE. Any EXIT = stuck forever |
| `#BSUB -w "ended(job)"` | Any terminal state (DONE or EXIT) is enough |

**Exam trap:** 1 out of 10 jobs fails → `done()` is permanently blocked. Answer: use `ended()` or resubmit the failed job.

**bjobs -A column to check:** EXIT > 0 means `done()` will never fire.

---

## 4. Cache and Memory Layout

**Golden rule:** Innermost loop = smallest stride = sequential memory.

**C-order strides (default NumPy, float64=8 bytes):**
```
shape (A, B, C, D) → strides (B·C·D·8,  C·D·8,  D·8,  8)
```
The last axis always has stride 8 (one element). The first axis has the largest stride.

**Given strides, find optimal loop order:**
1. List axes and strides: e.g. i=600, j=40, k=8, l=200
2. Sort ascending: k=8, j=40, l=200, i=600
3. Innermost loop → k (stride 8), outermost → i (stride 600)

**CPU vs CUDA layout rules:**
- **CPU:** innermost loop maps to last axis (smallest stride) → channels-last (H,W,C) better than (C,H,W)
- **CUDA:** `threadIdx.x` maps to last axis (columns) for coalescing → same preference

**Byte size of an array:**
```
elements × bytes_per_element
np.random.rand(1, SIZE) float64 with SIZE=40 → 40 × 8 = 320 bytes
np.random.rand(SIZE, SIZE) float64 with SIZE=35 → 35×35×8 = 9800 bytes
```

**Transpose is a view (no copy), just reverses strides:**
```python
a.strides == (32, 8)  →  a.T.strides == (8, 32)
np.shares_memory(a, a.T) == True
```

---

## 5. NumPy Broadcasting

**Rules:**
1. Right-align shapes, pad left with 1s
2. Each pair: must be equal OR one is 1
3. Output dim = max of pair
4. Any mismatch → ValueError

**Worked examples:**
```
(N, H, W, 3) - (3,)         pads→ (1,1,1,3)    OK  → (N,H,W,3)
(N, H, W, 3) - (H, W)       pads→ (1,1,H,W)    last: 3 vs W → ERROR if W≠3
(32, 224, 224, 3) - (3,)     → OK → (32,224,224,3)
(100, 1, 6, 3) + (100, 1, 3) pads→ (1,100,1,3) → (100,100,6,3)
(3, 4) + (3,)                pads→ (1,3)   last: 4 vs 3 → ERROR
(3, 4) + (3, 1)              →           OK → (3,4)  ← fix with [:, None]
```

**Common exam question — subtract mean from images:**
```python
images.shape = (N, H, W, 3)
mim.shape    = (H, W)          # mean image

images - mim                   # pads (1,1,H,W) → last 3 vs W → wrong/error
images - mim[:, :, None]       # (H,W,1) → broadcasts channel dim → CORRECT
```

**Pairwise distance pattern (haversine style):**
```python
diff = p1[:, None, :] - p2[None, :, :]   # (N,1,2) - (1,M,2) → (N,M,2)
```

**Fix rule:** if `mu.shape == (N,)` and you need to subtract per-sample from `(N, features)` → use `mu[:, None]` to get `(N, 1)`.

---

## 6. Profiling

**cProfile output columns:**

| Column | Meaning |
|--------|---------|
| `ncalls` | times the function was called |
| `tottime` | time in THIS function only (no callees) |
| `cumtime` | total time including all callees |
| `percall` | cumtime / ncalls |

**Where to optimize:** Highest `cumtime` function when scaled to production workload.
- If profiling used 10 samples but production uses 1000 → ncalls scales 100×
- Fixed-cost functions (load, save) do NOT scale → ignore them

**ncalls from cProfile:** `process_sample` called 10 times → 10 samples in the subset.

**Line profiler — reading hits and FLOP/s:**
- `Hits` = how many times that line ran
- Time unit: usually microseconds (1s = 10⁶ μs)
- FLOP/s = (FLOPs per hit × total hits) / total_time_in_seconds

---

## 7. Parallelism Strategy

**The GIL decision tree:**
```
Does the inner work release the GIL?
├── No (pure Python loop, dict ops) → multiprocessing Pool
└── Yes → ThreadPool works
    ├── NumPy ufuncs and matmul → yes, releases GIL
    ├── Blosc compress/decompress → yes
    └── Numba @jit(nogil=True) → yes
```

**Reading `time` output:**
```
real 0m5.2s   ← wall-clock (what matters)
user 1m20s    ← total CPU across all cores
sys  0m0.1s
```
More cores → real ↓, user stays similar or ↑ (multiprocessing adds user overhead).

**Static vs dynamic scheduling:**
- **Static** (equal chunks): use when all tasks take similar time — pi estimation, matrix multiply
- **Dynamic** (one task at a time): use when task times vary — Mandelbrot (edge ≠ interior pixels)

**Pool() default workers:** `os.cpu_count()` = all node cores, NOT LSF-allocated cores. Always: `Pool(n_proc)`.

**Threading vs multiprocessing for NumPy sum:**
- NumPy `.sum()` releases the GIL → ThreadPool works, no pickling overhead → prefer ThreadPool
- Row sums: each thread gets a contiguous row → fast
- Column sums: strided access → slower, but threading still correct

---

## 8. GPU / CUDA Kernels

**Thread indexing pattern:**
```python
@cuda.jit
def my_kernel(arr, out, h, w):
    i, j = cuda.grid(2)          # i=row index, j=col index
    if i < h and j < w:          # ALWAYS bounds-check
        out[i, j] = arr[i, j] * 2
```

**Block configuration for coalescing (C-order/row-major):**
- `threadIdx.x` → column axis (last axis in C-order)
- Adjacent threads in a warp differ in `x` → they access adjacent columns → coalesced
- `1×256` beats `256×1` for 1D row-major data
- `16×16` or `32×32` good default for 2D

**Grid size:**
```python
threads = (16, 16)
blocks  = (math.ceil(W / 16), math.ceil(H / 16))  # x=cols, y=rows
```

**Layout question pattern:**
- Innermost loop over k (channels) → image[k, i, j] strides C×H×W → bad on CPU
- For CUDA: `row, col = cuda.grid(2)` accessing `x[row, col]` → C-order (H,W) is coalesced
- Channel inner loop → `image[k, row, col]` → k varies, row/col fixed → strided → bad
- Fix: channels-last `image[row, col, k]` → k sequential in memory

**Device function:** `@cuda.jit(device=True)` → callable only from GPU, not from host.

---

## 9. GPU Memory Transfers

**Implicit (auto) transfer rules (Numba default):**
- NumPy array argument → 1 HtoD at call time
- Output / input-output array → 1 DtoH after kernel returns
- Array transferred in = array transferred out

**Counting transfers in a loop:**
```python
for x in x_all:                      # m images
    kernel[grid, block](x, y, n)     # x: 1 HtoD per iter = m HtoD
                                     # y: input-output → m HtoD + m DtoH
# Total: 2m HtoD + m DtoH  (very wasteful)

# Optimal: move y to device once before loop
y_d = cuda.to_device(y)
for x in x_all:
    kernel[grid, block](x, y_d, n)   # x: m HtoD; y_d: 0 HtoD
y = y_d.copy_to_host()               # 1 DtoH
# Total: m HtoD + 1 DtoH
```

**Transfer speed from nsys profiler:**
```
size = 25000 MB (HtoD, 2 transfers)  →  each = 12500 MB
time = 2.5s (HtoD total)             →  each = 1.25s
speed = 12500 MB / 1.25s = 10 GB/s
```

---

## 10. Parallel Reduction

**Requirements for correctness:**
- **Associativity required:** `(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)` — tree combines in any order
- **Commutativity needed** if using unordered aggregation

| Operator | Associative | Commutative | Safe? |
|----------|-------------|-------------|-------|
| `sum` | Yes | Yes | Yes |
| `max` / `min` | Yes | Yes | Yes |
| `abs(x+y)` | No | — | **NO** |
| matrix multiply | Yes | No | Partially |

**abssum trap:** `abs(a+b) ≠ abs(a) + abs(b)` → gives wrong results in parallel tree.

**Binary tree speedup:** N elements → ceil(log₂(N)) rounds of pool.map.
- N=64 → 6 rounds
- Each round halves active elements at doubling stride

**Tree vs linear comparison:**
- Linear (row-sum then serial combine): O(N/p) + O(p) final serial
- Tree: O(log₂ N) rounds → much faster for large N

---

## 11. Pandas Dtype Optimization

**Decision table:**

| Column type | Action |
|-------------|--------|
| `object`, low cardinality (< ~1% unique) | → `category` |
| `object`, date strings | → `datetime64[ns]` via `pd.to_datetime()` |
| `int64`, small range | → smallest int that fits |
| `float64`, low precision needed | → `float32` |
| `int64`, range 0–8 | → `uint8` (range 0–255) |
| `int64`, range up to ~65k | → `uint16` |

**Integer type ranges:**
```
uint8:  0–255          int8:  -128–127
uint16: 0–65535        int16: -32768–32767
uint32: 0–4.3B         int32: ±2.1B
```

**float32 exact integers:** up to 2²⁴ = 16,777,216 — safe for most IDs.

**Chunked processing max rows:**
```python
bytes_per_row = 4 + 8 + 2   # uint32 + float64 + int16
max_rows = 200_000_000 // bytes_per_row
```

**Fast repeated date queries:**
```python
df_idx = df.set_index('date').sort_index()
df_idx.loc['2023-06-15']    # O(log n) binary search vs O(n) boolean mask every time
```

**TextFileReader (chunked CSV):** forward-only iterator — exhausted after first pass. Store results during iteration; reopen to iterate again.

**memory_usage(deep=True):** object columns: shallow = 8 bytes/pointer; deep = actual string heap size (~50 bytes/string).

---

## 12. Zarr and Memory-Mapped Files

**np.memmap modes:**

| Mode | Behaviour |
|------|-----------|
| `'r'` | read-only, file must exist |
| `'r+'` | read-write, file must exist, no truncation |
| `'w+'` | read-write, creates/truncates file |
| `'c'` | copy-on-write: changes not saved to disk |

**Zarr block shape for column access:**
```python
x = zarr.open(fname, mode='r')   # shape (1000, 100000)
for i in columns:
    s += x[:, i].sum()           # reads full column each time
```
Best block = tall and narrow → `(1000, 100)` loads one column per block read.
- `(10, 10000)` = wide and short → loads many unused columns per block
- `(1000, 100)` = full height, narrow → each block covers exactly the column needed

**Block memory:** `rows × cols × bytes_per_element`
```
(1000, 100) float64 → 1000 × 100 × 8 = 800,000 bytes = 0.8 MB
```

---

## 13. NUMA and numactl

**What NUMA is:** Dual-socket servers have two NUMA nodes. Each socket has its own DRAM. Accessing the other socket's memory is ~2.1× slower (distance ratio 21/10).

**First-touch allocation (default):** The process that first writes a page owns it. Main process writes → all data on socket 0 → socket 1 workers see slow remote access.

**numactl commands:**

| Command | Effect |
|---------|--------|
| `numactl --interleave=all python ...` | Round-robin pages across all NUMA nodes |
| `numactl --membind=0 python ...` | All memory on node 0 only |
| `numactl --cpunodebind=0 python ...` | Run only on socket 0 cores |
| `numactl --cpunodebind=0 --membind=0` | Restrict both cores AND memory to node 0 |

**When to use what:**
- `--interleave=all` → best for multi-socket jobs (equalises latency for all workers)
- `--membind=0` → maximum local bandwidth for socket-0-only jobs
- `--cpunodebind=0 --membind=0` → pin everything to one socket

**Speedup curve with NUMA:**
- Without numactl: speedup rises until ~50% threads (socket 0 full), then **drops** (socket 1 pays remote penalty)
- With numactl --interleave=all: speedup keeps rising for all cores

**numactl --interleave=all does NOT restrict CPU affinity.** It only affects memory placement. All cores remain available.

**span[hosts=1] is NOT numactl.** It ensures all cores are on one node; it does NOT set any memory policy.

---

## 14. HPC Pitfalls

**The 4 classic pitfalls (from Week 13):**

### Pitfall 1 — Missing export on thread variables
```bash
OMP_NUM_THREADS=8      # WRONG: local variable, Python never sees it
export OMP_NUM_THREADS=8   # CORRECT
```

### Pitfall 2 — Subshell export trap
```bash
(
  export OMP_NUM_THREADS=8   # lives only in subshell, dies when ) closes
)
python matmul.py             # OMP_NUM_THREADS is NOT set here
```

### Pitfall 3 — Output routed through LSF channel (slow)
```bash
python -u script.py                          # 80s for 100k print lines (LSF channel)
python -u script.py > /work3/out_$LSB_JOBID.txt  # 3s (direct file redirect)
```
`-u` = unbuffered output (needed for accurate timing). `LSB_JOBID` works in shell redirects; `%J` only works in `#BSUB` lines.

### Pitfall 4 — ThreadPool + multi-threaded NumPy oversubscription
```bash
# OMP_NUM_THREADS=8, ThreadPool(8) → 8 × 8 = 64 threads on 8 cores → slower
# Fix: export OMP_NUM_THREADS=1, ThreadPool(8) → 8 threads on 8 cores → fast
```

**Memory calculation:** Each 1000×1000 float64 matrix = 8 MB. Three such arrays = 24 MB. Add ~1 GB Python overhead → request `rusage[mem=4GB]` for safety. Too low = OOM kill. Too high = long queue wait.

**Login node rule:** Never run computation on login nodes. Use `bsub < submit.sh`.

---

## 15. Blosc Compression

**When Blosc is faster than NumPy:**
- **Zeros** and **tiled (repetitive) data** → high compression ratio → smaller I/O → faster overall
- **Random data** → incompressible → Blosc overhead exceeds any benefit → use NumPy directly

**Codec comparison:**

| Codec | Speed | Ratio | Best for |
|-------|-------|-------|---------|
| lz4 | Fast compress + decompress | Moderate | Default choice |
| zstd | Slower compress, similar decompress | Better ratio | When space matters |
| blosclz | Similar to lz4 | Moderate | Alternative to lz4 |

**zstd vs lz4 for zeros/tiles:**
- Writing (compression): zstd is **slower**
- Reading (decompression): similar speed
- File size: zstd is **smaller**
- Answer: "Reading is the same, writing is slower, but it uses less space"

**Compression ratio for zeros using lz4:** 100 or above (zeros compress to near-zero bytes).

**SHUFFLE filter:**
- Rearranges bytes by significance position within each element before compression
- Helps for multi-byte numeric types (float32, float64)
- `BITSHUFFLE` > `SHUFFLE` > `NOSHUFFLE` in compression ratio for structured floats

**Lossless guarantee:** Blosc is always lossless. `pack_array`/`unpack_array` preserves dtype, shape, and all values exactly. Thread count does NOT affect the compressed content.

**pack_array vs compress:**
- `blosc.pack_array(arr)` → serialises array + metadata (dtype, shape) → use `unpack_array` to recover
- `blosc.compress(bytes)` → raw bytes only → `decompress` returns bytes, not array

---

## 16. Floating-Point Dtypes

**Type ranges and precision:**

| Dtype | Bits | Range | Notes |
|-------|------|-------|-------|
| float64 | 64 | ±1.8×10³⁰⁸ | 15–17 significant digits |
| float32 | 32 | ±3.4×10³⁸ | 6–7 significant digits |
| float16 | 16 | ±65504 | 3–4 significant digits |
| int64 | 64 | ±9.2×10¹⁸ | exact |
| int32 | 32 | ±2.1×10⁹ | exact |
| int8 | 8 | -128 to 127 | wraps on overflow |
| uint8 | 8 | 0 to 255 | wraps on overflow |

**float16 precision loss:**
```python
np.float16(10000) + np.float16(1) == np.float16(10000)  # True! 1 lost
np.float16(100) + np.float16(0.1) ≈ 100.06              # rounding
```
float16 spacing near 10000 is 8 — any addition smaller than 4 is lost.

**int8 overflow — wraps silently:**
```python
np.int8(127) + np.int8(1) == -128   # wraps around
```

**float32 exact integers:** up to 2²⁴ = 16,777,216. Values above this lose integer precision.

---

## 17. Numba JIT

**Key decorators:**
```python
@jit(nopython=True)        # = @njit: no Python fallback, fastest
@jit(nopython=True, nogil=True)  # also releases GIL → use with ThreadPool
@cuda.jit                  # GPU kernel
@cuda.jit(device=True)     # device function: called from GPU only, not host
```

**First-call warmup:** The first call compiles the function (takes seconds). Timing must exclude the first call:
```python
func(x)          # warmup — do NOT time this
t0 = time()
func(x)          # time this
```

**nogil=True + ThreadPool:**
```python
# Works because nogil releases the GIL, threads run truly in parallel
with ThreadPool(8) as p:
    results = p.map(my_numba_func, data)
```
Without `nogil=True`, threads still block on the GIL → no speedup.

**Unsupported types in nopython mode:** Python dicts, lists of mixed types, arbitrary Python objects → will raise `TypingError`. Must use NumPy arrays or Numba-supported types.

---

## 18. PyArrow and Apache Arrow

**read_csv returns Table, not DataFrame:**
```python
import pyarrow.csv as pa_csv
table = pa_csv.read_csv('data.csv')    # pyarrow.Table
df    = table.to_pandas()              # pandas.DataFrame (copy for string cols)
```

**Column access returns ChunkedArray:**
```python
col = table['value']           # pyarrow.ChunkedArray (not Series, not ndarray)
col.to_numpy()                 # convert to numpy
```

**Why Arrow uses less memory than Pandas:**
- Pandas `object` column: 8-byte pointer per element + Python str on heap (~50 bytes each)
- Arrow string column: contiguous UTF-8 buffer + offsets array → much smaller

**Type overrides at load time:**
```python
# CORRECT way (Arrow API):
opts = pa_csv.ConvertOptions(column_types={'value': pa.float32()})
table = pa_csv.read_csv('data.csv', convert_options=opts)

# WRONG (pandas API, not valid for pa_csv.read_csv):
table = pa_csv.read_csv('data.csv', dtype={'value': 'float32'})  # TypeError
```

**Parquet column projection (read only needed columns):**
```python
df = pd.read_parquet('file.parquet', columns=['id', 'value'])
# Only those byte ranges are read from disk — other columns never loaded
```

**Slicing is zero-copy:** Arrow slice creates a view over the same buffer (new offset + length). No data copied.

**Arrow vs Pandas load speed (DMI dataset):**
- Arrow: ~3s, ~507 MB in memory
- Pandas: ~11s, ~2045 MB in memory
- Arrow → Pandas conversion: +700ms, memory inflates further

---

## 19. Quick Reference Card

### Amdahl's Law
```
S(p) = 1 / ((1-F) + F/p)
S_max = 1 / (1-F)
F = 1 - 1/S_max
T(1) = T(p) / ((1-F) + F/p)
```

### C-Order Strides (float64)
```
shape (A, B, C, D) → strides (B·C·D·8,  C·D·8,  D·8,  8)
Innermost loop → axis with stride 8 (last axis)
```

### Broadcasting
```
Right-align → pair each dim → equal or one is 1 → else ValueError
Output dim = max of pair
```

### CUDA Thread Block
```
C-order: threadIdx.x → last axis (columns)
1×256 > 256×1 for 1D row-major
16×16 good default for 2D
Always: if i < h and j < w: ...
```

### Job Array
```bash
#BSUB -J job[1-N]         # 1-based indexing
idx = int(sys.argv[1]) - 1   # Python: subtract 1
done(job)  → all must succeed   ended(job) → any terminal state
%J_%I.out  → per-element files
```

### Threading Decision
```
Pure Python → Pool (multiprocessing)
NumPy / nogil Numba / Blosc → ThreadPool
```

### Dtype Sizes (bytes)
```
float64=8  float32=4  int64=8  int32=4  int16=2  uint8=1  bool=1
```

### Blosc Rule
```
zeros/tiles → Blosc is faster AND smaller
random data → Blosc is similar size, adds overhead → stick with NumPy
```

### numactl
```
--interleave=all  → spread pages across both NUMA nodes (best for multi-socket)
--membind=0       → all memory on node 0
--cpunodebind=0   → restrict cores to socket 0
```

---

## 20. Exam Traps

| Trap | Correct answer |
|------|----------------|
| "Use threading for pure Python loops" | No → GIL blocks → use multiprocessing |
| "`Pool()` uses LSF-allocated cores" | No → uses `os.cpu_count()` (all node cores); pass explicit count |
| "`done(job)` tolerates 1 failed element" | No → any EXIT permanently blocks it; use `ended()` |
| "Column access is faster in NumPy" | No → row access; C-order makes columns strided |
| "CUDA `256×1` block is best for row-major" | No → `1×256`; threadIdx.x maps to columns for coalescing |
| "`span[hosts=1]` applies numactl policy" | No → only ensures 1 node; call numactl explicitly |
| "Export inside `()` works" | No → subshell; parent never inherits the variable |
| "abssum works in parallel reduction" | No → not associative → wrong results |
| "`$LSB_JOBINDEX` is 0-based" | No → 1-based; always subtract 1 in Python |
| "Reuse TextFileReader in a second loop" | No → exhausted after first pass; outputs nothing |
| "Blosc is always faster than NumPy" | No → only for compressible data (zeros, tiles); random data = similar |
| "zstd reads faster, writes slower than lz4" | No → reads are similar; **writing** (compression) is slower |
| "Arrow to Pandas is zero-copy for all types" | No → string columns require copying and allocating Python objects |
| "`pa_csv.read_csv(..., dtype=...)` works" | No → TypeError; use `ConvertOptions(column_types={...})` |
| "Numba @jit first call is fast to time" | No → first call compiles; always warm up before timing |
| "More threads = smaller Blosc output" | No → thread count does not affect compressed content, only speed |
| "LSF memory is total job memory" | No → `rusage[mem=X]` is **per core** |
| "Routing stdout to `-o` is fast for many prints" | No → LSF channel is slow; use shell redirect `>` to scratch |
