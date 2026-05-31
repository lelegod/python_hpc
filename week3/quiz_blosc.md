# Blosc Quiz

*Autolab: Multiple choice questions about exercise 2.5 in week 3*

---

## Q1: For which arrays was using blosc with compression faster than NumPy? *(select all that apply)*

- [x] **zeros**
- [x] **tile**
- [ ] random

> Zeros and tiled arrays have high redundancy and compress extremely well, making blosc read/write faster overall. Random data has low redundancy and compresses poorly — blosc overhead exceeds any benefit.

---

## Q2: Which statement below is true when it comes to using compressed data?

- [ ] Compressed data is always slower but uses less space
- [ ] Compressed data is always faster and also uses less space
- [x] **When data contains repeated parts compression is both faster and uses less space**
- [ ] Compressed data makes little to no difference

> Compression trades CPU time for I/O. When data is highly compressible (zeros, tiles), the compressed form is so much smaller that read/write is faster even after decompression overhead.

---

## Q3: What compression ratio does blosc achieve over numpy (size of .npy file / size of .bl file) for the zero arrays using lz4?

- [ ] None, they are about the same
- [ ] 1-10
- [ ] 10-100
- [x] **100 or above**

> All-zero arrays compress to near-zero size under lz4. The ratio of the raw `.npy` to the compressed `.bl` file is typically well above 100×.

---

## Q4: What changes when using zstd instead of lz4 (for zeros and tiles data at least)?

- [ ] Reading and writing are both faster, but space use is the same
- [ ] Reading and writing are both faster, but it uses more space
- [ ] Reading and writing are both slower, but it uses less space
- [x] **Reading is the same, writing is slower, but it uses less space**
- [ ] Reading is slower, writing is the same, but it uses less space

> zstd compresses more aggressively than lz4, so **compression (writing) is slower**. Decompression (reading) speed is similar between lz4 and zstd. The payoff is smaller file sizes.

---

*Submission history: version 6 = 100.0/100*
