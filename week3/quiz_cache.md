# Cache Quiz

*Autolab: Multiple choice questions about exercises 1.4 + 1.6 from week 3*

---

## Q1: If SIZE is 40, how many bytes does the row vector `mat = np.random.rand(1, SIZE)` occupy?

- [ ] 40
- [ ] 400
- [x] **320**
- [ ] 160

> float64 = 8 bytes per element. 1 × 40 × 8 = 320 bytes.

---

## Q2: If SIZE is 35, how many bytes does the row vector `mat = np.random.rand(SIZE, SIZE)` occupy?

- [ ] 280
- [x] **9800**
- [ ] 1225
- [ ] 4900

> float64 = 8 bytes per element. 35 × 35 = 1225 elements × 8 = 9800 bytes.

---

## Q3: Which of the following most resembles your plot from exercise 1.4?

*(Image-based question — based on your experimental results)*

The correct plot shows **row access staying flat/fast** while **col access slows down** as SIZE grows beyond cache limits. Row-major iteration (sequential access) remains cache-friendly; column-major iteration (strided access) degrades.

---

## Q4: When do the performance of the row and column scaling start to diverge?

- [ ] They do not; they remain more or less the same
- [x] **After the array no longer fits in the L1 cache**
- [ ] After the array no longer fits in the L2 cache
- [ ] After the array no longer fits in the L3 cache

> Once the array exceeds L1 capacity, stride-based column access can no longer be served from L1 and must go to L2/RAM, causing the row vs. col split to appear.

---

## Q5: Which of the following most resembles your plot from exercise 1.6?

*(Image-based question — based on your experimental results)*

The correct plot shows **a curve that rises, peaks, then drops** as SIZE increases — reflecting the transition from L1 → L2 → L3 → RAM with visible bandwidth cliff.

---

*Submission history: version 3 = 100.0/100*
