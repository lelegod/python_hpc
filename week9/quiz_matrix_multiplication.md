# Matrix Multiplication Quiz

*Autolab: Questions for exercise 1.3 in week 9*

---

## Q1: Which loop ordering is the most cache efficient?

- [ ] Outermost is i, middle is j, innermost is k  (ijk)
- [ ] Outermost is k, middle is i, innermost is j  (kij)
- [x] **Outermost is i, middle is k, innermost is j  (ikj)**

> For C[i,j] += A[i,k] * B[k,j] in row-major (NumPy default) storage:
> - **ikj** order: the inner j-loop accesses `B[k,j]` sequentially (row of B) and `C[i,j]` sequentially — both are cache-friendly.
> - ijk order: the inner k-loop accesses `B[k,j]` with stride (column of B) — cache-unfriendly.
> - kij order: B is accessed row-wise but C is accessed with a stride pattern.

---

*Submission history: version 1 = 100.0/100*
