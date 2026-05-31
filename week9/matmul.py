import numpy as np
from numba import jit

@jit(nopython=True)
def matmul(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for k in range(A.shape[1]):      # swap k and j vs original ijk
            for j in range(B.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C
