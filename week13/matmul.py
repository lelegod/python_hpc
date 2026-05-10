from time import perf_counter as time
import numpy as np

def matmuls(A, B):
    assert A.shape[0] == B.shape[0], "A and B must hold same number of matrices"
    assert A.shape[2] == B.shape[1], "A and B must hold compatible matrices"
    n = A.shape[0]
    C = np.empty((n, A.shape[1], B.shape[2]))
    for i in range(n):
        C[i] = np.matmul(A[i], B[i])
    return C

if __name__ == '__main__':
    A = np.random.rand(100, 1000, 1000)
    B = np.random.rand(100, 1000, 1000)
    t0 = time()
    C = matmuls(A, B)
    t1 = time()
    print(t1 - t0)