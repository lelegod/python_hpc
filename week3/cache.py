import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

n = 100
SIZE_LIST = np.logspace(1, 4, n)
matrix_size = SIZE_LIST**2 * 8
MLOP_S_row = np.zeros(n)
MLOP_S_col = np.zeros(n)

for i, SIZE in enumerate(SIZE_LIST):
    mat = np.random.rand(int(SIZE), int(SIZE))
    operations = SIZE

    start = perf_counter()
    double_column = 2 * mat[:, 0]
    end = perf_counter()
    MLOP_S_col[i] = operations / (end - start) / 1000000

    start = perf_counter()
    double_row = 2 * mat[0, :]
    end = perf_counter()
    MLOP_S_row[i] = operations / (end - start) / 1000000

plt.subplots(1, 1, figsize=(10, 6))
plt.plot(matrix_size, MLOP_S_row, color='blue', label='row')
plt.plot(matrix_size, MLOP_S_col, color='orange', label='col')

L1 = 512 * 1024
L2 = 16 * 1024**2
L3 = 22 * 1024**2
plt.axvline(L1, linestyle='--', label='L1')
plt.axvline(L2, linestyle='--', label='L2')
plt.axvline(L3, linestyle='--', label='L3')

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig("cache.jpg")