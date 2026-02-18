import numpy as np
from time import perf_counter

SIZE = 100

mat = np.random.rand(SIZE, SIZE)
start = perf_counter()
double_column = 2 * mat[:, 0]
double_row = 2 * mat[0, :]
end = perf_counter()

print(f"{end - start}")