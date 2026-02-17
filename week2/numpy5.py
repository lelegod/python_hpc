import numpy as np
import sys
from time import perf_counter

def save_mean():
    A, p = np.load(sys.argv[1]), int(sys.argv[2])
    start = perf_counter()
    np.save("saved.npy", np.linalg.matrix_power(A, p + 1))
    end = perf_counter()
    print(f"{start - end}")
    pass

if __name__ == "__main__":
    save_mean()