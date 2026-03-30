import numpy as np

def distmat_1d(x, y):
    return abs(x[:, None] - y)

if __name__ == "__main__":
    x = np.array([1,3])
    y = np.array([1,2,3])
    print(distmat_1d(x, y))