import numpy as np

def outer(x, y):
    return x[:, None] * y

if __name__ == "__main__":
    x = np.array([1,3])
    y = np.array([1,2,3])
    print(outer(x, y))