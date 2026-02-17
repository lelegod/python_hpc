import numpy as np
import sys

def save_mean():
    M = np.load(sys.argv[1])
    np.save("cols.npy", np.mean(M, axis=0))
    np.save("rows.npy", np.mean(M, axis=1))
    pass

if __name__ == "__main__":
    save_mean()