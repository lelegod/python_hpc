import numpy as np
import sys

def save_diag():
    e = np.array(sys.argv[1:]).astype(float)
    np.save("saved.npy", np.diag(e))
    pass

if __name__ == "__main__":
    save_diag()