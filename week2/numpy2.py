import numpy as np
import sys

def magnitude():
    v = np.array(sys.argv[1:]).astype(float)
    return np.linalg.norm(v)

if __name__ == "__main__":
    print(f"{magnitude()}")