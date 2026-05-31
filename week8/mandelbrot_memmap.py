import sys
import numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mandelbrotref import mandelbrot

def main():
    N = int(sys.argv[1])

    mm = np.memmap('mandelbrot.raw', dtype='int32', mode='w+', shape=(N, N))
    result = mandelbrot(N)
    mm[:] = result
    del mm  # flush to disk

if __name__ == '__main__':
    main()
