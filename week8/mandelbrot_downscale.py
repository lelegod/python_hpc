import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    path = sys.argv[1]
    N = int(sys.argv[2])
    n = int(sys.argv[3])

    mm = np.memmap(path, dtype='int32', mode='r', shape=(N, N))
    downscaled = mm[::n, ::n]

    plt.imshow(downscaled, cmap='hot', extent=(-2, 2, -2, 2))
    plt.axis('off')
    plt.savefig('mandelbrot.png', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    main()
