import multiprocessing
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_escape_time(c):
    z = 0
    for i in range(100):
        z = z**2 + c
        if np.abs(z) > 2.0:
            return i
    return 100

def generate_mandelbrot_set(points, num_processes):
    chunk_size = len(points) // num_processes
    pool = multiprocessing.Pool(num_processes)
    escape_times = pool.map(mandelbrot_escape_time, points, chunksize=chunk_size)
    pool.close()
    pool.join()
    return np.array(escape_times)

def plot_mandelbrot(escape_times):
    plt.imshow(escape_times, cmap='hot', extent=(-2, 2, -2, 2))
    plt.axis('off')
    plt.savefig('mandelbrot.png', bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    width = 800
    height = 800
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2
    num_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    x_values = np.linspace(xmin, xmax, width)
    y_values = np.linspace(ymin, ymax, height)
    points = np.array([complex(x, y) for x in x_values for y in y_values])

    t0 = time.time()
    mandelbrot_set = generate_mandelbrot_set(points, num_proc)
    elapsed = time.time() - t0

    print(f"{num_proc},{elapsed:.4f}")

    mandelbrot_set = mandelbrot_set.reshape((height, width))
    plot_mandelbrot(mandelbrot_set)
