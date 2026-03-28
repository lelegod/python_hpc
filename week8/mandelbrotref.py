import numpy as np

def mandelbrot_escape_time(c):
    z = 0
    for i in range(100):
        z = z**2 + c
        if np.abs(z) > 2:
            return i
    return 100


def mandelbrot(size):
    xpts, ypts = np.meshgrid(
        # Add 1 to size to make it compatible with previous weeks
        np.linspace(-2, 2, size+1)[:-1],
        np.linspace(-2, 2, size+1)[:-1],
    )
    points = 1j * xpts.ravel() + ypts.ravel()
    mandelbrot = np.array([mandelbrot_escape_time(c) for c in points])
    mandelbrot = mandelbrot.reshape((size, size))
    return mandelbrot