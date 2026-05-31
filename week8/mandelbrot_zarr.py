import sys
import numpy as np
import zarr
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mandelbrotref import mandelbrot_escape_time

def compute_chunk(row_start, col_start, chunk_rows, chunk_cols, N):
    all_pts = np.linspace(-2, 2, N + 1)[:-1]
    x = all_pts[col_start:col_start + chunk_cols]
    y = all_pts[row_start:row_start + chunk_rows]
    xpts, ypts = np.meshgrid(x, y)
    points = 1j * xpts.ravel() + ypts.ravel()
    result = np.array([mandelbrot_escape_time(c) for c in points])
    return result.reshape(chunk_rows, chunk_cols)

def main():
    N = int(sys.argv[1])
    C = int(sys.argv[2])

    z = zarr.open('mandelbrot.zarr', mode='w', shape=(N, N),
                  chunks=(C, C), dtype='int32')

    for row_start in range(0, N, C):
        for col_start in range(0, N, C):
            chunk_rows = min(C, N - row_start)
            chunk_cols = min(C, N - col_start)
            z[row_start:row_start + chunk_rows,
              col_start:col_start + chunk_cols] = compute_chunk(
                row_start, col_start, chunk_rows, chunk_cols, N)

if __name__ == '__main__':
    main()
