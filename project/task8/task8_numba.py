import math
import numpy as np
from numba import cuda


@cuda.jit
def jacobi_kernel(u, u_new, mask):
    i, j = cuda.grid(2)
    n, m = u.shape
    if i < n and j < m:
        if i == 0 or i == n - 1 or j == 0 or j == m - 1:
            u_new[i, j] = u[i, j]
        else:
            if mask[i - 1, j - 1]:
                u_new[i, j] = 0.25 * (u[i, j - 1] + u[i, j + 1] + u[i - 1, j] + u[i + 1, j])
            else:
                u_new[i, j] = u[i, j]


def jacobi_numba(u, interior_mask, max_iter):
    """
    Run the Jacobi solver on the GPU for a fixed number of iterations.

    Args:
        u: 2D NumPy array of shape (N+2, N+2) with boundary padding (initial temperature).
        interior_mask: 2D boolean/uint8 mask of shape (N, N) indicating update points.
        max_iter: number of Jacobi iterations to perform.

    Returns:
        result: 2D NumPy array of the same shape as u after max_iter updates.
    """
    u_dev = cuda.to_device(u)
    mask_dev = cuda.to_device(interior_mask.astype(np.uint8))
    u_new_dev = cuda.device_array_like(u_dev)

    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(u.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(u.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    for _ in range(max_iter):
        jacobi_kernel[blocks_per_grid, threads_per_block](u_dev, u_new_dev, mask_dev)
        u_dev, u_new_dev = u_new_dev, u_dev

    return u_dev.copy_to_host()
