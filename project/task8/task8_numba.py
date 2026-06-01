import math
import numpy as np
from numba import cuda


@cuda.jit
def jacobi_kernel(u, u_new, mask):
    # [Claude] INEFFICIENCY — non-coalesced memory access: cuda.grid(2) returns
    # (x, y) where x varies within a warp. Assigning x→i (row) means warp threads
    # stride across rows → stride = n_cols in memory → non-coalesced.
    # Fix: `j, i = cuda.grid(2)` so x→j (column) → stride-1 → coalesced reads.
    i, j = cuda.grid(2)
    n, m = u.shape

    # Guard: launched grid may be larger than the array (ceiling division on block count).
    if i < n and j < m:

        # Border rows/cols hold fixed boundary conditions — copy unchanged.
        if i == 0 or i == n - 1 or j == 0 or j == m - 1:
            u_new[i, j] = u[i, j]
        else:
            # [Claude] INEFFICIENCY — warp divergence: threads in the same warp
            # that straddle a wall boundary serialize — GPU runs both branches
            # back-to-back with half the warp idle each time. Impact depends on
            # how fragmented the mask boundary is across warp-aligned columns.
            # mask is (N, N); u is (N+2, N+2), hence the -1 offset to align them.
            if mask[i - 1, j - 1]:
                # Standard 5-point Jacobi stencil (average of 4 neighbours).
                # [Claude] INEFFICIENCY — no shared memory tiling: all 5 reads hit
                # global memory every iteration. Adjacent threads share 3 of these
                # values (e.g. u[i, j+1] for thread j == u[i, j] for thread j+1)
                # but each thread reloads them independently. A shared-memory tile
                # would load each value once per block, cutting global reads ~5×.
                u_new[i, j] = 0.25 * (u[i, j - 1] + u[i, j + 1] + u[i - 1, j] + u[i + 1, j])
            else:
                # Wall/exterior point — freeze value, no update.
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

    # 16×16 = 256 threads per block — standard balance between occupancy and register pressure.
    threads_per_block = (16, 16)
    # [Claude] INEFFICIENCY — block grid is also backwards: shape[0] (rows) mapped
    # to x-blocks and shape[1] (cols) to y-blocks, consistent with the i,j swap bug
    # above. When fixing the kernel to `j, i = cuda.grid(2)`, this must also swap:
    # blocks_x should cover columns (shape[1]) and blocks_y should cover rows (shape[0]).
    blocks_per_grid_x = math.ceil(u.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(u.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    for _ in range(max_iter):
        jacobi_kernel[blocks_per_grid, threads_per_block](u_dev, u_new_dev, mask_dev)
        # Ping-pong buffers: swap device pointers instead of copying — avoids
        # a full array copy each iteration (O(N²) device memory write saved).
        u_dev, u_new_dev = u_new_dev, u_dev

    return u_dev.copy_to_host()
