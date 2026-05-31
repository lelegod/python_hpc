from os.path import join
import sys

import numpy as np
import cupy as cp


def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2), dtype=cp.float32)
    u[1:-1, 1:-1] = cp.asarray(np.load(join(load_dir, f"{bid}_domain.npy")))
    interior_mask = cp.asarray(np.load(join(load_dir, f"{bid}_interior.npy")))
    return u, interior_mask


@cp.fuse()
def jacobi_kernel(u_center, u_left, u_right, u_up, u_down, mask):
    u_new = 0.25 * (u_left + u_right + u_up + u_down)
    return cp.where(mask, u_new, u_center)

def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = cp.copy(u)

    for i in range(max_iter):
        # Check convergence first
        if i % 100 == 0:
            # Calculate the proposed new values
            u_new_temp = 0.25 * (u[..., 1:-1, :-2] + u[..., 1:-1, 2:] + u[..., :-2, 1:-1] + u[..., 2:, 1:-1])
            delta = cp.max(cp.abs(u[..., 1:-1, 1:-1] - u_new_temp) * interior_mask)
            if delta < atol:
                u[..., 1:-1, 1:-1] = cp.where(interior_mask, u_new_temp, u[..., 1:-1, 1:-1])
                break
                
        # Update using the fused kernel
        u[..., 1:-1, 1:-1] = jacobi_kernel(
            u[..., 1:-1, 1:-1],
            u[..., 1:-1, :-2],   
            u[..., 1:-1, 2:],    
            u[..., :-2, 1:-1],   
            u[..., 2:, 1:-1],    
            interior_mask
        )

    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = (cp.sum(u_interior > 18) / u_interior.size * 100).item()
    pct_below_15 = (cp.sum(u_interior < 15) / u_interior.size * 100).item()
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    BATCH_SIZE = 200

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))

    for i in range(0, N, BATCH_SIZE):
        batch_bids = building_ids[i:i + BATCH_SIZE]
        current_batch_size = len(batch_bids)

        batch_u0 = cp.empty((current_batch_size, 514, 514), dtype=cp.float32)
        batch_interior_mask = cp.empty((current_batch_size, 512, 512), dtype='bool')

        for j, bid in enumerate(batch_bids):
            u0, interior_mask = load_data(LOAD_DIR, bid)
            batch_u0[j] = u0
            batch_interior_mask[j] = interior_mask

        # Run jacobi iterations for the batch
        batch_u = jacobi(batch_u0, batch_interior_mask, MAX_ITER, ABS_TOL)

        for j, bid in enumerate(batch_bids):
            stats = summary_stats(batch_u[j], batch_interior_mask[j])
            print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))