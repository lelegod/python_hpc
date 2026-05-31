import sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from os.path import join, dirname, abspath, exists
from os import makedirs, environ
from time import perf_counter

# CONFIG
LOAD_DIR = environ.get(
    "DTU_HPC_LOAD_DIR",
    "D:/Program Files/SEM2/python_hpc/project/data/dtu/projects/02613_2025/data/modified_swiss_dwellings/",
)
REPORT_DIR = join(dirname(abspath(__file__)), "report")
makedirs(REPORT_DIR, exist_ok=True)
MAX_ITER = 20_000
ABS_TOL  = 1e-4
STAT_KEYS = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]

def load_data(bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))
    interior_mask = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi(u, interior_mask, max_iter=MAX_ITER, atol=ABS_TOL):
    u = np.copy(u)
    for _ in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        "mean_temp":    float(u_interior.mean()),
        "std_temp":     float(u_interior.std()),
        "pct_above_18": float(np.sum(u_interior > 18) / u_interior.size * 100),
        "pct_below_15": float(np.sum(u_interior < 15) / u_interior.size * 100),
    }

def task1(building_ids):
    print("TASK 1:")
    print(f"Total buildings: {len(building_ids)}")

    sample_ids = building_ids[:3]
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    for row, bid in enumerate(sample_ids):
        u0, interior_mask = load_data(bid)

        axes[row, 0].imshow(u0, cmap="hot", vmin=0, vmax=25)
        axes[row, 0].set_title(f"Building {bid} — Initial conditions")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(interior_mask, cmap="gray")
        axes[row, 1].set_title(f"Building {bid} — Interior mask")
        axes[row, 1].axis("off")

    plt.tight_layout()
    out = join(REPORT_DIR, "task1_input_visualization.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Saved: {out}\n")

def task2(building_ids):
    print("TASK 2:")

    N = 20
    ids_subset = building_ids[:N]

    start = perf_counter()
    save_time = 0
    for bid in ids_subset:
        u0, mask = load_data(bid)
        u = jacobi(u0, mask)

        save_start = perf_counter()
        np.save(join(REPORT_DIR, f"{bid}_result.npy"), u)
        s = summary_stats(u, mask)
        print(f"{bid}, " + ", ".join(f"{s[k]:.4f}" for k in STAT_KEYS))
        save_time += perf_counter() - save_start
    elapsed = perf_counter() - start - save_time

    print(f"Time for {N} buildings:  {elapsed:.2f} s")
    print(f"Time per building:       {elapsed/N:.2f} s")
    print(f"Estimated total (4571):  {elapsed/N * 4571 / 60:.1f} min")

def task3(building_ids):
    print("TASK 3:")

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    for row, bid in enumerate(building_ids[:3]):
        u0, mask = load_data(bid)
        result_path = join(REPORT_DIR, f"{bid}_result.npy")
        if not exists(result_path):
            u = jacobi(u0, mask)
            np.save(result_path, u)
        else:
            u = np.load(result_path)

        axes[row, 0].imshow(u0, cmap="hot", vmin=0, vmax=25)
        axes[row, 0].set_title(f"Building {bid} — Before (initial)")
        axes[row, 0].axis("off")

        im = axes[row, 1].imshow(u, cmap="hot", vmin=0, vmax=25)
        axes[row, 1].set_title(f"Building {bid} — After (converged)")
        axes[row, 1].axis("off")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Temperature (°C)")
    out = join(REPORT_DIR, "task3_simulation_results.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Saved: {out}\n")

def chunking(bids, n_workers):
    """Takes a list of building_ids and then dividing it to n_workers, with the last worker taking the reminaing tasks"""
    size = len(bids) // n_workers
    chunks = [bids[i * size: (i + 1) * size] for i in range(n_workers - 1)]
    chunks.append(bids[(n_workers - 1) * size:])
    return chunks

def process_chunk(bids):
    """Process a list of building_ids"""
    results = []
    for bid in bids:
        u0, mask = load_data(bid)
        u = jacobi(u0, mask)
        results.append((bid, summary_stats(u, mask)))
    return results

def run_parallel(bids, n_workers):
    chunks = chunking(bids, n_workers)
    with Pool(processes=n_workers) as pool:
        nested = pool.map(process_chunk, chunks)
    return [item for sub in nested for item in sub]

def task4(building_ids, N=1):
    import cProfile
    import pstats
    import io
    print("TASK 4:")

    out_txt = join(REPORT_DIR, "task4_profile.txt")
    with open(out_txt, "w") as f:
        for bid in building_ids[:N]:
            u0, mask = load_data(bid)
            pr = cProfile.Profile()
            pr.enable()
            jacobi(u0, mask)
            pr.disable()

            stream = io.StringIO()
            ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
            ps.print_stats(20)

            f.write(f"=== Building {bid} ===\n")
            f.write(stream.getvalue())
            f.write("\n")

    print(f"Profile saved: {out_txt}\n")

def task5(building_ids, N=50, max_workers=16):
    print(f"TASK 5 (N={N}, max_workers={max_workers}):")

    bids = building_ids[:N]
    worker_counts = sorted({w for w in [1, 2, 4, 8, 16] if w <= max_workers})

    times = {}
    first_results = None

    for w in worker_counts:
        t0 = perf_counter()
        results = run_parallel(bids, w)
        times[w] = perf_counter() - t0
        if first_results is None:
            first_results = results
        print(f"  workers={w:2d}  time={times[w]:.2f}s")

    t1 = times[1]
    print(f"\n{'Workers':>8} {'Time(s)':>10} {'Speedup':>10}")
    speedups = []
    for w in worker_counts:
        s = t1 / times[w]
        speedups.append(s)
        print(f"{w:>8} {times[w]:>10.2f} {s:>10.3f}")

    best_s = max(speedups)
    best_w = worker_counts[speedups.index(best_s)]
    if best_w > 1:
        p = (1 / best_s - 1) / (1 / best_w - 1)
        p = max(0.0, min(1.0, p))
        max_s = 1 / (1 - p) if p < 1 else float("inf")
        print(f"\nAmdahl: parallel fraction p ≈ {p:.3f}")
        print(f"Theoretical max speedup: {max_s:.2f}x")
    else:
        p = None

    t_best = min(times.values())
    print(f"Estimated time for all 4571 buildings: {t_best / N * 4571 / 60:.1f} min")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(worker_counts, speedups, "o-", label="Measured speedup")
    ax.plot(worker_counts, worker_counts, "--", color="gray", label="Ideal (linear)")
    if p is not None:
        amdahl = [1 / ((1 - p) + p / w) for w in worker_counts]
        ax.plot(worker_counts, amdahl, "r--", label=f"Amdahl (p={p:.2f})")
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Static scheduling speedup (N={N} buildings)")
    ax.legend()
    ax.grid(True)
    out = join(REPORT_DIR, "task5_speedup.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Speedup plot saved: {out}")

    out_csv = join(REPORT_DIR, "task5_results.csv")
    with open(out_csv, "w") as f:
        f.write("building_id, " + ", ".join(STAT_KEYS) + "\n")
        for bid, stats in first_results:
            f.write(f"{bid}, " + ", ".join(f"{stats[k]:.4f}" for k in STAT_KEYS) + "\n")
    print(f"CSV results saved:  {out_csv}\n")

if __name__ == "__main__":
    with open(join(LOAD_DIR, "building_ids.txt")) as f:
        building_ids = f.read().splitlines()

    task1(building_ids)
    task2(building_ids)
    task3(building_ids)
    task4(building_ids)
    task5(building_ids, max_workers=8)