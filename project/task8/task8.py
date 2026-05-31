from __future__ import annotations

import argparse
import csv
import sys
from os import makedirs
from os.path import abspath, dirname, join
from time import perf_counter

import numpy as np

sys.path.insert(0, dirname(abspath(__file__)))
from heat_cpu import (
    DEFAULT_LOAD_DIR,
    MAX_ITER,
    STAT_KEYS,
    load_data,
    read_building_ids,
    select_building_ids,
    summary_stats_numpy,
)

from task8_numba import jacobi_numba


def parse_args() -> argparse.Namespace:
    default_report_dir = join(dirname(abspath(__file__)), "report")
    parser = argparse.ArgumentParser(
        description="Task 8: Numba CUDA Jacobi solver on the DTU HPC project."
    )
    parser.add_argument("--load-dir", default=DEFAULT_LOAD_DIR)
    parser.add_argument("--report-dir", default=default_report_dir)
    parser.add_argument("--n-buildings", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument(
        "--sample-mode", choices=["head", "spread"], default="head"
    )
    return parser.parse_args()


def warm_up() -> None:
    u = np.zeros((8, 8), dtype=np.float64)
    mask = np.ones((6, 6), dtype=np.bool_)
    jacobi_numba(u, mask, max_iter=2)


def main() -> None:
    args = parse_args()
    makedirs(args.report_dir, exist_ok=True)

    all_ids = read_building_ids(args.load_dir)
    building_ids = select_building_ids(all_ids, args.n_buildings, args.sample_mode)

    print("Warming up CUDA kernel...", flush=True)
    warm_up()
    print("Warm-up done.", flush=True)

    results: list[dict[str, object]] = []
    total_start = perf_counter()

    for bid in building_ids:
        u0, interior_mask = load_data(args.load_dir, bid)

        start = perf_counter()
        u = jacobi_numba(u0, interior_mask, args.max_iter)
        solver_seconds = perf_counter() - start

        stats = summary_stats_numpy(u, interior_mask)
        results.append(
            {
                "building_id": bid,
                "solver_seconds": solver_seconds,
                **stats,
            }
        )
        print(
            f"{bid}: {solver_seconds:.3f}s  mean={stats['mean_temp']:.2f}°C",
            flush=True,
        )

    total_seconds = perf_counter() - total_start

    # CSV output
    out_csv = join(args.report_dir, "task8_results.csv")
    fieldnames = ["building_id", "solver_seconds", *STAT_KEYS]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})

    mean_solver = sum(r["solver_seconds"] for r in results) / len(results)
    n_total = len(all_ids)
    est_all = mean_solver * n_total

    notes_path = join(args.report_dir, "task8_notes.txt")
    with open(notes_path, "w", encoding="utf-8") as fh:
        fh.write("Task 8 – Numba CUDA Jacobi\n")
        fh.write("==========================\n\n")
        fh.write(f"Buildings tested : {len(building_ids)}\n")
        fh.write(f"Max iterations   : {args.max_iter}\n")
        fh.write(f"Total wall time  : {total_seconds:.2f} s\n")
        fh.write(f"Mean solver time : {mean_solver:.4f} s/building\n")
        fh.write(
            f"Estimated time for all {n_total} buildings: "
            f"{est_all / 60:.1f} min ({est_all:.0f} s)\n"
        )

    print(f"\nResults saved : {out_csv}")
    print(f"Notes saved   : {notes_path}")
    print(f"Mean solver   : {mean_solver:.4f} s/building")
    print(
        f"Estimated all : {est_all / 60:.1f} min for {n_total} buildings"
    )


if __name__ == "__main__":
    main()
