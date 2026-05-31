from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool, freeze_support
from os import environ, makedirs
from os.path import abspath, dirname, join
from time import perf_counter

from heat_cpu import (
    ABS_TOL,
    DEFAULT_LOAD_DIR,
    MAX_ITER,
    STAT_KEYS,
    read_building_ids,
    require_numba,
    solve_building,
    warm_up_numba,
)


WORKER_CONFIG = {
    "load_dir": DEFAULT_LOAD_DIR,
    "max_iter": MAX_ITER,
    "atol": ABS_TOL,
}


def parse_args() -> argparse.Namespace:
    default_report_dir = join(dirname(abspath(__file__)), "report")
    parser = argparse.ArgumentParser(
        description="Task 11 job-array worker using the fastest CPU JIT configuration."
    )
    parser.add_argument("--load-dir", default=DEFAULT_LOAD_DIR)
    parser.add_argument("--report-dir", default=default_report_dir)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--atol", type=float, default=ABS_TOL)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for testing. 0 means use the full dataset before sharding.",
    )
    return parser.parse_args()


def init_worker(load_dir: str, max_iter: int, atol: float) -> None:
    WORKER_CONFIG["load_dir"] = load_dir
    WORKER_CONFIG["max_iter"] = max_iter
    WORKER_CONFIG["atol"] = atol
    warm_up_numba()


def worker_solve_building(building_id: str) -> dict[str, object]:
    result = solve_building(
        WORKER_CONFIG["load_dir"],
        building_id,
        "numba",
        max_iter=WORKER_CONFIG["max_iter"],
        atol=WORKER_CONFIG["atol"],
    )
    return {
        "building_id": result.building_id,
        "iterations": result.iterations,
        "solver_seconds": result.solver_seconds,
        **result.stats,
    }


def shard_building_ids(building_ids: list[str], task_index: int, task_count: int) -> list[str]:
    if task_count <= 0:
        raise ValueError("task_count must be positive.")
    if task_index < 0 or task_index >= task_count:
        raise ValueError("task_index must be in [0, task_count).")
    return building_ids[task_index::task_count]


def run_subset(
    building_ids: list[str],
    load_dir: str,
    n_workers: int,
    chunksize: int,
    max_iter: int,
    atol: float,
) -> tuple[list[dict[str, object]], float]:
    if n_workers <= 0:
        raise ValueError("--workers must be positive.")

    start = perf_counter()
    if n_workers == 1:
        init_worker(load_dir, max_iter, atol)
        results = [worker_solve_building(building_id) for building_id in building_ids]
    else:
        with Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(load_dir, max_iter, atol),
        ) as pool:
            iterator = pool.imap_unordered(
                worker_solve_building,
                building_ids,
                chunksize=chunksize,
            )
            results = list(iterator)

    wall_seconds = perf_counter() - start
    results.sort(key=lambda item: item["building_id"])
    return results, wall_seconds


def write_results_csv(path: str, rows: list[dict[str, object]]) -> None:
    fieldnames = ["building_id", "iterations", "solver_seconds", *STAT_KEYS]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def write_summary(path: str, rows: list[dict[str, object]], wall_seconds: float, args: argparse.Namespace) -> None:
    total_solver_seconds = sum(float(row["solver_seconds"]) for row in rows)
    mean_solver_seconds = total_solver_seconds / len(rows) if rows else 0.0
    total_iterations = sum(int(row["iterations"]) for row in rows)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Task 11 array worker summary\n")
        handle.write("============================\n\n")
        handle.write("Configuration\n")
        handle.write("-------------\n")
        handle.write("engine: numba\n")
        handle.write("scheduler: dynamic\n")
        handle.write(f"workers: {args.workers}\n")
        handle.write(f"chunksize: {args.chunksize}\n")
        handle.write(f"task_index: {args.task_index}\n")
        handle.write(f"task_count: {args.task_count}\n")
        handle.write(f"dataset size for this shard: {len(rows)} buildings\n")
        handle.write(f"wall time: {wall_seconds:.4f} s\n")
        handle.write(f"mean solver time per building: {mean_solver_seconds:.4f} s\n")
        handle.write(f"total solver time across buildings: {total_solver_seconds:.4f} s\n")
        handle.write(f"total Jacobi iterations: {total_iterations}\n")


def main() -> None:
    args = parse_args()
    require_numba()
    makedirs(args.report_dir, exist_ok=True)

    env_jobindex = environ.get("LSB_JOBINDEX")
    env_jobcount = environ.get("LSB_JOBINDEX_END")

    task_index = args.task_index
    task_count = args.task_count

    if env_jobindex is not None and task_index == 0:
        task_index = int(env_jobindex) - 1
    if env_jobcount is not None and task_count == 0:
        task_count = int(env_jobcount)

    if task_count <= 0:
        raise ValueError(
            "task_count could not be determined. Pass --task-count or run inside an LSF job array."
        )
    args.task_index = task_index
    args.task_count = task_count

    building_ids = read_building_ids(args.load_dir)
    if args.limit > 0:
        building_ids = building_ids[: args.limit]
    shard_ids = shard_building_ids(building_ids, task_index, task_count)

    print(f"Running task 11 worker {task_index + 1}/{task_count} on {len(shard_ids)} buildings...")
    print("Configuration: engine=numba, scheduler=dynamic")
    print(f"Workers: {args.workers}")
    print(f"Chunksize: {args.chunksize}")

    results, wall_seconds = run_subset(
        shard_ids,
        args.load_dir,
        args.workers,
        args.chunksize,
        args.max_iter,
        args.atol,
    )

    stem = f"task11_array_{task_index + 1:03d}_of_{task_count:03d}"
    results_path = join(args.report_dir, f"{stem}_results.csv")
    summary_path = join(args.report_dir, f"{stem}_summary.txt")
    write_results_csv(results_path, results)
    write_summary(summary_path, results, wall_seconds, args)

    print(f"Results saved: {results_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    freeze_support()
    main()
