import argparse
import csv
from multiprocessing import Pool, cpu_count, freeze_support
from os import makedirs
from os.path import abspath, dirname, join
from time import perf_counter

import matplotlib.pyplot as plt

from heat_cpu import (
    ABS_TOL,
    DEFAULT_LOAD_DIR,
    HAS_NUMBA,
    MAX_ITER,
    STAT_KEYS,
    chunk_building_ids,
    read_building_ids,
    require_numba,
    select_building_ids,
    solve_building,
    warm_up_numba,
)

# Small experiment script for tasks 6 and 7.
WORKER_CONFIG = {
    "load_dir": DEFAULT_LOAD_DIR,
    "engine": "numpy",
    "max_iter": MAX_ITER,
    "atol": ABS_TOL,
}


def parse_args():
    default_report_dir = join(dirname(abspath(__file__)), "report")
    parser = argparse.ArgumentParser(
        description="Task 6/7 CPU optimization experiments for the DTU HPC project."
    )
    parser.add_argument("--load-dir", default=DEFAULT_LOAD_DIR)
    parser.add_argument("--report-dir", default=default_report_dir)
    parser.add_argument("--n-buildings", type=int, default=50)
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 16))
    parser.add_argument("--worker-counts", default="")
    parser.add_argument("--scheduler", choices=["static", "dynamic"], default="dynamic")
    parser.add_argument("--engine", choices=["numpy", "numba"], default="numba")
    parser.add_argument("--sample-mode", choices=["head", "spread"], default="spread")
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--dynamic-chunksizes", default="1,2,4,8")
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--atol", type=float, default=ABS_TOL)
    return parser.parse_args()


def resolve_worker_counts(max_workers, worker_counts_arg):
    if max_workers <= 0:
        raise ValueError("--workers must be positive.")

    if worker_counts_arg.strip():
        counts = sorted({int(part) for part in worker_counts_arg.split(",") if part.strip()})
    else:
        counts = sorted({count for count in [1, 2, 4, 8, 16, max_workers] if count <= max_workers})
    return [count for count in counts if count > 0]


def parse_chunksizes(chunksizes_arg):
    values = sorted({int(part) for part in chunksizes_arg.split(",") if part.strip()})
    return [value for value in values if value > 0]


def init_worker(load_dir, engine, max_iter, atol):
    WORKER_CONFIG["load_dir"] = load_dir
    WORKER_CONFIG["engine"] = engine
    WORKER_CONFIG["max_iter"] = max_iter
    WORKER_CONFIG["atol"] = atol

    if engine == "numba":
        warm_up_numba()


def worker_solve_building(building_id):
    result = solve_building(
        WORKER_CONFIG["load_dir"],
        building_id,
        WORKER_CONFIG["engine"],
        max_iter=WORKER_CONFIG["max_iter"],
        atol=WORKER_CONFIG["atol"],
    )
    return {
        "building_id": result.building_id,
        "iterations": result.iterations,
        "solver_seconds": result.solver_seconds,
        **result.stats,
    }


def worker_solve_chunk(building_ids):
    return [worker_solve_building(building_id) for building_id in building_ids]


def run_task5_static_numpy(
    building_ids,
    load_dir,
    n_workers,
    max_iter,
    atol,
):
    """Static NumPy baseline matching the task 5 chunking/timing structure."""
    chunks = chunk_building_ids(building_ids, n_workers)

    start = perf_counter()
    with Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(load_dir, "numpy", max_iter, atol),
    ) as pool:
        nested = pool.map(worker_solve_chunk, chunks)

    results = [item for chunk in nested for item in chunk]
    wall_seconds = perf_counter() - start
    results.sort(key=lambda item: item["building_id"])
    return results, wall_seconds


def run_schedule(
    building_ids,
    load_dir,
    engine,
    scheduler,
    n_workers,
    chunksize,
    max_iter,
    atol,
):
    if engine == "numpy" and scheduler == "static":
        return run_task5_static_numpy(
            building_ids,
            load_dir,
            n_workers,
            max_iter,
            atol,
        )

    if n_workers == 1:
        start = perf_counter()
        init_worker(load_dir, engine, max_iter, atol)
        if scheduler == "static":
            chunks = chunk_building_ids(building_ids, 1)
            results = [item for chunk in chunks for item in worker_solve_chunk(chunk)]
        elif scheduler == "dynamic":
            results = [worker_solve_building(building_id) for building_id in building_ids]
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")
        results.sort(key=lambda item: item["building_id"])
        return results, perf_counter() - start

    start = perf_counter()
    with Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(load_dir, engine, max_iter, atol),
    ) as pool:
        if scheduler == "static":
            chunks = chunk_building_ids(building_ids, n_workers)
            nested = pool.map(worker_solve_chunk, chunks)
            results = [item for chunk in nested for item in chunk]
        elif scheduler == "dynamic":
            iterator = pool.imap_unordered(
                worker_solve_building,
                building_ids,
                chunksize=chunksize,
            )
            results = list(iterator)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

    wall_seconds = perf_counter() - start
    results.sort(key=lambda item: item["building_id"])
    return results, wall_seconds


def benchmark_single_building(load_dir, building_id, max_iter, atol):
    timings = {}
    for engine in ["numpy", "numba"]:
        if engine == "numba" and not HAS_NUMBA:
            continue
        if engine == "numba":
            warm_up_numba()
        result = solve_building(load_dir, building_id, engine, max_iter=max_iter, atol=atol)
        timings[engine] = {
            "solver_seconds": result.solver_seconds,
            "iterations": float(result.iterations),
            **result.stats,
        }
    return timings


def benchmark_suite(
    building_ids,
    args,
    worker_counts,
):
    configurations = [("numpy", "static"), ("numpy", "dynamic")]
    if HAS_NUMBA:
        configurations.append(("numba", "dynamic"))

    rows = []
    for engine, scheduler in configurations:
        for n_workers in worker_counts:
            results, wall_seconds = run_schedule(
                building_ids,
                args.load_dir,
                engine,
                scheduler,
                n_workers,
                args.chunksize,
                args.max_iter,
                args.atol,
            )
            mean_solver_seconds = sum(row["solver_seconds"] for row in results) / len(results)
            total_solver_seconds = sum(row["solver_seconds"] for row in results)
            rows.append(
                {
                    "engine": engine,
                    "scheduler": scheduler,
                    "workers": n_workers,
                    "chunksize": args.chunksize if scheduler == "dynamic" else 0,
                    "wall_seconds": wall_seconds,
                    "mean_solver_seconds": mean_solver_seconds,
                    "total_solver_seconds": total_solver_seconds,
                }
            )
    return rows


def benchmark_chunksizes(
    building_ids,
    args,
    chunksizes,
):
    if args.scheduler != "dynamic":
        return []
    if args.engine == "numba" and not HAS_NUMBA:
        return []

    rows = []
    for chunksize in chunksizes:
        _, wall_seconds = run_schedule(
            building_ids,
            args.load_dir,
            args.engine,
            args.scheduler,
            args.workers,
            chunksize,
            args.max_iter,
            args.atol,
        )
        rows.append(
            {
                "engine": args.engine,
                "scheduler": args.scheduler,
                "workers": args.workers,
                "chunksize": chunksize,
                "wall_seconds": wall_seconds,
            }
        )
    return rows


def write_results_csv(path, rows):
    fieldnames = ["building_id", "iterations", "solver_seconds", *STAT_KEYS]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def write_benchmark_csv(path, rows):
    fieldnames = [
        "engine",
        "scheduler",
        "workers",
        "chunksize",
        "wall_seconds",
        "mean_solver_seconds",
        "total_solver_seconds",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_chunksize_csv(path, rows):
    fieldnames = ["engine", "scheduler", "workers", "chunksize", "wall_seconds"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_speedup_plot(path, benchmark_rows):
    baseline_by_config = {}
    grouped = {}
    for row in benchmark_rows:
        key = (row["engine"], row["scheduler"])
        grouped.setdefault(key, []).append(row)
        if row["workers"] == 1:
            baseline_by_config[key] = row["wall_seconds"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["workers"])
        baseline = baseline_by_config[key]
        speedups = [baseline / row["wall_seconds"] for row in rows]
        label = f"{key[0]} + {key[1]}"
        ax.plot([row["workers"] for row in rows], speedups, "o-", label=label)

    max_workers = max(row["workers"] for row in benchmark_rows)
    ideal_workers = list(range(1, max_workers + 1))
    ax.plot(ideal_workers, ideal_workers, "--", color="gray", label="ideal")
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Speedup")
    ax.set_title("Task 6 and 7 speedup comparison")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_notes(
    path,
    selected_ids,
    primary_results,
    primary_wall_seconds,
    benchmark_rows,
    single_building,
    chunk_rows,
    args,
):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Task 6/7 benchmark summary\n")
        handle.write("=========================\n\n")
        handle.write(f"Subset size: {len(selected_ids)}\n")
        handle.write(f"Sample mode: {args.sample_mode}\n")
        handle.write(f"Primary configuration: engine={args.engine}, scheduler={args.scheduler}\n")
        handle.write(f"Primary workers: {args.workers}\n")
        handle.write(f"Primary chunksize: {args.chunksize}\n")
        handle.write(f"Primary wall time: {primary_wall_seconds:.4f} s\n")
        handle.write(
            "Primary mean solver time per building: "
            f"{sum(row['solver_seconds'] for row in primary_results) / len(primary_results):.4f} s\n\n"
        )

        handle.write("Single-building comparison\n")
        handle.write("--------------------------\n")
        for engine, values in single_building.items():
            handle.write(
                f"{engine}: solver_seconds={values['solver_seconds']:.4f}, "
                f"iterations={int(values['iterations'])}, "
                f"mean_temp={values['mean_temp']:.4f}, "
                f"std_temp={values['std_temp']:.4f}\n"
            )

        handle.write("\nWorker sweep\n")
        handle.write("-----------\n")
        for row in sorted(benchmark_rows, key=lambda item: (item["engine"], item["scheduler"], item["workers"])):
            handle.write(
                f"{row['engine']} + {row['scheduler']}: workers={row['workers']}, "
                f"wall_seconds={row['wall_seconds']:.4f}, "
                f"mean_solver_seconds={row['mean_solver_seconds']:.4f}\n"
            )

        if chunk_rows:
            handle.write("\nChunksize sweep\n")
            handle.write("---------------\n")
            for row in chunk_rows:
                handle.write(
                    f"chunksize={row['chunksize']}, workers={row['workers']}, "
                    f"wall_seconds={row['wall_seconds']:.4f}\n"
                )


def main():
    args = parse_args()
    if args.engine == "numba":
        require_numba()

    makedirs(args.report_dir, exist_ok=True)
    worker_counts = resolve_worker_counts(args.workers, args.worker_counts)
    chunk_sizes = parse_chunksizes(args.dynamic_chunksizes)

    all_building_ids = read_building_ids(args.load_dir)
    selected_ids = select_building_ids(all_building_ids, args.n_buildings, args.sample_mode)

    if HAS_NUMBA:
        warm_up_numba()

    single_building = benchmark_single_building(
        args.load_dir,
        selected_ids[0],
        args.max_iter,
        args.atol,
    )

    benchmark_rows = benchmark_suite(selected_ids, args, worker_counts)
    primary_results, primary_wall_seconds = run_schedule(
        selected_ids,
        args.load_dir,
        args.engine,
        args.scheduler,
        args.workers,
        args.chunksize,
        args.max_iter,
        args.atol,
    )
    chunk_rows = benchmark_chunksizes(selected_ids, args, chunk_sizes)

    results_csv = join(args.report_dir, "task6_task7_results.csv")
    benchmark_csv = join(args.report_dir, "task6_task7_benchmarks.csv")
    chunksize_csv = join(args.report_dir, "task6_task7_chunksizes.csv")
    plot_path = join(args.report_dir, "task6_task7_speedup.png")
    notes_path = join(args.report_dir, "task6_task7_notes.txt")

    write_results_csv(results_csv, primary_results)
    write_benchmark_csv(benchmark_csv, benchmark_rows)
    if chunk_rows:
        write_chunksize_csv(chunksize_csv, chunk_rows)
    save_speedup_plot(plot_path, benchmark_rows)
    write_notes(
        notes_path,
        selected_ids,
        primary_results,
        primary_wall_seconds,
        benchmark_rows,
        single_building,
        chunk_rows,
        args,
    )

    print(f"Primary results saved: {results_csv}")
    print(f"Benchmark results saved: {benchmark_csv}")
    if chunk_rows:
        print(f"Chunksize sweep saved: {chunksize_csv}")
    print(f"Speedup plot saved: {plot_path}")
    print(f"Notes saved: {notes_path}")


if __name__ == "__main__":
    freeze_support()
    main()
