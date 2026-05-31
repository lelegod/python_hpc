from __future__ import annotations

import argparse
import csv
from pathlib import Path
from os.path import abspath, dirname, join

from heat_cpu import STAT_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge task 11 job-array shard outputs into one CSV and summary."
    )
    parser.add_argument("--report-dir", default=join(dirname(abspath(__file__)), "report"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    shard_paths = sorted(report_dir.glob("task11_array_*_results.csv"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard result CSVs found in {report_dir}")

    rows: list[dict[str, str]] = []
    for path in shard_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)

    rows.sort(key=lambda row: row["building_id"])

    merged_csv = report_dir / "task11_array_merged_results.csv"
    fieldnames = ["building_id", "iterations", "solver_seconds", *STAT_KEYS]
    with merged_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})

    solver_seconds = [float(row["solver_seconds"]) for row in rows]
    iterations = [int(row["iterations"]) for row in rows]

    summary_path = report_dir / "task11_array_merged_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Task 11 merged array summary\n")
        handle.write("============================\n\n")
        handle.write(f"number of shard files: {len(shard_paths)}\n")
        handle.write(f"total buildings: {len(rows)}\n")
        handle.write(f"mean solver time per building: {sum(solver_seconds) / len(solver_seconds):.4f} s\n")
        handle.write(f"total solver time across buildings: {sum(solver_seconds):.4f} s\n")
        handle.write(f"total Jacobi iterations: {sum(iterations)}\n\n")
        handle.write("Aggregate statistics across buildings\n")
        handle.write("------------------------------------\n")
        for key in STAT_KEYS:
            values = [float(row[key]) for row in rows]
            handle.write(
                f"{key}: mean={sum(values) / len(values):.4f}, "
                f"min={min(values):.4f}, max={max(values):.4f}\n"
            )

    print(f"Merged CSV saved: {merged_csv}")
    print(f"Merged summary saved: {summary_path}")


if __name__ == "__main__":
    main()
