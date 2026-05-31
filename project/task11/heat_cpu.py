from __future__ import annotations

from dataclasses import dataclass
from os import environ
from os.path import join
from time import perf_counter

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only when numba is unavailable
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


DEFAULT_LOAD_DIR = environ.get(
    "DTU_HPC_LOAD_DIR",
    "/dtu/projects/02613_2025/data/modified_swiss_dwellings/",
)
MAX_ITER = 20_000
ABS_TOL = 1e-4
STAT_KEYS = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]


@dataclass(frozen=True)
class SolveResult:
    building_id: str
    stats: dict[str, float]
    solver_seconds: float
    iterations: int


def load_data(load_dir: str, bid: str) -> tuple[np.ndarray, np.ndarray]:
    size = 512
    u = np.zeros((size + 2, size + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(np.bool_)
    return u, interior_mask


def read_building_ids(load_dir: str) -> list[str]:
    with open(join(load_dir, "building_ids.txt"), encoding="utf-8") as handle:
        return handle.read().splitlines()


def select_building_ids(building_ids: list[str], n_buildings: int, mode: str) -> list[str]:
    if n_buildings <= 0:
        raise ValueError("n_buildings must be positive.")
    if n_buildings > len(building_ids):
        raise ValueError(
            f"Requested {n_buildings} buildings, but only {len(building_ids)} are available."
        )

    if mode == "head":
        return building_ids[:n_buildings]

    if mode != "spread":
        raise ValueError(f"Unsupported sample mode: {mode}")

    if n_buildings == 1:
        return [building_ids[0]]

    positions = np.linspace(0, len(building_ids) - 1, num=n_buildings)
    seen = set()
    selected = []
    for pos in positions:
        idx = int(round(float(pos)))
        if idx not in seen:
            selected.append(building_ids[idx])
            seen.add(idx)

    if len(selected) < n_buildings:
        for idx, bid in enumerate(building_ids):
            if idx not in seen:
                selected.append(bid)
                if len(selected) == n_buildings:
                    break

    return selected


def chunk_building_ids(building_ids: list[str], n_workers: int) -> list[list[str]]:
    if n_workers <= 0:
        raise ValueError("n_workers must be positive.")
    size = len(building_ids) // n_workers
    chunks = [building_ids[i * size : (i + 1) * size] for i in range(n_workers - 1)]
    chunks.append(building_ids[(n_workers - 1) * size :])
    return [chunk for chunk in chunks if chunk]


def jacobi_numpy(
    u: np.ndarray,
    interior_mask: np.ndarray,
    max_iter: int = MAX_ITER,
    atol: float = ABS_TOL,
) -> tuple[np.ndarray, int]:
    u = np.array(u, copy=True)
    interior = u[1:-1, 1:-1]

    for iteration in range(max_iter):
        u_new = 0.25 * (
            u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1]
        )
        interior_new = u_new[interior_mask]
        delta = np.abs(interior[interior_mask] - interior_new).max()
        interior[interior_mask] = interior_new
        if delta < atol:
            return u, iteration + 1

    return u, max_iter


def summary_stats_numpy(u: np.ndarray, interior_mask: np.ndarray) -> dict[str, float]:
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        "mean_temp": float(u_interior.mean()),
        "std_temp": float(u_interior.std()),
        "pct_above_18": float(np.sum(u_interior > 18.0) / u_interior.size * 100.0),
        "pct_below_15": float(np.sum(u_interior < 15.0) / u_interior.size * 100.0),
    }


@njit(cache=True)
def _jacobi_numba_core(
    u_initial: np.ndarray,
    interior_mask: np.ndarray,
    max_iter: int,
    atol: float,
) -> tuple[np.ndarray, int]:
    u_old = np.copy(u_initial)
    u_new = np.copy(u_initial)
    n_rows, n_cols = u_old.shape

    for iteration in range(max_iter):
        delta = 0.0
        for i in range(1, n_rows - 1):
            for j in range(1, n_cols - 1):
                if interior_mask[i - 1, j - 1]:
                    updated = 0.25 * (
                        u_old[i, j - 1]
                        + u_old[i, j + 1]
                        + u_old[i - 1, j]
                        + u_old[i + 1, j]
                    )
                    diff = abs(u_old[i, j] - updated)
                    if diff > delta:
                        delta = diff
                    u_new[i, j] = updated
                else:
                    u_new[i, j] = u_old[i, j]

        if delta < atol:
            return u_new, iteration + 1

        u_old, u_new = u_new, u_old

    return u_old, max_iter


def jacobi_numba(
    u: np.ndarray,
    interior_mask: np.ndarray,
    max_iter: int = MAX_ITER,
    atol: float = ABS_TOL,
) -> tuple[np.ndarray, int]:
    require_numba()
    return _jacobi_numba_core(u, interior_mask, max_iter, atol)


@njit(cache=True)
def _summary_stats_numba_core(u: np.ndarray, interior_mask: np.ndarray) -> tuple[float, float, float, float]:
    total = 0.0
    total_sq = 0.0
    count = 0
    above_18 = 0
    below_15 = 0

    n_rows, n_cols = interior_mask.shape
    for i in range(n_rows):
        for j in range(n_cols):
            if interior_mask[i, j]:
                value = u[i + 1, j + 1]
                total += value
                total_sq += value * value
                count += 1
                if value > 18.0:
                    above_18 += 1
                if value < 15.0:
                    below_15 += 1

    mean_temp = total / count
    variance = total_sq / count - mean_temp * mean_temp
    if variance < 0.0:
        variance = 0.0
    std_temp = np.sqrt(variance)
    pct_above_18 = above_18 / count * 100.0
    pct_below_15 = below_15 / count * 100.0
    return mean_temp, std_temp, pct_above_18, pct_below_15


def summary_stats_numba(u: np.ndarray, interior_mask: np.ndarray) -> dict[str, float]:
    require_numba()
    mean_temp, std_temp, pct_above_18, pct_below_15 = _summary_stats_numba_core(
        u, interior_mask
    )
    return {
        "mean_temp": float(mean_temp),
        "std_temp": float(std_temp),
        "pct_above_18": float(pct_above_18),
        "pct_below_15": float(pct_below_15),
    }


def solve_building(
    load_dir: str,
    building_id: str,
    engine: str,
    max_iter: int = MAX_ITER,
    atol: float = ABS_TOL,
) -> SolveResult:
    u0, interior_mask = load_data(load_dir, building_id)

    start = perf_counter()
    if engine == "numpy":
        u, iterations = jacobi_numpy(u0, interior_mask, max_iter=max_iter, atol=atol)
        stats = summary_stats_numpy(u, interior_mask)
    elif engine == "numba":
        u, iterations = jacobi_numba(u0, interior_mask, max_iter=max_iter, atol=atol)
        stats = summary_stats_numba(u, interior_mask)
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    solver_seconds = perf_counter() - start
    return SolveResult(
        building_id=building_id,
        stats=stats,
        solver_seconds=solver_seconds,
        iterations=iterations,
    )


def warm_up_numba() -> None:
    require_numba()
    u = np.zeros((8, 8), dtype=np.float64)
    mask = np.ones((6, 6), dtype=np.bool_)
    solved, _ = jacobi_numba(u, mask, max_iter=2, atol=0.0)
    summary_stats_numba(solved, mask)


def require_numba() -> None:
    if not HAS_NUMBA:
        raise RuntimeError(
            "Numba is required for engine='numba'. Install it from requirements.txt first."
        )
