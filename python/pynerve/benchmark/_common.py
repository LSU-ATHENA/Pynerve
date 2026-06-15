from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .._compute_api import compute_persistence
from .._validation import validate_nonempty_string, validate_nonnegative_int, validate_positive_int

_BENCHMARK_RECOVERABLE_ERRORS = (ImportError, RuntimeError, ValueError)


def _benchmark_dataset(dataset: str, n_samples: int) -> np.ndarray:
    dataset = validate_nonempty_string(dataset, "dataset")
    n_samples = validate_positive_int(n_samples, "n_samples")
    rng = np.random.default_rng(0)
    loaders: dict[str, Callable[[int], np.ndarray]] = {
        "spheres": lambda n: np.asarray(rng.normal(size=(n, 4)), dtype=np.float64),
        "torus": lambda n: np.asarray(rng.uniform(size=(n, 3)), dtype=np.float64),
        "swiss_roll": lambda n: np.asarray(rng.normal(size=(n, 3)), dtype=np.float64),
    }
    try:
        return np.asarray(loaders[dataset](n_samples), dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {dataset}") from exc


def _compute_nerve_persistence(data: np.ndarray, max_dim: int) -> object:
    max_dim = validate_nonnegative_int(max_dim, "max_dim")
    return compute_persistence(data, max_dim=max_dim)


def _time_runs(n_runs: int, fn: Callable[[], object]) -> list[float]:
    n_runs = validate_positive_int(n_runs, "n_runs")
    timings: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return timings
