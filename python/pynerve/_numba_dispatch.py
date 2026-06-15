"""Dispatch and benchmark helpers for Numba kernels."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ._numba_distance import numba_pairwise_distances
from ._numba_graph import numba_connected_components
from ._numba_reduction import numba_column_reduction
from ._numba_representations import numba_betti_curve
from ._numba_simplices import numba_vr_edges
from ._validation import validate_positive_int

_OPERATIONS = {
    "pairwise_distances": numba_pairwise_distances,
    "vr_edges": numba_vr_edges,
    "column_reduction": numba_column_reduction,
    "betti_curve": numba_betti_curve,
    "connected_components": numba_connected_components,
}


def compute_with_numba(operation: str, *args: Any, **kwargs: Any) -> Any:
    if not isinstance(operation, str) or not operation:
        raise ValueError("operation must be a non-empty string")
    if operation not in _OPERATIONS:
        raise ValueError(f"Unknown operation: {operation}")

    return _OPERATIONS[operation](*args, **kwargs)


def benchmark_numba_vs_numpy(
    func_numba: Callable[..., Any],
    func_numpy: Callable[..., Any],
    setup_fn: Callable[..., Any],
    n_trials: int = 5,
) -> dict[str, Any]:
    if not callable(func_numba):
        raise TypeError("func_numba must be callable")
    if not callable(func_numpy):
        raise TypeError("func_numpy must be callable")
    if not callable(setup_fn):
        raise TypeError("setup_fn must be callable")
    n_trials = validate_positive_int(n_trials, "n_trials")
    data = setup_fn()
    if not isinstance(data, tuple):
        raise TypeError("setup_fn must return a tuple of positional arguments")

    data_copy = copy.deepcopy(data)
    func_numba(*data_copy)

    numba_times = []
    for _ in range(n_trials):
        data_copy = copy.deepcopy(data)
        start = time.perf_counter()
        result_numba = func_numba(*data_copy)
        end = time.perf_counter()
        numba_times.append(end - start)

    numpy_times = []
    for _ in range(n_trials):
        data_copy = copy.deepcopy(data)
        start = time.perf_counter()
        result_numpy = func_numpy(*data_copy)
        end = time.perf_counter()
        numpy_times.append(end - start)

    numba_mean = float(np.mean(numba_times))
    numpy_mean = float(np.mean(numpy_times))
    if (
        np.isfinite(numba_mean)
        and np.isfinite(numpy_mean)
        and numba_mean > 0.0
        and numpy_mean >= 0.0
    ):
        speedup = numpy_mean / numba_mean
    else:
        speedup = 1.0

    return {
        "numba_mean": numba_mean,
        "numpy_mean": numpy_mean,
        "speedup": speedup if np.isfinite(speedup) else 1.0,
        "results_match": bool(
            np.allclose(result_numba, result_numpy)  # pyright: ignore[reportPossiblyUnboundVariable]  # pyright: ignore[reportPossiblyUnboundVariable]
        ),
    }
