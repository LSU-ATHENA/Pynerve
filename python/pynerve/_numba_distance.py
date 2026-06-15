"""Numba distance kernels."""

from __future__ import annotations

import numpy as np

from ._numba_compat import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def numba_pairwise_distances(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    dim = points.shape[1]
    dists = np.zeros((n, n), dtype=np.float64)

    for i in prange(n):
        for j in range(i + 1, n):
            dist_sq = 0.0
            for k in range(dim):
                diff = points[i, k] - points[j, k]
                dist_sq += diff * diff
            dists[i, j] = np.sqrt(dist_sq)
            dists[j, i] = dists[i, j]

    return dists


@njit(cache=True)
def numba_nearest_neighbors(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = points.shape[0]
    dim = points.shape[1]
    if k <= 0:
        return np.empty((n, 0), dtype=np.float64), np.empty((n, 0), dtype=np.int64)

    distances = np.full((n, k), np.inf, dtype=np.float64)
    indices = np.zeros((n, k), dtype=np.int64)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            dist_sq = 0.0
            for d in range(dim):
                diff = points[i, d] - points[j, d]
                dist_sq += diff * diff
            dist = np.sqrt(dist_sq)

            if dist < distances[i, k - 1]:
                idx = k - 2
                while idx >= 0 and dist < distances[i, idx]:
                    distances[i, idx + 1] = distances[i, idx]
                    indices[i, idx + 1] = indices[i, idx]
                    idx -= 1
                distances[i, idx + 1] = dist
                indices[i, idx + 1] = j

    return distances, indices
