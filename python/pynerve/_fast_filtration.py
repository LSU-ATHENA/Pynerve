"""Filtration construction helpers."""

from __future__ import annotations

import numpy as np

from ._fast_simplices import enumerate_simplices_fast
from ._validation import validate_nonnegative_int


def sort_filtration_fast(
    simplices: np.ndarray, filtration_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    simplices = np.asarray(simplices)
    filtration_values = np.asarray(filtration_values, dtype=float)
    if simplices.shape[0] != filtration_values.shape[0]:
        raise ValueError("simplices and filtration_values must have matching lengths")
    if not np.isfinite(filtration_values).all():
        raise ValueError("filtration_values must be finite")
    sorted_indices = np.argsort(filtration_values, kind="mergesort")
    sorted_filtration = filtration_values[sorted_indices]
    sorted_simplices = simplices[sorted_indices]

    return sorted_indices, sorted_simplices, sorted_filtration


def vietoris_rips_filtration_fast(
    points: np.ndarray, max_dist: float, max_dim: int = 2
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    max_dim = validate_nonnegative_int(max_dim, "max_dim")
    simplices = enumerate_simplices_fast(points, max_dist, max_dim)

    filtration_values = [np.zeros(len(simplices[0]), dtype=float)]

    if len(simplices) > 1:
        edges = simplices[1]
        diff = points[edges[:, 0]] - points[edges[:, 1]]
        edge_dists = np.sqrt(np.sum(diff**2, axis=1))
        filtration_values.append(edge_dists)

    for d in range(2, len(simplices)):
        d_simplices = simplices[d]
        d_filtration = np.empty(len(d_simplices), dtype=float)

        for si, simplex in enumerate(d_simplices):
            max_sq = 0.0
            nv = len(simplex)
            for i in range(nv):
                vi = simplex[i]
                for j in range(i + 1, nv):
                    vj = simplex[j]
                    dist_sq = np.sum((points[vi] - points[vj]) ** 2)
                    max_sq = max(max_sq, dist_sq)
            d_filtration[si] = np.sqrt(max_sq)

        filtration_values.append(d_filtration)

    return simplices, filtration_values
