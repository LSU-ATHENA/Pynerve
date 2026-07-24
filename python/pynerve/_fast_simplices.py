"""Vectorized simplex construction and enumeration helpers."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ._fast_distance import pairwise_distances_fast
from ._validation import validate_max_dist, validate_nonnegative_int, validate_points


def _validate_simplex(simplex: np.ndarray) -> np.ndarray:
    array = np.asarray(simplex)
    if array.ndim != 1:
        raise ValueError("simplex must be a 1D array")
    if array.size == 0:
        raise ValueError("simplex must be non-empty")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("simplex must contain integer vertex indices")
    array = array.astype(np.int64, copy=False)
    if (array < 0).any():
        raise ValueError("simplex vertex indices must be non-negative")
    if np.unique(array).size != array.size:
        raise ValueError("simplex vertex indices must be unique")
    return array


def vr_edges_fast(
    points: np.ndarray, max_dist: float, return_dists: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    points = validate_points(points)
    _md = validate_max_dist(max_dist)
    assert _md is not None
    dists = pairwise_distances_fast(points)
    n = dists.shape[0]

    i_idx, j_idx = np.triu_indices(n, k=1)
    edge_dists = dists[i_idx, j_idx]

    mask = edge_dists <= _md
    edges = np.column_stack([i_idx[mask], j_idx[mask]])

    if return_dists:
        return edges, edge_dists[mask]
    return edges


def enumerate_simplices_fast(
    points: np.ndarray, max_dist: float, max_dim: int = 2
) -> list[np.ndarray]:
    points = validate_points(points)
    _md = validate_max_dist(max_dist)
    assert _md is not None
    max_dim = validate_nonnegative_int(max_dim, "max_dim")
    n = points.shape[0]
    simplices: list[np.ndarray] = [np.arange(n, dtype=np.int64).reshape(-1, 1)]
    if max_dim <= 0:
        return simplices

    edges_result = vr_edges_fast(points, _md)
    assert isinstance(edges_result, np.ndarray)
    edges = edges_result.astype(np.int64, copy=False)
    simplices.append(edges)
    if len(edges) == 0:
        return simplices

    adjacency = csr_matrix(
        (np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])), shape=(n, n)
    )
    adjacency = adjacency + adjacency.T

    for _dim in range(2, max_dim + 1):
        higher_simplices = []

        if len(simplices[-1]) > 0:
            prev_simplices = simplices[-1]

            for simplex in prev_simplices:
                common_neighbors = adjacency[simplex[0]].toarray().ravel().astype(bool)
                for v in simplex[1:]:
                    common_neighbors &= adjacency[v].toarray().ravel().astype(bool)

                for w in np.where(common_neighbors)[0]:
                    if w > max(simplex):
                        new_simplex = np.append(simplex, w)
                        higher_simplices.append(new_simplex)

        if len(higher_simplices) > 0:
            simplices.append(np.array(higher_simplices))
        else:
            break

    return simplices


def simplex_boundary_fast(simplex: np.ndarray) -> np.ndarray:
    simplex = _validate_simplex(simplex)
    dim = len(simplex)

    boundary = np.zeros((dim, dim - 1), dtype=simplex.dtype)

    for i in range(dim):
        boundary[i] = np.concatenate([simplex[:i], simplex[i + 1 :]])

    return boundary
