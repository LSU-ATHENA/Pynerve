"""Numba simplex kernels."""

from __future__ import annotations

import numpy as np

from ._numba_compat import njit


@njit(cache=True)
def numba_vr_edges(points: np.ndarray, max_dist: float) -> np.ndarray:
    if max_dist < 0.0:
        raise ValueError("max_dist must be non-negative")
    n = points.shape[0]
    dim = points.shape[1]

    max_edges = n * (n - 1) // 2
    edges = np.zeros((max_edges, 2), dtype=np.int64)
    edge_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = 0.0
            for k in range(dim):
                diff = points[i, k] - points[j, k]
                dist_sq += diff * diff

            if dist_sq <= max_dist * max_dist:
                edges[edge_count, 0] = i
                edges[edge_count, 1] = j
                edge_count += 1

    return edges[:edge_count]


@njit(cache=True)
def numba_triangle_enumeration(edges: np.ndarray, n_vertices: int) -> np.ndarray:
    if n_vertices < 3 or edges.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)

    max_triangles = n_vertices * (n_vertices - 1) * (n_vertices - 2) // 6
    triangles = np.zeros((max_triangles, 3), dtype=np.int64)
    tri_count = 0

    for i in range(edges.shape[0]):
        u, v = edges[i]

        u_neighbors = edges[edges[:, 0] == u, 1]
        u_neighbors = np.concatenate((u_neighbors, edges[edges[:, 1] == u, 0]))

        v_neighbors = edges[edges[:, 0] == v, 1]
        v_neighbors = np.concatenate((v_neighbors, edges[edges[:, 1] == v, 0]))

        for w in u_neighbors:
            if w in v_neighbors and w > v:
                triangles[tri_count] = [u, v, w]
                tri_count += 1

                if tri_count >= max_triangles:
                    return triangles[:tri_count]

    return triangles[:tri_count]


@njit(cache=True)
def numba_simplex_boundary(simplex: np.ndarray) -> np.ndarray:
    dim = len(simplex)
    if dim == 0:
        return np.empty((0, 0), dtype=simplex.dtype)
    boundary = np.zeros((dim, dim - 1), dtype=simplex.dtype)

    for i in range(dim):
        idx = 0
        for j in range(dim):
            if j != i:
                boundary[i, idx] = simplex[j]
                idx += 1

    return boundary
