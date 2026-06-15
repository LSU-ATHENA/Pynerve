"""Numba graph kernels."""

from __future__ import annotations

import numpy as np

from ._numba_compat import njit


@njit(cache=True)
def numba_connected_components(edges: np.ndarray, n_vertices: int) -> np.ndarray:
    if n_vertices < 0:
        raise ValueError(f"n_vertices must be non-negative, got {n_vertices}")
    parent = np.arange(n_vertices)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for i in range(edges.shape[0]):
        union(edges[i, 0], edges[i, 1])

    labels = np.zeros(n_vertices, dtype=np.int64)
    label_map: dict[int, int] = {}
    next_label = 0

    for i in range(n_vertices):
        root = find(i)
        if root not in label_map:
            label_map[root] = next_label
            next_label += 1
        labels[i] = label_map[root]

    return labels


@njit(cache=True)
def numba_mst_kruskal(edges: np.ndarray, weights: np.ndarray, n_vertices: int) -> np.ndarray:
    if n_vertices < 0:
        raise ValueError(f"n_vertices must be non-negative, got {n_vertices}")
    if n_vertices <= 1:
        return np.empty((0, 2), dtype=np.int64)
    if edges.shape[0] != weights.shape[0]:
        raise ValueError(
            f"edges and weights must have matching lengths, got {edges.shape[0]} and {weights.shape[0]}"
        )

    sorted_idx = np.argsort(weights)
    edges = edges[sorted_idx]

    parent = np.arange(n_vertices)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mst = np.zeros((n_vertices - 1, 2), dtype=np.int64)
    mst_count = 0

    for i in range(edges.shape[0]):
        u, v = edges[i]

        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            mst[mst_count] = [u, v]
            mst_count += 1
            parent[root_v] = root_u

        if mst_count == n_vertices - 1:
            break

    return mst[:mst_count]
