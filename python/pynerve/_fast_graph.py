"""Fast graph operations: connected components, minimum spanning tree."""

from __future__ import annotations

import numpy as np

from ._validation import validate_nonnegative_int, validate_points


def _validate_edges(edges: np.ndarray, n_vertices: int) -> np.ndarray:
    array = np.asarray(edges)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"edges must have shape (m, 2), got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"edges must contain integer vertex indices, got dtype {array.dtype}")
    array = array.astype(np.int64, copy=False)
    if array.min() < 0 or array.max() >= n_vertices:
        raise ValueError(
            f"edge indices are out of bounds [0, {n_vertices - 1}], got min {array.min()} max {array.max()}"
        )
    return array


def _compute_connected_components(n_vertices: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(n_vertices, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for e in edges:
        union(e[0], e[1])
    return parent


def connected_components_fast(edges: np.ndarray, n_vertices: int) -> np.ndarray:
    n_vertices = validate_nonnegative_int(n_vertices, "n_vertices")
    edges = _validate_edges(edges, n_vertices)
    if edges.size == 0:
        return np.arange(n_vertices, dtype=np.int64)

    parent = _compute_connected_components(n_vertices, edges)
    labels = np.array([_find_root(parent, i) for i in range(n_vertices)])

    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}

    return np.array([label_map[label] for label in labels], dtype=np.int64)


def _find_root(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def minimum_spanning_tree_fast(points: np.ndarray, edges: np.ndarray | None = None) -> np.ndarray:
    points = validate_points(points)
    n = points.shape[0]
    if n <= 1:
        return np.empty((0, 2), dtype=np.int64)

    if edges is None:
        i_idx, j_idx = np.triu_indices(n, k=1)
        edges_arr = np.column_stack([i_idx, j_idx]).astype(np.int64)
    else:
        edges_arr = edges
    edges_arr = _validate_edges(edges_arr, n)
    if edges_arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    weights = np.linalg.norm(points[edges_arr[:, 0]] - points[edges_arr[:, 1]], axis=1)
    sorted_idx = np.argsort(weights)
    sorted_edges = edges_arr[sorted_idx]

    parent = np.arange(n, dtype=np.int64)

    mst_edges: list[np.ndarray] = []
    for e in sorted_edges:
        root_u = _find_root(parent, int(e[0]))
        root_v = _find_root(parent, int(e[1]))
        if root_u != root_v:
            mst_edges.append(e)
            parent[root_v] = root_u

        if len(mst_edges) == n - 1:
            break

    return np.asarray(mst_edges, dtype=np.int64).reshape(-1, 2)
