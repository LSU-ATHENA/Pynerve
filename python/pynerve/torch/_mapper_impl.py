"""Internal Mapper implementation and helper implementations."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any, cast

import torch
from torch import Tensor

from .._validation import validate_finite_scalar as _validate_finite_scalar
from .._validation import validate_finite_tensor as _validate_finite_tensor
from .._validation import validate_positive_int as _validate_positive_int

_FILTER_FUNCTIONS = {"pca_2d", "pca_1d", "eccentricity", "identity"}
_CLUSTERERS = {"dbscan", "single_linkage", "connected"}


def _mapper_python(
    point_cloud: Tensor,
    filter_function: str,
    cover_resolution: int,
    cover_overlap: float,
    clusterer: str,
    dbscan_eps: float,
    dbscan_min_samples: int,
    return_graph: bool,
) -> dict[str, Any]:
    """Pure Python implementation."""
    (
        cover_resolution,
        cover_overlap,
        dbscan_eps,
        dbscan_min_samples,
    ) = _validate_mapper_inputs(
        point_cloud, cover_resolution, cover_overlap, dbscan_eps, dbscan_min_samples
    )

    if filter_function == "pca_2d":
        filter_vals = _filter_pca_python(point_cloud, 2)
    elif filter_function == "pca_1d":
        filter_vals = _filter_pca_python(point_cloud, 1)
    elif filter_function == "eccentricity":
        filter_vals = _filter_eccentricity_python(point_cloud)
    elif filter_function == "identity":
        filter_vals = point_cloud[:, : min(2, point_cloud.shape[1])]
    else:
        raise ValueError(f"Unknown filter_function: {filter_function}")

    return _mapper_from_filter_values(
        point_cloud,
        filter_vals,
        cover_resolution,
        cover_overlap,
        clusterer,
        dbscan_eps,
        dbscan_min_samples,
        return_graph,
    )


def _mapper_python_custom(
    point_cloud: Tensor,
    filter_function: Callable[[Tensor], Tensor],
    cover_resolution: int,
    cover_overlap: float,
    clusterer: str,
    dbscan_eps: float,
    dbscan_min_samples: int,
    return_graph: bool,
) -> dict[str, Any]:
    """Pure Python Mapper implementation that supports callable filters."""
    (
        cover_resolution,
        cover_overlap,
        dbscan_eps,
        dbscan_min_samples,
    ) = _validate_mapper_inputs(
        point_cloud, cover_resolution, cover_overlap, dbscan_eps, dbscan_min_samples
    )
    filter_vals = filter_function(point_cloud)
    if filter_vals.dim() == 1:
        filter_vals = filter_vals.unsqueeze(1)

    return _mapper_from_filter_values(
        point_cloud,
        filter_vals,
        cover_resolution,
        cover_overlap,
        clusterer,
        dbscan_eps,
        dbscan_min_samples,
        return_graph,
    )


def _validate_mapper_inputs(
    point_cloud: Tensor,
    cover_resolution: int,
    cover_overlap: float,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> tuple[int, float, float, int]:
    _validate_finite_tensor(point_cloud, "point_cloud")
    if point_cloud.dim() != 2 or point_cloud.shape[0] == 0:
        raise ValueError("point_cloud must be a non-empty 2D tensor")
    cover_resolution = _validate_positive_int(cover_resolution, "cover_resolution")
    cover_overlap = _validate_finite_scalar(cover_overlap, "cover_overlap")
    dbscan_eps = _validate_finite_scalar(dbscan_eps, "dbscan_eps")
    dbscan_min_samples = _validate_positive_int(dbscan_min_samples, "dbscan_min_samples")
    if cover_resolution <= 0:
        raise ValueError("cover_resolution must be positive")
    if not 0.0 <= cover_overlap < 1.0:
        raise ValueError("cover_overlap must be in [0, 1)")
    if dbscan_eps <= 0:
        raise ValueError("dbscan_eps must be positive")
    if dbscan_min_samples <= 0:
        raise ValueError("dbscan_min_samples must be positive")
    return cover_resolution, cover_overlap, dbscan_eps, dbscan_min_samples


def _validate_filter_vals(filter_vals: Tensor, point_cloud: Tensor) -> Tensor:
    _validate_finite_tensor(filter_vals, "filter values")
    if filter_vals.dim() == 1:
        filter_vals = filter_vals.unsqueeze(1)
    if filter_vals.shape[0] != point_cloud.shape[0]:
        raise ValueError("filter values must have one row per point")
    if filter_vals.shape[1] < 1:
        raise ValueError("filter values must have at least one dimension")
    return filter_vals


def _build_1d_cover(
    filter_vals: Tensor, cover_resolution: int, cover_overlap: float
) -> list[dict[str, Any]]:
    f_min = filter_vals.min().item()
    f_max = filter_vals.max().item()
    f_range = max(f_max - f_min, 1e-12)
    eps = f_range * 0.01
    f_min -= eps
    f_max += eps
    f_range = f_max - f_min
    interval_size = f_range / cover_resolution
    overlap_size = interval_size * cover_overlap

    cover_sets = []
    for i in range(cover_resolution):
        interval_start = f_min + i * interval_size - overlap_size / 2
        interval_end = f_min + (i + 1) * interval_size + overlap_size / 2
        in_interval = (filter_vals[:, 0] >= interval_start) & (filter_vals[:, 0] <= interval_end)
        indices = torch.where(in_interval)[0].tolist()
        if indices:
            cover_sets.append(
                {
                    "indices": indices,
                    "range": (interval_start, interval_end),
                    "center": (interval_start + interval_end) / 2,
                }
            )
    return cover_sets


def _run_clusterer(
    points_in_set: Tensor, clusterer: str, dbscan_eps: float, dbscan_min_samples: int
) -> Tensor:
    if clusterer == "dbscan":
        return _dbscan_python(points_in_set, dbscan_eps, dbscan_min_samples)
    if clusterer == "single_linkage":
        return _single_linkage_python(points_in_set, dbscan_eps)
    if clusterer == "connected":
        return torch.zeros(len(points_in_set), dtype=torch.long)
    raise ValueError(f"Unknown clusterer: {clusterer}")


def _process_cluster_labels(
    cluster_labels: Tensor,
    indices: list[int],
    cover_idx: int,
    node_id: int,
    point_cloud: Tensor,
    filter_vals: Tensor,
    nodes: list[dict[str, Any]],
) -> int:
    for label in torch.unique(cluster_labels):
        if label < 0:
            continue
        mask = cluster_labels == label
        cluster_indices = [indices[i] for i in range(len(indices)) if mask[i]]
        if not cluster_indices:
            continue
        cluster_points = point_cloud[cluster_indices]
        cluster_filter_vals = filter_vals[cluster_indices]
        nodes.append(
            {
                "id": node_id,
                "point_indices": cluster_indices,
                "centroid": cluster_points.mean(0),
                "filter_centroid": cluster_filter_vals.mean(0),
                "cover_index": cover_idx,
            }
        )
        node_id += 1
    return node_id


def _build_edges_from_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            set_i = set(nodes[i]["point_indices"])
            set_j = set(nodes[j]["point_indices"])
            overlap = set_i & set_j
            if overlap:
                union = set_i | set_j
                weight = len(overlap) / len(union) if union else 0.0
                edges.append({"source": nodes[i]["id"], "target": nodes[j]["id"], "weight": weight})
    return edges


def _build_result_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    filter_vals: Tensor,
    return_graph: bool,
) -> dict[str, Any]:
    result = {"nodes": nodes, "edges": edges, "filter_values": filter_vals}
    if return_graph:
        graph = _build_networkx_graph(nodes, edges)
        if graph is not None:
            result["graph"] = graph
    return result


def _mapper_from_filter_values(
    point_cloud: Tensor,
    filter_vals: Tensor,
    cover_resolution: int,
    cover_overlap: float,
    clusterer: str,
    dbscan_eps: float,
    dbscan_min_samples: int,
    return_graph: bool,
) -> dict[str, Any]:
    filter_vals = _validate_filter_vals(filter_vals, point_cloud)

    cover_sets = (
        _build_1d_cover(filter_vals, cover_resolution, cover_overlap)
        if filter_vals.shape[1] == 1
        else _create_grid_cover(filter_vals, cover_resolution, cover_overlap)
    )

    nodes: list[dict[str, Any]] = []
    node_id = 0
    for cover_idx, cover_set in enumerate(cover_sets):
        indices = cover_set["indices"]
        if not indices:
            continue
        cluster_labels = _run_clusterer(
            point_cloud[indices], clusterer, dbscan_eps, dbscan_min_samples
        )
        node_id = _process_cluster_labels(
            cluster_labels, indices, cover_idx, node_id, point_cloud, filter_vals, nodes
        )

    edges = _build_edges_from_nodes(nodes)
    return _build_result_graph(nodes, edges, filter_vals, return_graph)


def _build_networkx_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> Any:
    with suppress(ImportError):
        import networkx as nx  # noqa: PLC0415

        graph = nx.Graph()
        for node in nodes:
            graph.add_node(
                node["id"],
                point_indices=node["point_indices"],
                centroid=node["centroid"].tolist(),
                filter_centroid=node["filter_centroid"].tolist(),
            )
        for edge in edges:
            graph.add_edge(edge["source"], edge["target"], weight=edge["weight"])
        return graph
    return None


def _create_grid_cover(
    filter_vals: Tensor, resolution: int, overlap: float
) -> list[dict[str, Any]]:
    """Create grid cover for multi-dimensional filter values."""
    n_dims = filter_vals.shape[1]
    if n_dims < 1:
        raise ValueError("filter values must have at least one dimension")
    if n_dims == 1:
        f_min = filter_vals.min().item()
        f_max = filter_vals.max().item()
        f_range = max(f_max - f_min, 1e-12)
        padding = f_range * 0.01
        f_min -= padding
        f_max += padding
        interval_size = (f_max - f_min) / resolution
        overlap_size = interval_size * overlap
        cover_sets = []
        for i in range(resolution):
            interval_start = f_min + i * interval_size - overlap_size / 2
            interval_end = f_min + (i + 1) * interval_size + overlap_size / 2
            in_interval = (filter_vals[:, 0] >= interval_start) & (
                filter_vals[:, 0] <= interval_end
            )
            indices = torch.where(in_interval)[0].tolist()
            if indices:
                cover_sets.append(
                    {
                        "indices": indices,
                        "range": (interval_start, interval_end),
                        "center": (interval_start + interval_end) / 2,
                    }
                )
        return cover_sets
    if n_dims > 2:
        return _create_grid_cover(filter_vals[:, :2], resolution, overlap)

    f_mins = filter_vals.min(0)[0].tolist()
    f_maxs = filter_vals.max(0)[0].tolist()
    eps = [max((f_maxs[i] - f_mins[i]) * 0.01, 1e-12) for i in range(n_dims)]
    f_mins = [f_mins[i] - eps[i] for i in range(n_dims)]
    f_maxs = [f_maxs[i] + eps[i] for i in range(n_dims)]

    cover_sets = []
    cell_size_x = (f_maxs[0] - f_mins[0]) / resolution
    cell_size_y = (f_maxs[1] - f_mins[1]) / resolution
    overlap_x = cell_size_x * overlap
    overlap_y = cell_size_y * overlap

    for i in range(resolution):
        for j in range(resolution):
            x_start = f_mins[0] + i * cell_size_x - overlap_x / 2
            x_end = f_mins[0] + (i + 1) * cell_size_x + overlap_x / 2
            y_start = f_mins[1] + j * cell_size_y - overlap_y / 2
            y_end = f_mins[1] + (j + 1) * cell_size_y + overlap_y / 2

            in_cell = (
                (filter_vals[:, 0] >= x_start)
                & (filter_vals[:, 0] <= x_end)
                & (filter_vals[:, 1] >= y_start)
                & (filter_vals[:, 1] <= y_end)
            )
            indices = torch.where(in_cell)[0].tolist()
            if indices:
                cover_sets.append(
                    {
                        "indices": indices,
                        "range": ((x_start, x_end), (y_start, y_end)),
                        "center": ((x_start + x_end) / 2, (y_start + y_end) / 2),
                    }
                )

    return cover_sets


def _dbscan_python(points: Tensor, eps: float, min_samples: int) -> Tensor:
    """DBSCAN clustering for the Python Mapper implementation."""
    n = points.shape[0]
    labels = torch.full((n,), -1, dtype=torch.long, device=points.device)
    visited = torch.zeros(n, dtype=torch.bool, device=points.device)
    cluster_id = 0

    def region_query(index: int) -> list[int]:
        distances = torch.norm(points - points[index], dim=1)
        return torch.where(distances <= eps)[0].tolist()

    for i in range(n):
        if visited[i]:
            continue

        visited[i] = True
        neighbors = region_query(i)
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seeds = list(neighbors)
        seed_set = set(seeds)
        j = 0
        while j < len(seeds):
            current = seeds[j]
            j += 1

            if not visited[current]:
                visited[current] = True
                current_neighbors = region_query(current)
                if len(current_neighbors) >= min_samples:
                    for neighbor in current_neighbors:
                        if neighbor not in seed_set:
                            seeds.append(neighbor)
                            seed_set.add(neighbor)

            if labels[current] == -1:
                labels[current] = cluster_id

        cluster_id += 1

    return labels


def _single_linkage_python(points: Tensor, threshold: float) -> Tensor:
    """Single-linkage clustering for the Python Mapper implementation."""
    n = points.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=points.device)
    distances = torch.cdist(points, points)
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] <= threshold:
                union(i, j)

    labels = torch.tensor([find(i) for i in range(n)], dtype=torch.long, device=points.device)
    unique = torch.unique(labels, sorted=True)
    mapping = {int(u): i for i, u in enumerate(unique)}
    return torch.tensor(
        [mapping[int(label)] for label in labels],
        dtype=torch.long,
        device=points.device,
    )


def _filter_pca_python(point_cloud: Tensor, n_components: int) -> Tensor:
    """Pure Python PCA filter."""
    _validate_finite_tensor(point_cloud, "point_cloud")
    if point_cloud.shape[0] < 2:
        return torch.zeros(
            (point_cloud.shape[0], min(n_components, point_cloud.shape[1])),
            dtype=point_cloud.dtype,
            device=point_cloud.device,
        )
    centered = point_cloud - point_cloud.mean(0)
    cov = torch.mm(centered.t(), centered) / (point_cloud.shape[0] - 1)
    _, eigenvectors = torch.linalg.eigh(cov)
    top_vectors = eigenvectors[:, -n_components:]
    return torch.mm(centered, top_vectors)


def _filter_eccentricity_python(point_cloud: Tensor) -> Tensor:
    """Pure Python eccentricity filter."""
    _validate_finite_tensor(point_cloud, "point_cloud")
    centroid = point_cloud.mean(0)
    return cast(Tensor, torch.norm(point_cloud - centroid, dim=1))


__all__ = [
    "_filter_eccentricity_python",
    "_filter_pca_python",
    "_mapper_python",
    "_mapper_python_custom",
    "_validate_finite_scalar",
    "_validate_finite_tensor",
    "_validate_positive_int",
]
