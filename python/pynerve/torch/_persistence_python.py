"""Deterministic Python python implementation kernels for torch persistence APIs."""

from __future__ import annotations

import math

import torch

_DISTANCE_FUNCTIONS = {
    "euclidean": lambda pts: torch.cdist(pts, pts, p=2),
    "manhattan": lambda pts: torch.cdist(pts, pts, p=1),
    "chebyshev": lambda pts: torch.cdist(pts, pts, p=float("inf")),
    "cosine": lambda pts: (
        1
        - torch.nn.functional.normalize(pts, dim=-1) @ torch.nn.functional.normalize(pts, dim=-1).T
    ).clamp(min=0),
}


def _validate_vr_inputs(points: torch.Tensor, max_radius: float) -> float:
    radius = float(max_radius)
    if radius <= 0 or math.isnan(radius):
        raise ValueError(f"max_radius must be positive, got {max_radius}")
    if points.dim() != 3:
        raise ValueError(f"points must be a rank-3 tensor, got rank {points.dim()}")
    if not torch.is_floating_point(points):
        raise TypeError("points must use a floating-point dtype")
    if points.shape[0] == 0:
        raise ValueError("points must contain at least one batch item")
    if points.shape[1] == 0:
        raise ValueError("points must contain at least one point")
    if points.shape[2] == 0:
        raise ValueError("points must contain at least one coordinate dimension")
    if not torch.isfinite(points).all().item():
        raise ValueError("points must contain only finite coordinates")
    return radius


def _compute_distance(pts: torch.Tensor, metric: str) -> torch.Tensor:
    fn = _DISTANCE_FUNCTIONS.get(metric)
    if fn is None:
        raise ValueError(f"Unsupported metric: {metric}")
    return fn(pts)


def _kruskal_batch(
    dist: torch.Tensor, n_points: int, radius: float, inf: torch.Tensor, diagram_slot: torch.Tensor
) -> None:
    parent = list(range(n_points))
    rank = [0] * n_points

    def find(x: int, parent: list[int] = parent) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(x: int, y: int, parent: list[int] = parent, rank: list[int] = rank) -> bool:
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True

    edges: list[tuple[float, int, int]] = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            edges.append((float(dist[i, j].detach().cpu().item()), i, j))
    edges.sort(key=lambda x: x[0])

    k = 0
    for edge_weight, i, j in edges:
        if edge_weight > radius:
            break
        if unite(i, j):
            diagram_slot[k, 0] = 0.0
            diagram_slot[k, 1] = edge_weight
            diagram_slot[k, 2] = 0.0
            k += 1
            if k == n_points - 1:
                break

    essential_roots: set[int] = set()
    for i in range(n_points):
        root = find(i)
        if root in essential_roots:
            continue
        essential_roots.add(root)
        diagram_slot[k, 0] = 0.0
        diagram_slot[k, 1] = inf
        diagram_slot[k, 2] = 0.0
        k += 1


def compute_vr_python(
    points: torch.Tensor,
    max_dim: int,
    metric: str,
    max_radius: float = float("inf"),
) -> torch.Tensor:
    """Exact H0 python implementation via Kruskal MST over pairwise distances."""
    del max_dim  # higher-dimensional pairs require native persistence kernels
    radius = _validate_vr_inputs(points, max_radius)

    batch_size, n_points, _ = points.shape
    max_pairs = n_points
    diagrams = torch.zeros((batch_size, max_pairs, 3), dtype=points.dtype, device=points.device)
    inf = torch.tensor(float("inf"), dtype=points.dtype, device=points.device)

    for b in range(batch_size):
        dist = _compute_distance(points[b], metric)
        _kruskal_batch(dist, n_points, radius, inf, diagrams[b])

    return diagrams


__all__ = ["compute_vr_python"]
