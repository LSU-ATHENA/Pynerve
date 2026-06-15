"""Helper functions for assembling persistence diagrams."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ._diagram import PersistenceDiagram


def _build_sorted_edges(dist: Tensor, n_points: int) -> list[tuple[float, int, int]]:
    edges: list[tuple[float, int, int]] = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            weight = float(dist[i, j].detach().cpu().item())
            if math.isfinite(weight):
                edges.append((weight, i, j))
    edges.sort(key=lambda item: item[0])
    return edges


def _diagram_from_distance_matrix_python(
    distance_matrix: Tensor, max_dim: int, *, single_input: bool
) -> PersistenceDiagram:
    batch_size, n_points, _ = distance_matrix.shape
    max_pairs = n_points
    diagrams = torch.zeros(
        (batch_size, max_pairs, 3),
        dtype=distance_matrix.dtype,
        device=distance_matrix.device,
    )
    mask = torch.zeros((batch_size, max_pairs), dtype=torch.bool, device=distance_matrix.device)
    num_pairs = torch.zeros(
        (batch_size, max_dim + 1), dtype=torch.long, device=distance_matrix.device
    )
    inf = torch.tensor(float("inf"), dtype=distance_matrix.dtype, device=distance_matrix.device)

    for b in range(batch_size):
        dist = distance_matrix[b]
        parent: list[int] = list(range(n_points))
        rank: list[int] = [0] * n_points

        def find(x: int) -> int:  # noqa: B023
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def unite(x: int, y: int) -> bool:  # noqa: B023
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

        edges = _build_sorted_edges(dist, n_points)

        k = 0
        for edge_weight, i, j in edges:
            if unite(i, j):
                diagrams[b, k, 0] = 0.0
                diagrams[b, k, 1] = edge_weight
                diagrams[b, k, 2] = 0.0
                mask[b, k] = True
                k += 1

        essential_roots: set[int] = set()
        for i in range(n_points):
            root = find(i)
            if root in essential_roots:
                continue
            essential_roots.add(root)
            diagrams[b, k, 0] = 0.0
            diagrams[b, k, 1] = inf
            diagrams[b, k, 2] = 0.0
            mask[b, k] = True
            k += 1
        num_pairs[b, 0] = k

    if single_input:
        diagrams = diagrams.squeeze(0)
        mask = mask.squeeze(0)
        num_pairs = num_pairs.squeeze(0)
    return PersistenceDiagram(diagrams, mask, num_pairs)


def _diagram_from_backend_parts(
    diagrams_list: list[Tensor],
    masks_list: list[Tensor],
    num_pairs_list: list[Tensor],
    *,
    batched: bool,
) -> PersistenceDiagram:
    if not diagrams_list:
        raise ValueError("backend returned no diagrams")
    if not batched and len(diagrams_list) == 1:
        return PersistenceDiagram(diagrams_list[0], masks_list[0], num_pairs_list[0])

    max_pairs = max(diagram.shape[0] for diagram in diagrams_list)
    max_pair_dims = max(diagram.shape[1] for diagram in diagrams_list)
    max_num_pair_dims = max(num_pairs.reshape(-1).shape[0] for num_pairs in num_pairs_list)

    padded_diagrams = []
    padded_masks = []
    padded_num_pairs = []
    for diagram, mask, num_pairs in zip(diagrams_list, masks_list, num_pairs_list, strict=True):
        if diagram.shape[0] < max_pairs or diagram.shape[1] < max_pair_dims:
            padded = diagram.new_zeros((max_pairs, max_pair_dims))
            padded[: diagram.shape[0], : diagram.shape[1]] = diagram
            diagram = padded
        if mask.shape[0] < max_pairs:
            padded_mask = mask.new_zeros((max_pairs,))
            padded_mask[: mask.shape[0]] = mask
            mask = padded_mask
        num_pairs = num_pairs.reshape(-1)
        if num_pairs.shape[0] < max_num_pair_dims:
            padded_counts = num_pairs.new_zeros((max_num_pair_dims,))
            padded_counts[: num_pairs.shape[0]] = num_pairs
            num_pairs = padded_counts
        padded_diagrams.append(diagram)
        padded_masks.append(mask)
        padded_num_pairs.append(num_pairs)

    return PersistenceDiagram(
        torch.stack(padded_diagrams, dim=0),
        torch.stack(padded_masks, dim=0),
        torch.stack(padded_num_pairs, dim=0),
    )
