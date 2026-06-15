"""High-level feature extraction from persistence-diagram statistics."""

from __future__ import annotations

from typing import Any

import torch

from ._statistics_core import (
    amplitude,
    betti_curve,
    betti_numbers_at_scale,
    max_persistence,
    mean_persistence,
    number_of_features,
    persistence_entropy,
    persistence_variance,
    total_persistence,
)


def all_statistics(diagram: torch.Tensor, dims: list[Any] | None = None) -> dict[str, Any]:
    """Compute a comprehensive dictionary of scalar topological statistics."""
    stats: dict[str, float] = {}
    dims = [None] if dims is None else dims

    for dim in dims:
        dim_str = f"_dim{dim}" if dim is not None else ""
        stats[f"num_features{dim_str}"] = number_of_features(diagram, dim=dim).item()
        stats[f"num_infinite{dim_str}"] = number_of_features(
            diagram, dim=dim, min_persistence=float("inf") - 1
        ).item()
        stats[f"total_persistence{dim_str}"] = total_persistence(diagram, dim=dim, p=1.0).item()
        stats[f"total_persistence_sq{dim_str}"] = total_persistence(diagram, dim=dim, p=2.0).item()
        stats[f"mean_persistence{dim_str}"] = mean_persistence(diagram, dim=dim).item()
        stats[f"max_persistence{dim_str}"] = max_persistence(diagram, dim=dim).item()
        stats[f"variance_persistence{dim_str}"] = persistence_variance(diagram, dim=dim).item()
        stats[f"persistence_entropy{dim_str}"] = persistence_entropy(diagram, dim=dim).item()
        stats[f"amplitude_bottleneck{dim_str}"] = amplitude(
            diagram, metric="bottleneck", dim=dim
        ).item()
        stats[f"amplitude_total{dim_str}"] = amplitude(
            diagram, metric="persistence", dim=dim
        ).item()

    return stats


def extract_features(diagram: torch.Tensor, dims: list[Any] | None = None) -> torch.Tensor:
    """Flatten `all_statistics` output into a 1D feature vector."""
    stats = all_statistics(diagram, dims=dims)
    return torch.tensor(list(stats.values()), dtype=torch.float32, device=diagram.device)


__all__ = [
    "total_persistence",
    "mean_persistence",
    "max_persistence",
    "persistence_variance",
    "persistence_entropy",
    "number_of_features",
    "betti_numbers_at_scale",
    "betti_curve",
    "amplitude",
    "all_statistics",
    "extract_features",
]
