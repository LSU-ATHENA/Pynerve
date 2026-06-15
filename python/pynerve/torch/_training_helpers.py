"""Helper functions for topology-aware training loops."""

from __future__ import annotations

import math
from typing import Any, cast

import torch

from . import kernels
from ._kernels_pairwise import (
    gaussian_kernel,
    linear_kernel,
    persistence_fisher_kernel,
    persistence_scale_space_kernel,
    sliced_wasserstein_kernel,
)

_KERNEL_FUNCTIONS: dict[str, Any] = {
    "gaussian": gaussian_kernel,
    "pss": persistence_scale_space_kernel,
    "sliced_wasserstein": sliced_wasserstein_kernel,
    "fisher": persistence_fisher_kernel,
    "linear": linear_kernel,
}


def compute_kernel_similarity(
    diagrams1: list[Any],
    diagrams2: list[Any],
    kernel: str = "gaussian",
    sigma: float = 0.5,
) -> torch.Tensor:
    """Compute kernel-based similarity between sets of diagrams."""
    if kernel not in _KERNEL_FUNCTIONS:
        raise ValueError(f"Unknown kernel: {kernel}")
    sigma = float(sigma)
    if kernel != "linear" and (sigma <= 0 or not math.isfinite(sigma)):
        raise ValueError("sigma must be finite and positive")
    if diagrams1 is diagrams2:
        kwargs: dict[str, Any] = {} if kernel == "linear" else {"sigma": sigma}
        return cast(
            torch.Tensor,
            kernels.compute_kernel_matrix(diagrams1, kernel=cast(Any, kernel), **kwargs),
        )
    kernel_fn = _KERNEL_FUNCTIONS[kernel]
    rows: list[torch.Tensor] = []
    for left in diagrams1:
        values: list[torch.Tensor] = []
        for right in diagrams2:
            value: Any = (
                kernel_fn(left, right)
                if kernel == "linear"
                else kernel_fn(left, right, sigma=sigma)
            )
            values.append(value if isinstance(value, torch.Tensor) else torch.tensor(value))
        rows.append(torch.stack(values) if values else torch.empty(0))
    return torch.stack(rows) if rows else torch.empty((0, len(diagrams2)))


def topological_batch_loss(
    batch_diagrams: list[Any],
    target_statistics: dict[str, float],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute average regularization loss against target topology statistics."""
    if len(batch_diagrams) == 0:
        raise ValueError("batch_diagrams must be non-empty")
    from .training_utils_impl import TopologicalRegularization  # noqa: PLC0415

    reg = TopologicalRegularization(target_complexity=target_statistics, weights=weights)
    total_loss: torch.Tensor | None = None
    for diagram in batch_diagrams:
        value: torch.Tensor = reg(diagram)
        total_loss = value if total_loss is None else total_loss + value
    assert total_loss is not None, "batch_diagrams is non-empty"
    return total_loss / len(batch_diagrams)


__all__ = [
    "compute_kernel_similarity",
    "topological_batch_loss",
]
