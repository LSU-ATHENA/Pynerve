"""PyTorch module wrapper for persistent homology computation (re-export facade)."""

from __future__ import annotations

from torch import Tensor

from ._ph_autograd import PersistentHomologyFunction
from ._ph_module import PersistentHomology


def compute_persistence_diagrams(
    points: Tensor,
    max_dim: int = 1,
    max_radius: float = float("inf"),
    metric: str = "euclidean",
) -> list[Tensor]:
    """Compute per-dimension persistence diagrams for a batched point cloud.

    Unlike :func:`pynerve.compute_persistence`, this function returns a list of
    tensors (one per homology dimension), each shaped ``(batch, max_pairs, 2)``.
    It is designed for use within PyTorch workflows.

    Args:
        points: Batched point cloud shaped ``(batch, N, D)``.
        max_dim: Maximum homology dimension (default: 1).
        max_radius: Filtration radius cutoff (default: inf).
        metric: Distance metric (default: "euclidean").

    Returns:
        List of tensors ``[dim_0_diagrams, dim_1_diagrams, ...]`` where each
        tensor has shape ``(batch, max_pairs, 2)`` with columns (birth, death).
    """
    ph = PersistentHomology(max_dim=max_dim, max_radius=max_radius, metric=metric)
    return ph(points)


__all__ = [
    "PersistentHomologyFunction",
    "PersistentHomology",
    "compute_persistence_diagrams",
]
