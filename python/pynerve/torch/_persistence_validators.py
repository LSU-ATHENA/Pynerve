"""Validation utilities for persistence API functions."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from ._backend import backend

_SUPPORTED_METRICS = {"euclidean", "manhattan", "chebyshev", "cosine"}


def _torch_backend() -> Any:
    backend._ensure_backends()
    return backend._torch_c


def _core_backend() -> Any:
    backend._ensure_backends()
    return backend._core_c


def _validate_max_dim(max_dim: int) -> int:
    if isinstance(max_dim, bool) or not isinstance(max_dim, int) or max_dim < 0:
        raise ValueError(f"max_dim must be a non-negative integer, got {max_dim!r}")
    return max_dim


def _validate_max_radius(max_radius: float) -> float:
    radius = float(max_radius)
    if radius <= 0 or math.isnan(radius):
        raise ValueError(f"max_radius must be positive, got {max_radius!r}")
    return radius


def _validate_metric(metric: str) -> str:
    if metric not in _SUPPORTED_METRICS:
        supported = ", ".join(sorted(_SUPPORTED_METRICS))
        raise ValueError(f"Unsupported metric {metric!r}; expected one of {supported}")
    return metric


def _validate_image_resolution(resolution: tuple[Any, ...]) -> tuple[int, int]:
    if len(resolution) != 2:
        raise ValueError("resolution must contain exactly two entries")
    rows = int(resolution[0])
    cols = int(resolution[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    return rows, cols


def _validate_persistence_image_diagram(diagram: Tensor) -> Tensor:
    if diagram.dim() != 2:
        raise ValueError(f"Expected 2D diagram tensor, got {diagram.dim()}D")
    if diagram.shape[-1] < 2:
        raise ValueError("diagram must have at least birth and death columns")
    if not torch.is_floating_point(diagram):
        raise TypeError("diagram must use a floating-point dtype")
    if diagram.numel() == 0:
        return diagram

    work = diagram[..., :2].to(dtype=torch.float64)
    births = work[:, 0]
    deaths = work[:, 1]
    if not torch.isfinite(births).all().item():
        raise ValueError("diagram births must be finite")
    if torch.isnan(deaths).any().item():
        raise ValueError("diagram deaths must not be NaN")
    finite_death_mask = torch.isfinite(deaths)
    if (
        finite_death_mask.any().item()
        and not (deaths[finite_death_mask] >= births[finite_death_mask]).all().item()
    ):
        raise ValueError("diagram finite deaths must be greater than or equal to births")
    return diagram


def _valid_vr_output_mask(diagram_tensor: Tensor, batch_size: int, python_output: bool) -> Tensor:
    max_pairs = diagram_tensor.shape[1]
    if python_output:
        return torch.ones((batch_size, max_pairs), dtype=torch.bool, device=diagram_tensor.device)
    row_has_value = (diagram_tensor[..., 0] != 0) | (diagram_tensor[..., 1] != 0)
    if max_pairs == 0:
        return row_has_value
    row_numbers = torch.arange(1, max_pairs + 1, device=diagram_tensor.device)
    valid_counts = (row_has_value.long() * row_numbers).max(dim=1).values
    row_indices = torch.arange(max_pairs, device=diagram_tensor.device)
    return row_indices.unsqueeze(0) < valid_counts.unsqueeze(1)


def _count_pairs_by_dimension(diagram_tensor: Tensor, mask: Tensor, max_dim: int) -> Tensor:
    batch_size = diagram_tensor.shape[0]
    if diagram_tensor.shape[-1] >= 3:
        dims = diagram_tensor[..., 2].long()
    else:
        dims = torch.zeros_like(diagram_tensor[..., 0], dtype=torch.long)
    num_pairs = torch.zeros(batch_size, max_dim + 1, dtype=torch.long, device=diagram_tensor.device)
    for b in range(batch_size):
        for d in range(max_dim + 1):
            num_pairs[b, d] = ((dims[b] == d) & mask[b]).sum()
    return num_pairs
