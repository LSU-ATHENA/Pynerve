"""Pairwise kernel primitives for persistence diagrams."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import Tensor

from ._distance_core import diagram_bottleneck, diagram_wasserstein
from ._vectorization_basis import (
    _validate_diagram,
    _validate_positive_finite,
    _validate_positive_int,
)

_SUPPORTED_DISTANCE_METRICS = {"wasserstein", "bottleneck", "euclidean"}


def _compute_distance_matrix(d1: Tensor, d2: Tensor, p: float = 2.0) -> Tensor:
    b1, d1_t = d1[:, 0], d1[:, 1]
    b2, d2_t = d2[:, 0], d2[:, 1]

    if p == 1.0:
        dist_b = torch.abs(b1.unsqueeze(1) - b2.unsqueeze(0))
        dist_d = torch.abs(d1_t.unsqueeze(1) - d2_t.unsqueeze(0))
        return dist_b + dist_d
    if p == 2.0:
        dist_b = (b1.unsqueeze(1) - b2.unsqueeze(0)) ** 2
        dist_d = (d1_t.unsqueeze(1) - d2_t.unsqueeze(0)) ** 2
        return torch.sqrt(dist_b + dist_d)
    if math.isinf(p):
        dist_b = torch.abs(b1.unsqueeze(1) - b2.unsqueeze(0))
        dist_d = torch.abs(d1_t.unsqueeze(1) - d2_t.unsqueeze(0))
        return torch.maximum(dist_b, dist_d)

    dist_b = torch.abs(b1.unsqueeze(1) - b2.unsqueeze(0)) ** p
    dist_d = torch.abs(d1_t.unsqueeze(1) - d2_t.unsqueeze(0)) ** p
    return (dist_b + dist_d) ** (1.0 / p)


def _validate_kernel_diagrams(d1: Tensor, d2: Tensor) -> tuple[Tensor, Tensor]:
    d1 = _validate_diagram(d1)
    d2 = _validate_diagram(d2)
    if d1.numel() > 0 and not torch.isfinite(d1[:, 1]).all().item():
        raise ValueError("kernel diagrams require finite deaths")
    if d2.numel() > 0 and not torch.isfinite(d2[:, 1]).all().item():
        raise ValueError("kernel diagrams require finite deaths")
    if d1.device != d2.device:
        raise ValueError("kernel diagrams must be on the same device")
    return d1, d2


def _validate_positive_norm(value: float, name: str) -> float:
    result = float(value)
    if result <= 0 or math.isnan(result):
        raise ValueError(f"{name} must be positive")
    return result


def _finite_coords(diagram: Tensor) -> Tensor:
    coords = diagram[:, :2]
    finite = torch.isfinite(coords[:, 1])
    return coords[finite]


def gaussian_kernel(
    d1: Tensor,
    d2: Tensor,
    sigma: float = 1.0,
    distance_metric: Literal["wasserstein", "bottleneck", "euclidean"] = "euclidean",
    p: float = 2.0,
) -> Tensor:
    """Gaussian kernel using Euclidean, Wasserstein, or bottleneck distance."""
    sigma = _validate_positive_finite(sigma, "sigma")
    p = _validate_positive_norm(p, "p")
    if distance_metric not in _SUPPORTED_DISTANCE_METRICS:
        raise ValueError("distance_metric must be 'wasserstein', 'bottleneck', or 'euclidean'")
    d1, d2 = _validate_kernel_diagrams(d1, d2)
    if distance_metric == "wasserstein":
        dist = diagram_wasserstein(d1, d2, p=p)
        return torch.exp(-(dist**2) / (2 * sigma**2))
    if distance_metric == "bottleneck":
        dist = diagram_bottleneck(d1, d2)
        return torch.exp(-(dist**2) / (2 * sigma**2))

    coords1 = _finite_coords(d1)
    coords2 = _finite_coords(d2)
    if len(coords1) == 0 or len(coords2) == 0:
        return torch.tensor(0.0, device=d1.device, dtype=d1.dtype)
    dist_matrix = _compute_distance_matrix(coords1, coords2, p=p)
    similarities = torch.exp(-(dist_matrix**2) / (2 * sigma**2))
    return similarities.sum()


def persistence_scale_space_kernel(
    d1: Tensor,
    d2: Tensor,
    sigma: float = 1.0,
    weight: float = 0.5,
) -> Tensor:
    """Scale-space kernel in birth-death/persistence coordinates."""
    sigma = _validate_positive_finite(sigma, "sigma")
    weight = float(weight)
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be finite and in [0, 1]")
    d1, d2 = _validate_kernel_diagrams(d1, d2)
    coords1 = _finite_coords(d1)
    coords2 = _finite_coords(d2)
    if len(coords1) == 0 or len(coords2) == 0:
        return torch.tensor(0.0, device=d1.device, dtype=d1.dtype)

    dist_matrix = _compute_distance_matrix(coords1, coords2, p=2.0)
    persistence1 = (coords1[:, 1] - coords1[:, 0]).unsqueeze(1)
    persistence2 = (coords2[:, 1] - coords2[:, 0]).unsqueeze(0)
    persistence_diff = (persistence1 - persistence2) ** 2
    combined_dist = (1 - weight) * dist_matrix**2 + weight * persistence_diff
    return torch.exp(-combined_dist / (2 * sigma**2)).sum()


def sliced_wasserstein_kernel(
    d1: Tensor,
    d2: Tensor,
    num_slices: int = 10,
    sigma: float = 1.0,
) -> Tensor:
    """Approximate Wasserstein kernel by 1D projections."""
    num_slices = _validate_positive_int(num_slices, "num_slices")
    sigma = _validate_positive_finite(sigma, "sigma")
    d1, d2 = _validate_kernel_diagrams(d1, d2)
    coords1 = _finite_coords(d1)
    coords2 = _finite_coords(d2)
    if len(coords1) == 0 or len(coords2) == 0:
        return torch.tensor(0.0, device=d1.device, dtype=d1.dtype)

    angles = torch.linspace(0, math.pi, num_slices + 1, device=d1.device, dtype=d1.dtype)[:-1]
    total_kernel = torch.tensor(0.0, device=d1.device, dtype=d1.dtype)

    for angle in angles:
        direction = torch.stack([torch.cos(angle), torch.sin(angle)])
        proj1 = torch.sort(coords1 @ direction)[0]
        proj2 = torch.sort(coords2 @ direction)[0]

        n1, n2 = len(proj1), len(proj2)
        if n1 < n2:
            proj1 = torch.cat([proj1, torch.zeros(n2 - n1, device=d1.device, dtype=d1.dtype)])
        elif n2 < n1:
            proj2 = torch.cat([proj2, torch.zeros(n1 - n2, device=d1.device, dtype=d1.dtype)])

        w_dist = torch.abs(proj1 - proj2).mean()
        total_kernel += torch.exp(-(w_dist**2) / (2 * sigma**2))

    return total_kernel / num_slices


def persistence_fisher_kernel(
    d1: Tensor,
    d2: Tensor,
    sigma: float = 1.0,
    bandwidth: float = 0.5,
) -> Tensor:
    """Persistence Fisher-style kernel with persistence weights."""
    sigma = _validate_positive_finite(sigma, "sigma")
    bandwidth = _validate_positive_finite(bandwidth, "bandwidth")
    d1, d2 = _validate_kernel_diagrams(d1, d2)
    coords1 = _finite_coords(d1)
    coords2 = _finite_coords(d2)
    if len(coords1) == 0 or len(coords2) == 0:
        return torch.tensor(0.0, device=d1.device, dtype=d1.dtype)

    pers1 = torch.clamp(coords1[:, 1] - coords1[:, 0], min=0)
    pers2 = torch.clamp(coords2[:, 1] - coords2[:, 0], min=0)
    weights1 = (
        pers1 / pers1.sum()
        if pers1.sum() > 0
        else torch.ones(len(coords1), device=d1.device, dtype=d1.dtype) / len(coords1)
    )
    weights2 = (
        pers2 / pers2.sum()
        if pers2.sum() > 0
        else torch.ones(len(coords2), device=d1.device, dtype=d1.dtype) / len(coords2)
    )

    dist_matrix = _compute_distance_matrix(coords1, coords2, p=2.0)
    effective_sigma = sigma * bandwidth
    spatial_kernel = torch.exp(-(dist_matrix**2) / (2 * effective_sigma**2))
    return (spatial_kernel * weights1.unsqueeze(1) * weights2.unsqueeze(0)).sum()


def linear_kernel(
    d1: Tensor,
    d2: Tensor,
    sigma: float = 1.0,  # noqa: ARG001
    vectorization: Literal["image", "landscape", "silhouette"] = "silhouette",
    **kwargs: Any,
) -> Tensor:
    """Linear kernel after fixed-size vectorization."""
    from . import vectorization as vec  # noqa: PLC0415

    d1, d2 = _validate_kernel_diagrams(d1, d2)
    v1 = vec.diagram_to_vector(d1, method=vectorization, **kwargs).flatten()
    v2 = vec.diagram_to_vector(d2, method=vectorization, **kwargs).flatten()
    return torch.dot(v1, v2)


__all__ = [
    "gaussian_kernel",
    "persistence_scale_space_kernel",
    "sliced_wasserstein_kernel",
    "persistence_fisher_kernel",
    "linear_kernel",
]
