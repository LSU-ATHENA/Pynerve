"""Reusable components for learnable Mapper models."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import ceil, isfinite
from typing import cast

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._constants import EPS
from .._validation import validate_positive_int

CoverElement = tuple[tuple[float, float], ...]


def _validate_positive_finite(value: float, name: str) -> float:
    result = float(value)
    if result <= 0 or not isfinite(result):
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_overlap(value: float, name: str) -> float:
    result = float(value)
    if result < 0 or result >= 1 or not isfinite(result):
        raise ValueError(f"{name} must be finite and in [0, 1)")
    return result


def _validate_hidden_dims(hidden_dims: Sequence[int], name: str) -> list[int]:
    if isinstance(hidden_dims, (str, bytes)) or not isinstance(hidden_dims, Sequence):
        raise TypeError(f"{name} must be a sequence of positive integers")
    return [validate_positive_int(dim, name) for dim in hidden_dims]


def _validate_floating_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must use a floating-point dtype")
    if tensor.numel() > 0 and not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


class LensFunction(nn.Module):
    """Learnable lens (filter) function for the Mapper algorithm.

    Maps point clouds to a lower-dimensional lens space via a
    configurable MLP with layer normalisation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ):
        """Initialise the learnable lens.

        :param input_dim: Dimensionality of input point features.
        :param output_dim: Dimensionality of the lens output.
        :param hidden_dims: Hidden-layer sizes; defaults to ``[128, 64]``.
        :param activation: Activation name ``"relu"`` or ``"leaky_relu"``.
        :raises ValueError: If a dimension is invalid or ``activation`` is
            unsupported.
        """
        super().__init__()
        input_dim = validate_positive_int(input_dim, "input_dim")
        output_dim = validate_positive_int(output_dim, "output_dim")
        if hidden_dims is None:
            hidden_dims = [128, 64]
        hidden_dims = _validate_hidden_dims(hidden_dims, "hidden_dims")

        self.input_dim = input_dim
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            layers.append(nn.LayerNorm(h))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the lens to a point cloud.

        :param x: Input tensor of shape ``(N, D)`` or ``(B, N, D)``.
        :returns: Lens-space tensor of shape ``(N, O)`` or ``(B, N, O)``.
        :raises ValueError: If the input dimension or tensor shape is invalid.
        """
        x = _validate_floating_tensor(x, "x")
        if x.dim() == 3:
            if x.shape[-1] != self.input_dim:
                raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[-1]}")
            batch_size, n_points, dim = x.shape
            x_flat = x.reshape(-1, dim)
            y_flat: torch.Tensor = self.network(x_flat)
            return cast(torch.Tensor, y_flat.view(batch_size, n_points, self.output_dim))
        if x.dim() == 2:
            if x.shape[-1] != self.input_dim:
                raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[-1]}")
            return cast(torch.Tensor, self.network(x))
        raise ValueError(f"Expected a 2D or 3D tensor, got {tuple(x.shape)}")


class AdaptiveCover(nn.Module):
    """Learnable cover that adapts resolution and overlap during training.

    Uses softmax over a resolution logit vector and a sigmoid-gated
    overlap parameter so the cover can be optimized end-to-end.
    """

    def __init__(
        self,
        min_resolution: int = 2,
        max_resolution: int = 20,
        min_overlap: float = 0.1,
        max_overlap: float = 0.5,
    ):
        """Initialise the adaptive cover.

        :param min_resolution: Minimum number of intervals per dimension.
        :param max_resolution: Maximum number of intervals per dimension.
        :param min_overlap: Minimum overlap fraction in ``[0, 1)``.
        :param max_overlap: Maximum overlap fraction in ``[0, 1)``.
        :raises ValueError: If bounds are invalid or ``min > max``.
        """
        super().__init__()
        min_resolution = validate_positive_int(min_resolution, "min_resolution")
        max_resolution = validate_positive_int(max_resolution, "max_resolution")
        min_overlap = _validate_overlap(min_overlap, "min_overlap")
        max_overlap = _validate_overlap(max_overlap, "max_overlap")
        if max_resolution < min_resolution:
            raise ValueError("resolution bounds must satisfy 1 <= min <= max")
        if min_overlap > max_overlap:
            raise ValueError("overlap bounds must satisfy 0 <= min <= max < 1")

        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap

        self.resolution_logits = nn.Parameter(torch.zeros(max_resolution - min_resolution + 1))
        self.overlap_param = nn.Parameter(torch.tensor(0.0))

    def get_cover_params(self) -> tuple[int, float]:
        """Return the current learned cover resolution and overlap.

        :returns: Tuple ``(resolution, overlap)``.
        """
        resolution_probs = F.softmax(self.resolution_logits, dim=0)
        resolutions = torch.arange(
            self.min_resolution,
            self.max_resolution + 1,
            dtype=torch.float32,
            device=resolution_probs.device,
        )
        soft_resolution = (resolution_probs * resolutions).sum()

        hard_resolution = int(soft_resolution.round().item())
        hard_resolution = max(self.min_resolution, min(self.max_resolution, hard_resolution))

        overlap_ratio = torch.sigmoid(self.overlap_param)
        overlap = self.min_overlap + overlap_ratio * (self.max_overlap - self.min_overlap)

        return hard_resolution, overlap.item()

    def create_cover(self, lens_values: torch.Tensor) -> list[CoverElement]:
        """Create a cover over the lens space using learned parameters.

        :param lens_values: Lens-space tensor of shape ``(N,)`` or ``(N, D)``.
        :returns: List of cover elements, each a tuple of ``(start, end)``
            intervals.
        :raises ValueError: If ``lens_values`` is empty or has invalid shape.
        """
        resolution, overlap = self.get_cover_params()
        lens_values = _validate_floating_tensor(lens_values, "lens_values")
        if lens_values.numel() == 0:
            raise ValueError("lens_values must be non-empty")

        if lens_values.dim() == 1:
            min_val = lens_values.min().item()
            max_val = lens_values.max().item()
            span = max(max_val - min_val, EPS)
            step = span / resolution
            interval_size = step * (1 + overlap)

            intervals: list[CoverElement] = []
            for i in range(resolution):
                start = min_val + i * step
                end = start + interval_size
                intervals.append(((start, end),))

            return intervals

        if lens_values.dim() != 2:
            raise ValueError(f"Expected 1D or 2D lens values, got {tuple(lens_values.shape)}")

        lens_dim = lens_values.shape[1]
        if lens_dim == 0:
            raise ValueError("lens_values must have at least one feature dimension")
        intervals_per_dim = max(1, int(ceil(resolution ** (1.0 / lens_dim))))
        per_dim: list[list[tuple[float, float]]] = []
        for d in range(lens_dim):
            min_val = lens_values[:, d].min().item()
            max_val = lens_values[:, d].max().item()
            span = max(max_val - min_val, EPS)
            step = span / intervals_per_dim
            interval_size = step * (1 + overlap)
            per_dim.append(
                [
                    (min_val + i * step, min_val + i * step + interval_size)
                    for i in range(intervals_per_dim)
                ]
            )

        return [tuple(box) for box in product(*per_dim)]


class SoftClusterAssignment(nn.Module):
    """Soft cluster assignment via temperature-scaled softmax of distances."""

    def __init__(self, temperature: float = 0.1):
        """Initialise the soft cluster assignment.

        :param temperature: Temperature for the softmax (must be positive
            and finite).
        :raises ValueError: If ``temperature`` is not positive and finite.
        """
        super().__init__()
        self.temperature = _validate_positive_finite(temperature, "temperature")

    def forward(self, data: torch.Tensor, cluster_centers: torch.Tensor) -> torch.Tensor:
        """Compute soft assignments from data points to cluster centers.

        :param data: Data tensor of shape ``(N, D)``.
        :param cluster_centers: Center tensor of shape ``(K, D)``.
        :returns: Assignment matrix of shape ``(N, K)`` where rows sum to 1.
        :raises ValueError: If shapes are invalid or tensors are empty.
        """
        data = _validate_floating_tensor(data, "data")
        cluster_centers = _validate_floating_tensor(cluster_centers, "cluster_centers")
        if data.dim() != 2 or cluster_centers.dim() != 2:
            raise ValueError("data and cluster_centers must be 2D tensors")
        if data.shape[0] == 0 or cluster_centers.shape[0] == 0:
            raise ValueError("data and cluster_centers must be non-empty")
        if data.shape[1] != cluster_centers.shape[1]:
            raise ValueError("data and cluster_centers must have matching feature dimensions")
        distances = torch.cdist(data, cluster_centers)
        return F.softmax(-distances / self.temperature, dim=1)
