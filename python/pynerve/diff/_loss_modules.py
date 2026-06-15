"""Individual topology loss modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._validation import validate_finite_tensor as _validate_finite_tensor
from .._validation import validate_positive_int as _validate_positive_int
from ._diagram_distances import PersistenceLoss
from ._loss_helpers import (
    _persistence_values,
    _validate_diagram_dimensions,
    _validate_diagram_sequence,
    _validate_non_negative_scalar,
    _validate_positive_scalar,
    _validate_target_betti,
)
from .ph_layer import compute_persistence_landscape


class BettiNumberLoss(nn.Module):
    """Differentiable approximation of Betti-number matching."""

    def __init__(self, threshold: float = 0.1, temperature: float = 0.1):
        """Initialize Betti number loss.

        :param threshold: Persistence threshold for considering a feature significant.
        :param temperature: Softness parameter controlling the sigmoid steepness.
        :raises ValueError: If ``threshold`` is negative or ``temperature`` is non-positive.
        """
        super().__init__()
        threshold = _validate_non_negative_scalar("threshold", threshold)
        temperature = _validate_positive_scalar("temperature", temperature)
        self.threshold = threshold
        self.temperature = temperature

    def soft_step(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable step function using sigmoid."""
        _validate_finite_tensor(x, "x")
        return torch.sigmoid((x - self.threshold) / self.temperature)

    def forward(self, diagram: torch.Tensor, target_betti: torch.Tensor) -> torch.Tensor:
        """Compute MSE between predicted and target Betti numbers.

        :param diagram: Persistence diagram of shape ``(N, 3)`` with columns
            ``(birth, death, dimension)``.
        :param target_betti: 1D tensor of target Betti numbers indexed by dimension.
        :returns: MSE loss between predicted and target Betti numbers.
        :raises ValueError: If ``target_betti`` is empty, not 1D, or contains
            negative values.
        :raises ValueError: If diagram dimensions are not finite non-negative integers
            or if any death is less than its birth.
        """
        persistence = _persistence_values(diagram, min_cols=3)
        dimensions = _validate_diagram_dimensions(diagram)
        _validate_target_betti(target_betti)

        max_dim = target_betti.shape[0] - 1
        pred_betti = torch.zeros_like(target_betti, dtype=torch.float32)

        for d in range(max_dim + 1):
            mask = (dimensions == d).float()
            significant = self.soft_step(persistence) * mask
            pred_betti[d] = significant.sum()
        return F.mse_loss(pred_betti, target_betti.float())


class DiagramComplexityLoss(nn.Module):
    """Regularize diagram complexity by persistence-derived summaries."""

    def __init__(self, measure: str = "total_persistence"):
        """Initialize topological complexity loss.

        :param measure: Complexity measure to use. One of ``"total_persistence"``,
            ``"persistence_entropy"``, ``"num_features"``, ``"max_persistence"``.
        :raises ValueError: If ``measure`` is not a recognized complexity measure.
        """
        super().__init__()
        if measure not in {
            "total_persistence",
            "persistence_entropy",
            "num_features",
            "max_persistence",
        }:
            raise ValueError("unknown complexity measure")
        self.measure = measure

    def forward(self, diagram: torch.Tensor) -> torch.Tensor:
        """Compute the selected complexity measure of a persistence diagram.

        :param diagram: Persistence diagram of shape ``(N, 2)`` or ``(N, 3)``.
        :returns: Scalar complexity value (total persistence, persistence entropy,
            number of features above threshold, or max persistence). Returns zero
            for an empty diagram.
        :raises ValueError: If any birth/death coordinate is non-finite or any
            death is less than its birth.
        """
        persistence = _persistence_values(diagram)
        if diagram.shape[0] == 0:
            return diagram.new_zeros(())

        if self.measure == "total_persistence":
            return persistence.sum()

        if self.measure == "persistence_entropy":
            total = persistence.sum()
            if total <= 0:
                return diagram.new_zeros(())
            p_norm = persistence / total
            entropy = -(p_norm * torch.log(p_norm + 1e-8)).sum()
            return entropy

        if self.measure == "num_features":
            return (persistence > 0.1).float().sum()

        if self.measure == "max_persistence":
            return persistence.max()

        raise ValueError(f"Unknown complexity measure: {self.measure}")


TopologicalComplexityLoss = DiagramComplexityLoss


class StabilityLoss(nn.Module):
    """Penalize sensitivity of persistence to input perturbations."""

    def __init__(self, epsilon: float = 0.01, num_samples: int = 5):
        """Initialize stability loss.

        :param epsilon: Magnitude of random perturbations applied to the input.
        :param num_samples: Number of perturbation samples to average over.
        :raises ValueError: If ``epsilon`` is negative or ``num_samples`` is not
            a positive integer.
        """
        super().__init__()
        epsilon = _validate_non_negative_scalar("epsilon", epsilon)
        num_samples = _validate_positive_int(num_samples, "num_samples")
        self.epsilon = epsilon
        self.num_samples = num_samples

    def forward(self, points: torch.Tensor, persistence_fn: Callable[..., Any]) -> torch.Tensor:
        """Penalize sensitivity of persistence to input perturbations.

        :param points: Input point cloud tensor.
        :param persistence_fn: Callable that maps points to a list of persistence
            diagrams (one per homology dimension).
        :returns: Stability penalty as ``ReLU(average_wasserstein - threshold)``.
        :raises TypeError: If ``persistence_fn`` is not callable.
        :raises ValueError: If ``persistence_fn`` returns a different number of
            diagrams across perturbations, or if any diagram has non-finite
            coordinates.
        """
        _validate_finite_tensor(points, "points")
        if not callable(persistence_fn):
            raise TypeError("persistence_fn must be callable")
        orig_diagrams = persistence_fn(points)
        _validate_diagram_sequence(orig_diagrams, "orig_diagrams")
        stability_loss = points.new_zeros(())

        for _ in range(self.num_samples):
            noise = torch.randn_like(points) * self.epsilon
            perturbed = points + noise
            pert_diagrams = persistence_fn(perturbed)
            _validate_diagram_sequence(pert_diagrams, "pert_diagrams")
            if len(orig_diagrams) != len(pert_diagrams):
                raise ValueError("persistence_fn returned mismatched diagram counts")

            for orig, pert in zip(orig_diagrams, pert_diagrams, strict=False):
                if orig.shape[0] > 0 and pert.shape[0] > 0:
                    dist = PersistenceLoss.diagram_wasserstein(
                        orig[:, :2], pert[:, :2], temperature=0.1
                    )
                    stability_loss += dist

        return F.relu(stability_loss / self.num_samples - self.epsilon * 10)


class MultiScaleTopologyLoss(nn.Module):
    """Optimize diagram agreement at multiple persistence scales."""

    def __init__(self, scales: tuple[float, ...] = (0.01, 0.1, 0.5, 1.0)):
        """Initialize multi-scale topology loss.

        :param scales: Tuple of persistence thresholds defining the scales at which
            diagram agreement is enforced.
        :raises ValueError: If any scale is non-positive or ``scales`` is empty.
        """
        super().__init__()
        if not scales:
            raise ValueError("scales must be non-empty and positive")
        self.scales = tuple(_validate_positive_scalar("scales", scale) for scale in scales)

    def forward(self, diagram: torch.Tensor, target_diagrams: list[torch.Tensor]) -> torch.Tensor:
        """Optimize diagram agreement at multiple persistence scales.

        :param diagram: Persistence diagram of shape ``(N, 2)`` or ``(N, 3)``.
        :param target_diagrams: List of target diagrams, one per scale.
        :returns: Sum of per-scale Wasserstein losses weighted by inverse scale.
        :raises ValueError: If ``target_diagrams`` length does not match ``scales``.
        :raises ValueError: If any diagram birth/death coordinate is non-finite
            or any death is less than its birth.
        """
        persistence = _persistence_values(diagram)
        if len(target_diagrams) != len(self.scales):
            raise ValueError("target_diagrams length must match scales")
        total_loss = diagram.new_zeros(())

        for scale, target in zip(self.scales, target_diagrams, strict=False):
            _persistence_values(target)
            mask = persistence >= scale

            if mask.sum() > 0 and target.shape[0] > 0:
                filtered = diagram[mask][:, :2]
                loss = PersistenceLoss.diagram_wasserstein(
                    filtered, target[:, :2], temperature=scale
                )
                total_loss = total_loss + loss / scale

        return total_loss


class LandscapeLoss(nn.Module):
    """L2 loss between persistence landscapes."""

    def __init__(self, n_layers: int = 5, resolution: int = 100):
        """Initialize landscape loss.

        :param n_layers: Number of landscape layers to compute.
        :param resolution: Number of discretization points for the landscape.
        :raises ValueError: If ``n_layers`` or ``resolution`` is not a positive
            integer.
        """
        super().__init__()
        n_layers = _validate_positive_int(n_layers, "n_layers")
        resolution = _validate_positive_int(resolution, "resolution")
        self.n_layers = n_layers
        self.resolution = resolution

    def landscape(self, diagram: torch.Tensor) -> torch.Tensor:
        """Compute the persistence landscape of a diagram.

        :param diagram: Persistence diagram of shape ``(N, 2)`` or ``(N, 3)``.
        :returns: Persistence landscape tensor with shape determined by
            ``n_layers`` and ``resolution``.
        """
        return compute_persistence_landscape(diagram, self.n_layers, self.resolution)

    def forward(self, diagram1: torch.Tensor, diagram2: torch.Tensor) -> torch.Tensor:
        """L2 distance between landscapes."""
        land1 = self.landscape(diagram1)
        land2 = self.landscape(diagram2)

        return F.mse_loss(land1, land2)
