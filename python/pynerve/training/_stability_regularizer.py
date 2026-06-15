"""Stability regularization for persistence diagrams."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite, sqrt
from typing import Any

import torch
from torch import nn

from .._torch_diagrams import birth_death as _birth_death


class StabilityRegularizer(nn.Module):
    """Regularizes persistence diagram stability under perturbation.

    Perturbs inputs, computes diagram distances (Wasserstein, bottleneck, or L2),
    and penalizes violations of the theoretical stability bound.

    :param epsilon: Magnitude of the additive Gaussian perturbation.
    :param num_perturbations: Number of random perturbations to sample.
    :param norm: Distance norm for diagram comparison (``"wasserstein"``, ``"bottleneck"``, or ``"l2"``).
    :param lambda_reg: Regularization strength.
    :raises ValueError: If any parameter is invalid.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        num_perturbations: int = 5,
        norm: str = "wasserstein",
        lambda_reg: float = 0.1,
    ):
        super().__init__()
        epsilon = float(epsilon)
        lambda_reg = float(lambda_reg)
        if not isfinite(epsilon):
            raise ValueError("epsilon must be finite")
        if not isfinite(lambda_reg):
            raise ValueError("lambda_reg must be finite")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if num_perturbations <= 0:
            raise ValueError("num_perturbations must be positive")
        if norm not in {"wasserstein", "bottleneck", "l2"}:
            raise ValueError("norm must be one of 'wasserstein', 'bottleneck', or 'l2'")
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative")

        self.epsilon = epsilon
        self.num_perturbations = num_perturbations
        self.norm = norm
        self.lambda_reg = lambda_reg

    def compute_theoretical_bound(self, perturbation_magnitude: float, n_points: int) -> float:
        """Compute the theoretical stability bound for a given perturbation.

        For the bottleneck norm, the bound is the perturbation magnitude.
        For Wasserstein, it is magnitude times sqrt(n_points).
        For L2, it is magnitude times 2.0.

        :param perturbation_magnitude: The perturbation strength.
        :param n_points: Number of points in the input.
        :returns: The theoretical stability bound.
        :raises ValueError: If parameters are invalid.
        """
        perturbation_magnitude = float(perturbation_magnitude)
        if not isfinite(perturbation_magnitude):
            raise ValueError("perturbation_magnitude must be finite")
        if n_points <= 0:
            raise ValueError("n_points must be positive")
        if self.norm == "bottleneck":
            return perturbation_magnitude
        if self.norm == "wasserstein":
            return perturbation_magnitude * sqrt(n_points)
        return perturbation_magnitude * 2.0

    def forward(
        self,
        points: torch.Tensor,
        persistence_fn: Callable[..., Any],
        prediction_fn: Callable[..., Any] | None = None,
    ) -> torch.Tensor:
        """Compute the stability regularization loss.

        Perturbs the input, computes persistence diagrams for original and
        perturbed data, measures the diagram distance, and penalizes
        violations of the theoretical bound.

        :param points: Input point cloud tensor.
        :param persistence_fn: A callable that maps points to diagrams.
        :param prediction_fn: Optional callable for task-specific prediction stability.
        :returns: Scalar stability regularization loss.
        :raises TypeError: If ``points`` is not a tensor or callables are not callable.
        :raises ValueError: If ``points`` is empty or contains non-finite values.
        """
        if not isinstance(points, torch.Tensor):
            raise TypeError("points must be a tensor")
        if points.numel() == 0:
            raise ValueError("points must be non-empty")
        if not torch.isfinite(points).all().item():
            raise ValueError("points must contain only finite values")
        if not callable(persistence_fn):
            raise TypeError("persistence_fn must be callable")
        if prediction_fn is not None and not callable(prediction_fn):
            raise TypeError("prediction_fn must be callable")

        orig_diagrams = persistence_fn(points)
        self._validate_diagram_sequence(orig_diagrams, "orig_diagrams")
        orig_predictions = prediction_fn(orig_diagrams) if prediction_fn is not None else None
        n_points = points.shape[-2] if points.dim() >= 2 else points.numel()
        theoretical_bound = points.new_tensor(
            self.compute_theoretical_bound(self.epsilon, n_points)
        )
        violations = []

        for _ in range(self.num_perturbations):
            perturbed = points + torch.randn_like(points) * self.epsilon
            pert_diagrams = persistence_fn(perturbed)
            self._validate_diagram_sequence(pert_diagrams, "pert_diagrams")

            if self.norm == "wasserstein":
                diagram_dist = self.wasserstein_distance(orig_diagrams, pert_diagrams)
            elif self.norm == "bottleneck":
                diagram_dist = self.bottleneck_distance(orig_diagrams, pert_diagrams)
            else:
                diagram_dist = self.l2_diagram_distance(orig_diagrams, pert_diagrams)

            violations.append(torch.relu(diagram_dist - theoretical_bound))

            if prediction_fn is not None and orig_predictions is not None:
                pert_predictions = prediction_fn(pert_diagrams)
                violations.append(torch.norm(orig_predictions - pert_predictions))

        return self.lambda_reg * torch.stack(violations).mean()

    def wasserstein_distance(
        self, diagrams1: list[torch.Tensor], diagrams2: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute the mean 2-Wasserstein distance between two batches of diagrams.

        Uses the Hungarian-style bipartite matching with diagonal projection.

        :param diagrams1: First list of persistence diagrams.
        :param diagrams2: Second list of persistence diagrams.
        :returns: Mean Wasserstein distance as a scalar tensor.
        :raises ValueError: If the diagram lists have different lengths.
        """
        self._validate_diagram_lists(diagrams1, diagrams2)
        if len(diagrams1) != len(diagrams2):
            raise ValueError("diagram lists must have matching lengths")

        distances = []
        for d1, d2 in zip(diagrams1, diagrams2, strict=True):
            d1 = self._birth_death(d1)
            d2 = self._birth_death(d2).to(device=d1.device, dtype=d1.dtype)

            if d1.shape[0] == 0 and d2.shape[0] == 0:
                distances.append(d1.new_zeros(()))
            elif d1.shape[0] == 0:
                distances.append(self._diagonal_distance(d2).sum())
            elif d2.shape[0] == 0:
                distances.append(self._diagonal_distance(d1).sum())
            else:
                pairwise = torch.cdist(d1, d2, p=2)
                forward = pairwise.min(dim=1).values.sum()
                backward = pairwise.min(dim=0).values.sum()
                distances.append(0.5 * (forward + backward))

        return torch.stack(distances).mean()

    def bottleneck_distance(
        self, diagrams1: list[torch.Tensor], diagrams2: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute the bottleneck distance between two batches of diagrams.

        Uses the infinity-norm matching cost rather than L2.

        :param diagrams1: First list of persistence diagrams.
        :param diagrams2: Second list of persistence diagrams.
        :returns: Maximum bottleneck distance across all diagram pairs.
        :raises ValueError: If the diagram lists have different lengths.
        """
        self._validate_diagram_lists(diagrams1, diagrams2)
        if len(diagrams1) != len(diagrams2):
            raise ValueError("diagram lists must have matching lengths")

        distances = []
        for d1, d2 in zip(diagrams1, diagrams2, strict=True):
            d1 = self._birth_death(d1)
            d2 = self._birth_death(d2).to(device=d1.device, dtype=d1.dtype)

            if d1.shape[0] == 0 and d2.shape[0] == 0:
                distances.append(d1.new_zeros(()))
            elif d1.shape[0] == 0:
                distances.append(self._diagonal_distance(d2).max())
            elif d2.shape[0] == 0:
                distances.append(self._diagonal_distance(d1).max())
            else:
                pairwise = torch.cdist(d1, d2, p=float("inf"))
                forward = pairwise.min(dim=1).values.max()
                backward = pairwise.min(dim=0).values.max()
                distances.append(torch.maximum(forward, backward))

        return torch.stack(distances).max()

    def l2_diagram_distance(
        self, diagrams1: list[torch.Tensor], diagrams2: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute a simple L2 diagram distance between two batches of diagrams.

        Matches diagram points by index (sort order) and computes the
        element-wise squared difference. Unmatched points are projected
        to the diagonal.

        :param diagrams1: First list of persistence diagrams.
        :param diagrams2: Second list of persistence diagrams.
        :returns: Mean L2 distance across all diagram pairs.
        :raises ValueError: If the diagram lists have different lengths.
        """
        self._validate_diagram_lists(diagrams1, diagrams2)
        if len(diagrams1) != len(diagrams2):
            raise ValueError("diagram lists must have matching lengths")

        distances = []
        for d1, d2 in zip(diagrams1, diagrams2, strict=True):
            d1 = self._birth_death(d1)
            d2 = self._birth_death(d2).to(device=d1.device, dtype=d1.dtype)
            min_len = min(d1.shape[0], d2.shape[0])
            if min_len == 0:
                distances.append(
                    self._diagonal_distance(d1).sum() + self._diagonal_distance(d2).sum()
                )
            else:
                matched = ((d1[:min_len] - d2[:min_len]) ** 2).sum()
                tail = self._diagonal_distance(d1[min_len:]).sum()
                tail = tail + self._diagonal_distance(d2[min_len:]).sum()
                distances.append(matched + tail)

        return torch.stack(distances).mean()

    @staticmethod
    def _birth_death(diagram: torch.Tensor) -> torch.Tensor:
        pairs = _birth_death(diagram)
        if pairs.shape[0] == 0:
            return pairs
        if not torch.isfinite(pairs).all().item():
            raise ValueError("diagram birth/death coordinates must be finite")
        if not (pairs[:, 1] >= pairs[:, 0]).all().item():
            raise ValueError("diagram deaths must be greater than or equal to births")
        return pairs

    @classmethod
    def _validate_diagram_sequence(cls, diagrams: list[torch.Tensor], name: str) -> None:
        if not isinstance(diagrams, (list, tuple)):
            raise TypeError(f"{name} must be a sequence of tensors")
        if not diagrams:
            raise ValueError(f"{name} must be non-empty")
        for diagram in diagrams:
            cls._birth_death(diagram)

    @classmethod
    def _validate_diagram_lists(
        cls, diagrams1: list[torch.Tensor], diagrams2: list[torch.Tensor]
    ) -> None:
        cls._validate_diagram_sequence(diagrams1, "diagrams1")
        cls._validate_diagram_sequence(diagrams2, "diagrams2")

    @staticmethod
    def _diagonal_distance(diagram: torch.Tensor) -> torch.Tensor:
        if diagram.shape[0] == 0:
            return diagram.new_zeros((0,))
        return torch.abs(diagram[:, 1] - diagram[:, 0]) / sqrt(2.0)
