"""Topological regularization losses for neural networks."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from .._validation import validate_finite_scalar as _finite_scalar
from .._validation import validate_finite_tensor as _validate_finite_tensor
from .._validation import validate_positive_int as _validate_positive_int
from ..torch._persistence_vr import vr_persistence


def _validate_non_negative_scalar(name: str, value: float) -> float:
    parsed = _finite_scalar(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _validate_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_birth_death(diagram: Tensor, name: str = "diagram") -> Tensor:
    if not isinstance(diagram, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if diagram.dim() != 2 or diagram.shape[1] < 2:
        raise ValueError(f"{name} must have at least birth/death columns")
    if not torch.is_floating_point(diagram):
        raise TypeError(f"{name} must use a floating-point dtype")
    if diagram.numel() == 0:
        return diagram[:, :2]
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    if not torch.isfinite(births).all().item():
        raise ValueError(f"{name} births must be finite")
    if torch.isnan(deaths).any().item():
        raise ValueError(f"{name} deaths must not be NaN")
    finite_deaths = torch.isfinite(deaths)
    if (
        finite_deaths.any().item()
        and not (deaths[finite_deaths] >= births[finite_deaths]).all().item()
    ):
        raise ValueError(f"{name} finite deaths must be greater than or equal to births")
    return diagram[finite_deaths, :2]


def _finite_persistence(diagram: Tensor, name: str = "diagram") -> Tensor:
    pairs = _finite_birth_death(diagram, name)
    if pairs.numel() == 0:
        return pairs.new_empty((0,))
    return pairs[:, 1] - pairs[:, 0]


def _as_batched_points(hidden_states: Tensor) -> Tensor:
    _validate_finite_tensor(hidden_states, "hidden_states")
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)
    if hidden_states.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got {hidden_states.shape}")
    if hidden_states.shape[-2] == 0:
        raise ValueError("hidden_states must contain at least one point")
    return hidden_states


def _deterministic_subsample(points: Tensor, max_points: int) -> Tensor:
    max_points = _validate_positive_int(max_points, "max_points")
    if points.shape[0] <= max_points:
        return points
    indices = (
        torch.linspace(
            0,
            points.shape[0] - 1,
            max_points,
            dtype=torch.float64,
            device=points.device,
        )
        .round()
        .to(torch.long)
    )
    return points[indices]


def _compute_diagram(points: Tensor, max_dim: int, max_points: int = 1000) -> list[Tensor]:
    points = _deterministic_subsample(points, max_points)
    diagram: Any = vr_persistence(
        points.detach(),
        max_dim=max_dim,
        max_radius=float("inf"),
    )
    diagram_tensor: Tensor = diagram.diagrams.squeeze(0)
    mask: Tensor = diagram.mask.squeeze(0)
    valid = diagram_tensor[mask]
    if valid.numel() == 0:
        return [points.new_empty((0, 2)) for _ in range(max_dim + 1)]
    dims = valid[:, 2].long()
    return [valid[dims == dim, :2].to(points.device) for dim in range(max_dim + 1)]


class TopologicalRegularizationLoss(nn.Module):
    """Penalty for matching target Betti counts."""

    def __init__(
        self,
        min_persistence: float = 0.1,
        target_betti: list[int] | None = None,
        max_dim: int = 1,
        weight: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        """Initialize the topological regularization loss.

        :param min_persistence: Minimum persistence threshold for
            significant features.
        :param target_betti: Target Betti numbers per dimension. If
            ``None``, penalizes extreme feature counts (0 or >100).
        :param max_dim: Maximum homology dimension to compute.
        :param weight: Multiplicative weight applied to the loss.
        :param reduction: Reduction mode (``'mean'``, ``'sum'``, or
            ``'none'``).
        """
        super().__init__()
        min_persistence = _validate_non_negative_scalar("min_persistence", min_persistence)
        max_dim = _validate_non_negative_int("max_dim", max_dim)
        weight = _validate_non_negative_scalar("weight", weight)
        if target_betti is not None:
            if any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in target_betti
            ):
                raise ValueError("target_betti values must be non-negative integers")
            target_betti = [int(value) for value in target_betti]
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.min_persistence = min_persistence
        self.target_betti = target_betti
        self.max_dim = max_dim
        self.weight = weight
        self.reduction = reduction

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute the topological regularization loss.

        :param hidden_states: Point cloud tensor of shape
            ``(batch, n_points, dim)`` or ``(n_points, dim)``.
        :returns: Scalar loss tensor if reduction is ``'mean'`` or
            ``'sum'``, otherwise per-sample loss tensor.
        """
        hidden_states = _as_batched_points(hidden_states)
        batch_size = hidden_states.shape[0]
        losses = []

        for b in range(batch_size):
            diagrams = _compute_diagram(hidden_states[b], self.max_dim)
            losses.append(
                self._compute_loss_from_diagrams(
                    diagrams,
                    hidden_states.device,
                    hidden_states.dtype,
                )
            )

        batch_losses = torch.stack(losses)

        if self.reduction == "mean":
            return cast(Tensor, batch_losses.mean() * self.weight)
        if self.reduction == "sum":
            return cast(Tensor, batch_losses.sum() * self.weight)
        return cast(Tensor, batch_losses * self.weight)

    def _compute_loss_from_diagrams(
        self,
        diagrams: list[Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        total_loss = torch.zeros((), device=device, dtype=dtype)

        for dim, diagram in enumerate(diagrams):
            if diagram.numel() == 0:
                continue

            persistence = _finite_persistence(diagram)
            significant = persistence > self.min_persistence
            n_significant = significant.sum().float()

            if self.target_betti is not None and dim < len(self.target_betti):
                target = diagram.new_tensor(float(self.target_betti[dim]))
                betti_loss = torch.abs(n_significant - target)
                total_loss = total_loss + betti_loss
            elif n_significant < 1:
                total_loss = total_loss + total_loss.new_tensor(1.0)
            elif n_significant > 100:
                total_loss = total_loss + (n_significant - 100) * 0.01

        return total_loss


class PersistenceEntropyLoss(nn.Module):
    """Penalty for matching a target persistence entropy."""

    def __init__(self, target_entropy: float = 2.0, max_dim: int = 1, weight: float = 1.0):
        """Initialize the persistence entropy loss.

        :param target_entropy: Target persistence entropy value.
        :param max_dim: Maximum homology dimension to compute.
        :param weight: Multiplicative weight applied to the loss.
        """
        super().__init__()
        target_entropy = _validate_non_negative_scalar("target_entropy", target_entropy)
        max_dim = _validate_non_negative_int("max_dim", max_dim)
        weight = _validate_non_negative_scalar("weight", weight)
        self.target_entropy = target_entropy
        self.max_dim = max_dim
        self.weight = weight

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute the persistence entropy loss.

        :param hidden_states: Point cloud tensor of shape
            ``(batch, n_points, dim)`` or ``(n_points, dim)``.
        :returns: Scalar loss tensor.
        """
        hidden_states = _as_batched_points(hidden_states)
        batch_size = hidden_states.shape[0]
        entropies = []

        for b in range(batch_size):
            diagrams = _compute_diagram(hidden_states[b], self.max_dim)
            entropies.append(
                self._compute_entropy(diagrams, hidden_states.device, hidden_states.dtype)
            )

        batch_entropies = torch.stack(entropies)
        loss = torch.abs(batch_entropies - self.target_entropy).mean()
        return cast(Tensor, loss * self.weight)

    def _compute_entropy(
        self,
        diagrams: list[Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        all_persistence = []

        for diagram in diagrams:
            if diagram.numel() > 0:
                persistence = _finite_persistence(diagram)
                persistence = persistence[persistence > 0]
                if len(persistence) > 0:
                    all_persistence.append(persistence)

        if not all_persistence:
            return torch.zeros((), device=device, dtype=dtype)

        all_p = torch.cat(all_persistence)
        if len(all_p) == 0:
            return torch.zeros((), device=device, dtype=dtype)

        p = all_p / all_p.sum()
        return -(p * torch.log(p + 1e-10)).sum()


class TopologicalComplexityLoss(nn.Module):
    """Penalty for feature counts outside a configured range."""

    def __init__(
        self,
        min_features: int = 5,
        max_features: int = 50,
        min_persistence: float = 0.1,
        max_dim: int = 1,
        weight: float = 1.0,
    ):
        """Initialize the topological complexity loss.

        :param min_features: Minimum acceptable number of significant
            features.
        :param max_features: Maximum acceptable number of significant
            features.
        :param min_persistence: Minimum persistence threshold for
            significant features.
        :param max_dim: Maximum homology dimension to compute.
        :param weight: Multiplicative weight applied to the loss.
        """
        super().__init__()
        min_features = _validate_non_negative_int("min_features", min_features)
        max_features = _validate_non_negative_int("max_features", max_features)
        if min_features < 0 or max_features < min_features:
            raise ValueError("feature bounds must satisfy 0 <= min_features <= max_features")
        min_persistence = _validate_non_negative_scalar("min_persistence", min_persistence)
        max_dim = _validate_non_negative_int("max_dim", max_dim)
        weight = _validate_non_negative_scalar("weight", weight)
        self.min_features = min_features
        self.max_features = max_features
        self.min_persistence = min_persistence
        self.max_dim = max_dim
        self.weight = weight

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute the topological complexity loss.

        :param hidden_states: Point cloud tensor of shape
            ``(batch, n_points, dim)`` or ``(n_points, dim)``.
        :returns: Scalar loss tensor.
        """
        hidden_states = _as_batched_points(hidden_states)
        batch_size = hidden_states.shape[0]
        losses = []

        for b in range(batch_size):
            diagrams = _compute_diagram(hidden_states[b], self.max_dim)
            n_features = self._count_features(diagrams)
            if n_features < self.min_features:
                loss = float(self.min_features - n_features)
            elif n_features > self.max_features:
                loss = float(n_features - self.max_features) * 0.1
            else:
                loss = 0.0
            losses.append(hidden_states.new_tensor(loss))

        return torch.stack(losses).mean() * self.weight

    def _count_features(self, diagrams: list[Tensor]) -> int:
        count = 0
        for diagram in diagrams:
            if diagram.numel() > 0:
                persistence = _finite_persistence(diagram)
                count += int((persistence > self.min_persistence).sum().item())
        return count


class DiagramMatchingLoss(nn.Module):
    """Distance between predicted and target persistence diagrams."""

    def __init__(
        self,
        distance_metric: Literal["wasserstein", "bottleneck"] = "wasserstein",
        p: float = 2.0,
        weight: float = 1.0,
    ):
        """Initialize the diagram matching loss.

        :param distance_metric: Distance metric (``'wasserstein'`` or
            ``'bottleneck'``).
        :param p: Exponent for the p-Wasserstein distance.
        :param weight: Multiplicative weight applied to the loss.
        """
        super().__init__()
        if distance_metric not in {"wasserstein", "bottleneck"}:
            raise ValueError("distance_metric must be 'wasserstein' or 'bottleneck'")
        p = _finite_scalar(p, "p")
        if p <= 0:
            raise ValueError("p must be positive")
        weight = _validate_non_negative_scalar("weight", weight)
        self.distance_metric = distance_metric
        self.p = p
        self.weight = weight

    def forward(
        self, pred_diagrams: list[list[Tensor]], target_diagrams: list[list[Tensor]]
    ) -> Tensor:
        """Compute the diagram matching loss.

        :param pred_diagrams: List of predicted persistence diagram
            batches.
        :param target_diagrams: List of target persistence diagram
            batches.
        :returns: Scalar loss tensor.
        :raises ValueError: If batch sizes do not match or are empty.
        """
        if len(pred_diagrams) != len(target_diagrams):
            raise ValueError("Batch sizes must match")
        if not pred_diagrams:
            raise ValueError("At least one diagram batch is required")

        losses = []
        for pred_batch, target_batch in zip(pred_diagrams, target_diagrams, strict=True):
            batch_loss = self._compute_batch_distance(pred_batch, target_batch)
            losses.append(batch_loss)

        return torch.stack(losses).mean() * self.weight

    def _compute_batch_distance(self, pred: list[Tensor], target: list[Tensor]) -> Tensor:
        if len(pred) != len(target):
            raise ValueError("Diagram dimensions must match")
        if not pred:
            raise ValueError("Diagram batches must be non-empty")
        total_distance: Tensor | None = None

        for p, t in zip(pred, target, strict=True):
            distance = self._diagram_distance(p, t)
            total_distance = distance if total_distance is None else total_distance + distance

        if total_distance is None:
            device = pred[0].device if pred else target[0].device
            return torch.tensor(0.0, device=device)
        return total_distance

    def _diagram_distance(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_xy = _finite_birth_death(pred, "pred")
        target_xy = _finite_birth_death(target, "target")
        device = pred.device if pred.numel() else target.device
        dtype = pred.dtype if pred.is_floating_point() else target.dtype
        if pred_xy.numel() == 0 and target_xy.numel() == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        if pred_xy.numel() == 0:
            return self._diagonal_distance(target_xy)
        if target_xy.numel() == 0:
            return self._diagonal_distance(pred_xy)

        pairwise = torch.cdist(pred_xy, target_xy, p=self.p)
        pred_to_target = pairwise.min(dim=1).values
        target_to_pred = pairwise.min(dim=0).values

        if self.distance_metric == "bottleneck":
            return cast(Tensor, torch.maximum(pred_to_target.max(), target_to_pred.max()))

        pred_cost = pred_to_target.pow(self.p).mean()
        target_cost = target_to_pred.pow(self.p).mean()
        return cast(Tensor, (pred_cost + target_cost).pow(1.0 / self.p))

    def _diagonal_distance(self, diagram: Tensor) -> Tensor:
        diagram = _finite_birth_death(diagram)
        if diagram.numel() == 0:
            return torch.tensor(0.0, device=diagram.device, dtype=diagram.dtype)
        diagonal = (diagram[:, 1] - diagram[:, 0]).abs() / (2.0**0.5)
        if self.distance_metric == "bottleneck":
            return cast(Tensor, diagonal.max())
        return cast(Tensor, diagonal.pow(self.p).mean().pow(1.0 / self.p))


__all__ = [
    "TopologicalRegularizationLoss",
    "PersistenceEntropyLoss",
    "TopologicalComplexityLoss",
    "DiagramMatchingLoss",
]
