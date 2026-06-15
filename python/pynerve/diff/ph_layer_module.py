"""Differentiable persistent homology layer backed by pynerve_internal."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from torch.autograd import Function

import torch
from torch import Tensor, nn

from .ph_layer import _compute_persistence_backward, _compute_persistence_forward


class DifferentiablePHFunction(Function):
    """Autograd Function for differentiable persistent homology.

    Computes persistence diagrams for a batch of point clouds with a
    differentiable forward pass and a custom backward pass via the
    persistence backward solver.
    """

    @staticmethod
    def forward(
        ctx: Any,
        points: Tensor,
        max_dim: int,
        max_radius: float,
        metric: str,
        reduction: str,
    ) -> tuple[Tensor, ...]:
        """Compute persistence diagrams for a batch of point clouds.

        :param ctx: Autograd context for saving tensors.
        :param points: Point cloud tensor of shape ``(batch, n_points, dim)``.
        :param max_dim: Maximum homology dimension to compute.
        :param max_radius: Filtration cutoff radius.
        :param metric: Distance metric name (e.g. ``"euclidean"``).
        :param reduction: Persistence reduction algorithm (e.g. ``"clearing"``).
        :returns: Tuple of per-dimension diagram tensors.
        :raises ValueError: If ``points`` is not 3D or ``max_dim`` is negative.
        """
        if points.dim() != 3:
            raise ValueError("points must have shape (batch, n_points, dim)")
        if max_dim < 0:
            raise ValueError("max_dim must be non-negative")

        ctx.save_for_backward(points)
        ctx.max_dim = max_dim
        ctx.max_radius = max_radius
        ctx.metric = metric
        ctx.reduction = reduction

        diagrams, pair_counts = _compute_persistence_forward(
            points,
            max_dim,
            "rips",
            return_counts=True,
            max_radius=max_radius,
            metric=metric,
            reduction=reduction,
        )
        ctx.pair_counts = pair_counts
        return tuple(diagrams)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor | None, None, None, None, None]:
        """Backward pass for differentiable persistence.

        :param ctx: Autograd context with saved tensors and metadata.
        :param grad_outputs: Gradients w.r.t. each diagram tensor.
        :returns: Tuple of ``(grad_points, None, None, None, None)`` where
            ``grad_points`` is the gradient w.r.t. the input points tensor.
        """
        (points,) = ctx.saved_tensors
        grad_points = _compute_persistence_backward(
            points,
            grad_outputs,
            ctx.max_dim,
            "rips",
            {
                "max_radius": ctx.max_radius,
                "metric": ctx.metric,
                "reduction": ctx.reduction,
            },
            getattr(ctx, "pair_counts", None),
        )
        return grad_points, None, None, None, None


class DifferentiablePersistentHomology(nn.Module):
    """Differentiable persistence homology layer.

    Wraps :class:`DifferentiablePHFunction` as a standard ``nn.Module``
    for use in PyTorch pipelines.
    """

    def __init__(
        self,
        max_dim: int = 1,
        max_radius: float = float("inf"),
        metric: str = "euclidean",
        reduction: str = "clearing",
    ):
        """Initialize the differentiable persistence layer.

        :param max_dim: Maximum homology dimension (must be non-negative).
        :param max_radius: Filtration cutoff radius (must be positive or
            infinite).
        :param metric: Distance metric name.
        :param reduction: Persistence reduction algorithm.
        :raises ValueError: If ``max_dim`` is negative or ``max_radius`` is
            NaN or non-positive.
        """
        super().__init__()
        max_radius = float(max_radius)
        if max_dim < 0:
            raise ValueError("max_dim must be non-negative")
        if math.isnan(max_radius) or max_radius <= 0:
            raise ValueError("max_radius must be positive or infinite")
        self.max_dim = max_dim
        self.max_radius = max_radius
        self.metric = metric
        self.reduction = reduction

    def forward(self, points: Tensor) -> list[Tensor]:
        """Compute persistence diagrams for a batch of point clouds.

        :param points: Point cloud tensor of shape ``(batch, n_points, dim)``.
        :returns: List of per-dimension diagram tensors.
        :raises ValueError: If ``points`` is not a 3D tensor.
        """
        if points.dim() != 3:
            raise ValueError("points must have shape (batch, n_points, dim)")
        return cast(
            list[Tensor],
            DifferentiablePHFunction.apply(
                points,
                self.max_dim,
                self.max_radius,
                self.metric,
                self.reduction,
            ),
        )


class TopologyLoss(nn.Module):
    """Loss function based on Wasserstein distance to target diagrams."""

    def __init__(
        self,
        target_diagrams: list[Tensor],
        wasserstein_p: float = 2.0,
    ):
        """Initialize the topology loss.

        :param target_diagrams: List of target persistence diagram tensors.
        :param wasserstein_p: Exponent ``p`` for the Wasserstein distance
            (must be positive).
        :raises ValueError: If ``wasserstein_p`` is not positive.
        """
        super().__init__()
        if wasserstein_p <= 0:
            raise ValueError("wasserstein_p must be positive")
        self.target_diagrams = target_diagrams
        self.p = wasserstein_p

    def forward(self, diagrams: list[Tensor]) -> Tensor:
        """Compute the approximate Wasserstein loss against target diagrams.

        :param diagrams: List of predicted diagram tensors.
        :returns: Scalar loss tensor.
        :raises ValueError: If ``diagrams`` is empty.
        """
        if not diagrams:
            raise ValueError("diagrams must be non-empty")
        total_loss = diagrams[0].new_zeros(())

        for current, target in zip(diagrams, self.target_diagrams, strict=False):
            if current.numel() == 0 or target.numel() == 0:
                continue

            dist = self._approximate_wasserstein(current, target)
            total_loss = total_loss + dist

        return total_loss

    def _approximate_wasserstein(self, d1: Tensor, d2: Tensor) -> Tensor:
        if d1.shape[0] == 0 and d2.shape[0] == 0:
            return d1.new_zeros(())
        if d1.shape[0] == 0 or d2.shape[0] == 0:
            non_empty = d1 if d2.shape[0] == 0 else d2
            return cast(
                Tensor, torch.abs(non_empty[:, 1] - non_empty[:, 0]).sum() / float(np.sqrt(2.0))
            )

        dists = torch.cdist(d1, d2, p=self.p)
        min_dists_1 = dists.min(dim=1)[0]
        min_dists_2 = dists.min(dim=0)[0]
        return (min_dists_1.sum() + min_dists_2.sum()) / 2


def topology_regularizer(
    diagrams: list[Tensor], target_betti: list[int], weight: float = 1.0
) -> Tensor:
    """Penalise deviations from a target Betti-number profile.

    :param diagrams: List of persistence diagram tensors.
    :param target_betti: Target Betti numbers per dimension.
    :param weight: Regularisation strength multiplier (must be
        non-negative).
    :returns: Scalar loss tensor.
    :raises ValueError: If ``diagrams`` is empty or ``weight`` is negative.
    """
    if not diagrams:
        raise ValueError("diagrams must be non-empty")
    if weight < 0:
        raise ValueError("weight must be non-negative")
    loss = diagrams[0].new_zeros(())

    for diag, target in zip(diagrams, target_betti, strict=False):
        n_features = diag.shape[0] if diag.numel() > 0 else 0
        loss = loss + weight * (n_features - target) ** 2

    return loss


def persistence_penalty(
    diagrams: list[Tensor], min_persistence: float = 0.1, weight: float = 1.0
) -> Tensor:
    """Penalise features whose persistence is below a minimum threshold.

    :param diagrams: List of persistence diagram tensors.
    :param min_persistence: Minimum allowed persistence (must be
        non-negative).
    :param weight: Penalty strength multiplier (must be non-negative).
    :returns: Scalar loss tensor.
    :raises ValueError: If ``diagrams`` is empty or ``min_persistence`` or
        ``weight`` is negative.
    """
    if not diagrams:
        raise ValueError("diagrams must be non-empty")
    if min_persistence < 0 or weight < 0:
        raise ValueError("min_persistence and weight must be non-negative")
    loss = diagrams[0].new_zeros(())

    for diag in diagrams:
        if diag.numel() == 0:
            continue

        persistence = diag[:, 1] - diag[:, 0]
        loss = loss + weight * torch.relu(min_persistence - persistence).sum()

    return loss


__all__ = [
    "DifferentiablePHFunction",
    "DifferentiablePersistentHomology",
    "TopologyLoss",
    "topology_regularizer",
    "persistence_penalty",
]
