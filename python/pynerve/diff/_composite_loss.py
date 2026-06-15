"""Composite topology loss combining multiple components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from ._diagram_distances import PersistenceLoss
from ._loss_helpers import _validate_non_negative_scalar
from ._loss_modules import BettiNumberLoss, DiagramComplexityLoss, StabilityLoss


class TopologyLoss(nn.Module):
    """Combined topology loss with configurable weights."""

    def __init__(
        self,
        wasserstein_weight: float = 1.0,
        betti_weight: float = 0.1,
        complexity_weight: float = 0.01,
        stability_weight: float = 0.0,
    ):
        """Initialize combined topology loss.

        :param wasserstein_weight: Weight for the Wasserstein distance component.
        :param betti_weight: Weight for the Betti number matching component.
        :param complexity_weight: Weight for the topological complexity
            regularization component.
        :param stability_weight: Weight for the stability penalty component.
        :raises ValueError: If any weight is negative.
        """
        super().__init__()
        wasserstein_weight = _validate_non_negative_scalar("wasserstein_weight", wasserstein_weight)
        betti_weight = _validate_non_negative_scalar("betti_weight", betti_weight)
        complexity_weight = _validate_non_negative_scalar("complexity_weight", complexity_weight)
        stability_weight = _validate_non_negative_scalar("stability_weight", stability_weight)

        self.wasserstein_weight = wasserstein_weight
        self.betti_weight = betti_weight
        self.complexity_weight = complexity_weight
        self.stability_weight = stability_weight

        self.betti_loss = BettiNumberLoss()
        self.complexity_loss = DiagramComplexityLoss()

    def forward(
        self,
        pred_diagram: torch.Tensor,
        target_diagram: torch.Tensor,
        target_betti: torch.Tensor | None = None,
        points: torch.Tensor | None = None,
        persistence_fn: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Compute the combined topology loss with configurable components.

        :param pred_diagram: Predicted persistence diagram of shape ``(N, 2)`` or ``(N, 3)``.
        :param target_diagram: Target persistence diagram.
        :param target_betti: Optional target Betti numbers for the Betti loss
            component. Required when ``betti_weight > 0``.
        :param points: Optional input point cloud for the stability loss component.
            Required when ``stability_weight > 0``.
        :param persistence_fn: Optional callable for stability loss computation.
            Required when ``stability_weight > 0``.
        :returns: Dictionary with per-component loss values and a ``"total"`` key
            containing the weighted sum.
        """
        losses = {}

        if self.wasserstein_weight > 0:
            losses["wasserstein"] = PersistenceLoss.diagram_wasserstein(
                pred_diagram, target_diagram
            )

        if self.betti_weight > 0 and target_betti is not None:
            losses["betti"] = self.betti_loss(pred_diagram, target_betti)

        if self.complexity_weight > 0:
            losses["complexity"] = self.complexity_loss(pred_diagram)

        if self.stability_weight > 0 and points is not None and persistence_fn is not None:
            stab_loss = StabilityLoss()
            losses["stability"] = stab_loss(points, persistence_fn)

        total = pred_diagram.new_zeros(())
        if "wasserstein" in losses:
            total = total + self.wasserstein_weight * losses["wasserstein"]
        if "betti" in losses:
            total = total + self.betti_weight * losses["betti"]
        if "complexity" in losses:
            total = total + self.complexity_weight * losses["complexity"]
        if "stability" in losses:
            total = total + self.stability_weight * losses["stability"]

        losses["total"] = total

        return losses
