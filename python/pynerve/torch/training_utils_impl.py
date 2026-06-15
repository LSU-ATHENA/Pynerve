"""Training losses, callbacks, and metrics for topology-aware models."""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import torch
from torch import nn

from . import diagram_bottleneck, diagram_wasserstein
from . import statistics as stats
from ._training_callbacks import DiagramVisualizationCallback, TopologicalEarlyStopping
from ._training_helpers import compute_kernel_similarity, topological_batch_loss
from ._training_metrics import DiagramMetric, TopologicalComplexityMetric


def _validate_finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_nonnegative_finite(value: float, name: str) -> float:
    result = _validate_finite_scalar(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_finite_mapping(
    values: dict[str, float] | None, name: str, *, nonnegative: bool = False
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in (values or {}).items():
        scalar = _validate_finite_scalar(value, f"{name}[{key!r}]")
        if nonnegative and scalar < 0:
            raise ValueError(f"{name}[{key!r}] must be non-negative")
        result[key] = scalar
    return result


class DiagramDistanceLoss(nn.Module):
    """Distance loss between predicted and target persistence diagrams."""

    def __init__(
        self,
        metric: Literal["wasserstein", "bottleneck"] = "wasserstein",
        p: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if metric not in {"wasserstein", "bottleneck"}:
            raise ValueError("metric must be 'wasserstein' or 'bottleneck'")
        p = _validate_nonnegative_finite(p, "p")
        if p == 0:
            raise ValueError("p must be positive")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.metric = metric
        self.p = p
        self.reduction = reduction

    def forward(self, pred_diagram: Any, target_diagram: Any) -> torch.Tensor:
        pred = pred_diagram.diagrams if hasattr(pred_diagram, "diagrams") else pred_diagram
        target = target_diagram.diagrams if hasattr(target_diagram, "diagrams") else target_diagram

        if self.metric == "wasserstein":
            distance = diagram_wasserstein(pred, target, p=self.p)
        elif self.metric == "bottleneck":
            distance = diagram_bottleneck(pred, target)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        return (
            distance
            if isinstance(distance, torch.Tensor)
            else torch.tensor(float(distance), dtype=torch.float32)
        )


class TopologicalRegularization(nn.Module):
    """Regularizer that penalizes deviation from target topological complexity."""

    def __init__(
        self,
        target_complexity: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
        penalty_type: Literal["l1", "l2", "smooth"] = "l2",
    ):
        super().__init__()
        if penalty_type not in {"l1", "l2", "smooth"}:
            raise ValueError("penalty_type must be 'l1', 'l2', or 'smooth'")
        self.target = _validate_finite_mapping(target_complexity, "target_complexity")
        self.weights = _validate_finite_mapping(weights, "weights", nonnegative=True)
        self.penalty_type = penalty_type

    def forward(self, diagram: Any) -> torch.Tensor:
        d: torch.Tensor = diagram.diagrams[diagram.mask] if hasattr(diagram, "mask") else diagram
        total_penalty = torch.tensor(0.0, dtype=torch.float32, device=d.device)

        if "h0_count" in self.target:
            total_penalty += self.weights.get("h0_count", 1.0) * self._penalty(
                cast(torch.Tensor, stats.number_of_features(d, dim=0)).float(),
                self.target["h0_count"],
            )
        if "h1_count" in self.target:
            total_penalty += self.weights.get("h1_count", 1.0) * self._penalty(
                cast(torch.Tensor, stats.number_of_features(d, dim=1)).float(),
                self.target["h1_count"],
            )
        if "mean_persistence" in self.target:
            total_penalty += self.weights.get("mean_persistence", 1.0) * self._penalty(
                cast(torch.Tensor, stats.mean_persistence(d)), self.target["mean_persistence"]
            )
        if "total_persistence" in self.target:
            total_penalty += self.weights.get("total_persistence", 1.0) * self._penalty(
                cast(torch.Tensor, stats.total_persistence(d)), self.target["total_persistence"]
            )
        return total_penalty

    def _penalty(self, actual: torch.Tensor, target: float) -> torch.Tensor:
        diff = actual - target
        if self.penalty_type == "l1":
            return torch.abs(diff)
        if self.penalty_type == "l2":
            return diff**2
        if self.penalty_type == "smooth":
            return torch.where(torch.abs(diff) < 1.0, 0.5 * diff**2, torch.abs(diff) - 0.5)
        raise RuntimeError(f"invalid penalty_type: {self.penalty_type}")


class PersistenceCrossEntropy(nn.Module):
    """Cross-entropy with optional topology-based confidence weighting."""

    def __init__(
        self,
        base_loss: str = "ce",
        confidence_weighting: bool = True,
        min_persistence_threshold: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        min_persistence_threshold = _validate_nonnegative_finite(
            min_persistence_threshold, "min_persistence_threshold"
        )
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.confidence_weighting = confidence_weighting
        self.min_threshold = min_persistence_threshold
        self.reduction = reduction
        if base_loss != "ce":
            raise ValueError(f"Unknown base loss: {base_loss}")
        self.base_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        diagrams: Any = None,
    ) -> torch.Tensor:
        loss = self.base_loss(logits, targets)
        if self.confidence_weighting and diagrams is not None:
            if hasattr(diagrams, "total_persistence"):
                total_pers = cast(torch.Tensor, diagrams.total_persistence(p=1.0))
            else:
                d = diagrams
                if d.dim() == 3:
                    total_pers = torch.stack(
                        [cast(torch.Tensor, stats.total_persistence(item, p=1.0)) for item in d]
                    )
                else:
                    total_pers = cast(torch.Tensor, stats.total_persistence(d, p=1.0)).view(1)
            if total_pers.numel() not in {1, loss.shape[0]}:
                raise ValueError("diagram batch size must match logits batch size")
            confidence = torch.sigmoid(total_pers - self.min_threshold)
            if confidence.numel() == loss.shape[0] and loss.dim() > 1:
                confidence = confidence.view(-1, *([1] * (loss.dim() - 1)))
            loss = loss * confidence

        if self.reduction == "mean":
            return cast(torch.Tensor, loss.mean())
        if self.reduction == "sum":
            return cast(torch.Tensor, loss.sum())
        return cast(torch.Tensor, loss)


__all__ = [
    "DiagramDistanceLoss",
    "TopologicalRegularization",
    "PersistenceCrossEntropy",
    "DiagramMetric",
    "TopologicalComplexityMetric",
    "DiagramVisualizationCallback",
    "TopologicalEarlyStopping",
    "compute_kernel_similarity",
    "topological_batch_loss",
]
