"""Training-time metric helpers for persistence diagrams."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from torch import Tensor

from .._validation import validate_nonnegative_int
from . import statistics as stats

_SUPPORTED_TRACK_STATS = {"total", "mean", "max", "count", "entropy"}


def _validate_nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if result < 0 or not math.isfinite(result):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


class DiagramMetric:
    """Accumulate summary statistics across a stream of diagrams.

    Maintains per-statistic lists that are flushed on :meth:`compute`.
    """

    def __init__(
        self,
        name: str = "diagram",
        dim: int | None = None,
        track_stats: list[str] | None = None,
    ) -> None:
        """Initialise the metric accumulator.

        :param name: Prefix for computed metric keys.
        :param dim: Homology dimension to monitor, or ``None`` for all.
        :param track_stats: Statistic names to track (defaults to
            ``["total", "mean", "max", "count", "entropy"]``).
        :raises TypeError: If ``track_stats`` is not a sequence.
        :raises ValueError: If a statistic name is unsupported.
        """
        self.name = name
        if dim is not None:
            dim = validate_nonnegative_int(dim, "dim")
        track_stats = track_stats or ["total", "mean", "max", "count", "entropy"]
        if isinstance(track_stats, str) or not isinstance(track_stats, Sequence):
            raise TypeError("track_stats must be a sequence of statistic names")
        unknown_stats = set(track_stats) - _SUPPORTED_TRACK_STATS
        if unknown_stats:
            raise ValueError(f"unsupported track_stats: {sorted(unknown_stats)}")
        self.dim = dim
        self.track_stats = track_stats
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated values."""
        self.values: dict[str, list[float]] = {stat: [] for stat in self.track_stats}

    def update(self, diagram: Tensor | Any) -> None:
        """Accumulate statistics from a single diagram.

        :param diagram: A :class:`PersistenceDiagram` or raw diagram tensor.
        """
        d = diagram.diagrams if hasattr(diagram, "diagrams") else diagram  # pyright: ignore[reportAttributeAccessIssue]
        if "total" in self.track_stats:
            self.values["total"].append(float(stats.total_persistence(d, dim=self.dim).item()))
        if "mean" in self.track_stats:
            self.values["mean"].append(float(stats.mean_persistence(d, dim=self.dim).item()))
        if "max" in self.track_stats:
            self.values["max"].append(float(stats.max_persistence(d, dim=self.dim).item()))
        if "count" in self.track_stats:
            self.values["count"].append(float(stats.number_of_features(d, dim=self.dim).item()))
        if "entropy" in self.track_stats:
            self.values["entropy"].append(float(stats.persistence_entropy(d, dim=self.dim).item()))

    def compute(self) -> dict[str, float]:
        """Compute mean and standard deviation of accumulated statistics.

        :returns: Dict mapping ``"{name}_{stat}_mean"`` and
            ``"{name}_{stat}_std"`` to their values. Empty stats are
            omitted.
        """
        result: dict[str, float] = {}
        for stat, values in self.values.items():
            if not values:
                continue
            mean_key = f"{self.name}_{stat}_mean"
            std_key = f"{self.name}_{stat}_std"
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            result[mean_key] = mean
            result[std_key] = variance**0.5
        return result


class TopologicalComplexityMetric:
    """Track feature-count complexity against a target complexity.

    Reports the current complexity, its distance to the target, and the
    running mean.
    """

    def __init__(self, target_complexity: float = 10.0) -> None:
        """Initialise the complexity metric.

        :param target_complexity: Desired number of topological features
            (must be finite and non-negative).
        :raises ValueError: If ``target_complexity`` is NaN or negative.
        """
        target_complexity = float(target_complexity)
        if not math.isfinite(target_complexity):
            raise ValueError("target_complexity must be finite")
        if target_complexity < 0:
            raise ValueError("target_complexity must be non-negative")
        self.target = target_complexity
        self.history: list[float] = []

    def update(self, diagram: Tensor | Any) -> None:
        """Record the feature count from a single diagram.

        :param diagram: A :class:`PersistenceDiagram` or raw diagram tensor.
        """
        d = diagram.diagrams if hasattr(diagram, "diagrams") else diagram  # pyright: ignore[reportAttributeAccessIssue]
        complexity = float(stats.number_of_features(d, min_persistence=0.1).item())
        self.history.append(complexity)

    def compute(self) -> dict[str, float]:
        """Compute the current complexity metrics.

        :returns: Dict with ``"complexity"``, ``"target_distance"``,
            ``"mean_complexity"``.
        """
        if not self.history:
            return {"complexity": 0.0, "target_distance": self.target}
        current = self.history[-1]
        return {
            "complexity": current,
            "target_distance": abs(current - self.target),
            "mean_complexity": sum(self.history) / len(self.history),
        }

    def reset(self) -> None:
        """Clear the history buffer."""
        self.history = []


__all__ = ["DiagramMetric", "TopologicalComplexityMetric"]
