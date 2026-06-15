"""Training callbacks for topology-aware workflows."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, cast

from torch import Tensor

from .._validation import validate_nonnegative_int, validate_positive_int
from . import statistics as stats


def _validate_nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if result < 0 or not math.isfinite(result):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


class DiagramVisualizationCallback:
    """Callback that logs diagram images/statistics during training.

    Writes persistence images and scalar summaries through a
    ``SummaryWriter``-compatible ``writer`` object.
    """

    def __init__(
        self,
        log_every: int = 10,
        writer: Any = None,
        max_diagrams: int = 4,
    ) -> None:
        """Initialise the visualisation callback.

        :param log_every: Log interval in steps/epochs (positive integer).
        :param writer: An object with ``add_image`` and ``add_scalar``
            methods (e.g. ``SummaryWriter``).  If ``None``, logging is
            skipped.
        :param max_diagrams: Maximum number of diagrams to visualise per
            step (positive integer).
        :raises ValueError: If ``log_every`` or ``max_diagrams`` is not
            positive.
        """
        self.log_every = validate_positive_int(log_every, "log_every")
        self.writer = writer
        self.max_diagrams = validate_positive_int(max_diagrams, "max_diagrams")
        self.step = 0

    def on_batch_end(self, diagram: Tensor | Any, batch_idx: int = 0) -> None:
        """Log persistence images at the end of a batch.

        :param diagram: A :class:`PersistenceDiagram` or raw diagram tensor.
        :param batch_idx: Current batch index.
        :raises ValueError: If ``batch_idx`` is negative.
        """
        batch_idx = validate_nonnegative_int(batch_idx, "batch_idx")
        if batch_idx % self.log_every != 0:
            return
        img = self._diagram_to_image(diagram)
        if self.writer is not None and img is not None:
            self.writer.add_image("diagram", img, self.step)
        self.step += 1

    def on_epoch_end(self, epoch: int, diagram: Tensor | Any) -> None:
        """Log persistence statistics at the end of an epoch.

        :param epoch: Current epoch index.
        :param diagram: A :class:`PersistenceDiagram` or raw diagram tensor.
        :raises ValueError: If ``epoch`` is negative.
        """
        epoch = validate_nonnegative_int(epoch, "epoch")
        if epoch % self.log_every != 0:
            return
        d = diagram.diagrams if hasattr(diagram, "diagrams") else diagram  # pyright: ignore[reportAttributeAccessIssue]
        total_pers = stats.total_persistence(d).item()
        num_features = stats.number_of_features(d).item()
        if self.writer is not None:
            self.writer.add_scalar("diagram/total_persistence", total_pers, epoch)
            self.writer.add_scalar("diagram/num_features", num_features, epoch)

    def _diagram_to_image(self, diagram: Tensor | Any) -> Tensor:
        from . import persistence_image  # noqa: PLC0415

        img = cast(Callable[..., Tensor], persistence_image)(diagram, resolution=(20, 20))
        return img.unsqueeze(0)


class TopologicalEarlyStopping:
    """Early stopping based on topological complexity behavior.

    Monitors the feature count of a persistence diagram and triggers
    stopping when the complexity stabilises or meets a target.
    """

    _VALID_MODES = {"approach", "stabilize", "increase", "decrease"}

    def __init__(
        self,
        patience: int = 10,
        target_complexity: float | None = None,
        mode: Literal["approach", "stabilize", "increase", "decrease"] = "approach",
        min_delta: float = 0.01,
    ) -> None:
        """Initialise the early-stopping monitor.

        :param patience: Number of consecutive stable steps before stopping
            (positive integer).
        :param target_complexity: Target feature count (required for
            ``"approach"`` mode).
        :param mode: Stopping criterion (``"approach"``, ``"stabilize"``,
            ``"increase"``, ``"decrease"``).
        :param min_delta: Minimum change that counts as a meaningful
            difference (non-negative).
        :raises ValueError: If ``patience`` is not positive, ``mode`` is
            unsupported, or ``min_delta`` is invalid.
        """
        if patience <= 0:
            raise ValueError("patience must be positive")
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self._VALID_MODES)}")
        self.patience = validate_positive_int(patience, "patience")
        self.target = (
            None
            if target_complexity is None
            else _validate_nonnegative_finite(target_complexity, "target_complexity")
        )
        self.mode = mode
        self.min_delta = _validate_nonnegative_finite(min_delta, "min_delta")
        self.history: list[float] = []
        self.counter = 0

    def __call__(self, diagram: Tensor | Any) -> bool:
        """Update state and return whether training should stop.

        :param diagram: A :class:`PersistenceDiagram` or raw diagram tensor.
        :returns: ``True`` if the stopping condition has been met.
        """
        d = diagram.diagrams if hasattr(diagram, "diagrams") else diagram  # pyright: ignore[reportAttributeAccessIssue]
        complexity = float(stats.number_of_features(d, min_persistence=0.1).item())
        self.history.append(complexity)

        if self.mode == "approach" and self.target is not None:
            distance = abs(complexity - self.target)
            self.counter = self.counter + 1 if distance < self.min_delta else 0
        elif self.mode == "stabilize" and len(self.history) >= 2:
            change = abs(self.history[-1] - self.history[-2])
            self.counter = self.counter + 1 if change < self.min_delta else 0
        elif self.mode == "increase" and len(self.history) >= 2:
            delta = self.history[-1] - self.history[-2]
            self.counter = self.counter + 1 if delta < self.min_delta else 0
        elif self.mode == "decrease" and len(self.history) >= 2:
            delta = self.history[-2] - self.history[-1]
            self.counter = self.counter + 1 if delta < self.min_delta else 0

        return self.counter >= self.patience

    def reset(self) -> None:
        """Clear the history and counter."""
        self.history = []
        self.counter = 0


__all__ = [
    "DiagramVisualizationCallback",
    "TopologicalEarlyStopping",
]
