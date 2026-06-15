"""Backend-dispatched persistence-diagram distance computation."""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np

import torch
from torch import Tensor

from ..exceptions import ValidationError
from ..typing import PersistenceDiagramLike
from ._backend import backend
from ._statistics_core import _valid_rows, _validate_stat_diagram

_DISTANCE_BACKEND_ERRORS = (AttributeError, RuntimeError, TypeError)


def _validate_distance_diagram(diagram: Tensor) -> Tensor:
    _validate_stat_diagram(diagram)
    return _validate_finite_deaths(diagram, "diagram")


def _validate_finite_deaths(diagram: Tensor, _name: str) -> Tensor:
    if diagram.numel() > 0 and not torch.isfinite(diagram[:, 1]).all().item():
        finite_mask = torch.isfinite(diagram[:, 1])
        diagram = diagram[finite_mask]
    return diagram


@runtime_checkable
class DiagramDistance(Protocol):
    """Protocol for diagram distance metrics."""

    def __call__(
        self, d1: Tensor | PersistenceDiagramLike, d2: Tensor | PersistenceDiagramLike
    ) -> Tensor:
        """Compute distance between two diagrams."""
        ...


class DistanceMetric(ABC):
    """Abstract base class for distance metrics."""

    def __init__(self) -> None:
        self._warned_implementation = False

    @abstractmethod
    def compute(self, d1: Tensor, d2: Tensor) -> Tensor:
        """Compute distance between two birth/death tensors."""
        ...

    @abstractmethod
    def _compute_torch_c(self, d1: Tensor, d2: Tensor) -> Tensor:
        """Torch C++ backend dispatch."""
        ...

    @abstractmethod
    def _compute_core_c(self, d1: Tensor, d2: Tensor) -> Tensor:
        """Core C++ backend dispatch."""
        ...

    @abstractmethod
    def _compute_python(self, d1: Tensor, d2: Tensor) -> Tensor:
        """Python fallback dispatch."""
        ...

    def _fallback_compute(
        self,
        d1: Tensor,
        d2: Tensor,
        name: str,
        *,
        try_torch_c: bool = True,
    ) -> Tensor:
        """Try torch C++ backend, then core C++ backend, then Python."""
        if try_torch_c and backend.torch_c_available:
            try:
                return self._compute_torch_c(d1, d2)
            except _DISTANCE_BACKEND_ERRORS as exc:
                warnings.warn(
                    f"Torch C backend failed for {name}, falling back: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if backend.core_c_available:
            try:
                return self._compute_core_c(d1, d2)
            except _DISTANCE_BACKEND_ERRORS as exc:
                warnings.warn(
                    f"Core C backend failed for {name}, falling back: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if not self._warned_implementation:
            warnings.warn(
                f"Using Python implementation for {name}. "
                "Install C++ extensions for faster computation.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_implementation = True

        return self._compute_python(d1, d2)

    def __call__(
        self, d1: Tensor | PersistenceDiagramLike, d2: Tensor | PersistenceDiagramLike
    ) -> Tensor:
        """Extract tensors and compute distance."""
        t1 = self._extract_tensor(d1)
        t2 = self._extract_tensor(d2)

        if t1.dim() not in [1, 2] or t2.dim() not in [1, 2]:
            raise ValidationError(f"Diagrams must be 1D or 2D, got {t1.dim()}D and {t2.dim()}D")

        if t1.dim() == 1:
            t1 = t1.unsqueeze(0)
        if t2.dim() == 1:
            t2 = t2.unsqueeze(0)

        if t1.shape[-1] < 2 or t2.shape[-1] < 2:
            raise ValidationError("Diagrams must have at least 2 columns (birth, death)")

        t1 = t1[..., :2]
        t2 = t2[..., :2]
        t1 = _validate_distance_diagram(t1)
        t2 = _validate_distance_diagram(t2)

        return self.compute(t1, t2)

    def _extract_tensor(self, diagram: Tensor | PersistenceDiagramLike) -> Tensor:
        """Extract tensor from diagram or diagram-like object."""
        if isinstance(diagram, Tensor):
            return _valid_rows(diagram) if diagram.dim() == 3 else diagram

        if hasattr(diagram, "_diagrams"):
            return _valid_rows(diagram._diagrams)[..., :2]  # pyright: ignore[reportAttributeAccessIssue]
        if hasattr(diagram, "diagrams"):
            return _valid_rows(diagram.diagrams)[..., :2]  # pyright: ignore[reportAttributeAccessIssue]
        raise ValidationError(f"Cannot extract tensor from {type(diagram)}")


def _sort_diagram_by_persistence(diagram: Tensor) -> Tensor:
    """Sort persistence diagram by descending persistence for stable distance computation."""
    if diagram.shape[0] <= 1:
        return diagram
    persistence = diagram[:, 1] - diagram[:, 0]
    _, indices = torch.sort(persistence, descending=True, stable=True)
    return diagram[indices]


class WassersteinDistance(DistanceMetric):
    """Wasserstein/p-Wasserstein distance between persistence diagrams."""

    def __init__(self, p: float = 2.0, q: float = 2.0) -> None:
        """Initialize Wasserstein distance with exponent p and q. Default p=2, q=2 correspond to standard 2-Wasserstein."""
        super().__init__()
        p = float(p)
        q = float(q)
        if not math.isfinite(p) or p <= 0:
            raise ValueError("p must be finite and positive")
        if math.isnan(q) or q <= 0:
            raise ValueError("q must be positive")
        self.p = p
        self.q = q

    def compute(self, d1: Tensor, d2: Tensor) -> Tensor:
        """Compute Wasserstein distance."""
        d1 = _sort_diagram_by_persistence(d1)
        d2 = _sort_diagram_by_persistence(d2)
        if _linear_sum_assignment() is not None:
            return self._compute_python(d1, d2)
        return self._fallback_compute(d1, d2, "Wasserstein distance", try_torch_c=False)

    def _compute_torch_c(self, d1: Tensor, d2: Tensor) -> Tensor:
        if backend._torch_c is None:
            raise RuntimeError("Torch C backend not available for Wasserstein distance")
        distance: Tensor = backend._torch_c.wasserstein_distance(d1, d2, self.p)
        return distance.to(device=d1.device, dtype=d1.dtype)

    def _compute_core_c(self, d1: Tensor, d2: Tensor) -> Tensor:
        result: Tensor = torch.ops.pynerve.ph_wasserstein(d1, d2, self.p, self.q)  # type: ignore[valid-type]
        return result

    def _compute_python(self, d1: Tensor, d2: Tensor) -> Tensor:
        return _wasserstein_python(d1, d2, self.p, self.q)


class BottleneckDistance(DistanceMetric):
    """Bottleneck distance between persistence diagrams."""

    def __init__(self) -> None:
        """Initialize the bottleneck distance metric."""
        super().__init__()

    def compute(self, d1: Tensor, d2: Tensor) -> Tensor:
        """Compute bottleneck distance."""
        return self._fallback_compute(d1, d2, "bottleneck distance")

    def _compute_torch_c(self, d1: Tensor, d2: Tensor) -> Tensor:
        if backend._torch_c is None:
            raise RuntimeError("Torch C backend not available for bottleneck distance")
        distance: Tensor = backend._torch_c.bottleneck_distance(d1, d2)
        return distance.to(device=d1.device, dtype=d1.dtype)

    def _compute_core_c(self, d1: Tensor, d2: Tensor) -> Tensor:
        result: Tensor = torch.ops.pynerve.ph_bottleneck(d1, d2)  # type: ignore[valid-type]
        return result

    def _compute_python(self, d1: Tensor, d2: Tensor) -> Tensor:
        return _bottleneck_python(d1, d2)


def _finite_points(diagram: Tensor) -> Tensor:
    return diagram[torch.isfinite(diagram[:, 1])]


def _point_distance(d1: Tensor, d2: Tensor, norm: float) -> Tensor:
    delta = torch.abs(d1[:, None, :] - d2[None, :, :])
    if math.isinf(norm):
        return delta.max(dim=2).values
    if norm == 1.0:
        return delta.sum(dim=2)
    if norm == 2.0:
        return torch.sqrt((delta**2).sum(dim=2))
    return (delta**norm).sum(dim=2) ** (1.0 / norm)


def _diagonal_distance(points: Tensor, norm: float) -> Tensor:
    persistence = points[:, 1] - points[:, 0]
    if math.isinf(norm):
        return persistence / 2.0
    return persistence * math.pow(2.0, 1.0 / norm - 1.0)


def _assignment_cost_matrix(d1: Tensor, d2: Tensor, norm: float) -> np.ndarray:
    n = len(d1)
    m = len(d2)
    cost = np.zeros((n + m, n + m), dtype=np.float64)

    if n and m:
        cost[:n, :m] = _point_distance(d1, d2, norm).detach().cpu().numpy()
    if n:
        diag1 = _diagonal_distance(d1, norm).detach().cpu().numpy()
        cost[:n, m:] = diag1[:, None]
    if m:
        diag2 = _diagonal_distance(d2, norm).detach().cpu().numpy()
        cost[n:, :m] = diag2[None, :]

    return cost


def _linear_sum_assignment() -> Any:
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        return linear_sum_assignment
    except ImportError:
        return None


def _greedy_wasserstein(cost: np.ndarray, p: float, n: int, m: int) -> float:
    pairs = sorted((float(cost[i, j]), i, j) for i in range(n) for j in range(m))
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    total = 0.0

    for distance, row, col in pairs:
        if row not in matched_rows and col not in matched_cols:
            matched_rows.add(row)
            matched_cols.add(col)
            total += distance**p

    for row in range(n):
        if row not in matched_rows:
            total += float(cost[row, m] ** p)
    for col in range(m):
        if col not in matched_cols:
            total += float(cost[n, col] ** p)

    return math.pow(total, 1.0 / p)


def _greedy_bottleneck(cost: np.ndarray, n: int, m: int) -> float:
    pairs = sorted((float(cost[i, j]), i, j) for i in range(n) for j in range(m))
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    max_distance = 0.0

    for distance, row, col in pairs:
        if row not in matched_rows and col not in matched_cols:
            matched_rows.add(row)
            matched_cols.add(col)
            max_distance = max(max_distance, distance)

    for row in range(n):
        if row not in matched_rows:
            max_distance = max(max_distance, float(cost[row, m]))
    for col in range(m):
        if col not in matched_cols:
            max_distance = max(max_distance, float(cost[n, col]))

    return max_distance


def _wasserstein_python(d1: Tensor, d2: Tensor, p: float, q: float) -> Tensor:
    d1_finite = _finite_points(d1)
    d2_finite = _finite_points(d2)
    cost = _assignment_cost_matrix(d1_finite, d2_finite, q)
    if cost.size == 0:
        return torch.tensor(0.0, device=d1.device, dtype=d1.dtype)

    assign = _linear_sum_assignment()
    if assign is not None:
        rows, cols = assign(cost**p)
        distance = float((cost[rows, cols] ** p).sum() ** (1.0 / p))
    else:
        distance = _greedy_wasserstein(cost, p, len(d1_finite), len(d2_finite))

    return torch.tensor(distance, device=d1.device, dtype=d1.dtype)


def _bottleneck_python(d1: Tensor, d2: Tensor) -> Tensor:
    d1_finite = _finite_points(d1)
    d2_finite = _finite_points(d2)
    cost = _assignment_cost_matrix(d1_finite, d2_finite, math.inf)
    if cost.size == 0:
        return torch.tensor(0.0, device=d1.device, dtype=d1.dtype)

    assign = _linear_sum_assignment()
    if assign is None:
        distance = _greedy_bottleneck(cost, len(d1_finite), len(d2_finite))
    else:
        distance = float(cost.max())
        for threshold in np.unique(cost):
            blocked = (cost > threshold).astype(np.int8)
            rows, cols = assign(blocked)
            if int(blocked[rows, cols].sum()) == 0:
                distance = float(threshold)
                break

    return torch.tensor(distance, device=d1.device, dtype=d1.dtype)


_wasserstein_distance = WassersteinDistance(p=2.0)
_bottleneck_distance = BottleneckDistance()


def diagram_wasserstein(
    d1: Tensor | PersistenceDiagramLike,
    d2: Tensor | PersistenceDiagramLike,
    p: float = 2.0,
    q: float = 2.0,
) -> Tensor:
    """Compute Wasserstein distance between persistence diagrams.

    :param d1: First persistence diagram (tensor or diagram-like object).
    :param d2: Second persistence diagram (tensor or diagram-like object).
    :param p: Exponent for combining costs (Wasserstein order). Defaults to 2.0.
    :param q: Internal norm exponent. Defaults to 2.0.
    :returns: A scalar tensor with the Wasserstein distance.
    :raises ValidationError: If either diagram is invalid.
    """
    if p != 2.0 or q != 2.0:
        metric = WassersteinDistance(p=p, q=q)
        return metric(d1, d2)
    return _wasserstein_distance(d1, d2)


def diagram_bottleneck(
    d1: Tensor | PersistenceDiagramLike, d2: Tensor | PersistenceDiagramLike
) -> Tensor:
    """Compute Bottleneck distance between persistence diagrams.

    :param d1: First persistence diagram (tensor or diagram-like object).
    :param d2: Second persistence diagram (tensor or diagram-like object).
    :returns: A scalar tensor with the bottleneck distance.
    :raises ValidationError: If either diagram is invalid.
    """
    return _bottleneck_distance(d1, d2)


__all__ = [
    "DistanceMetric",
    "WassersteinDistance",
    "BottleneckDistance",
    "diagram_wasserstein",
    "diagram_bottleneck",
]
