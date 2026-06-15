"""Core statistical operators for persistence diagrams."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any, Literal

import torch

from .._constants import EPS
from ._vectorization_basis import _validate_positive_finite, _validate_positive_int

_SUPPORTED_AMPLITUDE_METRICS = {"persistence", "bottleneck", "wasserstein"}


def _validate_stat_diagram(diagram: torch.Tensor) -> torch.Tensor:
    if diagram.dim() not in (2, 3):
        raise ValueError("diagram must be a 2D or 3D tensor")
    if diagram.shape[-1] < 2:
        raise ValueError("diagram must have at least birth and death columns")
    if not torch.is_floating_point(diagram):
        raise TypeError("diagram must use a floating-point dtype")
    if diagram.numel() == 0:
        return diagram

    work = diagram[..., :2].to(dtype=torch.float64)
    births = work[..., 0]
    deaths = work[..., 1]
    if not torch.isfinite(births).all().item():
        raise ValueError("diagram births must be finite")
    if torch.isnan(deaths).any().item():
        raise ValueError("diagram deaths must not be NaN")
    finite_death_mask = torch.isfinite(deaths)
    if (
        finite_death_mask.any().item()
        and not (deaths[finite_death_mask] >= births[finite_death_mask]).all().item()
    ):
        raise ValueError("diagram finite deaths must be greater than or equal to births")
    return diagram


def _validate_nonnegative_threshold(value: float, name: str) -> float:
    result = float(value)
    if math.isnan(result) or result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_entropy_base(base: float) -> float:
    result = _validate_positive_finite(base, "base")
    if result == 1.0:
        raise ValueError("base must not be 1")
    return result


def _valid_rows(diagram: torch.Tensor) -> torch.Tensor:
    diagram = _validate_stat_diagram(diagram)
    if diagram.numel() == 0:
        return diagram.reshape(0, diagram.shape[-1])
    if diagram.dim() == 3:
        _, max_pairs, _ = diagram.shape
        row_has_value = (diagram[..., 0] != 0) | (diagram[..., 1] != 0)
        if max_pairs == 0:
            return diagram.reshape(0, diagram.shape[-1])
        row_numbers = torch.arange(1, max_pairs + 1, device=diagram.device)
        valid_counts = (row_has_value.long() * row_numbers).max(dim=1).values
        row_indices = torch.arange(max_pairs, device=diagram.device)
        mask = row_indices.unsqueeze(0) < valid_counts.unsqueeze(1)
        return diagram[mask]
    if diagram.dim() != 2:
        raise ValueError("diagram must be a 2D or 3D tensor")
    row_has_value = (diagram[:, 0] != 0) | (diagram[:, 1] != 0)
    if not row_has_value.any():
        return diagram[:0]
    row_numbers = torch.arange(1, diagram.shape[0] + 1, device=diagram.device)
    valid_count = int((row_has_value.long() * row_numbers).max().item())
    return diagram[:valid_count]


def _split_finite_persistence(diagram: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    diagram = _valid_rows(diagram)
    if diagram.numel() == 0:
        return torch.empty(0, dtype=diagram.dtype, device=diagram.device)

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    if dim is not None and diagram.shape[1] >= 3:
        dims = diagram[:, 2].long()
        keep = dims == dim
        births = births[keep]
        deaths = deaths[keep]

    if deaths.numel() > 0 and not torch.isfinite(deaths).all().item():
        raise ValueError("finite-persistence statistics require finite deaths")
    persistence = deaths - births
    return torch.clamp(persistence, min=0)


def _batch_or_scalar(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    @functools.wraps(fn)
    def wrapper(diagram: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if diagram.dim() == 3:
            return torch.stack([fn(diagram[i], *args, **kwargs) for i in range(diagram.shape[0])])
        return fn(diagram, *args, **kwargs)

    return wrapper


@_batch_or_scalar
def total_persistence(
    diagram: torch.Tensor, dim: int | None = None, p: float = 1.0
) -> torch.Tensor:
    """Sum of finite persistences, optionally raised to the power ``p``.

    :param diagram: Persistence diagram tensor (2D). Use 3D for batched input.
    :param dim: Filter to a specific homology dimension, or ``None`` for all.
    :param p: Exponent applied to each persistence value before summing.
    :returns: Scalar total persistence (or 1D tensor for 3D input).
    """
    p = _validate_positive_finite(p, "p")
    persistence = _split_finite_persistence(diagram, dim=dim)
    if persistence.numel() == 0:
        return torch.tensor(0.0, dtype=diagram.dtype, device=diagram.device)
    return persistence.sum() if p == 1.0 else torch.pow(persistence, p).sum()


@_batch_or_scalar
def mean_persistence(diagram: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Mean of finite persistences. Use 3D input for batched computation."""
    persistence = _split_finite_persistence(diagram, dim=dim)
    if persistence.numel() == 0:
        return torch.tensor(0.0, dtype=diagram.dtype, device=diagram.device)
    return persistence.mean()


@_batch_or_scalar
def max_persistence(diagram: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Maximum finite persistence. Use 3D input for batched computation."""
    persistence = _split_finite_persistence(diagram, dim=dim)
    if persistence.numel() == 0:
        return torch.tensor(0.0, dtype=diagram.dtype, device=diagram.device)
    return persistence.max()


@_batch_or_scalar
def persistence_variance(diagram: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Population variance. Use 3D input for batched computation."""
    persistence = _split_finite_persistence(diagram, dim=dim)
    if persistence.numel() < 2:
        return torch.tensor(0.0, dtype=diagram.dtype, device=diagram.device)
    return persistence.var(unbiased=False)


@_batch_or_scalar
def persistence_entropy(
    diagram: torch.Tensor,
    dim: int | None = None,
    base: float = math.e,
) -> torch.Tensor:
    """Entropy. Use 3D input for batched computation."""
    base = _validate_entropy_base(base)
    persistence = _split_finite_persistence(diagram, dim=dim)
    if persistence.numel() == 0:
        return torch.tensor(0.0, dtype=diagram.dtype, device=diagram.device)

    total = persistence.sum()
    if total < EPS:
        return torch.tensor(0.0, dtype=diagram.dtype, device=diagram.device)
    probs = persistence[persistence > 0] / total
    log_probs = torch.log(probs) / torch.log(
        torch.tensor(base, dtype=diagram.dtype, device=diagram.device)
    )
    return -(probs * log_probs).sum()


@_batch_or_scalar
def number_of_features(
    diagram: torch.Tensor,
    dim: int | None = None,
    min_persistence: float = 0.0,
) -> torch.Tensor:
    """Count features whose persistence meets a minimum threshold.

    :param diagram: Persistence diagram tensor (2D or 3D).
    :param dim: Filter to a specific homology dimension, or ``None`` for all.
    :param min_persistence: Minimum persistence threshold (non-negative).
    :returns: Integer count tensor.
    :raises ValueError: If ``min_persistence`` is negative or NaN.
    """
    min_persistence = _validate_nonnegative_threshold(min_persistence, "min_persistence")
    diagram = _valid_rows(diagram)
    if diagram.numel() == 0:
        return torch.tensor(0, device=diagram.device)

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    mask = torch.ones(len(births), dtype=torch.bool, device=diagram.device)
    if dim is not None and diagram.shape[1] >= 3:
        mask &= diagram[:, 2].long() == dim

    persistence = deaths - births
    finite_mask = torch.isfinite(deaths)
    persistence = torch.where(
        finite_mask, persistence, torch.tensor(float("inf"), device=diagram.device)
    )
    mask &= persistence >= min_persistence
    return mask.sum()


def betti_numbers_at_scale(
    diagram: torch.Tensor, threshold: float, dim: int | None = None
) -> torch.Tensor:
    """Betti number at a fixed persistence threshold.

    Equivalent to :func:`number_of_features` with
    ``min_persistence=threshold``.

    :param diagram: Persistence diagram tensor (2D or 3D).
    :param threshold: Minimum persistence threshold.
    :param dim: Filter to a specific homology dimension, or ``None`` for all.
    :returns: Integer count tensor.
    """
    return number_of_features(diagram, dim=dim, min_persistence=threshold)


@_batch_or_scalar
def betti_curve(
    diagram: torch.Tensor, num_samples: int = 100, dim: int | None = None
) -> torch.Tensor:
    """Betti curve sampled at evenly-spaced persistence thresholds.

    :param diagram: Persistence diagram tensor (2D or 3D).
    :param num_samples: Number of threshold samples (positive integer).
    :param dim: Filter to a specific homology dimension, or ``None`` for all.
    :returns: 1D tensor of Betti numbers of length ``num_samples``.
    :raises ValueError: If ``num_samples`` is not positive.
    """
    num_samples = _validate_positive_int(num_samples, "num_samples")
    persistence = _split_finite_persistence(diagram, dim=dim)
    if persistence.numel() == 0:
        return torch.zeros(num_samples, dtype=diagram.dtype, device=diagram.device)

    p_max = persistence.max().item()
    thresholds = torch.linspace(0, p_max, num_samples, dtype=diagram.dtype, device=diagram.device)
    betti_numbers = torch.stack([(persistence > t).sum() for t in thresholds])
    return betti_numbers.to(dtype=diagram.dtype)


@_batch_or_scalar
def amplitude(
    diagram: torch.Tensor,
    metric: Literal["persistence", "bottleneck", "wasserstein"] = "persistence",
    p: float = 2.0,
    dim: int | None = None,
) -> torch.Tensor:
    """Amplitude (norm) of the persistence diagram.

    :param diagram: Persistence diagram tensor (2D or 3D).
    :param metric: Amplitude metric (``"persistence"``, ``"bottleneck"``,
        or ``"wasserstein"``).
    :param p: Exponent for ``"persistence"`` or ``"wasserstein"``
        metrics (must be positive and finite).
    :param dim: Filter to a specific homology dimension, or ``None`` for all.
    :returns: Scalar amplitude (zero if diagram is empty).
    :raises ValueError: If ``metric`` is unsupported or ``p`` is invalid.
    """
    p = _validate_positive_finite(p, "p")
    if metric not in _SUPPORTED_AMPLITUDE_METRICS:
        raise ValueError("metric must be 'persistence', 'bottleneck', or 'wasserstein'")
    persistence = _split_finite_persistence(diagram, dim=dim)
    if persistence.numel() == 0:
        return torch.tensor(0.0, dtype=diagram.dtype, device=diagram.device)
    if metric == "bottleneck":
        return persistence.max()
    if metric == "persistence":
        return persistence.sum() if p == 1.0 else torch.pow(persistence, p).sum()
    sum_pow = persistence.sum() if p == 1.0 else torch.pow(persistence, p).sum()
    return sum_pow if p == 1.0 else torch.pow(sum_pow, 1.0 / p)


__all__ = [
    "total_persistence",
    "mean_persistence",
    "max_persistence",
    "persistence_variance",
    "persistence_entropy",
    "number_of_features",
    "betti_numbers_at_scale",
    "betti_curve",
    "amplitude",
]
