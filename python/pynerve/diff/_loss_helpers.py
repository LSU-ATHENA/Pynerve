"""Validation helpers for topology loss functions."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .._validation import validate_diagram as _validate_diagram
from .._validation import validate_finite_scalar as _finite_scalar
from .._validation import validate_finite_tensor as _validate_finite_tensor


def _validate_non_negative_scalar(name: str, value: float) -> float:
    parsed = _finite_scalar(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _validate_positive_scalar(name: str, value: float) -> float:
    parsed = _finite_scalar(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _persistence_values(diagram: torch.Tensor, min_cols: int = 2) -> torch.Tensor:
    _validate_diagram(diagram, min_cols=min_cols)
    birth_death = diagram[:, :2]
    if not torch.isfinite(birth_death).all().item():
        raise ValueError("diagram birth/death coordinates must be finite")
    persistence = birth_death[:, 1] - birth_death[:, 0]
    if (persistence < 0).any().item():
        raise ValueError("diagram deaths must be greater than or equal to births")
    return persistence


def _validate_diagram_dimensions(diagram: torch.Tensor) -> torch.Tensor:
    dimensions = diagram[:, 2]
    if not torch.isfinite(dimensions).all().item() or (dimensions < 0).any().item():
        raise ValueError("diagram dimensions must be finite and non-negative")
    if not torch.allclose(dimensions, dimensions.round()):
        raise ValueError("diagram dimensions must be integer-valued")
    return dimensions.long()


def _validate_target_betti(target_betti: torch.Tensor) -> None:
    _validate_finite_tensor(target_betti, "target_betti")
    if target_betti.dim() != 1 or target_betti.numel() == 0:
        raise ValueError("target_betti must be a non-empty 1D tensor")
    if (target_betti < 0).any().item():
        raise ValueError("target_betti must contain only non-negative values")


def _validate_diagram_sequence(diagrams: Sequence[torch.Tensor], name: str) -> None:
    if not isinstance(diagrams, (list, tuple)):
        raise TypeError(f"{name} must be a sequence of tensors")
    if not diagrams:
        raise ValueError(f"{name} must be non-empty")
    for diagram in diagrams:
        _persistence_values(diagram)
