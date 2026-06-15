"""Internal utilities for topology samplers."""

from __future__ import annotations

import numpy as np

import torch

from .._torch_diagrams import persistence as _persistence_values
from .._validation import validate_diagram as _validate_torch_diagram


def _validate_diagram(diagram: torch.Tensor) -> None:
    _validate_torch_diagram(diagram, name="diagram")
    if diagram.shape[0] == 0:
        return
    if not torch.isfinite(diagram[:, 2]).all().item() or (diagram[:, 2] < 0).any().item():
        raise ValueError("diagram dimensions must be finite and non-negative")


def _persistence(diagram: torch.Tensor) -> torch.Tensor:
    _validate_diagram(diagram)
    return _persistence_values(diagram)


def _shuffle(values: list[int], rng: np.random.Generator) -> list[int]:
    values = list(values)
    rng.shuffle(values)
    return values
