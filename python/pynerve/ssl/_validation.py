"""Shared validation utilities for SSL tasks."""

from __future__ import annotations

import torch

from .._torch_diagrams import validate_diagram as _validate_diagram


def _validate_ssl_diagram(diagram: torch.Tensor) -> None:
    _validate_diagram(diagram)
    if diagram.shape[0] == 0:
        return
    if not torch.isfinite(diagram[:, :2]).all().item():
        raise ValueError("diagram birth/death coordinates must be finite")
    if not (diagram[:, 1] >= diagram[:, 0]).all().item():
        raise ValueError("diagram deaths must be greater than or equal to births")
