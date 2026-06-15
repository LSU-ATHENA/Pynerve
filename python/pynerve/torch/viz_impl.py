"""Visualization helpers that expose plotting-ready data arrays."""

from __future__ import annotations

import math

import torch

from . import PersistenceDiagram
from ._viz_data import (
    _as_plot_tensor,
    diagram_to_betti_data,
    diagram_to_heatmap_data,
    diagram_to_histogram_data,
    diagram_to_image_data,
    diagram_to_landscape_data,
    diagram_to_scatter_data,
)


def get_plot_limits(
    diagram: PersistenceDiagram | torch.Tensor,
    padding: float = 0.1,
) -> tuple[float, float, float, float]:
    """Get symmetric axis limits that include the diagonal with padding."""
    padding = float(padding)
    if padding < 0 or not math.isfinite(padding):
        raise ValueError("padding must be finite and non-negative")
    tensor = _as_plot_tensor(diagram)
    births = tensor[..., 0]
    deaths = tensor[..., 1]
    finite_mask = torch.isfinite(deaths)
    if not finite_mask.any():
        return 0.0, 1.0, 0.0, 1.0

    births = births[finite_mask]
    deaths = deaths[finite_mask]
    x_min, x_max = births.min().item(), births.max().item()
    y_min, y_max = deaths.min().item(), deaths.max().item()
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    return min_val, max_val, min_val, max_val


__all__ = [
    "diagram_to_scatter_data",
    "diagram_to_histogram_data",
    "diagram_to_image_data",
    "diagram_to_landscape_data",
    "diagram_to_betti_data",
    "diagram_to_heatmap_data",
    "get_plot_limits",
]
