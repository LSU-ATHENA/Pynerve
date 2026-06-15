"""Persistence image computation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._validation import validate_diagram_array
from .exceptions import InvalidArgumentError
from .typing import PersistenceDiagramLike


def _to_diagram_array(diagram: PersistenceDiagramLike) -> np.ndarray:
    if isinstance(diagram, np.ndarray):
        arr: np.ndarray = diagram
    elif isinstance(diagram, PersistenceDiagramLike) and hasattr(diagram, "pairs_array"):
        arr = diagram.pairs_array
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr, dtype=np.float64)
    elif isinstance(diagram, PersistenceDiagramLike) and hasattr(diagram, "pairs"):
        pairs: Any = diagram.pairs
        try:
            import torch  # noqa: PLC0415

            pairs = (
                pairs.detach().cpu().numpy()
                if isinstance(pairs, torch.Tensor)
                else np.asarray(pairs)
            )
        except ImportError:
            pairs = np.asarray(pairs)
        arr = np.asarray(pairs, dtype=np.float64)
    else:
        arr = np.asarray(diagram, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return validate_diagram_array(arr)


def _normalize_image_resolution(resolution: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(resolution, tuple):
        if len(resolution) != 2:
            raise InvalidArgumentError(
                "resolution tuple must contain (height, width)", parameter="resolution"
            )
        height, width = int(resolution[0]), int(resolution[1])
    else:
        height = width = int(resolution)
    if height <= 0 or width <= 0:
        raise InvalidArgumentError("resolution must be positive", parameter="resolution")
    return height, width


def _finite_range(values: np.ndarray, explicit: tuple[float, float] | None) -> tuple[float, float]:
    if explicit is not None:
        low, high = float(explicit[0]), float(explicit[1])
    elif values.size:
        low, high = float(np.min(values)), float(np.max(values))
    else:
        low, high = 0.0, 1.0
    if not np.isfinite([low, high]).all() or high < low:
        raise InvalidArgumentError(
            "image range bounds must be finite and ordered", parameter="explicit"
        )
    if high == low:
        high = low + 1.0
    return low, high


def persistence_image(
    diagram: PersistenceDiagramLike,
    *,
    resolution: int | tuple[int, int] = 20,
    sigma: float = 0.1,
    birth_range: tuple[float, float] | None = None,
    persistence_range: tuple[float, float] | None = None,
    weight: str = "persistence",
) -> np.ndarray:
    """Convert a NumPy-compatible persistence diagram to a persistence image."""
    sigma_value = float(sigma)
    if not np.isfinite(sigma_value) or sigma_value <= 0.0:
        raise InvalidArgumentError("sigma must be finite and positive", parameter="sigma")
    height, width = _normalize_image_resolution(resolution)
    array = _to_diagram_array(diagram)
    if array.size == 0:
        return np.zeros((height, width), dtype=np.float64)

    births = array[:, 0]
    deaths = array[:, 1]
    persistence = deaths - births
    finite = np.isfinite(deaths) & np.isfinite(persistence) & (persistence > 0.0)
    births = births[finite]
    persistence = persistence[finite]
    if births.size == 0:
        return np.zeros((height, width), dtype=np.float64)

    birth_low, birth_high = _finite_range(births, birth_range)
    persistence_low, persistence_high = _finite_range(persistence, persistence_range)
    birth_axis = np.linspace(birth_low, birth_high, width, dtype=np.float64)
    persistence_axis = np.linspace(persistence_low, persistence_high, height, dtype=np.float64)
    birth_grid, persistence_grid = np.meshgrid(birth_axis, persistence_axis)

    sigma_sq = sigma_value * sigma_value
    if weight not in {"persistence", "uniform"}:
        raise InvalidArgumentError("weight must be 'persistence' or 'uniform'", parameter="weight")

    weights = persistence if weight == "persistence" else np.ones(len(births))
    diff_b = birth_grid[None, :, :] - births[:, None, None]
    diff_p = persistence_grid[None, :, :] - persistence[:, None, None]
    gaussians = weights[:, None, None] * np.exp(-(diff_b**2 + diff_p**2) / (2.0 * sigma_sq))
    image: np.ndarray = gaussians.sum(axis=0)
    return image


__all__ = [
    "persistence_image",
]
