"""Data-conversion helpers for plotting persistence diagrams."""

from __future__ import annotations

from typing import Any

import numpy as np

import torch
from torch import Tensor

from . import PersistenceDiagram
from . import vectorization as vec
from ._statistics_core import _validate_stat_diagram
from ._vectorization_basis import _validate_positive_int


def _validate_optional_dim(dim: int | None) -> int | None:
    if dim is None:
        return None
    result = int(dim)
    if result < 0:
        raise ValueError("dim must be non-negative")
    return result


def _validate_viz_tensor(tensor: Tensor) -> Tensor:
    tensor = _validate_stat_diagram(tensor)
    if tensor.numel() > 0 and not torch.isfinite(tensor[..., 1]).all().item():
        raise ValueError("visualization diagrams require finite deaths")
    if tensor.numel() > 0 and tensor.shape[-1] >= 3:
        dims = tensor[..., 2]
        if not torch.isfinite(dims).all().item():
            raise ValueError("diagram dimensions must be finite")
        if not (dims >= 0).all().item():
            raise ValueError("diagram dimensions must be non-negative")
        if not (dims == torch.floor(dims)).all().item():
            raise ValueError("diagram dimensions must be integers")
    return tensor


def _as_tensor(diagram: PersistenceDiagram | Tensor) -> Tensor:
    tensor = diagram.diagrams if isinstance(diagram, PersistenceDiagram) else diagram
    return _validate_viz_tensor(tensor)


def _as_plot_tensor(diagram: PersistenceDiagram | Tensor) -> Tensor:
    if isinstance(diagram, PersistenceDiagram):
        return _validate_viz_tensor(diagram.diagrams[diagram.mask])
    return _as_tensor(diagram)


def _as_single_tensor(diagram: PersistenceDiagram | Tensor) -> Tensor:
    if isinstance(diagram, PersistenceDiagram):
        if diagram.batch_size != 1:
            raise ValueError("expected a single diagram, got a batched diagram tensor")
        return _validate_viz_tensor(diagram.diagrams[0][diagram.mask[0]])
    tensor = _as_tensor(diagram)
    if tensor.dim() == 3:
        if tensor.shape[0] != 1:
            raise ValueError("expected a single diagram, got a batched diagram tensor")
        tensor = tensor.squeeze(0)
    return tensor


def _to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _padded_bounds(values: Tensor) -> tuple[float, float]:
    min_value = float(values.min().item())
    max_value = float(values.max().item())
    span = max_value - min_value
    padding = span * 0.1 if span > 0 else 0.5
    return min_value - padding, max_value + padding


def diagram_to_scatter_data(
    diagram: PersistenceDiagram | Tensor,
    dim: int | None = None,
) -> dict[str, np.ndarray]:
    """Convert diagram to scatter-plot data arrays.

    :param diagram: A :class:`PersistenceDiagram` or raw diagram tensor.
    :param dim: Filter to a specific homology dimension, or ``None`` for all.
    :returns: Dict with ``"births"``, ``"deaths"``, ``"dims"``,
        ``"persistence"`` NumPy arrays.
    :raises ValueError: If ``dim`` is negative or diagram data is invalid.
    """
    dim = _validate_optional_dim(dim)
    tensor = _as_plot_tensor(diagram)
    births = tensor[..., 0]
    deaths = tensor[..., 1]
    persistence = deaths - births
    finite_mask = torch.isfinite(deaths)
    births = births[finite_mask]
    deaths = deaths[finite_mask]
    persistence = persistence[finite_mask]

    if tensor.shape[-1] >= 3:
        dims = tensor[..., 2][finite_mask]
        if dim is not None:
            dim_mask = dims == dim
            births = births[dim_mask]
            deaths = deaths[dim_mask]
            persistence = persistence[dim_mask]
            dims = dims[dim_mask]
    else:
        dims = torch.zeros_like(births)

    return {
        "births": _to_numpy(births),
        "deaths": _to_numpy(deaths),
        "dims": _to_numpy(dims),
        "persistence": _to_numpy(persistence),
    }


def diagram_to_histogram_data(
    diagram: PersistenceDiagram | Tensor,
    num_bins: int = 50,
    dim: int | None = None,
) -> dict[str, Any]:
    """Convert diagram to persistence histogram data.

    :param diagram: A :class:`PersistenceDiagram` or raw diagram tensor.
    :param num_bins: Number of histogram bins (positive integer).
    :param dim: Filter to a specific homology dimension, or ``None`` for all.
    :returns: Dict with ``"values"``, ``"bins"``, ``"title"``.
    :raises ValueError: If ``num_bins`` is not positive or ``dim`` is
        negative.
    """
    num_bins = _validate_positive_int(num_bins, "num_bins")
    dim = _validate_optional_dim(dim)
    tensor = _as_plot_tensor(diagram)
    births = tensor[..., 0]
    deaths = tensor[..., 1]
    persistence = deaths - births
    if dim is not None and tensor.shape[-1] >= 3:
        persistence = persistence[tensor[..., 2] == dim]
    persistence = persistence[torch.isfinite(persistence) & (persistence > 0)]

    values = _to_numpy(persistence)
    if len(values) > 0:
        low, high = _padded_bounds(persistence)
        bins = _to_numpy(torch.linspace(low, high, num_bins + 1))
    else:
        bins = _to_numpy(torch.linspace(0, 1, num_bins + 1))
    title = (
        f"Persistence Distribution (dim={dim})" if dim is not None else "Persistence Distribution"
    )
    return {"values": values, "bins": bins, "title": title}


def diagram_to_image_data(
    diagram: PersistenceDiagram | Tensor,
    resolution: tuple[int, int] = (50, 50),
    sigma: float | None = None,
) -> Tensor:
    """Convert diagram to a persistence image tensor.

    :param diagram: A :class:`PersistenceDiagram` or raw diagram tensor.
    :param resolution: ``(height, width)`` of the output image.
    :param sigma: Gaussian kernel width; defaults to an auto-computed value.
    :returns: Persistence image as a 2D tensor.
    :raises ValueError: If the diagram is batched (batch_size > 1).
    """
    return vec.persistence_image(
        _as_single_tensor(diagram),
        resolution=resolution,
        sigma=sigma,
        weight_fn="persistence",
    )


def diagram_to_landscape_data(
    diagram: PersistenceDiagram | Tensor,
    k: int = 5,
    num_samples: int = 100,
) -> dict[str, Any]:
    """Convert diagram to persistence landscape data.

    :param diagram: A :class:`PersistenceDiagram` or raw diagram tensor.
    :param k: Number of landscape layers.
    :param num_samples: Number of x-axis sample points (positive integer).
    :returns: Dict with ``"landscapes"``, ``"x_values"``, ``"k"``.
    :raises ValueError: If ``num_samples`` is not positive.
    """
    num_samples = _validate_positive_int(num_samples, "num_samples")
    tensor = _as_single_tensor(diagram)
    landscapes = vec.persistence_landscape(tensor, k=k, num_samples=num_samples)
    births = tensor[..., 0]
    deaths = tensor[..., 1]
    finite_mask = torch.isfinite(deaths)
    if finite_mask.any():
        x_min = births[finite_mask].min().item()
        x_max = deaths[finite_mask].max().item()
        x_values = torch.linspace(
            x_min, x_max, num_samples, dtype=tensor.dtype, device=tensor.device
        )
    else:
        x_values = torch.linspace(0, 1, num_samples, dtype=tensor.dtype, device=tensor.device)
    return {
        "landscapes": _to_numpy(landscapes),
        "x_values": _to_numpy(x_values),
        "k": k,
    }


def diagram_to_betti_data(
    diagram: PersistenceDiagram | Tensor,
    num_samples: int = 100,
    dim: int | None = None,
) -> dict[str, Any]:
    """Convert diagram to Betti curve data.

    :param diagram: A :class:`PersistenceDiagram` or raw diagram tensor.
    :param num_samples: Number of threshold samples (positive integer).
    :param dim: Filter to a specific homology dimension, or ``None`` for all.
    :returns: Dict with ``"thresholds"``, ``"betti_numbers"``.
    :raises ValueError: If ``num_samples`` is not positive.
    """
    from . import statistics  # noqa: PLC0415

    num_samples = _validate_positive_int(num_samples, "num_samples")
    dim = _validate_optional_dim(dim)
    tensor = _as_single_tensor(diagram)
    betti = statistics.betti_curve(tensor, num_samples=num_samples, dim=dim)
    births = tensor[..., 0]
    deaths = tensor[..., 1]
    finite_mask = torch.isfinite(deaths)

    if finite_mask.any():
        max_pers = (deaths - births)[finite_mask].max().item()
        thresholds = torch.linspace(
            0, max_pers, num_samples, dtype=tensor.dtype, device=tensor.device
        )
    else:
        thresholds = torch.linspace(0, 1, num_samples, dtype=tensor.dtype, device=tensor.device)
    return {
        "thresholds": _to_numpy(thresholds),
        "betti_numbers": _to_numpy(betti),
    }


def diagram_to_heatmap_data(
    diagram: PersistenceDiagram | Tensor,
    grid_size: int = 20,
) -> dict[str, Any]:
    """Convert diagram to 2D histogram (heatmap) data.

    :param diagram: A :class:`PersistenceDiagram` or raw diagram tensor.
    :param grid_size: Number of bins per axis (positive integer).
    :returns: Dict with ``"grid"``, ``"birth_edges"``, ``"death_edges"``,
        ``"birth_min"``, ``"birth_max"``, ``"death_min"``, ``"death_max"``.
    :raises ValueError: If ``grid_size`` is not positive.
    """
    grid_size = _validate_positive_int(grid_size, "grid_size")
    tensor = _as_plot_tensor(diagram)
    births = tensor[..., 0]
    deaths = tensor[..., 1]
    finite_mask = torch.isfinite(deaths)
    births = births[finite_mask]
    deaths = deaths[finite_mask]

    if len(births) == 0:
        return {
            "grid": _to_numpy(
                torch.zeros((grid_size, grid_size), dtype=tensor.dtype, device=tensor.device)
            ),
            "birth_edges": _to_numpy(
                torch.linspace(0, 1, grid_size + 1, dtype=tensor.dtype, device=tensor.device)
            ),
            "death_edges": _to_numpy(
                torch.linspace(0, 1, grid_size + 1, dtype=tensor.dtype, device=tensor.device)
            ),
        }

    b_min, b_max = _padded_bounds(births)
    d_min, d_max = _padded_bounds(deaths)

    birth_edges = torch.linspace(
        b_min, b_max, grid_size + 1, dtype=tensor.dtype, device=tensor.device
    )
    death_edges = torch.linspace(
        d_min, d_max, grid_size + 1, dtype=tensor.dtype, device=tensor.device
    )
    grid = torch.zeros((grid_size, grid_size), dtype=tensor.dtype, device=tensor.device)

    births_a = _to_numpy(births)
    deaths_a = _to_numpy(deaths)
    birth_edges_a = _to_numpy(birth_edges)
    death_edges_a = _to_numpy(death_edges)
    for i in range(len(births)):
        b_idx = min(int((births_a[i] - b_min) / (b_max - b_min) * grid_size), grid_size - 1)
        d_idx = min(int((deaths_a[i] - d_min) / (d_max - d_min) * grid_size), grid_size - 1)
        grid[d_idx, b_idx] += 1

    return {
        "grid": _to_numpy(grid),
        "birth_edges": birth_edges_a,
        "death_edges": death_edges_a,
        "birth_min": b_min,
        "birth_max": b_max,
        "death_min": d_min,
        "death_max": d_max,
    }


__all__ = [
    "diagram_to_scatter_data",
    "diagram_to_histogram_data",
    "diagram_to_image_data",
    "diagram_to_landscape_data",
    "diagram_to_betti_data",
    "diagram_to_heatmap_data",
]
