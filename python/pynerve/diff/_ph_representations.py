"""Differentiable persistence-vectorization helpers."""

from __future__ import annotations

import math
from typing import cast

import torch

from .._constants import EPS_1e_6
from .._validation import validate_diagram as _validate_diagram


def compute_persistence_landscape(
    diagram: torch.Tensor, n_layers: int = 5, resolution: int = 100
) -> torch.Tensor:
    if n_layers <= 0 or resolution <= 0:
        raise ValueError("n_layers and resolution must be positive")
    diagram = _validate_diagram(diagram)
    if diagram.shape[0] == 0:
        return diagram.new_zeros((n_layers, resolution))

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    midpoints = (births + deaths) / 2
    heights = (deaths - births) / 2

    x_min = births.min()
    x_max = deaths.max()
    if x_max <= x_min:
        x_max = x_min + diagram.new_tensor(EPS_1e_6)
    t = torch.linspace(
        x_min,
        x_max,
        resolution,
        device=diagram.device,
        dtype=diagram.dtype,
    )

    landscapes_list: list[torch.Tensor] = []
    for midpoint, height in zip(midpoints, heights, strict=False):
        tent = height - torch.abs(t - midpoint)
        landscapes_list.append(torch.clamp(tent, min=0))

    landscapes = torch.stack(landscapes_list)
    landscapes, _ = torch.sort(landscapes, dim=0, descending=True)
    if landscapes.shape[0] < n_layers:
        padding = diagram.new_zeros((n_layers - landscapes.shape[0], resolution))
        landscapes = torch.cat([landscapes, padding], dim=0)
    return cast(torch.Tensor, landscapes[:n_layers])


def persistence_image(
    diagram: torch.Tensor, resolution: int = 20, sigma: float = 0.1
) -> torch.Tensor:
    """Differentiable persistence image for torch tensors.

    .. note::
        The canonical NumPy implementation is in :func:`pynerve._image_utils.persistence_image`.
        For a more feature-rich torch variant see
        :func:`pynerve.torch._persistence_image.persistence_image` (supports
        ``weight_fn``, auto device placement) and
        :func:`pynerve.torch._vectorization_basis.persistence_image` (supports
        ``birth_range``, ``death_range``, custom weight functions).
        This variant is kept minimal for autograd compatibility.
    """
    sigma = float(sigma)
    if resolution <= 0 or sigma <= 0 or not math.isfinite(sigma):
        raise ValueError("resolution and sigma must be finite and positive")
    diagram = _validate_diagram(diagram)
    if diagram.shape[0] == 0:
        return diagram.new_zeros((resolution, resolution))

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    persistence = deaths - births

    b_min, b_max = births.min(), births.max()
    d_min, d_max = deaths.min(), deaths.max()
    if b_max <= b_min:
        b_max = b_min + diagram.new_tensor(EPS_1e_6)
    if d_max <= d_min:
        d_max = d_min + diagram.new_tensor(EPS_1e_6)
    b_range = torch.linspace(b_min, b_max, resolution, device=diagram.device, dtype=diagram.dtype)
    d_range = torch.linspace(d_min, d_max, resolution, device=diagram.device, dtype=diagram.dtype)
    birth_grid, death_grid = torch.meshgrid(b_range, d_range, indexing="ij")

    image = diagram.new_zeros((resolution, resolution))
    for birth, death, weight in zip(births, deaths, persistence, strict=False):
        image += weight * torch.exp(
            -((birth_grid - birth) ** 2 + (death_grid - death) ** 2) / (2 * sigma**2)
        )

    return image
