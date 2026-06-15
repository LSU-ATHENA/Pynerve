"""Core vectorization primitives for persistence diagrams."""

from __future__ import annotations

import math
from typing import Literal

import torch

from .._validation import validate_positive_int as _validate_positive_int
from ..exceptions import DtypeError, ShapeError, ValidationError
from ..torch._persistence_validators import (
    _validate_image_resolution as _validate_resolution,  # noqa: PLC0415
)

_SUPPORTED_WEIGHT_FNS = {"constant", "linear", "persistence"}


def _validate_diagram(diagram: torch.Tensor, name: str = "diagram") -> torch.Tensor:
    if not isinstance(diagram, torch.Tensor):
        raise DtypeError(f"{name} must be a torch.Tensor")
    if diagram.dim() != 2:
        raise ShapeError(
            f"{name} must be a 2D tensor, got {diagram.dim()}D",
            parameter=name,
            expected_ndim=2,
            actual_ndim=diagram.dim(),
        )
    if diagram.shape[-1] < 2:
        raise ShapeError(
            f"{name} must have at least 2 columns",
            parameter=name,
        )
    if not torch.is_floating_point(diagram):
        raise DtypeError(f"{name} must use a floating-point dtype")
    if diagram.numel() == 0:
        return diagram
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    if not torch.isfinite(births).all().item():
        raise ValidationError("diagram births must be finite", parameter=name)
    if torch.isnan(deaths).any().item():
        raise ValidationError("diagram deaths must not be NaN", parameter=name)
    finite_death_mask = torch.isfinite(deaths)
    if (
        finite_death_mask.any().item()
        and not (deaths[finite_death_mask] >= births[finite_death_mask]).all().item()
    ):
        raise ValidationError(
            "diagram finite deaths must be greater than or equal to births",
            parameter=name,
        )
    return diagram


def _validate_positive_finite(value: float, name: str) -> float:
    result = float(value)
    if result <= 0 or not math.isfinite(result):
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_range(
    range_value: tuple[float, float] | None, name: str
) -> tuple[float, float] | None:
    if range_value is None:
        return None
    if len(range_value) != 2:
        raise ValueError(f"{name} must contain exactly two entries")
    lower = float(range_value[0])
    upper = float(range_value[1])
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"{name} bounds must be finite")
    if lower > upper:
        raise ValueError(f"{name} minimum must not exceed maximum")
    return lower, upper


def _padded_range(values: torch.Tensor) -> tuple[float, float]:
    min_value = float(values.min().item())
    max_value = float(values.max().item())
    span = max_value - min_value
    padding = span * 0.1 if span > 0 else 0.5
    return min_value - padding, max_value + padding


def _finite_birth_death(diagram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite (birth, death) coordinates."""
    diagram = _validate_diagram(diagram)
    if diagram.numel() == 0:
        return torch.empty(0, dtype=diagram.dtype, device=diagram.device), torch.empty(
            0, dtype=diagram.dtype, device=diagram.device
        )
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    if not torch.isfinite(deaths).all().item():
        finite_mask = torch.isfinite(deaths)
        births = births[finite_mask]
        deaths = deaths[finite_mask]
    return births, deaths


def persistence_image(
    diagram: torch.Tensor,
    resolution: tuple[int, int] = (20, 20),
    sigma: float | None = None,
    birth_range: tuple[float, float] | None = None,
    death_range: tuple[float, float] | None = None,
    weight_fn: Literal["constant", "linear", "persistence"] = "persistence",
    normalize: bool = True,
) -> torch.Tensor:
    """Convert a diagram to a Gaussian-smoothed persistence image.

    Supports batched input (3D): returns a 3D tensor ``(batch, H, W)``.
    """
    if diagram.dim() == 3:
        imgs = [
            persistence_image(
                diagram[i], resolution, sigma, birth_range, death_range, weight_fn, normalize
            )
            for i in range(diagram.shape[0])
        ]
        return torch.stack(imgs)
    _validate_diagram(diagram)
    resolution = _validate_resolution(resolution)
    birth_range = _validate_range(birth_range, "birth_range")
    death_range = _validate_range(death_range, "death_range")
    if weight_fn not in _SUPPORTED_WEIGHT_FNS:
        raise ValueError("weight_fn must be 'constant', 'linear', or 'persistence'")
    births, deaths = _finite_birth_death(diagram)
    if births.numel() == 0:
        return torch.zeros(resolution, dtype=diagram.dtype, device=diagram.device)

    if birth_range is None:
        birth_range = _padded_range(births)
    if death_range is None:
        death_range = _padded_range(deaths)
    if sigma is None:
        extent = max(birth_range[1] - birth_range[0], death_range[1] - death_range[0])
        sigma = extent / max(resolution) * 2.0
    sigma = _validate_positive_finite(sigma, "sigma")

    if weight_fn in ("linear", "persistence"):
        weights = torch.clamp(deaths - births, min=0)
    else:
        weights = torch.ones_like(births)

    x = torch.linspace(
        birth_range[0],
        birth_range[1],
        resolution[1],
        dtype=diagram.dtype,
        device=diagram.device,
    )
    y = torch.linspace(
        death_range[0],
        death_range[1],
        resolution[0],
        dtype=diagram.dtype,
        device=diagram.device,
    )
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    image = torch.zeros(resolution, dtype=diagram.dtype, device=diagram.device)

    for w, b, d in zip(weights, births, deaths, strict=True):
        image += w * torch.exp(-((xx - b) ** 2 + (yy - d) ** 2) / (2 * sigma**2))

    image = image.flip(0)
    if normalize and image.sum() > 0:
        image = image / image.sum()
    return image


def adaptive_persistence_image(
    diagram: torch.Tensor,
    target_resolution: int = 20,
    min_sigma: float = 0.1,
    max_sigma: float = 10.0,
) -> torch.Tensor:
    """Adaptively choose resolution and sigma, then compute an image.

    Supports batched input (3D): returns ``(batch, H, H)``.
    """
    if diagram.dim() == 3:
        images = [
            adaptive_persistence_image(diagram[i], target_resolution, min_sigma, max_sigma)
            for i in range(diagram.shape[0])
        ]
        return torch.stack(images)
    _validate_diagram(diagram)
    target_resolution = _validate_positive_int(target_resolution, "target_resolution")
    min_sigma = _validate_positive_finite(min_sigma, "min_sigma")
    max_sigma = _validate_positive_finite(max_sigma, "max_sigma")
    if min_sigma > max_sigma:
        raise ValueError("expected min_sigma <= max_sigma")
    births, deaths = _finite_birth_death(diagram)
    if births.numel() == 0:
        return torch.zeros(
            (target_resolution, target_resolution),
            dtype=diagram.dtype,
            device=diagram.device,
        )

    b_range = births.max() - births.min()
    d_range = deaths.max() - deaths.min()
    spread = torch.maximum(b_range, d_range)
    n_features = len(births)
    spread_value = float(spread.item())
    if spread_value == 0:
        sigma = min_sigma
        resolution = target_resolution
    else:
        estimated_sigma = spread / (
            2 * torch.sqrt(torch.tensor(float(n_features), device=diagram.device))
        )
        sigma = float(torch.clamp(estimated_sigma, min_sigma, max_sigma).item())
        min_resolution = max(1, target_resolution // 2)
        resolution = min(max(int(spread_value / sigma * 2), min_resolution), target_resolution * 2)
    return persistence_image(diagram, resolution=(resolution, resolution), sigma=sigma)


def persistence_landscape(
    diagram: torch.Tensor,
    k: int = 5,
    num_samples: int = 100,
    x_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Compute the first `k` persistence landscape functions.

    Supports batched input (3D): returns ``(batch, k, num_samples)``.
    """
    if diagram.dim() == 3:
        landscapes = [
            persistence_landscape(diagram[i], k, num_samples, x_range)
            for i in range(diagram.shape[0])
        ]
        return torch.stack(landscapes)
    _validate_diagram(diagram)
    k = _validate_positive_int(k, "k")
    num_samples = _validate_positive_int(num_samples, "num_samples")
    x_range = _validate_range(x_range, "x_range")
    births, deaths = _finite_birth_death(diagram)
    if births.numel() == 0:
        return torch.zeros((k, num_samples), dtype=diagram.dtype, device=diagram.device)

    midpoints = (births + deaths) / 2
    heights = (deaths - births) / 2
    if x_range is None:
        x_min = births.min().item()
        x_max = deaths.max().item()
        pad = (x_max - x_min) * 0.1
        x_range = (x_min - pad, x_max + pad)
    x = torch.linspace(
        x_range[0], x_range[1], num_samples, dtype=diagram.dtype, device=diagram.device
    )
    tents = torch.clamp(
        heights.unsqueeze(1) - torch.abs(x.unsqueeze(0) - midpoints.unsqueeze(1)), min=0
    )

    if len(births) >= k:
        landscape, _ = torch.topk(tents, k, dim=0)
        return landscape
    landscape = torch.zeros((k, num_samples), dtype=diagram.dtype, device=diagram.device)
    landscape[: len(births)] = tents
    return landscape


def persistence_silhouette(
    diagram: torch.Tensor,
    num_samples: int = 100,
    weight_fn: Literal["constant", "linear", "persistence"] = "persistence",
    x_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Compute weighted silhouette curve from persistence tents.

    Supports batched input (3D): returns ``(batch, num_samples)``.
    """
    if diagram.dim() == 3:
        silhouettes = [
            persistence_silhouette(diagram[i], num_samples, weight_fn, x_range)
            for i in range(diagram.shape[0])
        ]
        return torch.stack(silhouettes)
    _validate_diagram(diagram)
    num_samples = _validate_positive_int(num_samples, "num_samples")
    x_range = _validate_range(x_range, "x_range")
    if weight_fn not in _SUPPORTED_WEIGHT_FNS:
        raise ValueError("weight_fn must be 'constant', 'linear', or 'persistence'")
    births, deaths = _finite_birth_death(diagram)
    if births.numel() == 0:
        return torch.zeros(num_samples, dtype=diagram.dtype, device=diagram.device)

    midpoints = (births + deaths) / 2
    heights = (deaths - births) / 2
    weights = (
        torch.clamp(deaths - births, min=0)
        if weight_fn in ("linear", "persistence")
        else torch.ones_like(births)
    )
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return torch.zeros(num_samples, dtype=diagram.dtype, device=diagram.device)

    if x_range is None:
        x_min = births.min().item()
        x_max = deaths.max().item()
        pad = (x_max - x_min) * 0.1
        x_range = (x_min - pad, x_max + pad)
    x = torch.linspace(
        x_range[0], x_range[1], num_samples, dtype=diagram.dtype, device=diagram.device
    )
    tents = torch.clamp(
        heights.unsqueeze(1) - torch.abs(x.unsqueeze(0) - midpoints.unsqueeze(1)), min=0
    )
    weighted_tents = tents * weights.unsqueeze(1)
    return weighted_tents.sum(dim=0) / weight_sum


__all__ = [
    "_validate_diagram",
    "_validate_positive_int",
    "persistence_image",
    "adaptive_persistence_image",
    "persistence_landscape",
    "persistence_silhouette",
]
