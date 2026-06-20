"""Spectral and histogram vectorizers for persistence diagrams."""

from __future__ import annotations

from typing import Literal

import torch

from ._vectorization_basis import (
    _finite_birth_death,
    _validate_diagram,
    _validate_positive_finite,
    _validate_positive_int,
)

_SUPPORTED_STATISTICS = {"count", "mean", "sum"}


def _validate_t_values(t_values: torch.Tensor | None, diagram: torch.Tensor) -> torch.Tensor:
    if t_values is None:
        return torch.logspace(-2, 0, 10, dtype=diagram.dtype, device=diagram.device)
    if not isinstance(t_values, torch.Tensor):
        raise TypeError("t_values must be a tensor")
    if t_values.dim() != 1:
        raise ValueError("t_values must be a rank-1 tensor")
    if not torch.is_floating_point(t_values):
        raise TypeError("t_values must use a floating-point dtype")
    if t_values.numel() == 0:
        raise ValueError("t_values must not be empty")
    if not torch.isfinite(t_values).all().item():
        raise ValueError("t_values must be finite")
    if not (t_values > 0).all().item():
        raise ValueError("t_values must be positive")
    return t_values.to(dtype=diagram.dtype, device=diagram.device)


def heat_kernel_signature(
    diagram: torch.Tensor,
    num_samples: int = 100,
    sigma: float = 0.1,
    t_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute heat-kernel signatures over persistence values.

    Supports batched input (3D): returns ``(batch, n_t, num_samples)``.
    """
    if diagram.dim() == 3:
        sigs = [
            heat_kernel_signature(diagram[i], num_samples, sigma, t_values)
            for i in range(diagram.shape[0])
        ]
        return torch.stack(sigs)
    num_samples = _validate_positive_int(num_samples, "num_samples")
    sigma = _validate_positive_finite(sigma, "sigma")
    diagram = _validate_diagram(diagram)
    t_values = _validate_t_values(t_values, diagram)
    assert t_values is not None
    births, deaths = _finite_birth_death(diagram)
    if births.numel() == 0:
        return torch.zeros(
            (t_values.numel(), num_samples), dtype=diagram.dtype, device=diagram.device
        )

    persistence = deaths - births
    p_min, p_max = persistence.min().item(), persistence.max().item()
    if p_max == p_min:
        p_min -= 0.5
        p_max += 0.5
    x = torch.linspace(p_min, p_max, num_samples, dtype=diagram.dtype, device=diagram.device)
    signatures: list[torch.Tensor] = []

    for t in t_values:
        heat = torch.zeros(num_samples, dtype=diagram.dtype, device=diagram.device)
        effective_sigma = sigma * torch.sqrt(t)
        for p in persistence:
            heat += torch.exp(-((x - p) ** 2) / (2 * effective_sigma**2))
        signatures.append(heat)
    return torch.stack(signatures)


def birth_death_curve(
    diagram: torch.Tensor,
    num_bins: int = 50,
    statistic: Literal["count", "mean", "sum"] = "count",
) -> torch.Tensor:
    """Bin births and aggregate counts/mean/sum persistence per bin.

    Supports batched input (3D): returns ``(batch, num_bins)``.
    """
    if diagram.dim() == 3:
        curves = [
            birth_death_curve(diagram[i], num_bins, statistic) for i in range(diagram.shape[0])
        ]
        return torch.stack(curves)
    num_bins = _validate_positive_int(num_bins, "num_bins")
    if statistic not in _SUPPORTED_STATISTICS:
        raise ValueError("statistic must be 'count', 'mean', or 'sum'")
    diagram = _validate_diagram(diagram)
    births, deaths = _finite_birth_death(diagram)
    if births.numel() == 0:
        return torch.zeros(num_bins, dtype=diagram.dtype, device=diagram.device)

    persistence = torch.clamp(deaths - births, min=0)
    b_min, b_max = births.min(), births.max()
    if float(b_max - b_min) < 1e-12:
        bin_idx = torch.zeros_like(births, dtype=torch.long)
    else:
        normalized = (births - b_min) / (b_max - b_min)
        bin_idx = torch.clamp((normalized * num_bins).long(), min=0, max=num_bins - 1)

    counts = torch.zeros(num_bins, dtype=diagram.dtype, device=diagram.device)
    counts.scatter_add_(0, bin_idx, torch.ones_like(births, dtype=diagram.dtype))
    if statistic == "count":
        return counts

    sums = torch.zeros(num_bins, dtype=diagram.dtype, device=diagram.device)
    sums.scatter_add_(0, bin_idx, persistence)
    if statistic == "sum":
        return sums
    means = torch.zeros_like(sums)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]
    return means


__all__ = ["heat_kernel_signature", "birth_death_curve"]
