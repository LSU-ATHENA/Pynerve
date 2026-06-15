"""Vietoris-Rips persistence with autograd support."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ._diagram import PersistenceDiagram
from ._persistence_python import compute_vr_python
from ._persistence_validators import (
    _count_pairs_by_dimension,
    _torch_backend,
    _validate_max_dim,
    _validate_max_radius,
    _validate_metric,
)


class _VRPersistenceFunction(torch.autograd.Function):
    """Autograd function for differentiable Vietoris-Rips persistence."""

    @staticmethod
    def forward(
        ctx: Any,
        points: Tensor,
        max_dim: int,
        max_radius: float,
        metric: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        max_dim = _validate_max_dim(max_dim)
        max_radius = _validate_max_radius(max_radius)
        metric = _validate_metric(metric)
        ctx.max_dim = max_dim
        ctx.max_radius = max_radius
        ctx.metric = metric

        torch_c = _torch_backend()
        if torch_c is not None:
            result = torch_c.vr_persistence_forward(points, max_dim, max_radius, metric, 0, 0.1)
            diagram = result[0]
            mask = result[1]
            num_pairs = result[2]
            birth_idx = result[3] if len(result) > 3 else None
            death_idx = result[4] if len(result) > 4 else None
            if birth_idx is not None and death_idx is not None:
                ctx.save_for_backward(points, birth_idx, death_idx)
                ctx.has_indices = True
            else:
                ctx.save_for_backward(points)
                ctx.has_indices = False
            ctx.has_native_backward = (
                hasattr(torch_c, "vr_persistence_backward") and ctx.has_indices
            )
        else:
            diagram = compute_vr_python(
                points, max_dim=max_dim, metric=metric, max_radius=max_radius
            )
            mask = torch.ones(diagram.shape[:2], dtype=torch.bool, device=diagram.device)
            num_pairs = _count_pairs_by_dimension(diagram, mask, max_dim)
            ctx.save_for_backward(points)
            ctx.has_indices = False
            ctx.has_native_backward = False
        ctx.mark_non_differentiable(mask, num_pairs)
        return diagram, mask, num_pairs

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any,
        grad_output: Tensor,
        _grad_mask: Tensor | None = None,
        _grad_num_pairs: Tensor | None = None,
    ) -> tuple[Tensor, None, None, None]:
        if ctx.has_indices:
            points, birth_idx, death_idx = ctx.saved_tensors
        else:
            points = ctx.saved_tensors[0]
            birth_idx = None
            death_idx = None

        torch_c = _torch_backend()
        if torch_c is not None and ctx.has_native_backward:
            empty_indices = torch.empty(0, dtype=torch.long, device=points.device)
            grad_points = torch_c.vr_persistence_backward(
                grad_output,
                points,
                birth_idx if birth_idx is not None else empty_indices,
                death_idx if death_idx is not None else empty_indices,
            )
        elif torch_c is not None and ctx.has_indices:
            grad_points = torch_c.vr_persistence_backward(grad_output, points, birth_idx, death_idx)
        else:
            grad_points = _compute_zerod_grad_analytical(points, birth_idx, death_idx, grad_output)
        return grad_points, None, None, None


def _compute_zerod_grad_analytical(
    points: Tensor,
    birth_idx: Tensor | None,
    death_idx: Tensor | None,
    grad_output: Tensor,
) -> Tensor:
    """Approximate analytical gradient python implementation for 0D persistence."""
    batch_size, n_points, _ = points.shape
    grad_points = torch.zeros_like(points)

    if birth_idx is None or death_idx is None:
        return grad_points

    for b in range(batch_size):
        for i in range(grad_output.shape[1]):
            if birth_idx[b, i] < 0 or death_idx[b, i] < 0:
                continue
            birth_grad = grad_output[b, i, 0]
            death_grad = grad_output[b, i, 1]
            b_idx = int(birth_idx[b, i].item())
            d_idx = int(death_idx[b, i].item())

            if 0 <= b_idx < n_points:
                grad_points[b, b_idx] -= birth_grad * 0.5
            if 0 <= d_idx < n_points:
                grad_points[b, d_idx] -= birth_grad * 0.5
            if 0 <= b_idx < n_points:
                grad_points[b, b_idx] += death_grad * 0.5
            if 0 <= d_idx < n_points:
                grad_points[b, d_idx] += death_grad * 0.5
    return grad_points


def vr_persistence(
    points: Tensor,
    max_dim: int = 1,
    max_radius: float = float("inf"),
    metric: str = "euclidean",
    return_simplices: bool = False,
) -> PersistenceDiagram | tuple[Any, ...]:
    """Compute Vietoris-Rips persistence from point clouds."""
    if not isinstance(return_simplices, bool):
        raise TypeError("return_simplices must be a boolean")
    max_dim = _validate_max_dim(max_dim)
    max_radius = _validate_max_radius(max_radius)
    metric = _validate_metric(metric)

    single_input = False
    if points.dim() == 2:
        points = points.unsqueeze(0)
        single_input = True
    if points.dim() != 3:
        raise ValueError(f"Expected 2D or 3D input, got {points.dim()}D")
    if points.shape[0] == 0:
        raise ValueError("points must contain at least one batch item")
    if points.shape[1] == 0:
        raise ValueError("points must contain at least one point")
    if points.shape[2] == 0:
        raise ValueError("points must contain at least one coordinate dimension")

    supported_dtypes = {torch.float32, torch.float64}
    if points.dtype == torch.bfloat16:
        if not points.is_cuda or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 requires CUDA with bfloat16 support")
        supported_dtypes.add(torch.bfloat16)
    if points.dtype not in supported_dtypes:
        raise TypeError(f"Unsupported dtype {points.dtype}. Use float32, float64, or bfloat16")
    if not torch.isfinite(points).all().item():
        raise ValueError("points must contain only finite coordinates")

    diagram_tensor, mask, num_pairs = _VRPersistenceFunction.apply(  # pyright: ignore[reportGeneralTypeIssues]
        points, max_dim, max_radius, metric
    )

    if single_input:
        diagram_tensor = diagram_tensor.squeeze(0)
        mask = mask.squeeze(0)
        num_pairs = num_pairs.squeeze(0)

    diagram = PersistenceDiagram(diagram_tensor, mask, num_pairs)
    if return_simplices:
        simplex_width = max_dim + 1
        simplices = torch.empty((0, simplex_width), dtype=torch.long, device=points.device)
        return diagram, simplices
    return diagram
