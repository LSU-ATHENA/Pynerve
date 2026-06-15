"""Autograd Function wrapper for persistent homology computation."""

from __future__ import annotations

from typing import Any, cast

from torch.autograd import Function

import torch
from torch import Tensor

from .._constants import EPS

_MEMORY_MODES: set[str] = {"standard", "memory_mapped", "streaming", "extreme"}
_REDUCTIONS: set[str] = {"standard", "clearing", "cohomology"}


class PersistentHomologyFunction(Function):
    """Differentiable persistent homology via a custom autograd Function.

    Wraps the compiled persistence engine so that forward calls produce
    persistence diagrams and backward passes compute a coarse subgradient
    based on pairwise distances between points.
    """

    @staticmethod
    def forward(
        ctx: Any,
        points: Tensor,
        max_dim: int,
        max_radius: float,
        metric: str,
        reduction: str,
        use_gpu: bool,
        memory_mode: str,
    ) -> tuple[Tensor, ...]:
        """Compute persistence diagrams for a batched point cloud.

        :param ctx: Autograd context for saving tensors.
        :param points: Point cloud tensor of shape ``(batch, N, D)``.
        :param max_dim: Maximum homology dimension to compute.
        :param max_radius: Filtration radius cutoff (positive or inf).
        :param metric: Distance metric (currently only ``"euclidean"``).
        :param reduction: Reduction strategy (``"standard"``,
            ``"clearing"``, or ``"cohomology"``).
        :param use_gpu: Whether to use GPU-accelerated computation.
        :param memory_mode: Memory management mode
            (``"standard"``, ``"memory_mapped"``, ``"streaming"``,
            or ``"extreme"``).
        :returns: Tuple of tensors, one per homology dimension, each
            shaped ``(batch, max_pairs, 2)`` with columns
            ``(birth, death)``.
        :raises ValueError: If *metric* is not ``"euclidean"`` or
            *reduction* is invalid.
        """
        from ._ph_module import _core_compute_result, _result_to_diagram_tensors  # noqa: PLC0415

        ctx.save_for_backward(points)
        ctx.max_dim = max_dim
        ctx.max_radius = max_radius
        ctx.metric = metric
        ctx.reduction = reduction
        ctx.use_gpu = use_gpu
        ctx.memory_mode = memory_mode

        if points.is_cuda:
            points_cpu = points.contiguous().to("cpu", non_blocking=True)
        else:
            points_cpu = points.contiguous()
        if metric != "euclidean":
            raise ValueError("this build only supports euclidean metric")
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {sorted(_REDUCTIONS)}")

        batch_size = points.shape[0]
        diagrams = []

        for b in range(batch_size):
            result = _core_compute_result(
                points_cpu[b],
                max_dim=max_dim,
                max_radius=max_radius,
                reduction=reduction,
            )
            diagrams.append(
                _result_to_diagram_tensors(result, max_dim, points.dtype, points.device)
            )

        batch_diagrams = []
        for dim in range(max_dim + 1):
            dim_diags = _pad_diagram_batch([d[dim] for d in diagrams])
            batch_diagrams.append(dim_diags)

        return cast(tuple[Tensor, ...], tuple(batch_diagrams))

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Tensor
    ) -> tuple[Tensor | None, None, None, None, None, None, None]:
        """Compute a coarse subgradient for persistence diagrams."""
        (points,) = ctx.saved_tensors

        grad_points = torch.zeros_like(points)

        batch_size = points.shape[0]
        n_points = points.shape[1]

        for b in range(batch_size):
            pts = points[b]
            dists = torch.cdist(pts, pts)

            for grad_diag in grad_outputs:
                if grad_diag is None or grad_diag.numel() == 0:
                    continue

                g = grad_diag[b] if grad_diag.dim() == 3 else grad_diag

                if g.shape[0] == 0:
                    continue

                grad_birth = g[:, 0].mean()
                grad_death = g[:, 1].mean()

                for i in range(n_points):
                    for j in range(i + 1, n_points):
                        if dists[i, j] > EPS:
                            diff = (pts[i] - pts[j]) / dists[i, j]
                            grad_contrib = (grad_birth + grad_death) * diff
                            grad_points[b, i] += grad_contrib
                            grad_points[b, j] -= grad_contrib

        return grad_points, None, None, None, None, None, None


def _pad_diagram_batch(diagrams: list[Tensor]) -> Tensor:
    max_pairs = max((diagram.shape[0] for diagram in diagrams), default=0)
    if max_pairs == 0:
        first = diagrams[0]
        return first.new_empty((len(diagrams), 0, 2))

    padded = []
    for diagram in diagrams:
        if diagram.shape[0] == max_pairs:
            padded.append(diagram)
            continue
        pad = diagram.new_zeros((max_pairs - diagram.shape[0], diagram.shape[1]))
        padded.append(torch.cat((diagram, pad), dim=0))
    return torch.stack(padded)
