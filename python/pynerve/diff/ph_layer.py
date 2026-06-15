"""Differentiable persistence layers for PyTorch."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn

from ._ph_representations import compute_persistence_landscape, persistence_image


def _effective_max_radius(points: torch.Tensor, max_radius: float) -> float:
    if math.isfinite(max_radius):
        return max_radius
    if points.shape[0] < 2:
        return 0.0
    radius_points = points.detach().to(device="cpu", dtype=torch.float64)
    distances = torch.cdist(radius_points, radius_points)
    finite = distances[torch.isfinite(distances)]
    return float(finite.max().item()) if finite.numel() else 0.0


class DifferentiablePersistenceFunction(torch.autograd.Function):
    """Custom autograd function for differentiable persistence diagram computation.

    Computes persistence diagrams from point clouds using an external C++ backend
    and backpropagates gradients through the diagram coordinates.
    """

    @staticmethod
    def forward(
        ctx: Any,
        points: torch.Tensor,
        max_dim: int,
        filtration_type: str,
        options: dict[str, Any],
    ) -> tuple[torch.Tensor, ...]:
        """Compute persistence diagrams for a batch of point clouds.

        :param ctx: Autograd context for saving tensors.
        :param points: Tensor of shape ``(batch, n_points, dim)``.
        :param max_dim: Maximum homology dimension to compute.
        :param filtration_type: Type of filtration (e.g. ``"rips"``).
        :param options: Additional keyword arguments for the persistence backend.
        :returns: Tuple of persistence diagram tensors, one per dimension.
        """
        ctx.max_dim = max_dim
        ctx.filtration_type = filtration_type
        ctx.input_shape = points.shape
        ctx.options = options
        ctx.save_for_backward(points)
        diagrams, pair_counts = _compute_persistence_forward(
            points, max_dim, filtration_type, return_counts=True, **options
        )
        ctx.pair_counts = pair_counts
        return tuple(diagrams)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_diagrams: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None, None]:
        """Compute gradients through the persistence diagram computation.

        :param ctx: Autograd context containing saved tensors and metadata.
        :param grad_diagrams: Gradients of the loss with respect to each persistence diagram.
        :returns: Tuple of ``(grad_points, None, None, None)`` matching the forward inputs.
        """
        (points,) = ctx.saved_tensors
        grad_points = _compute_persistence_backward(
            points,
            grad_diagrams,
            ctx.max_dim,
            ctx.filtration_type,
            ctx.options,
            getattr(ctx, "pair_counts", None),
        )
        return grad_points, None, None, None


def _validate_persistence_inputs(
    points: torch.Tensor, max_dim: int, filtration_type: str, kwargs: dict[str, Any]
) -> tuple[float, str, str]:
    if filtration_type != "rips":
        raise ValueError("only rips filtration is supported")
    if points.dim() != 3:
        raise ValueError("points must have shape (batch, n_points, dim)")
    if max_dim < 0:
        raise ValueError("max_dim must be non-negative")
    if points.shape[0] == 0 or points.shape[1] == 0 or points.shape[2] == 0:
        raise ValueError("points must have non-empty batch, point, and feature dimensions")
    if not torch.isfinite(points).all():
        raise ValueError("points must contain only finite values")

    max_radius = kwargs.get("max_radius", float("inf"))
    if max_radius is None:
        max_radius = float("inf")
    max_radius = float(max_radius)
    if max_radius <= 0 or math.isnan(max_radius):
        raise ValueError("max_radius must be positive")
    metric = kwargs.get("metric", "euclidean")
    reduction = kwargs.get("reduction", "clearing")
    return max_radius, metric, reduction


def _setup_persistence_options(
    _core: Any, max_dim: int, max_radius: float, metric: str, reduction: str
) -> Any:
    options = _core.PersistenceOptions()
    options.max_dim = max_dim
    options.max_radius = max_radius if math.isfinite(max_radius) else 0.0
    if hasattr(options, "metric"):
        options.metric = metric
    elif metric != "euclidean":
        raise ValueError("this build only supports euclidean metric")
    if hasattr(options, "reduction"):
        options.reduction = reduction
    elif reduction != "clearing":
        raise ValueError("this build only supports clearing reduction")
    return options


def _process_single_batch(
    points: torch.Tensor, b: int, max_dim: int, _core: Any, options: Any, max_radius: float
) -> tuple[list[torch.Tensor], list[int]]:
    pts_np = points[b].detach().cpu().numpy()
    options.max_radius = _effective_max_radius(points[b], max_radius)
    result = _core.compute_persistence(pts_np, options)

    dim_diagrams: list[list[list[float]]] = [[] for _ in range(max_dim + 1)]
    for pair in result["pairs"]:
        birth, death, dim = pair
        if dim <= max_dim:
            dim_diagrams[dim].append([birth, death])

    batch_diagrams = []
    batch_counts = []
    for dim in range(max_dim + 1):
        batch_counts.append(len(dim_diagrams[dim]))
        if dim_diagrams[dim]:
            diag_tensor = torch.tensor(dim_diagrams[dim], dtype=points.dtype, device=points.device)
        else:
            diag_tensor = torch.empty((0, 2), dtype=points.dtype, device=points.device)
        batch_diagrams.append(diag_tensor)

    return batch_diagrams, batch_counts


def _pad_diagrams_across_batches(
    diagrams: list[list[torch.Tensor]],
    max_dim: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    result_diagrams = []
    for dim in range(max_dim + 1):
        max_pairs = max((d[dim].shape[0] for d in diagrams), default=0)
        if max_pairs > 0:
            padded = []
            for d in diagrams:
                if d[dim].shape[0] < max_pairs:
                    padding = torch.zeros(
                        max_pairs - d[dim].shape[0], 2, dtype=d[dim].dtype, device=d[dim].device
                    )
                    padded.append(torch.cat([d[dim], padding], dim=0))
                else:
                    padded.append(d[dim])
            result_diagrams.append(torch.stack(padded))
        else:
            result_diagrams.append(torch.empty(batch_size, 0, 2, dtype=dtype, device=device))
    return result_diagrams


def _compute_persistence_forward(
    points: torch.Tensor,
    max_dim: int,
    filtration_type: str,
    return_counts: bool = False,
    **kwargs: Any,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[list[int]]]:
    import pynerve_internal as _core  # noqa: PLC0415

    max_radius, metric, reduction = _validate_persistence_inputs(
        points, max_dim, filtration_type, kwargs
    )
    options = _setup_persistence_options(_core, max_dim, max_radius, metric, reduction)

    batch_size = points.shape[0]
    diagrams: list[list[torch.Tensor]] = []
    pair_counts: list[list[int]] = []

    for b in range(batch_size):
        batch_diagrams, batch_counts = _process_single_batch(
            points, b, max_dim, _core, options, max_radius
        )
        diagrams.append(batch_diagrams)
        pair_counts.append(batch_counts)

    result_diagrams = _pad_diagrams_across_batches(
        diagrams, max_dim, batch_size, points.dtype, points.device
    )

    if return_counts:
        return result_diagrams, pair_counts
    return result_diagrams


def _accumulate_merge_gradients(
    pts: torch.Tensor,
    grad_diag: torch.Tensor,
    dists: torch.Tensor,
    dist_grad: torch.Tensor,
    grad_target: torch.Tensor,
    b: int,
) -> None:
    n_points = pts.shape[0]
    edges = [(dists[i, j].item(), i, j) for i in range(n_points) for j in range(i + 1, n_points)]
    edges.sort()

    parent = list(range(n_points))

    def find(x: int, parent: list[int] = parent) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merge_edges = []
    for _, i, j in edges:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            merge_edges.append((i, j))

    for pair_idx in range(grad_diag.shape[0]):
        if pair_idx >= len(merge_edges):
            break
        pair_grad = grad_diag[pair_idx, 0] + grad_diag[pair_idx, 1]
        i, j = merge_edges[pair_idx]
        grad_target[b, i] += pair_grad * dist_grad[i, j]
        grad_target[b, j] -= pair_grad * dist_grad[i, j]


def _compute_persistence_backward(
    points: torch.Tensor,
    grad_diagrams: tuple[torch.Tensor, ...],
    max_dim: int,
    filtration_type: str,
    kwargs: dict[str, Any],
    pair_counts: list[list[int]] | None = None,
) -> torch.Tensor:
    del kwargs
    batch_size = points.shape[0]
    grad_points = torch.zeros_like(points)
    if filtration_type != "rips":
        return grad_points

    for b in range(batch_size):
        pts = points[b]
        dists = torch.cdist(pts, pts)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_grad = diff / (dists.unsqueeze(-1) + 1e-8)

        for dim in range(min(max_dim + 1, len(grad_diagrams))):
            grad_diag = grad_diagrams[dim]
            if grad_diag.dim() == 3:
                grad_diag = grad_diag[b]
            if pair_counts is not None:
                grad_diag = grad_diag[: pair_counts[b][dim]]
            if grad_diag.shape[0] == 0 or dim != 0:
                continue
            _accumulate_merge_gradients(pts, grad_diag, dists, dist_grad, grad_points, b)

    return grad_points


class DifferentiableVietorisRips(nn.Module):
    """Differentiable Vietoris-Rips persistence module.

    Wraps :class:`DifferentiablePersistenceFunction` to compute persistence diagrams
    for point clouds using the Vietoris-Rips filtration with gradient support.
    """

    def __init__(self, max_dim: int = 1, max_radius: float | None = None):
        """Initialize the differentiable Vietoris-Rips persistence module.

        :param max_dim: Maximum homology dimension to compute.
        :param max_radius: Maximum radius for the Rips filtration. If ``None``,
            uses the finite maximum pairwise distance.
        :raises ValueError: If ``max_dim`` is negative.
        """
        super().__init__()
        if max_dim < 0:
            raise ValueError("max_dim must be non-negative")
        self.max_dim = max_dim
        self.max_radius = max_radius

    def forward(self, points: torch.Tensor, **options: Any) -> list[torch.Tensor]:
        """Compute persistence diagrams for a batch of point clouds.

        :param points: Tensor of shape ``(batch, n_points, dim)``.
        :param options: Additional keyword arguments forwarded to the persistence backend.
        :returns: List of persistence diagram tensors, one per homology dimension.
        :raises ValueError: If ``points`` does not have 3 dimensions.
        """
        if points.dim() != 3:
            raise ValueError("points must have shape (batch, n_points, dim)")
        opts: dict[str, Any] = dict(options)
        if self.max_radius is not None:
            opts["max_radius"] = self.max_radius
        return cast(
            list[torch.Tensor],
            DifferentiablePersistenceFunction.apply(points, self.max_dim, "rips", opts),
        )


class DifferentiableAlphaComplex(nn.Module):
    """Differentiable Alpha complex persistence layer."""

    def __init__(self, max_dim: int = 2):
        """Initialize the differentiable Alpha complex persistence module.

        :param max_dim: Maximum homology dimension to compute.
        :raises ValueError: If ``max_dim`` is negative.
        """
        super().__init__()
        if max_dim < 0:
            raise ValueError("max_dim must be non-negative")
        self.max_dim = max_dim

    def forward(self, points: torch.Tensor) -> list[torch.Tensor]:
        """Compute Alpha complex persistence diagrams (not available in this build).

        :param points: Tensor of shape ``(batch, n_points, dim)``.
        :returns: Never returns; always raises ``RuntimeError``.
        :raises RuntimeError: Always, because this functionality is not exposed.
        """
        del points
        raise RuntimeError("differentiable alpha persistence is not exposed by this build")


class DifferentiableCubical(nn.Module):
    """Differentiable Cubical persistence layer."""

    def __init__(self, max_dim: int = 2, sublevel: bool = True):
        """Initialize the differentiable cubical persistence module.

        :param max_dim: Maximum homology dimension to compute.
        :param sublevel: If ``True``, use sublevel set filtration.
        :raises ValueError: If ``max_dim`` is negative.
        """
        super().__init__()
        if max_dim < 0:
            raise ValueError("max_dim must be non-negative")
        self.max_dim = max_dim
        self.sublevel = sublevel

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Compute cubical persistence diagrams (not available in this build).

        :param image: Input tensor.
        :returns: Never returns; always raises ``RuntimeError``.
        :raises RuntimeError: Always, because this functionality is not exposed.
        """
        del image
        raise RuntimeError("differentiable cubical persistence is not exposed by this build")


class FiltrationLearningLayer(nn.Module):
    """Learns a per-point filtration value for persistence computation.

    Uses a multi-layer perceptron with ReLU activations and layer normalization
    to map point coordinates to scalar filtration values.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None):
        """Initialize the filtration learning layer.

        :param input_dim: Dimensionality of each input point.
        :param hidden_dims: Hidden layer dimensions for the MLP. Defaults to ``[64, 64]``.
        :raises ValueError: If ``input_dim`` is not positive or any hidden dimension
            is not positive.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        hidden_dims = [64, 64] if hidden_dims is None else hidden_dims

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            if h <= 0:
                raise ValueError("hidden dimensions must be positive")
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.LayerNorm(h)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())

        self.filtration_net = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Compute learned filtration values for a batch of point clouds.

        :param points: Tensor of shape ``(batch, n_points, dim)``.
        :returns: Tensor of shape ``(batch, n_points)`` with scalar filtration values.
        :raises ValueError: If ``points`` does not have 3 dimensions.
        """
        if points.dim() != 3:
            raise ValueError("points must have shape (batch, n_points, dim)")
        batch_size, n_points, dim = points.shape
        x = points.view(-1, dim)
        values = cast(torch.Tensor, self.filtration_net(x)).squeeze(-1)
        return cast(torch.Tensor, values.view(batch_size, n_points))


class LearnableFiltrationPersistence(nn.Module):
    """End-to-end learnable persistence module combining filtration learning with Vietoris-Rips."""

    def __init__(self, input_dim: int, max_dim: int = 1, hidden_dims: list[int] | None = None):
        """Initialize the learnable filtration persistence module.

        :param input_dim: Dimensionality of each input point.
        :param max_dim: Maximum homology dimension for persistence computation.
        :param hidden_dims: Hidden layer dimensions for the filtration MLP. Defaults to ``None``
            (uses ``[64, 64]`` internally).
        """
        super().__init__()
        self.filtration = FiltrationLearningLayer(input_dim, hidden_dims)
        self.persistence = DifferentiableVietorisRips(max_dim=max_dim)

    def forward(self, points: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Compute persistence diagrams with learned filtration values.

        :param points: Tensor of shape ``(batch, n_points, dim)``.
        :returns: Tuple of ``(diagrams, filt_values)`` where ``diagrams`` is a list of
            persistence diagram tensors and ``filt_values`` has shape ``(batch, n_points)``.
        """
        filt_values = self.filtration(points)
        diagrams = self.persistence(points, weights=filt_values.softmax(dim=1))
        return diagrams, filt_values


__all__ = [
    "DifferentiableVietorisRips",
    "FiltrationLearningLayer",
    "LearnableFiltrationPersistence",
    "compute_persistence_landscape",
    "persistence_image",
]
