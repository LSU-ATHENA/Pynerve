"""Sparse and windowed persistent homology layers."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Literal, cast

import numpy as np
from torch.autograd import Function

import torch
from torch import Tensor, nn

from .._constants import EPS_1e_6
from .._validation import validate_finite_tensor as _validate_finite_tensor
from .._validation import validate_positive_int as _validate_positive_int
from .persistent_homology import PersistentHomology


def _validate_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_max_radius(max_radius: float) -> float:
    radius = float(max_radius)
    if math.isnan(radius) or radius <= 0:
        raise ValueError("max_radius must be positive")
    return radius


def _validate_probability(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return parsed


def _validate_landmark_ratio(landmark_ratio: float) -> float:
    ratio = float(landmark_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("landmark_ratio must be in (0, 1]")
    return ratio


def _validate_finite_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


class SparsePHFunction(Function):
    """Autograd function for landmark-based persistence homology.

    Computes approximate persistence diagrams using farthest point sampling
    and witness complexes. The operation is non-differentiable.
    """

    @staticmethod
    def forward(
        ctx: Any,
        points: Tensor,
        max_dim: int,
        max_radius: float,
        landmark_ratio: float,
        metric: str,
    ) -> tuple[Tensor, ...]:
        """Compute sparse persistence diagrams.

        :param ctx: Context object for saving tensors for backward.
        :param points: Points tensor of shape (batch_size, n_points, dim).
        :param max_dim: Maximum homology dimension to compute.
        :param max_radius: Maximum radius for persistence computation.
        :param landmark_ratio: Ratio of points to use as landmarks, in (0, 1].
        :param metric: Distance metric (e.g. 'euclidean', 'manhattan').
        :returns: Tuple of persistence diagram tensors, one per batch element.
        """
        batch_size, n_points, _ = points.shape
        n_landmarks = max(1, int(n_points * landmark_ratio))
        host_points = points.detach().cpu()

        diagrams = []
        for b in range(batch_size):
            landmarks, _ = farthest_point_sampling(host_points[b], n_landmarks)
            d = compute_witness_persistence(
                landmarks.numpy(),
                host_points[b].numpy(),
                max_dim=max_dim,
                max_radius=max_radius,
                metric=metric,
            )
            diagrams.append(torch.as_tensor(d, dtype=points.dtype, device=points.device))

        outputs = tuple(diagrams)
        ctx.mark_non_differentiable(*outputs)
        return outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[None, None, None, None, None]:
        """Backward pass (non-differentiable).

        :param ctx: Context object from forward.
        :param grad_outputs: Gradients of loss with respect to outputs.
        :returns: Tuple of None for each input (non-differentiable).
        """
        del ctx, grad_outputs
        return None, None, None, None, None


class SparsePH(nn.Module):
    """Landmark-based approximate persistence layer."""

    def __init__(
        self,
        max_dim: int = 1,
        max_radius: float = float("inf"),
        landmark_ratio: float = 0.1,
        metric: str = "euclidean",
        reduction: str = "mean",
    ):
        """Initialize sparse persistence layer.

        :param max_dim: Maximum homology dimension to compute.
        :param max_radius: Maximum radius for persistence computation.
        :param landmark_ratio: Ratio of points to use as landmarks, in (0, 1].
        :param metric: Distance metric (e.g. 'euclidean', 'manhattan').
        :param reduction: Reduction mode, ``'none'`` or ``'mean'``.
        :raises ValueError: If any parameter is invalid.
        """
        super().__init__()
        max_dim = _validate_non_negative_int("max_dim", max_dim)
        max_radius = _validate_max_radius(max_radius)
        landmark_ratio = _validate_landmark_ratio(landmark_ratio)
        if reduction not in {"none", "mean"}:
            raise ValueError("reduction must be 'none' or 'mean'")
        self.max_dim = max_dim
        self.max_radius = max_radius
        self.landmark_ratio = landmark_ratio
        self.metric = metric
        self.reduction = reduction

    def forward(self, points: Tensor) -> Tensor | list[Tensor]:
        """Compute persistence features from points.

        :param points: Points tensor of shape (batch_size, n_points, dim).
        :returns: If ``reduction='none'``, a list of per-sample diagram tensors.
            If ``reduction='mean'``, a stacked tensor of mean persistence features.
        :raises ValueError: If points is not a valid 3D tensor.
        """
        _validate_finite_tensor(points, "points")
        if points.dim() != 3:
            raise ValueError(f"Expected 3D tensor (batch, n, dim), got {points.shape}")
        if points.shape[1] == 0:
            raise ValueError("points must contain at least one point")

        diagrams = SparsePHFunction.apply(
            points, self.max_dim, self.max_radius, self.landmark_ratio, self.metric
        )
        if self.reduction == "none":
            return list(diagrams)  # pyright: ignore[reportArgumentType]
        if not diagrams:
            return points.new_zeros((points.shape[0], 0))
        feature_dim = diagrams[0].shape[-1] if diagrams[0].ndim > 1 else 0
        reduced = []
        for diagram in diagrams:
            if diagram.numel() == 0:
                reduced.append(points.new_zeros((feature_dim,)))
                continue
            finite_rows = (
                torch.isfinite(diagram).all(dim=1) if diagram.dim() > 1 else torch.isfinite(diagram)
            )
            finite_diagram = diagram[finite_rows]
            reduced.append(
                finite_diagram.mean(dim=0)
                if finite_diagram.numel() > 0
                else points.new_zeros((feature_dim,))
            )
        return torch.stack(reduced)


class WindowedPH(nn.Module):
    """Sliding-window persistence feature layer."""

    def __init__(
        self,
        window_size: int = 512,
        stride: int = 256,
        max_dim: int = 1,
        max_radius: float = float("inf"),
        overlap_handling: Literal["mean", "max", "concat"] = "concat",
        metric: str = "euclidean",
    ):
        """Initialize windowed persistence layer.

        :param window_size: Number of points per sliding window.
        :param stride: Stride between consecutive windows.
        :param max_dim: Maximum homology dimension to compute.
        :param max_radius: Maximum radius for persistence computation.
        :param overlap_handling: Strategy for merging overlapping windows
            (``'mean'``, ``'max'``, or ``'concat'``).
        :param metric: Distance metric (e.g. 'euclidean', 'manhattan').
        :raises ValueError: If any parameter is invalid.
        """
        super().__init__()
        window_size = _validate_positive_int(window_size, "window_size")
        stride = _validate_positive_int(stride, "stride")
        max_dim = _validate_non_negative_int("max_dim", max_dim)
        max_radius = _validate_max_radius(max_radius)
        if overlap_handling not in {"mean", "max", "concat"}:
            raise ValueError("overlap_handling must be 'mean', 'max', or 'concat'")
        self.window_size = window_size
        self.stride = stride
        self.max_dim = max_dim
        self.max_radius = max_radius
        self.overlap_handling = overlap_handling
        self.metric = metric

        self.local_ph = PersistentHomology(max_dim=max_dim, max_radius=max_radius, metric=metric)

    def forward(self, points: Tensor) -> Tensor:
        """Compute windowed persistence features.

        :param points: Points tensor of shape (batch_size, seq_len, dim).
        :returns: Persistence features tensor with shape dependent on
            ``overlap_handling`` mode.
        :raises ValueError: If points is not a valid 3D tensor.
        """
        _validate_finite_tensor(points, "points")
        if points.dim() != 3:
            raise ValueError(f"Expected 3D tensor (batch, seq_len, dim), got {points.shape}")
        batch_size, seq_len, dim = points.shape

        windows = []
        for start in range(0, seq_len - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = points[:, start:end, :]
            windows.append(window)

        if not windows:
            return torch.zeros(batch_size, 0, dtype=points.dtype, device=points.device)

        windows_tensor = torch.stack(windows, dim=1)
        num_windows = windows_tensor.shape[1]
        windows_flat = windows_tensor.reshape(-1, self.window_size, dim)
        diagrams = self.local_ph(windows_flat)
        features = self._diagrams_to_features(diagrams)
        feature_dim = features.shape[-1]
        features = features.reshape(batch_size, num_windows, feature_dim)

        if self.overlap_handling == "mean":
            return features.mean(dim=1)
        if self.overlap_handling == "max":
            return features.max(dim=1)[0]
        return features.reshape(batch_size, -1)

    def _diagrams_to_features(self, diagrams: list[Tensor]) -> Tensor:
        features: list[Tensor] = []
        for diagram in diagrams:
            if diagram.numel() == 0:
                return (
                    torch.zeros(5, dtype=torch.float32) if not features else torch.stack(features)
                )
            if diagram.dim() == 3:
                for b in range(diagram.shape[0]):
                    feat = self._compute_diagram_features(diagram[b])
                    features.append(feat)
            elif diagram.dim() == 2:
                features.append(self._compute_diagram_features(diagram))
            else:
                raise ValueError(f"diagram must be 2D or 3D, got {diagram.dim()}D")
        return torch.stack(features)

    def _compute_diagram_features(self, diagram: Tensor) -> Tensor:
        dtype = diagram.dtype if diagram.is_floating_point() else torch.float32
        if diagram.numel() == 0 or diagram.shape[-1] < 2:
            return torch.zeros(5, dtype=dtype, device=diagram.device)
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        if not torch.isfinite(births).all().item():
            raise ValueError("diagram births must be finite")
        finite_deaths = torch.isfinite(deaths)
        if (
            finite_deaths.any().item()
            and not (deaths[finite_deaths] >= births[finite_deaths]).all().item()
        ):
            raise ValueError("diagram finite deaths must be greater than or equal to births")
        persistence = deaths[finite_deaths] - births[finite_deaths]
        if persistence.numel() == 0:
            return torch.zeros(5, dtype=dtype, device=diagram.device)
        std = (
            persistence.std(unbiased=False) if len(persistence) > 1 else persistence.new_tensor(0.0)
        )
        return torch.stack(
            (
                persistence.new_tensor(float(len(diagram))),
                persistence.mean(),
                persistence.max(),
                std,
                (persistence > 0.1).sum().to(dtype=persistence.dtype),
            )
        )


class TopologyAttention(nn.Module):
    """Attention layer with learned topological cluster bias."""

    def __init__(
        self,
        n_heads: int = 8,
        dim: int = 768,
        n_clusters: int = 16,
        dropout: float = 0.0,
    ):
        """Initialize topology-aware attention layer.

        :param n_heads: Number of attention heads.
        :param dim: Total embedding dimension (must be divisible by n_heads).
        :param n_clusters: Number of topological cluster centers.
        :param dropout: Dropout probability in [0, 1).
        :raises ValueError: If any parameter is invalid.
        """
        super().__init__()
        n_heads = _validate_positive_int(n_heads, "n_heads")
        dim = _validate_positive_int(dim, "dim")
        n_clusters = _validate_positive_int(n_clusters, "n_clusters")
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        dropout = _validate_probability("dropout", dropout)
        self.n_heads = n_heads
        self.dim = dim
        self.n_clusters = n_clusters
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, dim))
        self.topo_temperature = nn.Parameter(torch.ones(1))

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Compute topology-biased multi-head attention.

        :param x: Input tensor of shape (batch_size, seq_len, dim).
        :param mask: Optional attention mask of shape (batch_size, seq_len,
            seq_len). True entries are attended to.
        :returns: Output tensor of shape (batch_size, seq_len, dim).
        :raises ValueError: If input validation fails.
        """
        _validate_finite_tensor(x, "x")
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (batch, seq_len, dim), got {x.shape}")
        batch, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected embedding dimension {self.dim}, got {dim}")
        _validate_finite_tensor(self.cluster_centers, "cluster_centers")
        _validate_finite_tensor(self.topo_temperature, "topo_temperature")

        topo_sim = torch.matmul(x, self.cluster_centers.T)
        topo_weights = torch.softmax(topo_sim / self.topo_temperature.clamp_min(EPS_1e_6), dim=-1)

        q = self.q_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        topo_bias = torch.matmul(topo_weights, topo_weights.transpose(-2, -1))
        topo_bias = topo_bias.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        attn_scores = attn_scores + topo_bias * 0.1

        if mask is not None:
            _validate_finite_tensor(mask, "mask")
            attention_mask = mask.to(dtype=torch.bool)
            if attention_mask.shape[-1] != seq_len:
                raise ValueError("mask has incompatible shape")
            if not attention_mask.any(dim=-1).all().item():
                raise ValueError("mask must leave at least one attention target per row")
            attn_scores = attn_scores.masked_fill(~attention_mask, float("-inf"))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, dim)
        out = cast(Tensor, self.out_proj(out))

        return out


def farthest_point_sampling(points: Tensor, n_samples: int) -> tuple[Tensor, Tensor]:
    """Select landmarks by deterministic farthest-point sampling."""
    n_samples = _validate_non_negative_int("n_samples", n_samples)
    _validate_finite_tensor(points, "points")
    if points.dim() != 2:
        raise ValueError("points must be a 2D tensor")
    n_points = points.shape[0]
    n_samples = min(n_samples, n_points)
    if n_samples <= 0:
        empty_indices = torch.empty(0, dtype=torch.long, device=points.device)
        return points[:0], empty_indices

    indices = torch.zeros(n_samples, dtype=torch.long, device=points.device)
    distances = torch.full((n_points,), float("inf"), device=points.device)

    indices[0] = 0

    for i in range(1, n_samples):
        last_selected = points[indices[i - 1]]
        dists = torch.norm(points - last_selected.unsqueeze(0), dim=1)
        distances = torch.minimum(distances, dists)
        indices[i] = torch.argmax(distances)

    landmarks = points[indices]
    return landmarks, indices


def compute_witness_persistence(
    landmarks: np.ndarray,
    witnesses: np.ndarray,
    max_dim: int = 1,
    max_radius: float = float("inf"),
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute persistence on a witness complex."""
    landmarks = _validate_finite_array("landmarks", landmarks)
    witnesses = _validate_finite_array("witnesses", witnesses)
    max_dim = _validate_non_negative_int("max_dim", max_dim)
    max_radius = _validate_max_radius(max_radius)
    if landmarks.shape[1] != witnesses.shape[1]:
        raise ValueError("landmarks and witnesses must have the same dimension")
    try:
        import pynerve_internal as _core  # noqa: PLC0415
    except ImportError:
        raise RuntimeError(
            "compute_witness_persistence is only available with C++ extensions."
        ) from None
    compute = getattr(_core, "compute_witness_persistence", None)
    if compute is None:
        import torch  # noqa: PLC0415

        from ..torch._persistence_api import witness_persistence as _witness  # noqa: PLC0415

        landmarks_t = torch.from_numpy(landmarks)
        witnesses_t = torch.from_numpy(witnesses)
        result = _witness(landmarks_t, witnesses_t, max_dim, max_radius)
        return result.diagrams[0].cpu().numpy()
    return cast(np.ndarray, compute(landmarks, witnesses, max_dim, max_radius, metric))


__all__ = [
    "SparsePH",
    "WindowedPH",
    "TopologyAttention",
    "farthest_point_sampling",
    "compute_witness_persistence",
]
