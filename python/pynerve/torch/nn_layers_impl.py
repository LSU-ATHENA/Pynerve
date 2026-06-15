"""PyTorch layers and modules for differentiable topological learning."""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import torch
from torch import nn

from .._validation import validate_finite_tensor as _validate_finite_tensor
from .._validation import validate_positive_int as _validate_positive_int
from . import preprocessing as prep
from . import vectorization as vec
from ._diagram import PersistenceDiagram
from ._persistence_api import (
    _validate_max_dim,
    _validate_max_radius,
    _validate_metric,
    _VRPersistenceFunction,
)


def _validate_probability(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1")
    return parsed


def _validate_positive_dims(name: str, dims: tuple[int, ...]) -> tuple[int, ...]:
    if not dims:
        raise ValueError(f"{name} must contain positive dimensions")
    parsed = []
    for dim in dims:
        parsed.append(_validate_positive_int(dim, name))
    return tuple(parsed)


def _validate_floating_tensor(name: str, value: torch.Tensor) -> None:
    _validate_finite_tensor(value, name)
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating-point dtype")


class PersistenceLayer(nn.Module):
    """Differentiable layer that computes persistence diagrams."""

    def __init__(
        self,
        max_dim: int = 1,
        max_radius: float = float("inf"),
        metric: str = "euclidean",
        preprocessing: dict[str, Any] | None = None,
        return_raw: bool = False,
    ):
        super().__init__()
        self.max_dim = _validate_max_dim(max_dim)
        self.max_radius = _validate_max_radius(max_radius)
        self.metric = _validate_metric(metric)
        self.preprocessing = preprocessing or {}
        self.return_raw = return_raw

    def forward(self, x: torch.Tensor) -> PersistenceDiagram | torch.Tensor:
        """Compute persistence diagrams from point clouds."""
        _validate_floating_tensor("x", x)
        if x.dim() not in {2, 3}:
            raise ValueError("x must be a 2D or 3D point cloud tensor")

        single_input = x.dim() == 2
        if single_input:
            x = x.unsqueeze(0)
        if x.shape[-2] == 0 or x.shape[-1] == 0:
            raise ValueError("x must contain at least one point and coordinate")

        diagrams, mask, num_pairs = _VRPersistenceFunction.apply(
            x, self.max_dim, self.max_radius, self.metric
        )

        if self.preprocessing:
            diagrams = diagrams.detach()
            diagrams = prep.clean_diagram(diagrams, **self.preprocessing)

        if self.return_raw:
            if single_input:
                diagrams = diagrams.squeeze(0)
            return cast(torch.Tensor, diagrams)

        diagrams.shape[0]

        if single_input:
            return PersistenceDiagram(diagrams.squeeze(0), mask.squeeze(0), num_pairs.squeeze(0))
        return PersistenceDiagram(diagrams, mask, num_pairs)


class VectorizationLayer(nn.Module):
    """Convert persistence diagrams to fixed-size vectors."""

    def __init__(
        self,
        method: Literal["image", "landscape", "silhouette", "heat", "histogram"] = "landscape",
        **method_params: Any,
    ) -> None:
        super().__init__()
        self.method = method
        self.method_params = method_params

        self.vec_fn = self._get_vectorization_fn()

    def _get_vectorization_fn(self) -> Any:
        method_map = {
            "image": vec.persistence_image,
            "landscape": vec.persistence_landscape,
            "silhouette": vec.persistence_silhouette,
            "heat": vec.heat_kernel_signature,
            "histogram": vec.birth_death_curve,
        }

        if self.method not in method_map:
            raise ValueError(f"Unknown method: {self.method}")

        return method_map[self.method]

    def forward(self, x: PersistenceDiagram) -> torch.Tensor:
        mask = None
        if isinstance(x, PersistenceDiagram):
            diagram = x.diagrams
            mask = x.mask
        else:
            diagram = x
        if diagram.dim() not in {2, 3} or diagram.shape[-1] < 2:
            raise ValueError("diagram must have shape (..., n_pairs, at least 2)")

        is_batched = diagram.dim() == 3

        if is_batched:
            batch_size = diagram.shape[0]
            vectors = []

            for i in range(batch_size):
                item = diagram[i][mask[i]] if mask is not None else diagram[i]
                vec_tensor = self.vec_fn(item, **self.method_params)
                vectors.append(vec_tensor.flatten())

            return torch.stack(vectors)
        else:
            diagram = diagram[mask] if mask is not None else diagram
            vec_tensor = self.vec_fn(diagram, **self.method_params)
            return cast(torch.Tensor, vec_tensor.flatten())


class TopologicalFeatureExtractor(nn.Module):
    """Pipeline layer: persistence computation + vectorization."""

    def __init__(
        self,
        max_dim: int = 1,
        max_radius: float = float("inf"),
        metric: str = "euclidean",
        preprocessing: dict[str, Any] | None = None,
        vectorization: str = "landscape",
        vectorization_params: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.persistence = PersistenceLayer(
            max_dim=max_dim,
            max_radius=max_radius,
            metric=metric,
            preprocessing=preprocessing,
        )

        self.vectorization = VectorizationLayer(
            method=cast(
                Literal["image", "landscape", "silhouette", "heat", "histogram"], vectorization
            ),
            **(vectorization_params or {}),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract topological features."""
        diagram = self.persistence(x)
        return cast(torch.Tensor, self.vectorization(diagram))


class DiagramPooling(nn.Module):
    """Pooling operations over collections of persistence diagrams."""

    def __init__(self, method: Literal["mean", "max", "sum", "attention"] = "mean", dim: int = 1):
        super().__init__()
        if method not in {"mean", "max", "sum", "attention"}:
            raise ValueError("unknown pooling method")
        self.method = method
        self.dim = dim

    def forward(self, diagrams: torch.Tensor) -> torch.Tensor:
        """Pool over multiple diagrams."""
        _validate_floating_tensor("diagrams", diagrams)
        if diagrams.numel() == 0:
            raise ValueError("diagrams must be non-empty")
        if self.method == "mean":
            return diagrams.mean(dim=self.dim)
        if self.method == "max":
            return diagrams.max(dim=self.dim)[0]
        if self.method == "sum":
            return diagrams.sum(dim=self.dim)
        if self.method == "attention":
            weights = torch.softmax(diagrams.sum(dim=-1), dim=self.dim)
            return (diagrams * weights.unsqueeze(-1)).sum(dim=self.dim)
        raise RuntimeError(f"unsupported pooling method: {self.method}")


class TopologicalAttention(nn.Module):
    """Attention block with optional topology-derived gating."""

    def __init__(self, feature_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        feature_dim = _validate_positive_int(feature_dim, "feature_dim")
        n_heads = _validate_positive_int(n_heads, "n_heads")
        dropout = _validate_probability("dropout", dropout)
        if feature_dim % n_heads != 0:
            raise ValueError("feature_dim must be divisible by a positive n_heads")

        self.feature_dim = feature_dim
        self.n_heads = n_heads

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        self.topo_gate = nn.Sequential(
            nn.Linear(1, max(1, feature_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(1, feature_dim // 4), feature_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = (feature_dim // n_heads) ** -0.5

    def forward(
        self, features: torch.Tensor, diagrams: PersistenceDiagram | None = None
    ) -> tuple[Any, ...]:
        """Apply topological attention."""
        _validate_floating_tensor("features", features)
        if features.dim() != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("features must have shape (batch, seq_len, feature_dim)")
        batch_size, seq_len, _ = features.shape
        del seq_len

        Q = self.query(features)  # noqa: N806
        K = self.key(features)  # noqa: N806
        V = self.value(features)  # noqa: N806

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if diagrams is not None:
            from . import statistics  # noqa: PLC0415

            if isinstance(diagrams, PersistenceDiagram):
                topo_strength: torch.Tensor = statistics.total_persistence(diagrams.diagrams)
            else:
                d = diagrams
                if d.dim() == 3:
                    topo_strength = torch.stack([statistics.total_persistence(item) for item in d])
                else:
                    topo_strength = cast(torch.Tensor, statistics.total_persistence(d)).view(1)
            if topo_strength.shape[0] != batch_size:
                raise ValueError("diagram batch size must match features")
            _validate_floating_tensor("topo_strength", topo_strength)
            gate = self.topo_gate(topo_strength.view(-1, 1))
            gate = gate.view(batch_size, 1, self.feature_dim)
            V = V * torch.sigmoid(gate)  # noqa: N806

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        return output, attn_weights


class PersistenceReadout(nn.Module):
    """MLP readout for vectorized persistence features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: tuple[int, ...] = (128,),
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        in_features = _validate_positive_int(in_features, "in_features")
        out_features = _validate_positive_int(out_features, "out_features")
        hidden_dims = _validate_positive_dims("hidden_dims", hidden_dims)
        dropout = _validate_probability("dropout", dropout)
        if activation not in {"relu", "gelu"}:
            raise ValueError("unknown activation")

        layers: list[nn.Module] = []
        prev_dim = in_features

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, out_features))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply readout MLP."""
        _validate_floating_tensor("x", x)
        return cast(torch.Tensor, self.mlp(x))


def make_topo_network(
    input_dim: int,
    hidden_dims: tuple[int, ...] = (128, 64),
    max_dim: int = 1,
    vectorization: str = "landscape",
    vectorization_params: dict[str, Any] | None = None,
    num_classes: int = 10,
    dropout: float = 0.2,
) -> nn.Module:
    """Create a compact end-to-end topological neural network."""
    input_dim = _validate_positive_int(input_dim, "input_dim")
    num_classes = _validate_positive_int(num_classes, "num_classes")
    hidden_dims = _validate_positive_dims("hidden_dims", hidden_dims)
    dropout = _validate_probability("dropout", dropout)
    max_dim = _validate_max_dim(max_dim)

    class TopoNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            layers: list[nn.Module] = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            self.encoder = nn.Sequential(*layers)

            self.tda = TopologicalFeatureExtractor(
                max_dim=max_dim,
                vectorization=vectorization,
                vectorization_params=vectorization_params,
            )

            self.classifier = nn.LazyLinear(num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _validate_floating_tensor("x", x)
            if x.dim() != 3 or x.shape[-1] != input_dim:
                raise ValueError("x must have shape (batch, n_points, input_dim)")
            features = self.encoder(x)
            topo_features = self.tda(features)
            return cast(torch.Tensor, self.classifier(topo_features))

    return TopoNetwork()


__all__ = [
    "PersistenceLayer",
    "VectorizationLayer",
    "TopologicalFeatureExtractor",
    "DiagramPooling",
    "TopologicalAttention",
    "PersistenceReadout",
    "make_topo_network",
]
