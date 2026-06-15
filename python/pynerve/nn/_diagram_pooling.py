"""Pooling layers for variable-size persistence diagrams."""

from __future__ import annotations

from numbers import Integral

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._validation import validate_finite_tensor as _validate_finite_tensor
from .._validation import validate_positive_int as _validate_positive_int
from ..torch._vectorization_basis import _validate_diagram as _validate_diagram_basis


def _validate_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_diagram(diagram: torch.Tensor) -> None:
    if diagram.dim() == 2:
        diagram = diagram.unsqueeze(0)
    for b in range(diagram.shape[0]):
        _validate_diagram_basis(diagram[b])
    if diagram.dim() != 3 or diagram.shape[-1] < 2:
        raise ValueError("diagram must have shape (batch, n_pairs, at least 2)")
    if diagram.numel() == 0:
        return
    birth_death = diagram[:, :, :2]
    if not (birth_death[:, :, 1] >= birth_death[:, :, 0]).all().item():
        raise ValueError("diagram deaths must be greater than or equal to births")


class DiagramPooling(nn.Module):
    """Learnable pooling for variable-size diagrams."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_prototypes: int = 8,
        pooling_type: str = "attention",
    ):
        super().__init__()
        in_channels = _validate_non_negative_int("in_channels", in_channels)
        out_channels = _validate_positive_int(out_channels, "out_channels")
        num_prototypes = _validate_positive_int(num_prototypes, "num_prototypes")
        if pooling_type not in {"attention", "persistence_clustering"}:
            raise ValueError("unknown pooling_type")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_prototypes = num_prototypes
        self.pooling_type = pooling_type
        self.value_transform = nn.Linear(in_channels, out_channels)

        if pooling_type == "attention":
            self.prototypes = nn.Parameter(torch.randn(num_prototypes, in_channels))
            self.attention_logits = nn.Linear(in_channels + 2, num_prototypes)
        elif pooling_type == "persistence_clustering":
            self.cluster_centers = nn.Parameter(torch.linspace(0, 1, num_prototypes))

    def forward(self, diagram: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Pool a diagram batch to ``[batch, num_prototypes, out_channels]``."""
        _validate_diagram(diagram)
        _validate_finite_tensor(features, "features")
        if features.dim() != 3:
            raise ValueError("features must have shape (batch, n_pairs, in_channels)")

        batch_size, n_pairs, _ = diagram.shape
        if features.shape != (batch_size, n_pairs, self.in_channels):
            raise ValueError("features must have shape (batch, n_pairs, in_channels)")
        if features.device != diagram.device:
            raise ValueError("diagram and features must be on the same device")

        if n_pairs == 0:
            return features.new_zeros((batch_size, self.num_prototypes, self.out_channels))

        transformed = self.value_transform(features)

        if self.pooling_type == "attention":
            birth_death = diagram[:, :, :2]
            x = torch.cat([features, birth_death.to(features.dtype)], dim=-1)
            logits = self.attention_logits(x)
            if self.in_channels:
                scale = self.in_channels**0.5
                logits = logits + torch.einsum("bni,pi->bnp", features, self.prototypes) / scale

            weights = F.softmax(logits, dim=1)
            pooled = torch.bmm(weights.transpose(1, 2), transformed)

            return pooled

        if self.pooling_type == "persistence_clustering":
            persistence = (diagram[:, :, 1] - diagram[:, :, 0]).to(features.dtype)
            dist_to_centers = torch.abs(
                persistence.unsqueeze(-1) - self.cluster_centers.to(features.dtype).view(1, 1, -1)
            )
            weights = F.softmax(-dist_to_centers, dim=-1)
            pooled = torch.bmm(weights.transpose(1, 2), transformed)
            normalizer = weights.sum(dim=1).clamp_min(1e-12).unsqueeze(-1)

            return pooled / normalizer

        raise RuntimeError(f"unsupported pooling_type: {self.pooling_type}")
