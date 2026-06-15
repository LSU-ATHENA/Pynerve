"""Primitive neural layers for persistence diagrams."""

from __future__ import annotations

from numbers import Integral
from typing import cast

import torch.nn.functional as F  # noqa: N812

import torch
from torch import Tensor, nn

from .._validation import validate_finite_tensor as _validate_finite_tensor
from .._validation import validate_positive_int as _validate_positive_int
from ..torch._vectorization_basis import _validate_diagram as _validate_diagram_basis

__all__ = [
    "DiagramConv1D",
    "DiagramDeepSet",
    "_validate_diagram",
    "_validate_non_negative_int",
    "_validate_positive_int",
]


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


class DiagramConv1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        persistence_stride: int = 1,
        use_persistence_weighting: bool = True,
    ):
        super().__init__()
        in_channels = _validate_non_negative_int("in_channels", in_channels)
        out_channels = _validate_positive_int(out_channels, "out_channels")
        kernel_size = _validate_positive_int(kernel_size, "kernel_size")
        persistence_stride = _validate_positive_int(persistence_stride, "persistence_stride")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_weighting = use_persistence_weighting

        self.conv = nn.Conv1d(
            2 + in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=persistence_stride,
            padding=kernel_size // 2,
            bias=True,
        )

        if use_persistence_weighting:
            gate_hidden = max(1, out_channels // 4)
            self.persistence_gate = nn.Sequential(
                nn.Linear(1, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, out_channels),
                nn.Sigmoid(),
            )

    def forward(self, diagram: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        _validate_diagram(diagram)
        batch_size, n_pairs, _ = diagram.shape
        if n_pairs == 0:
            return diagram.new_zeros(batch_size, self.out_channels, 0)

        birth_death = diagram[:, :, :2]

        if features is not None:
            _validate_finite_tensor(features, "features")
            if features.shape != (batch_size, n_pairs, self.in_channels):
                raise ValueError("features must have shape (batch, n_pairs, in_channels)")
            x = torch.cat([birth_death, features], dim=-1)
        else:
            if self.in_channels != 0:
                raise ValueError("features are required when in_channels is positive")
            x = birth_death

        out = cast(Tensor, self.conv(x.permute(0, 2, 1)))

        if self.use_weighting:
            persistence = (diagram[:, :, 1] - diagram[:, :, 0]).unsqueeze(-1)
            gate = cast(Tensor, self.persistence_gate(persistence)).permute(0, 2, 1)
            if gate.shape[-1] != out.shape[-1]:
                gate = F.interpolate(gate, size=out.shape[-1], mode="nearest")
            out = out * gate

        return out


class DiagramDeepSet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        out_channels: int,
        pooling: str = "persistence_weighted",
    ):
        super().__init__()
        in_channels = _validate_non_negative_int("in_channels", in_channels)
        out_channels = _validate_positive_int(out_channels, "out_channels")
        if not hidden_channels:
            raise ValueError("hidden dimensions must be non-empty")
        hidden_channels = [_validate_positive_int(h, "hidden_channels") for h in hidden_channels]
        if pooling not in {"sum", "mean", "max", "persistence_weighted"}:
            raise ValueError("unknown pooling mode")

        self.pooling = pooling

        psi_layers = []
        prev = in_channels + 2
        for h in hidden_channels:
            psi_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.LayerNorm(h)])
            prev = h
        psi_layers.append(nn.Linear(prev, hidden_channels[-1]))
        self.psi = nn.Sequential(*psi_layers)
        self.psi_first: nn.Linear = cast(nn.Linear, psi_layers[0])

        phi_layers: list[nn.Module] = []
        prev = hidden_channels[-1]
        for h in hidden_channels:
            phi_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        phi_layers.append(nn.Linear(prev, out_channels))
        self.phi = nn.Sequential(*phi_layers)
        self.phi_last: nn.Linear = cast(nn.Linear, phi_layers[-1])

        if pooling == "persistence_weighted":
            self.weight_net = nn.Sequential(
                nn.Linear(1, hidden_channels[0]),
                nn.ReLU(),
                nn.Linear(hidden_channels[0], 1),
                nn.Softplus(),
            )

    def forward(self, diagram: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        _validate_diagram(diagram)
        batch_size, n_pairs, _ = diagram.shape
        if n_pairs == 0:
            return cast(Tensor, diagram.new_zeros(batch_size, self.phi_last.out_features))

        birth_death = diagram[:, :, :2]

        if features is not None:
            _validate_finite_tensor(features, "features")
            if features.shape[:2] != (batch_size, n_pairs):
                raise ValueError("features must match diagram batch and pair dimensions")
            if features.shape[-1] != self.psi_first.in_features - 2:
                raise ValueError("features must have in_channels columns")
            x = torch.cat([birth_death, features], dim=-1)
        else:
            if self.psi_first.in_features != 2:
                raise ValueError("features are required for this DeepSet")
            x = birth_death

        x_flat = x.view(-1, x.shape[-1])
        psi_out = self.psi(x_flat).view(batch_size, n_pairs, -1)

        if self.pooling == "sum":
            pooled = psi_out.sum(dim=1)
        elif self.pooling == "mean":
            pooled = psi_out.mean(dim=1)
        elif self.pooling == "max":
            pooled = psi_out.max(dim=1)[0]
        elif self.pooling == "persistence_weighted":
            persistence = (diagram[:, :, 1] - diagram[:, :, 0]).unsqueeze(-1)
            weights = self.weight_net(persistence)
            pooled = (psi_out * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return cast(Tensor, self.phi(pooled))
