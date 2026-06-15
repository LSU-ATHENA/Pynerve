"""Composed convolutional network for persistence diagrams."""

from __future__ import annotations

from typing import cast

import torch.nn.functional as F  # noqa: N812

import torch
from torch import Tensor, nn

from ._diagram_conv_layers import (
    DiagramConv1D,
    _validate_diagram,
    _validate_non_negative_int,
    _validate_positive_int,
)
from ._diagram_pooling import DiagramPooling


class DiagramConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: list[int] | None = None,
        out_dim: int = 10,
        num_prototypes: int = 8,
        pooling: str = "persistence_weighted",
    ):
        super().__init__()
        in_channels = _validate_non_negative_int("in_channels", in_channels)
        out_dim = _validate_positive_int(out_dim, "out_dim")
        num_prototypes = _validate_positive_int(num_prototypes, "num_prototypes")
        hidden_channels = [64, 128, 256] if hidden_channels is None else hidden_channels
        if not hidden_channels:
            raise ValueError("hidden_channels must contain positive dimensions")
        hidden_channels = [_validate_positive_int(ch, "hidden_channels") for ch in hidden_channels]
        pooling_type = "attention" if pooling == "persistence_weighted" else pooling
        if pooling_type not in {"attention", "persistence_clustering"}:
            raise ValueError("unknown pooling mode")

        conv_layers = []
        prev_ch = in_channels
        for h in hidden_channels:
            conv_layers.append(DiagramConv1D(prev_ch, h, use_persistence_weighting=True))
            prev_ch = h
        self.convs = nn.ModuleList(conv_layers)
        self.in_channels = in_channels

        self.pool = DiagramPooling(
            hidden_channels[-1],
            hidden_channels[-1],
            num_prototypes,
            pooling_type=pooling_type,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_prototypes * hidden_channels[-1], 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, out_dim),
        )

    def forward(self, diagram: torch.Tensor) -> torch.Tensor:
        _validate_diagram(diagram)
        if diagram.dim() != 3 or diagram.shape[-1] < max(2, self.in_channels):
            raise ValueError("diagram must have shape (batch, n_pairs, channels)")

        diagram = diagram.where(torch.isfinite(diagram), torch.zeros_like(diagram))
        features = diagram[:, :, : self.in_channels] if self.in_channels else None

        for conv in self.convs:
            conv_out = F.relu(conv(diagram, features))
            features = conv_out.transpose(1, 2)

        if features is None:
            raise RuntimeError("DiagramConvNet requires at least one convolution layer")

        return cast(Tensor, self.classifier(self.pool(diagram, features)))


__all__ = ["DiagramConvNet"]
