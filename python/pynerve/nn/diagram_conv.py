"""Neural architectures for persistence diagrams."""

from __future__ import annotations

from ._diagram_attention import DiagramMultiHeadAttention, DiagramTransformerBlock
from ._diagram_conv_layers import DiagramConv1D, DiagramDeepSet
from ._diagram_conv_net import DiagramConvNet
from ._diagram_pooling import DiagramPooling

__all__ = [
    "DiagramConv1D",
    "DiagramDeepSet",
    "DiagramMultiHeadAttention",
    "DiagramTransformerBlock",
    "DiagramPooling",
    "DiagramConvNet",
]
