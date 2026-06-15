"""Attention blocks for persistence diagrams."""

from __future__ import annotations

import math
from typing import cast

import torch.nn.functional as F  # noqa: N812

import torch
from torch import Tensor, nn

from .._constants import EPS
from .._validation import validate_finite_tensor as _validate_finite_tensor
from .._validation import validate_positive_int as _validate_positive_int
from ..torch._vectorization_basis import _validate_diagram as _validate_diagram_basis


def _validate_probability(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1")
    return parsed


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


class DiagramMultiHeadAttention(nn.Module):
    """Multi-head self-attention for persistence diagrams."""

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_birth_death_positional: bool = True,
    ):
        super().__init__()
        d_model = _validate_positive_int(d_model, "d_model")
        num_heads = _validate_positive_int(num_heads, "num_heads")
        dropout = _validate_probability("dropout", dropout)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by a positive num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_positional = use_birth_death_positional

        self.qkv_proj = nn.Linear(d_model + 2, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        if use_birth_death_positional:
            self.positional_encoder = nn.Sequential(
                nn.Linear(2, max(1, d_model // 2)),
                nn.ReLU(),
                nn.Linear(max(1, d_model // 2), d_model),
            )

    def forward(
        self,
        diagram: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply multi-head attention."""
        _validate_diagram(diagram)
        _validate_finite_tensor(features, "features")
        if features.shape != (diagram.shape[0], diagram.shape[1], self.d_model):
            raise ValueError("features must have shape (batch, n_pairs, d_model)")
        if diagram.device != features.device:
            raise ValueError("diagram and features must be on the same device")

        batch_size, n_pairs, _ = diagram.shape
        if n_pairs == 0:
            return features.new_zeros(batch_size, 0, self.d_model)

        birth_death = diagram[:, :, :2].to(features.dtype)
        if self.use_positional:
            pos_enc = self.positional_encoder(birth_death)
            features = features + pos_enc

        x = torch.cat([features, birth_death], dim=-1)

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, n_pairs, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_pairs, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_pairs, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        birth_diff = birth_death[:, :, 0:1] - birth_death[:, :, 0:1].transpose(-2, -1)
        death_diff = birth_death[:, :, 1:2] - birth_death[:, :, 1:2].transpose(-2, -1)
        topo_dist = torch.sqrt(birth_diff**2 + death_diff**2 + EPS)
        scores = scores - topo_dist.unsqueeze(1) * 0.1

        if mask is not None:
            _validate_finite_tensor(mask, "mask")
            attention_mask = self._attention_mask(mask, batch_size, n_pairs, scores.shape).to(
                dtype=torch.bool
            )
            if not attention_mask.any(dim=-1).all().item():
                raise ValueError("mask must leave at least one attention target per row")
            scores = scores.masked_fill(~attention_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_pairs, self.d_model)
        return cast(Tensor, self.out_proj(out))

    def _attention_mask(
        self,
        mask: torch.Tensor,
        batch_size: int,
        n_pairs: int,
        score_shape: torch.Size,
    ) -> torch.Tensor:
        if mask.dim() == 2:
            if mask.shape != (batch_size, n_pairs):
                raise ValueError("2D mask must have shape (batch, n_pairs)")
            return mask[:, None, None, :]
        if mask.dim() == 3:
            if mask.shape != (batch_size, n_pairs, n_pairs):
                raise ValueError("3D mask must have shape (batch, n_pairs, n_pairs)")
            return mask[:, None, :, :]
        if mask.shape == score_shape:
            return mask
        raise ValueError("mask has incompatible shape")


class DiagramTransformerBlock(nn.Module):
    """Transformer block for persistence diagrams."""

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = _validate_positive_int(d_ff, "d_ff")
        dropout = _validate_probability("dropout", dropout)

        self.attention = DiagramMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        diagram: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply transformer block."""
        attended = self.attention(diagram, features, mask)
        features = self.norm1(features + self.dropout(attended))

        ff_out = self.feed_forward(features)
        features = self.norm2(features + self.dropout(ff_out))

        return features
