"""Graph encoder for differentiable Mapper graphs."""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._validation import validate_positive_int


def _validate_node_features(node_features: torch.Tensor, expected_dim: int) -> torch.Tensor:
    if not isinstance(node_features, torch.Tensor):
        raise TypeError("node_features must be a tensor")
    if not torch.is_floating_point(node_features):
        raise TypeError("node_features must use a floating-point dtype")
    if node_features.dim() != 2:
        raise ValueError("node_features must be a 2D tensor")
    if node_features.shape[0] == 0:
        raise ValueError("node_features must be non-empty")
    if node_features.shape[1] != expected_dim:
        raise ValueError(
            f"expected node feature dimension {expected_dim}, got {node_features.shape[1]}"
        )
    if not torch.isfinite(node_features).all().item():
        raise ValueError("node_features must contain only finite values")
    return node_features


def _validated_edges(edges: torch.Tensor, n_nodes: int, device: torch.device) -> torch.Tensor:
    if not isinstance(edges, torch.Tensor):
        raise TypeError("edges must be a tensor")
    if edges.dim() != 2 or edges.shape[0] != 2:
        raise ValueError("edges must have shape (2, n_edges)")
    if torch.is_floating_point(edges) or edges.dtype == torch.bool:
        raise TypeError("edges must contain integer node indices")
    edges = edges.to(device=device, dtype=torch.long)
    if edges.numel() > 0 and (edges.min() < 0 or edges.max() >= n_nodes):
        raise ValueError("edges contain node indices outside node_features")
    return edges


class MapperGraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        node_dim = validate_positive_int(node_dim, "node_dim")
        hidden_dim = validate_positive_int(hidden_dim, "hidden_dim")
        output_dim = validate_positive_int(output_dim, "output_dim")
        num_layers = validate_positive_int(num_layers, "num_layers")

        self.node_dim = node_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_d = node_dim if i == 0 else hidden_dim
            out_d = output_dim if i == num_layers - 1 else hidden_dim

            self.convs.append(nn.Linear(in_d * 2, out_d))
            self.norms.append(nn.LayerNorm(out_d))

    def forward(self, node_features: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        x = _validate_node_features(node_features, self.node_dim)
        edges = _validated_edges(edges, x.shape[0], x.device)

        for conv, norm in zip(self.convs, self.norms, strict=False):
            if edges.numel() > 0:
                src, dst = edges[0], edges[1]

                degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
                degree.index_add_(0, dst, torch.ones_like(src, dtype=x.dtype))
                degree = degree.clamp(min=1)

                neighbor_feats = torch.zeros_like(x)
                neighbor_feats.index_add_(0, dst, x[src])
                neighbor_feats = neighbor_feats / degree.unsqueeze(1)

                x = torch.cat([x, neighbor_feats], dim=1)
            else:
                x = torch.cat([x, torch.zeros_like(x)], dim=1)

            x = conv(x)
            x = norm(x)
            x = F.relu(x)

        return x.mean(dim=0)
