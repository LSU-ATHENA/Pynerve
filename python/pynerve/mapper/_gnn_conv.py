"""Graph convolution layers for Mapper graphs."""

from __future__ import annotations

from typing import cast

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._validation import validate_positive_int
from ._gnn_validation import _validate_floating_tensor, _validated_edges


class MapperNodeEncoder(nn.Module):
    """Encoder for Mapper graph nodes.

    Encodes node features through stacked :class:`MapperGraphConv` layers
    with residual connections, layer norm, and ReLU activations.  When
    ``use_lens_positional`` is enabled, lens coordinate values are encoded
    via a small MLP and added as positional information.

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        use_lens_positional: bool = True,
    ):
        """Initialize the node encoder.

        :param input_dim: Dimensionality of input node features.
        :param hidden_dim: Dimensionality of hidden representations.
        :param num_layers: Number of graph convolution layers.
        :param use_lens_positional: Whether to encode lens values as
            positional information.

        """
        super().__init__()
        input_dim = validate_positive_int(input_dim, "input_dim")
        hidden_dim = validate_positive_int(hidden_dim, "hidden_dim")
        num_layers = validate_positive_int(num_layers, "num_layers")

        self.input_dim = input_dim
        self.use_lens_positional = use_lens_positional
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(MapperGraphConv(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        if use_lens_positional:
            lens_hidden_dim = max(1, hidden_dim // 4)
            self.lens_encoder = nn.Sequential(
                nn.Linear(2, lens_hidden_dim),
                nn.ReLU(),
                nn.Linear(lens_hidden_dim, hidden_dim),
            )

    def forward(
        self,
        node_features: torch.Tensor,
        edges: torch.Tensor,
        lens_values: torch.Tensor | None = None,
        cover_overlap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode node features through graph convolution layers.

        :param node_features: Node feature tensor of shape ``(n_nodes, input_dim)``.
        :param edges: Edge index tensor of shape ``(2, n_edges)``.
        :param lens_values: Optional lens coordinate tensor of shape
            ``(n_nodes, 2)`` used as positional encoding.
        :param cover_overlap: Optional edge weight tensor passed to
            :meth:`MapperGraphConv.forward`.
        :returns: Encoded node representations of shape
            ``(n_nodes, hidden_dim)``.
        :raises ValueError: If ``node_features`` has wrong shape or
            dimension.
        :raises TypeError: If ``node_features`` or ``lens_values`` are not
            floating-point tensors.

        """
        node_features = _validate_floating_tensor(node_features, "node_features")
        if node_features.dim() != 2:
            raise ValueError("node_features must be a 2D tensor")
        if node_features.shape[0] == 0:
            raise ValueError("node_features must be non-empty")
        if node_features.shape[1] != self.input_dim:
            raise ValueError(
                f"expected node feature dimension {self.input_dim}, got {node_features.shape[1]}"
            )
        edges = _validated_edges(edges, node_features.shape[0], node_features.device)

        x = self.input_proj(node_features)

        if self.use_lens_positional and lens_values is not None:
            lens_values = _validate_floating_tensor(lens_values, "lens_values")
            assert lens_values is not None
            if lens_values.shape != (node_features.shape[0], 2):
                raise ValueError("lens_values must have shape (n_nodes, 2)")
            pos_enc = self.lens_encoder(lens_values.to(device=x.device, dtype=x.dtype))
            x = x + pos_enc

        for conv, ln in zip(self.convs, self.layer_norms, strict=False):
            x_new = conv(x, edges, cover_overlap)
            x = ln(x + x_new)
            x = F.relu(x)

        return cast(torch.Tensor, x)


class MapperGraphConv(nn.Module):
    """Message-passing graph convolution layer.

    Transforms both self and neighbour features, then aggregates neighbour
    messages via mean-pooling.  Supports optional edge gating through
    ``edge_weights`` that are passed through a sigmoid-gated MLP.

    """

    def __init__(self, in_dim: int, out_dim: int):
        """Initialize the graph convolution layer.

        :param in_dim: Input feature dimension.
        :param out_dim: Output feature dimension.

        """
        super().__init__()
        in_dim = validate_positive_int(in_dim, "in_dim")
        out_dim = validate_positive_int(out_dim, "out_dim")

        self.in_dim = in_dim
        self.self_transform = nn.Linear(in_dim, out_dim)
        self.neighbor_transform = nn.Linear(in_dim, out_dim)
        self.edge_gate = nn.Sequential(nn.Linear(1, out_dim), nn.Sigmoid())

    def forward(
        self,
        node_features: torch.Tensor,
        edges: torch.Tensor,
        edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply graph convolution to node features.

        :param node_features: Node feature tensor of shape ``(n_nodes, in_dim)``.
        :param edges: Edge index tensor of shape ``(2, n_edges)``.
        :param edge_weights: Optional edge weight tensor of shape
            ``(n_edges,)``.  When provided, neighbour messages are gated
            by a learned sigmoid MLP on these weights.
        :returns: Updated node features of shape ``(n_nodes, out_dim)``.
        :raises ValueError: If ``node_features`` has wrong shape or
            dimension, or ``edge_weights`` has wrong shape.
        :raises TypeError: If ``node_features`` or ``edge_weights`` are not
            floating-point tensors.

        """
        node_features = _validate_floating_tensor(node_features, "node_features")
        if node_features.dim() != 2:
            raise ValueError("node_features must be a 2D tensor")
        if node_features.shape[1] != self.in_dim:
            raise ValueError(
                f"expected node feature dimension {self.in_dim}, got {node_features.shape[1]}"
            )

        n_nodes = node_features.shape[0]
        if n_nodes == 0:
            raise ValueError("node_features must be non-empty")
        edges = _validated_edges(edges, n_nodes, node_features.device)

        self_out = self.self_transform(node_features)

        if edges.numel() > 0:
            src, dst = edges[0], edges[1]
            neighbor_feats = self.neighbor_transform(node_features[src])

            if edge_weights is not None:
                edge_weights = _validate_floating_tensor(edge_weights, "edge_weights")
                assert edge_weights is not None
                if edge_weights.dim() != 1 or edge_weights.shape[0] != edges.shape[1]:
                    raise ValueError("edge_weights must have shape (n_edges,)")
                edge_weights = edge_weights.to(
                    device=node_features.device, dtype=node_features.dtype
                )
                assert edge_weights is not None
                neighbor_feats = neighbor_feats * self.edge_gate(edge_weights.unsqueeze(-1))

            neighbor_out = torch.zeros(
                n_nodes,
                self_out.shape[1],
                device=node_features.device,
                dtype=node_features.dtype,
            )
            neighbor_out.index_add_(0, dst, neighbor_feats)

            degrees = torch.zeros(n_nodes, device=node_features.device, dtype=node_features.dtype)
            degrees.index_add_(0, dst, torch.ones_like(dst, dtype=node_features.dtype))
            degrees = degrees.clamp(min=1).unsqueeze(-1)
            neighbor_out = neighbor_out / degrees
        else:
            neighbor_out = torch.zeros_like(self_out)

        return cast(torch.Tensor, self_out + neighbor_out)
