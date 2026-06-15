"""Hierarchical graph pooling for Mapper graphs."""

from __future__ import annotations

import torch
from torch import nn

from .._validation import validate_positive_int
from ._gnn_validation import _validate_floating_tensor, _validate_probability, _validated_edges


class HierarchicalMapperPooling(nn.Module):
    """Hierarchical graph pooling for Mapper graphs.

    Stacks multiple :class:`MapperPoolingLayer` instances to produce a
    sequence of progressively coarser node feature tensors.  The output is
    a list where index 0 holds the original node features and subsequent
    entries correspond to each pooling level.

    """

    def __init__(self, node_dim: int, num_levels: int = 3, pool_ratio: float = 0.5):
        """Initialize hierarchical pooling.

        :param node_dim: Dimensionality of node features.
        :param num_levels: Number of pooling levels.
        :param pool_ratio: Fraction of nodes to retain at each pooling
            step.  Must be in ``(0, 1]``.

        """
        super().__init__()
        node_dim = validate_positive_int(node_dim, "node_dim")
        num_levels = validate_positive_int(num_levels, "num_levels")
        pool_ratio = _validate_probability(pool_ratio, "pool_ratio")

        self.node_dim = node_dim
        self.num_levels = num_levels
        self.pool_ratio = pool_ratio

        self.pool_layers = nn.ModuleList()
        for _ in range(num_levels):
            self.pool_layers.append(MapperPoolingLayer(node_dim, node_dim, pool_ratio))

    def forward(self, node_features: torch.Tensor, edges: torch.Tensor) -> list[torch.Tensor]:
        """Apply hierarchical pooling to node features.

        :param node_features: Node feature tensor of shape
            ``(n_nodes, node_dim)``.
        :param edges: Edge index tensor of shape ``(2, n_edges)``.
        :returns: List of pooled node feature tensors, one per level.
            The first element is the original input.
        :raises ValueError: If ``node_features`` has wrong shape or
            dimension.
        :raises TypeError: If ``node_features`` is not a floating-point
            tensor.

        """
        node_features = _validate_floating_tensor(node_features, "node_features")
        if node_features.dim() != 2:
            raise ValueError("node_features must be a 2D tensor")
        if node_features.shape[0] == 0:
            raise ValueError("node_features must be non-empty")
        if node_features.shape[1] != self.node_dim:
            raise ValueError(
                f"expected node feature dimension {self.node_dim}, got {node_features.shape[1]}"
            )

        pooled_features = [node_features]
        current_edges = _validated_edges(edges, node_features.shape[0], node_features.device)

        for pool_layer in self.pool_layers:
            node_features, current_edges = pool_layer(node_features, current_edges)
            pooled_features.append(node_features)

        return pooled_features


class MapperPoolingLayer(nn.Module):
    """Single graph pooling layer.

    Selects the top-*k* nodes by a learned score function, then assigns
    every node to its nearest top node via Euclidean distance.  Clusters
    are averaged, projected, and new edges are induced from the assignment.

    """

    def __init__(self, in_dim: int, out_dim: int, pool_ratio: float):
        """Initialize the pooling layer.

        :param in_dim: Input feature dimension.
        :param out_dim: Output feature dimension.
        :param pool_ratio: Fraction of nodes to retain.  Must be in
            ``(0, 1]``.

        """
        super().__init__()
        in_dim = validate_positive_int(in_dim, "in_dim")
        out_dim = validate_positive_int(out_dim, "out_dim")
        pool_ratio = _validate_probability(pool_ratio, "pool_ratio")

        self.in_dim = in_dim
        self.pool_ratio = pool_ratio
        self.score_net = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(
        self, node_features: torch.Tensor, edges: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool the graph to a smaller set of nodes.

        :param node_features: Node feature tensor of shape
            ``(n_nodes, in_dim)``.
        :param edges: Edge index tensor of shape ``(2, n_edges)``.
        :returns: Tuple of ``(pooled_features, pooled_edges)`` where
            ``pooled_features`` has shape ``(n_pooled, out_dim)`` and
            ``pooled_edges`` has shape ``(2, n_pooled_edges)``.
        :raises ValueError: If ``node_features`` has wrong shape or
            dimension, or the graph is empty.
        :raises TypeError: If ``node_features`` is not a floating-point
            tensor.

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
            raise ValueError("cannot pool an empty graph")
        edges = _validated_edges(edges, n_nodes, node_features.device)
        n_pooled = max(1, int(n_nodes * self.pool_ratio))

        scores = self.score_net(node_features).squeeze(-1)
        _, top_indices = torch.topk(scores, n_pooled)

        assignments = torch.argmin(torch.cdist(node_features, node_features[top_indices]), dim=1)

        pooled_features = torch.zeros(
            n_pooled,
            node_features.shape[1],
            device=node_features.device,
            dtype=node_features.dtype,
        )
        pooled_features.index_add_(0, assignments, node_features)

        cluster_sizes = torch.bincount(assignments, minlength=n_pooled).to(
            device=node_features.device, dtype=node_features.dtype
        )
        pooled_features = pooled_features / cluster_sizes.unsqueeze(-1).clamp(min=1)

        pooled_features = self.proj(pooled_features)

        if edges.numel() > 0:
            src, dst = edges[0], edges[1]
            pooled_src = assignments[src]
            pooled_dst = assignments[dst]

            pooled_edges = torch.stack([pooled_src, pooled_dst])
            pooled_edges = torch.unique(pooled_edges, dim=1)

            mask = pooled_edges[0] != pooled_edges[1]
            pooled_edges = pooled_edges[:, mask]
        else:
            pooled_edges = torch.zeros((2, 0), dtype=torch.long, device=node_features.device)

        return pooled_features, pooled_edges
