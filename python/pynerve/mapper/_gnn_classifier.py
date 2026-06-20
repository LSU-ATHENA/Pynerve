"""End-to-end Mapper GNN classifier."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from .._validation import validate_positive_int
from ._gnn_conv import MapperNodeEncoder
from ._gnn_pooling import HierarchicalMapperPooling
from ._gnn_readout import TopologyAwareReadout
from ._gnn_validation import _validate_floating_tensor, _validated_edges


class MapperGNNClassifier(nn.Module):
    """End-to-end classifier for Mapper graphs.

    Combines :class:`MapperNodeEncoder`, optional
    :class:`HierarchicalMapperPooling`, :class:`TopologyAwareReadout`, and
    a final MLP classifier into a single model.  Supports batched graph
    classification via ``batch_vector``.

    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 10,
        num_gnn_layers: int = 3,
        use_hierarchical: bool = True,
    ):
        """Initialize the classifier.

        :param node_dim: Dimensionality of input node features.
        :param hidden_dim: Dimensionality of hidden representations.
        :param num_classes: Number of output classes.
        :param num_gnn_layers: Number of GNN layers in the encoder.
        :param use_hierarchical: Whether to use hierarchical pooling.

        """
        super().__init__()
        node_dim = validate_positive_int(node_dim, "node_dim")
        hidden_dim = validate_positive_int(hidden_dim, "hidden_dim")
        num_classes = validate_positive_int(num_classes, "num_classes")
        num_gnn_layers = validate_positive_int(num_gnn_layers, "num_gnn_layers")

        self.node_dim = node_dim
        self.encoder = MapperNodeEncoder(node_dim, hidden_dim, num_gnn_layers)
        self.use_hierarchical = use_hierarchical
        self.hierarchical_levels = 3
        if use_hierarchical:
            self.hierarchical_pool = HierarchicalMapperPooling(
                hidden_dim, num_levels=self.hierarchical_levels
            )
            self.hierarchical_fusion = nn.Sequential(
                nn.Linear(128 * (self.hierarchical_levels + 1), 128),
                nn.ReLU(),
            )

        self.readout = TopologyAwareReadout(hidden_dim, 128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edges: torch.Tensor,
        lens_values: torch.Tensor | None = None,
        batch_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Classify a (batch of) Mapper graph(s).

        :param node_features: Node feature tensor of shape
            ``(n_nodes, node_dim)``.
        :param edges: Edge index tensor of shape ``(2, n_edges)``.
        :param lens_values: Optional lens coordinate tensor of shape
            ``(n_nodes, 2)``.
        :param batch_vector: Optional batch assignment tensor of shape
            ``(n_nodes,)`` with integer graph indices.  When provided,
            each graph in the batch is processed independently and
            the output is stacked.
        :returns: Class logits of shape ``(num_classes,)`` for a single
            graph, or ``(n_batches, num_classes)`` for batched input.
        :raises ValueError: If ``node_features`` has wrong shape or
            dimension.
        :raises TypeError: If ``node_features`` is not a floating-point
            tensor, or ``batch_vector`` has an invalid dtype.

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
        edges = _validated_edges(edges, node_features.shape[0], node_features.device)
        if batch_vector is not None:
            if not isinstance(batch_vector, torch.Tensor):
                raise TypeError("batch_vector must be a tensor")
            if torch.is_floating_point(batch_vector) or batch_vector.dtype == torch.bool:
                raise TypeError("batch_vector must contain integer graph indices")
            if batch_vector.shape != (node_features.shape[0],):
                raise ValueError("batch_vector must have shape (n_nodes,)")

        x = self.encoder(node_features, edges, lens_values)

        if batch_vector is not None:
            batch_vector = batch_vector.to(device=x.device)
            graph_embeddings = []
            assert batch_vector is not None
            for batch_id in batch_vector.unique(sorted=True):
                mask = batch_vector == batch_id
                local_edges = self._induced_edges(edges, mask)
                graph_embeddings.append(self._read_graph(x[mask], local_edges))
            graph_embedding = torch.stack(graph_embeddings)
        else:
            graph_embedding = self._read_graph(x, edges)

        return cast(torch.Tensor, self.classifier(graph_embedding))

    def _read_graph(self, node_features: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if self.use_hierarchical:
            levels = self.hierarchical_pool(node_features, edges)
            level_embeddings = torch.cat([self.readout(level) for level in levels], dim=-1)
            return cast(torch.Tensor, self.hierarchical_fusion(level_embeddings))
        return cast(torch.Tensor, self.readout(node_features))

    @staticmethod
    def _induced_edges(edges: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if edges.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=mask.device)

        edge_mask = mask[edges[0]] & mask[edges[1]]
        selected_edges = edges[:, edge_mask]
        if selected_edges.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=mask.device)

        node_ids = torch.nonzero(mask, as_tuple=False).flatten()
        node_lookup = torch.full((mask.shape[0],), -1, dtype=torch.long, device=mask.device)
        node_lookup[node_ids] = torch.arange(node_ids.numel(), device=mask.device)
        return node_lookup[selected_edges]
