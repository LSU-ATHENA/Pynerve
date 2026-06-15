"""Graph-level readout for Mapper graphs."""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._validation import validate_positive_int
from ._gnn_validation import _validate_floating_tensor


class TopologyAwareReadout(nn.Module):
    """Readout combining mean, max, and attention-based pooling.

    Produces a fixed-size graph-level representation by concatenating
    three complementary statistics: a mean-pooled projection, a max-pooled
    projection, and an attention-weighted sum.  This captures multiple
    aspects of the graph topology in a single vector.

    """

    def __init__(self, node_dim: int, output_dim: int = 64):
        """Initialize the readout.

        :param node_dim: Dimensionality of node features.
        :param output_dim: Dimensionality of the output representation.
            Must be at least 3.
        :raises ValueError: If ``output_dim < 3``.

        """
        super().__init__()
        node_dim = validate_positive_int(node_dim, "node_dim")
        output_dim = validate_positive_int(output_dim, "output_dim")
        if output_dim < 3:
            raise ValueError("output_dim must be at least 3")

        self.node_dim = node_dim
        self.output_dim = output_dim
        mean_dim = output_dim // 3
        max_dim = output_dim // 3
        attention_dim = output_dim - mean_dim - max_dim

        self.mean_proj = nn.Linear(node_dim, mean_dim)
        self.max_proj = nn.Linear(node_dim, max_dim)
        self.attention_score = nn.Sequential(
            nn.Linear(node_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.attention_value_proj = nn.Linear(node_dim, attention_dim)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Compute graph-level representation via multi-pooling.

        :param node_features: Node feature tensor of shape
            ``(n_nodes, node_dim)``.
        :returns: Graph-level representation of shape ``(output_dim,)``.
        :raises ValueError: If ``node_features`` has wrong shape or
            dimension, or the graph is empty.
        :raises TypeError: If ``node_features`` is not a floating-point
            tensor.

        """
        node_features = _validate_floating_tensor(node_features, "node_features")
        if node_features.dim() != 2:
            raise ValueError("node_features must be a 2D tensor")
        if node_features.shape[0] == 0:
            raise ValueError("cannot read out an empty graph")
        if node_features.shape[1] != self.node_dim:
            raise ValueError(
                f"expected node feature dimension {self.node_dim}, got {node_features.shape[1]}"
            )

        mean_pooled = node_features.mean(dim=0)
        mean_out = self.mean_proj(mean_pooled)

        max_pooled = node_features.max(dim=0)[0]
        max_out = self.max_proj(max_pooled)

        attn_scores = self.attention_score(node_features).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=0)
        attn_pooled = (node_features * attn_weights.unsqueeze(-1)).sum(dim=0)
        attn_out = self.attention_value_proj(attn_pooled)

        return torch.cat([mean_out, max_out, attn_out], dim=-1)
