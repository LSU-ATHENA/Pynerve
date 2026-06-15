"""Composed learnable Mapper models."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from .._constants import EPS
from .._validation import validate_positive_int
from ._learnable_mapper_components import (
    AdaptiveCover,
    CoverElement,
    LensFunction,
    SoftClusterAssignment,
)
from ._learnable_mapper_graph import MapperGraphEncoder


def _validate_floating_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must use a floating-point dtype")
    if tensor.numel() > 0 and not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


class DifferentiableMapper(nn.Module):
    """Differentiable Mapper pipeline for point cloud data.

    Applies a learnable lens function, creates an adaptive cover over the
    lens space, performs soft cluster assignment of data points to cover
    elements, and encodes the resulting Mapper graph via a graph neural
    network encoder.
    """

    def __init__(
        self,
        input_dim: int,
        lens_output_dim: int = 2,
        lens_hidden_dims: list[int] | None = None,
        min_cover_resolution: int = 2,
        max_cover_resolution: int = 10,
        cluster_method: str = "soft",
        temperature: float = 0.1,
    ):
        """Initialize the DifferentiableMapper.

        :param input_dim: Dimensionality of input point cloud data.
        :param lens_output_dim: Output dimensionality of the lens function.
        :param lens_hidden_dims: Hidden layer dimensions for the lens network.
        :param min_cover_resolution: Minimum number of intervals per lens
            dimension.
        :param max_cover_resolution: Maximum number of intervals per lens
            dimension.
        :param cluster_method: Cluster assignment method (currently only
            ``"soft"``).
        :param temperature: Temperature for the soft cluster assignment.

        :raises TypeError: If *input_dim* or *lens_output_dim* cannot be
            cast to a positive integer.
        :raises ValueError: If *cluster_method* is not ``"soft"``.
        """
        super().__init__()
        input_dim = validate_positive_int(input_dim, "input_dim")
        lens_output_dim = validate_positive_int(lens_output_dim, "lens_output_dim")
        if cluster_method != "soft":
            raise ValueError("DifferentiableMapper currently supports cluster_method='soft'")

        self.input_dim = input_dim
        self.lens_output_dim = lens_output_dim
        self.cluster_method = cluster_method

        self.lens = LensFunction(input_dim, lens_output_dim, lens_hidden_dims)
        self.cover = AdaptiveCover(min_cover_resolution, max_cover_resolution)
        self.soft_cluster = SoftClusterAssignment(temperature)

        self.graph_encoder = MapperGraphEncoder(node_dim=input_dim, hidden_dim=64, output_dim=32)

    def forward(self, data: torch.Tensor, return_graph: bool = False) -> dict[str, torch.Tensor]:
        """Run the DifferentiableMapper pipeline on a batch of point clouds.

        :param data: Input tensor of shape ``(batch, n_points, input_dim)``.
        :param return_graph: If ``True``, include the raw Mapper graph dict
            for each batch item in the output.

        :returns: Dictionary with keys:

            - ``lens_values``: Lens function outputs, shape
              ``(batch, n_points, lens_output_dim)``.
            - ``cover_intervals``: List of cover element intervals per batch
              item.
            - ``cluster_assignments``: Soft assignment matrix per batch item.
            - ``graph_embedding``: Stacked graph embeddings, shape
              ``(batch, 32)``.
            - ``mapper_graph``: (optional) List of Mapper graph dicts per
              batch item.

        :raises TypeError: If *data* is not a floating-point tensor.
        :raises ValueError: If *data* has wrong shape, batch size, dimension,
            or contains non-finite values.
        """
        data = _validate_floating_tensor(data, "data")
        if data.dim() != 3:
            raise ValueError(f"Expected data shape (batch, n_points, dim), got {tuple(data.shape)}")
        batch_size, n_points, dim = data.shape
        if batch_size == 0:
            raise ValueError("DifferentiableMapper requires at least one batch item")
        if dim != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {dim}")
        if n_points == 0:
            raise ValueError("DifferentiableMapper requires at least one point")

        lens_values = self.lens(data)

        all_intervals = []
        all_assignments = []

        for b in range(batch_size):
            lv = lens_values[b]
            intervals = self.cover.create_cover(lv)
            all_intervals.append(intervals)
            assignments = self._cluster_in_cover(lv, data[b], intervals)
            all_assignments.append(assignments)

        graph_embeddings = []
        mapper_graphs = []

        for b in range(batch_size):
            graph = self._build_mapper_graph(data[b], all_assignments[b])

            if return_graph:
                mapper_graphs.append(graph)

            embedding = self.graph_encoder(graph["node_features"], graph["edges"])
            graph_embeddings.append(embedding)

        result = {
            "lens_values": lens_values,
            "cover_intervals": all_intervals,
            "cluster_assignments": all_assignments,
            "graph_embedding": torch.stack(graph_embeddings),
        }

        if return_graph:
            result["mapper_graph"] = mapper_graphs

        return result

    def _cluster_in_cover(
        self,
        lens_values: torch.Tensor,
        data: torch.Tensor,
        intervals: list[CoverElement],
    ) -> torch.Tensor:
        n_points = data.shape[0]
        cluster_centers: list[torch.Tensor] = []

        for cover_box in intervals:
            if lens_values.dim() == 1:
                start, end = cover_box[0]
                in_cover = (lens_values >= start) & (lens_values <= end)
            else:
                in_cover = torch.ones(n_points, dtype=torch.bool, device=data.device)
                for d, (start, end) in enumerate(cover_box):
                    in_cover &= (lens_values[:, d] >= start) & (lens_values[:, d] <= end)

            center = data.mean(dim=0) if in_cover.sum() == 0 else data[in_cover].mean(dim=0)

            cluster_centers.append(center)

        cluster_centers_tensor = torch.stack(cluster_centers)

        return cast(torch.Tensor, self.soft_cluster(data, cluster_centers_tensor))

    def _build_mapper_graph(
        self,
        data: torch.Tensor,
        assignments: torch.Tensor,
    ) -> dict[str, Any]:
        n_clusters = assignments.shape[1]

        node_features_list: list[torch.Tensor] = []
        for c in range(n_clusters):
            weights = assignments[:, c : c + 1]
            weighted_sum = (data * weights).sum(dim=0)
            weight_total = weights.sum() + EPS
            node_feat = weighted_sum / weight_total
            node_features_list.append(node_feat)

        node_features = torch.stack(node_features_list)

        edges_list: list[list[int]] = []
        edge_weights_list: list[float] = []

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                shared = (assignments[:, i] * assignments[:, j]).sum()

                if shared.item() > 0.01:
                    edges_list.append([i, j])
                    edges_list.append([j, i])
                    edge_weights_list.extend([shared.item(), shared.item()])

        edges = (
            torch.tensor(edges_list, dtype=torch.long, device=data.device).t()
            if edges_list
            else torch.zeros((2, 0), dtype=torch.long, device=data.device)
        )
        edge_weights = (
            torch.tensor(edge_weights_list, dtype=data.dtype, device=data.device)
            if edge_weights_list
            else torch.empty(0, dtype=data.dtype, device=data.device)
        )

        return {
            "node_features": node_features,
            "edges": edges,
            "edge_weights": edge_weights,
            "n_nodes": n_clusters,
            "n_edges": len(edge_weights_list),
        }


class MapperAutoencoder(nn.Module):
    """Autoencoder that uses a DifferentiableMapper as the encoder backbone.

    Encodes point clouds through the Mapper pipeline and a linear
    projection into a latent space, then decodes through a multilayer
    perceptron.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        mapper_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
    ):
        """Initialize the MapperAutoencoder.

        :param input_dim: Dimensionality of input point cloud data.
        :param latent_dim: Dimensionality of the latent bottleneck.
        :param mapper_hidden_dims: Hidden dims for the Mapper's lens network.
        :param decoder_hidden_dims: Hidden dims for the decoder MLP (default
            ``[64, 128]``).

        :raises TypeError: If *input_dim*, *latent_dim*, or any value in
            *decoder_hidden_dims* cannot be cast to a positive integer.
        """
        super().__init__()
        input_dim = validate_positive_int(input_dim, "input_dim")
        latent_dim = validate_positive_int(latent_dim, "latent_dim")
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 128]

        self.mapper = DifferentiableMapper(
            input_dim=input_dim, lens_output_dim=2, lens_hidden_dims=mapper_hidden_dims
        )

        self.encoder_proj = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, latent_dim))

        decoder_layers = []
        prev = latent_dim
        for h in decoder_hidden_dims:
            h = validate_positive_int(h, "decoder_hidden_dims")
            decoder_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode a batch of point clouds to latent representations.

        :param data: Input point cloud of shape
            ``(batch, n_points, input_dim)``.

        :returns: Latent tensor of shape ``(batch, latent_dim)``.

        :raises TypeError: If *data* is not a floating-point tensor.
        :raises ValueError: If *data* contains non-finite values.
        """
        data = _validate_floating_tensor(data, "data")
        mapper_out = self.mapper(data)
        return cast(torch.Tensor, self.encoder_proj(mapper_out["graph_embedding"]))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent codes back to the original input space.

        :param latent: Latent tensor of shape ``(batch, latent_dim)``.

        :returns: Reconstructed output of shape ``(batch, input_dim)``.

        :raises TypeError: If *latent* is not a floating-point tensor.
        :raises ValueError: If *latent* has wrong shape or contains
            non-finite values.
        """
        latent = _validate_floating_tensor(latent, "latent")
        if latent.dim() != 2 or latent.shape[1] != self.encoder_proj[-1].out_features:
            raise ValueError("latent must have shape (batch, latent_dim)")
        return cast(torch.Tensor, self.decoder(latent))

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode a batch of point clouds end-to-end.

        :param data: Input point cloud of shape
            ``(batch, n_points, input_dim)``.

        :returns: ``(reconstruction, latent)`` tuple, where *reconstruction*
            has shape ``(batch, input_dim)`` and *latent* has shape
            ``(batch, latent_dim)``.
        """
        latent = self.encode(data)
        reconstruction = self.decode(latent)
        return reconstruction, latent
