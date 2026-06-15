"""Tests for pynerve mapper subpackage."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _cuda_functional() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        torch.cuda.current_device()
        return True
    except Exception:
        return False


# synthetic test helpers


def _make_diagram(
    n_pairs: int = 5,
    max_val: float = 1.0,
    dims: bool = False,
    device: str = "cpu",
    seed: int = 42,
) -> torch.Tensor:
    """Construct a valid diagram tensor with birth < death."""
    rng = torch.Generator(device=device).manual_seed(seed)
    births = torch.rand(n_pairs, generator=rng, device=device) * max_val * 0.8
    deaths = births + torch.rand(n_pairs, generator=rng, device=device) * max_val * 0.4 + 0.01
    pairs = torch.stack([births, deaths], dim=1)
    if dims:
        dims_t = torch.randint(0, 3, (n_pairs, 1), generator=rng, device=device).float()
        return torch.cat([pairs, dims_t], dim=1)
    return pairs


def _make_points(
    batch: int = 2, n_points: int = 10, dim: int = 3, device: str = "cpu", seed: int = 42
) -> torch.Tensor:
    rng = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(batch, n_points, dim, generator=rng, device=device)


# mapper tests


class TestLensFunction:
    def test_2d_input(self):
        from pynerve.mapper._learnable_mapper_components import LensFunction

        lens = LensFunction(input_dim=5, output_dim=2, hidden_dims=[16, 8])
        x = torch.randn(10, 5)
        out = lens(x)
        assert out.shape == (10, 2)
        assert torch.isfinite(out).all()

    def test_3d_input(self):
        from pynerve.mapper._learnable_mapper_components import LensFunction

        lens = LensFunction(input_dim=3, output_dim=2, hidden_dims=[8])
        x = torch.randn(2, 10, 3)
        out = lens(x)
        assert out.shape == (2, 10, 2)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        from pynerve.mapper._learnable_mapper_components import LensFunction

        lens = LensFunction(input_dim=3, output_dim=2)
        x = torch.randn(10, 3, requires_grad=True)
        out = lens(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_leaky_relu_activation(self):
        from pynerve.mapper._learnable_mapper_components import LensFunction

        lens = LensFunction(input_dim=3, output_dim=2, activation="leaky_relu")
        out = lens(torch.randn(5, 3))
        assert out.shape == (5, 2)

    def test_wrong_input_dim_raises(self):
        from pynerve.mapper._learnable_mapper_components import LensFunction

        lens = LensFunction(input_dim=5, output_dim=2)
        with pytest.raises(ValueError):
            lens(torch.randn(10, 3))
        with pytest.raises(ValueError):
            lens(torch.randn(2, 10, 3))


class TestAdaptiveCover:
    def test_get_cover_params(self):
        from pynerve.mapper._learnable_mapper_components import AdaptiveCover

        cover = AdaptiveCover(min_resolution=2, max_resolution=10)
        resolution, overlap = cover.get_cover_params()
        assert 2 <= resolution <= 10
        assert 0.1 <= overlap <= 0.5

    def test_create_cover_1d(self):
        from pynerve.mapper._learnable_mapper_components import AdaptiveCover

        cover = AdaptiveCover(min_resolution=3, max_resolution=3)
        lens = torch.linspace(0, 1, 10)
        intervals = cover.create_cover(lens)
        assert len(intervals) == 3

    def test_create_cover_2d(self):
        from pynerve.mapper._learnable_mapper_components import AdaptiveCover

        cover = AdaptiveCover(min_resolution=4, max_resolution=4)
        lens = torch.rand(10, 2)
        intervals = cover.create_cover(lens)
        assert len(intervals) > 0

    def test_gradient_flow(self):
        from pynerve.mapper._learnable_mapper_components import AdaptiveCover

        cover = AdaptiveCover()
        cover.get_cover_params()
        params = list(cover.parameters())
        assert len(params) > 0
        cover.resolution_logits.sum().backward()


class TestSoftClusterAssignment:
    def test_forward_shape(self):
        from pynerve.mapper._learnable_mapper_components import SoftClusterAssignment

        cluster = SoftClusterAssignment(temperature=0.1)
        data = torch.randn(10, 5)
        centers = torch.randn(3, 5)
        assignments = cluster(data, centers)
        assert assignments.shape == (10, 3)
        row_sums = assignments.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(10), atol=1e-5)

    def test_gradient_flow(self):
        from pynerve.mapper._learnable_mapper_components import SoftClusterAssignment

        cluster = SoftClusterAssignment(temperature=0.1)
        data = torch.randn(10, 5, requires_grad=True)
        centers = torch.randn(3, 5, requires_grad=True)
        assignments = cluster(data, centers)
        loss = assignments.sum()
        loss.backward()
        assert data.grad is not None
        assert centers.grad is not None


class TestMapperGraphEncoder:
    def test_forward(self):
        from pynerve.mapper._learnable_mapper_graph import MapperGraphEncoder

        enc = MapperGraphEncoder(node_dim=5, hidden_dim=16, output_dim=8, num_layers=2)
        feats = torch.randn(6, 5)
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = enc(feats, edges)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()

    def test_no_edges(self):
        from pynerve.mapper._learnable_mapper_graph import MapperGraphEncoder

        enc = MapperGraphEncoder(node_dim=5, hidden_dim=16, output_dim=8)
        feats = torch.randn(4, 5)
        edges = torch.zeros((2, 0), dtype=torch.long)
        out = enc(feats, edges)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        from pynerve.mapper._learnable_mapper_graph import MapperGraphEncoder

        enc = MapperGraphEncoder(node_dim=3, hidden_dim=8, output_dim=4)
        feats = torch.randn(5, 3, requires_grad=True)
        edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = enc(feats, edges)
        loss = out.sum()
        loss.backward()
        assert feats.grad is not None
        assert torch.isfinite(feats.grad).all()


class TestDifferentiableMapper:
    def test_forward_basic(self):
        from pynerve.mapper._learnable_mapper_models import DifferentiableMapper

        mapper = DifferentiableMapper(
            input_dim=3,
            lens_output_dim=2,
            lens_hidden_dims=[16, 8],
            min_cover_resolution=3,
            max_cover_resolution=3,
        )
        data = torch.randn(2, 8, 3)
        result = mapper(data)
        assert "graph_embedding" in result
        assert result["graph_embedding"].shape == (2, 32)
        assert "lens_values" in result
        assert result["lens_values"].shape == (2, 8, 2)
        assert "cluster_assignments" in result
        assert torch.isfinite(result["graph_embedding"]).all()

    def test_return_graph(self):
        from pynerve.mapper._learnable_mapper_models import DifferentiableMapper

        mapper = DifferentiableMapper(
            input_dim=2,
            lens_output_dim=2,
            min_cover_resolution=2,
            max_cover_resolution=2,
        )
        data = torch.randn(1, 5, 2)
        result = mapper(data, return_graph=True)
        assert "mapper_graph" in result
        assert len(result["mapper_graph"]) == 1

    def test_gradient_flow(self):
        from pynerve.mapper._learnable_mapper_models import DifferentiableMapper

        mapper = DifferentiableMapper(
            input_dim=2,
            lens_output_dim=2,
            min_cover_resolution=2,
            max_cover_resolution=2,
        )
        data = torch.randn(1, 5, 2, requires_grad=True)
        result = mapper(data)
        loss = result["graph_embedding"].sum()
        loss.backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()

    def test_graph_node_features_dim(self):
        from pynerve.mapper._learnable_mapper_models import DifferentiableMapper

        mapper = DifferentiableMapper(
            input_dim=4,
            lens_output_dim=2,
            min_cover_resolution=2,
            max_cover_resolution=2,
        )
        data = torch.randn(1, 8, 4)
        result = mapper(data, return_graph=True)
        graph = result["mapper_graph"][0]
        assert graph["node_features"].dim() == 2
        assert graph["node_features"].shape[1] == 4
        assert graph["edges"].shape[0] == 2


class TestMapperAutoencoder:
    def test_encode_decode(self):
        from pynerve.mapper._learnable_mapper_models import MapperAutoencoder

        ae = MapperAutoencoder(
            input_dim=3,
            latent_dim=16,
            mapper_hidden_dims=[8],
            decoder_hidden_dims=[8, 16],
        )
        data = torch.randn(2, 5, 3)
        reconstruction, latent = ae(data)
        assert reconstruction.shape == (2, 3)
        assert latent.shape == (2, 16)
        assert torch.isfinite(reconstruction).all()

    def test_gradient_flow(self):
        from pynerve.mapper._learnable_mapper_models import MapperAutoencoder

        ae = MapperAutoencoder(
            input_dim=2, latent_dim=8, mapper_hidden_dims=[8], decoder_hidden_dims=[8]
        )
        data = torch.randn(1, 5, 2, requires_grad=True)
        reconstruction, _ = ae(data)
        loss = ((reconstruction - data[:, 0, :]) ** 2).sum()
        loss.backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()


# mapper / mapper_gnn


class TestMapperGraphConv:
    def test_forward_basic(self):
        from pynerve.mapper.mapper_gnn import MapperGraphConv

        conv = MapperGraphConv(in_dim=8, out_dim=8)
        feats = torch.randn(5, 8)
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = conv(feats, edges)
        assert out.shape == (5, 8)
        assert torch.isfinite(out).all()

    def test_with_edge_weights(self):
        from pynerve.mapper.mapper_gnn import MapperGraphConv

        conv = MapperGraphConv(in_dim=4, out_dim=4)
        feats = torch.randn(4, 4)
        edges = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        weights = torch.tensor([0.5, 1.0])
        out = conv(feats, edges, weights)
        assert out.shape == (4, 4)

    def test_no_edges(self):
        from pynerve.mapper.mapper_gnn import MapperGraphConv

        conv = MapperGraphConv(in_dim=8, out_dim=8)
        feats = torch.randn(3, 8)
        edges = torch.zeros((2, 0), dtype=torch.long)
        out = conv(feats, edges)
        assert out.shape == (3, 8)

    def test_gradient_flow(self):
        from pynerve.mapper.mapper_gnn import MapperGraphConv

        conv = MapperGraphConv(in_dim=4, out_dim=4)
        feats = torch.randn(4, 4, requires_grad=True)
        edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = conv(feats, edges)
        loss = out.sum()
        loss.backward()
        assert feats.grad is not None
        assert torch.isfinite(feats.grad).all()


class TestMapperNodeEncoder:
    def test_forward_basic(self):
        from pynerve.mapper.mapper_gnn import MapperNodeEncoder

        enc = MapperNodeEncoder(input_dim=8, hidden_dim=16, num_layers=2)
        feats = torch.randn(6, 8)
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = enc(feats, edges)
        assert out.shape == (6, 16)
        assert torch.isfinite(out).all()

    def test_with_lens_values(self):
        from pynerve.mapper.mapper_gnn import MapperNodeEncoder

        enc = MapperNodeEncoder(input_dim=8, hidden_dim=16, num_layers=2, use_lens_positional=True)
        feats = torch.randn(4, 8)
        edges = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        lens = torch.randn(4, 2)
        out = enc(feats, edges, lens_values=lens)
        assert out.shape == (4, 16)

    def test_gradient_flow(self):
        from pynerve.mapper.mapper_gnn import MapperNodeEncoder

        enc = MapperNodeEncoder(input_dim=4, hidden_dim=8, num_layers=1)
        feats = torch.randn(5, 4, requires_grad=True)
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = enc(feats, edges)
        loss = out.sum()
        loss.backward()
        assert feats.grad is not None
        assert torch.isfinite(feats.grad).all()


class TestMapperPoolingLayer:
    def test_forward_basic(self):
        from pynerve.mapper.mapper_gnn import MapperPoolingLayer

        pool = MapperPoolingLayer(in_dim=8, out_dim=8, pool_ratio=0.5)
        feats = torch.randn(10, 8)
        edges = torch.tensor([[0, 1, 8], [1, 8, 9]], dtype=torch.long)
        out_feats, out_edges = pool(feats, edges)
        assert out_feats.dim() == 2
        assert out_feats.shape[1] == 8
        assert out_edges.dim() == 2
        assert out_edges.shape[0] == 2

    def test_empty_edges(self):
        from pynerve.mapper.mapper_gnn import MapperPoolingLayer

        pool = MapperPoolingLayer(in_dim=8, out_dim=8, pool_ratio=0.5)
        feats = torch.randn(6, 8)
        out_feats, out_edges = pool(feats, torch.zeros((2, 0), dtype=torch.long))
        assert out_feats.dim() == 2
        assert out_edges.numel() == 0

    def test_gradient_flow(self):
        from pynerve.mapper.mapper_gnn import MapperPoolingLayer

        pool = MapperPoolingLayer(in_dim=4, out_dim=4, pool_ratio=0.5)
        feats = torch.randn(6, 4, requires_grad=True)
        edges = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        out_feats, _ = pool(feats, edges)
        loss = out_feats.sum()
        loss.backward()
        assert feats.grad is not None
        assert torch.isfinite(feats.grad).all()


class TestHierarchicalMapperPooling:
    def test_forward(self):
        from pynerve.mapper.mapper_gnn import HierarchicalMapperPooling

        pool = HierarchicalMapperPooling(node_dim=8, num_levels=2, pool_ratio=0.5)
        feats = torch.randn(10, 8)
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        pooled = pool(feats, edges)
        assert len(pooled) == 3  # original + 2 levels
        assert all(p.shape[1] == 8 for p in pooled)

    def test_gradient_flow(self):
        from pynerve.mapper.mapper_gnn import HierarchicalMapperPooling

        pool = HierarchicalMapperPooling(node_dim=4, num_levels=1, pool_ratio=0.5)
        feats = torch.randn(6, 4, requires_grad=True)
        edges = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        pooled = pool(feats, edges)
        loss = sum(p.sum() for p in pooled)
        loss.backward()
        assert feats.grad is not None
        assert torch.isfinite(feats.grad).all()


class TestTopologyAwareReadout:
    def test_forward_shape(self):
        from pynerve.mapper.mapper_gnn import TopologyAwareReadout

        readout = TopologyAwareReadout(node_dim=8, output_dim=64)
        feats = torch.randn(5, 8)
        out = readout(feats)
        assert out.shape == (64,)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        from pynerve.mapper.mapper_gnn import TopologyAwareReadout

        readout = TopologyAwareReadout(node_dim=4, output_dim=12)
        feats = torch.randn(5, 4, requires_grad=True)
        out = readout(feats)
        loss = out.sum()
        loss.backward()
        assert feats.grad is not None


class TestMapperGNNClassifier:
    def test_forward_flat_graph(self):
        from pynerve.mapper.mapper_gnn import MapperGNNClassifier

        clf = MapperGNNClassifier(
            node_dim=8, hidden_dim=16, num_classes=3, num_gnn_layers=2, use_hierarchical=False
        )
        feats = torch.randn(6, 8)
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = clf(feats, edges)
        assert out.shape == (3,)
        assert torch.isfinite(out).all()

    def test_forward_hierarchical(self):
        from pynerve.mapper.mapper_gnn import MapperGNNClassifier

        clf = MapperGNNClassifier(
            node_dim=8, hidden_dim=16, num_classes=3, num_gnn_layers=2, use_hierarchical=True
        )
        feats = torch.randn(10, 8)
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = clf(feats, edges)
        assert out.shape == (3,)

    def test_with_batch_vector(self):
        from pynerve.mapper.mapper_gnn import MapperGNNClassifier

        clf = MapperGNNClassifier(node_dim=4, hidden_dim=8, num_classes=2, use_hierarchical=False)
        feats = torch.randn(8, 4)
        edges = torch.tensor([[0, 1, 4, 5], [1, 2, 5, 6]], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        out = clf(feats, edges, batch_vector=batch)
        assert out.shape == (2, 2)  # 2 graphs x 2 classes

    def test_gradient_flow(self):
        from pynerve.mapper.mapper_gnn import MapperGNNClassifier

        clf = MapperGNNClassifier(
            node_dim=4, hidden_dim=8, num_classes=2, num_gnn_layers=1, use_hierarchical=False
        )
        feats = torch.randn(5, 4, requires_grad=True)
        edges = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        out = clf(feats, edges)
        loss = out.sum()
        loss.backward()
        assert feats.grad is not None
        assert torch.isfinite(feats.grad).all()
