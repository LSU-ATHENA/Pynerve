"""Comprehensive tests for pynerve.torch subpackage.

Covers the public API surface: lazy imports, PersistenceDiagram, diagram
distance functions, data utilities, nn layers, preprocessing, statistics,
vectorization, kernels, mapper, sklearn transformers, tensorboard logging,
and visualisation helpers.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from pynerve.exceptions import ValidationError  # noqa: E402


def _torch_backend_available() -> bool:
    """Check if the PyTorch C++ extension (pynerve_torch_internal) is present."""
    try:
        import nerve_torch_internal  # noqa: F401

        return True
    except ImportError:
        try:
            import pynerve_torch_internal  # noqa: F401

            return True
        except ImportError:
            return False


_torch_backend = pytest.mark.skipif(
    not _torch_backend_available(),
    reason="pynerve_torch_internal not available",
)


def _core_backend_available() -> bool:
    """Check if the core C++ extension (pynerve_internal) is present."""
    try:
        import pynerve_internal  # noqa: F401

        return True
    except ImportError:
        return False


_core_backend = pytest.mark.skipif(
    not _core_backend_available(),
    reason="pynerve_internal not available",
)


def _networkx_available() -> bool:
    try:
        import networkx  # noqa: F401

        return True
    except ImportError:
        return False


def _make_diagram_tensor(
    batch: int = 2,
    pairs: int = 4,
    seed: int = 42,
) -> torch.Tensor:
    """Create a valid persistence diagram tensor (birth, death, dim) with birth < death."""
    torch.manual_seed(seed)
    births = torch.rand(batch, pairs, 1)
    deaths = births + torch.rand(batch, pairs, 1) + 0.1
    dims = torch.randint(0, 2, (batch, pairs, 1)).float()
    return torch.cat([births, deaths, dims], dim=-1)


def _make_2d_diagram(pairs: int = 3, seed: int = 42) -> torch.Tensor:
    """Create a 2D (unbatched) diagram tensor with (birth, death) columns."""
    torch.manual_seed(seed)
    births = torch.rand(pairs, 1)
    deaths = births + torch.rand(pairs, 1) + 0.1
    return torch.cat([births, deaths], dim=-1)


def _make_point_cloud(
    n_points: int = 8,
    dim: int = 3,
    batch: int = 1,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)
    if batch == 1:
        return torch.rand(n_points, dim)
    return torch.rand(batch, n_points, dim)


# Lazy import mechanism


# helpers


# nn_layers


class TestPersistenceLayer:
    """PersistenceLayer construction and forward pass."""

    def test_construct_default(self) -> None:
        from pynerve.torch.nn_layers import PersistenceLayer

        layer = PersistenceLayer()
        assert layer.max_dim == 1
        assert layer.max_radius == float("inf")
        assert layer.metric == "euclidean"

    def test_construct_custom(self) -> None:
        from pynerve.torch.nn_layers import PersistenceLayer

        layer = PersistenceLayer(max_dim=2, max_radius=1.0, metric="manhattan", return_raw=True)
        assert layer.max_dim == 2
        assert layer.max_radius == 1.0
        assert layer.return_raw is True

    def test_is_nn_module(self) -> None:
        from pynerve.torch.nn_layers import PersistenceLayer

        assert isinstance(PersistenceLayer(), torch.nn.Module)

    def test_forward_invalid_dim_raises(self) -> None:
        from pynerve.torch.nn_layers import PersistenceLayer

        layer = PersistenceLayer()
        with pytest.raises((TypeError, ValueError, ValidationError)):
            layer(torch.randn(8))

    def test_forward_nan_raises(self) -> None:
        from pynerve.torch.nn_layers import PersistenceLayer

        layer = PersistenceLayer()
        x = torch.tensor([[float("nan"), 0.0]], dtype=torch.float32)
        with pytest.raises((TypeError, ValueError, ValidationError)):
            layer(x)

    @_torch_backend
    def test_forward_happy_path_2d(self) -> None:
        from pynerve.torch.nn_layers import PersistenceLayer

        layer = PersistenceLayer(max_dim=0)
        x = torch.rand(5, 2)
        out = layer(x)
        from pynerve.torch._diagram import PersistenceDiagram

        assert isinstance(out, PersistenceDiagram)

    @_torch_backend
    def test_forward_return_raw(self) -> None:
        from pynerve.torch.nn_layers import PersistenceLayer

        layer = PersistenceLayer(max_dim=0, return_raw=True)
        x = torch.rand(5, 2)
        out = layer(x)
        assert isinstance(out, torch.Tensor)

    @_torch_backend
    def test_forward_batched(self) -> None:
        from pynerve.torch.nn_layers import PersistenceLayer

        layer = PersistenceLayer(max_dim=0)
        x = torch.rand(2, 5, 2)
        out = layer(x)
        from pynerve.torch._diagram import PersistenceDiagram

        assert isinstance(out, PersistenceDiagram)
        assert out.batch_size == 2

    def test_forward_3d_no_backend_falls_back(self) -> None:
        from pynerve.torch.nn_layers import PersistenceLayer

        layer = PersistenceLayer(max_dim=0)
        x = torch.rand(1, 3, 2)
        result = layer(x)
        assert result is not None

    def test_forward_empty_points_raises(self) -> None:
        from pynerve.torch.nn_layers import PersistenceLayer

        layer = PersistenceLayer()
        with pytest.raises((ValueError, ValidationError)):
            layer(torch.rand(0, 2))

    def test_forward_empty_coords_raises(self) -> None:
        from pynerve.torch.nn_layers import PersistenceLayer

        layer = PersistenceLayer()
        with pytest.raises(ValueError):
            layer(torch.rand(3, 0))


class TestVectorizationLayer:
    """VectorizationLayer construction and forward pass."""

    def test_construct_default(self) -> None:
        from pynerve.torch.nn_layers import VectorizationLayer

        layer = VectorizationLayer()
        assert layer.method == "landscape"

    def test_construct_custom(self) -> None:
        from pynerve.torch.nn_layers import VectorizationLayer

        layer = VectorizationLayer(method="image", resolution=(5, 5), sigma=0.2)
        assert layer.method == "image"

    def test_is_nn_module(self) -> None:
        from pynerve.torch.nn_layers import VectorizationLayer

        assert isinstance(VectorizationLayer(), torch.nn.Module)

    def test_invalid_method_raises(self) -> None:
        from pynerve.torch.nn_layers import VectorizationLayer

        with pytest.raises(ValueError):
            VectorizationLayer(method="bogus")

    def test_forward_with_persistence_diagram(self) -> None:
        from pynerve.torch import PersistenceDiagram
        from pynerve.torch.nn_layers import VectorizationLayer

        tensor = _make_diagram_tensor(batch=2, pairs=3)
        pd = PersistenceDiagram(tensor)
        layer = VectorizationLayer(method="silhouette", num_samples=10)
        out = layer(pd)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 2

    def test_forward_with_raw_tensor(self) -> None:
        from pynerve.torch.nn_layers import VectorizationLayer

        tensor = _make_diagram_tensor(batch=1, pairs=3).squeeze(0)
        layer = VectorizationLayer(method="landscape", k=1, num_samples=5)
        out = layer(tensor)
        assert isinstance(out, torch.Tensor)

    def test_forward_invalid_shape_raises(self) -> None:
        from pynerve.torch.nn_layers import VectorizationLayer

        layer = VectorizationLayer()
        with pytest.raises(ValueError):
            layer(torch.randn(10))

    def test_forward_batched_with_mask(self) -> None:
        from pynerve.torch import PersistenceDiagram
        from pynerve.torch.nn_layers import VectorizationLayer

        tensor = _make_diagram_tensor(batch=2, pairs=4)
        mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
        pd = PersistenceDiagram(tensor, mask=mask)
        layer = VectorizationLayer(method="silhouette", num_samples=8)
        out = layer(pd)
        assert out.shape[0] == 2


class TestTopologicalFeatureExtractor:
    """TopologicalFeatureExtractor pipeline."""

    def test_construct_default(self) -> None:
        from pynerve.torch.nn_layers import TopologicalFeatureExtractor

        tfe = TopologicalFeatureExtractor()
        assert tfe.persistence.max_dim == 1

    def test_is_nn_module(self) -> None:
        from pynerve.torch.nn_layers import TopologicalFeatureExtractor

        assert isinstance(TopologicalFeatureExtractor(), torch.nn.Module)

    @_torch_backend
    def test_forward_2d(self) -> None:
        from pynerve.torch.nn_layers import TopologicalFeatureExtractor

        tfe = TopologicalFeatureExtractor(
            max_dim=0,
            max_radius=3.0,
            vectorization="silhouette",
            vectorization_params={"num_samples": 8},
        )
        x = torch.rand(5, 2)
        out = tfe(x)
        assert isinstance(out, torch.Tensor)
        assert out.dim() == 2
        assert out.shape[0] == 1

    @_torch_backend
    def test_forward_batched(self) -> None:
        from pynerve.torch.nn_layers import TopologicalFeatureExtractor

        tfe = TopologicalFeatureExtractor(
            max_dim=0,
            max_radius=3.0,
            vectorization="silhouette",
            vectorization_params={"num_samples": 8},
        )
        x = torch.rand(2, 5, 2)
        out = tfe(x)
        assert out.shape[0] == 2


class TestDiagramPooling:
    """DiagramPooling module."""

    def test_construct_default(self) -> None:
        from pynerve.torch.nn_layers import DiagramPooling

        pool = DiagramPooling()
        assert pool.method == "mean"

    def test_construct_custom(self) -> None:
        from pynerve.torch.nn_layers import DiagramPooling

        pool = DiagramPooling(method="sum", dim=-1)
        assert pool.method == "sum"

    def test_invalid_method_raises(self) -> None:
        from pynerve.torch.nn_layers import DiagramPooling

        with pytest.raises(ValueError):
            DiagramPooling(method="invalid")

    def test_mean_pooling(self) -> None:
        from pynerve.torch.nn_layers import DiagramPooling

        pool = DiagramPooling(method="mean")
        x = torch.rand(2, 5, 3)
        out = pool(x)
        assert out.shape == (2, 3)

    def test_max_pooling(self) -> None:
        from pynerve.torch.nn_layers import DiagramPooling

        pool = DiagramPooling(method="max")
        x = torch.rand(2, 5, 3)
        out = pool(x)
        assert out.shape == (2, 3)

    def test_sum_pooling(self) -> None:
        from pynerve.torch.nn_layers import DiagramPooling

        pool = DiagramPooling(method="sum")
        x = torch.rand(2, 5, 3)
        out = pool(x)
        assert out.shape == (2, 3)

    def test_attention_pooling(self) -> None:
        from pynerve.torch.nn_layers import DiagramPooling

        pool = DiagramPooling(method="attention")
        x = torch.rand(2, 5, 3)
        out = pool(x)
        assert out.shape == (2, 3)

    def test_non_float_raises(self) -> None:
        from pynerve.torch.nn_layers import DiagramPooling

        pool = DiagramPooling()
        with pytest.raises((TypeError, ValueError)):
            pool(torch.randint(0, 2, (2, 3, 3)))

    def test_empty_raises(self) -> None:
        from pynerve.torch.nn_layers import DiagramPooling

        pool = DiagramPooling()
        with pytest.raises(ValueError):
            pool(torch.rand(0, 3, 3))

    def test_is_nn_module(self) -> None:
        from pynerve.torch.nn_layers import DiagramPooling

        assert isinstance(DiagramPooling(), torch.nn.Module)


class TestTopologicalAttention:
    """TopologicalAttention module."""

    def test_construct(self) -> None:
        from pynerve.torch.nn_layers import TopologicalAttention

        attn = TopologicalAttention(feature_dim=64, n_heads=4, dropout=0.0)
        assert attn.feature_dim == 64
        assert attn.n_heads == 4

    def test_is_nn_module(self) -> None:
        from pynerve.torch.nn_layers import TopologicalAttention

        assert isinstance(TopologicalAttention(feature_dim=64), torch.nn.Module)

    def test_forward_no_diagrams(self) -> None:
        from pynerve.torch.nn_layers import TopologicalAttention

        attn = TopologicalAttention(feature_dim=8, n_heads=2, dropout=0.0)
        x = torch.rand(2, 5, 8)
        out, weights = attn(x)
        assert out.shape == (2, 5, 8)
        assert weights.shape == (2, 5, 5)

    def test_invalid_feature_dim_raises(self) -> None:
        from pynerve.torch.nn_layers import TopologicalAttention

        with pytest.raises((ValueError, ValidationError)):
            TopologicalAttention(feature_dim=0)

    def test_indivisible_dim_raises(self) -> None:
        from pynerve.torch.nn_layers import TopologicalAttention

        with pytest.raises((ValueError, ValidationError)):
            TopologicalAttention(feature_dim=10, n_heads=3)

    def test_forward_wrong_shape_raises(self) -> None:
        from pynerve.torch.nn_layers import TopologicalAttention

        attn = TopologicalAttention(feature_dim=8, n_heads=2)
        with pytest.raises((ValueError, ValidationError)):
            attn(torch.rand(2, 5, 4))

    def test_gradient_flows(self) -> None:
        from pynerve.torch.nn_layers import TopologicalAttention

        attn = TopologicalAttention(feature_dim=8, n_heads=2)
        x = torch.rand(2, 5, 8, requires_grad=True)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0


class TestPersistenceReadout:
    """PersistenceReadout module."""

    def test_construct(self) -> None:
        from pynerve.torch.nn_layers import PersistenceReadout

        readout = PersistenceReadout(in_features=64, out_features=10)
        assert isinstance(readout, torch.nn.Module)

    def test_forward(self) -> None:
        from pynerve.torch.nn_layers import PersistenceReadout

        readout = PersistenceReadout(in_features=16, out_features=4, hidden_dims=(32,))
        x = torch.rand(2, 16)
        out = readout(x)
        assert out.shape == (2, 4)

    def test_invalid_activation_raises(self) -> None:
        from pynerve.torch.nn_layers import PersistenceReadout

        with pytest.raises((ValueError, ValidationError)):
            PersistenceReadout(in_features=8, out_features=4, activation="sigmoid")

    def test_invalid_dropout_raises(self) -> None:
        from pynerve.torch.nn_layers import PersistenceReadout

        with pytest.raises((ValueError, ValidationError)):
            PersistenceReadout(in_features=8, out_features=4, dropout=1.5)

    def test_gradient_flow(self) -> None:
        from pynerve.torch.nn_layers import PersistenceReadout

        readout = PersistenceReadout(in_features=8, out_features=4, hidden_dims=(16,))
        x = torch.rand(2, 8, requires_grad=True)
        out = readout(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_gelu_activation(self) -> None:
        from pynerve.torch.nn_layers import PersistenceReadout

        readout = PersistenceReadout(
            in_features=8, out_features=4, hidden_dims=(16,), activation="gelu"
        )
        x = torch.rand(2, 8)
        out = readout(x)
        assert out.shape == (2, 4)

    def test_non_float_raises(self) -> None:
        from pynerve.torch.nn_layers import PersistenceReadout

        readout = PersistenceReadout(in_features=8, out_features=4)
        with pytest.raises((TypeError, ValueError, ValidationError)):
            readout(torch.randint(0, 2, (2, 8)))


class TestMakeTopoNetwork:
    """make_topo_network factory."""

    @_torch_backend
    def test_construct_and_forward(self) -> None:
        from pynerve.torch.nn_layers import make_topo_network

        net = make_topo_network(
            input_dim=2,
            hidden_dims=(16,),
            max_dim=0,
            vectorization="silhouette",
            vectorization_params={"num_samples": 8},
            num_classes=3,
            dropout=0.1,
        )
        assert isinstance(net, torch.nn.Module)
        x = torch.rand(2, 5, 2)
        out = net(x)
        assert out.shape == (2, 3)

    def test_invalid_input_dim_raises(self) -> None:
        from pynerve.torch.nn_layers import make_topo_network

        with pytest.raises((ValueError, ValidationError)):
            make_topo_network(input_dim=0)

    def test_forward_wrong_shape_raises(self) -> None:
        from pynerve.torch.nn_layers import make_topo_network

        net = make_topo_network(input_dim=2, hidden_dims=(4,), num_classes=2)
        with pytest.raises((ValueError, ValidationError)):
            net(torch.rand(2, 5, 3))
