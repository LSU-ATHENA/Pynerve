"""Correctness tests for pynerve.nn subpackage."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402

# Helpers


def _has_torch_backend() -> bool:
    try:
        from pynerve.torch._persistence_validators import _torch_backend  # noqa: PLC0415

        return _torch_backend() is not None
    except ImportError:
        return False


_torch_skip = pytest.mark.skipif(
    not _has_torch_backend(), reason="torch C++ backend (pynerve_torch_internal) required"
)


def _has_core_backend() -> bool:
    """Check if the C++ persistence backend (pynerve_internal) is importable."""
    try:
        import pynerve_internal  # noqa: F401

        return True
    except ImportError:
        return False


def _has_internal_backend() -> bool:
    """Check if pynerve_internal (witness persistence) is importable."""
    try:
        import pynerve_internal  # noqa: F401

        return True
    except ImportError:
        return False


_core_skip = pytest.mark.skipif(
    not _has_core_backend(), reason="C++ persistence backend not available"
)
_internal_skip = pytest.mark.skipif(
    not _has_internal_backend(), reason="pynerve_internal C++ extension not available"
)

BATCH = 2
N_POINTS = 8
DIM = 3
SEED = 42


@pytest.fixture(scope="module")
def rng():
    torch.manual_seed(SEED)
    return torch.Generator().manual_seed(SEED)


@pytest.fixture(scope="module")
def point_cloud_2d() -> Tensor:
    """Single point cloud (N, D)."""
    torch.manual_seed(SEED)
    return torch.rand(N_POINTS, DIM)


@pytest.fixture(scope="module")
def point_cloud_batch() -> Tensor:
    """Batch of point clouds (B, N, D)."""
    torch.manual_seed(SEED)
    return torch.rand(BATCH, N_POINTS, DIM)


@pytest.fixture(scope="module")
def diagram_batch() -> Tensor:
    """Fake persistence diagram batch (B, pairs, 3) with birth<death."""
    torch.manual_seed(SEED)
    births = torch.rand(BATCH, 4, 1)
    deaths = births + torch.rand(BATCH, 4, 1) + 0.1
    dims = torch.randint(0, 2, (BATCH, 4, 1)).float()
    return torch.cat([births, deaths, dims], dim=-1)


@pytest.fixture(scope="module")
def simple_diagram() -> Tensor:
    """A simple (batch, 2, 3) diagram for attention / pooling tests."""
    return torch.tensor(
        [
            [[0.0, 1.0, 0.0], [0.2, 0.8, 1.0]],
            [[0.1, 0.9, 0.0], [0.3, 0.7, 1.0]],
        ],
        dtype=torch.float32,
    )


# diagram_conv (DiagramConv1D, DiagramDeepSet, DiagramMultiHeadAttention,
#               DiagramTransformerBlock, DiagramPooling, DiagramConvNet)


class TestDiagramConv1D:
    """Tests for DiagramConv1D."""

    @pytest.fixture(scope="class")
    def conv(self):
        from pynerve.nn.diagram_conv import DiagramConv1D

        return DiagramConv1D(in_channels=4, out_channels=16, kernel_size=3)

    def test_forward_shape(self, conv, diagram_batch):
        features = torch.randn(BATCH, 4, 4)
        out = conv(diagram_batch, features)
        assert out.shape[:2] == (BATCH, 16)

    def test_no_features_with_in_channels_zero(self, diagram_batch):
        from pynerve.nn.diagram_conv import DiagramConv1D

        conv = DiagramConv1D(in_channels=0, out_channels=8, kernel_size=3)
        out = conv(diagram_batch)
        assert out.shape[:2] == (BATCH, 8)

    def test_empty_diagram_returns_zeros(self, conv):
        empty = torch.zeros(BATCH, 0, 3)
        out = conv(empty, torch.zeros(BATCH, 0, 4))
        assert out.shape == (BATCH, 16, 0)

    def test_no_persistence_weighting(self, diagram_batch):
        from pynerve.nn.diagram_conv import DiagramConv1D

        conv = DiagramConv1D(
            in_channels=4, out_channels=16, kernel_size=3, use_persistence_weighting=False
        )
        features = torch.randn(BATCH, 4, 4)
        out = conv(diagram_batch, features)
        assert out.shape[:2] == (BATCH, 16)

    def test_is_nn_module(self, conv):
        assert isinstance(conv, nn.Module)

    def test_gradient_flow(self, conv, diagram_batch):
        features = torch.randn(BATCH, 4, 4, requires_grad=True)
        out = conv(diagram_batch, features)
        loss = out.sum()
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0


class TestDiagramDeepSet:
    """Tests for DiagramDeepSet."""

    @pytest.fixture(scope="class")
    def deepset(self):
        from pynerve.nn.diagram_conv import DiagramDeepSet

        return DiagramDeepSet(
            in_channels=2,
            hidden_channels=[16, 32],
            out_channels=10,
            pooling="persistence_weighted",
        )

    def test_forward_shape(self, deepset, diagram_batch):
        features = torch.randn(BATCH, 4, 2)
        out = deepset(diagram_batch, features)
        assert out.shape == (BATCH, 10)

    def test_sum_pooling(self, diagram_batch):
        from pynerve.nn.diagram_conv import DiagramDeepSet

        ds = DiagramDeepSet(in_channels=2, hidden_channels=[8], out_channels=6, pooling="sum")
        features = torch.randn(BATCH, 4, 2)
        out = ds(diagram_batch, features)
        assert out.shape == (BATCH, 6)

    def test_mean_pooling(self, diagram_batch):
        from pynerve.nn.diagram_conv import DiagramDeepSet

        ds = DiagramDeepSet(in_channels=2, hidden_channels=[8], out_channels=6, pooling="mean")
        features = torch.randn(BATCH, 4, 2)
        out = ds(diagram_batch, features)
        assert out.shape == (BATCH, 6)

    def test_max_pooling(self, diagram_batch):
        from pynerve.nn.diagram_conv import DiagramDeepSet

        ds = DiagramDeepSet(in_channels=2, hidden_channels=[8], out_channels=6, pooling="max")
        features = torch.randn(BATCH, 4, 2)
        out = ds(diagram_batch, features)
        assert out.shape == (BATCH, 6)

    def test_empty_diagram_returns_zeros(self, deepset):
        empty = torch.zeros(BATCH, 0, 3)
        out = deepset(empty)
        assert out.shape == (BATCH, 10)
        assert (out == 0).all()

    def test_gradient_flow(self, deepset, diagram_batch):
        features = torch.randn(BATCH, 4, 2, requires_grad=True)
        out = deepset(diagram_batch, features)
        loss = out.sum()
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_is_nn_module(self, deepset):
        assert isinstance(deepset, nn.Module)

    def test_invalid_pooling_raises(self):
        from pynerve.nn.diagram_conv import DiagramDeepSet

        with pytest.raises(ValueError):
            DiagramDeepSet(in_channels=2, hidden_channels=[8], out_channels=6, pooling="bad")

    def test_empty_hidden_channels_raises(self):
        from pynerve.nn.diagram_conv import DiagramDeepSet

        with pytest.raises(ValueError):
            DiagramDeepSet(in_channels=2, hidden_channels=[], out_channels=6)


class TestDiagramMultiHeadAttention:
    """Tests for DiagramMultiHeadAttention."""

    @pytest.fixture(scope="class")
    def attn(self):
        from pynerve.nn.diagram_conv import DiagramMultiHeadAttention

        return DiagramMultiHeadAttention(d_model=64, num_heads=4, dropout=0.0)

    def test_forward_shape(self, attn, diagram_batch):
        features = torch.randn(BATCH, 4, 64)
        out = attn(diagram_batch, features)
        assert out.shape == (BATCH, 4, 64)

    def test_with_mask(self, attn, diagram_batch):
        features = torch.randn(BATCH, 4, 64)
        mask = torch.ones(BATCH, 4)
        mask[0, -1] = 0
        out = attn(diagram_batch, features, mask)
        assert out.shape == (BATCH, 4, 64)

    def test_without_positional_encoding(self, diagram_batch):
        from pynerve.nn.diagram_conv import DiagramMultiHeadAttention

        attn = DiagramMultiHeadAttention(
            d_model=64, num_heads=4, dropout=0.0, use_birth_death_positional=False
        )
        features = torch.randn(BATCH, 4, 64)
        out = attn(diagram_batch, features)
        assert out.shape == (BATCH, 4, 64)

    def test_empty_diagram_returns_zeros(self, attn):
        empty = torch.zeros(BATCH, 0, 3)
        features = torch.zeros(BATCH, 0, 64)
        out = attn(empty, features)
        assert out.shape == (BATCH, 0, 64)

    def test_gradient_flow(self, attn, diagram_batch):
        features = torch.randn(BATCH, 4, 64, requires_grad=True)
        out = attn(diagram_batch, features)
        loss = out.sum()
        loss.backward()
        assert features.grad is not None
        assert (features.grad.abs().sum() > 0).item()

    def test_is_nn_module(self, attn):
        assert isinstance(attn, nn.Module)


class TestDiagramTransformerBlock:
    """Tests for DiagramTransformerBlock."""

    def test_forward_shape(self, diagram_batch):
        from pynerve.nn.diagram_conv import DiagramTransformerBlock

        block = DiagramTransformerBlock(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        features = torch.randn(BATCH, 4, 64)
        out = block(diagram_batch, features)
        assert out.shape == (BATCH, 4, 64)

    def test_gradient_flow(self, diagram_batch):
        from pynerve.nn.diagram_conv import DiagramTransformerBlock

        block = DiagramTransformerBlock(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        features = torch.randn(BATCH, 4, 64, requires_grad=True)
        out = block(diagram_batch, features)
        loss = out.sum()
        loss.backward()
        assert features.grad is not None
        assert (features.grad.abs().sum() > 0).item()

    def test_is_nn_module(self):
        from pynerve.nn.diagram_conv import DiagramTransformerBlock

        assert isinstance(DiagramTransformerBlock(d_model=64, num_heads=4, d_ff=128), nn.Module)


class TestDiagramPooling:
    """Tests for DiagramPooling."""

    def test_attention_pooling_shape(self, diagram_batch):
        from pynerve.nn.diagram_conv import DiagramPooling

        pool = DiagramPooling(
            in_channels=16, out_channels=32, num_prototypes=8, pooling_type="attention"
        )
        features = torch.randn(BATCH, 4, 16)
        out = pool(diagram_batch, features)
        assert out.shape == (BATCH, 8, 32)

    def test_persistence_clustering_shape(self, diagram_batch):
        from pynerve.nn.diagram_conv import DiagramPooling

        pool = DiagramPooling(
            in_channels=16, out_channels=32, num_prototypes=8, pooling_type="persistence_clustering"
        )
        features = torch.randn(BATCH, 4, 16)
        out = pool(diagram_batch, features)
        assert out.shape == (BATCH, 8, 32)

    def test_empty_diagram_returns_zeros(self):
        from pynerve.nn.diagram_conv import DiagramPooling

        pool = DiagramPooling(in_channels=16, out_channels=32, num_prototypes=8)
        empty = torch.zeros(BATCH, 0, 3)
        features = torch.zeros(BATCH, 0, 16)
        out = pool(empty, features)
        assert out.shape == (BATCH, 8, 32)
        assert (out == 0).all()

    def test_gradient_flow(self, diagram_batch):
        from pynerve.nn.diagram_conv import DiagramPooling

        pool = DiagramPooling(in_channels=16, out_channels=32, num_prototypes=8)
        features = torch.randn(BATCH, 4, 16, requires_grad=True)
        out = pool(diagram_batch, features)
        loss = out.sum()
        loss.backward()
        assert features.grad is not None
        assert (features.grad.abs().sum() > 0).item()

    def test_is_nn_module(self):
        from pynerve.nn.diagram_conv import DiagramPooling

        assert isinstance(
            DiagramPooling(in_channels=16, out_channels=32, num_prototypes=8), nn.Module
        )


class TestDiagramConvNet:
    """Tests for DiagramConvNet."""

    def test_forward_shape(self, diagram_batch):
        from pynerve.nn.diagram_conv import DiagramConvNet

        net = DiagramConvNet(
            in_channels=3,
            hidden_channels=[32, 64],
            out_dim=10,
            num_prototypes=8,
            pooling="persistence_weighted",
        )
        out = net(diagram_batch)
        assert out.shape == (BATCH, 10)

    def test_gradient_flow(self, diagram_batch):
        from pynerve.nn.diagram_conv import DiagramConvNet

        net = DiagramConvNet(in_channels=3, hidden_channels=[32, 64], out_dim=10, num_prototypes=4)
        # diagram_batch has requires_grad=False by default, so we test params
        out = net(diagram_batch)
        loss = out.sum()
        loss.backward()
        params = list(net.parameters())
        grad_norms = [p.grad.norm().item() for p in params if p.grad is not None]
        assert len(grad_norms) > 0
        assert sum(grad_norms) > 0

    def test_is_nn_module(self):
        from pynerve.nn.diagram_conv import DiagramConvNet

        assert isinstance(DiagramConvNet(in_channels=3, hidden_channels=[32], out_dim=5), nn.Module)

    def test_invalid_pooling_raises(self):
        from pynerve.nn.diagram_conv import DiagramConvNet

        with pytest.raises(ValueError):
            DiagramConvNet(in_channels=3, hidden_channels=[32], out_dim=5, pooling="bad")
