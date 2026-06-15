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


class TestTopologyAttention:
    """Tests for TopologyAttention module."""

    def test_forward_shape(self):
        from pynerve.nn.sparse_ph import TopologyAttention

        ta = TopologyAttention(n_heads=4, dim=64, n_clusters=8, dropout=0.0)
        x = torch.randn(BATCH, 10, 64)
        out = ta(x)
        assert out.shape == (BATCH, 10, 64)

    def test_with_mask(self):
        from pynerve.nn.sparse_ph import TopologyAttention

        ta = TopologyAttention(n_heads=4, dim=64, n_clusters=8, dropout=0.0)
        x = torch.randn(BATCH, 10, 64)
        mask = torch.ones(BATCH, 1, 1, 10)
        mask[0, :, :, -1] = 0
        out = ta(x, mask=mask)
        assert out.shape == (BATCH, 10, 64)

    def test_gradient_flow(self):
        from pynerve.nn.sparse_ph import TopologyAttention

        ta = TopologyAttention(n_heads=4, dim=64, n_clusters=8, dropout=0.0)
        x = torch.randn(BATCH, 10, 64, requires_grad=True)
        out = ta(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert (x.grad.abs().sum() > 0).item()

    def test_is_nn_module(self):
        from pynerve.nn.sparse_ph import TopologyAttention

        assert isinstance(TopologyAttention(), nn.Module)

    def test_invalid_dim_raises(self):
        from pynerve.nn.sparse_ph import TopologyAttention

        with pytest.raises(ValueError):
            TopologyAttention(dim=65, n_heads=4)  # not divisible

    def test_mismatched_embedding_dim_raises(self):
        from pynerve.nn.sparse_ph import TopologyAttention

        ta = TopologyAttention(dim=64, n_heads=4)
        x = torch.randn(BATCH, 10, 32)  # wrong dim
        with pytest.raises(ValueError):
            ta(x)


class TestFarthestPointSampling:
    """Tests for farthest_point_sampling."""

    def test_returns_landmarks_and_indices(self, point_cloud_2d):
        from pynerve.nn.sparse_ph import farthest_point_sampling

        landmarks, indices = farthest_point_sampling(point_cloud_2d, 4)
        assert landmarks.shape == (4, DIM)
        assert indices.shape == (4,)
        assert indices.dtype == torch.long

    def test_n_samples_exceeds_points(self, point_cloud_2d):
        from pynerve.nn.sparse_ph import farthest_point_sampling

        landmarks, indices = farthest_point_sampling(point_cloud_2d, 100)
        assert landmarks.shape[0] == N_POINTS
        assert indices.shape[0] == N_POINTS

    def test_zero_samples_returns_empty(self, point_cloud_2d):
        from pynerve.nn.sparse_ph import farthest_point_sampling

        landmarks, indices = farthest_point_sampling(point_cloud_2d, 0)
        assert landmarks.shape[0] == 0
        assert indices.shape[0] == 0

    def test_all_indices_unique(self, point_cloud_2d):
        from pynerve.nn.sparse_ph import farthest_point_sampling

        _, indices = farthest_point_sampling(point_cloud_2d, 5)
        assert len(set(indices.tolist())) == 5
