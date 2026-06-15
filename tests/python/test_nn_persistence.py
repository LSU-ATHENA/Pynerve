"""Correctness tests for pynerve.nn subpackage."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from pynerve.exceptions import ValidationError  # noqa: E402

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


# persistent_homology


class TestPersistentHomology:
    """Tests for PersistentHomology module."""

    @pytest.fixture(scope="class")
    def ph(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        return PersistentHomology(max_dim=1, max_radius=float("inf"))

    @_core_skip
    def test_forward_returns_list_of_tensors(self, ph, point_cloud_batch):
        diagrams = ph(point_cloud_batch)
        assert isinstance(diagrams, list)
        assert len(diagrams) == 2  # dims 0 and 1

    @_core_skip
    def test_each_tensor_is_3d(self, ph, point_cloud_batch):
        diagrams = ph(point_cloud_batch)
        for d in diagrams:
            assert d.dim() == 3
            assert d.shape[0] == BATCH
            assert d.shape[2] == 2  # birth, death

    @_core_skip
    def test_max_dim_zero(self, point_cloud_batch):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph0 = PersistentHomology(max_dim=0, max_radius=float("inf"))
        diagrams = ph0(point_cloud_batch)
        assert len(diagrams) == 1

    @_core_skip
    @_core_skip
    def test_cohomology_reduction(self, point_cloud_batch):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=1, reduction="cohomology")
        diagrams = ph(point_cloud_batch)
        assert len(diagrams) == 2

    @_core_skip
    def test_finite_max_radius(self, point_cloud_batch):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=0, max_radius=1.0)
        diagrams = ph(point_cloud_batch)
        assert len(diagrams) == 1

    @_core_skip
    def test_gradient_flows(self, point_cloud_batch):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=0, max_radius=float("inf"))
        pts = point_cloud_batch.clone().requires_grad_(True)
        diagrams = ph(pts)
        dg = diagrams[0]
        loss = dg.sum()
        loss.backward()
        assert pts.grad is not None
        assert (pts.grad.abs().sum() > 0).item()

    def test_invalid_max_dim_raises(self):
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.nn.persistent_homology import PersistentHomology

        with pytest.raises(InvalidArgumentError):
            PersistentHomology(max_dim=-1)

    def test_invalid_max_radius_raises(self):
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.nn.persistent_homology import PersistentHomology

        with pytest.raises(InvalidArgumentError):
            PersistentHomology(max_radius=0.0)
        with pytest.raises(InvalidArgumentError):
            PersistentHomology(max_radius=-1.0)

    def test_invalid_reduction_raises(self):
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.nn.persistent_homology import PersistentHomology

        with pytest.raises(InvalidArgumentError):
            PersistentHomology(reduction="unknown")

    def test_is_nn_module(self, ph):
        assert isinstance(ph, nn.Module)

    @_core_skip
    def test_non_3d_input_raises(self, ph):

        with pytest.raises(ValidationError):
            ph(torch.randn(5, 3))

    @_core_skip
    def test_empty_batch_raises(self, ph):

        with pytest.raises(ValidationError):
            ph(torch.zeros(0, 8, 3))


class TestComputePersistenceDiagrams:
    """Tests for compute_persistence_diagrams free function."""

    @_core_skip
    def test_returns_list_of_tensors(self, point_cloud_batch):
        from pynerve.nn.persistent_homology import compute_persistence_diagrams

        diagrams = compute_persistence_diagrams(point_cloud_batch, max_dim=1)
        assert isinstance(diagrams, list)
        assert len(diagrams) == 2

    @_core_skip
    def test_each_tensor_has_correct_shape(self, point_cloud_batch):
        from pynerve.nn.persistent_homology import compute_persistence_diagrams

        diagrams = compute_persistence_diagrams(point_cloud_batch, max_dim=1)
        for d in diagrams:
            assert d.dim() == 3
            assert d.shape[0] == BATCH
            assert d.shape[2] == 2


# sparse_ph (SparsePH, WindowedPH, TopologyAttention, farthest_point_sampling)


class TestSparsePH:
    """Tests for SparsePH module."""

    def test_forward_with_reduction_mean(self, point_cloud_batch):
        from pynerve.nn.sparse_ph import SparsePH

        sph = SparsePH(max_dim=0, landmark_ratio=0.5, reduction="mean")
        out = sph(point_cloud_batch)
        assert isinstance(out, Tensor)
        assert out.dim() == 2

    def test_forward_with_reduction_none(self, point_cloud_batch):
        from pynerve.nn.sparse_ph import SparsePH

        sph = SparsePH(max_dim=0, landmark_ratio=0.5, reduction="none")
        out = sph(point_cloud_batch)
        assert isinstance(out, list)

    def test_invalid_landmark_ratio_raises(self):
        from pynerve.nn.sparse_ph import SparsePH

        with pytest.raises(ValueError):
            SparsePH(landmark_ratio=0.0)
        with pytest.raises(ValueError):
            SparsePH(landmark_ratio=1.5)

    def test_invalid_reduction_raises(self):
        from pynerve.nn.sparse_ph import SparsePH

        with pytest.raises(ValueError):
            SparsePH(reduction="invalid")

    def test_is_nn_module(self):
        from pynerve.nn.sparse_ph import SparsePH

        assert isinstance(SparsePH(), nn.Module)


class TestWindowedPH:
    """Tests for WindowedPH module."""

    def test_forward_with_concat(self, point_cloud_batch):
        from pynerve.nn.sparse_ph import WindowedPH

        wph = WindowedPH(window_size=4, stride=2, max_dim=0, overlap_handling="concat")
        out = wph(point_cloud_batch)
        assert isinstance(out, Tensor)
        assert out.dim() == 2

    def test_forward_with_mean(self, point_cloud_batch):
        from pynerve.nn.sparse_ph import WindowedPH

        wph = WindowedPH(window_size=4, stride=2, max_dim=0, overlap_handling="mean")
        out = wph(point_cloud_batch)
        assert isinstance(out, Tensor)
        assert out.dim() == 2

    def test_forward_with_max(self, point_cloud_batch):
        from pynerve.nn.sparse_ph import WindowedPH

        wph = WindowedPH(window_size=4, stride=2, max_dim=0, overlap_handling="max")
        out = wph(point_cloud_batch)
        assert isinstance(out, Tensor)
        assert out.dim() == 2

    def test_window_larger_than_sequence_returns_empty(self, point_cloud_batch):
        from pynerve.nn.sparse_ph import WindowedPH

        wph = WindowedPH(window_size=100, stride=50, max_dim=0)
        out = wph(point_cloud_batch)
        assert out.shape[0] == BATCH
        assert out.shape[1] == 0

    def test_invalid_overlap_handling_raises(self):
        from pynerve.nn.sparse_ph import WindowedPH

        with pytest.raises(ValueError):
            WindowedPH(overlap_handling="bad")

    def test_is_nn_module(self):
        from pynerve.nn.sparse_ph import WindowedPH

        assert isinstance(WindowedPH(), nn.Module)
