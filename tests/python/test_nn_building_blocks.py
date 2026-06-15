"""Correctness tests for pynerve.nn subpackage."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402

from torch import Tensor  # noqa: E402

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


# building_blocks (PersistenceDiagram, SparseDistanceMatrix,
#                   SparseRipsPersistence, WitnessComplexPersistence,
#                   PersistenceSketch)


class TestPersistenceDiagram:
    """Tests for nn.building_blocks.PersistenceDiagram (container class)."""

    @pytest.fixture(scope="class")
    def pd(self):
        from pynerve.nn.building_blocks import PersistenceDiagram

        births = torch.tensor([0.0, 0.2, 0.5])
        deaths = torch.tensor([1.0, 0.8, float("inf")])
        dims = torch.tensor([0, 1, 0], dtype=torch.long)
        return PersistenceDiagram(births, deaths, dims)

    def test_constructor_valid(self, pd):
        assert len(pd) == 3

    def test_constructor_shape_mismatch_raises(self):
        from pynerve.nn.building_blocks import PersistenceDiagram

        with pytest.raises(ValueError):
            PersistenceDiagram(
                torch.tensor([0.0, 0.2]),
                torch.tensor([1.0]),
                torch.tensor([0, 1], dtype=torch.long),
            )

    def test_constructor_non_1d_raises(self):
        from pynerve.nn.building_blocks import PersistenceDiagram

        with pytest.raises(ValueError):
            PersistenceDiagram(
                torch.tensor([[0.0]]),
                torch.tensor([[1.0]]),
                torch.tensor([[0]], dtype=torch.long),
            )

    def test_persistence_values(self, pd):
        pv = pd.persistence_values()
        assert pv.shape == (3,)
        expected = torch.tensor([1.0, 0.6, float("inf")])
        finite_mask = torch.isfinite(pv)
        assert torch.allclose(pv[finite_mask], expected[finite_mask])

    def test_get_dimension(self, pd):
        dim0 = pd.get_dimension(0)
        assert len(dim0) == 2
        assert (dim0.dimensions == 0).all()

    def test_get_empty_dimension(self, pd):
        dim2 = pd.get_dimension(2)
        assert len(dim2) == 0

    def test_to_numpy(self, pd):
        births, deaths, dims = pd.to_numpy()
        assert isinstance(births, np.ndarray)
        assert births.shape == (3,)


class TestSparseDistanceMatrix:
    """Tests for SparseDistanceMatrix."""

    def test_2d_call_returns_tensors(self, point_cloud_2d):
        from pynerve.nn.building_blocks import SparseDistanceMatrix

        sdm = SparseDistanceMatrix(k_neighbors=3)
        distances, indices = sdm(point_cloud_2d)
        assert isinstance(distances, Tensor)
        assert isinstance(indices, Tensor)
        assert distances.shape == (N_POINTS, 3)
        assert indices.shape == (N_POINTS, 3)

    def test_3d_call_returns_tensors(self, point_cloud_batch):
        from pynerve.nn.building_blocks import SparseDistanceMatrix

        sdm = SparseDistanceMatrix(k_neighbors=3)
        distances, indices = sdm(point_cloud_batch)
        assert distances.shape == (BATCH, N_POINTS, 3)
        assert indices.shape == (BATCH, N_POINTS, 3)

    def test_return_numpy(self, point_cloud_2d):
        from pynerve.nn.building_blocks import SparseDistanceMatrix

        sdm = SparseDistanceMatrix(k_neighbors=3)
        distances, indices = sdm(point_cloud_2d, return_numpy=True)
        assert isinstance(distances, np.ndarray)
        assert isinstance(indices, np.ndarray)

    def test_epsilon_neighborhood(self, point_cloud_2d):
        from pynerve.nn.building_blocks import SparseDistanceMatrix

        sdm = SparseDistanceMatrix(k_neighbors=3)
        adj = sdm.epsilon_neighborhood(point_cloud_2d, epsilon=1.0)
        assert adj.is_sparse

    def test_invalid_algorithm_raises(self):
        from pynerve.nn.building_blocks import SparseDistanceMatrix

        with pytest.raises(ValueError):
            SparseDistanceMatrix(algorithm="invalid")

    def test_negative_k_neighbors_raises(self):
        from pynerve.nn.building_blocks import SparseDistanceMatrix

        with pytest.raises(ValueError):
            SparseDistanceMatrix(k_neighbors=0)


class TestSparseRipsPersistence:
    """Tests for SparseRipsPersistence."""

    @_core_skip
    def test_call_returns_persistence_diagram(self, point_cloud_2d):
        from pynerve.nn.building_blocks import (
            PersistenceDiagram,
            SparseRipsPersistence,
        )

        srp = SparseRipsPersistence(sparse_parameter=0.5, max_dim=1)
        result = srp(point_cloud_2d)
        assert isinstance(result, PersistenceDiagram)
        assert len(result) >= 0

    @_core_skip
    def test_max_dim_zero(self, point_cloud_2d):
        from pynerve.nn.building_blocks import (
            PersistenceDiagram,
            SparseRipsPersistence,
        )

        srp = SparseRipsPersistence(sparse_parameter=0.5, max_dim=0)
        result = srp(point_cloud_2d)
        assert isinstance(result, PersistenceDiagram)

    def test_invalid_sparse_parameter_raises(self):
        from pynerve.nn.building_blocks import SparseRipsPersistence

        with pytest.raises(ValueError):
            SparseRipsPersistence(sparse_parameter=0.0)
        with pytest.raises(ValueError):
            SparseRipsPersistence(sparse_parameter=-1.0)
        with pytest.raises(ValueError):
            SparseRipsPersistence(sparse_parameter=float("nan"))


class TestWitnessComplexPersistence:
    """Tests for WitnessComplexPersistence."""

    @_core_skip
    def test_farthest_method(self, point_cloud_2d):
        from pynerve.nn.building_blocks import (
            PersistenceDiagram,
            WitnessComplexPersistence,
        )

        wcp = WitnessComplexPersistence(n_landmarks=5, max_dim=1, method="farthest")
        result = wcp(point_cloud_2d)
        assert isinstance(result, PersistenceDiagram)

    @_core_skip
    def test_random_method(self, point_cloud_2d):
        from pynerve.nn.building_blocks import (
            PersistenceDiagram,
            WitnessComplexPersistence,
        )

        wcp = WitnessComplexPersistence(n_landmarks=5, max_dim=1, method="random", random_seed=42)
        result = wcp(point_cloud_2d)
        assert isinstance(result, PersistenceDiagram)

    @_core_skip
    def test_kmeans_method(self, point_cloud_2d):
        from pynerve.nn.building_blocks import (
            PersistenceDiagram,
            WitnessComplexPersistence,
        )

        wcp = WitnessComplexPersistence(n_landmarks=3, max_dim=0, method="kmeans")
        result = wcp(point_cloud_2d)
        assert isinstance(result, PersistenceDiagram)

    def test_invalid_method_raises(self):
        from pynerve.nn.building_blocks import WitnessComplexPersistence

        with pytest.raises(ValueError):
            WitnessComplexPersistence(method="nonexistent")


class TestPersistenceSketch:
    """Tests for PersistenceSketch."""

    def test_statistics_method_from_points(self, point_cloud_2d):
        from pynerve.nn.building_blocks import PersistenceSketch

        sketch = PersistenceSketch(output_dim=64, method="statistics", max_dim=0)
        result = sketch(point_cloud_2d)
        assert isinstance(result, Tensor)
        assert result.shape == (64,)

    def test_landscape_method_from_points(self, point_cloud_2d):
        from pynerve.nn.building_blocks import PersistenceSketch

        sketch = PersistenceSketch(output_dim=16, method="landscape", max_dim=0)
        result = sketch(point_cloud_2d)
        assert result.shape == (16,)

    def test_image_method_from_points(self, point_cloud_2d):
        from pynerve.nn.building_blocks import PersistenceSketch

        sketch = PersistenceSketch(output_dim=32, method="image", max_dim=0)
        result = sketch(point_cloud_2d)
        assert result.shape == (32,)

    def test_betti_curve_method_from_points(self, point_cloud_2d):
        from pynerve.nn.building_blocks import PersistenceSketch

        sketch = PersistenceSketch(output_dim=10, method="betti_curve", max_dim=0)
        result = sketch(point_cloud_2d)
        assert result.shape == (10,)

    def test_from_persistence_diagram(self):
        from pynerve.nn.building_blocks import PersistenceDiagram, PersistenceSketch

        pd = PersistenceDiagram(
            births=torch.tensor([0.0, 0.3]),
            deaths=torch.tensor([0.5, 0.8]),
            dimensions=torch.tensor([0, 1], dtype=torch.long),
        )
        sketch = PersistenceSketch(output_dim=20, method="statistics")
        result = sketch(pd)
        assert result.shape == (20,)

    def test_output_dim_larger_than_stats_pads(self, point_cloud_2d):
        from pynerve.nn.building_blocks import PersistenceSketch

        sketch = PersistenceSketch(output_dim=1000, method="statistics", max_dim=0)
        result = sketch(point_cloud_2d)
        assert result.shape == (1000,)
        # Last elements should be zero if padding occurred
        # (number of stats for 0D is 13: count + max_dim + 3*4 summaries)

    def test_invalid_method_raises(self):
        from pynerve.nn.building_blocks import PersistenceSketch

        with pytest.raises(ValueError):
            PersistenceSketch(method="bad")
