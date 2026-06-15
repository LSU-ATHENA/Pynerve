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


# topo_regularization


class TestTopologicalRegularizationLoss:
    """Tests for TopologicalRegularizationLoss."""

    def test_loss_is_non_negative(self, point_cloud_2d):
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss

        loss_fn = TopologicalRegularizationLoss(min_persistence=0.1, max_dim=0)
        loss = loss_fn(point_cloud_2d)
        assert isinstance(loss, Tensor)
        assert loss.item() >= 0

    def test_batched_input(self, point_cloud_batch):
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss

        loss_fn = TopologicalRegularizationLoss(min_persistence=0.1, max_dim=0)
        loss = loss_fn(point_cloud_batch)
        assert loss.dim() == 0

    def test_reduction_sum(self, point_cloud_batch):
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss

        loss_fn = TopologicalRegularizationLoss(reduction="sum", max_dim=0)
        loss = loss_fn(point_cloud_batch)
        assert loss.dim() == 0

    def test_reduction_none(self, point_cloud_batch):
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss

        loss_fn = TopologicalRegularizationLoss(reduction="none", max_dim=0)
        loss = loss_fn(point_cloud_batch)
        assert loss.shape == (BATCH,)

    def test_with_target_betti(self, point_cloud_2d):
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss

        loss_fn = TopologicalRegularizationLoss(target_betti=[1, 0], max_dim=1, min_persistence=0.1)
        loss = loss_fn(point_cloud_2d)
        assert loss.item() >= 0

    def test_is_nn_module(self):
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss

        assert isinstance(TopologicalRegularizationLoss(), nn.Module)

    def test_invalid_reduction_raises(self):
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss

        with pytest.raises(ValueError):
            TopologicalRegularizationLoss(reduction="max")


class TestPersistenceEntropyLoss:
    """Tests for PersistenceEntropyLoss."""

    def test_loss_is_non_negative(self, point_cloud_2d):
        from pynerve.nn.topo_regularization import PersistenceEntropyLoss

        loss_fn = PersistenceEntropyLoss(target_entropy=2.0, max_dim=0)
        loss = loss_fn(point_cloud_2d)
        assert loss.item() >= 0

    def test_batched_input(self, point_cloud_batch):
        from pynerve.nn.topo_regularization import PersistenceEntropyLoss

        loss_fn = PersistenceEntropyLoss(target_entropy=2.0, max_dim=0)
        loss = loss_fn(point_cloud_batch)
        assert loss.dim() == 0

    def test_is_nn_module(self):
        from pynerve.nn.topo_regularization import PersistenceEntropyLoss

        assert isinstance(PersistenceEntropyLoss(), nn.Module)

    def test_zero_for_empty_diagrams(self):
        from pynerve.nn.topo_regularization import PersistenceEntropyLoss

        loss_fn = PersistenceEntropyLoss(target_entropy=2.0, max_dim=0)
        # Single point produces trivial diagram
        loss = loss_fn(torch.tensor([[0.0, 0.0, 0.0]]))
        assert loss.item() >= 0

    def test_finite_output(self, point_cloud_2d):
        from pynerve.nn.topo_regularization import PersistenceEntropyLoss

        loss_fn = PersistenceEntropyLoss(target_entropy=2.0, max_dim=0)
        loss = loss_fn(point_cloud_2d)
        assert torch.isfinite(loss).item()


class TestTopologicalComplexityLoss:
    """Tests for TopologicalComplexityLoss."""

    def test_loss_is_non_negative(self, point_cloud_2d):
        from pynerve.nn.topo_regularization import TopologicalComplexityLoss

        loss_fn = TopologicalComplexityLoss(min_features=5, max_features=50, max_dim=0)
        loss = loss_fn(point_cloud_2d)
        assert loss.item() >= 0

    def test_batched_input(self, point_cloud_batch):
        from pynerve.nn.topo_regularization import TopologicalComplexityLoss

        loss_fn = TopologicalComplexityLoss(min_features=5, max_features=50, max_dim=0)
        loss = loss_fn(point_cloud_batch)
        assert loss.dim() == 0

    def test_is_nn_module(self):
        from pynerve.nn.topo_regularization import TopologicalComplexityLoss

        assert isinstance(TopologicalComplexityLoss(), nn.Module)

    def test_invalid_bounds_raises(self):
        from pynerve.nn.topo_regularization import TopologicalComplexityLoss

        with pytest.raises(ValueError):
            TopologicalComplexityLoss(min_features=50, max_features=5)


class TestDiagramMatchingLoss:
    """Tests for DiagramMatchingLoss."""

    def test_identical_diagrams_give_zero_loss(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss_fn = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0)
        diag = torch.tensor([[0.0, 1.0], [0.2, 0.8]])
        loss = loss_fn([[diag]], [[diag]])
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_identical_diagrams_bottleneck_zero_loss(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss_fn = DiagramMatchingLoss(distance_metric="bottleneck")
        diag = torch.tensor([[0.0, 1.0], [0.2, 0.8]])
        loss = loss_fn([[diag]], [[diag]])
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_different_diagrams_give_positive_loss(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss_fn = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0)
        pred = torch.tensor([[0.0, 1.0], [0.2, 0.8]])
        target = torch.tensor([[0.1, 1.2], [0.3, 0.7]])
        loss = loss_fn([[pred]], [[target]])
        assert loss.item() > 0

    def test_batched_diagrams(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss_fn = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0)
        d1 = torch.tensor([[0.0, 1.0], [0.2, 0.8]])
        d2 = torch.tensor([[0.1, 1.2], [0.3, 0.7]])
        d3 = torch.tensor([[0.0, 0.5]])
        d4 = torch.tensor([[0.1, 0.6]])
        loss = loss_fn([[d1, d3]], [[d2, d4]])
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_empty_pred_nonempty_target(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss_fn = DiagramMatchingLoss(distance_metric="wasserstein")
        pred = torch.empty((0, 2))
        target = torch.tensor([[0.0, 1.0]])
        loss = loss_fn([[pred]], [[target]])
        assert loss.item() >= 0

    def test_both_empty_returns_zero(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss_fn = DiagramMatchingLoss(distance_metric="wasserstein")
        empty = torch.empty((0, 2))
        loss = loss_fn([[empty]], [[empty]])
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_is_nn_module(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        assert isinstance(DiagramMatchingLoss(), nn.Module)

    def test_invalid_distance_metric_raises(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        with pytest.raises(ValueError):
            DiagramMatchingLoss(distance_metric="euclidean")

    def test_mismatched_batch_sizes_raises(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss_fn = DiagramMatchingLoss()
        d1 = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError):
            loss_fn([[d1]], [[]])

    def test_empty_batch_raises(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss_fn = DiagramMatchingLoss()
        with pytest.raises(ValueError):
            loss_fn([], [])

    def test_bottleneck_metric(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss_fn = DiagramMatchingLoss(distance_metric="bottleneck")
        pred = torch.tensor([[0.0, 1.0]])
        target = torch.tensor([[0.1, 1.1]])
        loss = loss_fn([[pred]], [[target]])
        assert loss.item() >= 0
