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


# Data module


class TestPersistenceDataset:
    """PersistenceDataset: construction and item access."""

    def test_construct_with_labels(self) -> None:
        from pynerve.torch.data import PersistenceDataset

        clouds = [torch.rand(5, 2), torch.rand(3, 2)]
        labels = torch.tensor([0, 1])
        ds = PersistenceDataset(clouds, labels=labels, cache=False)
        assert len(ds) == 2

    def test_construct_with_cache(self) -> None:
        from pynerve.torch.data import PersistenceDataset

        clouds = [torch.rand(4, 2)]
        ds = PersistenceDataset(clouds, cache=True)
        assert len(ds) == 1

    def test_construct_empty(self) -> None:
        from pynerve.torch.data import PersistenceDataset

        ds = PersistenceDataset([])
        assert len(ds) == 0

    def test_labels_length_mismatch_raises(self) -> None:
        from pynerve.torch.data import PersistenceDataset

        clouds = [torch.rand(5, 2), torch.rand(3, 2)]
        labels = torch.tensor([0])
        with pytest.raises(ValueError):
            PersistenceDataset(clouds, labels=labels)

    def test_invalid_point_cloud_raises(self) -> None:
        from pynerve.torch.data import PersistenceDataset

        with pytest.raises((TypeError, ValueError)):
            PersistenceDataset([torch.tensor([0, 1])])

    @_torch_backend
    def test_getitem_returns_diagram(self) -> None:
        from pynerve.torch.data import PersistenceDataset

        clouds = [torch.rand(5, 2)]
        ds = PersistenceDataset(clouds, cache=False)
        item = ds[0]
        from pynerve.torch import PersistenceDiagram

        assert isinstance(item, PersistenceDiagram)

    @_torch_backend
    def test_getitem_with_labels(self) -> None:
        from pynerve.torch.data import PersistenceDataset

        clouds = [torch.rand(5, 2)]
        labels = torch.tensor([42])
        ds = PersistenceDataset(clouds, labels=labels, cache=False)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    @_torch_backend
    def test_cache_hit(self) -> None:
        from pynerve.torch import PersistenceDiagram
        from pynerve.torch.data import PersistenceDataset

        clouds = [torch.rand(5, 2)]
        ds = PersistenceDataset(clouds, cache=True)
        item1 = ds[0]
        assert isinstance(item1, PersistenceDiagram)

    def test_invalid_max_dim_raises(self) -> None:
        from pynerve.torch.data import PersistenceDataset

        with pytest.raises((ValueError, TypeError)):
            PersistenceDataset([torch.rand(3, 2)], max_dim=-1)


class TestPointCloudDataset:
    """PointCloudDataset: construction and item access."""

    def test_construct(self) -> None:
        from pynerve.torch.data import PointCloudDataset

        clouds = [torch.rand(5, 2), torch.rand(3, 2)]
        ds = PointCloudDataset(clouds)
        assert len(ds) == 2

    def test_construct_with_labels(self) -> None:
        from pynerve.torch.data import PointCloudDataset

        clouds = [torch.rand(5, 2)]
        labels = torch.tensor([0])
        ds = PointCloudDataset(clouds, labels=labels)
        diagram, label = ds[0]
        assert label.item() == 0

    def test_getitem_without_labels(self) -> None:
        from pynerve.torch.data import PointCloudDataset

        cloud = torch.rand(5, 2)
        ds = PointCloudDataset([cloud])
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (5, 2)

    def test_labels_length_mismatch_raises(self) -> None:
        from pynerve.torch.data import PointCloudDataset

        clouds = [torch.rand(5, 2)]
        labels = torch.tensor([0, 1])
        with pytest.raises(ValueError):
            PointCloudDataset(clouds, labels=labels)

    def test_empty_clouds(self) -> None:
        from pynerve.torch.data import PointCloudDataset

        ds = PointCloudDataset([])
        assert len(ds) == 0

    def test_invalid_cloud_raises(self) -> None:
        from pynerve.torch.data import PointCloudDataset

        with pytest.raises((TypeError, ValueError)):
            PointCloudDataset([torch.tensor([[0, 1]], dtype=torch.int64)])


class TestCollateDiagrams:
    """collate_diagrams (tested extensively in test_torch_ops_contracts.py)."""

    def test_empty_batch(self) -> None:
        from pynerve.torch import PersistenceDiagram
        from pynerve.torch.data import collate_diagrams

        result = collate_diagrams([])
        assert isinstance(result, PersistenceDiagram)
        assert result.batch_size == 0

    def test_batch_with_labels(self) -> None:
        from pynerve.torch import PersistenceDiagram
        from pynerve.torch.data import collate_diagrams

        tensor = _make_diagram_tensor(batch=1, pairs=2)
        pd = PersistenceDiagram(tensor)
        labels = [torch.tensor(0), torch.tensor(1)]
        batch_in = [(pd, labels[0]), (pd, labels[1])]
        result = collate_diagrams(batch_in)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[1].shape == (2,)

    def test_batch_without_labels(self) -> None:
        from pynerve.torch import PersistenceDiagram
        from pynerve.torch.data import collate_diagrams

        tensor = _make_diagram_tensor(batch=1, pairs=2)
        pd = PersistenceDiagram(tensor)
        result = collate_diagrams([pd, pd])
        from pynerve.torch import PersistenceDiagram

        assert isinstance(result, PersistenceDiagram)
        assert result.batch_size == 2


class TestCollatePointClouds:
    """collate_point_clouds (tested extensively in test_torch_ops_contracts.py)."""

    def test_empty_batch(self) -> None:
        from pynerve.torch.data import collate_point_clouds

        result = collate_point_clouds([])
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 0

    def test_basic_padding(self) -> None:
        from pynerve.torch.data import collate_point_clouds

        a = torch.rand(5, 2)
        b = torch.rand(3, 2)
        result = collate_point_clouds([a, b])
        assert result.shape == (2, 5, 2)

    def test_with_labels(self) -> None:
        from pynerve.torch.data import collate_point_clouds

        a = torch.rand(5, 2)
        b = torch.rand(3, 2)
        result = collate_point_clouds([(a, torch.tensor(0)), (b, torch.tensor(1))])
        assert isinstance(result, tuple)
        assert result[0].shape == (2, 5, 2)
        assert result[1].shape == (2,)

    def test_non_tensor_raises(self) -> None:
        from pynerve.torch.data import collate_point_clouds

        with pytest.raises(TypeError):
            collate_point_clouds([[1.0, 2.0]])

    def test_nan_pad_value_raises(self) -> None:
        from pynerve.torch.data import collate_point_clouds

        a = torch.rand(3, 2)
        with pytest.raises(ValueError):
            collate_point_clouds([a], pad_value=float("nan"))


class TestCreateDataloader:
    """create_dataloader construction and validation."""

    def test_create_with_persistence_dataset(self) -> None:
        from pynerve.torch.data import PointCloudDataset, create_dataloader

        ds = PointCloudDataset([torch.rand(5, 2)])
        loader = create_dataloader(ds, batch_size=1, shuffle=False)
        assert loader.batch_size == 1

    def test_invalid_batch_size_raises(self) -> None:
        from pynerve.torch.data import PointCloudDataset, create_dataloader

        ds = PointCloudDataset([torch.rand(5, 2)])
        with pytest.raises((ValueError, ValidationError)):
            create_dataloader(ds, batch_size=0)

    def test_invalid_num_workers_raises(self) -> None:
        from pynerve.torch.data import PointCloudDataset, create_dataloader

        ds = PointCloudDataset([torch.rand(5, 2)])
        with pytest.raises((TypeError, ValueError, ValidationError)):
            create_dataloader(ds, batch_size=1, num_workers=1.5)
