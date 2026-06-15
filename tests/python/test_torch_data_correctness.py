"""Numerical correctness tests for torch data utilities."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestPersistenceDataset:
    """Numerical correctness for PersistenceDataset."""

    def test_len_matches_input(self) -> None:
        from pynerve.torch.data import PersistenceDataset

        clouds = [torch.randn(10, 2) for _ in range(5)]
        ds = PersistenceDataset(clouds, max_dim=0, max_radius=5.0)
        assert len(ds) == 5

    def test_getitem_returns_diagram(self) -> None:
        from pynerve.torch.data import PersistenceDataset

        clouds = [torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)]
        ds = PersistenceDataset(clouds, max_dim=0, max_radius=5.0)
        diagram = ds[0]
        deaths = diagram.deaths()
        finite_mask = torch.isfinite(deaths)
        finite_deaths = deaths[finite_mask]
        assert finite_deaths.numel() >= 1
        assert finite_deaths[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_getitem_with_labels(self) -> None:
        from pynerve.torch.data import PersistenceDataset

        clouds = [torch.randn(10, 2) for _ in range(3)]
        labels = torch.tensor([0, 1, 0], dtype=torch.long)
        ds = PersistenceDataset(clouds, labels=labels, max_dim=0, max_radius=5.0)
        diagram, label = ds[1]
        assert label.item() == 1

    def test_cache_returns_same_object(self) -> None:
        from pynerve.torch.data import PersistenceDataset

        clouds = [torch.randn(10, 2) for _ in range(2)]
        ds = PersistenceDataset(clouds, max_dim=0, max_radius=5.0, cache=True)
        d1 = ds[0]
        d2 = ds[0]
        assert d1.diagrams.shape == d2.diagrams.shape


class TestPointCloudDataset:
    """Numerical correctness for PointCloudDataset."""

    def test_getitem_returns_tensor(self) -> None:
        from pynerve.torch.data import PointCloudDataset

        clouds = [torch.randn(8, 3) for _ in range(4)]
        ds = PointCloudDataset(clouds)
        pc = ds[2]
        assert pc.shape == (8, 3)

    def test_getitem_with_labels(self) -> None:
        from pynerve.torch.data import PointCloudDataset

        clouds = [torch.randn(5, 2) for _ in range(3)]
        labels = torch.tensor([10, 20, 30], dtype=torch.float32)
        ds = PointCloudDataset(clouds, labels=labels)
        pc, label = ds[0]
        assert label.item() == 10


class TestCollateDiagrams:
    """Numerical correctness for collate_diagrams."""

    def test_collate_two_diagrams(self) -> None:
        from pynerve.torch import PersistenceDiagram
        from pynerve.torch.data import collate_diagrams

        d1 = PersistenceDiagram(
            torch.tensor([[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]]], dtype=torch.float64)
        )
        d2 = PersistenceDiagram(torch.tensor([[[0.0, 3.0, 0.0]]], dtype=torch.float64))
        batched = collate_diagrams([d1, d2])
        assert batched.batch_size == 2

    def test_collate_with_labels(self) -> None:
        from pynerve.torch import PersistenceDiagram
        from pynerve.torch.data import collate_diagrams

        d1 = PersistenceDiagram(torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64))
        d2 = PersistenceDiagram(torch.tensor([[[0.0, 2.0, 0.0]]], dtype=torch.float64))
        labels = [torch.tensor(0), torch.tensor(1)]
        batch_data = collate_diagrams([(d1, labels[0]), (d2, labels[1])])
        assert isinstance(batch_data, tuple)
        assert batch_data[1][0].item() == 0

    def test_collate_empty(self) -> None:
        from pynerve.torch.data import collate_diagrams

        batched = collate_diagrams([])
        assert batched.batch_size == 0


class TestCollatePointClouds:
    """Numerical correctness for collate_point_clouds."""

    def test_collate_pads_to_largest(self) -> None:
        from pynerve.torch.data import collate_point_clouds

        small = torch.randn(3, 2)
        large = torch.randn(7, 2)
        batched = collate_point_clouds([small, large])
        assert batched.shape == (2, 7, 2)
        assert torch.allclose(batched[0, :3], small)
        assert (batched[0, 3:] == 0).all()

    def test_collate_custom_pad(self) -> None:
        from pynerve.torch.data import collate_point_clouds

        a = torch.randn(2, 2)
        b = torch.randn(4, 2)
        batched = collate_point_clouds([a, b], pad_value=-1.0)
        assert (batched[0, 2:] == -1.0).all()

    def test_collate_with_labels(self) -> None:
        from pynerve.torch.data import collate_point_clouds

        a = torch.randn(3, 2)
        b = torch.randn(5, 2)
        labels = [torch.tensor(0), torch.tensor(1)]
        result = collate_point_clouds([(a, labels[0]), (b, labels[1])])
        assert isinstance(result, tuple)
        assert result[1][0].item() == 0
        assert result[1][1].item() == 1


class TestCreateDataloader:
    """Numerical correctness for create_dataloader."""

    def test_dataloader_diagram_batch(self) -> None:
        from pynerve.torch.data import PersistenceDataset, create_dataloader

        clouds = [torch.randn(10, 2) for _ in range(6)]
        ds = PersistenceDataset(clouds, max_dim=0, max_radius=5.0)
        loader = create_dataloader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert batch.batch_size == 2

    def test_dataloader_point_cloud_batch(self) -> None:
        from pynerve.torch.data import PointCloudDataset, create_dataloader

        clouds = [torch.randn(10, 2) for _ in range(4)]
        ds = PointCloudDataset(clouds)
        loader = create_dataloader(ds, batch_size=2, shuffle=False)
        batch: torch.Tensor = next(iter(loader))
        assert batch.shape[0] == 2
