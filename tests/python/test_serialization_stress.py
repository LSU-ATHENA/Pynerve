"""Serialization round-trip and large-scale stress tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


class TestPersistenceDiagramSerialization:
    """torch.save/load round-trip preserves numerical values."""

    def test_torch_save_load_roundtrip(self):
        from pynerve.torch import PersistenceDiagram

        orig = PersistenceDiagram(
            torch.tensor(
                [[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 3.0, 0.0]]],
                dtype=torch.float64,
            )
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)
            torch.save(orig, path)
            loaded = torch.load(path, weights_only=False)
            path.unlink()
        torch.testing.assert_close(orig.diagrams, loaded.diagrams)
        assert loaded.batch_size == orig.batch_size

    def test_torch_save_load_batched(self):
        from pynerve.torch import PersistenceDiagram

        orig = PersistenceDiagram(
            torch.tensor(
                [[[0.0, 1.0, 0.0]], [[0.0, 5.0, 0.0]]],
                dtype=torch.float64,
            )
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)
            torch.save(orig, path)
            loaded = torch.load(path, weights_only=False)
            path.unlink()
        assert loaded.batch_size == 2
        torch.testing.assert_close(orig.diagrams, loaded.diagrams)

    def test_torch_save_partial_diagram(self):
        from pynerve.torch import PersistenceDiagram

        diagrams = torch.zeros(1, 5, 3, dtype=torch.float64)
        masks = torch.zeros(1, 5, dtype=torch.bool)
        masks[0, :2] = True
        diagrams[0, 0] = torch.tensor([0.0, 1.0, 0.0])
        diagrams[0, 1] = torch.tensor([0.0, 2.0, 1.0])
        num_pairs = torch.tensor([[2, 1]], dtype=torch.long)
        orig = PersistenceDiagram(diagrams, masks, num_pairs)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)
            torch.save(orig, path)
            loaded = torch.load(path, weights_only=False)
            path.unlink()
        torch.testing.assert_close(orig.diagrams, loaded.diagrams)
        assert loaded.batch_size == orig.batch_size

    def test_nn_persistence_diagram_to_numpy(self):
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram

        pd = PersistenceDiagram(
            births=torch.tensor([0.0, 1.0]),
            deaths=torch.tensor([2.0, float("inf")]),
            dimensions=torch.tensor([0, 1]),
        )
        births_np, deaths_np, dims_np = pd.to_numpy()
        import numpy as np

        np.testing.assert_array_almost_equal(births_np, [0.0, 1.0])
        np.testing.assert_array_almost_equal(deaths_np, [2.0, float("inf")])
        np.testing.assert_array_equal(dims_np, [0, 1])

    def test_nn_persistence_diagram_persistence_values(self):
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram

        pd = PersistenceDiagram(
            births=torch.tensor([0.0, 2.0]),
            deaths=torch.tensor([1.0, 5.0]),
            dimensions=torch.tensor([0, 0]),
        )
        pv = pd.persistence_values()
        torch.testing.assert_close(pv, torch.tensor([1.0, 3.0]))

    def test_nn_persistence_diagram_get_dimension(self):
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram

        pd = PersistenceDiagram(
            births=torch.tensor([0.0, 0.0, 0.0]),
            deaths=torch.tensor([1.0, 2.0, 3.0]),
            dimensions=torch.tensor([0, 1, 0]),
        )
        pd0 = pd.get_dimension(0)
        assert pd0.births.shape == (2,)
        pd1 = pd.get_dimension(1)
        assert pd1.births.shape == (1,)

    def test_nn_persistence_diagram_len(self):
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram

        pd = PersistenceDiagram(
            births=torch.tensor([0.0, 0.0]),
            deaths=torch.tensor([1.0, 2.0]),
            dimensions=torch.tensor([0, 0]),
        )
        assert len(pd) == 2


class TestLargeScaleStress:
    """Stress tests with large diagrams."""

    def test_statistics_10000_points(self):
        from pynerve.torch.statistics import max_persistence, mean_persistence, total_persistence

        births = torch.zeros(10000, dtype=torch.float64)
        deaths = torch.arange(1.0, 10001.0, dtype=torch.float64)
        d = torch.stack([births, deaths], dim=1)
        tp = total_persistence(d, p=1.0)
        assert tp.item() == pytest.approx(50005000.0, rel=1e-10)
        mp = mean_persistence(d)
        assert mp.item() == pytest.approx(5000.5, rel=1e-10)
        mx = max_persistence(d)
        assert mx.item() == pytest.approx(10000.0, rel=1e-10)

    def test_stats_100k_points(self):
        from pynerve.torch.statistics import mean_persistence, total_persistence

        births = torch.zeros(100000, dtype=torch.float64)
        deaths = torch.arange(1.0, 100001.0, dtype=torch.float64)
        d = torch.stack([births, deaths], dim=1)
        tp = total_persistence(d, p=1.0)
        assert tp.item() == pytest.approx(5000050000.0, rel=1e-8)
        mp = mean_persistence(d)
        assert mp.item() == pytest.approx(50000.5, rel=1e-8)

    def test_wasserstein_1000_points(self):
        from pynerve.torch import diagram_wasserstein

        births = torch.rand(1000) * 10
        d1 = torch.stack([births, births + torch.rand(1000) * 5 + 0.1], dim=1)
        d2 = torch.stack([births, births + torch.rand(1000) * 5 + 0.1], dim=1)
        dist = diagram_wasserstein(d1, d2)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert torch.isfinite(val)
        assert val.item() >= 0

    def test_persistence_image_resolution_100(self):
        from pynerve.torch.vectorization import persistence_image

        births = torch.rand(500) * 10
        d = torch.stack([births, births + torch.rand(500) * 5 + 0.1], dim=1)
        img = persistence_image(d, resolution=(100, 100), sigma=0.5)
        assert img.shape == (100, 100)
        assert torch.isfinite(img).all()

    def test_landscape_many_points(self):
        from pynerve.torch.vectorization import persistence_landscape

        births = torch.rand(500) * 10
        d = torch.stack([births, births + torch.rand(500) * 5 + 0.1], dim=1)
        xr: tuple[float, float] = (0.0, 15.0)
        land = persistence_landscape(d, k=5, num_samples=50, x_range=xr)
        assert land.shape == (5, 50)
        assert torch.isfinite(land).all()

    def test_entropy_10000_points(self):
        from pynerve.torch.statistics import persistence_entropy

        births = torch.zeros(10000, dtype=torch.float64)
        deaths = torch.arange(1.0, 10001.0, dtype=torch.float64)
        d = torch.stack([births, deaths], dim=1)
        ent = persistence_entropy(d)
        assert torch.isfinite(ent)
        assert ent.item() > 0
