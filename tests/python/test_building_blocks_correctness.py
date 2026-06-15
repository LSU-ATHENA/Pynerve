"""Direct numerical tests for building blocks persistence.

Covers _zero_dimensional_diagram, landmark selection,
and PersistenceSketch internal methods.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestZeroDimensionalDiagram:
    """Pure-Python H0 persistence via Prim's algorithm."""

    def test_two_points(self) -> None:
        from pynerve.nn._building_blocks_persistence import _zero_dimensional_diagram

        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        d = _zero_dimensional_diagram(pts)
        births, deaths, dims = d.births, d.deaths, d.dimensions
        finite = deaths[torch.isfinite(deaths)]
        assert finite[0].item() == pytest.approx(1.0, abs=1e-10)
        assert births[0].item() == pytest.approx(0.0, abs=1e-10)
        assert dims[0].item() == 0
        assert not torch.isfinite(deaths[1])

    def test_three_collinear(self) -> None:
        from pynerve.nn._building_blocks_persistence import _zero_dimensional_diagram

        pts = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=torch.float64)
        d = _zero_dimensional_diagram(pts)
        finite = d.deaths[torch.isfinite(d.deaths)].sort().values
        assert finite[0].item() == pytest.approx(0.5, abs=1e-10)
        assert finite[1].item() == pytest.approx(0.5, abs=1e-10)

    def test_single_point(self) -> None:
        from pynerve.nn._building_blocks_persistence import _zero_dimensional_diagram

        pts = torch.tensor([[5.0, 5.0]], dtype=torch.float64)
        d = _zero_dimensional_diagram(pts)
        assert not torch.isfinite(d.deaths[0])
        assert d.births[0].item() == pytest.approx(0.0, abs=1e-10)

    def test_two_identical_points(self) -> None:
        from pynerve.nn._building_blocks_persistence import _zero_dimensional_diagram

        pts = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        d = _zero_dimensional_diagram(pts)
        finite = d.deaths[torch.isfinite(d.deaths)]
        assert finite[0].item() == pytest.approx(0.0, abs=1e-10)

    def test_equilateral_triangle_deaths(self) -> None:
        from pynerve.nn._building_blocks_persistence import _zero_dimensional_diagram

        s3 = 3**0.5
        pts = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, s3]], dtype=torch.float64)
        d = _zero_dimensional_diagram(pts)
        finite = d.deaths[torch.isfinite(d.deaths)].sort().values
        assert finite[0].item() == pytest.approx(2.0, abs=1e-6)
        assert finite[1].item() == pytest.approx(2.0, abs=1e-6)


class TestFarthestLandmarks:
    """Farthest-point landmark selection."""

    def test_selects_correct_first(self) -> None:
        from pynerve.nn._building_blocks_persistence import _farthest_landmarks

        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]], dtype=torch.float64)
        idx = _farthest_landmarks(pts, 3)
        assert idx[0].item() == 0
        assert idx.shape[0] == 3

    def test_returns_unique_indices(self) -> None:
        from pynerve.nn._building_blocks_persistence import _farthest_landmarks

        pts = torch.randn(50, 2)
        idx = _farthest_landmarks(pts, 20)
        assert len(set(idx.tolist())) == 20

    def test_three_collinear_points(self) -> None:
        from pynerve.nn._building_blocks_persistence import _farthest_landmarks

        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
        idx = _farthest_landmarks(pts, 3)
        assert idx[0].item() == 0
        assert idx[1].item() == 2
        assert idx[2].item() == 1


class TestGreedyCoverLandmarks:
    """Greedy cover landmark selection."""

    def test_cover_radius_selects_subset(self) -> None:
        from pynerve.nn._building_blocks_persistence import _greedy_cover_landmarks

        pts = torch.tensor([[0.0, 0.0], [0.1, 0.1], [10.0, 0.0], [10.1, 0.0]], dtype=torch.float64)
        idx = _greedy_cover_landmarks(pts, cover_radius=1.0)
        assert idx.shape[0] == 2
        assert 0 in idx.tolist() or 1 in idx.tolist()
        assert 2 in idx.tolist() or 3 in idx.tolist()


class TestKMeansLandmarks:
    """k-means landmark computation."""

    def test_basic_clustering(self) -> None:
        from pynerve.nn._building_blocks_persistence import _kmeans_landmarks

        pts = torch.tensor(
            [[0.0, 0.0], [0.0, 0.1], [10.0, 0.0], [10.0, 0.1]],
            dtype=torch.float64,
        )
        centers = _kmeans_landmarks(pts, 2)
        assert centers.shape == (2, 2)
        assert centers[:, 0].sort().values[0].item() < 5.0
        assert centers[:, 0].sort().values[1].item() > 5.0


class TestPersistenceSketchStatistics:
    """PersistenceSketch._statistics_from_diagram with known values."""

    def test_statistics_vector_values(self) -> None:
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=30, method="statistics")
        pd = PersistenceDiagram(
            births=torch.tensor([1.0, 3.0, 5.0]),
            deaths=torch.tensor([2.0, 7.0, float("inf")]),
            dimensions=torch.tensor([0, 0, 0]),
        )
        v = sketch._statistics_from_diagram(pd)
        assert v[0].item() == pytest.approx(3.0, abs=1e-10)
        assert v[1].item() == pytest.approx(0.0, abs=1e-10)
        assert v[2].item() == pytest.approx(3.0, abs=1e-10)
        assert v[4].item() == pytest.approx(1.0, abs=1e-10)
        assert v[5].item() == pytest.approx(5.0, abs=1e-10)
        assert v[6].item() == pytest.approx(4.5, abs=1e-10)
        assert v[12].item() == pytest.approx(1.0, abs=1e-10)
        assert v[13].item() == pytest.approx(4.0, abs=1e-10)

    def test_empty_diagram(self) -> None:
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=10, method="statistics")
        pd = PersistenceDiagram(
            births=torch.tensor([]),
            deaths=torch.tensor([]),
            dimensions=torch.tensor([], dtype=torch.long),
        )
        v = sketch._statistics_from_diagram(pd)
        assert v[0].item() == pytest.approx(0.0, abs=1e-10)

    def test_statistics_finite_deaths_summary(self) -> None:
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=30, method="statistics")
        pd = PersistenceDiagram(
            births=torch.tensor([0.0, 0.0]),
            deaths=torch.tensor([2.0, 4.0]),
            dimensions=torch.tensor([0, 0]),
        )
        v = sketch._statistics_from_diagram(pd)
        assert v[6].item() == pytest.approx(3.0, abs=1e-10)
        assert v[8].item() == pytest.approx(2.0, abs=1e-10)
        assert v[9].item() == pytest.approx(4.0, abs=1e-10)

    def test_persistence_summary(self) -> None:
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=30, method="statistics")
        pd = PersistenceDiagram(
            births=torch.tensor([0.0, 0.0]),
            deaths=torch.tensor([1.0, 3.0]),
            dimensions=torch.tensor([0, 0]),
        )
        v = sketch._statistics_from_diagram(pd)
        assert v[10].item() == pytest.approx(2.0, abs=1e-10)
        assert v[12].item() == pytest.approx(1.0, abs=1e-10)
        assert v[13].item() == pytest.approx(3.0, abs=1e-10)


class TestPersistenceSketchBettiCurve:
    """PersistenceSketch._betti_curve_from_diagram with known values."""

    def test_betti_curve_values(self) -> None:
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=4, method="betti_curve")
        pd = PersistenceDiagram(
            births=torch.tensor([0.0, 0.0]),
            deaths=torch.tensor([1.0, 3.0]),
            dimensions=torch.tensor([0, 0]),
        )
        v = sketch._betti_curve_from_diagram(pd)
        assert v.shape == (4,)

    def test_betti_curve_empty(self) -> None:
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=5, method="betti_curve")
        pd = PersistenceDiagram(
            births=torch.tensor([]),
            deaths=torch.tensor([]),
            dimensions=torch.tensor([], dtype=torch.long),
        )
        v = sketch._betti_curve_from_diagram(pd)
        assert (v == 0).all()


class TestPersistenceSketchLandscape:
    """PersistenceSketch._landscape_from_diagram with known values."""

    def test_landscape_single_point(self) -> None:
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=5, method="landscape")
        pd = PersistenceDiagram(
            births=torch.tensor([0.0]),
            deaths=torch.tensor([4.0]),
            dimensions=torch.tensor([0]),
        )
        v = sketch._landscape_from_diagram(pd)
        assert v[0].item() == pytest.approx(4.0, abs=1e-10)
        assert v[1:].sum().item() == pytest.approx(0.0, abs=1e-10)

    def test_landscape_two_points_sorted(self) -> None:
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=5, method="landscape")
        pd = PersistenceDiagram(
            births=torch.tensor([0.0, 0.0]),
            deaths=torch.tensor([2.0, 5.0]),
            dimensions=torch.tensor([0, 0]),
        )
        v = sketch._landscape_from_diagram(pd)
        assert v[0].item() == pytest.approx(5.0, abs=1e-10)
        assert v[1].item() == pytest.approx(2.0, abs=1e-10)
        assert v[2:].sum().item() == pytest.approx(0.0, abs=1e-10)

    def test_landscape_empty(self) -> None:
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=5, method="landscape")
        pd = PersistenceDiagram(
            births=torch.tensor([]),
            deaths=torch.tensor([]),
            dimensions=torch.tensor([], dtype=torch.long),
        )
        v = sketch._landscape_from_diagram(pd)
        assert (v == 0).all()


class TestPersistenceSketchImage:
    """PersistenceSketch._image_from_diagram with known values."""

    def test_image_single_point(self) -> None:
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=5, method="image")
        pd = PersistenceDiagram(
            births=torch.tensor([2.0]),
            deaths=torch.tensor([5.0]),
            dimensions=torch.tensor([0]),
        )
        v = sketch._image_from_diagram(pd)
        assert v.shape == (5,)
        assert v.sum().item() == pytest.approx(3.0, abs=1e-10)

    def test_image_multiple_points(self) -> None:
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=5, method="image")
        pd = PersistenceDiagram(
            births=torch.tensor([0.0, 2.0]),
            deaths=torch.tensor([2.0, 5.0]),
            dimensions=torch.tensor([0, 0]),
        )
        v = sketch._image_from_diagram(pd)
        assert v.sum().item() == pytest.approx(5.0, abs=1e-10)
        assert v[0].item() > 0
        assert v[4].item() == pytest.approx(3.0, abs=1e-10)

    def test_image_empty(self) -> None:
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=5, method="image")
        pd = PersistenceDiagram(
            births=torch.tensor([]),
            deaths=torch.tensor([]),
            dimensions=torch.tensor([], dtype=torch.long),
        )
        v = sketch._image_from_diagram(pd)
        assert (v == 0).all()


class TestPersistenceSketchForward:
    """PersistenceSketch.__call__ integration."""

    def test_forward_statistics(self) -> None:
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=20, method="statistics")
        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        v = sketch(pts)
        assert v.shape == (20,)

    def test_forward_landscape(self) -> None:
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=10, method="landscape")
        pts = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
        v = sketch(pts)
        assert v.shape == (10,)

    def test_forward_empty_diagram(self) -> None:
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        from pynerve.nn._building_blocks_persistence import PersistenceSketch

        sketch = PersistenceSketch(output_dim=8, method="statistics")
        pd = PersistenceDiagram(
            births=torch.tensor([]),
            deaths=torch.tensor([]),
            dimensions=torch.tensor([], dtype=torch.long),
        )
        v = sketch(pd)
        assert v.shape == (8,)
