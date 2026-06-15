"""Numerical correctness tests for torch mapper filter functions."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestPcaFilter:
    """Numerical correctness for _filter_pca_python."""

    def test_three_collinear_points(self) -> None:
        from pynerve.torch._mapper_impl import _filter_pca_python

        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        result = _filter_pca_python(pts, n_components=1)
        assert result.shape == (3, 1)
        assert result[0].item() == pytest.approx(-1.0 / 3, abs=1e-6)
        assert result[1].item() == pytest.approx(2.0 / 3, abs=1e-6)
        assert result[2].item() == pytest.approx(-1.0 / 3, abs=1e-6)

    def test_two_components_on_plane(self) -> None:
        from pynerve.torch._mapper_impl import _filter_pca_python

        pts = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=torch.float64,
        )
        result = _filter_pca_python(pts, n_components=2)
        assert result.shape == (4, 2)

    def test_single_point(self) -> None:
        from pynerve.torch._mapper_impl import _filter_pca_python

        pts = torch.tensor([[5.0, 5.0]], dtype=torch.float64)
        result = _filter_pca_python(pts, n_components=1)
        assert result.shape == (1, 1)
        assert result[0].item() == pytest.approx(0.0, abs=1e-10)

    def test_two_identical_points(self) -> None:
        from pynerve.torch._mapper_impl import _filter_pca_python

        pts = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        result = _filter_pca_python(pts, n_components=1)
        assert result.shape == (2, 1)
        assert (result == 0).all()


class TestEccentricityFilter:
    """Numerical correctness for _filter_eccentricity_python."""

    def test_three_collinear_points(self) -> None:
        from pynerve.torch._mapper_impl import _filter_eccentricity_python

        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        result = _filter_eccentricity_python(pts)
        assert result.shape == (3,)
        assert result[0].item() == pytest.approx(1.0 / 3, abs=1e-6)
        assert result[1].item() == pytest.approx(2.0 / 3, abs=1e-6)
        assert result[2].item() == pytest.approx(1.0 / 3, abs=1e-6)

    def test_single_point(self) -> None:
        from pynerve.torch._mapper_impl import _filter_eccentricity_python

        pts = torch.tensor([[5.0, 5.0]], dtype=torch.float64)
        result = _filter_eccentricity_python(pts)
        assert result[0].item() == pytest.approx(0.0, abs=1e-10)

    def test_two_points(self) -> None:
        from pynerve.torch._mapper_impl import _filter_eccentricity_python

        pts = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
        result = _filter_eccentricity_python(pts)
        assert result[0].item() == pytest.approx(1.0, abs=1e-6)
        assert result[1].item() == pytest.approx(1.0, abs=1e-6)

    def test_identical_points(self) -> None:
        from pynerve.torch._mapper_impl import _filter_eccentricity_python

        pts = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        result = _filter_eccentricity_python(pts)
        assert (result == 0).all()
