"""Numerical correctness tests for torch visualization functions."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_D = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=torch.float64)


class TestDiagramToScatterData:
    """Numerical correctness for diagram_to_scatter_data."""

    def test_scatter_values(self) -> None:
        from pynerve.torch.viz import diagram_to_scatter_data

        s = diagram_to_scatter_data(_D)
        np.testing.assert_array_equal(s["births"], np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(s["deaths"], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(s["persistence"], np.array([1.0, 2.0, 3.0]))

    def test_dim_filter(self) -> None:
        from pynerve.torch.viz import diagram_to_scatter_data

        d3 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float64)
        s = diagram_to_scatter_data(d3, dim=0)
        assert len(s["births"]) == 1
        assert s["births"][0] == 0.0

    def test_empty(self) -> None:
        from pynerve.torch.viz import diagram_to_scatter_data

        s = diagram_to_scatter_data(torch.empty(0, 2, dtype=torch.float64))
        assert len(s["births"]) == 0


class TestDiagramToHistogramData:
    """Numerical correctness for diagram_to_histogram_data."""

    def test_bin_boundaries(self) -> None:
        from pynerve.torch.viz import diagram_to_histogram_data

        h = diagram_to_histogram_data(_D, num_bins=3)
        assert len(h["bins"]) == 4
        assert h["bins"][0] == pytest.approx(0.8, abs=1e-1)
        assert h["bins"][-1] == pytest.approx(3.2, abs=1e-1)

    def test_empty(self) -> None:
        from pynerve.torch.viz import diagram_to_histogram_data

        h = diagram_to_histogram_data(torch.empty(0, 2, dtype=torch.float64), num_bins=3)
        assert len(h["values"]) == 0


class TestDiagramToHeatmapData:
    """Numerical correctness for diagram_to_heatmap_data."""

    def test_grid_shape(self) -> None:
        from pynerve.torch.viz import diagram_to_heatmap_data

        hm = diagram_to_heatmap_data(_D, grid_size=4)
        assert hm["grid"].shape == (4, 4)

    def test_grid_non_negative(self) -> None:
        from pynerve.torch.viz import diagram_to_heatmap_data

        hm = diagram_to_heatmap_data(_D, grid_size=4)
        assert (hm["grid"] >= 0).all()

    def test_empty_diagram(self) -> None:
        from pynerve.torch.viz import diagram_to_heatmap_data

        hm = diagram_to_heatmap_data(torch.empty(0, 2, dtype=torch.float64), grid_size=4)
        assert (hm["grid"] == 0).all()


class TestDiagramToImageData:
    """Numerical correctness for diagram_to_image_data."""

    def test_output_shape(self) -> None:
        from pynerve.torch.viz import diagram_to_image_data

        img = diagram_to_image_data(_D, resolution=(4, 4))
        assert img.shape == (4, 4)

    def test_non_negative(self) -> None:
        from pynerve.torch.viz import diagram_to_image_data

        img = diagram_to_image_data(_D, resolution=(4, 4))
        assert (img >= 0).all()

    def test_empty_diagram(self) -> None:
        from pynerve.torch.viz import diagram_to_image_data

        img = diagram_to_image_data(torch.empty(0, 2, dtype=torch.float64), resolution=(4, 4))
        assert (img == 0).all()


class TestDiagramToLandscapeData:
    """Numerical correctness for diagram_to_landscape_data."""

    def test_x_values(self) -> None:
        from pynerve.torch.viz import diagram_to_landscape_data

        ld = diagram_to_landscape_data(_D, k=2, num_samples=5)
        assert len(ld["x_values"]) == 5
        assert ld["x_values"][0] == pytest.approx(0.0, abs=1e-1)
        assert ld["x_values"][-1] == pytest.approx(3.0, abs=1e-1)

    def test_landscape_shape(self) -> None:
        from pynerve.torch.viz import diagram_to_landscape_data

        ld = diagram_to_landscape_data(_D, k=2, num_samples=5)
        assert ld["landscapes"].shape == (2, 5)

    def test_empty(self) -> None:
        from pynerve.torch.viz import diagram_to_landscape_data

        ld = diagram_to_landscape_data(torch.empty(0, 2, dtype=torch.float64), k=2, num_samples=5)
        assert (ld["landscapes"] == 0).all()


class TestDiagramToBettiData:
    """Numerical correctness for diagram_to_betti_data."""

    def test_betti_values(self) -> None:
        from pynerve.torch.viz import diagram_to_betti_data

        bd = diagram_to_betti_data(_D, num_samples=4)
        assert len(bd["thresholds"]) == 4
        assert len(bd["betti_numbers"]) == 4
        assert (bd["betti_numbers"] >= 0).all()

    def test_empty(self) -> None:
        from pynerve.torch.viz import diagram_to_betti_data

        bd = diagram_to_betti_data(torch.empty(0, 2, dtype=torch.float64), num_samples=3)
        assert (bd["betti_numbers"] == 0).all()


class TestGetPlotLimits:
    """Numerical correctness for get_plot_limits."""

    def test_limits_with_padding(self) -> None:
        from pynerve.torch.viz import get_plot_limits

        limits = get_plot_limits(_D, padding=1.0)
        assert len(limits) == 4
        assert limits[0] == pytest.approx(-1.0, abs=1e-6)
        assert limits[1] == pytest.approx(5.0, abs=1e-6)
        assert limits[2] == pytest.approx(-1.0, abs=1e-6)
        assert limits[3] == pytest.approx(5.0, abs=1e-6)

    def test_limits_zero_padding(self) -> None:
        from pynerve.torch.viz import get_plot_limits

        limits = get_plot_limits(_D, padding=0.0)
        assert limits[0] == pytest.approx(0.0, abs=1e-6)
        assert limits[1] == pytest.approx(3.0, abs=1e-6)
        assert limits[2] == pytest.approx(0.0, abs=1e-6)
        assert limits[3] == pytest.approx(3.0, abs=1e-6)
