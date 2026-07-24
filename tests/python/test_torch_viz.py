"""Tests for torch/viz_impl.py and torch/_viz_data.py — viz data converters."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pynerve.torch._diagram import PersistenceDiagram
from pynerve.torch._viz_data import (
    _as_plot_tensor,
    _as_single_tensor,
    _as_tensor,
    _to_numpy,
    _validate_optional_dim,
    _validate_viz_tensor,
    diagram_to_betti_data,
    diagram_to_heatmap_data,
    diagram_to_histogram_data,
    diagram_to_image_data,
    diagram_to_landscape_data,
    diagram_to_scatter_data,
)
from pynerve.torch.viz_impl import get_plot_limits


# ── _to_numpy ──────────────────────────────────────────────────────────────


class TestToNumpy:
    def test_basic(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = _to_numpy(t)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, np.array([1.0, 2.0, 3.0]))


# ── _validate_optional_dim ─────────────────────────────────────────────────


class TestValidateOptionalDim:
    def test_returns_none(self):
        assert _validate_optional_dim(None) is None

    def test_valid(self):
        assert _validate_optional_dim(0) == 0
        assert _validate_optional_dim(5) == 5

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_optional_dim(-1)


# ── _validate_viz_tensor ──────────────────────────────────────────────────


class TestValidateVizTensor:
    def test_valid(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = _validate_viz_tensor(d)
        assert result is d

    def test_infinite_death_raises(self):
        d = torch.tensor([[0.0, float("inf"), 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="finite deaths"):
            _validate_viz_tensor(d)


# ── _as_tensor / _as_plot_tensor ───────────────────────────────────────────


class TestAsTensor:
    def test_plain_tensor(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = _as_tensor(d)
        assert result is not None

    def test_persistence_diagram(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        result = _as_tensor(pd)
        assert result is not None


class TestAsPlotTensor:
    def test_plain_tensor(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = _as_plot_tensor(d)
        assert result is not None

    def test_persistence_diagram_applies_mask(self):
        d = torch.tensor([[[0.0, 1.0, 0], [2.0, 3.0, 0]]], dtype=torch.float32)
        mask = torch.tensor([[True, False]], dtype=torch.bool)
        pd = PersistenceDiagram(d, mask=mask)
        result = _as_plot_tensor(pd)
        assert result.shape[0] == 1  # only masked rows


class TestAsSingleTensor:
    def test_plain_2d_tensor(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = _as_single_tensor(d)
        assert result.dim() == 2

    def test_3d_batch_size_1(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        result = _as_single_tensor(d)
        assert result.dim() == 2

    def test_3d_batch_size_gt_1_raises(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[2.0, 3.0, 0]]], dtype=torch.float32
        )
        with pytest.raises(ValueError, match="single"):
            _as_single_tensor(d)

    def test_persistence_diagram_batch_gt_1_raises(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[2.0, 3.0, 0]]], dtype=torch.float32
        )
        pd = PersistenceDiagram(d)
        with pytest.raises(ValueError, match="single"):
            _as_single_tensor(pd)


# ── get_plot_limits ────────────────────────────────────────────────────────


class TestGetPlotLimits:
    def test_basic(self):
        d = torch.tensor([[0.0, 2.0, 0], [1.0, 4.0, 0]], dtype=torch.float32)
        x_min, x_max, y_min, y_max = get_plot_limits(d)
        assert x_min < 0.0
        assert x_max > 4.0
        assert x_min == y_min  # symmetric

    def test_all_infinite_deaths(self):
        d = torch.tensor([[0.0, float("inf"), 0]], dtype=torch.float32)
        result = get_plot_limits(d)
        assert result == (0.0, 1.0, 0.0, 1.0)

    def test_custom_padding(self):
        d = torch.tensor([[0.0, 10.0, 0]], dtype=torch.float32)
        x_min, x_max, _, _ = get_plot_limits(d, padding=0.5)
        assert x_min < 0.0
        assert x_max > 10.0

    def test_invalid_padding_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="non-negative"):
            get_plot_limits(d, padding=-1.0)

    def test_persistence_diagram_input(self):
        d = torch.tensor([[[0.0, 1.0, 0], [2.0, 5.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        result = get_plot_limits(pd)
        assert len(result) == 4


# ── diagram_to_scatter_data ────────────────────────────────────────────────


class TestDiagramToScatterData:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 1]], dtype=torch.float32)
        result = diagram_to_scatter_data(d)
        assert "births" in result
        assert "deaths" in result
        assert "dims" in result
        assert "persistence" in result
        assert isinstance(result["births"], np.ndarray)

    def test_dim_filter(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 1]], dtype=torch.float32)
        result = diagram_to_scatter_data(d, dim=0)
        assert len(result["dims"]) == 1

    def test_negative_dim_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="non-negative"):
            diagram_to_scatter_data(d, dim=-1)

    def test_empty(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = diagram_to_scatter_data(d)
        assert len(result["births"]) == 0


# ── diagram_to_histogram_data ──────────────────────────────────────────────


class TestDiagramToHistogramData:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = diagram_to_histogram_data(d)
        assert "values" in result
        assert "bins" in result
        assert "title" in result

    def test_dim_filter(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 1]], dtype=torch.float32)
        result = diagram_to_histogram_data(d, dim=0)
        assert len(result["values"]) == 1

    def test_empty(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = diagram_to_histogram_data(d)
        assert isinstance(result["values"], np.ndarray)

    def test_invalid_num_bins_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises((ValueError, TypeError)):
            diagram_to_histogram_data(d, num_bins=0)


# ── diagram_to_image_data ─────────────────────────────────────────────────


class TestDiagramToImageData:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = diagram_to_image_data(d, resolution=(10, 10))
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 10)

    def test_custom_resolution(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = diagram_to_image_data(d, resolution=(5, 8))
        assert result.shape == (5, 8)

    def test_batched_raises(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[2.0, 3.0, 0]]], dtype=torch.float32
        )
        with pytest.raises(ValueError, match="single"):
            diagram_to_image_data(d)


# ── diagram_to_landscape_data ──────────────────────────────────────────────


class TestDiagramToLandscapeData:
    def test_basic(self):
        d = torch.tensor([[0.0, 2.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = diagram_to_landscape_data(d, k=3, num_samples=20)
        assert "landscapes" in result
        assert "x_values" in result
        assert result["k"] == 3
        assert result["landscapes"].shape == (3, 20)

    def test_invalid_num_samples_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises((ValueError, TypeError)):
            diagram_to_landscape_data(d, num_samples=0)


# ── diagram_to_betti_data ──────────────────────────────────────────────────


class TestDiagramToBettiData:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = diagram_to_betti_data(d, num_samples=10)
        assert "thresholds" in result
        assert "betti_numbers" in result
        assert len(result["betti_numbers"]) == 10

    def test_dim_filter(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 1]], dtype=torch.float32)
        result = diagram_to_betti_data(d, dim=0)
        assert len(result["betti_numbers"]) == 100


# ── diagram_to_heatmap_data ────────────────────────────────────────────────


class TestDiagramToHeatmapData:
    def test_basic(self):
        d = torch.tensor([[0.0, 2.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = diagram_to_heatmap_data(d, grid_size=10)
        assert "grid" in result
        assert "birth_edges" in result
        assert "death_edges" in result
        assert result["grid"].shape == (10, 10)

    def test_empty(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = diagram_to_heatmap_data(d)
        assert result["grid"].shape == (20, 20)

    def test_invalid_grid_size_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises((ValueError, TypeError)):
            diagram_to_heatmap_data(d, grid_size=0)
