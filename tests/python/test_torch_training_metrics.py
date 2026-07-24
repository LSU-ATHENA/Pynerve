"""Tests for torch/_training_metrics.py — DiagramMetric and TopologicalComplexityMetric."""

from __future__ import annotations

import pytest
import torch

from pynerve.torch._diagram import PersistenceDiagram
from pynerve.torch._training_metrics import (
    DiagramMetric,
    TopologicalComplexityMetric,
    _validate_nonnegative_finite,
)


# ── _validate_nonnegative_finite ───────────────────────────────────────────


class TestValidateNonnegativeFinite:
    def test_valid(self):
        assert _validate_nonnegative_finite(0.0, "x") == 0.0
        assert _validate_nonnegative_finite(5.0, "x") == 5.0

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_nonnegative_finite(-1.0, "x")

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_nonnegative_finite(float("nan"), "x")

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_nonnegative_finite(float("inf"), "x")


# ── DiagramMetric ──────────────────────────────────────────────────────────


class TestDiagramMetric:
    def test_construction_defaults(self):
        metric = DiagramMetric()
        assert metric.name == "diagram"
        assert metric.dim is None
        assert set(metric.track_stats) == {"total", "mean", "max", "count", "entropy"}

    def test_construction_custom_name(self):
        metric = DiagramMetric(name="h1")
        assert metric.name == "h1"

    def test_construction_custom_dim(self):
        metric = DiagramMetric(dim=0)
        assert metric.dim == 0

    def test_construction_custom_track_stats(self):
        metric = DiagramMetric(track_stats=["total", "count"])
        assert set(metric.track_stats) == {"total", "count"}

    def test_construction_invalid_dim_raises(self):
        with pytest.raises((ValueError, TypeError)):
            DiagramMetric(dim=-1)

    def test_construction_invalid_track_stats_raises(self):
        with pytest.raises(ValueError, match="unsupported"):
            DiagramMetric(track_stats=["invalid_stat"])

    def test_construction_string_track_stats_raises(self):
        with pytest.raises(TypeError, match="sequence"):
            DiagramMetric(track_stats="total")  # type: ignore[arg-type]

    def test_update_tensor(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        metric = DiagramMetric()
        metric.update(d)
        assert len(metric.values["total"]) == 1

    def test_update_persistence_diagram(self):
        d = torch.tensor([[[0.0, 1.0, 0], [2.0, 5.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        metric = DiagramMetric()
        metric.update(pd)

    def test_compute_returns_means_and_stds(self):
        d1 = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        d2 = torch.tensor([[1.0, 3.0, 0]], dtype=torch.float32)
        metric = DiagramMetric(track_stats=["total", "count"])
        metric.update(d1)
        metric.update(d2)
        result = metric.compute()
        assert "diagram_total_mean" in result
        assert "diagram_count_mean" in result

    def test_compute_empty_returns_empty(self):
        metric = DiagramMetric()
        result = metric.compute()
        assert result == {}

    def test_reset_clears_values(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        metric = DiagramMetric()
        metric.update(d)
        metric.reset()
        assert metric.compute() == {}

    def test_dim_filter_passed_to_statistics(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 1]], dtype=torch.float32)
        metric = DiagramMetric(dim=0, track_stats=["count"])
        metric.update(d)
        result = metric.compute()
        assert "diagram_count_mean" in result

    def test_single_value_std_is_zero(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        metric = DiagramMetric(track_stats=["total"])
        metric.update(d)
        result = metric.compute()
        assert "diagram_total_std" in result
        assert result["diagram_total_std"] == 0.0


# ── TopologicalComplexityMetric ────────────────────────────────────────────


class TestTopologicalComplexityMetric:
    def test_construction_defaults(self):
        tcm = TopologicalComplexityMetric()
        assert tcm.target == 10.0

    def test_construction_custom_target(self):
        tcm = TopologicalComplexityMetric(target_complexity=5.0)
        assert tcm.target == 5.0

    def test_construction_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            TopologicalComplexityMetric(target_complexity=float("nan"))

    def test_construction_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            TopologicalComplexityMetric(target_complexity=-1.0)

    def test_update_and_compute(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=torch.float32)
        tcm = TopologicalComplexityMetric(target_complexity=5.0)
        tcm.update(d)
        result = tcm.compute()
        assert "complexity" in result
        assert "target_distance" in result
        assert "mean_complexity" in result

    def test_compute_empty(self):
        tcm = TopologicalComplexityMetric()
        result = tcm.compute()
        assert result["complexity"] == 0.0
        assert result["target_distance"] == 10.0

    def test_multiple_updates(self):
        d1 = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        d2 = torch.tensor([[0.0, 2.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        tcm = TopologicalComplexityMetric(target_complexity=1.0)
        tcm.update(d1)
        tcm.update(d2)
        result = tcm.compute()
        assert result["mean_complexity"] > 0

    def test_reset(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        tcm = TopologicalComplexityMetric()
        tcm.update(d)
        tcm.reset()
        result = tcm.compute()
        assert result["complexity"] == 0.0

    def test_update_persistence_diagram(self):
        d = torch.tensor([[[0.0, 1.0, 0], [2.0, 5.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        tcm = TopologicalComplexityMetric(target_complexity=1.0)
        tcm.update(pd)
        result = tcm.compute()
        assert result["complexity"] >= 0
