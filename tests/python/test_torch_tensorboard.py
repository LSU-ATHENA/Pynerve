"""Tests for torch/tensorboard.py — log functions and DiagramSummaryWriter."""

from __future__ import annotations

import pytest
import torch

from pynerve.torch._diagram import PersistenceDiagram
from pynerve.torch.tensorboard import (
    DiagramSummaryWriter,
    log_betti_curve,
    log_diagram,
    log_landscape,
    log_statistics,
)


# ── Mock writer ────────────────────────────────────────────────────────────


class MockWriter:
    """Mimics SummaryWriter for testing without torch.utils.tensorboard."""

    def __init__(self):
        self.images = []
        self.scalars = []
        self.histograms = []

    def add_image(self, tag, img_tensor, global_step):
        self.images.append((tag, img_tensor.shape, global_step))

    def add_scalar(self, tag, scalar_value, global_step):
        self.scalars.append((tag, scalar_value, global_step))

    def add_histogram(self, tag, values, global_step):
        self.histograms.append((tag, values.shape, global_step))

    def close(self):
        pass


# ── log_diagram ────────────────────────────────────────────────────────────


class TestLogDiagram:
    def test_image_method(self):
        writer = MockWriter()
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        log_diagram(writer, d, step=0, method="image")
        assert len(writer.images) == 1

    def test_heatmap_method(self):
        writer = MockWriter()
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        log_diagram(writer, d, step=0, method="heatmap")
        assert len(writer.images) == 1

    def test_scatter_method(self):
        writer = MockWriter()
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        log_diagram(writer, d, step=0, method="scatter", dim=0)
        assert len(writer.histograms) >= 1

    def test_invalid_method_raises(self):
        writer = MockWriter()
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="method"):
            log_diagram(writer, d, step=0, method="bad")  # type: ignore[arg-type]

    def test_persistence_diagram_input(self):
        writer = MockWriter()
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        log_diagram(writer, pd, step=0, method="image")
        assert len(writer.images) == 1

    def test_custom_tag(self):
        writer = MockWriter()
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        log_diagram(writer, d, step=0, tag="custom", method="image")
        assert writer.images[0][0] == "custom"


# ── log_landscape ──────────────────────────────────────────────────────────


class TestLogLandscape:
    def test_basic(self):
        writer = MockWriter()
        d = torch.tensor([[0.0, 2.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        log_landscape(writer, d, step=0, k=2)
        assert len(writer.scalars) > 0

    def test_invalid_k_raises(self):
        writer = MockWriter()
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="positive"):
            log_landscape(writer, d, step=0, k=0)

    def test_persistence_diagram_input(self):
        writer = MockWriter()
        d = torch.tensor([[[0.0, 2.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        log_landscape(writer, pd, step=0, k=2)
        assert len(writer.scalars) > 0


# ── log_betti_curve ────────────────────────────────────────────────────────


class TestLogBettiCurve:
    def test_basic(self):
        writer = MockWriter()
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        log_betti_curve(writer, d, step=0, num_samples=10)
        assert len(writer.scalars) > 0

    def test_invalid_num_samples_raises(self):
        writer = MockWriter()
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="positive"):
            log_betti_curve(writer, d, step=0, num_samples=0)

    def test_dim_parameter(self):
        writer = MockWriter()
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 1]], dtype=torch.float32)
        log_betti_curve(writer, d, step=0, num_samples=5, dim=0)
        assert len(writer.scalars) > 0


# ── log_statistics ─────────────────────────────────────────────────────────


class TestLogStatistics:
    def test_basic(self):
        writer = MockWriter()
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        log_statistics(writer, d, step=0)
        assert len(writer.scalars) > 0

    def test_custom_dims(self):
        writer = MockWriter()
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 1]], dtype=torch.float32)
        log_statistics(writer, d, step=0, dims=[0, 1])
        # total, mean, max, num, entropy × 2 dims = 10 scalars
        assert len(writer.scalars) >= 5


# ── DiagramSummaryWriter ───────────────────────────────────────────────────


class TestDiagramSummaryWriter:
    def test_construction(self):
        pytest.importorskip("torch.utils.tensorboard")
        sw = DiagramSummaryWriter(log_dir="/tmp/test_tb", comment="test")
        # Cleanup
        if hasattr(sw, "_writer"):
            sw._writer.close()

    def test_add_diagram(self):
        pytest.importorskip("torch.utils.tensorboard")
        sw = DiagramSummaryWriter(log_dir="/tmp/test_tb_diag")
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        sw.add_diagram(d, global_step=0, method="image")
        if hasattr(sw, "_writer"):
            sw._writer.close()

    def test_add_landscape(self):
        pytest.importorskip("torch.utils.tensorboard")
        sw = DiagramSummaryWriter(log_dir="/tmp/test_tb_land")
        d = torch.tensor([[0.0, 2.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        sw.add_landscape(d, global_step=0, k=2)
        if hasattr(sw, "_writer"):
            sw._writer.close()

    def test_add_betti_curve(self):
        pytest.importorskip("torch.utils.tensorboard")
        sw = DiagramSummaryWriter(log_dir="/tmp/test_tb_betti")
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        sw.add_betti_curve(d, global_step=0, num_samples=5)
        if hasattr(sw, "_writer"):
            sw._writer.close()

    def test_add_diagram_stats(self):
        pytest.importorskip("torch.utils.tensorboard")
        sw = DiagramSummaryWriter(log_dir="/tmp/test_tb_stats")
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        sw.add_diagram_stats(d, global_step=0)
        if hasattr(sw, "_writer"):
            sw._writer.close()

    def test_getattr_delegates_to_writer(self):
        pytest.importorskip("torch.utils.tensorboard")
        sw = DiagramSummaryWriter(log_dir="/tmp/test_tb_attr")
        # getattr should delegate to internal SummaryWriter
        assert hasattr(sw, "add_scalar")
        if hasattr(sw, "_writer"):
            sw._writer.close()

    def test_context_manager(self):
        pytest.importorskip("torch.utils.tensorboard")
        with DiagramSummaryWriter(log_dir="/tmp/test_tb_ctx") as sw:
            d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
            sw.add_diagram(d, global_step=0)
