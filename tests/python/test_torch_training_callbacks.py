"""Tests for torch/_training_callbacks.py — TopologicalEarlyStopping and DiagramVisualizationCallback."""

from __future__ import annotations

import pytest
import torch

from pynerve.torch._diagram import PersistenceDiagram
from pynerve.torch._training_callbacks import (
    DiagramVisualizationCallback,
    TopologicalEarlyStopping,
    _validate_nonnegative_finite,
)


# ── _validate_nonnegative_finite ───────────────────────────────────────────


class TestValidateNonnegativeFinite:
    def test_valid(self):
        assert _validate_nonnegative_finite(0.0, "x") == 0.0
        assert _validate_nonnegative_finite(100.0, "x") == 100.0

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_nonnegative_finite(-0.1, "x")

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_nonnegative_finite(float("nan"), "x")


# ── DiagramVisualizationCallback ───────────────────────────────────────────


class TestDiagramVisualizationCallback:
    def test_construction_defaults(self):
        cb = DiagramVisualizationCallback()
        assert cb.log_every > 0
        assert cb.writer is None
        assert cb.step == 0

    def test_construction_custom_params(self):
        cb = DiagramVisualizationCallback(log_every=5, max_diagrams=8)
        assert cb.log_every == 5
        assert cb.max_diagrams == 8

    def test_construction_invalid_log_every_raises(self):
        with pytest.raises((ValueError, TypeError)):
            DiagramVisualizationCallback(log_every=0)

    def test_construction_invalid_max_diagrams_raises(self):
        with pytest.raises((ValueError, TypeError)):
            DiagramVisualizationCallback(max_diagrams=0)

    def test_on_batch_end_no_writer(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=torch.float32)
        cb = DiagramVisualizationCallback(log_every=1)
        cb.on_batch_end(d, batch_idx=1)

    def test_on_epoch_end_no_writer(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        cb = DiagramVisualizationCallback(log_every=1)
        cb.on_epoch_end(epoch=1, diagram=d)

    def test_on_batch_end_skips_non_log_steps(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        cb = DiagramVisualizationCallback(log_every=10)
        initial_step = cb.step
        cb.on_batch_end(d, batch_idx=5)
        assert cb.step == initial_step  # didn't increment

    def test_on_epoch_end_skips_non_log_epochs(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        cb = DiagramVisualizationCallback(log_every=10)
        cb.on_epoch_end(epoch=5, diagram=d)  # should skip

    def test_invalid_batch_idx_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        cb = DiagramVisualizationCallback()
        with pytest.raises(ValueError, match="non-negative"):
            cb.on_batch_end(d, batch_idx=-1)

    def test_invalid_epoch_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        cb = DiagramVisualizationCallback()
        with pytest.raises(ValueError, match="non-negative"):
            cb.on_epoch_end(epoch=-1, diagram=d)


# ── TopologicalEarlyStopping ───────────────────────────────────────────────


class TestTopologicalEarlyStopping:
    def test_construction_defaults(self):
        es = TopologicalEarlyStopping()
        assert es.patience == 10
        assert es.mode == "approach"
        assert es.target is None

    def test_construction_approach_with_target(self):
        es = TopologicalEarlyStopping(target_complexity=20.0, mode="approach")
        assert es.target == 20.0

    def test_construction_stabilize(self):
        es = TopologicalEarlyStopping(mode="stabilize")
        assert es.mode == "stabilize"

    def test_construction_invalid_patience_raises(self):
        with pytest.raises(ValueError, match="patience"):
            TopologicalEarlyStopping(patience=0)

    def test_construction_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            TopologicalEarlyStopping(mode="bad")  # type: ignore[arg-type]

    def test_construction_invalid_min_delta_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            TopologicalEarlyStopping(min_delta=-1.0)

    def test_approach_mode_converges(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=torch.float32)
        es = TopologicalEarlyStopping(
            patience=3, target_complexity=2.0, mode="approach", min_delta=5.0
        )
        for _ in range(4):
            stop = es(d)
        assert stop  # should converge after patience

    def test_stabilize_mode(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        es = TopologicalEarlyStopping(patience=3, mode="stabilize", min_delta=100.0)
        for _ in range(5):
            stop = es(d)
        assert stop

    def test_increase_mode(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        es = TopologicalEarlyStopping(patience=3, mode="increase", min_delta=100.0)
        for _ in range(5):
            stop = es(d)
        assert stop  # no increase → counter grows

    def test_decrease_mode(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        es = TopologicalEarlyStopping(patience=3, mode="decrease", min_delta=100.0)
        for _ in range(5):
            stop = es(d)
        assert stop

    def test_reset(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=torch.float32)
        es = TopologicalEarlyStopping(patience=5, mode="stabilize", min_delta=100.0)
        for _ in range(3):
            es(d)
        es.reset()
        assert len(es.history) == 0
        assert es.counter == 0

    def test_approach_without_target_no_op(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        es = TopologicalEarlyStopping(patience=100, mode="approach")
        # approach mode without target: counter stays 0
        result = es(d)
        assert not result

    def test_history_appended(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        es = TopologicalEarlyStopping(patience=10)
        es(d)
        assert len(es.history) == 1

    def test_persistence_diagram_input(self):
        d = torch.tensor([[[0.0, 1.0, 0], [2.0, 5.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        es = TopologicalEarlyStopping(patience=10)
        result = es(pd)
        assert isinstance(result, bool)
