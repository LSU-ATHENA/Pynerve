"""Tests for torch/training_utils_impl.py -- loss functions and validators."""

from __future__ import annotations

import pytest
import torch

from pynerve.torch.training_utils_impl import (
    DiagramDistanceLoss,
    PersistenceCrossEntropy,
    TopologicalRegularization,
    _validate_finite_mapping,
    _validate_finite_scalar,
    _validate_nonnegative_finite,
)


# _validate_finite_scalar 


class TestValidateFiniteScalar:
    def test_valid(self):
        assert _validate_finite_scalar(3.14, "x") == 3.14

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            _validate_finite_scalar(float("nan"), "x")

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            _validate_finite_scalar(float("inf"), "x")


# _validate_nonnegative_finite 


class TestValidateNonnegativeFinite:
    def test_valid(self):
        assert _validate_nonnegative_finite(0.0, "x") == 0.0
        assert _validate_nonnegative_finite(42.0, "x") == 42.0

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_nonnegative_finite(-1.0, "x")

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            _validate_nonnegative_finite(float("nan"), "x")


# _validate_finite_mapping 


class TestValidateFiniteMapping:
    def test_valid(self):
        result = _validate_finite_mapping({"a": 1.0, "b": 2.0}, "x")
        assert result == {"a": 1.0, "b": 2.0}

    def test_none_returns_empty(self):
        result = _validate_finite_mapping(None, "x")
        assert result == {}

    def test_empty_returns_empty(self):
        result = _validate_finite_mapping({}, "x")
        assert result == {}

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            _validate_finite_mapping({"a": float("inf")}, "x")

    def test_nonnegative_mode(self):
        result = _validate_finite_mapping({"a": 0.0, "b": 5.0}, "x", nonnegative=True)
        assert result == {"a": 0.0, "b": 5.0}

    def test_nonnegative_mode_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_finite_mapping({"a": -1.0}, "x", nonnegative=True)


# DiagramDistanceLoss 


class TestDiagramDistanceLoss:
    def test_construction_wasserstein(self):
        loss = DiagramDistanceLoss(metric="wasserstein", p=2.0)
        assert loss.metric == "wasserstein"
        assert loss.p == 2.0

    def test_construction_bottleneck(self):
        loss = DiagramDistanceLoss(metric="bottleneck")
        assert loss.metric == "bottleneck"

    def test_construction_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            DiagramDistanceLoss(metric="euclidean")  # type: ignore[arg-type]

    def test_construction_p_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            DiagramDistanceLoss(p=0.0)

    def test_construction_p_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            DiagramDistanceLoss(p=-1.0)

    def test_construction_invalid_reduction_raises(self):
        with pytest.raises(ValueError, match="reduction"):
            DiagramDistanceLoss(reduction="max")  # type: ignore[arg-type]

    def test_construction_valid_reductions(self):
        for r in ("mean", "sum", "none"):
            loss = DiagramDistanceLoss(reduction=r)
            assert loss.reduction == r


# TopologicalRegularization 


class TestTopologicalRegularization:
    def test_construction_defaults(self):
        reg = TopologicalRegularization()
        assert reg.target == {}
        assert reg.weights == {}
        assert reg.penalty_type == "l2"

    def test_construction_with_targets(self):
        reg = TopologicalRegularization(
            target_complexity={"h0_count": 10.0, "mean_persistence": 5.0}
        )
        assert "h0_count" in reg.target

    def test_construction_invalid_penalty_type_raises(self):
        with pytest.raises(ValueError, match="penalty_type"):
            TopologicalRegularization(penalty_type="bad")  # type: ignore[arg-type]

    def test_construction_penalty_types(self):
        for pt in ("l1", "l2", "smooth"):
            reg = TopologicalRegularization(penalty_type=pt)
            assert reg.penalty_type == pt

    def test_penalty_l2(self):
        reg = TopologicalRegularization(penalty_type="l2")
        result = reg._penalty(torch.tensor(5.0), 3.0)
        assert result.item() == 4.0  # (5-3)^2

    def test_penalty_l1(self):
        reg = TopologicalRegularization(penalty_type="l1")
        result = reg._penalty(torch.tensor(5.0), 3.0)
        assert result.item() == 2.0  # |5-3|

    def test_penalty_smooth_near(self):
        reg = TopologicalRegularization(penalty_type="smooth")
        result = reg._penalty(torch.tensor(3.5), 3.0)
        assert result.item() == 0.5 * 0.25  # 0.5 * diff^2 when |diff| < 1

    def test_penalty_smooth_far(self):
        reg = TopologicalRegularization(penalty_type="smooth")
        result = reg._penalty(torch.tensor(5.0), 3.0)
        assert result.item() == 1.5  # |diff| - 0.5 when |diff| >= 1

    def test_forward_no_targets_returns_zero(self):
        reg = TopologicalRegularization()
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = reg(d)
        assert result.item() == 0.0

    def test_forward_with_h0_count(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=torch.float32)
        reg = TopologicalRegularization(
            target_complexity={"h0_count": 100.0}, weights={"h0_count": 1.0}
        )
        result = reg(d)
        # 2 features vs target 100 -> l2 penalty: (2-100)^2 = 9604
        assert result.item() == pytest.approx(9604.0)


# PersistenceCrossEntropy 


class TestPersistenceCrossEntropy:
    def test_construction_defaults(self):
        loss = PersistenceCrossEntropy()
        assert loss.confidence_weighting is True
        assert loss.min_threshold == 0.0

    def test_construction_no_weighting(self):
        loss = PersistenceCrossEntropy(confidence_weighting=False)
        assert loss.confidence_weighting is False

    def test_construction_custom_threshold(self):
        loss = PersistenceCrossEntropy(min_persistence_threshold=0.5)
        assert loss.min_threshold == 0.5

    def test_construction_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            PersistenceCrossEntropy(min_persistence_threshold=-1.0)

    def test_construction_invalid_reduction_raises(self):
        with pytest.raises(ValueError, match="reduction"):
            PersistenceCrossEntropy(reduction="max")  # type: ignore[arg-type]

    def test_construction_unknown_base_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            PersistenceCrossEntropy(base_loss="mse")  # type: ignore[arg-type]

    def test_forward_basic_no_diagrams(self):
        loss = PersistenceCrossEntropy(confidence_weighting=False)
        logits = torch.tensor([[0.5, 0.5, 0.0], [0.1, 0.1, 0.8]], dtype=torch.float32)
        targets = torch.tensor([0, 2])
        result = loss(logits, targets)
        assert result.dim() == 0  # scalar

    def test_forward_with_diagrams_no_confidence(self):
        loss = PersistenceCrossEntropy(confidence_weighting=False)
        logits = torch.randn(4, 3, dtype=torch.float32)
        targets = torch.tensor([0, 1, 2, 0])
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[1.0, 2.0, 0]], [[2.0, 3.0, 0]], [[3.0, 4.0, 0]]],
            dtype=torch.float32,
        )
        result = loss(logits, targets, diagrams=d)
        assert result.dim() == 0

    def test_forward_reduction_none(self):
        loss = PersistenceCrossEntropy(confidence_weighting=False, reduction="none")
        logits = torch.tensor([[0.5, 0.5], [0.2, 0.8]], dtype=torch.float32)
        targets = torch.tensor([0, 1])
        result = loss(logits, targets)
        assert result.shape == (2,)

    def test_forward_reduction_sum(self):
        loss = PersistenceCrossEntropy(confidence_weighting=False, reduction="sum")
        logits = torch.tensor([[0.5, 0.5], [0.2, 0.8]], dtype=torch.float32)
        targets = torch.tensor([0, 1])
        result = loss(logits, targets)
        assert result.dim() == 0
