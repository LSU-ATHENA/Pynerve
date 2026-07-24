"""Tests for torch/_statistics_core.py — statistical descriptors for persistence diagrams."""

from __future__ import annotations

import math

import pytest
import torch

from pynerve.torch._statistics_core import (
    _batch_or_scalar,
    _split_finite_persistence,
    _valid_rows,
    _validate_stat_diagram,
    amplitude,
    betti_curve,
    betti_numbers_at_scale,
    max_persistence,
    mean_persistence,
    number_of_features,
    persistence_entropy,
    persistence_variance,
    total_persistence,
)


# _validate_stat_diagram 


class TestValidateStatDiagram:
    def test_valid_2d(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 1]], dtype=torch.float32)
        result = _validate_stat_diagram(d)
        assert result is d

    def test_valid_3d(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        result = _validate_stat_diagram(d)
        assert result is d

    def test_1d_raises(self):
        d = torch.tensor([1.0, 2.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="2D or 3D"):
            _validate_stat_diagram(d)

    def test_wrong_dtype_raises(self):
        d = torch.tensor([[0, 1, 0]], dtype=torch.int64)
        with pytest.raises(TypeError, match="floating-point"):
            _validate_stat_diagram(d)

    def test_birth_not_finite_raises(self):
        d = torch.tensor([[float("nan"), 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="births must be finite"):
            _validate_stat_diagram(d)

    def test_death_nan_raises(self):
        d = torch.tensor([[0.0, float("nan"), 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="NaN"):
            _validate_stat_diagram(d)

    def test_death_less_than_birth_raises(self):
        d = torch.tensor([[5.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="greater than or equal"):
            _validate_stat_diagram(d)

    def test_empty_returns_self(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = _validate_stat_diagram(d)
        assert result.numel() == 0

    def test_two_column_diagram(self):
        d = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        result = _validate_stat_diagram(d)
        assert result is d


# _valid_rows 


class TestValidRows:
    def test_2d_all_valid(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=torch.float32)
        result = _valid_rows(d)
        assert result.shape[0] == 2

    def test_2d_trailing_zeros_removed(self):
        d = torch.tensor([[0.0, 1.0, 0], [0.0, 0.0, 0]], dtype=torch.float32)
        result = _valid_rows(d)
        assert result.shape[0] == 1

    def test_2d_all_zeros(self):
        d = torch.tensor([[0.0, 0.0, 0], [0.0, 0.0, 0]], dtype=torch.float32)
        result = _valid_rows(d)
        assert result.shape[0] == 0

    def test_3d_batched(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0], [0.0, 0.0, 0]],
             [[1.0, 2.0, 0], [2.0, 3.0, 0]]],
            dtype=torch.float32,
        )
        result = _valid_rows(d)
        assert result.dim() == 2

    def test_empty(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = _valid_rows(d)
        assert result.shape[0] == 0


# _split_finite_persistence 


class TestSplitFinitePersistence:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = _split_finite_persistence(d)
        assert result.shape[0] == 2
        assert torch.allclose(result, torch.tensor([1.0, 2.0], dtype=torch.float32))

    def test_dimension_filter(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 1]], dtype=torch.float32)
        result = _split_finite_persistence(d, dim=0)
        assert result.shape[0] == 1
        assert torch.allclose(result, torch.tensor([1.0], dtype=torch.float32))

    def test_infinite_death_raises(self):
        d = torch.tensor([[0.0, float("inf"), 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="finite deaths"):
            _split_finite_persistence(d)

    def test_empty(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = _split_finite_persistence(d)
        assert result.numel() == 0


# _batch_or_scalar 


class TestBatchOrScalar:
    def test_wraps_to_handle_3d(self):
        @_batch_or_scalar
        def sum_persistence(d):
            return (d[:, 1] - d[:, 0]).sum()

        d2d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        d3d = torch.stack([d2d, d2d])

        r2d = sum_persistence(d2d)
        r3d = sum_persistence(d3d)
        assert r2d.dim() == 0
        assert r3d.dim() == 1
        assert r3d.shape[0] == 2


# total_persistence 


class TestTotalPersistence:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = total_persistence(d)
        assert result.item() == 3.0  # 1 + 2

    def test_with_power(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = total_persistence(d, p=2.0)
        assert result.item() == 1.0

    def test_empty_returns_zero(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = total_persistence(d)
        assert result.item() == 0.0

    def test_batched_3d(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[1.0, 3.0, 0]]], dtype=torch.float32
        )
        result = total_persistence(d)
        assert result.dim() == 1
        assert result.shape[0] == 2

    def test_dimension_filter(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 1]], dtype=torch.float32)
        result = total_persistence(d, dim=0)
        assert result.item() == 1.0


# mean_persistence 


class TestMeanPersistence:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = mean_persistence(d)
        assert result.item() == 1.5

    def test_empty_returns_zero(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = mean_persistence(d)
        assert result.item() == 0.0


# max_persistence 


class TestMaxPersistence:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 10.0, 0]], dtype=torch.float32)
        result = max_persistence(d)
        assert result.item() == 9.0

    def test_empty_returns_zero(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = max_persistence(d)
        assert result.item() == 0.0


# persistence_variance 


class TestPersistenceVariance:
    def test_basic(self):
        d = torch.tensor([[0.0, 2.0, 0], [0.0, 4.0, 0]], dtype=torch.float32)
        result = persistence_variance(d)
        assert result.item() == 1.0  # var([2, 4], unbiased=False)

    def test_single_point_returns_zero(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = persistence_variance(d)
        assert result.item() == 0.0

    def test_empty_returns_zero(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = persistence_variance(d)
        assert result.item() == 0.0


# persistence_entropy 


class TestPersistenceEntropy:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = persistence_entropy(d)
        assert result.item() > 0

    def test_empty_returns_zero(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = persistence_entropy(d)
        assert result.item() == 0.0

    def test_uniform_distribution(self):
        d = torch.tensor([[0.0, 1.0, 0], [0.0, 1.0, 0], [0.0, 1.0, 0]], dtype=torch.float32)
        result = persistence_entropy(d)
        assert result.item() > 0

    def test_invalid_base(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="base"):
            persistence_entropy(d, base=1.0)

    def test_dimension_filter(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 1]], dtype=torch.float32)
        result0 = persistence_entropy(d, dim=0)
        result1 = persistence_entropy(d, dim=1)
        assert result0.item() >= 0  # single-point entropy is zero
        assert result1.item() >= 0


# number_of_features 


class TestNumberOfFeatures:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0], [2.0, 2.5, 0]], dtype=torch.float32)
        result = number_of_features(d)
        assert result.item() == 3

    def test_with_min_persistence(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 10.0, 0]], dtype=torch.float32)
        result = number_of_features(d, min_persistence=5.0)
        assert result.item() == 1

    def test_dimension_filter(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 1]], dtype=torch.float32)
        result = number_of_features(d, dim=0)
        assert result.item() == 1

    def test_empty_returns_zero(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = number_of_features(d)
        assert result.item() == 0

    def test_infinite_death_included(self):
        d = torch.tensor([[0.0, float("inf"), 0]], dtype=torch.float32)
        result = number_of_features(d)
        assert result.item() == 1


# betti_numbers_at_scale 


class TestBettiNumbersAtScale:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = betti_numbers_at_scale(d, threshold=0.5)
        assert result.item() == 2


# betti_curve 


class TestBettiCurve:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 5.0, 0]], dtype=torch.float32)
        result = betti_curve(d, num_samples=10)
        assert result.shape[0] == 10
        # At threshold 0, all features included; at max, none
        assert result[0].item() > 0
        assert result[-1].item() >= 0

    def test_empty(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = betti_curve(d, num_samples=5)
        assert torch.all(result == 0)

    def test_batched_3d(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[1.0, 3.0, 0]]], dtype=torch.float32
        )
        result = betti_curve(d, num_samples=5)
        assert result.dim() == 2
        assert result.shape == (2, 5)

    def test_invalid_num_samples_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises((ValueError, TypeError)):
            betti_curve(d, num_samples=0)


# amplitude 


class TestAmplitude:
    def test_persistence_amplitude(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = amplitude(d, metric="persistence")
        assert result.item() == 5.0  # 1^2 + 2^2 = 5

    def test_bottleneck_amplitude(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 10.0, 0]], dtype=torch.float32)
        result = amplitude(d, metric="bottleneck")
        assert result.item() == 9.0

    def test_wasserstein_amplitude(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = amplitude(d, metric="wasserstein", p=2.0)
        assert result.item() > 0

    def test_empty_returns_zero(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = amplitude(d)
        assert result.item() == 0.0

    def test_invalid_metric_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="metric"):
            amplitude(d, metric="bad")  # type: ignore[arg-type]

    def test_dimension_filter(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 1]], dtype=torch.float32)
        result = amplitude(d, dim=0, metric="persistence")
        assert result.item() == 1.0
