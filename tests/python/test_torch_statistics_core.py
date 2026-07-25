"""Tests for torch/_statistics_core.py — statistical operators for persistence diagrams."""

from __future__ import annotations

import math

import pytest
import torch
from _test_helpers import make_diag_2d, make_diag_3d

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")

torch = pytest.importorskip("torch")


def _batched_diag(batch=3, n=5):
    """Create batched diagrams (batch, n, 2)."""
    births = torch.rand(batch, n) * 0.5
    deaths = births + torch.rand(batch, n) * 0.5 + 0.01
    return torch.stack([births, deaths], dim=-1)


class TestValidateStatDiagram:
    def test_valid_2d(self):
        from pynerve.torch._statistics_core import _validate_stat_diagram
        _validate_stat_diagram(make_diag_2d(5))

    def test_valid_3d(self):
        from pynerve.torch._statistics_core import _validate_stat_diagram
        _validate_stat_diagram(_batched_diag(2, 5))

    def test_wrong_dim(self):
        from pynerve.torch._statistics_core import _validate_stat_diagram
        with pytest.raises(ValueError, match="2D or 3D"):
            _validate_stat_diagram(torch.rand(2, 3, 4, 5))

    def test_too_few_cols(self):
        from pynerve.torch._statistics_core import _validate_stat_diagram
        with pytest.raises(ValueError, match="birth and death"):
            _validate_stat_diagram(torch.rand(5, 1))

    def test_not_floating(self):
        from pynerve.torch._statistics_core import _validate_stat_diagram
        with pytest.raises(TypeError, match="floating"):
            _validate_stat_diagram(torch.zeros(3, 2, dtype=torch.int32))

    def test_nan_births(self):
        from pynerve.torch._statistics_core import _validate_stat_diagram
        d = torch.tensor([[float("nan"), 1.0]])
        with pytest.raises(ValueError, match="births"):
            _validate_stat_diagram(d)

    def test_nan_deaths(self):
        from pynerve.torch._statistics_core import _validate_stat_diagram
        d = torch.tensor([[0.0, float("nan")]])
        with pytest.raises(ValueError, match="deaths"):
            _validate_stat_diagram(d)

    def test_death_before_birth(self):
        from pynerve.torch._statistics_core import _validate_stat_diagram
        d = torch.tensor([[1.0, 0.5]])
        with pytest.raises(ValueError, match="deaths"):
            _validate_stat_diagram(d)

    def test_empty(self):
        from pynerve.torch._statistics_core import _validate_stat_diagram
        result = _validate_stat_diagram(torch.empty(0, 2))
        assert result.numel() == 0


class TestTotalPersistence:
    def test_basic(self):
        from pynerve.torch._statistics_core import total_persistence
        d = make_diag_2d(5)
        result = total_persistence(d)
        assert result > 0

    def test_p2(self):
        from pynerve.torch._statistics_core import total_persistence
        d = make_diag_2d(5)
        result = total_persistence(d, p=2.0)
        assert result > 0

    def test_empty(self):
        from pynerve.torch._statistics_core import total_persistence
        result = total_persistence(torch.empty(0, 2))
        assert result.item() == 0.0

    def test_batched(self):
        from pynerve.torch._statistics_core import total_persistence
        d = _batched_diag(3, 5)
        result = total_persistence(d)
        assert result.shape == (3,)
        assert (result > 0).all()

    def test_with_dim(self):
        from pynerve.torch._statistics_core import total_persistence
        d = make_diag_3d(5)
        result = total_persistence(d, dim=0)
        assert result >= 0

    def test_invalid_p(self):
        from pynerve.torch._statistics_core import total_persistence
        with pytest.raises(ValueError, match="positive|finite"):
            total_persistence(make_diag_2d(3), p=0.0)


class TestMeanPersistence:
    def test_basic(self):
        from pynerve.torch._statistics_core import mean_persistence
        d = make_diag_2d(5)
        result = mean_persistence(d)
        assert result > 0

    def test_empty(self):
        from pynerve.torch._statistics_core import mean_persistence
        result = mean_persistence(torch.empty(0, 2))
        assert result.item() == 0.0

    def test_batched(self):
        from pynerve.torch._statistics_core import mean_persistence
        d = _batched_diag(3, 5)
        result = mean_persistence(d)
        assert result.shape == (3,)


class TestMaxPersistence:
    def test_basic(self):
        from pynerve.torch._statistics_core import max_persistence
        d = make_diag_2d(5)
        result = max_persistence(d)
        assert result > 0

    def test_empty(self):
        from pynerve.torch._statistics_core import max_persistence
        result = max_persistence(torch.empty(0, 2))
        assert result.item() == 0.0

    def test_batched(self):
        from pynerve.torch._statistics_core import max_persistence
        d = _batched_diag(3, 5)
        result = max_persistence(d)
        assert result.shape == (3,)


class TestPersistenceVariance:
    def test_basic(self):
        from pynerve.torch._statistics_core import persistence_variance
        d = make_diag_2d(10)
        result = persistence_variance(d)
        assert result >= 0

    def test_single_feature(self):
        from pynerve.torch._statistics_core import persistence_variance
        d = make_diag_2d(1)
        result = persistence_variance(d)
        assert result.item() == 0.0

    def test_empty(self):
        from pynerve.torch._statistics_core import persistence_variance
        result = persistence_variance(torch.empty(0, 2))
        assert result.item() == 0.0

    def test_batched(self):
        from pynerve.torch._statistics_core import persistence_variance
        d = _batched_diag(3, 10)
        result = persistence_variance(d)
        assert result.shape == (3,)


class TestPersistenceEntropy:
    def test_basic(self):
        from pynerve.torch._statistics_core import persistence_entropy
        d = make_diag_2d(10)
        result = persistence_entropy(d)
        assert result >= 0

    def test_base2(self):
        from pynerve.torch._statistics_core import persistence_entropy
        d = make_diag_2d(10)
        result = persistence_entropy(d, base=2.0)
        assert result >= 0

    def test_empty(self):
        from pynerve.torch._statistics_core import persistence_entropy
        result = persistence_entropy(torch.empty(0, 2))
        assert result.item() == 0.0

    def test_batched(self):
        from pynerve.torch._statistics_core import persistence_entropy
        d = _batched_diag(3, 10)
        result = persistence_entropy(d)
        assert result.shape == (3,)

    def test_invalid_base(self):
        from pynerve.torch._statistics_core import persistence_entropy
        with pytest.raises(ValueError, match="base"):
            persistence_entropy(make_diag_2d(3), base=1.0)

    def test_invalid_base_zero(self):
        from pynerve.torch._statistics_core import persistence_entropy
        with pytest.raises(ValueError, match="positive|finite"):
            persistence_entropy(make_diag_2d(3), base=0.0)


class TestNumberOfFeatures:
    def test_basic(self):
        from pynerve.torch._statistics_core import number_of_features
        d = make_diag_2d(10)
        result = number_of_features(d)
        assert result.item() == 10

    def test_with_threshold(self):
        from pynerve.torch._statistics_core import number_of_features
        d = make_diag_2d(10)
        result = number_of_features(d, min_persistence=100.0)
        assert result.item() == 0

    def test_with_dim(self):
        from pynerve.torch._statistics_core import number_of_features
        d = make_diag_3d(10)
        result = number_of_features(d, dim=0)
        assert result >= 0

    def test_empty(self):
        from pynerve.torch._statistics_core import number_of_features
        result = number_of_features(torch.empty(0, 2))
        assert result.item() == 0

    def test_batched(self):
        from pynerve.torch._statistics_core import number_of_features
        d = _batched_diag(3, 5)
        result = number_of_features(d)
        assert result.shape == (3,)

    def test_negative_threshold(self):
        from pynerve.torch._statistics_core import number_of_features
        with pytest.raises(ValueError, match="non-negative"):
            number_of_features(make_diag_2d(3), min_persistence=-1.0)


class TestBettiNumbersAtScale:
    def test_basic(self):
        from pynerve.torch._statistics_core import betti_numbers_at_scale
        d = make_diag_2d(10)
        result = betti_numbers_at_scale(d, threshold=0.0)
        assert result.item() == 10

    def test_high_threshold(self):
        from pynerve.torch._statistics_core import betti_numbers_at_scale
        d = make_diag_2d(10)
        result = betti_numbers_at_scale(d, threshold=100.0)
        assert result.item() == 0

    def test_with_dim(self):
        from pynerve.torch._statistics_core import betti_numbers_at_scale
        d = make_diag_3d(10)
        result = betti_numbers_at_scale(d, threshold=0.0, dim=0)
        assert result >= 0


class TestBettiCurve:
    def test_basic(self):
        from pynerve.torch._statistics_core import betti_curve
        d = make_diag_2d(10)
        result = betti_curve(d, num_samples=50)
        assert result.shape == (50,)

    def test_empty(self):
        from pynerve.torch._statistics_core import betti_curve
        result = betti_curve(torch.empty(0, 2), num_samples=20)
        assert result.shape == (20,)
        assert (result == 0).all()

    def test_batched(self):
        from pynerve.torch._statistics_core import betti_curve
        d = _batched_diag(3, 5)
        result = betti_curve(d, num_samples=20)
        assert result.shape == (3, 20)

    def test_invalid_num_samples(self):
        from pynerve.torch._statistics_core import betti_curve
        with pytest.raises(ValueError, match="positive"):
            betti_curve(make_diag_2d(3), num_samples=0)


class TestAmplitude:
    def test_persistence(self):
        from pynerve.torch._statistics_core import amplitude
        d = make_diag_2d(5)
        result = amplitude(d, metric="persistence", p=2.0)
        assert result > 0

    def test_bottleneck(self):
        from pynerve.torch._statistics_core import amplitude
        d = make_diag_2d(5)
        result = amplitude(d, metric="bottleneck")
        assert result > 0

    def test_wasserstein(self):
        from pynerve.torch._statistics_core import amplitude
        d = make_diag_2d(5)
        result = amplitude(d, metric="wasserstein", p=2.0)
        assert result > 0

    def test_empty(self):
        from pynerve.torch._statistics_core import amplitude
        result = amplitude(torch.empty(0, 2), metric="persistence")
        assert result.item() == 0.0

    def test_batched(self):
        from pynerve.torch._statistics_core import amplitude
        d = _batched_diag(3, 5)
        result = amplitude(d, metric="persistence")
        assert result.shape == (3,)

    def test_invalid_metric(self):
        from pynerve.torch._statistics_core import amplitude
        with pytest.raises(ValueError, match="metric"):
            amplitude(make_diag_2d(3), metric="bad")

    def test_invalid_p(self):
        from pynerve.torch._statistics_core import amplitude
        with pytest.raises(ValueError, match="positive|finite"):
            amplitude(make_diag_2d(3), p=0.0)

    def test_p1(self):
        from pynerve.torch._statistics_core import amplitude
        d = make_diag_2d(5)
        result = amplitude(d, metric="persistence", p=1.0)
        assert result > 0


class TestValidRows:
    def test_basic(self):
        from pynerve.torch._statistics_core import _valid_rows
        d = make_diag_2d(5)
        result = _valid_rows(d)
        assert result.shape[0] == 5

    def test_with_zeros(self):
        from pynerve.torch._statistics_core import _valid_rows
        # _valid_rows returns rows up to the last non-zero row (not just non-zero rows)
        d = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.5, 0.8]])
        result = _valid_rows(d)
        # The last non-zero row is index 2, so all 3 rows are returned
        assert result.shape[0] == 3

    def test_empty(self):
        from pynerve.torch._statistics_core import _valid_rows
        result = _valid_rows(torch.empty(0, 2))
        assert result.numel() == 0

    def test_all_zeros(self):
        from pynerve.torch._statistics_core import _valid_rows
        d = torch.zeros(3, 2)
        result = _valid_rows(d)
        assert result.shape[0] == 0

    def test_batched(self):
        from pynerve.torch._statistics_core import _valid_rows
        d = _batched_diag(2, 3)
        result = _valid_rows(d)
        assert result.shape[-1] == 2


class TestSplitFinitePersistence:
    def test_basic(self):
        from pynerve.torch._statistics_core import _split_finite_persistence
        d = make_diag_2d(5)
        result = _split_finite_persistence(d)
        assert result.shape == (5,)
        assert (result > 0).all()

    def test_with_dim(self):
        from pynerve.torch._statistics_core import _split_finite_persistence
        d = make_diag_3d(10)
        result = _split_finite_persistence(d, dim=0)
        assert result.shape[0] <= 10

    def test_empty(self):
        from pynerve.torch._statistics_core import _split_finite_persistence
        result = _split_finite_persistence(torch.empty(0, 2))
        assert result.numel() == 0

    def test_inf_deaths(self):
        from pynerve.torch._statistics_core import _split_finite_persistence
        d = torch.tensor([[0.0, 1.0], [0.5, float("inf")]])
        with pytest.raises(ValueError, match="finite deaths"):
            _split_finite_persistence(d)
