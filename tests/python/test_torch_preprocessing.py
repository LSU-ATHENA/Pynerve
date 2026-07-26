"""Tests for torch/preprocessing.py -- diagram preprocessing utilities."""

from __future__ import annotations

import pytest
import torch

from pynerve.torch.preprocessing import (
    _pad_diagrams,
    clean_diagram,
    handle_infinite_deaths,
    normalize_diagram,
    remove_outliers,
    subsample_diagram,
    threshold_diagram,
)


# _pad_diagrams 


class TestPadDiagrams:
    def test_uniform_already_padded(self):
        d1 = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=torch.float32)
        d2 = torch.tensor([[0.0, 3.0, 0], [2.0, 4.0, 0]], dtype=torch.float32)
        result = _pad_diagrams([d1, d2])
        assert result.shape == (2, 2, 3)

    def test_uneven_padding(self):
        d1 = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        d2 = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0], [2.0, 3.0, 0]], dtype=torch.float32)
        result = _pad_diagrams([d1, d2])
        assert result.shape == (2, 3, 3)
        # First diagram's extra rows should be zeros
        assert torch.all(result[0, 1:] == 0)

    def test_all_empty(self):
        d1 = torch.empty((0, 3), dtype=torch.float32)
        d2 = torch.empty((0, 3), dtype=torch.float32)
        result = _pad_diagrams([d1, d2])
        assert result.shape == (2, 0, 3)

    def test_mixed_empty_nonempty(self):
        d1 = torch.empty((0, 3), dtype=torch.float32)
        d2 = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = _pad_diagrams([d1, d2])
        assert result.shape == (2, 1, 3)
        assert torch.all(result[0] == 0)  # empty diagram padded to match

    def test_single_diagram(self):
        d1 = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=torch.float32)
        result = _pad_diagrams([d1])
        assert result.shape == (1, 2, 3)

    def test_different_col_counts(self):
        d1 = torch.tensor([[0.0, 1.0, 0, 5.0]], dtype=torch.float32)
        d2 = torch.tensor([[0.0, 1.0, 0, 5.0]], dtype=torch.float32)
        result = _pad_diagrams([d1, d2])
        assert result.shape[-1] == 4


# handle_infinite_deaths 


class TestHandleInfiniteDeaths:
    def test_no_infinite_leaves_unchanged(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 1]], dtype=torch.float32)
        result = handle_infinite_deaths(d, strategy="max")
        assert torch.allclose(result, d)

    def test_strategy_max_replaces_inf(self):
        d = torch.tensor([[0.0, 1.0, 0], [2.0, float("inf"), 0]], dtype=torch.float32)
        result = handle_infinite_deaths(d, strategy="max")
        assert torch.isfinite(result[:, 1]).all()
        assert result[1, 1] > 1.0  # max_finite * factor

    def test_strategy_max_all_infinite(self):
        """When all deaths are infinite, fall back to births."""
        d = torch.tensor([[0.0, float("inf"), 0], [2.0, float("inf"), 0]], dtype=torch.float32)
        result = handle_infinite_deaths(d, strategy="max")
        assert torch.isfinite(result[:, 1]).all()

    def test_strategy_remove(self):
        d = torch.tensor([[0.0, 1.0, 0], [2.0, float("inf"), 0]], dtype=torch.float32)
        result = handle_infinite_deaths(d, strategy="remove")
        assert result.shape[0] == 1
        assert result[0, 1] == 1.0

    def test_strategy_large_value(self):
        d = torch.tensor([[0.0, float("inf"), 0]], dtype=torch.float32)
        result = handle_infinite_deaths(d, strategy="large_value", large_value_factor=100.0)
        assert result[0, 1] == 100.0

    def test_large_value_too_small_raises(self):
        d = torch.tensor([[50.0, float("inf"), 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="large_value_factor"):
            handle_infinite_deaths(d, strategy="large_value", large_value_factor=10.0)

    def test_invalid_strategy_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="strategy"):
            handle_infinite_deaths(d, strategy="bad")  # type: ignore[arg-type]

    def test_batched_3d(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0], [2.0, float("inf"), 0]],
             [[1.0, 3.0, 1], [4.0, float("inf"), 1]]],
            dtype=torch.float32,
        )
        result = handle_infinite_deaths(d, strategy="max")
        assert result.dim() == 3
        assert torch.isfinite(result[:, :, 1]).all()

    def test_large_value_factor_validation(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises((ValueError, TypeError)):
            handle_infinite_deaths(d, strategy="max", large_value_factor=-1.0)


# threshold_diagram 


class TestThresholdDiagram:
    def test_min_persistence(self):
        d = torch.tensor([[0.0, 0.1, 0], [1.0, 5.0, 0]], dtype=torch.float32)
        result = threshold_diagram(d, min_persistence=1.0)
        assert result.shape[0] == 1
        assert result[0, 1] == 5.0

    def test_max_persistence(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 10.0, 0]], dtype=torch.float32)
        result = threshold_diagram(d, max_persistence=5.0)
        assert result.shape[0] == 1
        assert result[0, 1] == 1.0

    def test_both_thresholds(self):
        d = torch.tensor([[0.0, 1.0, 0], [0.0, 3.0, 0], [0.0, 10.0, 0]], dtype=torch.float32)
        result = threshold_diagram(d, min_persistence=0.5, max_persistence=5.0)
        assert result.shape[0] == 2

    def test_infinite_death_passes_max(self):
        d = torch.tensor([[0.0, float("inf"), 0]], dtype=torch.float32)
        result = threshold_diagram(d, min_persistence=1.0)
        # inf persistence passes min_persistence
        assert result.shape[0] == 1

    def test_max_less_than_min_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="max_persistence"):
            threshold_diagram(d, min_persistence=5.0, max_persistence=1.0)

    def test_batched_3d(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0], [1.0, 10.0, 0]],
             [[0.0, 0.5, 0], [2.0, 5.0, 0]]],
            dtype=torch.float32,
        )
        result = threshold_diagram(d, min_persistence=2.0)
        assert result.dim() == 3

    def test_empty_diagram(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = threshold_diagram(d, min_persistence=1.0)
        assert result.shape[0] == 0


# normalize_diagram 


class TestNormalizeDiagram:
    def test_minmax_normalization(self):
        d = torch.tensor([[0.0, 1.0, 0], [2.0, 5.0, 0]], dtype=torch.float32)
        result = normalize_diagram(d, method="minmax")
        assert result[0, 0] == 0.0  # min birth -> 0
        assert result[1, 0] == 1.0  # max birth -> 1

    def test_standard_normalization(self):
        d = torch.tensor([[0.0, 1.0, 0], [2.0, 5.0, 0]], dtype=torch.float32)
        result = normalize_diagram(d, method="standard")
        # mean-centered, std-scaled
        assert torch.allclose(result[:, 0].mean(), torch.tensor(0.0), atol=1e-6)

    def test_none_method_returns_clone(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = normalize_diagram(d, method="none")
        assert torch.allclose(result, d)

    def test_empty_diagram(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = normalize_diagram(d, method="minmax")
        assert result.numel() == 0

    def test_custom_ranges(self):
        d = torch.tensor([[1.0, 3.0, 0]], dtype=torch.float32)
        result = normalize_diagram(d, method="minmax", birth_range=(0.0, 10.0), death_range=(0.0, 10.0))
        assert 0.0 <= result[0, 0] <= 1.0

    def test_invalid_method_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="method"):
            normalize_diagram(d, method="bad")  # type: ignore[arg-type]

    def test_batched_3d(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[2.0, 5.0, 0]]], dtype=torch.float32
        )
        result = normalize_diagram(d, method="minmax")
        assert result.dim() == 3
        assert result.shape == (2, 1, 3)

    def test_infinite_death_raises(self):
        d = torch.tensor([[0.0, float("inf"), 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="finite deaths"):
            normalize_diagram(d, method="minmax")


# subsample_diagram 


class TestSubsampleDiagram:
    def test_most_persistent(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 10.0, 0], [2.0, 3.0, 0]], dtype=torch.float32)
        result = subsample_diagram(d, max_features=2, strategy="most_persistent")
        assert result.shape[0] == 2
        # Most persistent is [1, 10] -> persistence 9
        assert torch.any((result[:, 1] - result[:, 0]) > 5.0)

    def test_uniform(self):
        d = torch.tensor(
            [[0.0, 1.0, 0], [2.0, 4.0, 0], [5.0, 7.0, 0], [8.0, 9.0, 0]],
            dtype=torch.float32,
        )
        result = subsample_diagram(d, max_features=2, strategy="uniform")
        assert result.shape[0] >= 1

    def test_random(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0], [2.0, 3.0, 0]], dtype=torch.float32)
        result = subsample_diagram(d, max_features=2, strategy="random")
        assert result.shape[0] == 2

    def test_already_smaller_than_max(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = subsample_diagram(d, max_features=5)
        assert result.shape[0] == 1

    def test_invalid_strategy_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="strategy"):
            subsample_diagram(d, max_features=1, strategy="bad")  # type: ignore[arg-type]

    def test_batched_3d(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0], [1.0, 5.0, 0]], [[2.0, 3.0, 0], [4.0, 6.0, 0]]],
            dtype=torch.float32,
        )
        result = subsample_diagram(d, max_features=1, strategy="most_persistent")
        assert result.dim() == 3

    def test_zero_features_returns_empty(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=torch.float32)
        result = subsample_diagram(d, max_features=0)
        assert result.shape[0] == 0


# remove_outliers 


class TestRemoveOutliers:
    def test_iqr_basic(self):
        d = torch.tensor(
            [[0.0, 1.0, 0], [0.5, 1.5, 0], [0.2, 1.2, 0], [10.0, 1000.0, 0]],
            dtype=torch.float32,
        )
        result = remove_outliers(d, method="iqr", threshold=1.0)
        assert result.shape[0] < 4

    def test_zscore_basic(self):
        d = torch.tensor(
            [[0.0, 1.0, 0], [0.5, 1.5, 0], [0.2, 1.2, 0],
             [0.3, 1.3, 0], [0.7, 1.7, 0], [0.4, 1.4, 0],
             [100.0, 1000.0, 0]],
            dtype=torch.float32,
        )
        result = remove_outliers(d, method="zscore", threshold=2.0)
        assert result.shape[0] == 6

    def test_isolation_forest(self):
        d = torch.tensor(
            [[0.0, 1.0, 0], [1.0, 2.0, 0], [0.5, 1.5, 0], [10.0, 100.0, 0]],
            dtype=torch.float32,
        )
        result = remove_outliers(d, method="isolation_forest")
        assert result.shape[0] > 0

    def test_invalid_method_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="method"):
            remove_outliers(d, method="bad")  # type: ignore[arg-type]

    def test_empty_diagram(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = remove_outliers(d, method="iqr")
        assert result.numel() == 0

    def test_negative_threshold_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="non-negative"):
            remove_outliers(d, threshold=-1.0)

    def test_single_point_diagram(self):
        """IQR with a single point should not crash."""
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = remove_outliers(d, method="iqr")
        assert result.shape[0] <= 1

    def test_two_point_diagram(self):
        """Z-score with two points should compute normally."""
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=torch.float32)
        result = remove_outliers(d, method="zscore", threshold=3.0)
        assert result.shape[0] >= 1


# clean_diagram 


class TestCleanDiagram:
    def test_full_pipeline(self):
        d = torch.tensor(
            [[0.0, 1.0, 0], [1.0, float("inf"), 0], [2.0, 3.0, 0], [0.0, 0.01, 0]],
            dtype=torch.float32,
        )
        result = clean_diagram(
            d,
            handle_inf=True,
            min_persistence=0.1,
            max_features=3,
            normalize=True,
            remove_outliers_flag=False,
        )
        assert result.dim() == 2
        # Infinite deaths handled, tiny persistence filtered
        assert torch.isfinite(result[:, 1]).all()

    def test_no_ops_returns_clone(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = clean_diagram(
            d, handle_inf=False, min_persistence=0.0, normalize=False
        )
        assert torch.allclose(result, d)

    def test_batched_3d(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0], [1.0, float("inf"), 0]],
             [[2.0, 3.0, 0], [0.0, 0.01, 0]]],
            dtype=torch.float32,
        )
        result = clean_diagram(
            d, handle_inf=True, min_persistence=0.1, normalize=True
        )
        assert result.dim() == 3
        assert torch.isfinite(result[:, :, 1]).all()

    def test_empty_batched(self):
        d = torch.empty((0, 2, 3), dtype=torch.float32)
        result = clean_diagram(d)
        assert result.dim() == 3

    def test_with_outlier_removal(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=torch.float32)
        result = clean_diagram(d, remove_outliers_flag=True)
        assert result.dim() == 2

    def test_invalid_dim_raises(self):
        d = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="2D or 3D"):
            clean_diagram(d)

    def test_all_options_combined(self):
        """Pipeline with all options enabled simultaneously."""
        d = torch.tensor(
            [[0.0, 1.0, 0], [1.0, float("inf"), 0], [2.0, 3.0, 0], [0.0, 0.01, 0]],
            dtype=torch.float32,
        )
        result = clean_diagram(
            d,
            handle_inf=True,
            min_persistence=0.1,
            max_features=2,
            normalize=True,
            remove_outliers_flag=True,
        )
        assert result.dim() == 2
        assert torch.isfinite(result[:, 1]).all()
        assert result.shape[0] <= 2

    def test_single_point_clean(self):
        """Clean a single-point diagram."""
        d = torch.tensor([[0.0, 5.0, 0]], dtype=torch.float32)
        result = clean_diagram(d, handle_inf=True, normalize=True)
        assert result.shape[0] == 1
        assert result[0, 0] == 0.0  # minmax: min birth goes to 0

    def test_large_value_factor_validation(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="positive"):
            handle_infinite_deaths(d, strategy="max", large_value_factor=-1.0)

    def test_handle_inf_strategy_remove_all_infinite(self):
        """When all deaths are infinite and strategy=remove, result is empty."""
        d = torch.tensor([[0.0, float("inf"), 0], [2.0, float("inf"), 0]], dtype=torch.float32)
        result = handle_infinite_deaths(d, strategy="remove")
        assert result.shape[0] == 0

    def test_subsample_uniform_clustered_births(self):
        """Uniform subsample on diagrams with clustered births."""
        d = torch.tensor(
            [[0.0, 1.0, 0], [0.1, 2.0, 0], [0.2, 1.5, 0], [5.0, 10.0, 0]],
            dtype=torch.float32,
        )
        result = subsample_diagram(d, max_features=2, strategy="uniform")
        assert result.shape[0] >= 1

    def test_normalize_single_point_standard(self):
        """Standard normalization on a single-point diagram."""
        d = torch.tensor([[5.0, 10.0, 0]], dtype=torch.float32)
        result = normalize_diagram(d, method="standard")
        # With a single point, std clamped to EPS, so result is near 0
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)

    def test_normalize_extreme_values(self):
        """Minmax normalization with extreme birth/death values."""
        d = torch.tensor([[0.0, 1.0, 0], [1e6, 2e6, 0]], dtype=torch.float32)
        result = normalize_diagram(d, method="minmax")
        assert result[0, 0] == 0.0
        assert result[1, 0] == 1.0
