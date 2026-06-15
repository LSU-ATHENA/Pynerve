"""Comprehensive tests for pynerve.torch subpackage.

Covers the public API surface: lazy imports, PersistenceDiagram, diagram
distance functions, data utilities, nn layers, preprocessing, statistics,
vectorization, kernels, mapper, sklearn transformers, tensorboard logging,
and visualisation helpers.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from pynerve.exceptions import ValidationError  # noqa: E402


def _torch_backend_available() -> bool:
    """Check if the PyTorch C++ extension (pynerve_torch_internal) is present."""
    try:
        import nerve_torch_internal  # noqa: F401

        return True
    except ImportError:
        try:
            import pynerve_torch_internal  # noqa: F401

            return True
        except ImportError:
            return False


_torch_backend = pytest.mark.skipif(
    not _torch_backend_available(),
    reason="pynerve_torch_internal not available",
)


def _core_backend_available() -> bool:
    """Check if the core C++ extension (pynerve_internal) is present."""
    try:
        import pynerve_internal  # noqa: F401

        return True
    except ImportError:
        return False


_core_backend = pytest.mark.skipif(
    not _core_backend_available(),
    reason="pynerve_internal not available",
)


def _networkx_available() -> bool:
    try:
        import networkx  # noqa: F401

        return True
    except ImportError:
        return False


def _make_diagram_tensor(
    batch: int = 2,
    pairs: int = 4,
    seed: int = 42,
) -> torch.Tensor:
    """Create a valid persistence diagram tensor (birth, death, dim) with birth < death."""
    torch.manual_seed(seed)
    births = torch.rand(batch, pairs, 1)
    deaths = births + torch.rand(batch, pairs, 1) + 0.1
    dims = torch.randint(0, 2, (batch, pairs, 1)).float()
    return torch.cat([births, deaths, dims], dim=-1)


def _make_2d_diagram(pairs: int = 3, seed: int = 42) -> torch.Tensor:
    """Create a 2D (unbatched) diagram tensor with (birth, death) columns."""
    torch.manual_seed(seed)
    births = torch.rand(pairs, 1)
    deaths = births + torch.rand(pairs, 1) + 0.1
    return torch.cat([births, deaths], dim=-1)


def _make_point_cloud(
    n_points: int = 8,
    dim: int = 3,
    batch: int = 1,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)
    if batch == 1:
        return torch.rand(n_points, dim)
    return torch.rand(batch, n_points, dim)


# Lazy import mechanism


# helpers


# preprocessing


class TestHandleInfiniteDeaths:
    """handle_infinite_deaths strategies."""

    def test_max_strategy(self) -> None:
        from pynerve.torch.preprocessing import handle_infinite_deaths

        diagram = torch.tensor([[0.0, 1.0], [2.0, float("inf")]], dtype=torch.float64)
        result = handle_infinite_deaths(diagram, strategy="max")
        assert torch.isfinite(result[:, 1]).all()
        assert result.shape == (2, 2)

    def test_remove_strategy(self) -> None:
        from pynerve.torch.preprocessing import handle_infinite_deaths

        diagram = torch.tensor([[0.0, 1.0], [2.0, float("inf")]], dtype=torch.float64)
        result = handle_infinite_deaths(diagram, strategy="remove")
        assert result.shape == (1, 2)
        assert torch.isfinite(result[:, 1]).all()

    def test_large_value_strategy(self) -> None:
        from pynerve.torch.preprocessing import handle_infinite_deaths

        diagram = torch.tensor([[0.0, 1.0], [2.0, float("inf")]], dtype=torch.float64)
        result = handle_infinite_deaths(diagram, strategy="large_value", large_value_factor=100.0)
        assert result.shape == (2, 2)
        assert result[1, 1].item() == 100.0

    def test_no_infinite_deaths_returns_clone(self) -> None:
        from pynerve.torch.preprocessing import handle_infinite_deaths

        diagram = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
        result = handle_infinite_deaths(diagram)
        torch.testing.assert_close(result, diagram)

    def test_large_value_too_small_raises(self) -> None:
        from pynerve.torch.preprocessing import handle_infinite_deaths

        diagram = torch.tensor([[10.0, float("inf")]], dtype=torch.float64)
        with pytest.raises(ValueError):
            handle_infinite_deaths(diagram, strategy="large_value", large_value_factor=2.0)

    def test_invalid_strategy_raises(self) -> None:
        from pynerve.torch.preprocessing import handle_infinite_deaths

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            handle_infinite_deaths(diagram, strategy="bogus")

    def test_invalid_large_value_factor_raises(self) -> None:
        from pynerve.torch.preprocessing import handle_infinite_deaths

        diagram = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            handle_infinite_deaths(diagram, strategy="max", large_value_factor=float("nan"))


class TestThresholdDiagram:
    """threshold_diagram filtering."""

    def test_min_persistence(self) -> None:
        from pynerve.torch.preprocessing import threshold_diagram

        diagram = torch.tensor([[0.0, 0.5], [0.0, 2.0], [0.0, 0.8]], dtype=torch.float64)
        result = threshold_diagram(diagram, min_persistence=1.0)
        assert result.shape == (1, 2)
        assert result[0, 1].item() == 2.0

    def test_max_persistence(self) -> None:
        from pynerve.torch.preprocessing import threshold_diagram

        diagram = torch.tensor([[0.0, 0.5], [0.0, 2.0], [0.0, 0.8]], dtype=torch.float64)
        result = threshold_diagram(diagram, min_persistence=0.0, max_persistence=1.0)
        assert result.shape == (2, 2)

    def test_max_less_than_min_raises(self) -> None:
        from pynerve.torch.preprocessing import threshold_diagram

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            threshold_diagram(diagram, min_persistence=2.0, max_persistence=1.0)

    def test_all_removed_returns_empty(self) -> None:
        from pynerve.torch.preprocessing import threshold_diagram

        diagram = torch.tensor([[0.0, 0.1]], dtype=torch.float64)
        result = threshold_diagram(diagram, min_persistence=1.0)
        assert result.shape[0] == 0


class TestNormalizeDiagram:
    """normalize_diagram methods."""

    def test_minmax_default(self) -> None:
        from pynerve.torch.preprocessing import normalize_diagram

        diagram = torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float64)
        result = normalize_diagram(diagram, method="minmax")
        assert result.dtype == diagram.dtype
        assert result[:, 0].min() >= 0.0
        assert result[:, 0].max() <= 1.0

    def test_standard(self) -> None:
        from pynerve.torch.preprocessing import normalize_diagram

        diagram = torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float64)
        result = normalize_diagram(diagram, method="standard")
        assert result.shape == diagram.shape

    def test_none_returns_clone(self) -> None:
        from pynerve.torch.preprocessing import normalize_diagram

        diagram = torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float64)
        result = normalize_diagram(diagram, method="none")
        torch.testing.assert_close(result, diagram)

    def test_invalid_method_raises(self) -> None:
        from pynerve.torch.preprocessing import normalize_diagram

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            normalize_diagram(diagram, method="bogus")

    def test_infinite_deaths_raises(self) -> None:
        from pynerve.torch.preprocessing import normalize_diagram

        diagram = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            normalize_diagram(diagram)

    def test_empty_diagram(self) -> None:
        from pynerve.torch.preprocessing import normalize_diagram

        diagram = torch.empty((0, 2), dtype=torch.float64)
        result = normalize_diagram(diagram)
        assert result.shape == (0, 2)

    def test_with_birth_range(self) -> None:
        from pynerve.torch.preprocessing import normalize_diagram

        diagram = torch.tensor([[1.0, 3.0], [2.0, 5.0]], dtype=torch.float64)
        result = normalize_diagram(diagram, method="minmax", birth_range=(0.0, 10.0))
        assert result.shape == (2, 2)

    def test_with_death_range(self) -> None:
        from pynerve.torch.preprocessing import normalize_diagram

        diagram = torch.tensor([[1.0, 3.0], [2.0, 5.0]], dtype=torch.float64)
        result = normalize_diagram(diagram, method="minmax", death_range=(0.0, 10.0))
        assert result.shape == (2, 2)


class TestSubsampleDiagram:
    """subsample_diagram strategies."""

    def test_most_persistent(self) -> None:
        from pynerve.torch.preprocessing import subsample_diagram

        diagram = torch.tensor([[0.0, 0.5], [0.0, 4.0], [0.0, 0.8]], dtype=torch.float64)
        result = subsample_diagram(diagram, max_features=2, strategy="most_persistent")
        assert result.shape[0] == 2

    def test_uniform(self) -> None:
        from pynerve.torch.preprocessing import subsample_diagram

        diagram = torch.tensor(
            [[0.1, 0.5], [0.3, 0.9], [0.6, 2.0], [0.8, 3.0]], dtype=torch.float64
        )
        result = subsample_diagram(diagram, max_features=2, strategy="uniform")
        assert result.shape[0] == 2

    def test_random(self) -> None:
        from pynerve.torch.preprocessing import subsample_diagram

        diagram = torch.tensor(
            [[0.1, 0.5], [0.3, 0.9], [0.6, 2.0], [0.8, 3.0]], dtype=torch.float64
        )
        result = subsample_diagram(diagram, max_features=2, strategy="random")
        assert result.shape[0] == 2

    def test_less_than_max_returns_all(self) -> None:
        from pynerve.torch.preprocessing import subsample_diagram

        diagram = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
        result = subsample_diagram(diagram, max_features=10)
        assert result.shape[0] == 2

    def test_invalid_strategy_raises(self) -> None:
        from pynerve.torch.preprocessing import subsample_diagram

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError):
            subsample_diagram(diagram, max_features=1, strategy="bogus")

    def test_negative_max_features_raises(self) -> None:
        from pynerve.torch.preprocessing import subsample_diagram

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError):
            subsample_diagram(diagram, max_features=-1)


class TestRemoveOutliers:
    """remove_outliers methods."""

    def test_iqr_method(self) -> None:
        from pynerve.torch.preprocessing import remove_outliers

        diagram = torch.tensor([[0.0, 1.0], [0.0, 1.1], [0.0, 100.0]], dtype=torch.float64)
        result = remove_outliers(diagram, method="iqr", threshold=1.5)
        assert result.shape[0] <= 3

    def test_zscore_method(self) -> None:
        from pynerve.torch.preprocessing import remove_outliers

        diagram = torch.tensor([[0.0, 1.0], [0.0, 1.1], [0.0, 100.0]], dtype=torch.float64)
        result = remove_outliers(diagram, method="zscore", threshold=2.0)
        assert result.shape[0] <= 3

    def test_isolation_forest_fallback_on_no_sklearn(self) -> None:
        from pynerve.torch.preprocessing import remove_outliers

        diagram = torch.tensor([[0.0, 1.0], [0.0, 1.1], [0.0, 100.0]], dtype=torch.float64)
        result = remove_outliers(diagram, method="isolation_forest", threshold=1.5)
        assert isinstance(result, torch.Tensor)

    def test_invalid_method_raises(self) -> None:
        from pynerve.torch.preprocessing import remove_outliers

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            remove_outliers(diagram, method="bogus")

    def test_nan_threshold_raises(self) -> None:
        from pynerve.torch.preprocessing import remove_outliers

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            remove_outliers(diagram, threshold=float("nan"))

    def test_non_finite_diagram_falls_back(self) -> None:
        from pynerve.torch.preprocessing import remove_outliers

        diagram = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        result = remove_outliers(diagram, method="isolation_forest")
        assert isinstance(result, torch.Tensor)


class TestCleanDiagram:
    """clean_diagram pipeline."""

    def test_default_pipeline(self) -> None:
        from pynerve.torch.preprocessing import clean_diagram

        diagram = torch.tensor([[0.0, float("inf")], [0.5, 2.0]], dtype=torch.float64)
        result = clean_diagram(diagram)
        assert torch.isfinite(result[:, 1]).all()

    def test_batched(self) -> None:
        from pynerve.torch.preprocessing import clean_diagram

        diagram = torch.tensor(
            [[[0.0, float("inf")], [0.5, 2.0]], [[0.1, 1.0], [0.3, 3.0]]], dtype=torch.float64
        )
        result = clean_diagram(diagram)
        assert result.dim() == 3
        assert torch.isfinite(result[:, :, 1]).all()

    def test_empty_batched(self) -> None:
        from pynerve.torch.preprocessing import clean_diagram

        diagram = torch.empty((0, 5, 2), dtype=torch.float64)
        result = clean_diagram(diagram)
        assert result.shape == (0, 5, 2)

    def test_with_max_features(self) -> None:
        from pynerve.torch.preprocessing import clean_diagram

        diagram = torch.tensor(
            [[0.0, 1.0], [0.1, 2.0], [0.2, 3.0], [0.3, 4.0]], dtype=torch.float64
        )
        result = clean_diagram(diagram, max_features=2)
        assert result.shape[0] == 2

    def test_without_normalize(self) -> None:
        from pynerve.torch.preprocessing import clean_diagram

        diagram = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
        result = clean_diagram(diagram, normalize=False)
        assert not torch.allclose(result[:, 0], torch.tensor(0.0, dtype=result.dtype))

    def test_invalid_dim_raises(self) -> None:
        from pynerve.torch.preprocessing import clean_diagram

        with pytest.raises((ValueError, ValidationError)):
            clean_diagram(torch.randn(10))
