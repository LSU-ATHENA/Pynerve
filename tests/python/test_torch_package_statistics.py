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


# statistics


class TestStatistics:
    """Top-level statistics functions (happy path + errors)."""

    def test_total_persistence(self) -> None:
        from pynerve.torch.statistics import total_persistence

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        result = total_persistence(diagram)
        assert result.dtype == diagram.dtype
        assert result.numel() == 1

    def test_persistence_entropy(self) -> None:
        from pynerve.torch.statistics import persistence_entropy

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        result = persistence_entropy(diagram)
        assert torch.isfinite(result)

    def test_betti_curve(self) -> None:
        from pynerve.torch.statistics import betti_curve

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        result = betti_curve(diagram, num_samples=5)
        assert result.shape == (5,)

    def test_betti_numbers_at_scale(self) -> None:
        from pynerve.torch.statistics import betti_numbers_at_scale

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        result = betti_numbers_at_scale(diagram, threshold=0.5)
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 1

    def test_number_of_features(self) -> None:
        from pynerve.torch.statistics import number_of_features

        diagram = torch.tensor([[0.0, 1.0], [0.0, 0.5]], dtype=torch.float64)
        result = number_of_features(diagram)
        assert result.numel() == 1

    def test_amplitude(self) -> None:
        from pynerve.torch.statistics import amplitude

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        result = amplitude(diagram, metric="wasserstein")
        assert torch.isfinite(result)

    def test_mean_persistence(self) -> None:
        from pynerve.torch.statistics import mean_persistence

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        result = mean_persistence(diagram)
        assert result.numel() == 1
        assert torch.isfinite(result)

    def test_max_persistence(self) -> None:
        from pynerve.torch.statistics import max_persistence

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        result = max_persistence(diagram)
        assert result.item() == 2.0

    def test_persistence_variance(self) -> None:
        from pynerve.torch.statistics import persistence_variance

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        result = persistence_variance(diagram)
        assert torch.isfinite(result)

    def test_all_statistics(self) -> None:
        from pynerve.torch.statistics import all_statistics

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        stats = all_statistics(diagram)
        assert isinstance(stats, dict)
        assert len(stats) > 0

    def test_extract_features(self) -> None:
        from pynerve.torch.statistics import extract_features

        diagram = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float64)
        features = extract_features(diagram, dims=[0, 1])
        assert features.dim() == 1
        assert torch.isfinite(features).all()

    # --- error cases ---

    def test_total_persistence_invalid_p_raises(self) -> None:
        from pynerve.torch.statistics import total_persistence

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            total_persistence(diagram, p=float("nan"))

    def test_persistence_entropy_invalid_base_raises(self) -> None:
        from pynerve.torch.statistics import persistence_entropy

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            persistence_entropy(diagram, base=0.0)
        with pytest.raises((ValueError, ValidationError)):
            persistence_entropy(diagram, base=1.0)

    def test_betti_curve_invalid_num_samples_raises(self) -> None:
        from pynerve.torch.statistics import betti_curve

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, TypeError, ValidationError)):
            betti_curve(diagram, num_samples=0)

    def test_amplitude_invalid_metric_raises(self) -> None:
        from pynerve.torch.statistics import amplitude

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            amplitude(diagram, metric="bogus")

    def test_non_finite_births_raises(self) -> None:
        from pynerve.torch.statistics import total_persistence

        bad = torch.tensor([[float("nan"), 1.0]], dtype=torch.float32)
        with pytest.raises((ValueError, ValidationError)):
            total_persistence(bad)

    def test_deaths_less_than_births_raises(self) -> None:
        from pynerve.torch.statistics import total_persistence

        bad = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
        with pytest.raises((ValueError, ValidationError)):
            total_persistence(bad)

    def test_infinite_deaths_raises(self) -> None:
        from pynerve.torch.statistics import total_persistence

        bad = torch.tensor([[0.0, float("inf")]], dtype=torch.float32)
        with pytest.raises((ValueError, ValidationError)):
            total_persistence(bad)

    def test_betti_numbers_at_scale_with_dims(self) -> None:
        from pynerve.torch.statistics import betti_numbers_at_scale

        diagram = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float64)
        result = betti_numbers_at_scale(diagram, threshold=0.5)
        assert isinstance(result, torch.Tensor)
