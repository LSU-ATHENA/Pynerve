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


# viz


class TestViz:
    """Visualisation helpers (tested extensively in test_torch_ops_contracts.py)."""

    def test_diagram_to_scatter_data(self) -> None:
        from pynerve.torch.viz import diagram_to_scatter_data

        diagram = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float64)
        data = diagram_to_scatter_data(diagram)
        assert "births" in data
        assert "deaths" in data

    def test_diagram_to_scatter_data_with_dim(self) -> None:
        from pynerve.torch.viz import diagram_to_scatter_data

        diagram = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float64)
        data = diagram_to_scatter_data(diagram, dim=0)
        assert len(data["births"]) >= 1

    def test_diagram_to_histogram_data(self) -> None:
        from pynerve.torch.viz import diagram_to_histogram_data

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        data = diagram_to_histogram_data(diagram, num_bins=4)
        assert "bins" in data
        assert "values" in data

    def test_diagram_to_heatmap_data(self) -> None:
        from pynerve.torch.viz import diagram_to_heatmap_data

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        data = diagram_to_heatmap_data(diagram, grid_size=4)
        assert "grid" in data
        assert data["grid"].shape == (4, 4)

    def test_diagram_to_image_data(self) -> None:
        from pynerve.torch.viz import diagram_to_image_data

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        img = diagram_to_image_data(diagram, resolution=(4, 4))
        assert img.shape == (4, 4)

    def test_diagram_to_landscape_data(self) -> None:
        from pynerve.torch.viz import diagram_to_landscape_data

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        data = diagram_to_landscape_data(diagram, k=2, num_samples=5)
        assert "landscapes" in data
        assert "x_values" in data

    def test_diagram_to_betti_data(self) -> None:
        from pynerve.torch.viz import diagram_to_betti_data

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        data = diagram_to_betti_data(diagram, num_samples=5)
        assert "thresholds" in data
        assert "betti_numbers" in data

    def test_get_plot_limits(self) -> None:
        from pynerve.torch.viz import get_plot_limits

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        limits = get_plot_limits(diagram)
        assert len(limits) == 4

    def test_get_plot_limits_with_padding(self) -> None:
        from pynerve.torch.viz import get_plot_limits

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        limits = get_plot_limits(diagram, padding=0.2)
        assert limits[0] >= 0.0

    def test_invalid_padding_raises(self) -> None:
        from pynerve.torch.viz import get_plot_limits

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            get_plot_limits(diagram, padding=float("nan"))

    def test_non_finite_births_raises(self) -> None:
        from pynerve.torch.viz import diagram_to_scatter_data

        bad = torch.tensor([[float("nan"), 1.0, 0.0]], dtype=torch.float32)
        with pytest.raises((ValueError, ValidationError)):
            diagram_to_scatter_data(bad)

    def test_invalid_dimensions_raises(self) -> None:
        from pynerve.torch.viz import diagram_to_scatter_data

        bad = torch.tensor([[0.0, 1.0, 0.5]], dtype=torch.float32)
        with pytest.raises((ValueError, ValidationError)):
            diagram_to_scatter_data(bad)

    def test_diagram_to_histogram_invalid_bins_raises(self) -> None:
        from pynerve.torch.viz import diagram_to_histogram_data

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, TypeError, ValidationError)):
            diagram_to_histogram_data(diagram, num_bins=0)

    def test_diagram_to_heatmap_invalid_grid_raises(self) -> None:
        from pynerve.torch.viz import diagram_to_heatmap_data

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, TypeError, ValidationError)):
            diagram_to_heatmap_data(diagram, grid_size=0)
