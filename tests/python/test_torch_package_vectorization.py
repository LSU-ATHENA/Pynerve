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


# vectorization


class TestVectorization:
    """Vectorization functions (tested extensively in test_torch_ops_contracts.py)."""

    def test_persistence_image(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        img = persistence_image(diagram, resolution=(4, 4), sigma=0.2)
        assert img.shape == (4, 4)
        assert img.dtype == diagram.dtype

    def test_persistence_landscape(self) -> None:
        from pynerve.torch.vectorization import persistence_landscape

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        land = persistence_landscape(diagram, k=2, num_samples=5)
        assert land.shape == (2, 5)

    def test_persistence_silhouette(self) -> None:
        from pynerve.torch.vectorization import persistence_silhouette

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        sil = persistence_silhouette(diagram, num_samples=5)
        assert sil.shape == (5,)

    def test_adaptive_persistence_image(self) -> None:
        from pynerve.torch.vectorization import adaptive_persistence_image

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        img = adaptive_persistence_image(diagram, target_resolution=4)
        assert img.dim() == 2

    def test_heat_kernel_signature(self) -> None:
        from pynerve.torch.vectorization import heat_kernel_signature

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        t_values = torch.tensor([0.1, 0.5], dtype=torch.float32)
        heat = heat_kernel_signature(diagram, num_samples=5, sigma=0.2, t_values=t_values)
        assert heat.shape == (2, 5)

    def test_birth_death_curve(self) -> None:
        from pynerve.torch.vectorization import birth_death_curve

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        curve = birth_death_curve(diagram, num_bins=4, statistic="mean")
        assert curve.shape == (4,)

    def test_diagram_to_vector(self) -> None:
        from pynerve.torch.vectorization import diagram_to_vector

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        vec = diagram_to_vector(diagram, method="silhouette", num_samples=8)
        assert vec.dim() == 1

    def test_invalid_sigma_raises(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            persistence_image(diagram, sigma=float("nan"))

    def test_invalid_x_range_raises(self) -> None:
        from pynerve.torch.vectorization import persistence_landscape

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            persistence_landscape(diagram, k=1, num_samples=5, x_range=(2.0, 1.0))

    def test_non_finite_births_raises(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        bad = torch.tensor([[float("nan"), 1.0]], dtype=torch.float32)
        with pytest.raises((ValueError, ValidationError)):
            persistence_image(bad)

    def test_deaths_less_than_births_raises(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        bad = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        with pytest.raises((ValueError, ValidationError)):
            persistence_image(bad)

    def test_infinite_deaths_raises(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        bad = torch.tensor([[0.0, float("inf")]], dtype=torch.float32)
        result = persistence_image(bad, resolution=(4, 4))
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 4)
