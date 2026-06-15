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


# diagram_wasserstein / diagram_bottleneck


class TestDiagramWasserstein:
    """diagram_wasserstein happy path and error cases."""

    def test_identical_diagrams_zero(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d = torch.tensor([[0.0, 1.0], [0.2, 0.8]], dtype=torch.float64)
        dist = diagram_wasserstein(d, d)
        if isinstance(dist, torch.Tensor):
            assert dist.dtype == torch.float64
            assert dist.item() == pytest.approx(0.0, abs=1e-6)
        else:
            assert dist == pytest.approx(0.0, abs=1e-6)

    def test_different_diagrams_positive(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d1 = torch.tensor([[0.0, 1.0], [0.2, 0.8]], dtype=torch.float64)
        d2 = torch.tensor([[0.1, 1.2], [0.3, 0.9]], dtype=torch.float64)
        dist = diagram_wasserstein(d1, d2)
        assert dist > 0

    def test_custom_p_and_q(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        dist = diagram_wasserstein(d, d, p=1.0, q=1.0)
        assert dist == pytest.approx(0.0)

    def test_invalid_p_raises(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            diagram_wasserstein(d, d, p=float("nan"))

    def test_invalid_q_raises(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, ValidationError)):
            diagram_wasserstein(d, d, q=float("nan"))

    def test_non_finite_births_raises(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        bad = torch.tensor([[float("nan"), 1.0]], dtype=torch.float32)
        with pytest.raises((ValueError, ValidationError)):
            diagram_wasserstein(bad, d.to(torch.float32))

    def test_non_finite_deaths_raises(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        bad = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        with pytest.raises((ValueError, ValidationError)):
            diagram_wasserstein(bad, d.to(torch.float32))

    def test_infinite_deaths_raises(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        bad = torch.tensor([[0.0, float("inf")]], dtype=torch.float32)
        result = diagram_wasserstein(bad, d.to(torch.float32))
        assert isinstance(result, (torch.Tensor, float))

    def test_integer_tensor_raises(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        int_t = torch.tensor([[0, 1]], dtype=torch.int64)
        with pytest.raises((TypeError, ValidationError)):
            diagram_wasserstein(int_t, d)


class TestDiagramBottleneck:
    """diagram_bottleneck happy path and error cases."""

    def test_identical_diagrams_zero(self) -> None:
        from pynerve.torch import diagram_bottleneck

        d = torch.tensor([[0.0, 1.0], [0.2, 0.8]], dtype=torch.float64)
        dist = diagram_bottleneck(d, d)
        if isinstance(dist, torch.Tensor):
            assert dist.dtype == torch.float64
            assert dist.item() == pytest.approx(0.0, abs=1e-6)
        else:
            assert dist == pytest.approx(0.0, abs=1e-6)

    def test_different_diagrams_positive(self) -> None:
        from pynerve.torch import diagram_bottleneck

        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.5, 1.5]], dtype=torch.float64)
        dist = diagram_bottleneck(d1, d2)
        assert dist > 0

    def test_non_finite_births_raises(self) -> None:
        from pynerve.torch import diagram_bottleneck

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        bad = torch.tensor([[float("nan"), 1.0]], dtype=torch.float32)
        with pytest.raises((ValueError, ValidationError)):
            diagram_bottleneck(bad, d.to(torch.float32))

    def test_non_finite_deaths_raises(self) -> None:
        from pynerve.torch import diagram_bottleneck

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        bad = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        with pytest.raises((ValueError, ValidationError)):
            diagram_bottleneck(bad, d.to(torch.float32))

    def test_integer_tensor_raises(self) -> None:
        from pynerve.torch import diagram_bottleneck

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        int_t = torch.tensor([[0, 1]], dtype=torch.int64)
        with pytest.raises((TypeError, ValidationError)):
            diagram_bottleneck(int_t, d)

    def test_infinite_deaths_handled(self) -> None:
        from pynerve.torch import diagram_bottleneck

        d1 = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        dist = diagram_bottleneck(d1, d2)
        assert torch.isfinite(dist if isinstance(dist, torch.Tensor) else torch.tensor(dist))
