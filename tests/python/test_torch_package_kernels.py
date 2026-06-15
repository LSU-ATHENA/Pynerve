"""Comprehensive tests for pynerve.torch subpackage.

Covers the public API surface: lazy imports, PersistenceDiagram, diagram
distance functions, data utilities, nn layers, preprocessing, statistics,
vectorization, kernels, mapper, sklearn transformers, tensorboard logging,
and visualisation helpers.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


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


# kernels


class TestKernels:
    """Kernel functions (tested extensively in test_torch_ops_contracts.py)."""

    def test_gaussian_kernel(self) -> None:
        from pynerve.torch.kernels import gaussian_kernel

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        result = gaussian_kernel(d1, d2, sigma=0.5)
        assert result.dtype == d1.dtype
        assert result.shape == () or result.numel() == 1

    def test_linear_kernel(self) -> None:
        from pynerve.torch.kernels import linear_kernel

        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        result = linear_kernel(d1, d2)
        assert result.numel() == 1
        assert torch.isfinite(result)

    def test_persistence_scale_space_kernel(self) -> None:
        from pynerve.torch.kernels import persistence_scale_space_kernel

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        result = persistence_scale_space_kernel(d1, d2, sigma=0.5)
        assert result.numel() == 1

    def test_sliced_wasserstein_kernel(self) -> None:
        from pynerve.torch.kernels import sliced_wasserstein_kernel

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        result = sliced_wasserstein_kernel(d1, d2, num_slices=10)
        assert result.numel() == 1

    def test_persistence_fisher_kernel(self) -> None:
        from pynerve.torch.kernels import persistence_fisher_kernel

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        result = persistence_fisher_kernel(d1, d2, bandwidth=1.0)
        assert result.numel() == 1

    def test_compute_kernel_matrix(self) -> None:
        from pynerve.torch.kernels import compute_kernel_matrix

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        matrix = compute_kernel_matrix([d, d], kernel="gaussian", sigma=0.5)
        assert matrix.shape == (2, 2)

    def test_normalize_kernel_matrix(self) -> None:
        from pynerve.torch.kernels import normalize_kernel_matrix

        matrix = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        normalized = normalize_kernel_matrix(matrix)
        assert normalized.shape == matrix.shape

    def test_center_kernel_matrix(self) -> None:
        from pynerve.torch.kernels import center_kernel_matrix

        matrix = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        centered = center_kernel_matrix(matrix)
        assert centered.shape == matrix.shape

    # error cases

    def test_invalid_sigma_raises(self) -> None:
        from pynerve.torch.kernels import gaussian_kernel

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError):
            gaussian_kernel(d, d, sigma=float("nan"))

    def test_invalid_distance_metric_raises(self) -> None:
        from pynerve.torch.kernels import gaussian_kernel

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError):
            gaussian_kernel(d, d, distance_metric="bogus")

    def test_unknown_kernel_raises(self) -> None:
        from pynerve.torch.kernels import compute_kernel_matrix

        with pytest.raises(ValueError):
            compute_kernel_matrix([], kernel="bogus")

    def test_zero_diagonal_normalize_raises(self) -> None:
        from pynerve.torch.kernels import normalize_kernel_matrix

        matrix = torch.zeros((2, 2), dtype=torch.float64)
        with pytest.raises(ValueError):
            normalize_kernel_matrix(matrix)

    def test_non_finite_center_raises(self) -> None:
        from pynerve.torch.kernels import center_kernel_matrix

        matrix = torch.tensor([[1.0, float("nan")], [0.0, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError):
            center_kernel_matrix(matrix)
