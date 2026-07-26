"""Tests for torch/_kernels_pairwise.py -- kernel functions for persistence diagrams."""

from __future__ import annotations

import pytest
import torch
from _test_helpers import make_diag_2d

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")

torch = pytest.importorskip("torch")


class TestComputeDistanceMatrix:
    def test_p1(self):
        from pynerve.torch._kernels_pairwise import _compute_distance_matrix
        d1 = make_diag_2d(3)
        d2 = make_diag_2d(4)
        result = _compute_distance_matrix(d1, d2, p=1.0)
        assert result.shape == (3, 4)
        assert (result >= 0).all()

    def test_p2(self):
        from pynerve.torch._kernels_pairwise import _compute_distance_matrix
        d1 = make_diag_2d(3)
        result = _compute_distance_matrix(d1, d1, p=2.0)
        assert result.shape == (3, 3)
        assert torch.allclose(torch.diag(result), torch.zeros(3), atol=1e-5)

    def test_pinf(self):
        from pynerve.torch._kernels_pairwise import _compute_distance_matrix
        import math
        d1 = make_diag_2d(3)
        result = _compute_distance_matrix(d1, d1, p=math.inf)
        assert result.shape == (3, 3)
        assert torch.allclose(torch.diag(result), torch.zeros(3), atol=1e-5)

    def test_p3(self):
        from pynerve.torch._kernels_pairwise import _compute_distance_matrix
        d1 = make_diag_2d(3)
        d2 = make_diag_2d(4)
        result = _compute_distance_matrix(d1, d2, p=3.0)
        assert result.shape == (3, 4)
        assert (result >= 0).all()


class TestValidateKernelDiagrams:
    def test_valid(self):
        from pynerve.torch._kernels_pairwise import _validate_kernel_diagrams
        d1, d2 = _validate_kernel_diagrams(make_diag_2d(3), make_diag_2d(4))
        assert d1.shape[0] == 3
        assert d2.shape[0] == 4

    def test_nan_deaths(self):
        from pynerve.torch._kernels_pairwise import _validate_kernel_diagrams
        d = torch.tensor([[0.0, float("nan")]])
        with pytest.raises(Exception, match="NaN"):
            _validate_kernel_diagrams(d, make_diag_2d(3))


class TestValidatePositiveNorm:
    def test_valid(self):
        from pynerve.torch._kernels_pairwise import _validate_positive_norm
        assert _validate_positive_norm(2.0, "p") == 2.0

    def test_zero(self):
        from pynerve.torch._kernels_pairwise import _validate_positive_norm
        with pytest.raises(ValueError, match="positive"):
            _validate_positive_norm(0.0, "p")

    def test_nan(self):
        from pynerve.torch._kernels_pairwise import _validate_positive_norm
        with pytest.raises(ValueError, match="positive"):
            _validate_positive_norm(float("nan"), "p")


class TestGaussianKernel:
    def test_euclidean(self):
        from pynerve.torch._kernels_pairwise import gaussian_kernel
        d1 = make_diag_2d(5)
        d2 = make_diag_2d(5)
        result = gaussian_kernel(d1, d2, sigma=1.0, distance_metric="euclidean")
        assert result >= 0

    def test_self_kernel(self):
        from pynerve.torch._kernels_pairwise import gaussian_kernel
        d = make_diag_2d(5)
        result = gaussian_kernel(d, d, sigma=1.0, distance_metric="euclidean")
        assert result > 0

    def test_invalid_sigma(self):
        from pynerve.torch._kernels_pairwise import gaussian_kernel
        with pytest.raises(ValueError, match="sigma|positive|finite"):
            gaussian_kernel(make_diag_2d(3), make_diag_2d(3), sigma=-1.0)

    def test_invalid_metric(self):
        from pynerve.torch._kernels_pairwise import gaussian_kernel
        with pytest.raises(ValueError, match="distance_metric"):
            gaussian_kernel(make_diag_2d(3), make_diag_2d(3), distance_metric="bad")

    def test_empty(self):
        from pynerve.torch._kernels_pairwise import gaussian_kernel
        result = gaussian_kernel(torch.empty(0, 2), make_diag_2d(3), distance_metric="euclidean")
        assert result.item() == 0.0

    def test_p1(self):
        from pynerve.torch._kernels_pairwise import gaussian_kernel
        d1 = make_diag_2d(5)
        d2 = make_diag_2d(5)
        result = gaussian_kernel(d1, d2, sigma=1.0, p=1.0)
        assert result >= 0

    @pytest.mark.skip(reason="wasserstein distance returns MagicMock with mocked C++ backend, torch.exp fails on non-Tensor")
    def test_wasserstein(self):
        from pynerve.torch._kernels_pairwise import gaussian_kernel
        d1 = make_diag_2d(5)
        d2 = make_diag_2d(5)
        result = gaussian_kernel(d1, d2, sigma=1.0, distance_metric="wasserstein")
        assert result is not None

    @pytest.mark.skip(reason="bottleneck distance returns MagicMock with mocked C++ backend, torch.exp fails on non-Tensor")
    def test_bottleneck(self):
        from pynerve.torch._kernels_pairwise import gaussian_kernel
        d1 = make_diag_2d(5)
        d2 = make_diag_2d(5)
        result = gaussian_kernel(d1, d2, sigma=1.0, distance_metric="bottleneck")
        assert result is not None


class TestPersistenceScaleSpaceKernel:
    def test_basic(self):
        from pynerve.torch._kernels_pairwise import persistence_scale_space_kernel
        d1 = make_diag_2d(5)
        d2 = make_diag_2d(5)
        result = persistence_scale_space_kernel(d1, d2, sigma=1.0, weight=0.5)
        assert result >= 0

    def test_weight_zero(self):
        from pynerve.torch._kernels_pairwise import persistence_scale_space_kernel
        d1 = make_diag_2d(3)
        d2 = make_diag_2d(3)
        result = persistence_scale_space_kernel(d1, d2, sigma=1.0, weight=0.0)
        assert result >= 0

    def test_weight_one(self):
        from pynerve.torch._kernels_pairwise import persistence_scale_space_kernel
        d1 = make_diag_2d(3)
        d2 = make_diag_2d(3)
        result = persistence_scale_space_kernel(d1, d2, sigma=1.0, weight=1.0)
        assert result >= 0

    def test_invalid_weight(self):
        from pynerve.torch._kernels_pairwise import persistence_scale_space_kernel
        with pytest.raises(ValueError, match="weight"):
            persistence_scale_space_kernel(make_diag_2d(3), make_diag_2d(3), weight=1.5)

    def test_invalid_sigma(self):
        from pynerve.torch._kernels_pairwise import persistence_scale_space_kernel
        with pytest.raises(ValueError, match="sigma|positive|finite"):
            persistence_scale_space_kernel(make_diag_2d(3), make_diag_2d(3), sigma=0.0)

    def test_empty(self):
        from pynerve.torch._kernels_pairwise import persistence_scale_space_kernel
        result = persistence_scale_space_kernel(torch.empty(0, 2), make_diag_2d(3))
        assert result.item() == 0.0


class TestSlicedWassersteinKernel:
    def test_basic(self):
        from pynerve.torch._kernels_pairwise import sliced_wasserstein_kernel
        d1 = make_diag_2d(5)
        d2 = make_diag_2d(5)
        result = sliced_wasserstein_kernel(d1, d2, num_slices=10, sigma=1.0)
        assert result >= 0

    def test_single_slice(self):
        from pynerve.torch._kernels_pairwise import sliced_wasserstein_kernel
        d1 = make_diag_2d(3)
        d2 = make_diag_2d(3)
        result = sliced_wasserstein_kernel(d1, d2, num_slices=1, sigma=1.0)
        assert result >= 0

    def test_invalid_num_slices(self):
        from pynerve.torch._kernels_pairwise import sliced_wasserstein_kernel
        with pytest.raises(ValueError, match="positive"):
            sliced_wasserstein_kernel(make_diag_2d(3), make_diag_2d(3), num_slices=0)

    def test_invalid_sigma(self):
        from pynerve.torch._kernels_pairwise import sliced_wasserstein_kernel
        with pytest.raises(ValueError, match="sigma|positive|finite"):
            sliced_wasserstein_kernel(make_diag_2d(3), make_diag_2d(3), sigma=-1.0)

    def test_empty(self):
        from pynerve.torch._kernels_pairwise import sliced_wasserstein_kernel
        result = sliced_wasserstein_kernel(torch.empty(0, 2), make_diag_2d(3))
        assert result.item() == 0.0


class TestPersistenceFisherKernel:
    def test_basic(self):
        from pynerve.torch._kernels_pairwise import persistence_fisher_kernel
        d1 = make_diag_2d(5)
        d2 = make_diag_2d(5)
        result = persistence_fisher_kernel(d1, d2, sigma=1.0, bandwidth=0.5)
        assert result >= 0

    def test_invalid_sigma(self):
        from pynerve.torch._kernels_pairwise import persistence_fisher_kernel
        with pytest.raises(ValueError, match="sigma|positive|finite"):
            persistence_fisher_kernel(make_diag_2d(3), make_diag_2d(3), sigma=0.0)

    def test_invalid_bandwidth(self):
        from pynerve.torch._kernels_pairwise import persistence_fisher_kernel
        with pytest.raises(ValueError, match="bandwidth|positive|finite"):
            persistence_fisher_kernel(make_diag_2d(3), make_diag_2d(3), bandwidth=-1.0)

    def test_empty(self):
        from pynerve.torch._kernels_pairwise import persistence_fisher_kernel
        result = persistence_fisher_kernel(torch.empty(0, 2), make_diag_2d(3))
        assert result.item() == 0.0


class TestLinearKernel:
    def test_silhouette(self):
        from pynerve.torch._kernels_pairwise import linear_kernel
        d1 = make_diag_2d(5)
        d2 = make_diag_2d(5)
        result = linear_kernel(d1, d2, vectorization="silhouette")
        assert result is not None

    def test_image(self):
        from pynerve.torch._kernels_pairwise import linear_kernel
        d1 = make_diag_2d(5)
        d2 = make_diag_2d(5)
        result = linear_kernel(d1, d2, vectorization="image")
        assert result is not None

    def test_landscape(self):
        from pynerve.torch._kernels_pairwise import linear_kernel
        d1 = make_diag_2d(5)
        d2 = make_diag_2d(5)
        result = linear_kernel(d1, d2, vectorization="landscape")
        assert result is not None


class TestFiniteCoords:
    def test_basic(self):
        from pynerve.torch._kernels_pairwise import _finite_coords
        d = make_diag_2d(5)
        result = _finite_coords(d)
        assert result.shape == (5, 2)

    def test_with_inf_death(self):
        from pynerve.torch._kernels_pairwise import _finite_coords
        d = torch.tensor([[0.0, 1.0], [0.5, float("inf")]])
        result = _finite_coords(d)
        assert result.shape == (1, 2)
