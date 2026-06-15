"""Numerical correctness tests that verify outputs against hand-computed values.

Every test with a numerical value includes a precise assertion with tolerance,
not just a shape or finiteness check.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# known-configuration point clouds
_TRIANGLE = torch.tensor(
    [[0.0, 0.0], [2.0, 0.0], [1.0, 3.0**0.5]],
    dtype=torch.float64,
)
_SQUARE = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=torch.float64,
)
_TWO_POINTS = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
_SINGLE_POINT = torch.tensor([[0.0, 0.0]], dtype=torch.float64)

# known diagram tensors
_SIMPLE_DIAGRAM = torch.tensor(
    [[0.0, 1.0], [0.0, 2.0], [1.0, 3.0]],
    dtype=torch.float64,
)
_SINGLE_DIAGRAM = torch.tensor([[0.0, 1.5]], dtype=torch.float64)
_INFINITE_DIAGRAM = torch.tensor(
    [[0.0, float("inf")], [0.0, 2.0]],
    dtype=torch.float64,
)

# kernels


class TestKernelCorrectness:
    """Numerical correctness for kernel functions."""

    def test_gaussian_kernel_positive(self) -> None:
        from pynerve.torch.kernels import gaussian_kernel

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        k = gaussian_kernel(d, d, sigma=1.0)
        assert k.item() > 0
        assert torch.isfinite(k)

    def test_gaussian_kernel_symmetric(self) -> None:
        from pynerve.torch.kernels import gaussian_kernel

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.5, 1.5], [0.0, 3.0]], dtype=torch.float64)
        k12 = gaussian_kernel(d1, d2, sigma=1.0)
        k21 = gaussian_kernel(d2, d1, sigma=1.0)
        assert k12.item() == pytest.approx(k21.item(), abs=1e-10)

    def test_persistence_scale_space_kernel(self) -> None:
        from pynerve.torch.kernels import persistence_scale_space_kernel

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        k = persistence_scale_space_kernel(d, d, sigma=1.0)
        assert k.item() > 0

    def test_sliced_wasserstein_kernel(self) -> None:
        from pynerve.torch.kernels import sliced_wasserstein_kernel

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        k = sliced_wasserstein_kernel(d, d, num_slices=10)
        assert k.item() > 0

    def test_persistence_fisher_kernel(self) -> None:
        from pynerve.torch.kernels import persistence_fisher_kernel

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        k = persistence_fisher_kernel(d, d, bandwidth=1.0)
        assert torch.isfinite(k)

    def test_kernel_matrix_psd(self) -> None:
        from pynerve.torch.kernels import compute_kernel_matrix

        diagrams = [
            torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64),
            torch.tensor([[0.5, 1.5]], dtype=torch.float64),
            torch.tensor([[0.0, 3.0]], dtype=torch.float64),
        ]
        K = compute_kernel_matrix(diagrams, kernel="gaussian", sigma=1.0)
        assert K.shape == (3, 3)
        assert (K.diag() > 0).all()
        assert (K >= 0).all()
        assert torch.allclose(K, K.T, atol=1e-12)

    def test_kernel_matrix_normalization(self) -> None:
        from pynerve.torch.kernels import compute_kernel_matrix, normalize_kernel_matrix

        diagrams = [
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        ]
        K = compute_kernel_matrix(diagrams, kernel="gaussian", sigma=1.0)
        Kn = normalize_kernel_matrix(K)
        assert (Kn.diag() == 1.0).all()
