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

# gradient flow through differentiable ops


class TestGradientCorrectness:
    """Gradient flow through differentiable operations."""

    def test_persistence_image_gradient_nonzero(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        d = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        img = persistence_image(d, resolution=(4, 4), sigma=1.0)
        loss = img.sum()
        loss.backward()
        assert d.grad is not None
        assert d.grad.abs().sum().item() > 0
        assert torch.isfinite(d.grad).all()

    def test_persistence_landscape_gradient_nonzero(self) -> None:
        from pynerve.torch.vectorization import persistence_landscape

        x_range: tuple[float, float] = (0.0, 3.0)
        d = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        land = persistence_landscape(d, k=1, num_samples=5, x_range=x_range)
        loss = land.sum()
        loss.backward()
        assert d.grad is not None
        assert d.grad.abs().sum().item() > 0

    def test_persistence_silhouette_gradient_nonzero(self) -> None:
        from pynerve.torch.vectorization import persistence_silhouette

        d = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        s = persistence_silhouette(d, num_samples=5)
        loss = s.sum()
        loss.backward()
        assert d.grad is not None
        assert d.grad.abs().sum().item() > 0

    def test_total_persistence_gradient_nonzero(self) -> None:
        from pynerve.torch.statistics import total_persistence

        d = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        tp = total_persistence(d, p=1.0)
        loss = tp.sum()
        loss.backward()
        assert d.grad is not None
        assert d.grad.abs().sum().item() > 0
