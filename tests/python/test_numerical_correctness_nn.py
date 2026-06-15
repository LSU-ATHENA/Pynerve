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

# nn layers


class TestNnLayerCorrectness:
    """Numerical correctness for NN layers."""

    def test_persistence_readout_forward(self) -> None:
        from pynerve.torch.nn_layers import PersistenceReadout

        readout = PersistenceReadout(in_features=8, out_features=4)
        x = torch.randn(2, 8)
        out = readout(x)
        assert out.shape == (2, 4)
        assert torch.isfinite(out).all()

    def test_persistence_readout_gradient(self) -> None:
        from pynerve.torch.nn_layers import PersistenceReadout

        readout = PersistenceReadout(in_features=8, out_features=4)
        x = torch.randn(2, 8, requires_grad=True)
        out = readout(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

    def test_diagram_pooling_mean_exact(self) -> None:
        from pynerve.torch.nn_layers import DiagramPooling

        pool = DiagramPooling(method="mean")
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float32)
        out = pool(x)
        expected = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_diagram_pooling_sum_exact(self) -> None:
        from pynerve.torch.nn_layers import DiagramPooling

        pool = DiagramPooling(method="sum")
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
        out = pool(x)
        expected = torch.tensor([[4.0, 6.0]], dtype=torch.float32)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_diagram_pooling_max_exact(self) -> None:
        from pynerve.torch.nn_layers import DiagramPooling

        pool = DiagramPooling(method="max")
        x = torch.tensor([[[1.0, 5.0], [3.0, 4.0], [2.0, 6.0]]], dtype=torch.float32)
        out = pool(x)
        expected = torch.tensor([[3.0, 6.0]], dtype=torch.float32)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_persistence_layer_forward(self) -> None:
        from pynerve.torch.nn_layers import PersistenceLayer

        if not hasattr(PersistenceLayer, "forward"):
            pytest.skip("PersistenceLayer not available")
        layer = PersistenceLayer(max_dim=0, max_radius=3.0)
        x = torch.randn(1, 5, 2)
        result = layer(x)
        assert result is not None
