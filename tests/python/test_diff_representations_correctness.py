"""Numerical correctness tests for diff persistence representations.

Verifies actual numerical values, not just shapes or finiteness.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestComputePersistenceLandscape:
    """Numerical correctness for compute_persistence_landscape."""

    def test_single_point_tent(self) -> None:
        from pynerve.diff._ph_representations import compute_persistence_landscape

        d = torch.tensor([[0.0, 4.0]], dtype=torch.float64)
        land = compute_persistence_landscape(d, n_layers=1, resolution=5)
        expected = torch.tensor([[0.0, 1.0, 2.0, 1.0, 0.0]], dtype=torch.float64)
        assert torch.allclose(land, expected, atol=1e-10)

    def test_two_points_landscape_both_layers(self) -> None:
        from pynerve.diff._ph_representations import compute_persistence_landscape

        d = torch.tensor([[0.0, 2.0], [0.0, 6.0]], dtype=torch.float64)
        land = compute_persistence_landscape(d, n_layers=2, resolution=7)
        expected_l1 = torch.tensor([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0], dtype=torch.float64)
        expected_l2 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(land[0], expected_l1, atol=1e-10)
        assert torch.allclose(land[1], expected_l2, atol=1e-10)

    def test_two_layers_monotonic(self) -> None:
        from pynerve.diff._ph_representations import compute_persistence_landscape

        d = torch.tensor([[0.0, 2.0], [0.0, 6.0]], dtype=torch.float64)
        land = compute_persistence_landscape(d, n_layers=3, resolution=10)
        assert (land[1] <= land[0] + 1e-12).all()
        assert (land[2] <= land[1] + 1e-12).all()

    def test_empty_diagram(self) -> None:
        from pynerve.diff._ph_representations import compute_persistence_landscape

        d = torch.empty(0, 2, dtype=torch.float64)
        land = compute_persistence_landscape(d, n_layers=3, resolution=10)
        assert (land == 0).all()

    def test_gradient_nonzero_and_sign(self) -> None:
        from pynerve.diff._ph_representations import compute_persistence_landscape

        d = torch.tensor([[1.0, 3.0]], dtype=torch.float64, requires_grad=True)
        land = compute_persistence_landscape(d, n_layers=1, resolution=10)
        loss = land.sum()
        loss.backward()
        assert d.grad is not None
        assert d.grad.abs().sum().item() > 0
        assert d.grad[0, 0].item() < 0
        assert d.grad[0, 1].item() > 0


class TestPersistenceImage:
    """Numerical correctness for persistence_image in _ph_representations."""

    def test_single_point_nonzero_mass(self) -> None:
        from pynerve.diff._ph_representations import persistence_image

        d = torch.tensor([[0.5, 2.5]], dtype=torch.float64)
        img = persistence_image(d, resolution=20, sigma=0.3)
        assert img.sum().item() > 0
        assert (img >= 0).all()

    def test_empty_diagram_zero(self) -> None:
        from pynerve.diff._ph_representations import persistence_image

        d = torch.empty(0, 2, dtype=torch.float64)
        img = persistence_image(d, resolution=5, sigma=0.5)
        assert (img == 0).all()

    def test_gradient_nonzero(self) -> None:
        from pynerve.diff._ph_representations import persistence_image

        d = torch.tensor([[0.5, 2.5]], dtype=torch.float64, requires_grad=True)
        img = persistence_image(d, resolution=20, sigma=0.3)
        loss = img.sum()
        loss.backward()
        assert d.grad is not None
        assert d.grad.abs().sum().item() > 0
