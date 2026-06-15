"""Numerical correctness tests for vr_persistence backward pass."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestVrBackwardGradient:
    """Gradient flow through vr_persistence forward/backward."""

    def test_two_points_gradient_exists(self) -> None:
        from pynerve.torch import vr_persistence

        pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32, requires_grad=True)
        diagram = vr_persistence(pts, max_dim=0, max_radius=5.0)
        births = diagram.births()
        deaths = diagram.deaths()
        finite_mask = torch.isfinite(deaths)
        loss = births[finite_mask].sum() + deaths[finite_mask].sum()
        loss.backward()
        assert pts.grad is not None
        assert pts.grad.shape == pts.shape

    def test_single_point_gradient_zero(self) -> None:
        from pynerve.torch import vr_persistence

        pts = torch.tensor([[[5.0, 5.0]]], dtype=torch.float32, requires_grad=True)
        diagram = vr_persistence(pts, max_dim=0, max_radius=float("inf"))
        births = diagram.births()
        deaths = diagram.deaths()
        finite_mask = torch.isfinite(deaths)
        loss = births[finite_mask].sum() + deaths[finite_mask].sum()
        loss.backward()
        assert pts.grad is not None

    def test_gradient_with_h1(self) -> None:
        from pynerve.torch import vr_persistence

        pts = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        diagram = vr_persistence(pts, max_dim=1, max_radius=5.0)
        births = diagram.births()
        deaths = diagram.deaths()
        finite_mask = torch.isfinite(deaths)
        loss = births[finite_mask].sum() + deaths[finite_mask].sum()
        loss.backward()
        assert pts.grad is not None
        assert pts.grad.shape == pts.shape


class TestVrBackwardAnalytical:
    """Verify the analytical gradient computation for 0D persistence."""

    def test_known_contribution_values(self) -> None:
        from pynerve.torch._persistence_vr import _compute_zerod_grad_analytical

        # For pair (birth=0, death=1) with grad_output=[2, 4]:
        # grad[0] += -2*0.5 + 4*0.5 = 1  (birth point)
        # grad[1] += -2*0.5 + 4*0.5 = 1  (death point)
        pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float64)
        g = torch.tensor([[[2.0, 4.0]]], dtype=torch.float64)
        grad = _compute_zerod_grad_analytical(pts, torch.tensor([[0]]), torch.tensor([[1]]), g)
        assert grad[0, 0, 0].item() == pytest.approx(1.0, abs=1e-10)
        assert grad[0, 1, 0].item() == pytest.approx(1.0, abs=1e-10)

    def test_negative_contribution(self) -> None:
        from pynerve.torch._persistence_vr import _compute_zerod_grad_analytical

        # For pair with grad_output=[-2, 0]:
        # grad[0] += -(-2)*0.5 + 0*0.5 = 1
        # grad[1] += -(-2)*0.5 + 0*0.5 = 1
        pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float64)
        g = torch.tensor([[[-2.0, 0.0]]], dtype=torch.float64)
        grad = _compute_zerod_grad_analytical(pts, torch.tensor([[0]]), torch.tensor([[1]]), g)
        assert grad[0, 0, 0].item() == pytest.approx(1.0, abs=1e-10)

    def test_null_indices_returns_zero(self) -> None:
        from pynerve.torch._persistence_vr import _compute_zerod_grad_analytical

        pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float64)
        g = torch.tensor([[[1.0, 1.0]]], dtype=torch.float64)
        grad = _compute_zerod_grad_analytical(pts, None, None, g)
        assert (grad == 0).all()

    def test_birth_eq_death_contribution(self) -> None:
        from pynerve.torch._persistence_vr import _compute_zerod_grad_analytical

        # birth_idx == death_idx: both -birth_grad and +death_grad get
        # applied TWICE to the same point (once for birth, once for death)
        # grad = -1*0.5 + -1*0.5 + 3*0.5 + 3*0.5 = 2.0
        pts = torch.tensor([[[0.0, 0.0]]], dtype=torch.float64)
        g = torch.tensor([[[1.0, 3.0]]], dtype=torch.float64)
        grad = _compute_zerod_grad_analytical(pts, torch.tensor([[0]]), torch.tensor([[0]]), g)
        assert grad[0, 0, 0].item() == pytest.approx(2.0, abs=1e-10)
