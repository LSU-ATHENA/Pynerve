"""Tests for pynerve stability subpackage."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


def _cuda_functional() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        torch.cuda.current_device()
        return True
    except Exception:
        return False


# synthetic test helpers


def _make_diagram(
    n_pairs: int = 5,
    max_val: float = 1.0,
    dims: bool = False,
    device: str = "cpu",
    seed: int = 42,
) -> torch.Tensor:
    """Construct a valid diagram tensor with birth < death."""
    rng = torch.Generator(device=device).manual_seed(seed)
    births = torch.rand(n_pairs, generator=rng, device=device) * max_val * 0.8
    deaths = births + torch.rand(n_pairs, generator=rng, device=device) * max_val * 0.4 + 0.01
    pairs = torch.stack([births, deaths], dim=1)
    if dims:
        dims_t = torch.randint(0, 3, (n_pairs, 1), generator=rng, device=device).float()
        return torch.cat([pairs, dims_t], dim=1)
    return pairs


def _make_points(
    batch: int = 2, n_points: int = 10, dim: int = 3, device: str = "cpu", seed: int = 42
) -> torch.Tensor:
    rng = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(batch, n_points, dim, generator=rng, device=device)


# training / _stability_regularizer


class TestStabilityRegularizer:
    def test_forward_non_negative(self):
        from pynerve.training._stability_regularizer import StabilityRegularizer

        reg = StabilityRegularizer(epsilon=0.01, num_perturbations=3, lambda_reg=0.1)

        def fn(_pts):
            return [torch.tensor([[0.0, 0.5], [0.1, 0.8]]), torch.tensor([[0.0, 0.3]])]

        pts = torch.randn(2, 5, 3)
        loss = reg(pts, fn)
        assert loss.item() >= -1e-7

    def test_non_negative_trivial(self):
        from pynerve.training._stability_regularizer import StabilityRegularizer

        reg = StabilityRegularizer(epsilon=0.0, num_perturbations=1, lambda_reg=0.1)

        d0 = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        d1 = torch.tensor([[0.0, 0.5], [0.1, 0.8]])

        def fn(pts):
            if pts is None:
                return [d0]
            return [d1]

        pts = torch.randn(1, 3, 2)
        loss = reg(pts, fn)
        assert loss.item() >= -1e-7

    def test_compute_theoretical_bound(self):
        from pynerve.training._stability_regularizer import StabilityRegularizer

        reg = StabilityRegularizer(epsilon=0.01, norm="wasserstein")
        bound = reg.compute_theoretical_bound(0.1, 100)
        assert bound >= 0
        assert bound == pytest.approx(0.1 * math.sqrt(100))

        reg_bn = StabilityRegularizer(epsilon=0.01, norm="bottleneck")
        assert reg_bn.compute_theoretical_bound(0.1, 100) == pytest.approx(0.1)

    def test_wasserstein_distance(self):
        from pynerve.training._stability_regularizer import StabilityRegularizer

        reg = StabilityRegularizer(norm="wasserstein")
        d1 = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        d2 = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        dist = reg.wasserstein_distance([d1], [d2])
        assert dist.item() < 0.01

    def test_bottleneck_distance(self):
        from pynerve.training._stability_regularizer import StabilityRegularizer

        reg = StabilityRegularizer(norm="bottleneck")
        d1 = torch.tensor([[0.0, 0.5]])
        d2 = torch.tensor([[0.0, 0.5]])
        dist = reg.bottleneck_distance([d1], [d2])
        assert dist.item() < 0.01

    def test_l2_diagram_distance(self):
        from pynerve.training._stability_regularizer import StabilityRegularizer

        reg = StabilityRegularizer(norm="l2")
        d1 = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        d2 = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        dist = reg.l2_diagram_distance([d1], [d2])
        assert dist.item() < 0.01

    def test_gradient_flow(self):
        from pynerve.training._stability_regularizer import StabilityRegularizer

        reg = StabilityRegularizer(epsilon=0.01, num_perturbations=2, lambda_reg=0.1)

        def fn(_pts):
            return [torch.tensor([[0.0, 0.5], [0.1, 0.8]]), torch.tensor([[0.0, 0.3]])]

        pts = torch.randn(1, 5, 2, requires_grad=True)
        loss = reg(pts, fn) + 0.0 * pts.sum()
        loss.backward()
        assert pts.grad is not None
        assert torch.isfinite(pts.grad).all()


# training / _stability_training


class TestPersistenceStabilityLoss:
    def test_non_negative(self):
        from pynerve.training._stability_training import PersistenceStabilityLoss

        loss_fn = PersistenceStabilityLoss(stability_weight=0.1)
        feats = torch.randn(10, 5)
        pert = feats + 0.01 * torch.randn(10, 5)
        loss = loss_fn(feats, pert, 0.01)
        assert loss.item() >= -1e-7

    def test_matching_same_features_trivial(self):
        from pynerve.training._stability_training import PersistenceStabilityLoss

        loss_fn = PersistenceStabilityLoss(stability_weight=1.0, lipschitz_constant=100.0)
        feats = torch.randn(10, 5)
        loss = loss_fn(feats, feats, 0.0)
        assert abs(loss.item()) < 1e-7

    def test_gradient_flow(self):
        from pynerve.training._stability_training import PersistenceStabilityLoss

        loss_fn = PersistenceStabilityLoss(stability_weight=0.1)
        feats = torch.randn(10, 5, requires_grad=True)
        pert = feats.detach() + 0.1 * torch.randn_like(feats)
        loss = loss_fn(feats, pert, 0.1)
        loss.backward()
        assert feats.grad is not None
        assert torch.isfinite(feats.grad).all()


class TestInterleavingRegularizer:
    def test_non_negative(self):
        from pynerve.training._stability_training import InterleavingRegularizer

        reg = InterleavingRegularizer(lambda_reg=0.1)
        f1 = torch.randn(10, 5)
        f2 = torch.randn(10, 5)
        loss = reg(f1, f2)
        assert loss.item() >= -1e-7

    def test_identical_zero(self):
        from pynerve.training._stability_training import InterleavingRegularizer

        reg = InterleavingRegularizer(lambda_reg=0.1)
        f = torch.randn(10, 5)
        loss = reg(f, f)
        assert abs(loss.item()) < 1e-7

    def test_gradient_flow(self):
        from pynerve.training._stability_training import InterleavingRegularizer

        reg = InterleavingRegularizer(lambda_reg=0.1)
        f1 = torch.randn(10, 5, requires_grad=True)
        f2 = torch.randn(10, 5)
        loss = reg(f1, f2)
        loss.backward()
        assert f1.grad is not None
        assert torch.isfinite(f1.grad).all()
