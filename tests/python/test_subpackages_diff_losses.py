"""Tests for pynerve differentiable topology loss subpackage."""

from __future__ import annotations

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


# diff / topology_loss


class TestPersistenceLoss:
    def test_softmin_basic(self):
        from pynerve.diff.topology_loss import PersistenceLoss

        x = torch.tensor([1.0, 2.0, 3.0])
        result = PersistenceLoss.softmin(x, temperature=0.1)
        assert result.item() <= x.numpy().min() + 0.5

    def test_softmax_basic(self):
        from pynerve.diff.topology_loss import PersistenceLoss

        x = torch.tensor([1.0, 2.0, 3.0])
        result = PersistenceLoss.softmax(x, temperature=1.0)
        assert result.shape == (3,)
        assert (result >= 0).all() and abs(result.sum().item() - 1.0) < 1e-5

    def test_diagram_wasserstein_non_negative(self):
        from pynerve.diff.topology_loss import PersistenceLoss

        d1 = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        d2 = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        w = PersistenceLoss.diagram_wasserstein(d1, d2)
        assert w.item() >= -1e-7

    def test_diagram_wasserstein_self_zero(self):
        from pynerve.diff.topology_loss import PersistenceLoss

        d = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        w = PersistenceLoss.diagram_wasserstein(d, d)
        assert w.item() < 0.15  # approximate, not exact due to Sinkhorn approximation

    def test_diagram_wasserstein_empty(self):
        from pynerve.diff.topology_loss import PersistenceLoss

        e1 = torch.empty((0, 2))
        e2 = torch.empty((0, 2))
        w = PersistenceLoss.diagram_wasserstein(e1, e2)
        assert w.item() == 0.0

    def test_diagram_wasserstein_one_empty(self):
        from pynerve.diff.topology_loss import PersistenceLoss

        d = torch.tensor([[0.0, 0.5]])
        e = torch.empty((0, 2))
        w = PersistenceLoss.diagram_wasserstein(d, e)
        assert w.item() >= 0

    def test_diagram_wasserstein_p1(self):
        from pynerve.diff.topology_loss import PersistenceLoss

        d1 = torch.tensor([[0.0, 1.0], [0.2, 0.5]])
        d2 = torch.tensor([[0.0, 1.0], [0.2, 0.6]])
        w = PersistenceLoss.diagram_wasserstein(d1, d2, p=1)
        assert torch.isfinite(w) and w.item() >= 0

    def test_diagram_bottleneck_non_negative(self):
        from pynerve.diff.topology_loss import PersistenceLoss

        d1 = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        d2 = torch.tensor([[0.0, 0.6], [0.2, 0.7]])
        b = PersistenceLoss.diagram_bottleneck(d1, d2)
        assert b.item() >= -1e-7

    def test_diagram_bottleneck_empty(self):
        from pynerve.diff.topology_loss import PersistenceLoss

        e1 = torch.empty((0, 2))
        e2 = torch.empty((0, 2))
        b = PersistenceLoss.diagram_bottleneck(e1, e2)
        assert b.item() == 0.0

    def test_persistence_kernel_non_negative(self):
        from pynerve.diff.topology_loss import PersistenceLoss

        d1 = torch.tensor([[0.0, 1.0], [0.2, 0.5]])
        d2 = torch.tensor([[0.1, 0.8], [0.3, 0.6]])
        k = PersistenceLoss.persistence_kernel(d1, d2, sigma=1.0)
        assert k.item() >= -1e-7

    def test_persistence_kernel_empty(self):
        from pynerve.diff.topology_loss import PersistenceLoss

        e = torch.empty((0, 2))
        d = torch.tensor([[0.0, 0.5]])
        k = PersistenceLoss.persistence_kernel(d, e)
        assert k.item() == 0.0

    def test_wasserstein_gradient_flow(self):
        from pynerve.diff.topology_loss import PersistenceLoss

        d1 = torch.tensor([[0.1, 0.6], [0.3, 0.9]], requires_grad=True)
        d2 = torch.tensor([[0.15, 0.55], [0.25, 0.85]])
        w = PersistenceLoss.diagram_wasserstein(d1, d2, temperature=0.5)
        w.backward()
        assert d1.grad is not None
        assert torch.isfinite(d1.grad).all()


class TestBettiNumberLoss:
    def test_non_negative(self):
        from pynerve.diff.topology_loss import BettiNumberLoss

        diagram = _make_diagram(10, dims=True)
        target = torch.tensor([3, 2])
        loss_fn = BettiNumberLoss(threshold=0.1, temperature=0.1)
        loss = loss_fn(diagram, target)
        assert loss.item() >= -1e-7

    def test_gradient_flow(self):
        from pynerve.diff.topology_loss import BettiNumberLoss

        diagram = _make_diagram(10, dims=True).requires_grad_(True)
        target = torch.tensor([2, 1, 1])
        loss_fn = BettiNumberLoss()
        loss = loss_fn(diagram, target)
        loss.backward()
        assert diagram.grad is not None
        assert torch.isfinite(diagram.grad).all()


class TestTopologicalComplexityLoss:
    def test_empty_zero(self):
        from pynerve.diff.topology_loss import TopologicalComplexityLoss

        empty = torch.empty((0, 2))
        lf = TopologicalComplexityLoss("total_persistence")
        assert abs(lf(empty).item()) < 1e-7

    def test_non_negative(self):
        from pynerve.diff.topology_loss import TopologicalComplexityLoss

        d = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        for measure in ("total_persistence", "max_persistence"):
            lf = TopologicalComplexityLoss(measure)
            assert lf(d).item() >= -1e-7

    def test_entropy_non_negative(self):
        from pynerve.diff.topology_loss import TopologicalComplexityLoss

        d = torch.tensor([[0.0, 1.0], [0.2, 0.5]])
        lf = TopologicalComplexityLoss("persistence_entropy")
        loss = lf(d)
        assert loss.item() >= -1e-7

    def test_num_features(self):
        from pynerve.diff.topology_loss import TopologicalComplexityLoss

        d = torch.tensor([[0.0, 0.2], [0.3, 0.4], [0.5, 0.6]])
        lf = TopologicalComplexityLoss("num_features")
        assert lf(d).item() >= 0

    def test_gradient_flow(self):
        from pynerve.diff.topology_loss import TopologicalComplexityLoss

        d = torch.tensor([[0.0, 0.5], [0.1, 0.8]], requires_grad=True)
        lf = TopologicalComplexityLoss("total_persistence")
        loss = lf(d)
        loss.backward()
        assert d.grad is not None
        assert torch.isfinite(d.grad).all()

    def test_unknown_measure_raises(self):
        from pynerve.diff.topology_loss import TopologicalComplexityLoss

        with pytest.raises(ValueError):
            TopologicalComplexityLoss("invalid")


class TestStabilityLoss:
    def test_empty_diagrams_zero(self):
        from pynerve.diff.topology_loss import StabilityLoss

        def fn(_pts):
            return [torch.empty((0, 2))]

        sl = StabilityLoss(epsilon=0.01, num_samples=2)
        pts = torch.zeros(1, 3, 2)
        loss = sl(pts, fn)
        assert abs(loss.item()) < 1e-7


class TestMultiScaleTopologyLoss:
    def test_non_negative(self):
        from pynerve.diff.topology_loss import MultiScaleTopologyLoss

        d = torch.tensor([[0.0, 0.5], [0.1, 0.8], [0.2, 1.0]])
        targets = [d.clone() for _ in range(4)]
        lf = MultiScaleTopologyLoss(scales=(0.01, 0.1, 0.5, 1.0))
        loss = lf(d, targets)
        assert loss.item() >= -1e-7

    def test_gradient_flow(self):
        from pynerve.diff.topology_loss import MultiScaleTopologyLoss

        d = torch.tensor([[0.1, 0.6], [0.3, 0.9], [0.5, 1.1]], requires_grad=True)
        targets = [torch.tensor([[0.15, 0.55], [0.25, 0.85]]) for _ in range(4)]
        lf = MultiScaleTopologyLoss(scales=(0.01, 0.1, 0.5, 1.0))
        loss = lf(d, targets)
        loss.backward()
        assert d.grad is not None
        assert torch.isfinite(d.grad).all()


class TestLandscapeLoss:
    def test_non_negative(self):
        from pynerve.diff.topology_loss import LandscapeLoss

        d1 = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        d2 = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        lf = LandscapeLoss(n_layers=3, resolution=50)
        loss = lf(d1, d2)
        assert loss.item() >= -1e-7

    def test_self_distance_zero(self):
        from pynerve.diff.topology_loss import LandscapeLoss

        d = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        lf = LandscapeLoss(n_layers=3, resolution=50)
        loss = lf(d, d)
        assert abs(loss.item()) < 1e-5

    def test_gradient_flow(self):
        from pynerve.diff.topology_loss import LandscapeLoss

        d1 = torch.tensor([[0.0, 0.5], [0.1, 0.8]], requires_grad=True)
        d2 = torch.tensor([[0.0, 0.6], [0.2, 0.7]])
        lf = LandscapeLoss(n_layers=3, resolution=50)
        loss = lf(d1, d2)
        loss.backward()
        assert d1.grad is not None
        assert torch.isfinite(d1.grad).all()


class TestCombinedTopologyLoss:
    def test_returns_dict(self):
        from pynerve.diff.topology_loss import TopologyLoss as CombinedTopologyLoss

        pred = _make_diagram(5, dims=True)
        target = _make_diagram(4, dims=True, seed=99)
        betti = torch.tensor([2, 1])
        loss_fn = CombinedTopologyLoss(
            wasserstein_weight=1.0,
            betti_weight=0.1,
            complexity_weight=0.01,
        )
        result = loss_fn(pred, target, target_betti=betti)
        assert isinstance(result, dict)
        assert "total" in result
        assert result["total"].item() >= -1e-7

    def test_gradient_flow(self):
        from pynerve.diff.topology_loss import TopologyLoss as CombinedTopologyLoss

        pred = _make_diagram(5, dims=True).requires_grad_(True)
        target = _make_diagram(4, dims=True, seed=99)
        betti = torch.tensor([2, 1])
        loss_fn = CombinedTopologyLoss(wasserstein_weight=1.0)
        result = loss_fn(pred, target, target_betti=betti)
        result["total"].backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
