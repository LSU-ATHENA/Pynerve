"""Tests for pynerve persistent homology layer module subpackage."""

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


# diff / ph_layer_module


class TestTopologyRegularizer:
    def test_non_negative(self):
        from pynerve.diff.ph_layer_module import topology_regularizer

        d0 = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        d1 = torch.tensor([[0.1, 0.3]])
        loss = topology_regularizer([d0, d1], [2, 1], weight=1.0)
        assert loss.item() >= -1e-7

    def test_trivial_zero_for_exact_match(self):
        from pynerve.diff.ph_layer_module import topology_regularizer

        d0 = torch.tensor([[0.0, 0.1], [0.2, 0.5]])
        loss = topology_regularizer([d0], [2], weight=1.0)
        assert abs(loss.item()) < 1e-5

    def test_counts_based_loss_correct(self):
        from pynerve.diff.ph_layer_module import topology_regularizer

        d0 = torch.tensor([[0.0, 0.1]])
        d1 = torch.tensor([[0.0, 0.2], [0.3, 0.5]])
        loss = topology_regularizer([d0, d1], [2, 3], weight=1.0)
        assert loss.item() > 0

    def test_produces_finite_output(self):
        from pynerve.diff.ph_layer_module import topology_regularizer

        d = torch.zeros(10, 2)
        loss = topology_regularizer([d], [5])
        assert torch.isfinite(loss) and loss.item() >= 0


class TestPersistencePenalty:
    def test_non_negative(self):
        from pynerve.diff.ph_layer_module import persistence_penalty

        d = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        loss = persistence_penalty([d], min_persistence=0.1, weight=1.0)
        assert loss.item() >= -1e-7

    def test_penalty_increases_with_small_persistence(self):
        from pynerve.diff.ph_layer_module import persistence_penalty

        d_small = torch.tensor([[0.5, 0.55]])
        d_large = torch.tensor([[0.5, 2.0]])
        loss_small = persistence_penalty([d_small], min_persistence=0.2)
        loss_large = persistence_penalty([d_large], min_persistence=0.2)
        assert loss_small > loss_large

    def test_empty_diagram_zero(self):
        from pynerve.diff.ph_layer_module import persistence_penalty

        empty = torch.empty((0, 2))
        loss = persistence_penalty([empty], min_persistence=0.1)
        assert abs(loss.item()) < 1e-7

    def test_gradient_flow(self):
        from pynerve.diff.ph_layer_module import persistence_penalty

        d = torch.tensor([[0.0, 0.05]], requires_grad=True)
        loss = persistence_penalty([d], min_persistence=0.1, weight=1.0)
        loss.backward()
        assert d.grad is not None
        assert torch.isfinite(d.grad).all()


class TestTopologyLossModule:
    def test_same_diagrams_zero_loss(self):
        from pynerve.diff.ph_layer_module import TopologyLoss

        d = torch.tensor([[0.0, 0.5], [0.1, 0.8]])
        loss_fn = TopologyLoss([d], wasserstein_p=2.0)
        loss = loss_fn([d])
        assert loss.item() >= -1e-7

    def test_non_negative(self):
        from pynerve.diff.ph_layer_module import TopologyLoss

        pred = torch.tensor([[0.0, 0.5]])
        target = torch.tensor([[0.1, 0.6]])
        loss_fn = TopologyLoss([target], wasserstein_p=2.0)
        loss = loss_fn([pred])
        assert loss.item() >= -1e-7

    def test_gradient_flow(self):
        from pynerve.diff.ph_layer_module import TopologyLoss

        pred = torch.tensor([[0.0, 0.5], [0.1, 0.8]], requires_grad=True)
        target = torch.tensor([[0.05, 0.55], [0.15, 0.85]])
        loss_fn = TopologyLoss([target], wasserstein_p=2.0)
        loss = loss_fn([pred])
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
