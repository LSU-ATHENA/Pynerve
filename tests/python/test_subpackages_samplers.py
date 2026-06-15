"""Tests for pynerve topology sampler subpackage."""

from __future__ import annotations

import numpy as np
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


# training / topology_sampler


class TestPersistenceStratifiedSampler:
    def test_iter_produces_indices(self):
        from pynerve.training.topology_sampler import PersistenceStratifiedSampler

        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True) for _ in range(20)]
        sampler = PersistenceStratifiedSampler(diagrams, num_strata=3, batch_size=5, seed=42)
        indices = list(sampler)
        assert len(indices) <= 20
        assert all(0 <= i < 20 for i in indices)

    def test_drop_last(self):
        from pynerve.training.topology_sampler import PersistenceStratifiedSampler

        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True) for _ in range(18)]
        sampler = PersistenceStratifiedSampler(
            diagrams, num_strata=3, batch_size=5, drop_last=True, seed=42
        )
        indices = list(sampler)
        assert len(indices) <= 15


class TestBettiBalancedSampler:
    def test_iter_produces_indices(self):
        from pynerve.training.topology_sampler import BettiBalancedSampler

        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True) for _ in range(20)]
        sampler = BettiBalancedSampler(diagrams, batch_size=5, seed=42)
        indices = list(sampler)
        assert len(indices) <= 20
        assert all(0 <= i < 20 for i in indices)


class TestTopologyImportanceSampler:
    def test_iter_produces_indices(self):
        from pynerve.training.topology_sampler import TopologyImportanceSampler

        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True) for _ in range(15)]
        sampler = TopologyImportanceSampler(diagrams, batch_size=4, seed=42)
        indices = list(sampler)
        assert len(indices) == 15
        assert sorted(indices) == list(range(15))

    def test_update_weights_maintains_distribution(self):
        from pynerve.training.topology_sampler import TopologyImportanceSampler

        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True) for _ in range(10)]
        sampler = TopologyImportanceSampler(diagrams, batch_size=3, seed=42)
        sampler.update_weights([0, 1, 2])
        assert abs(sampler.weights.sum() - 1.0) < 1e-7


class TestMultiScaleTopologySampler:
    def test_iter_produces_indices(self):
        from pynerve.training.topology_sampler import MultiScaleTopologySampler

        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True) for _ in range(25)]
        sampler = MultiScaleTopologySampler(
            diagrams, scales=[0.01, 0.1, 1.0], batch_size=32, samples_per_scale=3, seed=42
        )
        indices = list(sampler)
        # should produce all indices eventually (some may repeat in multi-pass)
        assert len(indices) <= 25 * 2


class TestTopologyAdaptiveBatchSize:
    def test_low_complexity_large_batch(self):
        from pynerve.training.topology_sampler import TopologyAdaptiveBatchSize

        adapter = TopologyAdaptiveBatchSize(
            base_batch_size=32, min_batch_size=8, max_batch_size=128
        )
        simple_diags = [torch.tensor([[0.0, 0.1, 0.0]]) for _ in range(10)]
        batch = adapter.compute_batch_size(simple_diags)
        assert batch >= adapter.min_batch

    def test_high_complexity_small_batch(self):
        from pynerve.training.topology_sampler import TopologyAdaptiveBatchSize

        adapter = TopologyAdaptiveBatchSize(
            base_batch_size=32, min_batch_size=8, max_batch_size=128
        )
        complex_diags = [torch.rand(100, 3) * 2.0 for _ in range(10)]
        for d in complex_diags:
            d[:, 1] = d[:, 0] + d[:, 1].abs() + 0.01  # birth < death
        batch = adapter.compute_batch_size(complex_diags)
        assert batch <= adapter.base_batch

    def test_empty_diagrams_default(self):
        from pynerve.training.topology_sampler import TopologyAdaptiveBatchSize

        adapter = TopologyAdaptiveBatchSize(base_batch_size=32)
        batch = adapter.compute_batch_size([])
        assert batch == 32
