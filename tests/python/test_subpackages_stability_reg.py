"""Tests for pynerve stability regularizer public API."""

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


# training / stability_reg (public API)


class TestStabilityRegPublicAPI:
    def test_imports_work(self):
        from pynerve.training.stability_reg import (
            CoherentPerturbationSampler,
            InterleavingRegularizer,
            PersistenceStabilityLoss,
            RobustTopologyTraining,
            StabilityRegularizer,
        )

        assert StabilityRegularizer is not None
        assert PersistenceStabilityLoss is not None
        assert InterleavingRegularizer is not None
        assert CoherentPerturbationSampler is not None
        assert RobustTopologyTraining is not None
