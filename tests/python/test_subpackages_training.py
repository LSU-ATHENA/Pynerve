"""Tests for pynerve training subpackage."""

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


class TestCoherentPerturbationSampler:
    def test_sample_perturbation_shape(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler

        sampler = CoherentPerturbationSampler(max_magnitude=0.1, seed=42)
        p = sampler.sample_perturbation((3, 4, 5))
        assert p.shape == (3, 4, 5)
        assert torch.isfinite(p).all()

    def test_apply_perturbation_shape_preserved(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler

        sampler = CoherentPerturbationSampler(max_magnitude=0.05, seed=42)
        pts = torch.randn(5, 3, 2)
        for noise_type in ("gaussian", "uniform", "scale"):
            result = sampler.apply_perturbation(pts, noise_type)
            assert result.shape == pts.shape
            assert torch.isfinite(result).all()

    def test_rotation_3d(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler

        sampler = CoherentPerturbationSampler(max_magnitude=0.1, seed=42, noise_types=["rotation"])
        pts = torch.randn(1, 5, 3)
        result = sampler.apply_perturbation(pts, "rotation")
        assert result.shape == pts.shape

    def test_reproducible(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler

        s1 = CoherentPerturbationSampler(max_magnitude=0.1, seed=42)
        s2 = CoherentPerturbationSampler(max_magnitude=0.1, seed=42)
        p1 = s1.sample_perturbation((5,))
        p2 = s2.sample_perturbation((5,))
        assert torch.equal(p1, p2)


class TestRobustTopologyTraining:
    def test_training_step_basic(self):
        from pynerve.training._stability_training import RobustTopologyTraining

        class DiagramModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 1)

            def forward(self, diagrams):
                flat = diagrams[0].view(1, -1)
                return self.linear(flat)

        model = DiagramModel()

        def fn(_pts):
            return [torch.tensor([[0.0, 0.5], [0.1, 0.8]]), torch.tensor([[0.0, 0.3]])]

        trainer = RobustTopologyTraining(model, fn, stability_weight=0.01, num_perturbations=2)
        pts = torch.randn(2, 3, 2)
        target = torch.tensor([[0.0]])
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        result = trainer.training_step(pts, target, torch.nn.MSELoss(), opt)
        assert "prediction_loss" in result
        assert "stability_loss" in result
        assert result["total_loss"] >= -1e-7
