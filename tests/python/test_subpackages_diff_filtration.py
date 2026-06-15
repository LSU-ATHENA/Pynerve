"""Tests for pynerve differentiable filtration subpackage."""

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


# diff / ph_layer (pure-torch components)


class TestFiltrationLearningLayer:
    def test_forward_shape(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=3, hidden_dims=[16, 8])
        points = _make_points(2, 5, 3)
        out = layer(points)
        assert out.shape == (2, 5)
        assert torch.isfinite(out).all()

    def test_output_non_negative(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=2, hidden_dims=[4])
        pt = torch.randn(1, 3, 2)
        out = layer(pt)
        assert (out >= -1e-7).all(), "Softplus output should be non-negative"

    def test_gradient_flow(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=2, hidden_dims=[4])
        pt = torch.randn(1, 3, 2, requires_grad=True)
        out = layer(pt)
        loss = out.sum()
        loss.backward()
        assert pt.grad is not None
        assert torch.isfinite(pt.grad).all()

    def test_parameter_count(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=5, hidden_dims=[16, 8])
        total = sum(p.numel() for p in layer.parameters())
        assert total > 0


class TestLearnableFiltrationPersistence:
    def test_module_construction(self):
        from pynerve.diff.ph_layer import LearnableFiltrationPersistence

        lfs = LearnableFiltrationPersistence(input_dim=3, max_dim=1)
        assert isinstance(lfs.filtration, torch.nn.Module)
        params = list(lfs.parameters())
        assert len(params) > 0

    def test_filtration_forward(self):
        from pynerve.diff.ph_layer import LearnableFiltrationPersistence

        lfs = LearnableFiltrationPersistence(input_dim=3, max_dim=1)
        pt = _make_points(1, 5, 3)
        filt = lfs.filtration(pt)
        assert filt.shape == (1, 5)


class TestDifferentiableVietorisRips:
    def test_module_construction(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips

        rips = DifferentiableVietorisRips(max_dim=1)
        assert rips.max_dim == 1
        assert rips.max_radius is None

    def test_validation_dimensions(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips

        rips = DifferentiableVietorisRips(max_dim=1)
        with pytest.raises(ValueError):
            rips(torch.randn(5, 3))  # needs batch dim
        # forward with non-empty requires pynerve_internal C++ extension
        try:
            import pynerve_internal  # noqa: F401
        except ImportError:
            pytest.skip("pynerve_internal C++ extension not available")
