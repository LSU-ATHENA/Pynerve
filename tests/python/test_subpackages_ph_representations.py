"""Tests for pynerve persistent homology representation subpackage."""

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


# diff / _ph_representations


class TestPersistenceLandscape:
    def test_landscape_shape_and_non_neg(self):
        from pynerve.diff.ph_layer import compute_persistence_landscape

        diagram = _make_diagram(5)
        out = compute_persistence_landscape(diagram, n_layers=3, resolution=50)
        assert out.shape == (3, 50)
        assert (out >= -1e-7).all(), "landscape values must be non-negative"

    def test_landscape_empty_diagram(self):
        from pynerve.diff.ph_layer import compute_persistence_landscape

        empty = torch.empty((0, 2))
        out = compute_persistence_landscape(empty, n_layers=2, resolution=20)
        assert out.shape == (2, 20)
        assert out.abs().max() == 0.0

    def test_landscape_gradient_flow(self):
        from pynerve.diff.ph_layer import compute_persistence_landscape

        diagram = _make_diagram(5).requires_grad_(True)
        out = compute_persistence_landscape(diagram, n_layers=2, resolution=30)
        loss = out.sum()
        loss.backward()
        assert diagram.grad is not None
        assert torch.isfinite(diagram.grad).all()

    def test_landscape_zero_persistence_all_zero(self):
        from pynerve.diff.ph_layer import compute_persistence_landscape

        d = torch.tensor([[0.5, 0.5], [0.7, 0.7]])
        out = compute_persistence_landscape(d, n_layers=2, resolution=20)
        assert (out == 0.0).all()

    def test_landscape_sorted_descending(self):
        from pynerve.diff.ph_layer import compute_persistence_landscape

        diagram = _make_diagram(10)
        out = compute_persistence_landscape(diagram, n_layers=4, resolution=40)
        for col in range(out.shape[1]):
            assert (out[:, col].diff() <= 1e-6).all(), "layers must be sorted descending"


class TestPersistenceImage:
    def test_image_shape(self):
        from pynerve.diff.ph_layer import persistence_image

        diagram = _make_diagram(5)
        out = persistence_image(diagram, resolution=20, sigma=0.1)
        assert out.shape == (20, 20)

    def test_empty_diagram(self):
        from pynerve.diff.ph_layer import persistence_image

        empty = torch.empty((0, 2))
        out = persistence_image(empty, resolution=10, sigma=0.1)
        assert out.abs().max() == 0.0

    def test_image_non_negative(self):
        from pynerve.diff.ph_layer import persistence_image

        diagram = _make_diagram(5)
        out = persistence_image(diagram, resolution=20, sigma=0.1)
        assert (out >= -1e-7).all()

    def test_image_gradient_flow(self):
        from pynerve.diff.ph_layer import persistence_image

        diagram = _make_diagram(5).requires_grad_(True)
        out = persistence_image(diagram, resolution=15, sigma=0.1)
        loss = out.sum()
        loss.backward()
        assert diagram.grad is not None
        assert torch.isfinite(diagram.grad).all()
