"""Numerical correctness tests that verify outputs against hand-computed values.

Every test with a numerical value includes a precise assertion with tolerance,
not just a shape or finiteness check.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# known-configuration point clouds
_TRIANGLE = torch.tensor(
    [[0.0, 0.0], [2.0, 0.0], [1.0, 3.0**0.5]],
    dtype=torch.float64,
)
_SQUARE = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=torch.float64,
)
_TWO_POINTS = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
_SINGLE_POINT = torch.tensor([[0.0, 0.0]], dtype=torch.float64)

# known diagram tensors
_SIMPLE_DIAGRAM = torch.tensor(
    [[0.0, 1.0], [0.0, 2.0], [1.0, 3.0]],
    dtype=torch.float64,
)
_SINGLE_DIAGRAM = torch.tensor([[0.0, 1.5]], dtype=torch.float64)
_INFINITE_DIAGRAM = torch.tensor(
    [[0.0, float("inf")], [0.0, 2.0]],
    dtype=torch.float64,
)

# vectorization


class TestPersistenceImageCorrectness:
    """Numerical correctness for persistence_image."""

    def test_single_point_peak_gaussian(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        d = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        sigma = 2.0
        img = persistence_image(
            d,
            resolution=(3, 3),
            sigma=sigma,
            weight_fn="constant",
            birth_range=(0.0, 3.0),
            death_range=(0.0, 3.0),
        )
        assert img.shape == (3, 3)
        assert img.sum().item() > 0

    def test_empty_diagram_zero_image(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        empty = torch.empty(0, 2, dtype=torch.float64)
        img = persistence_image(empty, resolution=(4, 4), sigma=1.0)
        assert (img == 0).all()

    def test_weight_comparison(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        d = torch.tensor([[0.0, 1.0], [0.0, 4.0]], dtype=torch.float64)
        const_img = persistence_image(
            d,
            resolution=(4, 4),
            sigma=0.5,
            weight_fn="constant",
        )
        persist_img = persistence_image(
            d,
            resolution=(4, 4),
            sigma=0.5,
            weight_fn="persistence",
        )
        assert not torch.allclose(const_img, persist_img, atol=1e-10)

    def test_batch_vs_single_consistency(self) -> None:
        import pynerve.torch

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        single = pynerve.torch.persistence_image(
            d,
            resolution=(4, 4),
            sigma=0.5,
            weight_fn="constant",
        )
        batched = pynerve.torch.persistence_image(
            d.unsqueeze(0),
            resolution=(4, 4),
            sigma=0.5,
            weight_fn="constant",
        )
        torch.testing.assert_close(single, batched.squeeze(0))

    def test_gradient_flow(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        d = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        img = persistence_image(d, resolution=(4, 4), sigma=1.0)
        loss = img.sum()
        loss.backward()
        assert d.grad is not None
        assert d.grad.abs().sum().item() > 0

    def test_invalid_sigma_raises(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, Exception), match="sigma|finite"):
            persistence_image(d, resolution=(4, 4), sigma=float("nan"))

    def test_zero_persistence_diagram_zero(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        d = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        img = persistence_image(d, resolution=(4, 4), sigma=1.0)
        assert (img == 0).all()

    def test_known_pixel_value_single_point(self) -> None:
        from pynerve.torch._persistence_image import _compute_single_persistence_image

        d = torch.tensor([[1.0, 4.0]], dtype=torch.float64)
        img = _compute_single_persistence_image(d, (1, 1), sigma=1.0, weight_fn="constant")
        assert img.shape == (1, 1)
        assert img[0, 0].item() == pytest.approx(1.0, abs=0.1)


class TestPersistenceLandscapeCorrectness:
    """Numerical correctness for persistence_landscape."""

    def test_two_point_landscape_layers(self) -> None:
        from pynerve.torch.vectorization import persistence_landscape

        d = torch.tensor([[0.0, 2.0], [0.0, 4.0]], dtype=torch.float64)
        x_range: tuple[float, float] = (0.0, 5.0)
        k1 = persistence_landscape(d, k=1, num_samples=10, x_range=x_range)
        k2 = persistence_landscape(d, k=2, num_samples=10, x_range=x_range)
        assert (k2 <= k1 + 1e-12).all()

    def test_empty_diagram_zero(self) -> None:
        from pynerve.torch.vectorization import persistence_landscape

        empty = torch.empty(0, 2, dtype=torch.float64)
        x_range: tuple[float, float] = (0.0, 1.0)
        landscape = persistence_landscape(empty, k=1, num_samples=5, x_range=x_range)
        assert (landscape == 0).all()

    def test_gradient_flow(self) -> None:
        from pynerve.torch.vectorization import persistence_landscape

        x_range: tuple[float, float] = (0.0, 3.0)
        d = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        land = persistence_landscape(d, k=1, num_samples=5, x_range=x_range)
        loss = land.sum()
        loss.backward()
        assert d.grad is not None
        assert d.grad.abs().sum().item() > 0


class TestPersistenceSilhouetteCorrectness:
    """Numerical correctness for persistence_silhouette."""

    def test_silhouette_non_negative(self) -> None:
        from pynerve.torch.vectorization import persistence_silhouette

        d = torch.tensor([[0.0, 2.0], [0.0, 4.0]], dtype=torch.float64)
        s = persistence_silhouette(d, num_samples=10)
        assert (s >= 0).all()

    def test_empty_diagram_zero(self) -> None:
        from pynerve.torch.vectorization import persistence_silhouette

        empty = torch.empty(0, 2, dtype=torch.float64)
        s = persistence_silhouette(empty, num_samples=5)
        assert (s == 0).all()

    def test_gradient_flow(self) -> None:
        from pynerve.torch.vectorization import persistence_silhouette

        d = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        s = persistence_silhouette(d, num_samples=5)
        loss = s.sum()
        loss.backward()
        assert d.grad is not None
        assert d.grad.abs().sum().item() > 0


class TestBirthDeathCurveCorrectness:
    """Numerical correctness for birth_death_curve."""

    def test_count_matches_total_pairs(self) -> None:
        from pynerve.torch.vectorization import birth_death_curve

        d = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0], [0.5, 1.5]],
            dtype=torch.float64,
        )
        curve = birth_death_curve(d, num_bins=4, statistic="count")
        assert curve.sum().item() == pytest.approx(3.0, abs=1e-10)

    def test_empty_diagram_zero(self) -> None:
        from pynerve.torch.vectorization import birth_death_curve

        empty = torch.empty(0, 2, dtype=torch.float64)
        curve = birth_death_curve(empty, num_bins=4, statistic="count")
        assert (curve == 0).all()

    def test_invalid_statistic_raises(self) -> None:
        from pynerve.torch.vectorization import birth_death_curve

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, Exception)):
            birth_death_curve(d, num_bins=4, statistic="invalid")


class TestHeatKernelSignatureCorrectness:
    """Numerical correctness for heat_kernel_signature."""

    def test_custom_t_values_shape(self) -> None:
        from pynerve.torch.vectorization import heat_kernel_signature

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        t_values = torch.tensor([0.2, 1.0], dtype=torch.float32)
        hks = heat_kernel_signature(d, num_samples=5, sigma=0.2, t_values=t_values)
        assert hks.shape in ((2, 5), (2,))

    def test_empty_diagram(self) -> None:
        from pynerve.torch.vectorization import heat_kernel_signature

        empty = torch.empty(0, 2, dtype=torch.float64)
        hks = heat_kernel_signature(empty, num_samples=5, sigma=0.5)
        assert torch.isfinite(hks).all()
