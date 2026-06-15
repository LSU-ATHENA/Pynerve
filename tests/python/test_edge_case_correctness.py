"""Edge case and extremal-value tests for statistics and preprocessing.

Tests that the library handles degenerate, boundary, and extreme inputs
with correct finite results (not NaN, not crash).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


class TestEdgeCaseStatistics:
    """Statistics with degenerate diagram configurations."""

    def test_total_persistence_inf_death_pairs_raises(self) -> None:
        from pynerve.torch.statistics import total_persistence

        d = torch.tensor([[0.0, float("inf")], [0.0, float("inf")]], dtype=torch.float64)
        with pytest.raises((ValueError, Exception), match="finite"):
            total_persistence(d, p=1.0)

    def test_single_point(self) -> None:
        from pynerve.torch.statistics import total_persistence

        d = torch.tensor([[0.0, 0.5]], dtype=torch.float64)
        tp = total_persistence(d, p=1.0)
        assert tp.item() == pytest.approx(0.5, abs=1e-10)

    def test_total_persistence_many_points(self) -> None:
        from pynerve.torch.statistics import total_persistence

        n = 10000
        births = torch.zeros(n, dtype=torch.float64)
        deaths = torch.full((n,), 0.001, dtype=torch.float64)
        d = torch.stack([births, deaths], dim=1)
        tp = total_persistence(d, p=1.0)
        assert tp.item() == pytest.approx(10.0, abs=1e-6)

    def test_total_persistence_very_small_values(self) -> None:
        from pynerve.torch.statistics import total_persistence

        d = torch.tensor([[0.0, 1e-10], [0.0, 2e-10]], dtype=torch.float64)
        tp = total_persistence(d, p=1.0)
        assert tp.item() == pytest.approx(3e-10, abs=1e-20)
        assert tp.item() > 0

    def test_total_persistence_very_large_values(self) -> None:
        from pynerve.torch.statistics import total_persistence

        d = torch.tensor([[0.0, 1e10], [0.0, 2e10]], dtype=torch.float64)
        tp = total_persistence(d, p=1.0)
        assert tp.item() == pytest.approx(3e10, abs=1e-4)
        assert torch.isfinite(tp)

    def test_entropy_single_element(self) -> None:
        from pynerve.torch.statistics import persistence_entropy

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        ent = persistence_entropy(d)
        assert ent.item() == pytest.approx(0.0, abs=1e-10)

    def test_entropy_three_identical_persistences(self) -> None:
        from pynerve.torch.statistics import persistence_entropy

        d = torch.tensor([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]], dtype=torch.float64)
        ent = persistence_entropy(d)
        expected = -math.log(1.0 / 3)
        assert ent.item() == pytest.approx(expected, abs=1e-10)

    def test_entropy_only_inf_deaths_raises(self) -> None:
        from pynerve.torch.statistics import persistence_entropy

        d = torch.tensor([[0.0, float("inf")], [0.0, float("inf")]], dtype=torch.float64)
        with pytest.raises((ValueError, Exception), match="finite"):
            persistence_entropy(d)

    def test_entropy_mixed_finite_inf_raises(self) -> None:
        from pynerve.torch.statistics import persistence_entropy

        d = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0], [0.0, float("inf")]],
            dtype=torch.float64,
        )
        with pytest.raises((ValueError, Exception), match="finite"):
            persistence_entropy(d)

    def test_number_of_features_all_inf(self) -> None:
        from pynerve.torch.statistics import number_of_features

        d = torch.tensor([[0.0, float("inf")], [0.0, float("inf")]], dtype=torch.float64)
        n = number_of_features(d, min_persistence=0.0)
        assert n.item() >= 0

    def test_amplitude_with_only_inf_raises(self) -> None:
        from pynerve.torch.statistics import amplitude

        d = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        for metric in ("bottleneck", "wasserstein", "persistence"):
            with pytest.raises((ValueError, Exception), match="finite"):
                amplitude(d, metric=metric)

    def test_mean_persistence_very_small(self) -> None:
        from pynerve.torch.statistics import mean_persistence

        d = torch.tensor([[0.0, 1e-15]], dtype=torch.float64)
        mp = mean_persistence(d)
        assert mp.item() == pytest.approx(1e-15, abs=1e-20)

    def test_persistence_variance_single_point(self) -> None:
        from pynerve.torch.statistics import persistence_variance

        d = torch.tensor([[0.0, 5.0]], dtype=torch.float64)
        pv = persistence_variance(d)
        assert pv.item() == pytest.approx(0.0, abs=1e-10)

    def test_betti_curve_all_essential_raises(self) -> None:
        from pynerve.torch.statistics import betti_curve

        d = torch.tensor([[0.0, float("inf")], [0.0, float("inf")]], dtype=torch.float64)
        with pytest.raises((ValueError, Exception), match="finite"):
            betti_curve(d, num_samples=5)


class TestEdgeCasePreprocessing:
    """Preprocessing with edge-case inputs."""

    def test_normalize_minmax_single_element(self) -> None:
        from pynerve.torch.preprocessing import normalize_diagram

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        normed = normalize_diagram(d, method="minmax")
        assert normed.shape == d.shape
        assert torch.isfinite(normed).all()

    def test_normalize_standard_single_element(self) -> None:
        from pynerve.torch.preprocessing import normalize_diagram

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        normed = normalize_diagram(d, method="standard")
        assert torch.isfinite(normed).all()

    def test_threshold_empty_diagram(self) -> None:
        from pynerve.torch.preprocessing import threshold_diagram

        d = torch.empty(0, 2, dtype=torch.float64)
        filtered = threshold_diagram(d, min_persistence=0.5)
        assert filtered.shape[0] == 0

    def test_threshold_removes_all(self) -> None:
        from pynerve.torch.preprocessing import threshold_diagram

        d = torch.tensor([[0.0, 0.1], [0.0, 0.2]], dtype=torch.float64)
        filtered = threshold_diagram(d, min_persistence=1.0)
        assert filtered.shape[0] == 0

    def test_threshold_with_max_persistence(self) -> None:
        from pynerve.torch.preprocessing import threshold_diagram

        d = torch.tensor(
            [[0.0, 0.5], [0.0, 2.0], [0.0, 5.0]],
            dtype=torch.float64,
        )
        filtered = threshold_diagram(d, min_persistence=1.0, max_persistence=3.0)
        assert filtered.shape[0] == 1
        assert filtered[0, 1].item() == pytest.approx(2.0, abs=1e-10)

    def test_subsample_all_strategies(self) -> None:
        from pynerve.torch.preprocessing import subsample_diagram

        d = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]],
            dtype=torch.float64,
        )
        for strategy in ("most_persistent", "uniform", "random"):
            result = subsample_diagram(d, max_features=2, strategy=strategy)
            assert result.shape[0] <= 2

    def test_subsample_most_persistent_ordering(self) -> None:
        from pynerve.torch.preprocessing import subsample_diagram

        d = torch.tensor(
            [[0.0, 1.0], [0.0, 10.0], [0.0, 5.0], [0.0, 20.0]],
            dtype=torch.float64,
        )
        result = subsample_diagram(d, max_features=2, strategy="most_persistent")
        assert result[0, 1].item() == pytest.approx(20.0, abs=1e-10)
        assert result[1, 1].item() == pytest.approx(10.0, abs=1e-10)

    def test_handle_inf_all_strategies(self) -> None:
        from pynerve.torch.preprocessing import handle_infinite_deaths

        d = torch.tensor([[0.0, 1.0], [0.0, float("inf")]], dtype=torch.float64)
        for strategy in ("max", "remove", "large_value"):
            if strategy == "large_value":
                result = handle_infinite_deaths(d, strategy=strategy, large_value_factor=50.0)
            else:
                result = handle_infinite_deaths(d, strategy=strategy)
            assert result.shape[-1] == 2

    def test_clean_diagram_without_normalization(self) -> None:
        from pynerve.torch.preprocessing import clean_diagram

        d = torch.tensor(
            [[0.0, float("inf")], [0.0, 2.0], [0.0, 0.05]],
            dtype=torch.float64,
        )
        cleaned = clean_diagram(d, handle_inf=True, min_persistence=0.5, normalize=False)
        assert torch.isfinite(cleaned).all()
        assert cleaned.shape[0] <= 2

    def test_clean_diagram_with_normalization(self) -> None:
        from pynerve.torch.preprocessing import clean_diagram

        d = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            dtype=torch.float64,
        )
        cleaned = clean_diagram(d, handle_inf=True, min_persistence=0.0, normalize=True)
        assert (cleaned >= 0).all() and (cleaned <= 1).all()


class TestEdgeCaseVectorization:
    """Vectorization with degenerate diagram inputs."""

    def test_persistence_image_large_sigma(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        img = persistence_image(d, resolution=(4, 4), sigma=100.0)
        assert (img >= 0).all()
        assert torch.isfinite(img).all()

    def test_persistence_image_tiny_sigma(self) -> None:
        from pynerve.torch.vectorization import persistence_image

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        img = persistence_image(d, resolution=(4, 4), sigma=1e-6)
        assert (img >= 0).all()
        assert torch.isfinite(img).all()

    def test_landscape_many_layers(self) -> None:
        from pynerve.torch.vectorization import persistence_landscape

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        x_range: tuple[float, float] = (0.0, 3.0)
        land = persistence_landscape(d, k=10, num_samples=5, x_range=x_range)
        assert land.shape == (10, 5)
        assert torch.isfinite(land).all()

    def test_landscape_k_greater_than_points(self) -> None:
        from pynerve.torch.vectorization import persistence_landscape

        d = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        x_range: tuple[float, float] = (0.0, 3.0)
        land = persistence_landscape(d, k=5, num_samples=4, x_range=x_range)
        assert (land[1:] == 0).all()

    def test_silhouette_with_entirely_inf(self) -> None:
        from pynerve.torch.vectorization import persistence_silhouette

        d = torch.tensor(
            [[0.0, float("inf")], [0.0, float("inf")]],
            dtype=torch.float64,
        )
        s = persistence_silhouette(d, num_samples=5)
        assert (s == 0).all()

    def test_birth_death_curve_many_bins(self) -> None:
        from pynerve.torch.vectorization import birth_death_curve

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=torch.float64)
        curve = birth_death_curve(d, num_bins=100, statistic="count")
        assert curve.shape == (100,)
        assert curve.sum().item() == pytest.approx(3.0, abs=1e-10)

    def test_birth_death_curve_sum_statistic(self) -> None:
        from pynerve.torch.vectorization import birth_death_curve

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=torch.float64)
        for stat in ("count", "sum", "mean"):
            curve = birth_death_curve(d, num_bins=4, statistic=stat)
            assert curve.shape == (4,)
            assert torch.isfinite(curve).all()


class TestEdgeCaseDistance:
    """Distance functions with edge case diagrams."""

    def test_wasserstein_both_empty(self) -> None:
        from pynerve.torch import diagram_wasserstein

        e = torch.empty(0, 2, dtype=torch.float64)
        dist = diagram_wasserstein(e, e)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_bottleneck_both_empty(self) -> None:
        from pynerve.torch import diagram_bottleneck

        e = torch.empty(0, 2, dtype=torch.float64)
        dist = diagram_bottleneck(e, e)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_wasserstein_identical_3d(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float64)
        dist = diagram_wasserstein(d, d)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_bottleneck_identical_3d(self) -> None:
        from pynerve.torch import diagram_bottleneck

        d = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float64)
        dist = diagram_bottleneck(d, d)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-6)

    def test_wasserstein_large_number_of_points(self) -> None:
        from pynerve.torch import diagram_wasserstein

        births = torch.rand(100) * 10
        d1 = torch.stack([births, births + torch.rand(100) * 5 + 0.1], dim=1)
        d2 = torch.stack([births, births + torch.rand(100) * 5 + 0.1], dim=1)
        dist = diagram_wasserstein(d1, d2)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert torch.isfinite(val)

    def test_bottleneck_large_number_of_points(self) -> None:
        from pynerve.torch import diagram_bottleneck

        births = torch.rand(100) * 10
        d1 = torch.stack([births, births + torch.rand(100) * 5 + 0.1], dim=1)
        d2 = torch.stack([births, births + torch.rand(100) * 5 + 0.1], dim=1)
        dist = diagram_bottleneck(d1, d2)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert torch.isfinite(val)
