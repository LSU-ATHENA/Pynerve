"""Error message / API misuse tests for all major operations."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

_D = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)


class TestStatisticsErrors:
    """Statistics operations give clear errors on invalid input."""

    def test_total_persistence_invalid_p(self):
        from pynerve.torch.statistics import total_persistence

        with pytest.raises((ValueError, Exception), match="positive|finite"):
            total_persistence(_D, p=float("nan"))

    def test_mean_persistence_nan_diagram(self):
        from pynerve.torch.statistics import mean_persistence

        d = torch.tensor([[0.0, float("nan")]], dtype=torch.float64)
        with pytest.raises((ValueError, Exception), match="finite|NaN"):
            mean_persistence(d)

    def test_inf_diagram_raises(self):
        from pynerve.torch.statistics import total_persistence

        d = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        with pytest.raises((ValueError, Exception), match="finite"):
            total_persistence(d)

    def test_negative_threshold_raises(self):
        from pynerve.torch.statistics import number_of_features

        with pytest.raises((ValueError, Exception)):
            number_of_features(_D, min_persistence=-1.0)

    def test_bad_dim_returns_full_total(self):
        from pynerve.torch.statistics import total_persistence

        r = total_persistence(_D, dim=999)
        assert r.item() == pytest.approx(3.0, abs=1e-10)

    def test_amplitude_unknown_metric(self):
        from pynerve.torch.statistics import amplitude

        with pytest.raises((ValueError, Exception)):
            amplitude(_D, metric="unknown")


class TestVectorizationErrors:
    """Vectorization operations give clear errors on invalid input."""

    def test_image_bad_resolution(self):
        from pynerve.torch.vectorization import persistence_image

        with pytest.raises((ValueError, Exception)):
            persistence_image(_D, resolution=(0, 4))

    def test_image_nan_sigma(self):
        from pynerve.torch.vectorization import persistence_image

        with pytest.raises((ValueError, Exception)):
            persistence_image(_D, resolution=(4, 4), sigma=float("nan"))

    def test_landscape_bad_k(self):
        from pynerve.torch.vectorization import persistence_landscape

        with pytest.raises((ValueError, Exception)):
            persistence_landscape(_D, k=0, num_samples=5)

    def test_curve_bad_statistic(self):
        from pynerve.torch.vectorization import birth_death_curve

        with pytest.raises((ValueError, Exception)):
            birth_death_curve(_D, statistic="invalid")

    def test_silhouette_bad_weight_fn(self):
        from pynerve.torch.vectorization import persistence_silhouette

        with pytest.raises((ValueError, Exception)):
            persistence_silhouette(_D, weight_fn="invalid")

    def test_heat_bad_num_samples(self):
        from pynerve.torch.vectorization import heat_kernel_signature

        with pytest.raises((ValueError, Exception)):
            heat_kernel_signature(_D, num_samples=-1)

    def test_betti_curve_bad_num_samples(self):
        from pynerve.torch.statistics import betti_curve

        with pytest.raises((ValueError, Exception)):
            betti_curve(_D, num_samples=0)


class TestPreprocessingErrors:
    """Preprocessing operations give clear errors on invalid input."""

    def test_handle_inf_bad_strategy(self):
        from pynerve.torch.preprocessing import handle_infinite_deaths

        with pytest.raises((ValueError, Exception)):
            handle_infinite_deaths(_D, strategy="invalid")

    def test_normalize_bad_method(self):
        from pynerve.torch.preprocessing import normalize_diagram

        with pytest.raises((ValueError, Exception)):
            normalize_diagram(_D, method="invalid")

    def test_threshold_bad_min(self):
        from pynerve.torch.preprocessing import threshold_diagram

        with pytest.raises((ValueError, Exception)):
            threshold_diagram(_D, min_persistence=-1.0)

    def test_subsample_bad_strategy(self):
        from pynerve.torch.preprocessing import subsample_diagram

        with pytest.raises((ValueError, Exception)):
            subsample_diagram(_D, max_features=1, strategy="invalid")

    def test_subsample_bad_max_features(self):
        from pynerve.torch.preprocessing import subsample_diagram

        with pytest.raises((ValueError, Exception)):
            subsample_diagram(_D, max_features=-1, strategy="most_persistent")


class TestDistanceErrors:
    """Distance operations give clear errors on invalid input."""

    def test_wasserstein_bad_p(self):
        from pynerve.torch import diagram_wasserstein

        with pytest.raises((ValueError, Exception)):
            diagram_wasserstein(_D, _D, p=float("nan"))

    def test_wasserstein_batch_mismatch(self):
        from pynerve.torch import diagram_wasserstein

        d1 = torch.stack([_D, _D], dim=0)
        d2 = torch.stack([_D, _D, _D], dim=0)
        with pytest.raises((ValueError, Exception)):
            diagram_wasserstein(d1, d2)

    def test_bottleneck_batch_mismatch(self):
        from pynerve.torch import diagram_bottleneck

        d1 = torch.stack([_D, _D], dim=0)
        d2 = torch.stack([_D, _D, _D], dim=0)
        with pytest.raises((ValueError, Exception)):
            diagram_bottleneck(d1, d2)

    def test_distance_wrong_dim_raises(self):
        from pynerve.torch import diagram_wasserstein

        with pytest.raises((ValueError, Exception)):
            diagram_wasserstein(torch.tensor(1.0), torch.tensor(1.0))
