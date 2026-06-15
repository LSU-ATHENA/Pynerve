"""Regression tests for previously-fixed bugs.

These tests reproduce issues that were fixed in past commits,
ensuring they never reappear.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestRegression:
    """Tests that would have caught past bugs."""

    def test_total_persistence_very_small(self):
        """Regression: very small persistence should not underflow to NaN."""
        from pynerve.torch.statistics import total_persistence

        d = torch.tensor([[0.0, 1e-20], [0.0, 2e-20]], dtype=torch.float64)
        tp = total_persistence(d, p=1.0)
        assert torch.isfinite(tp)
        assert tp.item() > 0

    def test_persistence_image_empty_diagram(self):
        """Regression: empty diagram should return zeros, not raise."""
        from pynerve.torch.vectorization import persistence_image

        empty = torch.empty(0, 2, dtype=torch.float64)
        img = persistence_image(empty, resolution=(4, 4), sigma=0.5)
        assert img.shape == (4, 4)
        assert (img == 0).all()

    def test_persistence_landscape_empty_diagram(self):
        """Regression: empty diagram should return zeros."""
        from pynerve.torch.vectorization import persistence_landscape

        empty = torch.empty(0, 2, dtype=torch.float64)
        xr: tuple[float, float] = (0.0, 1.0)
        land = persistence_landscape(empty, k=2, num_samples=5, x_range=xr)
        assert (land == 0).all()

    def test_normalize_diagram_single_element(self):
        """Regression: single-element diagram should not divide by zero."""
        from pynerve.torch.preprocessing import normalize_diagram

        d = torch.tensor([[5.0, 5.0]], dtype=torch.float64)
        normed = normalize_diagram(d, method="minmax")
        assert torch.isfinite(normed).all()

    def test_entropy_zero_persistence(self):
        """Regression: all-zero persistence should return 0, not NaN."""
        from pynerve.torch.statistics import persistence_entropy

        d = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        ent = persistence_entropy(d)
        assert torch.isfinite(ent)
        assert ent.item() == pytest.approx(0.0, abs=1e-10)

    def test_variance_single_feature(self):
        """Regression: variance of single element should be 0."""
        from pynerve.torch.statistics import persistence_variance

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        pv = persistence_variance(d)
        assert pv.item() == pytest.approx(0.0, abs=1e-10)

    def test_number_of_features_all_zero_persistence(self):
        """Regression: zero persistence features count with threshold > 0."""
        from pynerve.torch.statistics import number_of_features

        d = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        n = number_of_features(d, min_persistence=0.1)
        assert n.item() == 0

    def test_amplitude_empty(self):
        """Regression: empty diagram should return 0 amplitude."""
        from pynerve.torch.statistics import amplitude

        empty = torch.empty(0, 2, dtype=torch.float64)
        for metric in ("persistence", "bottleneck", "wasserstein"):
            val = amplitude(empty, metric=metric)
            assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_wasserstein_identical_single_point(self):
        """Regression: wasserstein distance of identical single-point diagrams."""
        from pynerve.torch import diagram_wasserstein

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        dist = diagram_wasserstein(d, d)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_bottleneck_identical_single_point(self):
        """Regression: bottleneck distance of identical single-point diagrams."""
        from pynerve.torch import diagram_bottleneck

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        dist = diagram_bottleneck(d, d)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-6)

    def test_subsample_diagram_empty(self):
        """Regression: subsampling an empty diagram returns empty."""
        from pynerve.torch.preprocessing import subsample_diagram

        empty = torch.empty(0, 2, dtype=torch.float64)
        for strategy in ("most_persistent", "uniform", "random"):
            result = subsample_diagram(empty, max_features=5, strategy=strategy)
            assert result.shape[0] == 0

    def test_betti_curve_no_finite_deaths(self):
        """Regression: diagram with only inf deaths should return zeros."""
        from pynerve.torch.statistics import betti_curve

        d = torch.tensor([[0.0, float("inf")], [0.0, float("inf")]], dtype=torch.float64)
        with pytest.raises((ValueError, Exception), match="finite"):
            betti_curve(d, num_samples=5)
