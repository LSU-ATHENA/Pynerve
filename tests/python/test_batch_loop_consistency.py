"""Batch consistency: 2D (single) == 3D (batched) produce identical values.

Now that operations natively support 3D batched input, verify that
the batched result matches per-element application.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

_D2 = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
_B2 = torch.stack([_D2, _D2], dim=0)

_D2_inf = torch.tensor([[0.0, 1.0], [0.0, float("inf")]], dtype=torch.float64)
_B2_inf = torch.stack([_D2_inf, _D2_inf], dim=0)


def _match(single, batched):
    """Assert batched == torch.stack([single, single]) within tolerance."""
    stacked = torch.stack([single, single])
    assert batched.shape == stacked.shape
    assert torch.allclose(batched, stacked, atol=1e-10)


class TestStatisticsBatch2Dvs3D:
    """Statistics: 3D input returns per-batch results matching 2D."""

    @pytest.mark.parametrize(
        "fn_name",
        [
            "total_persistence",
            "mean_persistence",
            "max_persistence",
            "persistence_variance",
            "number_of_features",
            "persistence_entropy",
        ],
    )
    def test_batch_matches_single(self, fn_name):
        import pynerve.torch.statistics as st

        fn = getattr(st, fn_name)
        single = fn(_D2)
        batched = fn(_B2)
        _match(single, batched)

    def test_amplitude(self):
        from pynerve.torch.statistics import amplitude

        single = amplitude(_D2)
        batched = amplitude(_B2)
        _match(single, batched)

    def test_betti_curve(self):
        from pynerve.torch.statistics import betti_curve

        single = betti_curve(_D2, num_samples=10)
        batched = betti_curve(_B2, num_samples=10)
        _match(single, batched)


class TestVectorizationBatch2Dvs3D:
    """Vectorization: 3D input returns batched result matching 2D."""

    def test_persistence_image(self):
        from pynerve.torch.vectorization import persistence_image as fn

        single = fn(_D2, resolution=(4, 4), sigma=0.5)
        batched = fn(_B2, resolution=(4, 4), sigma=0.5)
        _match(single, batched)

    def test_persistence_landscape(self):
        from pynerve.torch.vectorization import persistence_landscape as fn

        xr: tuple[float, float] = (0.0, 4.0)
        single = fn(_D2, k=2, num_samples=5, x_range=xr)
        batched = fn(_B2, k=2, num_samples=5, x_range=xr)
        _match(single, batched)

    def test_persistence_silhouette(self):
        from pynerve.torch.vectorization import persistence_silhouette as fn

        single = fn(_D2, num_samples=5)
        batched = fn(_B2, num_samples=5)
        _match(single, batched)

    def test_birth_death_curve(self):
        from pynerve.torch.vectorization import birth_death_curve as fn

        single = fn(_D2, num_bins=4)
        batched = fn(_B2, num_bins=4)
        _match(single, batched)

    def test_adaptive_persistence_image(self):
        from pynerve.torch.vectorization import adaptive_persistence_image as fn

        single = fn(_D2, target_resolution=4)
        batched = fn(_B2, target_resolution=4)
        assert batched.shape[0] == 2
        assert torch.allclose(batched[0], single)


class TestDistanceBatch2Dvs3D:
    """Distance functions: batched 3D matches per-element 2D."""

    def test_wasserstein_batch(self):
        from pynerve.torch import diagram_wasserstein as fn

        single = fn(_D2, _D2)
        batched = fn(_B2, _B2)
        assert batched.shape == (2,)
        assert batched[0].item() == pytest.approx(single.item(), abs=1e-10)

    def test_bottleneck_batch(self):
        from pynerve.torch import diagram_bottleneck as fn

        single = fn(_D2, _D2)
        batched = fn(_B2, _B2)
        assert batched.shape == (2,)
        assert batched[0].item() == pytest.approx(single.item(), abs=1e-6)

    def test_wasserstein_batch_mismatch_raises(self):
        from pynerve.torch import diagram_wasserstein as fn

        d3 = torch.stack([_D2] * 3, dim=0)
        with pytest.raises((ValueError, Exception)):
            fn(_B2, d3)


class TestPreprocessingBatch2Dvs3D:
    """Preprocessing: 3D input matches per-element 2D."""

    def test_handle_infinite_deaths(self):
        from pynerve.torch.preprocessing import handle_infinite_deaths as fn

        single = fn(_D2_inf, strategy="max")
        batched = fn(_B2_inf, strategy="max")
        assert batched.shape[0] == 2
        assert torch.allclose(batched[0], single)

    def test_normalize_diagram(self):
        from pynerve.torch.preprocessing import normalize_diagram as fn

        single = fn(_D2, method="minmax")
        batched = fn(_B2, method="minmax")
        assert batched.shape[0] == 2
        assert torch.allclose(batched[0], single)

    def test_threshold_diagram(self):
        from pynerve.torch.preprocessing import threshold_diagram as fn

        single = fn(_D2, min_persistence=0.5)
        batched = fn(_B2, min_persistence=0.5)
        assert batched.shape[0] == 2
        assert torch.allclose(batched[0], single)

    def test_subsample_diagram(self):
        from pynerve.torch.preprocessing import subsample_diagram as fn

        single = fn(_D2, max_features=1, strategy="most_persistent")
        batched = fn(_B2, max_features=1, strategy="most_persistent")
        assert batched.shape[0] == 2
        assert torch.allclose(batched[0], single)


class TestBatchOutputShapes:
    """3D batch input produces correctly-shaped batched output."""

    def test_statistics_output_shape(self):
        import pynerve.torch.statistics as st

        d3 = torch.stack([_D2, _D2 * 2, _D2 * 3], dim=0)
        for name in [
            "total_persistence",
            "mean_persistence",
            "max_persistence",
            "persistence_variance",
        ]:
            result = getattr(st, name)(d3)
            assert result.shape == (3,), f"{name}: expected (3,) got {result.shape}"

    def test_vectorization_output_shape(self):
        from pynerve.torch.vectorization import (
            persistence_image,
            persistence_landscape,
            persistence_silhouette,
        )

        d3 = torch.stack([_D2, _D2 * 2, _D2 * 3], dim=0)
        img = persistence_image(d3, resolution=(4, 4), sigma=0.5)
        assert img.shape == (3, 4, 4)
        land = persistence_landscape(d3, k=2, num_samples=5)
        assert land.shape == (3, 2, 5)
        sil = persistence_silhouette(d3, num_samples=5)
        assert sil.shape == (3, 5)

    def test_preprocessing_output_shape(self):
        from pynerve.torch.preprocessing import handle_infinite_deaths, normalize_diagram

        d3 = torch.stack([_D2, _D2 * 2, _D2 * 3], dim=0)
        normed = normalize_diagram(d3, method="minmax")
        assert normed.shape[0] == 3
        d3_inf = torch.stack([_D2_inf, _D2_inf, _D2_inf], dim=0)
        handled = handle_infinite_deaths(d3_inf, strategy="max")
        assert handled.shape[0] == 3
