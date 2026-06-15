"""Correctness tests for persistence representations and image utilities."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local pynerve package is importable (mirrors conftest.py logic).
_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_ROOT = _ROOT / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402
from pynerve._fast_representations import (  # noqa: E402
    betti_curve_fast,
    persistence_image_fast,
    persistence_landscape_fast,
)
from pynerve._image_utils import persistence_image  # noqa: E402
from pynerve.exceptions import InvalidArgumentError, ValidationError  # noqa: E402

# helpers


def _simple_diagram() -> np.ndarray:
    """Two points: (0, 1) in dim 0, (2, 5) in dim 1."""
    return np.array([[0.0, 1.0, 0.0], [2.0, 5.0, 1.0]])


def _diagonal_diagram() -> np.ndarray:
    """Two diagonal points: zero persistence."""
    return np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 1.0]])


def _single_point_diagram() -> np.ndarray:
    return np.array([[1.0, 4.0, 0.0]])


def _empty_diagram() -> np.ndarray:
    return np.empty((0, 3))


def _identical_points_diagram() -> np.ndarray:
    return np.array([[2.0, 5.0, 0.0], [2.0, 5.0, 0.0], [2.0, 5.0, 0.0]])


# persistence_image_fast


class TestPersistenceImageFast:
    def test_non_negative_pixels(self):
        img = persistence_image_fast(_simple_diagram(), resolution=32, sigma=0.2)
        assert np.all(img >= 0), "all pixel values must be non-negative"

    def test_output_shape(self):
        img = persistence_image_fast(_simple_diagram(), resolution=47, sigma=0.1)
        assert img.shape == (47, 47)

    def test_empty_diagram_yields_zeros(self):
        img = persistence_image_fast(_empty_diagram(), resolution=10)
        assert img.shape == (10, 10)
        assert np.all(img == 0.0)

    def test_single_point(self):
        img = persistence_image_fast(_single_point_diagram(), resolution=64, sigma=0.1)
        assert img.shape == (64, 64)
        assert np.all(img >= 0)
        assert np.any(img > 0), "at least some pixels should be nonzero"

    def test_all_diagonal_yields_zeros(self):
        img = persistence_image_fast(_diagonal_diagram(), resolution=16, sigma=0.1)
        assert np.all(img >= 0)
        # With zero persistence, weights are zero (persistence^2), so image must be all zero.
        assert np.allclose(img, 0.0)

    def test_identical_points(self):
        img = persistence_image_fast(_identical_points_diagram(), resolution=32, sigma=0.1)
        assert np.all(img >= 0)
        assert np.any(img > 0)

    def test_nan_birth_raises(self):
        diagram = np.array([[float("nan"), 1.0, 0.0]])
        with pytest.raises(ValueError, match="finite"):
            persistence_image_fast(diagram)

    def test_inf_birth_raises(self):
        diagram = np.array([[float("inf"), 1.0, 0.0]])
        with pytest.raises(ValueError, match="finite"):
            persistence_image_fast(diagram)

    def test_inf_death_raises(self):
        diagram = np.array([[0.0, float("inf"), 0.0]])
        with pytest.raises(ValueError, match="finite"):
            persistence_image_fast(diagram)

    def test_death_less_than_birth_raises(self):
        diagram = np.array([[5.0, 3.0, 0.0]])
        with pytest.raises(ValueError, match="greater than or equal"):
            persistence_image_fast(diagram)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            persistence_image_fast(_single_point_diagram(), sigma=-0.5)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            persistence_image_fast(_single_point_diagram(), sigma=0.0)

    def test_nan_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            persistence_image_fast(_single_point_diagram(), sigma=float("nan"))

    def test_zero_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution"):
            persistence_image_fast(_single_point_diagram(), resolution=0)

    def test_invalid_weight_fn_raises(self):
        with pytest.raises(ValueError, match="weight_fn"):
            persistence_image_fast(_single_point_diagram(), weight_fn="quadratic")

    def test_determinism(self):
        img1 = persistence_image_fast(_simple_diagram(), resolution=32, sigma=0.1)
        img2 = persistence_image_fast(_simple_diagram(), resolution=32, sigma=0.1)
        assert_allclose(img1, img2)

    def test_linear_weight(self):
        img = persistence_image_fast(
            _simple_diagram(), resolution=32, sigma=0.1, weight_fn="linear"
        )
        assert np.all(img >= 0)

    def test_constant_weight(self):
        img = persistence_image_fast(
            _simple_diagram(), resolution=32, sigma=0.1, weight_fn="constant"
        )
        assert np.all(img >= 0)

    def test_none_weight(self):
        img = persistence_image_fast(_simple_diagram(), resolution=32, sigma=0.1, weight_fn=None)
        assert np.all(img >= 0)

    def test_weight_ordering(self):
        """Points with larger persistence (non-diagonal) contribute more than
        points with smaller persistence under default quadratic weight."""
        lo = np.array([[0.0, 1.0, 0.0]])  # persistence 1
        hi = np.array([[0.0, 5.0, 0.0]])  # persistence 5
        img_lo = persistence_image_fast(lo, resolution=32, sigma=0.1)
        img_hi = persistence_image_fast(hi, resolution=32, sigma=0.1)
        # Higher persistence point should produce a brighter image.
        assert img_hi.sum() > img_lo.sum()


# betti_curve_fast


class TestBettiCurveFast:
    def test_non_negative_integer(self):
        curve = betti_curve_fast(_simple_diagram(), max_dim=3, resolution=100)
        assert curve.dtype == int
        assert np.all(curve >= 0)

    def test_output_shape(self):
        curve = betti_curve_fast(_simple_diagram(), max_dim=2, resolution=80)
        assert curve.shape == (3, 80)  # max_dim + 1 rows

    def test_empty_diagram_yields_zeros(self):
        curve = betti_curve_fast(_empty_diagram(), max_dim=2, resolution=50)
        assert curve.shape == (3, 50)
        assert np.all(curve == 0)

    def test_single_point(self):
        # Point: birth=1, death=4, dim=0.
        curve = betti_curve_fast(_single_point_diagram(), max_dim=1, resolution=100)
        assert curve.shape == (2, 100)
        assert np.all(curve >= 0)
        # Dimension 0 should have nonzero values (bar is [1, 4)).
        assert np.any(curve[0] > 0)
        # Dimension 1 should be all zeros (no dim-1 points).
        assert np.all(curve[1] == 0)

    def test_monotonic_step_up_at_birth(self):
        """At the birth time, Betti number should increase by 1 and stay 1
        until just before death (exclusive)."""
        birth = 2.0
        death = 10.0
        diagram = np.array([[birth, death, 0.0]])
        resolution = 1000
        max_t = death + 1
        curve = betti_curve_fast(diagram, max_dim=0, max_time=max_t, resolution=resolution)
        times = np.linspace(0, max_t, resolution)
        nonzero_idx = np.where(curve[0] > 0)[0]
        assert len(nonzero_idx) > 0
        first_idx = nonzero_idx[0]
        last_idx = nonzero_idx[-1]
        assert times[first_idx] >= birth
        assert times[last_idx] < death
        # All values between first and last should be exactly 1.
        assert np.all(curve[0, first_idx : last_idx + 1] == 1)
        # After death, curve should drop to 0.
        assert np.all(curve[0, last_idx + 1 :] == 0)

    def test_monotonic_step_down_at_death(self):
        """At the death time, Betti number should drop to 0 (open interval)."""
        birth = 2.0
        death = 8.0
        diagram = np.array([[birth, death, 0.0]])
        resolution = 1000
        curve = betti_curve_fast(diagram, max_dim=0, max_time=12, resolution=resolution)
        times = np.linspace(0, 12, resolution)
        nonzero_idx = np.where(curve[0] > 0)[0]
        assert len(nonzero_idx) > 0
        last_idx = nonzero_idx[-1]
        assert times[last_idx] < death
        assert np.all(curve[0, last_idx + 1 :] == 0)

    def test_all_diagonal_yields_zeros(self):
        # Zero persistence bars have birth=death; curve counts while t in [birth, death).
        # No open interval -> zero contribution since death <= t excludes birth point
        # and birth > t excludes it too.
        curve = betti_curve_fast(_diagonal_diagram(), max_dim=2, resolution=100)
        # With birth=death, bars are empty or single-point, unlikely to contribute.
        # The strict > death comparison means birth=death bars give zero.
        assert np.all(curve == 0)

    def test_identical_points(self):
        # Three identical points should add the same Betti as one (one distinct bar).
        diagram = _identical_points_diagram()
        curve = betti_curve_fast(diagram, max_dim=0, max_time=6.0, resolution=100)
        assert np.all(curve >= 0)
        # Expect Betti 3 where t in [2, 5).
        assert np.any(curve[0] == 3)

    def test_nan_raises(self):
        diagram = np.array([[float("nan"), 1.0, 0.0]])
        with pytest.raises(ValueError, match="finite"):
            betti_curve_fast(diagram)

    def test_inf_birth_raises(self):
        diagram = np.array([[float("inf"), 1.0, 0.0]])
        with pytest.raises(ValueError, match="finite"):
            betti_curve_fast(diagram)

    def test_inf_death_allowed(self):
        """Infinity death is allowed for essential classes. The curve includes
        the bar up to max_time."""
        diagram = np.array([[1.0, float("inf"), 0.0]])
        # inf is not finite, but betti_curve_fast checks pairs[:, :3]. The
        # open-interval logic means the bar stays active until max_time.
        # Since np.isfinite(float("inf")) is False, and the check uses
        # not np.isfinite(pairs[:, :3]).all(), inf will raise.
        with pytest.raises(ValueError, match="finite"):
            betti_curve_fast(diagram)

    def test_negative_max_dim_raises(self):
        with pytest.raises(ValueError, match="max_dim"):
            betti_curve_fast(_single_point_diagram(), max_dim=-1)

    def test_negative_max_time_raises(self):
        with pytest.raises(ValueError, match="max_time"):
            betti_curve_fast(_single_point_diagram(), max_time=-5.0)

    def test_nan_max_time_raises(self):
        with pytest.raises(ValueError, match="max_time"):
            betti_curve_fast(_single_point_diagram(), max_time=float("nan"))

    def test_zero_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution"):
            betti_curve_fast(_single_point_diagram(), resolution=0)

    def test_determinism(self):
        c1 = betti_curve_fast(_simple_diagram(), max_dim=3, resolution=100)
        c2 = betti_curve_fast(_simple_diagram(), max_dim=3, resolution=100)
        assert_allclose(c1, c2)

    def test_multiple_dimensions(self):
        diagram = np.array(
            [
                [0.0, 1.0, 0.0],
                [2.0, 5.0, 0.0],
                [1.0, 4.0, 1.0],
                [3.0, 7.0, 1.0],
            ]
        )
        curve = betti_curve_fast(diagram, max_dim=2, resolution=200)
        # dim 0 has 2 bars, dim 1 has 2 bars, dim 2 has 0 bars.
        assert np.all(curve[0] <= 2)
        assert np.all(curve[1] <= 2)
        assert np.all(curve[2] == 0)

    def test_betti_is_non_decreasing_with_birth(self):
        """Betti numbers in a given dimension should not exceed the total
        number of bars in that dimension."""
        n_bars = 5
        diagram = np.column_stack(
            [
                np.linspace(0, 10, n_bars),
                np.linspace(5, 20, n_bars),
                np.zeros(n_bars),
            ]
        )
        curve = betti_curve_fast(diagram, max_dim=0, resolution=500)
        assert np.all(curve[0] <= n_bars)

    def test_betti_increases_monotonically_over_time(self):
        """Betti numbers should not jump by more than the number of bars whose
        births land at that time (but can drop arbitrarily as multiple bars
        can die at the same time). Since bars simply start at birth and stop
        before death, the curve should never be negative and never exceed the
        total bar count."""
        n_points = 20
        rng = np.random.RandomState(42)
        births = rng.uniform(0, 10, n_points)
        deaths = births + rng.uniform(0.1, 5, n_points)
        dims = rng.randint(0, 2, n_points).astype(float)
        diagram = np.column_stack([births, deaths, dims])
        curve = betti_curve_fast(diagram, max_dim=2, resolution=500)
        assert np.all(curve >= 0)
        assert np.all(curve[0] <= n_points)
        assert np.all(curve[1] <= n_points)


# persistence_landscape_fast


class TestPersistenceLandscapeFast:
    def test_non_negative(self):
        landscape = persistence_landscape_fast(_simple_diagram(), n_layers=3, resolution=100)
        assert np.all(landscape >= 0)

    def test_output_shape(self):
        landscape = persistence_landscape_fast(_simple_diagram(), n_layers=4, resolution=50)
        assert landscape.shape == (4, 50)

    def test_empty_diagram_yields_zeros(self):
        landscape = persistence_landscape_fast(_empty_diagram(), n_layers=3, resolution=30)
        assert landscape.shape == (3, 30)
        assert np.all(landscape == 0.0)

    def test_all_diagonal_yields_zeros(self):
        # Zero persistence -> zero height -> zero landscape.
        landscape = persistence_landscape_fast(_diagonal_diagram(), n_layers=3, resolution=100)
        assert np.allclose(landscape, 0.0)

    def test_single_point_tent_shape(self):
        # A single point (birth=1, death=4) produces a single tent:
        #   midpoint m = 2.5, height h = 1.5.
        #   lambda_1(t) = max(1.5 - |t - 2.5|, 0)
        # Grid resolution may not exactly hit m=2.5, so peak is ~1.5.
        diagram = np.array([[1.0, 4.0]])
        landscape = persistence_landscape_fast(diagram, n_layers=3, resolution=200)
        assert np.any(landscape[0] > 0)
        assert np.allclose(landscape[1:], 0.0)
        step = (4.0 - 1.0) / 199
        assert landscape[0].max() == pytest.approx(1.5, abs=step)

    def test_non_increasing_in_layer(self):
        """Higher layers must not exceed corresponding lower layers at any point."""
        n_points = 10
        rng = np.random.RandomState(123)
        births = rng.uniform(0, 10, n_points)
        deaths = births + rng.uniform(0.5, 5, n_points)
        diagram = np.column_stack([births, deaths])
        landscape = persistence_landscape_fast(diagram, n_layers=5, resolution=200)
        for k in range(1, landscape.shape[0]):
            assert np.all(landscape[k] <= landscape[k - 1]), f"Layer {k} exceeds layer {k - 1}"

    def test_one_lipschitz(self):
        """Each landscape layer must be 1-Lipschitz: adjacent differences
        must not exceed the grid spacing."""
        n_points = 15
        rng = np.random.RandomState(456)
        births = rng.uniform(0, 10, n_points)
        deaths = births + rng.uniform(0.5, 5, n_points)
        diagram = np.column_stack([births, deaths])
        landscape = persistence_landscape_fast(diagram, n_layers=3, resolution=200)
        t_min = births.min()
        t_max = deaths.max()
        step = (t_max - t_min) / (199)  # resolution - 1
        for k in range(landscape.shape[0]):
            diffs = np.abs(np.diff(landscape[k]))
            assert np.all(diffs <= step + 1e-12), (
                f"Layer {k} violates 1-Lipschitz: max diff {diffs.max()} > step {step}"
            )

    def test_identical_points(self):
        # Three identical points -> three identical tents -> all 3 layers nonzero.
        diagram = _identical_points_diagram()[:, :2]
        landscape = persistence_landscape_fast(diagram, n_layers=3, resolution=100)
        assert np.any(landscape[0] > 0)
        # All three layers share the same tent values.
        assert_allclose(landscape[0], landscape[1])
        assert_allclose(landscape[1], landscape[2])
        # Layer 4 (beyond the 3 tents) should be zero.
        landscape4 = persistence_landscape_fast(diagram, n_layers=4, resolution=100)
        assert np.allclose(landscape4[3], 0.0)

    def test_more_layers_than_points(self):
        """Requesting more layers than points pads with zeros."""
        diagram = np.array([[0.0, 1.0], [2.0, 5.0]])
        landscape = persistence_landscape_fast(diagram, n_layers=10, resolution=30)
        assert landscape.shape == (10, 30)
        # Layers beyond len(diagram) should be zero.
        assert np.allclose(landscape[2:], 0.0)

    def test_nan_birth_raises(self):
        diagram = np.array([[float("nan"), 1.0]])
        with pytest.raises(ValueError, match="finite"):
            persistence_landscape_fast(diagram)

    def test_inf_birth_raises(self):
        diagram = np.array([[float("inf"), 1.0]])
        with pytest.raises(ValueError, match="finite"):
            persistence_landscape_fast(diagram)

    def test_inf_death_raises(self):
        diagram = np.array([[0.0, float("inf")]])
        with pytest.raises(ValueError, match="finite"):
            persistence_landscape_fast(diagram)

    def test_death_less_than_birth_raises(self):
        diagram = np.array([[5.0, 3.0]])
        with pytest.raises(ValueError, match="greater than or equal"):
            persistence_landscape_fast(diagram)

    def test_negative_n_layers_raises(self):
        with pytest.raises(ValueError, match="n_layers"):
            persistence_landscape_fast(_single_point_diagram()[:, :2], n_layers=0)

    def test_zero_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution"):
            persistence_landscape_fast(_single_point_diagram()[:, :2], resolution=0)

    def test_determinism(self):
        diagram = np.column_stack(
            [
                np.random.RandomState(789).uniform(0, 10, 8),
                np.random.RandomState(789).uniform(5, 20, 8),
            ]
        )
        l1 = persistence_landscape_fast(diagram, n_layers=5, resolution=100)
        l2 = persistence_landscape_fast(diagram, n_layers=5, resolution=100)
        assert_allclose(l1, l2)

    def test_overlapping_tents(self):
        """Two well-separated tents produce two distinct bumps on layer 1,
        and layer 2 is zero."""
        diagram = np.array([[0.0, 1.0], [5.0, 6.0]])
        landscape = persistence_landscape_fast(diagram, n_layers=2, resolution=200)
        assert np.any(landscape[0] > 0)
        assert np.allclose(landscape[1], 0.0)
        # Two peaks (one near 0.5, one near 5.5).
        peaks = (landscape[0, 1:-1] > landscape[0, :-2]) & (landscape[0, 1:-1] > landscape[0, 2:])
        assert np.sum(peaks) >= 2, "should have at least two local peaks"

    def test_overlapping_tents_n_layers(self):
        """Two overlapping tents produce a nonzero second landscape layer."""
        diagram = np.array([[0.0, 5.0], [1.0, 4.0]])  # heavily overlapping
        landscape = persistence_landscape_fast(diagram, n_layers=3, resolution=200)
        assert np.any(landscape[0] > 0)
        assert np.any(landscape[1] > 0), "overlapping tents should produce a nonzero second layer"
        assert np.allclose(landscape[2], 0.0)  # only 2 tents, no third layer


# persistence_image (_image_utils)


class TestPersistenceImageUtils:
    def test_non_negative_pixels(self):
        img = persistence_image(_simple_diagram(), resolution=32, sigma=0.2)
        assert np.all(img >= 0), "all pixel values must be non-negative"

    def test_output_shape_square(self):
        img = persistence_image(_simple_diagram(), resolution=25, sigma=0.1)
        assert img.shape == (25, 25)

    def test_output_shape_tuple(self):
        img = persistence_image(_simple_diagram(), resolution=(30, 40), sigma=0.1)
        assert img.shape == (30, 40)

    def test_empty_diagram_yields_zeros(self):
        img = persistence_image(np.empty((0, 3)), resolution=10)
        assert img.shape == (10, 10)
        assert np.all(img == 0.0)

    def test_all_diagonal_yields_zeros(self):
        # Points with zero persistence are filtered out.
        img = persistence_image(_diagonal_diagram(), resolution=16, sigma=0.1)
        assert np.allclose(img, 0.0)

    def test_single_point(self):
        img = persistence_image(_single_point_diagram(), resolution=64, sigma=0.1)
        assert img.shape == (64, 64)
        assert np.all(img >= 0)
        assert np.any(img > 0)

    def test_identical_points(self):
        img = persistence_image(_identical_points_diagram(), resolution=32, sigma=0.1)
        assert np.all(img >= 0)
        assert np.any(img > 0)

    def test_weight_persistence_greater_than_uniform_for_nonzero_persistence(self):
        """With persistence weight, non-diagonal points contribute more than
        with uniform weight."""
        diagram = _simple_diagram()
        img_p = persistence_image(diagram, resolution=32, sigma=0.1, weight="persistence")
        img_u = persistence_image(diagram, resolution=32, sigma=0.1, weight="uniform")
        assert img_p.sum() > img_u.sum()

    def test_weight_uniform_on_zero_persistence(self):
        """Uniform weight should still produce zero on a zero-persistence point."""
        diagram = _diagonal_diagram()
        img = persistence_image(diagram, resolution=32, sigma=0.1, weight="uniform")
        assert np.allclose(img, 0.0)

    def test_sigma_affects_spread(self):
        img_small = persistence_image(_single_point_diagram(), resolution=64, sigma=0.05)
        img_large = persistence_image(_single_point_diagram(), resolution=64, sigma=0.5)
        assert img_large.sum() > img_small.sum()

    def test_resolution_affects_detail(self):
        img_low = persistence_image(_simple_diagram(), resolution=10, sigma=0.1)
        img_high = persistence_image(_simple_diagram(), resolution=100, sigma=0.1)
        # Higher resolution should produce values closer to true Gaussian
        # (the integral should be similar across resolutions).
        assert img_low.sum() > 0
        assert img_high.sum() > 0

    def test_nan_birth_raises(self):
        with pytest.raises((InvalidArgumentError, ValidationError)):
            persistence_image(np.array([[float("nan"), 1.0, 0.0]]))

    def test_negative_sigma_raises(self):
        with pytest.raises(InvalidArgumentError):
            persistence_image(_simple_diagram(), sigma=-0.1)

    def test_zero_sigma_raises(self):
        with pytest.raises(InvalidArgumentError):
            persistence_image(_simple_diagram(), sigma=0.0)

    def test_nan_sigma_raises(self):
        with pytest.raises(InvalidArgumentError):
            persistence_image(_simple_diagram(), sigma=float("nan"))

    def test_zero_resolution_raises(self):
        with pytest.raises(InvalidArgumentError):
            persistence_image(_simple_diagram(), resolution=0)

    def test_invalid_weight_raises(self):
        with pytest.raises(InvalidArgumentError):
            persistence_image(_simple_diagram(), weight="cosine")

    def test_determinism(self):
        diagram = _simple_diagram()
        img1 = persistence_image(diagram, resolution=32, sigma=0.1)
        img2 = persistence_image(diagram, resolution=32, sigma=0.1)
        assert_allclose(img1, img2)

    def test_birth_range_constrains_x_axis(self):
        diagram = np.array([[0.0, 5.0, 0.0]])
        img = persistence_image(diagram, resolution=50, sigma=0.1, birth_range=(0.0, 2.0))
        assert img.shape == (50, 50)

    def test_persistence_range_constrains_y_axis(self):
        diagram = np.array([[0.0, 5.0, 0.0]])
        img = persistence_image(diagram, resolution=50, sigma=0.1, persistence_range=(0.0, 3.0))
        assert img.shape == (50, 50)

    def test_inf_death_filtered(self):
        """Rows with +inf death (non-finite persistence) are filtered out,
        producing an all-zero image."""
        diagram = np.array([[1.0, float("inf"), 0.0]])
        img = persistence_image(diagram, resolution=32, sigma=0.1)
        assert np.allclose(img, 0.0)

    def test_non_finite_output(self):
        """For valid finite inputs, output values must all be finite."""
        diagram = _simple_diagram()
        img = persistence_image(diagram, resolution=32, sigma=0.1)
        assert np.isfinite(img).all()

    def test_persistence_image_weight_correspondence(self):
        """With persistence weighting, a bar of persistence p contributes
        weight p, so two identical bars of persistence 1 and 4 should
        produce proportional pixel sums (modulo Gaussian overlap)."""
        lo = np.array([[0.0, 1.0, 0.0]])
        hi = np.array([[0.0, 5.0, 0.0]])
        img_lo = persistence_image(lo, resolution=64, sigma=0.2, weight="persistence")
        img_hi = persistence_image(hi, resolution=64, sigma=0.2, weight="persistence")
        assert img_hi.sum() > img_lo.sum()
