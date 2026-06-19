from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_ROOT = _ROOT / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pynerve.algorithms import (
    gaussian_kernel_matrix,
    knn,
    pairwise_distances,
    persistence_heat_vector,
    persistence_image,
    persistence_landscape,
    persistence_silhouette,
)
from pynerve.exceptions import ValidationError
from pynerve.exceptions import InvalidArgumentError, ValidationError


def _simple_diagram() -> np.ndarray:
    return np.array([[0.0, 1.0, 0.0], [2.0, 5.0, 1.0]])


def _diagonal_diagram() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 1.0]])


def _single_diagram() -> np.ndarray:
    return np.array([[1.0, 4.0, 0.0]])


def _empty_diagram() -> np.ndarray:
    return np.empty((0, 3))


def _identical_diagram() -> np.ndarray:
    return np.array([[2.0, 5.0, 0.0], [2.0, 5.0, 0.0], [2.0, 5.0, 0.0]])


def _large_diagram() -> np.ndarray:
    rng = np.random.RandomState(42)
    births = rng.uniform(0, 10, 20)
    deaths = births + rng.uniform(0.5, 5, 20)
    dims = rng.randint(0, 3, 20).astype(float)
    return np.column_stack([births, deaths, dims])


def _infinite_death_diagram() -> np.ndarray:
    return np.array([[1.0, float("inf"), 0.0], [0.0, 2.0, 1.0]])


class TestPairwiseDistances:
    def test_output_shape_and_dtype(self):
        pts = np.random.RandomState(42).uniform(0, 5, (10, 3))
        d = pairwise_distances(pts)
        assert d.shape == (10, 10)
        assert d.dtype == np.float64

    def test_symmetric(self):
        pts = np.random.RandomState(7).normal(size=(8, 4))
        d = pairwise_distances(pts)
        assert_allclose(d, d.T)

    def test_diagonal_zero(self):
        pts = np.random.RandomState(11).uniform(-1, 1, (6, 5))
        d = pairwise_distances(pts)
        assert_allclose(np.diag(d), 0.0, atol=1e-12)

    def test_self_distance_zero(self):
        pts = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        d = pairwise_distances(pts)
        assert_allclose(np.diag(d), 0.0, atol=1e-12)

    def test_euclidean_known(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        d = pairwise_distances(pts, metric="euclidean")
        assert_allclose(d[0, 1], 5.0)
        assert_allclose(d[1, 0], 5.0)

    def test_cityblock_known(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        d = pairwise_distances(pts, metric="cityblock")
        assert_allclose(d[0, 1], 7.0)

    def test_cosine(self):
        pts = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        d = pairwise_distances(pts, metric="cosine")
        assert d.shape == (3, 3)
        assert_allclose(np.diag(d), 0.0, atol=1e-12)
        assert np.all(d >= 0)

    def test_invalid_metric_raises(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises((ValueError, ValidationError)):
            pairwise_distances(pts, metric="invalid_metric")

    def test_1d_array_raises(self):
        with pytest.raises(ValidationError, match="2-D"):
            pairwise_distances(np.array([0.0, 1.0, 2.0]))

    def test_3d_array_raises(self):
        with pytest.raises(ValidationError, match="2-D"):
            pairwise_distances(np.zeros((2, 3, 4)))

    def test_not_ndarray_raises(self):
        with pytest.raises(ValidationError, match="2-D"):
            pairwise_distances([1.0, 2.0, 3.0])

    def test_single_point(self):
        pts = np.array([[1.0, 2.0]])
        d = pairwise_distances(pts)
        assert d.shape == (1, 1)
        assert_allclose(d[0, 0], 0.0)

    def test_determinism(self):
        pts = np.random.RandomState(99).uniform(0, 10, (5, 3))
        d1 = pairwise_distances(pts)
        d2 = pairwise_distances(pts)
        assert_allclose(d1, d2)

    def test_use_simd_ignored(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        d1 = pairwise_distances(pts, _use_simd=True)
        d2 = pairwise_distances(pts, _use_simd=False)
        assert_allclose(d1, d2)


class TestKNN:
    def test_output_shapes(self):
        pts = np.random.RandomState(5).uniform(0, 5, (15, 3))
        dist, idx = knn(pts, k=5)
        assert dist.shape == (15, 5)
        assert idx.shape == (15, 5)

    def test_distances_non_negative(self):
        pts = np.random.RandomState(13).uniform(0, 10, (10, 2))
        dist, _ = knn(pts, k=3)
        assert np.all(dist >= 0)

    def test_does_not_return_self(self):
        pts = np.random.RandomState(17).normal(size=(10, 3))
        _, idx = knn(pts, k=3)
        for i in range(10):
            assert i not in idx[i]

    def test_sorted_distances(self):
        pts = np.random.RandomState(19).uniform(0, 10, (8, 4))
        dist, _ = knn(pts, k=4)
        for i in range(8):
            assert np.all(np.diff(dist[i]) >= 0)

    def test_brute_force_vs_kd_tree(self):
        pts = np.random.RandomState(23).uniform(0, 5, (20, 2))
        d_bf, i_bf = knn(pts, k=3, algorithm="brute_force")
        d_kd, i_kd = knn(pts, k=3, algorithm="kd_tree")
        assert_allclose(d_bf, d_kd)
        assert_allclose(i_bf, i_kd)

    def test_kd_tree_single_point(self):
        pts = np.array([[0.0, 0.0]])
        dist, idx = knn(pts, k=1, algorithm="kd_tree")
        assert dist.shape == (1, 1)
        assert idx.shape == (1, 1)

    def test_k_equals_zero_raises(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(InvalidArgumentError, match="k must be"):
            knn(pts, k=0)

    def test_k_negative_raises(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(InvalidArgumentError, match="k must be"):
            knn(pts, k=-1)

    def test_1d_array_raises(self):
        with pytest.raises(ValidationError, match="2-D"):
            knn(np.array([0.0, 1.0, 2.0]), k=2)

    def test_invalid_algorithm_raises(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(InvalidArgumentError, match="algorithm must be"):
            knn(pts, k=1, algorithm="flann")

    def test_small_k_equals_one(self):
        pts = np.random.RandomState(29).uniform(0, 5, (12, 3))
        dist, idx = knn(pts, k=1)
        assert dist.shape == (12, 1)
        for i in range(len(pts)):
            assert i not in idx[i]

    def test_large_k_equals_n_minus_one(self):
        n = 5
        pts = np.random.RandomState(31).uniform(0, 5, (n, 2))
        dist, idx = knn(pts, k=n - 1)
        assert dist.shape == (n, n - 1)
        for i in range(n):
            assert i not in idx[i]


class TestPersistenceLandscape:
    def test_output_shape(self):
        ls = persistence_landscape(_simple_diagram(), num_levels=5, resolution=100)
        assert ls.shape == (5, 100)

    def test_non_negative(self):
        ls = persistence_landscape(_large_diagram(), num_levels=3, resolution=80)
        assert np.all(ls >= 0)

    def test_empty_diagram_yields_zeros(self):
        ls = persistence_landscape(_empty_diagram(), num_levels=3, resolution=30)
        assert ls.shape == (3, 30)
        assert np.all(ls == 0.0)

    def test_diagonal_diagram_yields_zeros(self):
        ls = persistence_landscape(_diagonal_diagram(), num_levels=3, resolution=100)
        assert_allclose(ls, 0.0)

    def test_single_point_landscape(self):
        ls = persistence_landscape(_single_diagram(), num_levels=3, resolution=200)
        assert ls[0].max() > 0
        assert_allclose(ls[1], 0.0)
        assert_allclose(ls[2], 0.0)

    def test_num_levels_output_shape(self):
        ls = persistence_landscape(_large_diagram(), num_levels=5, resolution=100)
        assert ls.shape == (5, 100)
        assert ls[0].max() >= 0

    def test_identical_points(self):
        ls = persistence_landscape(_identical_diagram(), num_levels=3, resolution=100)
        assert_allclose(ls[0], ls[1])
        assert_allclose(ls[1], ls[2])
        assert ls[0].max() > 0

    def test_more_levels_than_points(self):
        ls = persistence_landscape(np.array([[0.0, 1.0]]), num_levels=5, resolution=30)
        assert ls.shape == (5, 30)
        assert_allclose(ls[1:], 0.0)

    def test_overlapping_tents_first_layer(self):
        diagram = np.array([[0.0, 5.0], [1.0, 4.0]])
        ls = persistence_landscape(diagram, num_levels=3, resolution=200)
        assert ls[0].max() > 0
        assert ls[1].max() > 0
        assert_allclose(ls[2], 0.0)

    def test_separated_tents(self):
        diagram = np.array([[0.0, 1.0], [5.0, 6.0]])
        ls = persistence_landscape(diagram, num_levels=2, resolution=200)
        assert ls[0].max() > 0
        assert_allclose(ls[1], 0.0)

    def test_determinism(self):
        d = _large_diagram()
        l1 = persistence_landscape(d, num_levels=3, resolution=50)
        l2 = persistence_landscape(d, num_levels=3, resolution=50)
        assert_allclose(l1, l2)

    def test_nan_birth_raises(self):
        d = np.array([[float("nan"), 1.0, 0.0]])
        with pytest.raises((ValidationError, ValueError)):
            persistence_landscape(d)

    def test_inf_birth_raises(self):
        d = np.array([[float("inf"), 1.0, 0.0]])
        with pytest.raises((ValidationError, ValueError)):
            persistence_landscape(d)

    def test_death_less_than_birth_raises(self):
        d = np.array([[5.0, 3.0, 0.0]])
        with pytest.raises((ValidationError, ValueError)):
            persistence_landscape(d)

    def test_infinite_death_filtered(self):
        d = np.array([[1.0, float("inf"), 0.0], [0.0, 2.0, 1.0]])
        ls = persistence_landscape(d, num_levels=3, resolution=100)
        assert ls.shape == (3, 100)
        assert np.all(ls >= 0)

    def test_t_max_edge_case(self):
        d = np.array([[0.0, 0.1]])
        ls = persistence_landscape(d, num_levels=2, resolution=10)
        assert ls.shape == (2, 10)
        assert ls[0].max() > 0

    def test_default_parameters(self):
        d = _simple_diagram()
        ls = persistence_landscape(d)
        assert ls.shape == (5, 100)

    def test_large_resolution(self):
        d = _single_diagram()
        ls = persistence_landscape(d, num_levels=2, resolution=500)
        assert ls.shape == (2, 500)
        assert np.all(ls >= 0)

    def test_single_level(self):
        d = _simple_diagram()
        ls = persistence_landscape(d, num_levels=1, resolution=50)
        assert ls.shape == (1, 50)
        assert np.all(ls >= 0)

    def test_two_column_diagram(self):
        d = np.array([[1.0, 3.0]])
        ls = persistence_landscape(d, num_levels=2, resolution=50)
        assert ls.shape == (2, 50)
        assert ls[0].max() > 0

    def test_many_points_landscape(self):
        rng = np.random.RandomState(123)
        n = 50
        births = rng.uniform(0, 10, n)
        deaths = births + rng.uniform(0.1, 5, n)
        d = np.column_stack([births, deaths, np.zeros(n)])
        ls = persistence_landscape(d, num_levels=4, resolution=100)
        assert ls.shape == (4, 100)
        assert np.all(ls >= 0)


class TestPersistenceImage:
    def test_output_shape_square(self):
        img = persistence_image(_simple_diagram(), resolution=32, sigma=0.1)
        assert img.shape == (32, 32)

    def test_non_negative(self):
        img = persistence_image(_simple_diagram(), resolution=32, sigma=0.2)
        assert np.all(img >= 0)

    def test_empty_diagram_yields_zeros(self):
        img = persistence_image(_empty_diagram(), resolution=10)
        assert img.shape == (10, 10)
        assert np.all(img == 0.0)

    def test_diagonal_diagram_yields_zeros(self):
        img = persistence_image(_diagonal_diagram(), resolution=16, sigma=0.1)
        assert_allclose(img, 0.0)

    def test_single_point(self):
        img = persistence_image(_single_diagram(), resolution=64, sigma=0.1)
        assert img.shape == (64, 64)
        assert np.all(img >= 0)
        assert np.any(img > 0)

    def test_identical_points(self):
        img = persistence_image(_identical_diagram(), resolution=32, sigma=0.1)
        assert np.all(img >= 0)
        assert np.any(img > 0)

    def test_determinism(self):
        d = _simple_diagram()
        img1 = persistence_image(d, resolution=32, sigma=0.1)
        img2 = persistence_image(d, resolution=32, sigma=0.1)
        assert_allclose(img1, img2)

    def test_default_parameters(self):
        img = persistence_image(_simple_diagram())
        assert img.shape == (64, 64)

    def test_nan_birth_raises(self):
        d = np.array([[float("nan"), 1.0, 0.0]])
        with pytest.raises((InvalidArgumentError, ValidationError)):
            persistence_image(d)

    def test_negative_sigma_raises(self):
        with pytest.raises(InvalidArgumentError):
            persistence_image(_simple_diagram(), sigma=-0.1)

    def test_zero_sigma_raises(self):
        with pytest.raises(InvalidArgumentError):
            persistence_image(_simple_diagram(), sigma=0.0)

    def test_nan_sigma_raises(self):
        with pytest.raises(InvalidArgumentError):
            persistence_image(_simple_diagram(), sigma=float("nan"))

    def test_infinite_death_filtered(self):
        d = np.array([[1.0, float("inf"), 0.0]])
        img = persistence_image(d, resolution=32, sigma=0.1)
        assert_allclose(img, 0.0)

    def test_sigma_affects_spread(self):
        d = _single_diagram()
        img_small = persistence_image(d, resolution=64, sigma=0.05)
        img_large = persistence_image(d, resolution=64, sigma=0.5)
        assert img_large.sum() > img_small.sum()

    def test_resolution_affects_detail(self):
        d = _simple_diagram()
        img_low = persistence_image(d, resolution=10, sigma=0.1)
        img_high = persistence_image(d, resolution=100, sigma=0.1)
        assert img_low.shape == (10, 10)
        assert img_high.shape == (100, 100)
        assert img_low.sum() > 0
        assert img_high.sum() > 0

    def test_finite_output(self):
        d = _simple_diagram()
        img = persistence_image(d, resolution=32, sigma=0.1)
        assert np.isfinite(img).all()

    def test_high_resolution_single_point(self):
        d = _single_diagram()
        img = persistence_image(d, resolution=128, sigma=0.1)
        assert img.shape == (128, 128)
        assert np.any(img > 0)


class TestPersistenceSilhouette:
    def test_output_shape(self):
        s = persistence_silhouette(_simple_diagram(), resolution=100)
        assert s.shape == (100,)

    def test_non_negative(self):
        s = persistence_silhouette(_large_diagram(), resolution=80)
        assert np.all(s >= 0)

    def test_empty_diagram_yields_zeros(self):
        s = persistence_silhouette(_empty_diagram(), resolution=30)
        assert s.shape == (30,)
        assert np.all(s == 0.0)

    def test_diagonal_diagram_yields_zeros(self):
        s = persistence_silhouette(_diagonal_diagram(), resolution=100)
        assert_allclose(s, 0.0)

    def test_single_point(self):
        s = persistence_silhouette(_single_diagram(), resolution=200)
        assert np.any(s > 0)
        assert np.all(s >= 0)

    def test_weight_power_zero(self):
        d = _simple_diagram()
        s = persistence_silhouette(d, resolution=100, weight_power=0.0)
        assert s.shape == (100,)
        assert np.all(s >= 0)

    def test_weight_power_two(self):
        d = _simple_diagram()
        s = persistence_silhouette(d, resolution=100, weight_power=2.0)
        assert s.shape == (100,)
        assert np.all(s >= 0)

    def test_weight_power_affects_result(self):
        d = _single_diagram()
        s1 = persistence_silhouette(d, resolution=100, weight_power=1.0)
        s2 = persistence_silhouette(d, resolution=100, weight_power=2.0)
        assert not np.allclose(s1, s2)

    def test_determinism(self):
        d = _large_diagram()
        s1 = persistence_silhouette(d, resolution=50)
        s2 = persistence_silhouette(d, resolution=50)
        assert_allclose(s1, s2)

    def test_default_parameters(self):
        d = _simple_diagram()
        s = persistence_silhouette(d)
        assert s.shape == (100,)

    def test_nan_birth_raises(self):
        d = np.array([[float("nan"), 1.0, 0.0]])
        with pytest.raises((ValidationError, ValueError)):
            persistence_silhouette(d)

    def test_inf_birth_raises(self):
        d = np.array([[float("inf"), 1.0, 0.0]])
        with pytest.raises((ValidationError, ValueError)):
            persistence_silhouette(d)

    def test_death_less_than_birth_raises(self):
        d = np.array([[5.0, 3.0, 0.0]])
        with pytest.raises((ValidationError, ValueError)):
            persistence_silhouette(d)

    def test_infinite_death_filtered(self):
        d = np.array([[1.0, float("inf"), 0.0], [0.0, 2.0, 1.0]])
        s = persistence_silhouette(d, resolution=100)
        assert s.shape == (100,)
        assert np.any(s > 0)

    def test_many_points(self):
        rng = np.random.RandomState(456)
        n = 30
        births = rng.uniform(0, 10, n)
        deaths = births + rng.uniform(0.1, 5, n)
        d = np.column_stack([births, deaths, np.zeros(n)])
        s = persistence_silhouette(d, resolution=100)
        assert s.shape == (100,)
        assert np.all(s >= 0)

    def test_two_column_diagram(self):
        d = np.array([[1.0, 3.0], [2.0, 6.0]])
        s = persistence_silhouette(d, resolution=50)
        assert s.shape == (50,)
        assert np.all(s >= 0)

    def test_identical_points(self):
        s = persistence_silhouette(_identical_diagram(), resolution=100)
        assert np.all(s >= 0)

    def test_t_max_edge_case(self):
        d = np.array([[0.0, 0.1]])
        s = persistence_silhouette(d, resolution=10)
        assert np.all(s >= 0)


class TestPersistenceHeatVector:
    def test_output_shape(self):
        v = persistence_heat_vector(_simple_diagram(), resolution=100)
        assert v.shape == (100,)

    def test_non_negative(self):
        v = persistence_heat_vector(_large_diagram(), resolution=80)
        assert np.all(v >= 0)

    def test_empty_diagram_yields_zeros(self):
        v = persistence_heat_vector(_empty_diagram(), resolution=30)
        assert v.shape == (30,)
        assert np.all(v == 0.0)

    def test_diagonal_diagram_yields_zeros(self):
        v = persistence_heat_vector(_diagonal_diagram(), resolution=100)
        assert_allclose(v, 0.0)

    def test_single_point(self):
        v = persistence_heat_vector(_single_diagram(), resolution=200, sigma=0.5)
        assert v.shape == (200,)
        assert np.any(v > 0)
        assert np.all(v >= 0)

    def test_determinism(self):
        d = _large_diagram()
        v1 = persistence_heat_vector(d, resolution=50)
        v2 = persistence_heat_vector(d, resolution=50)
        assert_allclose(v1, v2)

    def test_default_parameters(self):
        d = _simple_diagram()
        v = persistence_heat_vector(d)
        assert v.shape == (100,)

    def test_sigma_affects_result(self):
        d = _single_diagram()
        v1 = persistence_heat_vector(d, resolution=100, sigma=0.1)
        v2 = persistence_heat_vector(d, resolution=100, sigma=2.0)
        assert not np.allclose(v1, v2)

    def test_t_parameter_zero(self):
        d = _single_diagram()
        v = persistence_heat_vector(d, resolution=50, t=0.0)
        assert v.shape == (50,)
        assert np.all(v >= 0)

    def test_t_parameter_large(self):
        d = _single_diagram()
        v = persistence_heat_vector(d, resolution=50, t=10.0)
        assert v.shape == (50,)
        assert np.all(v >= 0)

    def test_t_parameter_affects_result(self):
        d = _simple_diagram()
        v1 = persistence_heat_vector(d, resolution=50, t=0.0)
        v2 = persistence_heat_vector(d, resolution=50, t=5.0)
        assert not np.allclose(v1, v2)

    def test_nan_birth_raises(self):
        d = np.array([[float("nan"), 1.0, 0.0]])
        with pytest.raises((ValidationError, ValueError)):
            persistence_heat_vector(d)

    def test_inf_birth_raises(self):
        d = np.array([[float("inf"), 1.0, 0.0]])
        with pytest.raises((ValidationError, ValueError)):
            persistence_heat_vector(d)

    def test_death_less_than_birth_raises(self):
        d = np.array([[5.0, 3.0, 0.0]])
        with pytest.raises((ValidationError, ValueError)):
            persistence_heat_vector(d)

    def test_infinite_death_filtered(self):
        d = np.array([[1.0, float("inf"), 0.0], [0.0, 2.0, 1.0]])
        v = persistence_heat_vector(d, resolution=100, sigma=1.0, t=1.0)
        assert v.shape == (100,)
        assert np.any(v > 0)

    def test_two_column_diagram(self):
        d = np.array([[1.0, 3.0], [2.0, 6.0]])
        v = persistence_heat_vector(d, resolution=50, sigma=1.0, t=1.0)
        assert v.shape == (50,)
        assert np.all(v >= 0)

    def test_many_points(self):
        rng = np.random.RandomState(789)
        n = 30
        births = rng.uniform(0, 10, n)
        deaths = births + rng.uniform(0.1, 5, n)
        d = np.column_stack([births, deaths, np.zeros(n)])
        v = persistence_heat_vector(d, resolution=100, sigma=2.0, t=0.5)
        assert v.shape == (100,)
        assert np.all(v >= 0)

    def test_identical_points(self):
        v = persistence_heat_vector(_identical_diagram(), resolution=100, sigma=1.0, t=1.0)
        assert np.all(v >= 0)

    def test_t_max_edge_case(self):
        d = np.array([[0.0, 0.1]])
        v = persistence_heat_vector(d, resolution=10)
        assert np.all(v >= 0)

    def test_finite_output(self):
        d = _large_diagram()
        v = persistence_heat_vector(d, resolution=50, sigma=1.0, t=1.0)
        assert np.isfinite(v).all()


class TestGaussianKernelMatrix:
    def test_output_shape_symmetric(self):
        d = _simple_diagram()
        K = gaussian_kernel_matrix(d)
        assert K.shape == (len(d), len(d))

    def test_non_negative(self):
        d = _large_diagram()
        K = gaussian_kernel_matrix(d, sigma=1.0)
        assert np.all(K >= 0)

    def test_max_value_one(self):
        d = _simple_diagram()
        K = gaussian_kernel_matrix(d, sigma=1.0)
        assert np.all(K <= 1.0 + 1e-12)

    def test_symmetric_matrix(self):
        d = _large_diagram()
        K = gaussian_kernel_matrix(d, sigma=1.0)
        assert_allclose(K, K.T)

    def test_self_kernel(self):
        d = _simple_diagram()
        K = gaussian_kernel_matrix(d, d, sigma=1.0)
        assert K.shape == (len(d), len(d))

    def test_different_diagrams(self):
        d1 = _simple_diagram()
        d2 = np.array([[0.5, 1.5, 0.0], [3.0, 6.0, 1.0]])
        K = gaussian_kernel_matrix(d1, d2, sigma=1.0)
        assert K.shape == (2, 2)
        assert np.all(K >= 0)
        assert np.all(K <= 1.0)

    def test_single_point_diagram(self):
        d1 = _single_diagram()
        d2 = _simple_diagram()
        K = gaussian_kernel_matrix(d1, d2, sigma=1.0)
        assert K.shape == (1, 2)
        assert np.all(K >= 0)

    def test_empty_first_diagram(self):
        d1 = np.empty((0, 3))
        d2 = _simple_diagram()
        K = gaussian_kernel_matrix(d1, d2, sigma=1.0)
        assert K.shape == (0, 2)

    def test_empty_second_diagram(self):
        d1 = _simple_diagram()
        d2 = np.empty((0, 3))
        K = gaussian_kernel_matrix(d1, d2, sigma=1.0)
        assert K.shape == (2, 0)

    def test_both_empty(self):
        K = gaussian_kernel_matrix(np.empty((0, 3)), sigma=1.0)
        assert K.shape == (0, 0)

    def test_sigma_affects_result(self):
        d = _simple_diagram()
        K1 = gaussian_kernel_matrix(d, sigma=0.1)
        K2 = gaussian_kernel_matrix(d, sigma=10.0)
        assert not np.allclose(K1, K2)

    def test_determinism(self):
        d = _large_diagram()
        K1 = gaussian_kernel_matrix(d, sigma=1.0)
        K2 = gaussian_kernel_matrix(d, sigma=1.0)
        assert_allclose(K1, K2)

    def test_kernel_of_same_point_is_one(self):
        d = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        K = gaussian_kernel_matrix(d, sigma=1.0)
        assert_allclose(K[0, 1], 1.0, atol=1e-10)

    def test_kernel_of_different_points_less_than_one(self):
        d = np.array([[0.0, 1.0, 0.0], [2.0, 5.0, 1.0]])
        K = gaussian_kernel_matrix(d, sigma=1.0)
        assert K[0, 1] < 1.0

    def test_nan_birth_raises(self):
        d = np.array([[float("nan"), 1.0]])
        with pytest.raises((ValidationError, ValueError)):
            gaussian_kernel_matrix(d)

    def test_inf_birth_raises(self):
        d = np.array([[float("inf"), 1.0]])
        with pytest.raises((ValidationError, ValueError)):
            gaussian_kernel_matrix(d)

    def test_death_less_than_birth_raises(self):
        d = np.array([[5.0, 3.0]])
        with pytest.raises((ValidationError, ValueError)):
            gaussian_kernel_matrix(d)

    def test_two_column_diagram(self):
        d = np.array([[0.0, 1.0], [2.0, 5.0]])
        K = gaussian_kernel_matrix(d, sigma=1.0)
        assert K.shape == (2, 2)
        assert np.all(K >= 0)

    def test_infinite_death_filtered(self):
        d = np.array([[1.0, float("inf"), 0.0], [0.0, 2.0, 1.0]])
        K = gaussian_kernel_matrix(d, sigma=1.0)
        assert K.shape == (1, 1)
        assert_allclose(K[0, 0], 1.0, atol=1e-10)

    def test_large_sigma_near_uniform(self):
        d = _simple_diagram()
        K = gaussian_kernel_matrix(d, sigma=1000.0)
        assert_allclose(K, 1.0, atol=1e-3)

    def test_tiny_sigma_diagonal_one(self):
        d = _simple_diagram()
        K = gaussian_kernel_matrix(d, sigma=0.001)
        assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_diagonal_points_filtered(self):
        d = _diagonal_diagram()
        K = gaussian_kernel_matrix(d, sigma=1.0)
        assert K.shape == (0, 0)


class TestModuleAll:
    def test_all_exports_match(self):
        from pynerve.algorithms import __all__ as exports

        expected = {
            "gaussian_kernel_matrix",
            "knn",
            "pairwise_distances",
            "persistence_heat_vector",
            "persistence_image",
            "persistence_landscape",
            "persistence_silhouette",
        }
        assert set(exports) == expected

    def test_all_functions_importable(self):
        for name in [
            "gaussian_kernel_matrix",
            "knn",
            "pairwise_distances",
            "persistence_heat_vector",
            "persistence_image",
            "persistence_landscape",
            "persistence_silhouette",
        ]:
            obj = getattr(__import__("pynerve.algorithms", fromlist=[name]), name)
            assert callable(obj), f"{name} should be callable"
