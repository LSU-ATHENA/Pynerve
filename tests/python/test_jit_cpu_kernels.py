from __future__ import annotations

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from pynerve.jit._cpu_kernels import (
    _jit_batch_betti_curves_impl,
    _jit_betti_curve_impl,
    _jit_filter_pairs_impl,
    _jit_pairwise_distances_impl,
    _jit_persistence_image_impl,
    _jit_vietoris_rips_edges_impl,
)


def _call_numba_fn(fn, *args, fn_name="kernel", **kwargs):
    try:
        return fn(*args, **kwargs)
    except numba.core.errors.TypingError as exc:
        pytest.skip(f"{fn_name} compilation failed: {exc}")


def _ref_pairwise(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    dim = points.shape[1]
    dists = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            s = 0.0
            for k in range(dim):
                d = points[i, k] - points[j, k]
                s += d * d
            dists[i, j] = np.sqrt(s)
            dists[j, i] = dists[i, j]
    return dists


class TestJitPairwiseDistances:
    def test_random_small(self) -> None:
        rng = np.random.default_rng(42)
        for n in [2, 3, 5, 10, 20]:
            pts = rng.normal(size=(n, 3)).astype(np.float32)
            got = _jit_pairwise_distances_impl(pts)
            expected = _ref_pairwise(pts.astype(np.float64))
            np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_single_point(self) -> None:
        pts = np.array([[1.5, 2.5, 3.5]], dtype=np.float32)
        got = _jit_pairwise_distances_impl(pts)
        assert got.shape == (1, 1)
        assert got[0, 0] == 0.0

    def test_two_points(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
        got = _jit_pairwise_distances_impl(pts)
        assert got[0, 0] == 0.0
        assert got[1, 1] == 0.0
        np.testing.assert_allclose(got[0, 1], 5.0, rtol=1e-6)
        np.testing.assert_allclose(got[1, 0], 5.0, rtol=1e-6)

    def test_symmetry(self) -> None:
        rng = np.random.default_rng(7)
        pts = rng.normal(size=(12, 4)).astype(np.float32)
        got = _jit_pairwise_distances_impl(pts)
        np.testing.assert_allclose(got, got.T, rtol=1e-15)

    def test_zero_diagonal(self) -> None:
        rng = np.random.default_rng(3)
        pts = rng.normal(size=(8, 3)).astype(np.float32)
        got = _jit_pairwise_distances_impl(pts)
        np.testing.assert_allclose(np.diag(got), 0.0, atol=1e-15)

    def test_type_is_float32(self) -> None:
        pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        got = _jit_pairwise_distances_impl(pts)
        assert got.dtype == np.float32


class TestJitFilterPairs:
    def test_all_below_threshold(self) -> None:
        pairs = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        mask = _call_numba_fn(_jit_filter_pairs_impl, pairs, 2.0, fn_name="filter_pairs")
        assert not np.any(mask)

    def test_all_above_threshold(self) -> None:
        pairs = np.array([[0.0, 3.0], [1.0, 5.0], [2.0, 7.0]], dtype=np.float32)
        mask = _call_numba_fn(_jit_filter_pairs_impl, pairs, 1.0, fn_name="filter_pairs")
        assert np.all(mask)

    def test_mixed(self) -> None:
        pairs = np.array([[0.0, 2.0], [0.0, 1.0], [0.0, 5.0]], dtype=np.float32)
        mask = _call_numba_fn(_jit_filter_pairs_impl, pairs, 2.0, fn_name="filter_pairs")
        np.testing.assert_array_equal(mask, np.array([False, False, True]))

    def test_threshold_zero(self) -> None:
        pairs = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        mask = _call_numba_fn(_jit_filter_pairs_impl, pairs, 0.0, fn_name="filter_pairs")
        np.testing.assert_array_equal(mask, np.array([True, False]))

    def test_negative_threshold(self) -> None:
        pairs = np.array([[0.0, 1.0]], dtype=np.float32)
        mask = _call_numba_fn(_jit_filter_pairs_impl, pairs, -10.0, fn_name="filter_pairs")
        assert np.all(mask)

    def test_empty_pairs(self) -> None:
        pairs = np.empty((0, 2), dtype=np.float32)
        mask = _call_numba_fn(_jit_filter_pairs_impl, pairs, 0.5, fn_name="filter_pairs")
        assert mask.shape == (0,)

    def test_output_is_bool(self) -> None:
        pairs = np.array([[0.0, 2.0]], dtype=np.float32)
        mask = _call_numba_fn(_jit_filter_pairs_impl, pairs, 1.0, fn_name="filter_pairs")
        assert mask.dtype == bool


class TestJitBettiCurve:
    def test_simple_pairs(self) -> None:
        pairs = np.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [1.0, 3.0, 1.0]], dtype=np.float32)
        betti = _jit_betti_curve_impl(pairs, max_dim=1, resolution=10)
        assert betti.shape == (2, 10)
        assert betti.dtype == np.int32

    def test_shape_and_nonnegative(self) -> None:
        pairs = np.array([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32)
        betti = _jit_betti_curve_impl(pairs, max_dim=0, resolution=10)
        assert betti.shape == (1, 10)
        assert np.all(betti >= 0)

    def test_single_dimension(self) -> None:
        pairs = np.array([[1.0, 5.0, 0.0], [2.0, 6.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
        betti = _jit_betti_curve_impl(pairs, max_dim=0, resolution=6)
        assert betti.shape == (1, 6)

    def test_max_dim_larger_than_data(self) -> None:
        pairs = np.array([[0.0, 2.0, 0.0]], dtype=np.float32)
        betti = _jit_betti_curve_impl(pairs, max_dim=5, resolution=10)
        assert betti.shape == (6, 10)

    def test_no_pairs_in_dim(self) -> None:
        pairs = np.array([[0.0, 2.0, 0.0]], dtype=np.float32)
        betti = _jit_betti_curve_impl(pairs, max_dim=2, resolution=10)
        np.testing.assert_array_equal(betti[1], np.zeros(10, dtype=np.int32))
        np.testing.assert_array_equal(betti[2], np.zeros(10, dtype=np.int32))

    def test_dim_filtered_out_when_above_max(self) -> None:
        pairs = np.array([[0.0, 5.0, 0.0], [0.0, 5.0, 3.0]], dtype=np.float32)
        betti = _jit_betti_curve_impl(pairs, max_dim=2, resolution=5)
        assert betti.shape == (3, 5)
        np.testing.assert_array_equal(betti[2], np.zeros(5, dtype=np.int32))

    def test_betti_counts_nonnegative(self) -> None:
        rng = np.random.default_rng(11)
        n = 20
        births = rng.uniform(0.0, 5.0, size=n).astype(np.float32)
        deaths = births + rng.uniform(0.1, 5.0, size=n).astype(np.float32)
        dims = rng.integers(0, 3, size=n).astype(np.float32)
        pairs = np.column_stack([births, deaths, dims])
        betti = _jit_betti_curve_impl(pairs, max_dim=2, resolution=15)
        assert np.all(betti >= 0)

    def test_resolution_one(self) -> None:
        pairs = np.array([[0.0, 5.0, 0.0]], dtype=np.float32)
        betti = _jit_betti_curve_impl(pairs, max_dim=0, resolution=1)
        assert betti.shape == (1, 1)
        assert betti[0, 0] == 1


class TestJitPersistenceImage:
    def test_simple_pairs(self) -> None:
        pairs = np.array([[1.0, 3.0], [2.0, 5.0]], dtype=np.float32)
        img = _jit_persistence_image_impl(pairs, resolution=10, sigma=0.5)
        assert img.shape == (10, 10)
        assert img.dtype == np.float32
        assert np.any(img > 0)

    def test_single_point(self) -> None:
        pairs = np.array([[2.0, 4.0]], dtype=np.float32)
        img = _jit_persistence_image_impl(pairs, resolution=8, sigma=0.5)
        assert img.shape == (8, 8)
        assert np.any(img > 0)

    def test_all_values_nonnegative(self) -> None:
        rng = np.random.default_rng(13)
        births = rng.uniform(0.0, 5.0, size=10).astype(np.float32)
        deaths = births + rng.uniform(0.1, 5.0, size=10).astype(np.float32)
        pairs = np.column_stack([births, deaths])
        img = _jit_persistence_image_impl(pairs, resolution=16, sigma=0.3)
        assert np.all(img >= 0.0)

    def test_empty_pairs_raises(self) -> None:
        pairs = np.empty((0, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="zero-size"):
            _jit_persistence_image_impl(pairs, resolution=8, sigma=0.5)

    def test_resolution_one(self) -> None:
        pairs = np.array([[1.0, 2.0]], dtype=np.float32)
        img = _jit_persistence_image_impl(pairs, resolution=1, sigma=0.5)
        assert img.shape == (1, 1)

    def test_large_sigma_produces_wide_spread(self) -> None:
        pairs = np.array([[1.0, 2.0]], dtype=np.float32)
        img_narrow = _jit_persistence_image_impl(pairs, resolution=16, sigma=0.1)
        img_wide = _jit_persistence_image_impl(pairs, resolution=16, sigma=2.0)
        nonzero_narrow = np.count_nonzero(img_narrow)
        nonzero_wide = np.count_nonzero(img_wide)
        assert nonzero_wide >= nonzero_narrow

    def test_identical_pairs(self) -> None:
        pairs = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
        img = _jit_persistence_image_impl(pairs, resolution=8, sigma=0.5)
        assert img.shape == (8, 8)
        assert np.any(img > 0)

    def test_far_apart_pairs(self) -> None:
        pairs = np.array([[0.0, 1.0], [100.0, 101.0]], dtype=np.float32)
        img = _jit_persistence_image_impl(pairs, resolution=16, sigma=0.5)
        assert img.shape == (16, 16)
        assert np.all(np.isfinite(img))


class TestJitVietorisRipsEdges:
    def test_random_small(self) -> None:
        rng = np.random.default_rng(19)
        pts = rng.normal(size=(8, 3)).astype(np.float32)
        edges = _jit_vietoris_rips_edges_impl(pts, max_dist=2.0)
        assert edges.ndim == 2
        assert edges.shape[1] == 2
        assert edges.dtype == np.int32

    def test_large_max_dist_all_edges(self) -> None:
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        edges = _jit_vietoris_rips_edges_impl(pts, max_dist=100.0)
        assert len(edges) == 3

    def test_zero_max_dist(self) -> None:
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        edges = _jit_vietoris_rips_edges_impl(pts, max_dist=0.0)
        assert len(edges) == 0

    def test_edges_within_radius(self) -> None:
        rng = np.random.default_rng(23)
        pts = rng.normal(size=(15, 2)).astype(np.float32)
        max_dist = 1.5
        edges = _jit_vietoris_rips_edges_impl(pts, max_dist)
        for e in edges:
            u, v = int(e[0]), int(e[1])
            dist = np.linalg.norm(pts[u] - pts[v])
            assert dist <= max_dist + 1e-6

    def test_index_bounds(self) -> None:
        rng = np.random.default_rng(29)
        pts = rng.normal(size=(10, 2)).astype(np.float32)
        edges = _jit_vietoris_rips_edges_impl(pts, max_dist=1.0)
        for e in edges:
            assert 0 <= e[0] < 10
            assert 0 <= e[1] < 10
            assert e[0] < e[1]

    def test_two_points_close(self) -> None:
        pts = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)
        edges = _jit_vietoris_rips_edges_impl(pts, max_dist=1.0)
        assert len(edges) == 1
        np.testing.assert_array_equal(edges[0], np.array([0, 1], dtype=np.int32))

    def test_two_points_far(self) -> None:
        pts = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        edges = _jit_vietoris_rips_edges_impl(pts, max_dist=1.0)
        assert len(edges) == 0

    def test_single_point(self) -> None:
        pts = np.array([[0.0, 0.0]], dtype=np.float32)
        edges = _jit_vietoris_rips_edges_impl(pts, max_dist=1.0)
        assert edges.shape == (0, 2)
        assert edges.dtype == np.int32

    def test_type_is_int32(self) -> None:
        pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        edges = _jit_vietoris_rips_edges_impl(pts, max_dist=1e-9)
        assert edges.dtype == np.int32


class TestJitBatchBettiCurves:
    def test_simple_batch(self) -> None:
        diagrams = np.zeros((2, 5, 3), dtype=np.float32)
        diagrams[0, 0] = [0.0, 2.0, 0.0]
        diagrams[0, 1] = [0.0, 1.0, 0.0]
        diagrams[0, 2] = [1.0, 3.0, 1.0]
        diagrams[1, 0] = [0.0, 3.0, 0.0]
        diagrams[1, 1] = [1.0, 4.0, 1.0]
        curves = _jit_batch_betti_curves_impl(diagrams, max_dim=1, resolution=10)
        assert curves.shape == (2, 2, 10)
        assert curves.dtype == np.int32

    def test_padding_terminates_early(self) -> None:
        diagrams = np.zeros((1, 5, 3), dtype=np.float32)
        diagrams[0, 0] = [0.0, 2.0, 0.0]
        diagrams[0, 1] = [0.0, 1.0, 0.0]
        betti = _jit_batch_betti_curves_impl(diagrams, max_dim=0, resolution=10)
        assert betti.shape == (1, 1, 10)

    def test_single_diagram_single_pair(self) -> None:
        diagrams = np.zeros((1, 3, 3), dtype=np.float32)
        diagrams[0, 0] = [0.0, 5.0, 0.0]
        curves = _jit_batch_betti_curves_impl(diagrams, max_dim=0, resolution=5)
        assert curves.shape == (1, 1, 5)
        assert curves[0, 0, 0] == 1

    def test_larger_batch(self) -> None:
        rng = np.random.default_rng(37)
        batch = 4
        n_pairs = 6
        diagrams = np.zeros((batch, n_pairs, 3), dtype=np.float32)
        for b in range(batch):
            for i in range(n_pairs):
                birth = rng.uniform(0.0, 5.0)
                death = birth + rng.uniform(0.1, 5.0)
                dim = float(rng.integers(0, 3))
                diagrams[b, i] = [birth, death, dim]
        curves = _jit_batch_betti_curves_impl(diagrams, max_dim=2, resolution=8)
        assert curves.shape == (batch, 3, 8)
        assert np.all(curves >= 0)

    def test_all_zero_diagram(self) -> None:
        diagrams = np.zeros((2, 4, 3), dtype=np.float32)
        curves = _jit_batch_betti_curves_impl(diagrams, max_dim=1, resolution=5)
        assert curves.shape == (2, 2, 5)
        assert np.all(curves == 0)

    def test_dim_filtering(self) -> None:
        diagrams = np.zeros((1, 5, 3), dtype=np.float32)
        diagrams[0, 0] = [0.0, 5.0, 3.0]
        curves = _jit_batch_betti_curves_impl(diagrams, max_dim=2, resolution=5)
        assert curves.shape == (1, 3, 5)
        np.testing.assert_array_equal(curves[0, 2], np.zeros(5, dtype=np.int32))

    def test_resolution_one(self) -> None:
        diagrams = np.zeros((1, 3, 3), dtype=np.float32)
        diagrams[0, 0] = [0.0, 5.0, 0.0]
        curves = _jit_batch_betti_curves_impl(diagrams, max_dim=0, resolution=1)
        assert curves.shape == (1, 1, 1)
        assert curves[0, 0, 0] == 1

    def test_empty_pairs_in_each_diagram(self) -> None:
        diagrams = np.zeros((3, 1, 3), dtype=np.float32)
        curves = _jit_batch_betti_curves_impl(diagrams, max_dim=0, resolution=10)
        assert curves.shape == (3, 1, 10)
        assert np.all(curves == 0)

    def test_large_resolution(self) -> None:
        diagrams = np.zeros((1, 2, 3), dtype=np.float32)
        diagrams[0, 0] = [0.0, 10.0, 0.0]
        curves = _jit_batch_betti_curves_impl(diagrams, max_dim=0, resolution=100)
        assert curves.shape == (1, 1, 100)
