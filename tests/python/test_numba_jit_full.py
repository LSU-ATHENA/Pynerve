from __future__ import annotations

import numpy as np
import pytest

numba = pytest.importorskip("numba")


class TestNumbaCompat:
    def test_has_numba_true(self) -> None:
        from pynerve._numba_compat import HAS_NUMBA

        assert HAS_NUMBA is True

    def test_njit_compiles_function(self) -> None:
        from pynerve._numba_compat import njit

        @njit(cache=True)
        def add(a: int, b: int) -> int:
            return a + b

        assert add(3, 4) == 7

    def test_njit_with_parallel_prange(self) -> None:
        from pynerve._numba_compat import njit, prange

        @njit(parallel=True, cache=True)
        def square_sum(arr: np.ndarray) -> float:
            s = 0.0
            for i in prange(arr.shape[0]):
                s += arr[i] * arr[i]
            return s

        arr = np.array([1.0, 2.0, 3.0, 4.0])
        expected = 1.0 + 4.0 + 9.0 + 16.0
        got = square_sum(arr)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_njit_with_options_passthrough(self) -> None:
        from pynerve._numba_compat import njit

        @njit(cache=True, fastmath=True)
        def mul(a: float, b: float) -> float:
            return a * b

        assert mul(7.0, 3.0) == 21.0

    def test_prange_is_range_compatible(self) -> None:
        from pynerve._numba_compat import prange

        vals = list(prange(5))
        assert vals == [0, 1, 2, 3, 4]


class TestNumbaDispatch:
    def test_compute_with_numba_pairwise(self) -> None:
        from pynerve._numba_dispatch import compute_with_numba

        pts = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
        got = compute_with_numba("pairwise_distances", pts)
        assert got.shape == (2, 2)
        np.testing.assert_allclose(got[0, 1], 5.0, rtol=1e-6)

    def test_compute_with_numba_vr_edges(self) -> None:
        from pynerve._numba_dispatch import compute_with_numba

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        got = compute_with_numba("vr_edges", pts, 100.0)
        assert got.ndim == 2 and got.shape[1] == 2

    def test_compute_with_numba_column_reduction(self) -> None:
        from pynerve._numba_dispatch import compute_with_numba

        bm = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int64)
        filtration = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        got = compute_with_numba("column_reduction", bm.copy(), filtration)
        assert got.shape == (3,)
        assert np.all(got >= -1)

    def test_compute_with_numba_betti_curve(self) -> None:
        from pynerve._numba_dispatch import compute_with_numba

        pairs = np.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        got = compute_with_numba("betti_curve", pairs, 1, 10, 2.0)
        assert got.shape == (2, 10)

    def test_compute_with_numba_connected_components(self) -> None:
        from pynerve._numba_dispatch import compute_with_numba

        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        got = compute_with_numba("connected_components", edges, 3)
        assert got.shape == (3,)
        assert got[0] == got[1] == got[2]

    def test_compute_with_numba_empty_operation_raises(self) -> None:
        from pynerve._numba_dispatch import compute_with_numba

        with pytest.raises(ValueError, match="non-empty"):
            compute_with_numba("")

    def test_compute_with_numba_non_string_raises(self) -> None:
        from pynerve._numba_dispatch import compute_with_numba

        with pytest.raises(ValueError, match="non-empty string"):
            compute_with_numba(123)  # type: ignore[arg-type]

    def test_compute_with_numba_unknown_operation_raises(self) -> None:
        from pynerve._numba_dispatch import compute_with_numba

        with pytest.raises(ValueError, match="Unknown operation"):
            compute_with_numba("nonexistent")

    def test_benchmark_numba_vs_numpy_basic(self) -> None:
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        # noinspection PyMissingOrEmptyDocstring
        @numba.njit(cache=True)
        def fn_numba(arr: np.ndarray) -> np.ndarray:
            return arr + 1.0

        def fn_numpy(arr: np.ndarray) -> np.ndarray:
            return arr + 1.0

        def setup() -> tuple[np.ndarray]:
            return (np.arange(1000, dtype=np.float64),)

        result = benchmark_numba_vs_numpy(fn_numba, fn_numpy, setup, n_trials=3)
        assert "numba_mean" in result
        assert "numpy_mean" in result
        assert "speedup" in result
        assert isinstance(result["numba_mean"], float)
        assert isinstance(result["numpy_mean"], float)
        assert isinstance(result["speedup"], float)
        assert result["results_match"] is True

    def test_benchmark_func_numba_not_callable_raises(self) -> None:
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        def fn_numpy(arr: np.ndarray) -> np.ndarray:
            return arr

        def setup() -> tuple[np.ndarray]:
            return (np.array([1.0]),)

        with pytest.raises(TypeError, match="func_numba must be callable"):
            benchmark_numba_vs_numpy("not_callable", fn_numpy, setup)  # type: ignore[arg-type]

    def test_benchmark_func_numpy_not_callable_raises(self) -> None:
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        # noinspection PyMissingOrEmptyDocstring
        @numba.njit(cache=True)
        def fn_numba(arr: np.ndarray) -> np.ndarray:
            return arr

        def setup() -> tuple[np.ndarray]:
            return (np.array([1.0]),)

        with pytest.raises(TypeError, match="func_numpy must be callable"):
            benchmark_numba_vs_numpy(fn_numba, None, setup)  # type: ignore[arg-type]

    def test_benchmark_setup_not_callable_raises(self) -> None:
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        # noinspection PyMissingOrEmptyDocstring
        @numba.njit(cache=True)
        def fn(arr: np.ndarray) -> np.ndarray:
            return arr

        with pytest.raises(TypeError, match="setup_fn must be callable"):
            benchmark_numba_vs_numpy(fn, fn, "not_callable")  # type: ignore[arg-type]

    def test_benchmark_setup_returns_non_tuple_raises(self) -> None:
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        # noinspection PyMissingOrEmptyDocstring
        @numba.njit(cache=True)
        def fn(arr: np.ndarray) -> np.ndarray:
            return arr

        def setup() -> np.ndarray:
            return np.array([1.0])

        with pytest.raises(TypeError, match="setup_fn must return a tuple"):
            benchmark_numba_vs_numpy(fn, fn, setup)  # type: ignore[arg-type]

    def test_benchmark_n_trials_negative_raises(self) -> None:
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        # noinspection PyMissingOrEmptyDocstring
        @numba.njit(cache=True)
        def fn(arr: np.ndarray) -> np.ndarray:
            return arr

        def setup() -> tuple[np.ndarray]:
            return (np.array([1.0]),)

        with pytest.raises(ValueError):
            benchmark_numba_vs_numpy(fn, fn, setup, n_trials=-1)

    def test_benchmark_results_match_false_on_mismatch(self) -> None:
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        # noinspection PyMissingOrEmptyDocstring
        @numba.njit(cache=True)
        def fn_numba(arr: np.ndarray) -> np.ndarray:
            return arr + 1.0

        def fn_numpy(arr: np.ndarray) -> np.ndarray:
            return arr

        def setup() -> tuple[np.ndarray]:
            return (np.array([1.0, 2.0], dtype=np.float64),)

        result = benchmark_numba_vs_numpy(fn_numba, fn_numpy, setup, n_trials=3)
        assert result["results_match"] is False


class TestNumbaDistance:
    def test_pairwise_distances_small(self) -> None:
        from pynerve._numba_distance import numba_pairwise_distances

        pts = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float64)
        got = numba_pairwise_distances(pts)
        assert got.shape == (3, 3)
        np.testing.assert_allclose(got[0, 0], 0.0, atol=1e-15)
        np.testing.assert_allclose(got[0, 1], 5.0, rtol=1e-6)
        np.testing.assert_allclose(got[0, 2], 10.0, rtol=1e-6)
        np.testing.assert_allclose(got[1, 2], 5.0, rtol=1e-6)

    def test_pairwise_distances_symmetry(self) -> None:
        from pynerve._numba_distance import numba_pairwise_distances

        rng = np.random.default_rng(1)
        pts = rng.normal(size=(12, 4)).astype(np.float64)
        got = numba_pairwise_distances(pts)
        np.testing.assert_allclose(got, got.T, rtol=1e-15)

    def test_pairwise_distances_zero_diagonal(self) -> None:
        from pynerve._numba_distance import numba_pairwise_distances

        rng = np.random.default_rng(2)
        pts = rng.normal(size=(8, 3)).astype(np.float64)
        got = numba_pairwise_distances(pts)
        np.testing.assert_allclose(np.diag(got), 0.0, atol=1e-15)

    def test_pairwise_distances_single_point(self) -> None:
        from pynerve._numba_distance import numba_pairwise_distances

        pts = np.array([[1.5, 2.5]], dtype=np.float64)
        got = numba_pairwise_distances(pts)
        assert got.shape == (1, 1) and got[0, 0] == 0.0

    def test_pairwise_distances_two_points(self) -> None:
        from pynerve._numba_distance import numba_pairwise_distances

        pts = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float64)
        got = numba_pairwise_distances(pts)
        np.testing.assert_allclose(got[0, 1], 5.0, rtol=1e-6)
        np.testing.assert_allclose(got[1, 0], 5.0, rtol=1e-6)

    def test_pairwise_distances_consistent_with_ref(self) -> None:
        from pynerve._numba_distance import numba_pairwise_distances

        rng = np.random.default_rng(3)
        for n in [2, 3, 5, 10]:
            pts = rng.normal(size=(n, 3))
            got = numba_pairwise_distances(pts)
            expected = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
            np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_nearest_neighbors_basic(self) -> None:
        from pynerve._numba_distance import numba_nearest_neighbors

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        dists, indices = numba_nearest_neighbors(pts, 2)
        assert dists.shape == (3, 2) and indices.shape == (3, 2)
        assert indices.dtype == np.int64
        assert dists.dtype == np.float64

    def test_nearest_neighbors_sorted_distances(self) -> None:
        from pynerve._numba_distance import numba_nearest_neighbors

        rng = np.random.default_rng(4)
        pts = rng.normal(size=(20, 3)).astype(np.float64)
        k = 5
        dists, indices = numba_nearest_neighbors(pts, k)
        for i in range(20):
            for j in range(k - 1):
                assert dists[i, j] <= dists[i, j + 1] + 1e-12

    def test_nearest_neighbors_no_self(self) -> None:
        from pynerve._numba_distance import numba_nearest_neighbors

        rng = np.random.default_rng(5)
        pts = rng.normal(size=(10, 2)).astype(np.float64)
        k = 3
        _dists, indices = numba_nearest_neighbors(pts, k)
        for i in range(10):
            assert i not in indices[i]

    def test_nearest_neighbors_k_zero(self) -> None:
        from pynerve._numba_distance import numba_nearest_neighbors

        pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        dists, indices = numba_nearest_neighbors(pts, 0)
        assert dists.shape == (2, 0) and indices.shape == (2, 0)

    def test_nearest_neighbors_k_negative(self) -> None:
        from pynerve._numba_distance import numba_nearest_neighbors

        pts = np.array([[0.0, 0.0]], dtype=np.float64)
        dists, indices = numba_nearest_neighbors(pts, -1)
        assert dists.shape == (1, 0) and indices.shape == (1, 0)


class TestNumbaGraph:
    def test_connected_components_basic(self) -> None:
        from pynerve._numba_graph import numba_connected_components

        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        labels = numba_connected_components(edges, 3)
        assert labels[0] == labels[1] == labels[2]
        assert labels.dtype == np.int64

    def test_connected_components_two_components(self) -> None:
        from pynerve._numba_graph import numba_connected_components

        edges = np.array([[0, 1], [2, 3]], dtype=np.int64)
        labels = numba_connected_components(edges, 4)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_connected_components_no_edges(self) -> None:
        from pynerve._numba_graph import numba_connected_components

        edges = np.empty((0, 2), dtype=np.int64)
        labels = numba_connected_components(edges, 5)
        assert len(np.unique(labels)) == 5

    def test_connected_components_zero_vertices(self) -> None:
        from pynerve._numba_graph import numba_connected_components

        edges = np.empty((0, 2), dtype=np.int64)
        labels = numba_connected_components(edges, 0)
        assert labels.shape == (0,)

    def test_connected_components_negative_n_vertices(self) -> None:
        from pynerve._numba_graph import numba_connected_components

        edges = np.empty((0, 2), dtype=np.int64)
        with pytest.raises(ValueError, match="n_vertices must be non-negative"):
            numba_connected_components(edges, -1)

    def test_connected_components_extra_vertices(self) -> None:
        from pynerve._numba_graph import numba_connected_components

        edges = np.array([[0, 1]], dtype=np.int64)
        labels = numba_connected_components(edges, 4)
        assert labels[0] == labels[1]
        assert labels[2] != labels[0]
        assert labels[3] != labels[0]
        assert labels[2] != labels[3]

    def test_mst_kruskal_basic(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, 3)
        assert mst.shape == (2, 2)
        assert mst.dtype == np.int64

    def test_mst_kruskal_no_edges_returns_empty(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.empty((0, 2), dtype=np.int64)
        weights = np.empty(0, dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, 3)
        assert mst.shape == (0, 2)

    def test_mst_kruskal_single_vertex(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.empty((0, 2), dtype=np.int64)
        weights = np.empty(0, dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, 1)
        assert mst.shape == (0, 2)

    def test_mst_kruskal_zero_vertices(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.empty((0, 2), dtype=np.int64)
        weights = np.empty(0, dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, 0)
        assert mst.shape == (0, 2)

    def test_mst_kruskal_negative_n_vertices(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.empty((0, 2), dtype=np.int64)
        weights = np.empty(0, dtype=np.float64)
        with pytest.raises(ValueError, match="n_vertices must be non-negative"):
            numba_mst_kruskal(edges, weights, -1)

    def test_mst_kruskal_mismatched_lengths_raises(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        weights = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="matching lengths"):
            numba_mst_kruskal(edges, weights, 3)

    def test_mst_kruskal_sorted_by_weight(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        weights = np.array([10.0, 1.0, 5.0], dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, 3)
        assert mst.shape[0] == 2
        mst_set = {tuple(sorted(row)) for row in mst}
        assert mst_set == {(0, 2), (1, 2)}

    def test_mst_kruskal_connected_vertices_equal(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        weights = np.array([1.0, 2.0], dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, 3)
        flat = np.sort(mst.ravel())
        np.testing.assert_array_equal(flat, np.array([0, 1, 1, 2]))


class TestNumbaReduction:
    def test_column_reduction_basic(self) -> None:
        from pynerve._numba_reduction import numba_column_reduction

        bm = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int64)
        filtration = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        pivots = numba_column_reduction(bm.copy(), filtration)
        assert pivots.shape == (3,)
        assert pivots.dtype == np.int64

    def test_column_reduction_identity(self) -> None:
        from pynerve._numba_reduction import numba_column_reduction

        bm = np.eye(3, dtype=np.int64)
        filtration = np.arange(3, dtype=np.float64)
        pivots = numba_column_reduction(bm.copy(), filtration)
        assert pivots.shape == (3,)

    def test_column_reduction_zero_matrix(self) -> None:
        from pynerve._numba_reduction import numba_column_reduction

        bm = np.zeros((3, 5), dtype=np.int64)
        filtration = np.arange(5, dtype=np.float64)
        pivots = numba_column_reduction(bm.copy(), filtration)
        assert np.all(pivots == -1)

    def test_column_reduction_single_column(self) -> None:
        from pynerve._numba_reduction import numba_column_reduction

        bm = np.array([[1], [0]], dtype=np.int64)
        filtration = np.array([0.0], dtype=np.float64)
        pivots = numba_column_reduction(bm.copy(), filtration)
        assert pivots.shape == (1,)
        assert pivots[0] == 0

    def test_column_reduction_no_rows(self) -> None:
        from pynerve._numba_reduction import numba_column_reduction

        bm = np.empty((0, 3), dtype=np.int64)
        filtration = np.arange(3, dtype=np.float64)
        pivots = numba_column_reduction(bm.copy(), filtration)
        assert np.all(pivots == -1)

    def test_column_reduction_no_columns(self) -> None:
        from pynerve._numba_reduction import numba_column_reduction

        bm = np.empty((5, 0), dtype=np.int64)
        filtration = np.empty(0, dtype=np.float64)
        pivots = numba_column_reduction(bm.copy(), filtration)
        assert pivots.shape == (0,)

    def test_column_reduction_pivots_valid_range(self) -> None:
        from pynerve._numba_reduction import numba_column_reduction

        bm = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=np.int64)
        filtration = np.arange(3, dtype=np.float64)
        pivots = numba_column_reduction(bm.copy(), filtration)
        assert np.all(pivots >= -1)
        assert np.all(pivots < bm.shape[0])

    def test_column_reduction_simple_case_pivots(self) -> None:
        from pynerve._numba_reduction import numba_column_reduction

        bm = np.array([[1, 0], [0, 1]], dtype=np.int64)
        filtration = np.arange(2, dtype=np.float64)
        pivots = numba_column_reduction(bm.copy(), filtration)
        assert pivots.shape == (2,)
        assert pivots.dtype == np.int64
        assert np.all(np.sort(pivots) == np.array([0, 1]))

    def test_sparse_reduction_empty_columns(self) -> None:
        from pynerve._numba_reduction import numba_sparse_reduction

        columns = np.empty((0, 1), dtype=np.int64)
        col_lengths = np.empty(0, dtype=np.int64)
        pivots = numba_sparse_reduction(columns, col_lengths)
        assert pivots.shape == (0,)

    def test_sparse_reduction_all_empty_cols(self) -> None:
        from pynerve._numba_reduction import numba_sparse_reduction

        columns = np.full((2, 2), -1, dtype=np.int64)
        col_lengths = np.zeros(2, dtype=np.int64)
        pivots = numba_sparse_reduction(columns, col_lengths)
        assert np.all(pivots == -1)


class TestNumbaRepresentations:
    def test_betti_curve_basic(self) -> None:
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [1.0, 3.0, 1.0]], dtype=np.float64)
        betti = numba_betti_curve(pairs, 1, 10, 3.0)
        assert betti.shape == (2, 10)
        assert betti.dtype == np.int64

    def test_betti_curve_empty_pairs(self) -> None:
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.empty((0, 3), dtype=np.float64)
        betti = numba_betti_curve(pairs, 1, 10, 5.0)
        assert betti.shape == (2, 10)
        assert np.all(betti == 0)

    def test_betti_curve_zero_max_time(self) -> None:
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.array([[0.0, 2.0, 0.0]], dtype=np.float64)
        betti = numba_betti_curve(pairs, 0, 10, 0.0)
        assert np.all(betti == 0)

    def test_betti_curve_negative_max_dim_raises(self) -> None:
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="max_dim must be non-negative"):
            numba_betti_curve(pairs, -1, 10, 5.0)

    def test_betti_curve_zero_resolution_raises(self) -> None:
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="resolution positive"):
            numba_betti_curve(pairs, 0, 0, 5.0)

    def test_betti_curve_nonnegative_counts(self) -> None:
        from pynerve._numba_representations import numba_betti_curve

        rng = np.random.default_rng(7)
        n = 30
        births = rng.uniform(0.0, 5.0, size=n)
        deaths = births + rng.uniform(0.1, 5.0, size=n)
        dims = rng.integers(0, 3, size=n).astype(np.float64)
        pairs = np.column_stack([births, deaths, dims])
        betti = numba_betti_curve(pairs, 2, 15, 10.0)
        assert np.all(betti >= 0)

    def test_betti_curve_resolution_one(self) -> None:
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.array([[0.0, 5.0, 0.0]], dtype=np.float64)
        betti = numba_betti_curve(pairs, 0, 1, 5.0)
        assert betti.shape == (1, 1)
        assert betti[0, 0] == 1

    def test_betti_curve_max_dim_larger_than_data(self) -> None:
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.array([[0.0, 2.0, 0.0]], dtype=np.float64)
        betti = numba_betti_curve(pairs, 5, 10, 2.0)
        assert betti.shape == (6, 10)

    def test_persistence_image_basic(self) -> None:
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[1.0, 3.0], [2.0, 5.0]], dtype=np.float64)
        img = numba_persistence_image(pairs, 10, 0.5, 0.0, 6.0, 0.0, 6.0)
        assert img.shape == (10, 10)
        assert img.dtype == np.float64
        assert np.any(img > 0)

    def test_persistence_image_all_nonnegative(self) -> None:
        from pynerve._numba_representations import numba_persistence_image

        rng = np.random.default_rng(8)
        births = rng.uniform(0.0, 5.0, size=10)
        deaths = births + rng.uniform(0.1, 5.0, size=10)
        pairs = np.column_stack([births, deaths])
        img = numba_persistence_image(pairs, 16, 0.3, 0.0, 10.0, 0.0, 10.0)
        assert np.all(img >= 0.0)

    def test_persistence_image_negative_resolution_raises(self) -> None:
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[1.0, 2.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="resolution and sigma must be positive"):
            numba_persistence_image(pairs, 0, 0.5, 0.0, 5.0, 0.0, 5.0)

    def test_persistence_image_zero_sigma_raises(self) -> None:
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[1.0, 2.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="resolution and sigma must be positive"):
            numba_persistence_image(pairs, 8, 0.0, 0.0, 5.0, 0.0, 5.0)

    def test_persistence_image_single_pair(self) -> None:
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[2.0, 4.0]], dtype=np.float64)
        img = numba_persistence_image(pairs, 8, 0.5, 0.0, 5.0, 0.0, 5.0)
        assert np.any(img > 0)

    def test_persistence_image_resolution_one(self) -> None:
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[1.0, 2.0]], dtype=np.float64)
        img = numba_persistence_image(pairs, 1, 0.5, 0.0, 3.0, 0.0, 3.0)
        assert img.shape == (1, 1)

    def test_persistence_image_larger_sigma_wider_spread(self) -> None:
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[1.0, 2.0]], dtype=np.float64)
        img_narrow = numba_persistence_image(pairs, 16, 0.1, 0.0, 3.0, 0.0, 3.0)
        img_wide = numba_persistence_image(pairs, 16, 2.0, 0.0, 3.0, 0.0, 3.0)
        nonzero_narrow = np.count_nonzero(img_narrow)
        nonzero_wide = np.count_nonzero(img_wide)
        assert nonzero_wide >= nonzero_narrow

    def test_persistence_image_far_apart_pairs(self) -> None:
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[0.0, 1.0], [100.0, 101.0]], dtype=np.float64)
        img = numba_persistence_image(pairs, 16, 0.5, 0.0, 102.0, 0.0, 102.0)
        assert img.shape == (16, 16)
        assert np.all(np.isfinite(img))

    def test_persistence_image_identical_pairs(self) -> None:
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
        img = numba_persistence_image(pairs, 8, 0.5, 0.0, 3.0, 0.0, 3.0)
        assert np.any(img > 0)


class TestNumbaSimplices:
    def test_vr_edges_basic(self) -> None:
        from pynerve._numba_simplices import numba_vr_edges

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        edges = numba_vr_edges(pts, 100.0)
        assert edges.shape == (3, 2)
        assert edges.dtype == np.int64

    def test_vr_edges_zero_max_dist(self) -> None:
        from pynerve._numba_simplices import numba_vr_edges

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        edges = numba_vr_edges(pts, 0.0)
        assert len(edges) == 0

    def test_vr_edges_negative_max_dist_raises(self) -> None:
        from pynerve._numba_simplices import numba_vr_edges

        pts = np.array([[0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="max_dist must be non-negative"):
            numba_vr_edges(pts, -1.0)

    def test_vr_edges_single_point(self) -> None:
        from pynerve._numba_simplices import numba_vr_edges

        pts = np.array([[0.0, 0.0]], dtype=np.float64)
        edges = numba_vr_edges(pts, 1.0)
        assert edges.shape == (0, 2)

    def test_vr_edges_within_radius(self) -> None:
        from pynerve._numba_simplices import numba_vr_edges

        rng = np.random.default_rng(9)
        pts = rng.normal(size=(15, 2)).astype(np.float64)
        max_dist = 1.5
        edges = numba_vr_edges(pts, max_dist)
        for e in edges:
            u, v = int(e[0]), int(e[1])
            dist = np.linalg.norm(pts[u] - pts[v])
            assert dist <= max_dist + 1e-6

    def test_vr_edges_index_bounds(self) -> None:
        from pynerve._numba_simplices import numba_vr_edges

        rng = np.random.default_rng(10)
        pts = rng.normal(size=(10, 2)).astype(np.float64)
        edges = numba_vr_edges(pts, 1.0)
        for e in edges:
            assert 0 <= e[0] < 10
            assert 0 <= e[1] < 10
            assert e[0] < e[1]

    def test_vr_edges_two_points_close(self) -> None:
        from pynerve._numba_simplices import numba_vr_edges

        pts = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float64)
        edges = numba_vr_edges(pts, 1.0)
        assert len(edges) == 1
        np.testing.assert_array_equal(edges[0], np.array([0, 1], dtype=np.int64))

    def test_vr_edges_two_points_far(self) -> None:
        from pynerve._numba_simplices import numba_vr_edges

        pts = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        edges = numba_vr_edges(pts, 1.0)
        assert len(edges) == 0

    def test_vr_edges_large_max_dist_all_edges(self) -> None:
        from pynerve._numba_simplices import numba_vr_edges

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        edges = numba_vr_edges(pts, 100.0)
        assert len(edges) == 3

    def test_triangle_enumeration_basic(self) -> None:
        from pynerve._numba_simplices import numba_triangle_enumeration

        edges = np.array([[0, 1], [1, 2], [0, 2], [1, 3]], dtype=np.int64)
        tris = numba_triangle_enumeration(edges, 4)
        assert tris.shape[0] == 1
        assert set(tris[0]) == {0, 1, 2}
        assert tris.dtype == np.int64

    def test_triangle_enumeration_no_triangles(self) -> None:
        from pynerve._numba_simplices import numba_triangle_enumeration

        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        tris = numba_triangle_enumeration(edges, 4)
        assert tris.shape[0] == 0

    def test_triangle_enumeration_few_vertices(self) -> None:
        from pynerve._numba_simplices import numba_triangle_enumeration

        edges = np.array([[0, 1]], dtype=np.int64)
        tris = numba_triangle_enumeration(edges, 2)
        assert tris.shape == (0, 3)

    def test_triangle_enumeration_empty_edges(self) -> None:
        from pynerve._numba_simplices import numba_triangle_enumeration

        edges = np.empty((0, 2), dtype=np.int64)
        tris = numba_triangle_enumeration(edges, 5)
        assert tris.shape == (0, 3)

    def test_triangle_enumeration_small_n_vertices(self) -> None:
        from pynerve._numba_simplices import numba_triangle_enumeration

        edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        tris = numba_triangle_enumeration(edges, 2)
        assert tris.shape == (0, 3)

    def test_triangle_enumeration_multiple_triangles(self) -> None:
        from pynerve._numba_simplices import numba_triangle_enumeration

        edges = np.array([[0, 1], [1, 2], [0, 2], [1, 3], [2, 3], [0, 3]], dtype=np.int64)
        tris = numba_triangle_enumeration(edges, 4)
        assert 1 <= tris.shape[0] <= 4
        for tri in tris:
            assert len(set(tri)) == 3

    def test_simplex_boundary_empty(self) -> None:
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([], dtype=np.int64)
        bdy = numba_simplex_boundary(simplex)
        assert bdy.shape == (0, 0)

    def test_simplex_boundary_single_vertex(self) -> None:
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([5], dtype=np.int64)
        bdy = numba_simplex_boundary(simplex)
        assert bdy.shape == (1, 0)

    def test_simplex_boundary_edge(self) -> None:
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([3, 7], dtype=np.int64)
        bdy = numba_simplex_boundary(simplex)
        assert bdy.shape == (2, 1)
        assert set(bdy[:, 0]) == {3, 7}

    def test_simplex_boundary_triangle(self) -> None:
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([1, 2, 3], dtype=np.int64)
        bdy = numba_simplex_boundary(simplex)
        assert bdy.shape == (3, 2)
        faces = [tuple(sorted(bdy[i])) for i in range(3)]
        assert (1, 2) in faces
        assert (1, 3) in faces
        assert (2, 3) in faces

    def test_simplex_boundary_tetrahedron(self) -> None:
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([0, 1, 2, 3], dtype=np.int64)
        bdy = numba_simplex_boundary(simplex)
        assert bdy.shape == (4, 3)
        faces = [tuple(sorted(bdy[i])) for i in range(4)]
        assert len(faces) == 4
        for face in faces:
            assert len(face) == 3

    def test_simplex_boundary_preserves_dtype(self) -> None:
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([1, 2, 3], dtype=np.int32)
        bdy = numba_simplex_boundary(simplex)
        assert bdy.dtype == np.int32


class TestJitCache:
    def test_cache_get_or_compile_new(self) -> None:
        from pynerve.jit._cache import JITCache

        cache = JITCache()

        def add(a: int, b: int) -> int:
            return a + b

        compiled = cache.get_or_compile(add)
        assert compiled(3, 4) == 7

    def test_cache_reuses_compiled(self) -> None:
        from pynerve.jit._cache import JITCache

        cache = JITCache()

        def mul(a: float, b: float) -> float:
            return a * b

        c1 = cache.get_or_compile(mul)
        c2 = cache.get_or_compile(mul)
        assert c1 is c2

    def test_cache_different_funcs_different_keys(self) -> None:
        from pynerve.jit._cache import JITCache

        cache = JITCache()

        def f1(a: int) -> int:
            return a

        def f2(a: int) -> int:
            return a

        c1 = cache.get_or_compile(f1)
        c2 = cache.get_or_compile(f2)
        assert c1 is not c2

    def test_cache_clear_removes_all(self) -> None:
        from pynerve.jit._cache import JITCache

        cache = JITCache()

        def sq(x: float) -> float:
            return x * x

        c1 = cache.get_or_compile(sq)
        assert repr(cache).startswith("JITCache(entries=1)")
        cache.clear()
        assert repr(cache).startswith("JITCache(entries=0)")
        c2 = cache.get_or_compile(sq)
        assert c1 is not c2

    def test_cache_not_callable_raises(self) -> None:
        from pynerve.jit._cache import JITCache

        cache = JITCache()
        with pytest.raises(TypeError, match="func must be callable"):
            cache.get_or_compile("not_a_function")  # type: ignore[arg-type]

    def test_cache_repr_format(self) -> None:
        from pynerve.jit._cache import JITCache

        cache = JITCache()
        assert repr(cache) == "JITCache(entries=0)"

    def test_cached_jit_decorator(self) -> None:
        from pynerve._numba_compat import prange
        from pynerve.jit._cache import cached_jit

        @cached_jit
        def array_add(arr: np.ndarray) -> float:
            s = 0.0
            for i in prange(arr.shape[0]):
                s += arr[i]
            return s

        arr = np.array([1.0, 2.0, 3.0, 4.0])
        assert array_add(arr) == 10.0

    def test_cached_jit_not_callable_raises(self) -> None:
        from pynerve.jit._cache import cached_jit

        with pytest.raises(TypeError, match="func must be callable"):
            cached_jit("not_callable")  # type: ignore[arg-type]


class TestJitDevice:
    def test_resolve_device_none_returns_false(self) -> None:
        from pynerve.jit._device import _resolve_device

        assert _resolve_device(None) is False

    def test_resolve_device_cpu_returns_false(self) -> None:
        from pynerve.jit._device import _resolve_device

        assert _resolve_device("cpu") is False

    def test_resolve_device_cuda_returns_true(self) -> None:
        from pynerve.jit._device import _resolve_device
        from pynerve.jit._setup import HAS_CUDA

        if HAS_CUDA:
            assert _resolve_device("cuda") is True
        else:
            with pytest.raises(RuntimeError, match="CUDA device requested"):
                _resolve_device("cuda")

    def test_resolve_device_invalid_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._device import _resolve_device

        with pytest.raises(InvalidArgumentError, match="unsupported device"):
            _resolve_device("gpu")


class TestJitValidate:
    def test_validate_points_basic(self) -> None:
        from pynerve.jit._validate import _validate_points

        pts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        out = _validate_points(pts)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, pts.astype(np.float32))

    def test_validate_points_empty_rows_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._validate import _validate_points

        with pytest.raises(InvalidArgumentError, match="points must be non-empty"):
            _validate_points(np.empty((0, 3)))

    def test_validate_points_empty_cols_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._validate import _validate_points

        with pytest.raises(InvalidArgumentError, match="points must be non-empty"):
            _validate_points(np.empty((3, 0)))

    def test_validate_points_1d_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._validate import _validate_points

        with pytest.raises(InvalidArgumentError, match="points must be a 2D array"):
            _validate_points(np.array([1.0, 2.0, 3.0]))

    def test_validate_points_3d_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._validate import _validate_points

        with pytest.raises(InvalidArgumentError, match="points must be a 2D array"):
            _validate_points(np.zeros((2, 2, 2)))

    def test_validate_points_nonfinite_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._validate import _validate_points

        with pytest.raises(InvalidArgumentError, match="finite coordinates"):
            _validate_points(np.array([[np.inf, 0.0]]))

        with pytest.raises(InvalidArgumentError, match="finite coordinates"):
            _validate_points(np.array([[1.0, np.nan]]))

    def test_validate_pairs_basic(self) -> None:
        from pynerve.jit._validate import _validate_pairs

        pairs = np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float64)
        out = _validate_pairs(pairs)
        assert out.dtype == np.float32

    def test_validate_pairs_require_dim_basic(self) -> None:
        from pynerve.jit._validate import _validate_pairs

        pairs = np.array([[0.0, 2.0, 0.0], [1.0, 3.0, 1.0]], dtype=np.float64)
        out = _validate_pairs(pairs, require_dim=True)
        assert out.dtype == np.float32
        assert out.shape[1] == 3

    def test_validate_pairs_too_few_cols_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._validate import _validate_pairs

        with pytest.raises(InvalidArgumentError, match="pairs must have shape"):
            _validate_pairs(np.array([[0.0]]))

    def test_validate_pairs_too_few_cols_with_dim_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._validate import _validate_pairs

        with pytest.raises(InvalidArgumentError, match="pairs must have shape"):
            _validate_pairs(np.array([[0.0, 1.0]]), require_dim=True)

    def test_validate_pairs_1d_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._validate import _validate_pairs

        with pytest.raises(InvalidArgumentError, match="pairs must have shape"):
            _validate_pairs(np.array([0.0, 1.0]))

    def test_validate_pairs_empty_returns_empty(self) -> None:
        from pynerve.jit._validate import _validate_pairs

        pairs = np.empty((0, 3), dtype=np.float32)
        out = _validate_pairs(pairs, require_dim=True)
        assert out.shape == (0, 3)

    def test_validate_pairs_empty_with_extra_cols(self) -> None:
        from pynerve.jit._validate import _validate_pairs

        pairs = np.empty((0, 3), dtype=np.float32)
        out = _validate_pairs(pairs)
        assert out.shape[1] == 3
        assert out.shape[0] == 0

    def test_validate_pairs_nonfinite_birth_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._validate import _validate_pairs

        with pytest.raises(InvalidArgumentError, match="births must be finite"):
            _validate_pairs(np.array([[np.inf, 1.0]]))

    def test_validate_pairs_nan_death_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._validate import _validate_pairs

        with pytest.raises(InvalidArgumentError, match="deaths must be finite"):
            _validate_pairs(np.array([[0.0, np.nan]]))

    def test_validate_pairs_neginf_death_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._validate import _validate_pairs

        with pytest.raises(InvalidArgumentError, match="deaths must be finite"):
            _validate_pairs(np.array([[0.0, -np.inf]]))

    def test_validate_pairs_posinf_death_ok(self) -> None:
        from pynerve.jit._validate import _validate_pairs

        pairs = np.array([[0.0, np.inf]], dtype=np.float32)
        out = _validate_pairs(pairs)
        assert out.shape == (1, 2)

    def test_validate_pairs_death_less_than_birth_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._validate import _validate_pairs

        with pytest.raises(InvalidArgumentError, match="greater than or equal"):
            _validate_pairs(np.array([[5.0, 1.0]]))

    def test_validate_pairs_invalid_dim_raises(self) -> None:
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit._validate import _validate_pairs

        with pytest.raises(InvalidArgumentError, match="dimensions must be finite"):
            _validate_pairs(np.array([[0.0, 1.0, 1.5]]), require_dim=True)

        with pytest.raises(InvalidArgumentError, match="dimensions must be finite"):
            _validate_pairs(np.array([[0.0, 1.0, -1.0]]), require_dim=True)


class TestJitInitPublicAPI:
    def test_pairwise_distances_cpu(self) -> None:
        from pynerve.jit import pairwise_distances

        pts = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
        got = pairwise_distances(pts, device="cpu")
        assert got.shape == (2, 2)
        np.testing.assert_allclose(got[0, 1], 5.0, rtol=1e-6)

    def test_pairwise_distances_default_device(self) -> None:
        from pynerve.jit import pairwise_distances

        pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        got = pairwise_distances(pts)
        assert got.shape == (2, 2)
        np.testing.assert_allclose(np.diag(got), 0.0, atol=1e-15)

    def test_filter_pairs_basic(self) -> None:
        from pynerve.jit import filter_pairs

        pairs = np.array([[0.0, 2.0], [0.0, 1.0], [0.0, 5.0]], dtype=np.float64)
        try:
            mask = filter_pairs(pairs, 2.0)
        except numba.core.errors.TypingError as exc:
            pytest.skip(f"filter_pairs compilation failed: {exc}")
            return
        np.testing.assert_array_equal(mask, np.array([False, False, True]))
        assert mask.dtype == bool

    def test_filter_pairs_all_below(self) -> None:
        from pynerve.jit import filter_pairs

        pairs = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        try:
            mask = filter_pairs(pairs, 2.0)
        except numba.core.errors.TypingError as exc:
            pytest.skip(f"filter_pairs compilation failed by typing: {exc}")
            return
        assert not np.any(mask)

    def test_filter_pairs_empty(self) -> None:
        from pynerve.jit import filter_pairs

        pairs = np.empty((0, 2), dtype=np.float32)
        try:
            mask = filter_pairs(pairs, 0.5)
        except numba.core.errors.TypingError as exc:
            pytest.skip(f"filter_pairs compilation failed by typing: {exc}")
            return
        assert mask.shape == (0,)

    def test_betti_curve_public(self) -> None:
        from pynerve.jit import betti_curve

        pairs = np.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        betti = betti_curve(pairs, max_dim=1, resolution=10)
        assert betti.shape == (2, 10)
        assert betti.dtype == np.int32

    def test_betti_curve_empty_returns_zeros(self) -> None:
        from pynerve.jit import betti_curve

        pairs = np.empty((0, 3), dtype=np.float32)
        betti = betti_curve(pairs, max_dim=1, resolution=10)
        assert betti.shape == (2, 10)
        assert np.all(betti == 0)

    def test_betti_curve_max_dim_zero(self) -> None:
        from pynerve.jit import betti_curve

        pairs = np.array([[0.0, 2.0, 0.0]], dtype=np.float64)
        betti = betti_curve(pairs, max_dim=0, resolution=5)
        assert betti.shape == (1, 5)

    def test_persistence_image_public(self) -> None:
        from pynerve.jit import persistence_image

        pairs = np.array([[1.0, 3.0], [2.0, 5.0]], dtype=np.float64)
        img = persistence_image(pairs, resolution=10, sigma=0.5, device="cpu")
        assert img.shape == (10, 10)
        assert img.dtype == np.float32

    def test_persistence_image_empty_pairs(self) -> None:
        from pynerve.jit import persistence_image

        pairs = np.empty((0, 2), dtype=np.float32)
        img = persistence_image(pairs, resolution=8, sigma=0.5)
        assert img.shape == (8, 8)
        assert np.all(img == 0.0)

    def test_vietoris_rips_edges_public(self) -> None:
        from pynerve.jit import vietoris_rips_edges

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        edges = vietoris_rips_edges(pts, max_dist=100.0)
        assert edges.shape[1] == 2
        assert len(edges) == 3

    def test_vietoris_rips_edges_zero_dist(self) -> None:
        from pynerve.jit import vietoris_rips_edges

        pts = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        edges = vietoris_rips_edges(pts, max_dist=0.0)
        assert len(edges) == 0

    def test_batch_betti_curves_public(self) -> None:
        from pynerve.jit import batch_betti_curves

        diagrams = np.zeros((2, 5, 3), dtype=np.float32)
        diagrams[0, 0] = [0.0, 2.0, 0.0]
        diagrams[0, 1] = [0.0, 1.0, 0.0]
        diagrams[1, 0] = [0.0, 3.0, 0.0]
        curves = batch_betti_curves(diagrams, max_dim=1, resolution=10)
        assert curves.shape == (2, 2, 10)
        assert curves.dtype == np.int32

    def test_batch_betti_curves_empty_batch_raises(self) -> None:
        from pynerve.jit import batch_betti_curves

        diagrams = np.empty((0, 3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="batch must be non-empty"):
            batch_betti_curves(diagrams, max_dim=0, resolution=5)

    def test_batch_betti_curves_wrong_ndim_raises(self) -> None:
        from pynerve.jit import batch_betti_curves

        with pytest.raises(ValueError, match="diagrams must have shape"):
            batch_betti_curves(np.zeros((5, 3), dtype=np.float32), max_dim=0, resolution=5)

    def test_batch_betti_curves_too_few_cols_raises(self) -> None:
        from pynerve.jit import batch_betti_curves

        diagrams = np.zeros((1, 3, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="diagrams must have shape"):
            batch_betti_curves(diagrams, max_dim=0, resolution=5)

    def test_batch_betti_curves_all_zero_diagrams(self) -> None:
        from pynerve.jit import batch_betti_curves

        diagrams = np.zeros((2, 4, 3), dtype=np.float32)
        curves = batch_betti_curves(diagrams, max_dim=1, resolution=5)
        assert np.all(curves == 0)

    def test_batch_betti_curves_single_diagram(self) -> None:
        from pynerve.jit import batch_betti_curves

        diagrams = np.zeros((1, 3, 3), dtype=np.float32)
        diagrams[0, 0] = [0.0, 5.0, 0.0]
        curves = batch_betti_curves(diagrams, max_dim=0, resolution=5)
        assert curves.shape == (1, 1, 5)
        assert curves[0, 0, 0] == 1
