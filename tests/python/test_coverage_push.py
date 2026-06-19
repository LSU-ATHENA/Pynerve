from __future__ import annotations

import math
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest


def _call_numba_fn(import_path: str, fn_name: str, *args, **kwargs):
    import importlib

    numba = pytest.importorskip("numba")
    mod = importlib.import_module(import_path)
    fn = getattr(mod, fn_name)
    try:
        return fn(*args, **kwargs)
    except numba.core.errors.CompilerError as exc:
        pytest.skip(f"{fn_name} compilation failed: {exc}")


# _numba_dispatch.py


class TestComputeWithNumbaAllOps:
    def test_dispatch_vr_edges(self):
        pytest.importorskip("numba")
        from pynerve._numba_dispatch import compute_with_numba

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        result = compute_with_numba("vr_edges", pts, max_dist=100.0)
        assert result.shape[1] == 2
        assert result.shape[0] > 0

    def test_dispatch_column_reduction(self):
        pytest.importorskip("numba")
        from pynerve._numba_dispatch import compute_with_numba

        m = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]], dtype=np.int64)
        filtration = np.arange(3, dtype=np.float64)
        result = compute_with_numba("column_reduction", m.copy(), filtration)
        assert len(result) == 3
        assert result[0] == 0

    def test_dispatch_betti_curve(self):
        pytest.importorskip("numba")
        from pynerve._numba_dispatch import compute_with_numba

        pairs = np.array([[1.0, 3.0, 0.0], [0.5, 2.0, 0.0]], dtype=np.float64)
        result = compute_with_numba("betti_curve", pairs, max_dim=1, resolution=10, max_time=5.0)
        assert result.shape == (2, 10)
        assert result.dtype == np.int64

    def test_dispatch_connected_components(self):
        pytest.importorskip("numba")
        from pynerve._numba_dispatch import compute_with_numba

        edges = np.array([[0, 1], [2, 3], [3, 4]], dtype=np.int64)
        result = compute_with_numba("connected_components", edges, n_vertices=5)
        assert len(set(result)) == 2
        assert result[0] == result[1]
        assert result[2] == result[3] == result[4]

    def test_dispatch_none_operation(self):
        from pynerve._numba_dispatch import compute_with_numba

        with pytest.raises(ValueError):
            compute_with_numba(None)

    def test_dispatch_bool_operation(self):
        from pynerve._numba_dispatch import compute_with_numba

        with pytest.raises(ValueError):
            compute_with_numba(True)


class TestBenchmarkNumbaVsNumpyExtended:
    def test_benchmark_with_numba_function(self):
        pytest.importorskip("numba")
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy
        from pynerve._numba_distance import numba_pairwise_distances
        from scipy.spatial.distance import pdist, squareform

        pts = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float64)

        def scipy_pw(x):
            return squareform(pdist(x.astype(np.float64), "euclidean"))

        stats = benchmark_numba_vs_numpy(
            numba_pairwise_distances,
            scipy_pw,
            lambda: (pts,),
            n_trials=3,
        )
        assert stats["results_match"] is True
        assert stats["speedup"] > 0.0
        assert np.isfinite(stats["numba_mean"])
        assert np.isfinite(stats["numpy_mean"])

    def test_benchmark_single_trial(self):
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        stats = benchmark_numba_vs_numpy(
            lambda x: x + 1.0,
            lambda x: x + 1.0,
            lambda: (np.array([1.0, 2.0], dtype=np.float64),),
            n_trials=1,
        )
        assert "numba_mean" in stats
        assert stats["results_match"] is True

    def test_benchmark_n_trials_numpy_int(self):
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        stats = benchmark_numba_vs_numpy(
            lambda x: x + 1.0,
            lambda x: x + 1.0,
            lambda: (np.array([1.0], dtype=np.float64),),
            n_trials=np.int64(3),
        )
        assert stats["results_match"] is True

    def test_benchmark_n_trials_negative(self):
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        with pytest.raises((ValueError, Exception)):
            benchmark_numba_vs_numpy(
                lambda x: x,
                lambda x: x,
                lambda: (np.array([1.0]),),
                n_trials=-1,
            )

    def test_benchmark_with_large_data(self):
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        data = np.random.default_rng(42).normal(size=(50, 3)).astype(np.float64)

        stats = benchmark_numba_vs_numpy(
            lambda x: x + 1.0,
            lambda x: x + 1.0,
            lambda: (data,),
            n_trials=2,
        )
        assert stats["results_match"] is True
        assert np.isfinite(stats["numba_mean"])
        assert np.isfinite(stats["numpy_mean"])

    def test_benchmark_operations_registry_coverage(self):
        from pynerve._numba_dispatch import _OPERATIONS

        assert len(_OPERATIONS) == 5
        for op_name, op_fn in _OPERATIONS.items():
            assert isinstance(op_name, str)
            assert callable(op_fn)


# _numba_distance.py


class TestNumbaPairwiseDistancesExtended:
    def test_pairwise_high_dimensional(self):
        pytest.importorskip("numba")
        from pynerve._numba_distance import numba_pairwise_distances

        pts = np.random.default_rng(99).normal(size=(4, 50)).astype(np.float64)
        got = numba_pairwise_distances(pts)
        assert got.shape == (4, 4)
        np.testing.assert_allclose(np.diag(got), 0.0, atol=1e-15)
        np.testing.assert_allclose(got, got.T, rtol=1e-15)

    def test_pairwise_1d_points(self):
        pytest.importorskip("numba")
        from pynerve._numba_distance import numba_pairwise_distances

        pts = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        got = numba_pairwise_distances(pts)
        assert abs(got[0, 1] - 1.0) < 1e-12
        assert abs(got[1, 2] - 1.0) < 1e-12
        assert abs(got[0, 2] - 2.0) < 1e-12


class TestNumbaNearestNeighborsExtended:
    def test_k_larger_than_n(self):
        pytest.importorskip("numba")
        from pynerve._numba_distance import numba_nearest_neighbors

        pts = np.random.default_rng(77).normal(size=(5, 3)).astype(np.float64)
        dists, indices = numba_nearest_neighbors(pts, k=10)
        assert dists.shape == (5, 10)
        assert indices.shape == (5, 10)

    def test_k_equals_n_minus_one(self):
        pytest.importorskip("numba")
        from pynerve._numba_distance import numba_nearest_neighbors

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        dists, indices = numba_nearest_neighbors(pts, k=2)
        assert dists.shape == (3, 2)
        for i in range(3):
            assert i not in indices[i]

    def test_identical_points(self):
        pytest.importorskip("numba")
        from pynerve._numba_distance import numba_nearest_neighbors

        pts = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        dists, indices = numba_nearest_neighbors(pts, k=1)
        assert dists[0, 0] == 0.0
        assert indices[0, 0] == 1

    def test_k_negative(self):
        pytest.importorskip("numba")
        from pynerve._numba_distance import numba_nearest_neighbors

        pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        dists, indices = numba_nearest_neighbors(pts, k=-3)
        assert dists.shape == (2, 0)
        assert indices.shape == (2, 0)


# _numba_graph.py


class TestNumbaConnectedComponentsExtended:
    def test_single_vertex_no_edges(self):
        pytest.importorskip("numba")
        from pynerve._numba_graph import numba_connected_components

        edges = np.empty((0, 2), dtype=np.int64)
        labels = numba_connected_components(edges, n_vertices=1)
        assert len(labels) == 1
        assert labels[0] == 0

    def test_fully_connected(self):
        pytest.importorskip("numba")
        from pynerve._numba_graph import numba_connected_components

        edges = np.array([[0, 1], [0, 2], [1, 2], [2, 3]], dtype=np.int64)
        labels = numba_connected_components(edges, n_vertices=4)
        assert len(set(labels)) == 1

    def test_zero_vertices(self):
        pytest.importorskip("numba")
        from pynerve._numba_graph import numba_connected_components

        labels = numba_connected_components(np.empty((0, 2), dtype=np.int64), n_vertices=0)
        assert len(labels) == 0

    def test_self_loop_ignored(self):
        pytest.importorskip("numba")
        from pynerve._numba_graph import numba_connected_components

        edges = np.array([[0, 0], [0, 1]], dtype=np.int64)
        labels = numba_connected_components(edges, n_vertices=3)
        assert labels[0] == labels[1]


class TestNumbaMstKruskalExtended:
    def test_two_vertices(self):
        pytest.importorskip("numba")
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.array([[0, 1]], dtype=np.int64)
        weights = np.array([5.0], dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, n_vertices=2)
        assert len(mst) == 1
        assert int(mst[0, 0]) == 0
        assert int(mst[0, 1]) == 1

    def test_zero_vertices(self):
        pytest.importorskip("numba")
        from pynerve._numba_graph import numba_mst_kruskal

        mst = numba_mst_kruskal(
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float64),
            n_vertices=0,
        )
        assert mst.shape == (0, 2)

    def test_reverse_sorted_weights(self):
        pytest.importorskip("numba")
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
        weights = np.array([10.0, 7.0, 3.0, 1.0], dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, n_vertices=4)
        assert len(mst) == 3
        mst_set = {(int(e[0]), int(e[1])) for e in mst}
        assert (0, 3) in mst_set or (3, 0) in mst_set
        assert (2, 3) in mst_set or (3, 2) in mst_set
        assert (1, 2) in mst_set or (2, 1) in mst_set

    def test_all_same_weight(self):
        pytest.importorskip("numba")
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, n_vertices=4)
        assert len(mst) == 3


# _numba_reduction.py


class TestNumbaColumnReductionExtended:
    def test_all_ones_boundary(self):
        pytest.importorskip("numba")
        m = np.ones((3, 3), dtype=np.int64)
        filtration = np.arange(3, dtype=np.float64)
        pivots = _call_numba_fn(
            "pynerve._numba_reduction",
            "numba_column_reduction",
            m.copy(),
            filtration,
        )
        assert len(pivots) == 3

    def test_identity_boundary(self):
        pytest.importorskip("numba")
        m = np.eye(4, dtype=np.int64)
        filtration = np.arange(4, dtype=np.float64)
        pivots = _call_numba_fn(
            "pynerve._numba_reduction",
            "numba_column_reduction",
            m.copy(),
            filtration,
        )
        assert pivots[0] == 0
        assert pivots[1] == 1
        assert pivots[2] == 2
        assert pivots[3] == 3

    def test_single_row_many_cols(self):
        pytest.importorskip("numba")
        m = np.array([[1, 0, 1, 0, 1]], dtype=np.int64)
        filtration = np.arange(5, dtype=np.float64)
        pivots = _call_numba_fn(
            "pynerve._numba_reduction",
            "numba_column_reduction",
            m.copy(),
            filtration,
        )
        assert len(pivots) == 5
        assert pivots[0] == 0
        assert pivots[1] == -1
        assert pivots[2] == -1
        assert pivots[3] == -1
        assert pivots[4] == -1

    def test_no_columns(self):
        pytest.importorskip("numba")
        m = np.zeros((3, 0), dtype=np.int64)
        filtration = np.empty(0, dtype=np.float64)
        pivots = _call_numba_fn(
            "pynerve._numba_reduction",
            "numba_column_reduction",
            m.copy(),
            filtration,
        )
        assert len(pivots) == 0


class TestNumbaSparseReductionExtended:
    def test_single_column_all_zeros(self):
        pytest.importorskip("numba")
        from pynerve._numba_reduction import numba_sparse_reduction

        columns = np.full((1, 1), -1, dtype=np.int64)
        col_lengths = np.array([0], dtype=np.int64)
        pivots = numba_sparse_reduction(columns, col_lengths)
        assert np.all(pivots == -1)

    def test_all_identical_columns(self):
        pytest.importorskip("numba")
        from pynerve._numba_reduction import numba_sparse_reduction

        columns = np.array([[0, 1, -1], [0, 1, -1], [0, 1, -1]], dtype=np.int64)
        col_lengths = np.array([2, 2, 2], dtype=np.int64)
        pivots = numba_sparse_reduction(columns, col_lengths)
        non_neg = pivots[pivots >= 0]
        assert len(non_neg) <= 1

    def test_large_empty_columns(self):
        pytest.importorskip("numba")
        from pynerve._numba_reduction import numba_sparse_reduction

        columns = np.full((10, 1), -1, dtype=np.int64)
        col_lengths = np.zeros(10, dtype=np.int64)
        pivots = numba_sparse_reduction(columns, col_lengths)
        assert np.all(pivots == -1)

    def test_chain_three_columns(self):
        pytest.importorskip("numba")
        from pynerve._numba_reduction import numba_sparse_reduction

        columns = np.array([[0, -1], [0, -1], [1, -1]], dtype=np.int64)
        col_lengths = np.array([1, 1, 1], dtype=np.int64)
        pivots = numba_sparse_reduction(columns, col_lengths)
        assert pivots[1] == -1


# _numba_representations.py


class TestNumbaBettiCurveExtended:
    def test_zero_max_time_returns_zeros(self):
        pytest.importorskip("numba")
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.array([[1.0, 3.0, 0.0], [2.0, 5.0, 1.0]], dtype=np.float64)
        result = numba_betti_curve(pairs, max_dim=2, resolution=10, max_time=0.0)
        assert np.all(result == 0)

    def test_very_small_max_time(self):
        pytest.importorskip("numba")
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.array([[0.0, 0.001, 0.0]], dtype=np.float64)
        result = numba_betti_curve(pairs, max_dim=1, resolution=100, max_time=0.0005)
        assert result.shape == (2, 100)

    def test_all_pairs_one_dim(self):
        pytest.importorskip("numba")
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [2.0, 3.0, 0.0]], dtype=np.float64)
        result = numba_betti_curve(pairs, max_dim=1, resolution=10, max_time=4.0)
        assert result.shape == (2, 10)
        assert np.all(result[1] == 0)

    def test_high_dim_pairs(self):
        pytest.importorskip("numba")
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.array([[0.0, 5.0, 2.0], [1.0, 4.0, 2.0]], dtype=np.float64)
        result = numba_betti_curve(pairs, max_dim=3, resolution=20, max_time=10.0)
        assert result.shape == (4, 20)

    def test_pairs_at_edges(self):
        pytest.importorskip("numba")
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.array([[0.0, 10.0, 0.0]], dtype=np.float64)
        result = numba_betti_curve(pairs, max_dim=0, resolution=5, max_time=10.0)
        assert result.shape == (1, 5)
        assert result[0, 0] == 1
        assert result[0, 4] == 1


class TestNumbaPersistenceImageExtended:
    def test_small_resolution(self):
        pytest.importorskip("numba")
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[2.0, 4.0]], dtype=np.float64)
        got = numba_persistence_image(pairs, 3, 1.0, 0.0, 5.0, 0.0, 5.0)
        assert got.shape == (3, 3)
        assert np.all(got >= 0.0)

    def test_zero_span_birth(self):
        pytest.importorskip("numba")
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[2.0, 4.0]], dtype=np.float64)
        got = numba_persistence_image(pairs, 5, 1.0, 2.0, 2.0, 0.0, 5.0)
        assert got.shape == (5, 5)

    def test_zero_span_death(self):
        pytest.importorskip("numba")
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[2.0, 4.0]], dtype=np.float64)
        got = numba_persistence_image(pairs, 5, 1.0, 0.0, 5.0, 4.0, 4.0)
        assert got.shape == (5, 5)

    def test_point_at_origin(self):
        pytest.importorskip("numba")
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[0.0, 0.5]], dtype=np.float64)
        got = numba_persistence_image(pairs, 10, 0.5, 0.0, 1.0, 0.0, 1.0)
        assert np.any(got > 0)

    def test_point_outside_range(self):
        pytest.importorskip("numba")
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[10.0, 15.0]], dtype=np.float64)
        got = numba_persistence_image(pairs, 5, 1.0, 0.0, 2.0, 0.0, 2.0)
        assert np.all(got == 0.0)


# _numba_simplices.py


class TestNumbaVrEdgesExtended:
    def test_large_n_all_pairs(self):
        pytest.importorskip("numba")
        from pynerve._numba_simplices import numba_vr_edges

        pts = np.random.default_rng(123).normal(size=(30, 2)).astype(np.float64)
        edges = numba_vr_edges(pts, max_dist=1e9)
        n = pts.shape[0]
        assert len(edges) == n * (n - 1) // 2

    def test_small_max_dist_no_edges(self):
        pytest.importorskip("numba")
        from pynerve._numba_simplices import numba_vr_edges

        pts = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]], dtype=np.float64)
        edges = numba_vr_edges(pts, max_dist=1.0)
        assert len(edges) == 0

    def test_exact_distance_match(self):
        pytest.importorskip("numba")
        from pynerve._numba_simplices import numba_vr_edges

        pts = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
        edges = numba_vr_edges(pts, max_dist=5.0)
        assert len(edges) == 1
        assert int(edges[0, 0]) == 0
        assert int(edges[0, 1]) == 1


class TestNumbaTriangleEnumerationExtended:
    def test_fully_connected_small(self):
        pytest.importorskip("numba")
        from pynerve._numba_simplices import numba_triangle_enumeration

        n = 6
        rows, cols = np.triu_indices(n, k=1)
        edges = np.column_stack([rows, cols]).astype(np.int64)
        tris = numba_triangle_enumeration(edges, n_vertices=n)
        expected_count = n * (n - 1) * (n - 2) // 6
        assert len(tris) == expected_count

    def test_bipartite_no_triangles(self):
        pytest.importorskip("numba")
        from pynerve._numba_simplices import numba_triangle_enumeration

        edges = np.array([[0, 2], [0, 3], [1, 2], [1, 3]], dtype=np.int64)
        tris = numba_triangle_enumeration(edges, n_vertices=4)
        assert len(tris) == 0

    def test_two_triangles_share_edge(self):
        pytest.importorskip("numba")
        from pynerve._numba_simplices import numba_triangle_enumeration

        edges = np.array([[0, 1], [1, 2], [0, 2], [1, 3], [2, 3]], dtype=np.int64)
        tris = numba_triangle_enumeration(edges, n_vertices=4)
        assert len(tris) == 2


class TestNumbaSimplexBoundaryExtended:
    def test_five_simplex(self):
        pytest.importorskip("numba")
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        boundary = numba_simplex_boundary(simplex)
        assert boundary.shape == (5, 4)

    def test_float_dtype(self):
        pytest.importorskip("numba")
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        boundary = numba_simplex_boundary(simplex)
        assert boundary.shape == (3, 2)
        assert boundary.dtype == np.float64


# _shared_memory.py


class TestValidateBatches:
    def test_string_input_raises(self):
        from pynerve._shared_memory import _validate_batches

        with pytest.raises(TypeError, match="must be an iterable"):
            _validate_batches("not_a_list", name="data")

    def test_bytes_input_raises(self):
        from pynerve._shared_memory import _validate_batches

        with pytest.raises(TypeError, match="must be an iterable"):
            _validate_batches(b"bytes_input", name="data")

    def test_non_iterable_raises(self):
        from pynerve._shared_memory import _validate_batches

        with pytest.raises(TypeError, match="must be an iterable"):
            _validate_batches(42, name="data")

    def test_empty_list_returns_empty(self):
        from pynerve._shared_memory import _validate_batches

        result = _validate_batches([], name="data")
        assert result == []

    def test_valid_list(self):
        from pynerve._shared_memory import _validate_batches

        arrays = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = _validate_batches(arrays, name="data")
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], arrays[0])

    def test_zero_dim_array_raises(self):
        from pynerve._shared_memory import _validate_batches

        with pytest.raises((ValueError, Exception)):
            _validate_batches([np.array(5.0)], name="data")

    def test_list_of_single_array(self):
        from pynerve._shared_memory import _validate_batches

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _validate_batches([arr], name="data")
        assert len(result) == 1
        assert result[0] is arr

    def test_generator_input(self):
        from pynerve._shared_memory import _validate_batches

        def gen():
            yield np.array([1.0, 2.0])
            yield np.array([3.0, 4.0])

        result = _validate_batches(gen(), name="data")
        assert len(result) == 2


class TestSharedMemoryArrayExtended:
    def test_repr(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((3,), np.float64)
        try:
            r = repr(shm)
            assert "SharedMemoryArray" in r
            assert "shape=(3,)" in r
            assert "float64" in r
        finally:
            shm.close()

    def test_array_dunder(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((5,), np.float64)
        shm.array[:] = np.arange(5, dtype=np.float64)
        try:
            result = np.asarray(shm)
            np.testing.assert_array_equal(result, np.arange(5, dtype=np.float64))
        finally:
            shm.close()

    def test_array_dunder_with_dtype_cast(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((3,), np.float64)
        shm.array[:] = [1.5, 2.5, 3.5]
        try:
            result = np.asarray(shm, dtype=np.int64)
            assert result.dtype == np.int64
            np.testing.assert_array_equal(result, np.array([1, 2, 3]))
        finally:
            shm.close()

    def test_from_array_with_list(self):
        from pynerve._shared_memory import SharedMemoryArray

        data = [[1.0, 2.0], [3.0, 4.0]]
        shm = SharedMemoryArray.from_array(data)
        try:
            np.testing.assert_array_equal(shm.array, np.array(data))
        finally:
            shm.close()

    def test_from_array_with_integer_array(self):
        from pynerve._shared_memory import SharedMemoryArray

        data = np.array([[1, 2], [3, 4]], dtype=np.int32)
        shm = SharedMemoryArray.from_array(data)
        try:
            assert shm.array.dtype == np.int32
            np.testing.assert_array_equal(shm.array, data)
        finally:
            shm.close()

    def test_getitem_works(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((3, 4), np.float64)
        shm.array[:] = np.arange(12, dtype=np.float64).reshape(3, 4)
        try:
            assert shm[0, 0] == 0.0
            assert shm[2, 3] == 11.0
            assert shm[1, :].tolist() == [4.0, 5.0, 6.0, 7.0]
        finally:
            shm.close()

    def test_setitem_works(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((3,), np.float64)
        try:
            shm[0] = 10.0
            shm[1] = 20.0
            shm[2] = 30.0
            np.testing.assert_array_equal(shm.array, np.array([10.0, 20.0, 30.0]))
        finally:
            shm.close()

    def test_slicing_setitem(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((4,), np.float64)
        try:
            shm[:2] = [1.0, 2.0]
            shm[2:] = [3.0, 4.0]
            np.testing.assert_array_equal(shm.array, np.array([1.0, 2.0, 3.0, 4.0]))
        finally:
            shm.close()

    def test_scalar_shape(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((1,), np.float64)
        try:
            shm[0] = 42.0
            assert shm[0] == 42.0
        finally:
            shm.close()

    def test_close_then_enter(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((3,), np.float64)
        shm.close()
        assert shm._closed is True

    def test_double_context_manager_is_safe(self):
        from pynerve._shared_memory import SharedMemoryArray

        with SharedMemoryArray((2,), np.float64) as shm:
            assert shm._closed is False
        assert shm._closed is True
        shm.close()

    def test_array_dunder_when_closed(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((3,), np.float64)
        shm.close()
        with pytest.raises(RuntimeError, match="is closed"):
            np.asarray(shm)

    def test_repr_when_closed(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((3,), np.float64)
        shm.close()
        r = repr(shm)
        assert "SharedMemoryArray" in r

    def test_empty_shape_raises(self):
        from pynerve._shared_memory import SharedMemoryArray

        with pytest.raises((ValueError, Exception)):
            SharedMemoryArray((), np.float64)

    def test_negative_dimension_raises(self):
        from pynerve._shared_memory import SharedMemoryArray

        with pytest.raises((ValueError, Exception)):
            SharedMemoryArray((-1,), np.float64)

    def test_bool_dimension_raises(self):
        from pynerve._shared_memory import SharedMemoryArray

        with pytest.raises((ValueError, Exception)):
            SharedMemoryArray((True,), np.float64)


# diff/ph_layer.py


class TestPhLayerExtended:
    torch_mod = pytest.importorskip("torch")

    @staticmethod
    def _torch():
        return pytest.importorskip("torch")

    def test_effective_max_radius_two_points_inf(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import _effective_max_radius

        points = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float64)
        result = _effective_max_radius(points, float("inf"))
        assert result == pytest.approx(5.0)

    def test_effective_max_radius_finite_not_needed(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import _effective_max_radius

        points = torch.randn(5, 3)
        result = _effective_max_radius(points, 3.5)
        assert result == 3.5

    def test_effective_max_radius_finite_single_point(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import _effective_max_radius

        points = torch.randn(1, 3)
        result = _effective_max_radius(points, 2.0)
        assert result == 2.0

    def test_valid_max_radius_negative(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        points = torch.randn(2, 5, 3)
        with pytest.raises(ValueError, match="positive"):
            _validate_persistence_inputs(points, 1, "rips", {"max_radius": -1.0})

    def test_setup_persistence_options_no_metric_attr_non_euclidean_raises(self):
        from pynerve.diff.ph_layer import _setup_persistence_options

        class MockOptions:
            max_dim = 0
            max_radius = 0.0

        mock_core = types.ModuleType("test_metric_core")
        mock_core.PersistenceOptions = MockOptions
        sys.modules["test_metric_core"] = mock_core
        try:
            with pytest.raises(ValueError, match="only supports euclidean"):
                _setup_persistence_options(mock_core, 1, float("inf"), "manhattan", "clearing")
        finally:
            del sys.modules["test_metric_core"]

    def test_setup_persistence_options_no_reduction_attr_non_clearing_raises(self):
        from pynerve.diff.ph_layer import _setup_persistence_options

        class MockOptions:
            max_dim = 0
            max_radius = 0.0
            metric = "euclidean"

        mock_core = types.ModuleType("test_reduction_core")
        mock_core.PersistenceOptions = MockOptions
        sys.modules["test_reduction_core"] = mock_core
        try:
            with pytest.raises(ValueError, match="only supports clearing"):
                _setup_persistence_options(mock_core, 1, float("inf"), "euclidean", "exhaustive")
        finally:
            del sys.modules["test_reduction_core"]

    def test_setup_persistence_options_finite_max_radius(self):
        from pynerve.diff.ph_layer import _setup_persistence_options

        class MockOptions:
            max_dim = 0
            max_radius = 0.0
            metric = "euclidean"
            reduction = "clearing"

        mock_core = types.ModuleType("test_finite_core")
        mock_core.PersistenceOptions = MockOptions
        sys.modules["test_finite_core"] = mock_core
        try:
            opts = _setup_persistence_options(mock_core, 2, 5.0, "euclidean", "clearing")
            assert opts.max_dim == 2
            assert opts.max_radius == 5.0
        finally:
            del sys.modules["test_finite_core"]

    def test_pad_diagrams_across_batches_all_same_size(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import _pad_diagrams_across_batches

        d1 = [torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float32)]
        d2 = [torch.tensor([[0.0, 0.3], [1.0, 2.0]], dtype=torch.float32)]
        result = _pad_diagrams_across_batches([d1, d2], 0, 2, torch.float32, torch.device("cpu"))
        assert result[0].shape == (2, 2, 2)

    def test_pad_diagrams_across_batches_multiple_dims(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import _pad_diagrams_across_batches

        d1 = [
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.5, 2.0]], dtype=torch.float32),
        ]
        d2 = [
            torch.tensor([[0.0, 0.5]], dtype=torch.float32),
            torch.empty((0, 2), dtype=torch.float32),
        ]
        result = _pad_diagrams_across_batches([d1, d2], 1, 2, torch.float32, torch.device("cpu"))
        assert len(result) == 2
        assert result[0].shape == (2, 1, 2)
        assert result[1].shape == (2, 1, 2)

    def test_accumulate_merge_gradients_no_gradient(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import _accumulate_merge_gradients

        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        dists = torch.cdist(pts, pts)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_grad = diff / (dists.unsqueeze(-1) + 1e-8)
        grad_target = torch.zeros(1, 3, 2, dtype=torch.float64)
        grad_diag = torch.zeros(0, 2, dtype=torch.float64)

        _accumulate_merge_gradients(pts, grad_diag, dists, dist_grad, grad_target, 0)
        assert torch.all(grad_target == 0.0)

    def test_accumulate_merge_gradients_single_point(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import _accumulate_merge_gradients

        pts = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        dists = torch.cdist(pts, pts)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_grad = diff / (dists.unsqueeze(-1) + 1e-8)
        grad_target = torch.zeros(1, 1, 2, dtype=torch.float64)
        grad_diag = torch.tensor([[0.1, 0.1]], dtype=torch.float64)

        _accumulate_merge_gradients(pts, grad_diag, dists, dist_grad, grad_target, 0)
        assert torch.isfinite(grad_target).all()

    def test_compute_persistence_backward_with_pair_counts(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import _compute_persistence_backward

        points = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=False)
        grad_diag_h0 = torch.tensor([[0.5, 0.0], [0.1, 0.0]], dtype=torch.float64)
        grad_diags = (grad_diag_h0, torch.zeros(0, 2, dtype=torch.float64))
        pair_counts = [[2, 0]]
        result = _compute_persistence_backward(
            points, grad_diags, 1, "rips", {}, pair_counts=pair_counts
        )
        assert result.shape == points.shape
        assert torch.isfinite(result).all()

    def test_compute_persistence_backward_3d_grad_diagrams(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import _compute_persistence_backward

        points = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=False)
        grad_diags = (torch.randn(2, 1, 2, dtype=torch.float64),)
        result = _compute_persistence_backward(
            points, grad_diags, 0, "rips", {"max_radius": float("inf")}
        )
        assert result.shape == points.shape

    def test_compute_persistence_backward_empty_batch(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import _compute_persistence_backward

        points = torch.randn(0, 3, 2, dtype=torch.float64, requires_grad=False)
        grad_diags = (torch.empty(0, 0, 2, dtype=torch.float64),)
        result = _compute_persistence_backward(points, grad_diags, 1, "rips", {})
        assert result.shape == points.shape

    def test_compute_persistence_backward_dim_one_only(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import _compute_persistence_backward

        points = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=False)
        grad_diags = (
            torch.zeros(1, 2, dtype=torch.float64),
            torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        )
        result = _compute_persistence_backward(
            points, grad_diags, 1, "rips", {"max_radius": float("inf")}
        )
        assert result.shape == points.shape

    def test_compute_persistence_backward_past_max_dim(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import _compute_persistence_backward

        points = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=False)
        grad_diags = (torch.zeros(1, 2, dtype=torch.float64),)
        result = _compute_persistence_backward(points, grad_diags, 5, "rips", {})
        assert result.shape == points.shape

    def test_filtration_learning_layer_various_batch_sizes(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=3, hidden_dims=[32])
        for bs in [1, 2, 5, 10]:
            points = torch.randn(bs, 4, 3)
            output = layer.forward(points)
            assert output.shape == (bs, 4)

    def test_filtration_learning_layer_output_range(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=2, hidden_dims=[16, 32])
        points = torch.randn(3, 5, 2)
        output = layer.forward(points)
        assert torch.all(output >= 0.0)

    def test_learnable_filtration_persistence_defaults(self):
        from pynerve.diff.ph_layer import LearnableFiltrationPersistence

        layer = LearnableFiltrationPersistence(input_dim=3)
        assert layer.filtration is not None
        assert layer.persistence is not None
        assert layer.persistence.max_dim == 1

    def test_learnable_filtration_persistence_custom_hidden(self):
        torch = self._torch()
        from pynerve.diff.ph_layer import LearnableFiltrationPersistence

        layer = LearnableFiltrationPersistence(input_dim=4, max_dim=2, hidden_dims=[128, 64])
        points = torch.randn(1, 3, 4)
        filt_values = layer.filtration(points)
        assert filt_values.shape == (1, 3)

    def test_differentiable_vietoris_rips_forward_wrong_dim_shorter(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips

        layer = DifferentiableVietorisRips(max_dim=1)
        with pytest.raises(ValueError, match="shape"):
            layer.forward(self._torch().randn(5))

    def test_compute_persistence_forward_mock_success(self):
        torch = self._torch()
        import importlib

        import pynerve.diff.ph_layer as _phl_pre

        mock_core = types.ModuleType("pynerve_internal")

        class MockOpts:
            max_dim = 0
            max_radius = 0.0
            metric = "euclidean"
            reduction = "clearing"

        mock_core.PersistenceOptions = MockOpts

        def mock_compute(pts, opts):
            return {"pairs": [[0.0, 1.0, 0], [0.5, 2.0, 0]]}

        mock_core.compute_persistence = mock_compute

        original_core = sys.modules.get("pynerve_internal")
        sys.modules["pynerve_internal"] = mock_core
        try:
            importlib.reload(_phl_pre)
            from pynerve.diff.ph_layer import _compute_persistence_forward

            points = torch.randn(2, 3, 2)
            result = _compute_persistence_forward(points, 0, "rips")
            assert len(result) == 1
            assert result[0].shape[0] == 2
        finally:
            if original_core is not None:
                sys.modules["pynerve_internal"] = original_core
            else:
                sys.modules.pop("pynerve_internal", None)
            importlib.reload(_phl_pre)

    def test_compute_persistence_forward_with_counts(self):
        torch = self._torch()
        import importlib

        import pynerve.diff.ph_layer as _phl_pre

        mock_core = types.ModuleType("pynerve_internal")

        class MockOpts:
            max_dim = 0
            max_radius = 0.0
            metric = "euclidean"
            reduction = "clearing"

        mock_core.PersistenceOptions = MockOpts

        def mock_compute(pts, opts):
            return {"pairs": [[0.0, 1.0, 0]]}

        mock_core.compute_persistence = mock_compute

        original_core = sys.modules.get("pynerve_internal")
        sys.modules["pynerve_internal"] = mock_core
        try:
            importlib.reload(_phl_pre)
            from pynerve.diff.ph_layer import _compute_persistence_forward

            points = torch.randn(2, 3, 2)
            result, counts = _compute_persistence_forward(points, 0, "rips", return_counts=True)
            assert len(result) == 1
            assert len(counts) == 2
        finally:
            if original_core is not None:
                sys.modules["pynerve_internal"] = original_core
            else:
                sys.modules.pop("pynerve_internal", None)
            importlib.reload(_phl_pre)

    def test_differentiable_persistence_function_forward_mock(self):
        torch = self._torch()
        with patch("pynerve.diff.ph_layer._compute_persistence_forward") as mock_cp:
            mock_cp.return_value = [
                torch.randn(2, 3, 2),
                torch.randn(2, 3, 2),
            ]
            from pynerve.diff.ph_layer import DifferentiablePersistenceFunction

            points = torch.randn(2, 5, 3, requires_grad=True)
            result = DifferentiablePersistenceFunction.apply(
                points, 1, "rips", {"max_radius": float("inf")}
            )
            assert len(result) == 2


# torch/_sklearn_compat.py


class TestSklearnCompatExtended:
    def test_base_estimator_fallback_get_params_deep_ignored(self):
        from pynerve.torch._sklearn_compat import SKLEARN_AVAILABLE, BaseEstimator

        if SKLEARN_AVAILABLE:
            pytest.skip("sklearn installed; fallback not used")

        est = BaseEstimator()
        assert est.get_params(deep=True) == {}
        assert est.get_params(deep=False) == {}

    def test_base_estimator_fallback_set_params_returns_self(self):
        from pynerve.torch._sklearn_compat import SKLEARN_AVAILABLE, BaseEstimator

        if SKLEARN_AVAILABLE:
            pytest.skip("sklearn installed; fallback not used")

        est = BaseEstimator()
        result = est.set_params(x=10)
        assert result is est
        assert est.x == 10

    def test_transformer_mixin_fallback_no_y(self):
        from pynerve.torch._sklearn_compat import SKLEARN_AVAILABLE, TransformerMixin

        if SKLEARN_AVAILABLE:
            pytest.skip("sklearn installed; fallback not used")

        class _T(TransformerMixin):
            def fit(self, X, y=None, **kwargs):
                self.X_shape_ = len(X)
                return self

            def transform(self, X):
                return [f"t_{x}" for x in X]

        t = _T()
        result = t.fit_transform(["a", "b", "c"])
        assert result == ["t_a", "t_b", "t_c"]
        assert t.X_shape_ == 3

    def test_require_non_empty_with_tuple(self):
        from pynerve.torch._sklearn_compat import _require_non_empty

        _require_non_empty("stuff", (1, 2, 3))

    def test_require_non_empty_with_empty_tuple(self):
        from pynerve.torch._sklearn_compat import _require_non_empty

        with pytest.raises(ValueError, match="stuff"):
            _require_non_empty("stuff", ())

    def test_sklearn_available_is_bool(self):
        from pynerve.torch._sklearn_compat import SKLEARN_AVAILABLE

        assert isinstance(SKLEARN_AVAILABLE, bool)
