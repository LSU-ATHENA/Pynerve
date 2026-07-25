"""Tests for numba kernels — graph, reduction, simplices, distance, representations."""

from __future__ import annotations

import numpy as np
import pytest

# Do NOT use mock_gpu_deps — numba must be real for these tests


class TestNumbaConnectedComponents:
    def test_single_component(self):
        from pynerve._numba_graph import numba_connected_components

        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        labels = numba_connected_components(edges, n_vertices=3)
        assert len(labels) == 3
        assert labels[0] == labels[1] == labels[2]

    def test_two_components(self):
        from pynerve._numba_graph import numba_connected_components

        edges = np.array([[0, 1], [2, 3]], dtype=np.int64)
        labels = numba_connected_components(edges, n_vertices=4)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_all_singletons(self):
        from pynerve._numba_graph import numba_connected_components

        edges = np.empty((0, 2), dtype=np.int64)
        labels = numba_connected_components(edges, n_vertices=5)
        assert len(labels) == 5
        assert len(np.unique(labels)) == 5

    def test_chain(self):
        from pynerve._numba_graph import numba_connected_components

        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
        labels = numba_connected_components(edges, n_vertices=5)
        assert len(np.unique(labels)) == 1

    def test_empty_edges_zero_vertices(self):
        from pynerve._numba_graph import numba_connected_components

        edges = np.empty((0, 2), dtype=np.int64)
        labels = numba_connected_components(edges, n_vertices=0)
        assert len(labels) == 0


class TestNumbaMSTKruskal:
    def test_basic_mst(self):
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        weights = np.array([1.0, 1.0, 1.4], dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, n_vertices=3)
        assert mst.shape[0] == 2

    def test_single_vertex(self):
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.empty((0, 2), dtype=np.int64)
        weights = np.empty(0, dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, n_vertices=1)
        assert mst.shape == (0, 2)

    def test_two_vertices(self):
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.array([[0, 1]], dtype=np.int64)
        weights = np.array([2.5], dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, n_vertices=2)
        assert mst.shape == (1, 2)

    def test_mst_n_minus_1_edges(self):
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]], dtype=np.int64)
        weights = np.array([1.0, 2.0, 3.0, 4.0, 0.5], dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, n_vertices=5)
        assert mst.shape[0] == 4

    def test_mismatched_lengths(self):
        from pynerve._numba_graph import numba_mst_kruskal

        edges = np.array([[0, 1]], dtype=np.int64)
        weights = np.array([1.0, 2.0], dtype=np.float64)
        with pytest.raises(ValueError, match="matching"):
            numba_mst_kruskal(edges, weights, n_vertices=3)


class TestNumbaColumnReduction:
    def test_simple_boundary(self):
        from pynerve._numba_reduction import numba_column_reduction

        boundary = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.int64)
        filtration = np.array([0.0, 1.0, 2.0])
        pivots = numba_column_reduction(boundary.copy(), filtration)
        assert pivots.shape == (3,)
        assert (pivots >= -1).all()

    def test_empty_matrix(self):
        from pynerve._numba_reduction import numba_column_reduction

        boundary = np.zeros((0, 0), dtype=np.int64)
        filtration = np.zeros(0)
        pivots = numba_column_reduction(boundary.copy(), filtration)
        assert pivots.shape == (0,)


class TestNumbaSparseReduction:
    def test_simple_sparse(self):
        from pynerve._numba_reduction import numba_sparse_reduction

        columns = np.array([[0], [1]], dtype=np.int64)
        col_lengths = np.array([1, 1], dtype=np.int64)
        pivots = numba_sparse_reduction(columns, col_lengths)
        assert pivots.shape == (2,)

    def test_single_column(self):
        from pynerve._numba_reduction import numba_sparse_reduction

        columns = np.array([[0]], dtype=np.int64)
        col_lengths = np.array([1], dtype=np.int64)
        pivots = numba_sparse_reduction(columns, col_lengths)
        assert pivots[0] == 0


class TestNumbaVREdges:
    def test_basic_edges(self):
        from pynerve._numba_simplices import numba_vr_edges

        points = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float64)
        edges = numba_vr_edges(points, max_dist=1.5)
        assert edges.shape[1] == 2
        # dist(0,1)=1.0 <= 1.5; dist(1,2)=2.0 > 1.5; dist(0,2)=3.0 > 1.5
        assert len(edges) == 1

    def test_no_edges_large_distance(self):
        from pynerve._numba_simplices import numba_vr_edges

        points = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        edges = numba_vr_edges(points, max_dist=1.0)
        assert len(edges) == 0

    def test_negative_max_dist_raises(self):
        from pynerve._numba_simplices import numba_vr_edges

        points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="non-negative"):
            numba_vr_edges(points, max_dist=-1.0)


class TestNumbaTriangleEnumeration:
    def test_basic_triangles(self):
        from pynerve._numba_simplices import numba_triangle_enumeration

        edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        triangles = numba_triangle_enumeration(edges, n_vertices=3)
        assert triangles.shape[1] == 3
        assert len(triangles) >= 1

    def test_no_triangles(self):
        from pynerve._numba_simplices import numba_triangle_enumeration

        edges = np.array([[0, 1], [2, 3]], dtype=np.int64)
        triangles = numba_triangle_enumeration(edges, n_vertices=4)
        assert len(triangles) == 0

    def test_empty_edges(self):
        from pynerve._numba_simplices import numba_triangle_enumeration

        edges = np.empty((0, 2), dtype=np.int64)
        triangles = numba_triangle_enumeration(edges, n_vertices=5)
        assert len(triangles) == 0


class TestNumbaSimplexBoundary:
    def test_edge_boundary(self):
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([3, 7], dtype=np.int64)
        boundary = numba_simplex_boundary(simplex)
        assert boundary.shape == (2, 1)
        assert boundary[0, 0] in {3, 7}
        assert boundary[1, 0] in {3, 7}

    def test_triangle_boundary(self):
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([1, 2, 3], dtype=np.int64)
        boundary = numba_simplex_boundary(simplex)
        assert boundary.shape == (3, 2)

    def test_single_vertex(self):
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([5], dtype=np.int64)
        boundary = numba_simplex_boundary(simplex)
        assert boundary.shape == (0, 0) or boundary.size == 0
