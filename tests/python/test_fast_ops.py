"""Correctness tests for the fast NumPy-based operations in pynerve.

Covers distance, simplex, and graph functions without requiring the C++ extension.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pynerve._fast_distance import (
    pairwise_distances_broadcast,
    pairwise_distances_fast,
    sparse_distance_matrix,
)
from pynerve._fast_graph import connected_components_fast, minimum_spanning_tree_fast
from pynerve._fast_simplices import (
    enumerate_simplices_fast,
    simplex_boundary_fast,
    vr_edges_fast,
)
from pynerve.exceptions import ValidationError
from scipy.spatial.distance import cdist

# Distance functions


class TestPairwiseDistancesFast:
    """Correctness tests for pairwise_distances_fast (scipy-based)."""

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev"])
    def test_known_1d(self, metric: str) -> None:
        points = np.array([[0.0], [3.0], [4.0], [7.0]])
        dists = pairwise_distances_fast(points, metric=metric)
        expected = cdist(points, points, metric=metric)
        np.testing.assert_allclose(dists, expected, rtol=1e-12)

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev"])
    def test_known_2d(self, metric: str) -> None:
        points = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        dists = pairwise_distances_fast(points, metric=metric)
        expected = cdist(points, points, metric=metric)
        np.testing.assert_allclose(dists, expected, rtol=1e-12)

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev"])
    def test_known_3d(self, metric: str) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dists = pairwise_distances_fast(points, metric=metric)
        expected = cdist(points, points, metric=metric)
        np.testing.assert_allclose(dists, expected, rtol=1e-12)

    def test_equivalence_scipy(self) -> None:
        rng = np.random.default_rng(42)
        points = rng.uniform(-10, 10, (20, 5))
        dists = pairwise_distances_fast(points, metric="euclidean")
        expected = cdist(points, points, metric="euclidean")
        np.testing.assert_allclose(dists, expected, rtol=1e-10)

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev"])
    def test_symmetry(self, metric: str) -> None:
        rng = np.random.default_rng(43)
        points = rng.uniform(-5, 5, (15, 3))
        dists = pairwise_distances_fast(points, metric=metric)
        np.testing.assert_allclose(dists, dists.T, rtol=1e-12)
        assert np.allclose(dists, dists.T)

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev"])
    def test_zero_diagonal(self, metric: str) -> None:
        rng = np.random.default_rng(44)
        points = rng.uniform(-5, 5, (10, 4))
        dists = pairwise_distances_fast(points, metric=metric)
        assert np.allclose(np.diag(dists), 0.0)

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev"])
    def test_non_negative(self, metric: str) -> None:
        rng = np.random.default_rng(45)
        points = rng.uniform(-5, 5, (12, 3))
        dists = pairwise_distances_fast(points, metric=metric)
        assert np.all(dists >= 0)

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev"])
    def test_triangle_inequality(self, metric: str) -> None:
        points = np.array([[0.0], [1.0], [2.5]])
        dists = pairwise_distances_fast(points, metric=metric)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    assert dists[i, j] <= dists[i, k] + dists[k, j] + 1e-12

    def test_single_point(self) -> None:
        points = np.array([[3.14, 2.71]])
        dists = pairwise_distances_fast(points)
        assert dists.shape == (1, 1)
        assert dists[0, 0] == 0.0

    def test_empty_input_zero_points(self) -> None:
        """pairwise_distances_fast with 0 points produces (1,1) via scipy squareform."""
        dists = pairwise_distances_fast(np.empty((0, 2)))
        assert dists.shape == (1, 1)
        assert dists[0, 0] == 0.0

    def test_determinism(self) -> None:
        rng = np.random.default_rng(46)
        points = rng.uniform(-10, 10, (25, 3))
        d1 = pairwise_distances_fast(points)
        d2 = pairwise_distances_fast(points)
        np.testing.assert_array_equal(d1, d2)

    def test_inf_input_raises(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, np.inf]])
        with pytest.raises(ValidationError):
            pairwise_distances_fast(points)

    def test_nan_input_raises(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, np.nan]])
        with pytest.raises(ValidationError):
            pairwise_distances_fast(points)

    def test_invalid_metric_raises(self) -> None:
        points = np.array([[0.0], [1.0]])
        with pytest.raises(ValueError):
            pairwise_distances_fast(points, metric="")


class TestPairwiseDistancesBroadcast:
    """Correctness tests for pairwise_distances_broadcast (NumPy broadcasting)."""

    def test_known_1d(self) -> None:
        points = np.array([[0.0], [3.0], [4.0], [7.0]])
        dists = pairwise_distances_broadcast(points)
        expected = cdist(points, points, metric="euclidean")
        np.testing.assert_allclose(dists, expected, rtol=1e-12)

    def test_known_2d(self) -> None:
        points = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        dists = pairwise_distances_broadcast(points)
        expected = cdist(points, points, metric="euclidean")
        np.testing.assert_allclose(dists, expected, rtol=1e-12)

    def test_known_3d(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dists = pairwise_distances_broadcast(points)
        expected = cdist(points, points, metric="euclidean")
        np.testing.assert_allclose(dists, expected, rtol=1e-12)

    def test_equivalence_pairwise_distances_fast(self) -> None:
        rng = np.random.default_rng(47)
        points = rng.uniform(-10, 10, (20, 5))
        d_broad = pairwise_distances_broadcast(points)
        d_fast = pairwise_distances_fast(points, metric="euclidean")
        np.testing.assert_allclose(d_broad, d_fast, rtol=1e-10)

    def test_symmetry(self) -> None:
        rng = np.random.default_rng(48)
        points = rng.uniform(-5, 5, (15, 3))
        dists = pairwise_distances_broadcast(points)
        np.testing.assert_allclose(dists, dists.T, rtol=1e-12)

    def test_zero_diagonal(self) -> None:
        rng = np.random.default_rng(49)
        points = rng.uniform(-5, 5, (10, 4))
        dists = pairwise_distances_broadcast(points)
        assert np.allclose(np.diag(dists), 0.0)

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(50)
        points = rng.uniform(-5, 5, (12, 3))
        dists = pairwise_distances_broadcast(points)
        assert np.all(dists >= 0)

    def test_single_point(self) -> None:
        points = np.array([[3.14, 2.71]])
        dists = pairwise_distances_broadcast(points)
        assert dists.shape == (1, 1)
        assert dists[0, 0] == 0.0

    def test_empty_input_zero_points(self) -> None:
        """pairwise_distances_broadcast with 0 points produces (0,0)."""
        dists = pairwise_distances_broadcast(np.empty((0, 2)))
        assert dists.shape == (0, 0)

    def test_determinism(self) -> None:
        rng = np.random.default_rng(51)
        points = rng.uniform(-10, 10, (25, 3))
        d1 = pairwise_distances_broadcast(points)
        d2 = pairwise_distances_broadcast(points)
        np.testing.assert_array_equal(d1, d2)

    def test_inf_input_raises(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, np.inf]])
        with pytest.raises(ValidationError):
            pairwise_distances_broadcast(points)

    def test_nan_input_raises(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, np.nan]])
        with pytest.raises(ValidationError):
            pairwise_distances_broadcast(points)


class TestSparseDistanceMatrix:
    """Correctness tests for sparse_distance_matrix."""

    def test_dense_output_equivalence(self) -> None:
        rng = np.random.default_rng(52)
        points = rng.uniform(-5, 5, (8, 3))
        full = pairwise_distances_fast(points)
        sparse = sparse_distance_matrix(points, max_dist=5.0, output_type="dense")
        full_clamped = full.copy()
        full_clamped[full > 5.0] = 0
        np.testing.assert_allclose(sparse, full_clamped, rtol=1e-12)

    def test_very_large_max_dist(self) -> None:
        rng = np.random.default_rng(53)
        points = rng.uniform(0, 1, (6, 2))
        full = pairwise_distances_fast(points)
        sparse = sparse_distance_matrix(points, max_dist=1e6, output_type="dense")
        np.testing.assert_allclose(sparse, full, rtol=1e-10)

    def test_zero_max_dist(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        sparse = sparse_distance_matrix(points, max_dist=0.0, output_type="dense")
        assert np.all(sparse == 0)

    def test_negative_max_dist_raises(self) -> None:
        points = np.array([[0.0], [1.0]])
        with pytest.raises(ValidationError):
            sparse_distance_matrix(points, max_dist=-1.0)

    def test_inf_max_dist_raises(self) -> None:
        points = np.array([[0.0], [1.0]])
        with pytest.raises(ValidationError):
            sparse_distance_matrix(points, max_dist=float("inf"))

    def test_nan_max_dist_raises(self) -> None:
        points = np.array([[0.0], [1.0]])
        with pytest.raises(ValidationError):
            sparse_distance_matrix(points, max_dist=float("nan"))

    def test_coo_output_nonnegative(self) -> None:
        rng = np.random.default_rng(54)
        points = rng.uniform(-3, 3, (10, 2))
        coo = sparse_distance_matrix(points, max_dist=2.0, output_type="coo")
        assert coo.data.min() >= 0

    def test_csr_output_same_data(self) -> None:
        rng = np.random.default_rng(55)
        points = rng.uniform(-3, 3, (10, 2))
        coo = sparse_distance_matrix(points, max_dist=2.0, output_type="coo")
        csr = sparse_distance_matrix(points, max_dist=2.0, output_type="csr")
        np.testing.assert_allclose(csr.data, coo.data, rtol=1e-12)

    def test_no_self_loops(self) -> None:
        """Sparse output should not include self-distances (distance 0)."""
        rng = np.random.default_rng(56)
        points = rng.uniform(-3, 3, (10, 2))
        coo = sparse_distance_matrix(points, max_dist=10.0, output_type="coo")
        assert np.all(coo.row != coo.col)

    def test_invalid_output_type_raises(self) -> None:
        points = np.array([[0.0], [1.0]])
        with pytest.raises(ValueError):
            sparse_distance_matrix(points, max_dist=1.0, output_type="invalid")


# Simplex functions


class TestVREdges:
    """Correctness tests for vr_edges_fast."""

    def test_known_triangle(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        edges = vr_edges_fast(points, max_dist=1.5)
        edges_set = {tuple(sorted(e)) for e in edges}
        # dist(0,1)=1.0, dist(0,2)=1.0, dist(1,2)=sqrt(2)~1.414 -- all < 1.5
        assert len(edges) == 3
        assert edges_set == {(0, 1), (0, 2), (1, 2)}

    def test_radius_threshold(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        edges = vr_edges_fast(points, max_dist=1.1)
        edges_set = {tuple(sorted(e)) for e in edges}
        # Only adjacent points (distance 1.0) are within 1.1
        assert edges_set == {(0, 1), (1, 2), (2, 3)}

    def test_zero_radius_no_edges(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 0.0]])
        edges = vr_edges_fast(points, max_dist=0.0)
        assert len(edges) == 0

    def test_no_self_loops(self) -> None:
        rng = np.random.default_rng(57)
        points = rng.uniform(-3, 3, (15, 4))
        edges = vr_edges_fast(points, max_dist=10.0)
        for e in edges:
            assert e[0] != e[1]

    def test_all_vertices_in_range(self) -> None:
        rng = np.random.default_rng(58)
        points = rng.uniform(-3, 3, (10, 2))
        edges = vr_edges_fast(points, max_dist=10.0)
        if len(edges) > 0:
            assert edges.min() >= 0
            assert edges.max() < 10

    def test_empty_point_set_zero_edges(self) -> None:
        """vr_edges_fast with 0 points produces (0,2) array."""
        edges = vr_edges_fast(np.empty((0, 2)), max_dist=1.0)
        assert edges.shape == (0, 2)

    def test_single_point_no_edges(self) -> None:
        points = np.array([[0.0, 0.0]])
        edges = vr_edges_fast(points, max_dist=1.0)
        assert len(edges) == 0

    def test_with_distances(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        edges, edists = vr_edges_fast(points, max_dist=1.1, return_dists=True)
        assert len(edges) == len(edists)
        assert edges.shape[1] == 2
        assert edists.ndim == 1
        for i, (a, b) in enumerate(edges):
            expected = np.linalg.norm(points[a] - points[b])
            assert math.isclose(edists[i], expected, rel_tol=1e-10)

    def test_determinism(self) -> None:
        rng = np.random.default_rng(59)
        points = rng.uniform(-3, 3, (12, 3))
        e1 = vr_edges_fast(points, max_dist=1.5)
        e2 = vr_edges_fast(points, max_dist=1.5)
        np.testing.assert_array_equal(e1, e2)

    def test_negative_max_dist_raises(self) -> None:
        points = np.array([[0.0], [1.0]])
        with pytest.raises(ValidationError):
            vr_edges_fast(points, max_dist=-0.1)


class TestEnumerateSimplices:
    """Correctness tests for enumerate_simplices_fast."""

    def test_known_triangle_maxdim2(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        simplices = enumerate_simplices_fast(points, max_dist=1.5, max_dim=2)
        # dim 0: 3 vertices
        assert len(simplices) == 3
        assert simplices[0].shape == (3, 1)  # vertices
        assert set(simplices[0].ravel()) == {0, 1, 2}
        # dim 1: 3 edges
        assert simplices[1].shape[0] == 3
        assert simplices[1].shape[1] == 2
        # dim 2: 1 triangle
        assert simplices[2].shape[0] == 1
        assert simplices[2].shape[1] == 3

    def test_four_point_square(self) -> None:
        """Square with edge length 1 and diagonal sqrt(2).

        radius=1.1: includes edges but not diagonal, so no triangles exist.
        When no higher simplices are found, enumeration stops early.
        """
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        simplices = enumerate_simplices_fast(points, max_dist=1.1, max_dim=2)
        # dim 0: 4 vertices
        assert simplices[0].shape == (4, 1)
        # dim 1: 4 edges (no diagonals within 1.1)
        assert simplices[1].shape[0] == 4
        # No higher simplices: enumeration stopped after dim 1
        assert len(simplices) == 2

    def test_max_dim_zero(self) -> None:
        points = np.array([[0.0], [1.0], [2.0]])
        simplices = enumerate_simplices_fast(points, max_dist=1.5, max_dim=0)
        assert len(simplices) == 1
        assert simplices[0].shape == (3, 1)

    def test_max_dim_one(self) -> None:
        points = np.array([[0.0], [1.0], [2.0]])
        simplices = enumerate_simplices_fast(points, max_dist=1.5, max_dim=1)
        assert len(simplices) == 2
        assert simplices[0].shape == (3, 1)  # dim 0
        assert simplices[1].shape == (2, 2)  # dim 1: edges (0,1), (1,2)

    def test_empty_point_set_zero_simplices(self) -> None:
        """enumerate_simplices_fast with 0 points returns empty groups per dimension."""
        simplices = enumerate_simplices_fast(np.empty((0, 2)), max_dist=1.0)
        # vertices group and potentially empty edge group
        assert len(simplices) >= 1
        assert simplices[0].shape[0] == 0
        for s in simplices:
            assert s.shape[0] == 0

    def test_single_point(self) -> None:
        points = np.array([[0.0, 0.0]])
        simplices = enumerate_simplices_fast(points, max_dist=1.0, max_dim=2)
        assert len(simplices) >= 1
        assert simplices[0].shape == (1, 1)
        # No edges or higher simplices
        if len(simplices) > 1:
            for s in simplices[1:]:
                assert s.shape[0] == 0

    def test_vertex_order(self) -> None:
        """Vertices in each simplex should be strictly increasing."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        simplices = enumerate_simplices_fast(points, max_dist=1.5, max_dim=2)
        for dim_idx, simps in enumerate(simplices):
            for row in simps:
                if dim_idx > 0:
                    for i in range(len(row) - 1):
                        assert row[i] < row[i + 1], (
                            f"non-increasing vertex order in dim {dim_idx}: {row}"
                        )

    def test_determinism(self) -> None:
        rng = np.random.default_rng(60)
        points = rng.uniform(-2, 2, (8, 2))
        s1 = enumerate_simplices_fast(points, max_dist=1.2, max_dim=2)
        s2 = enumerate_simplices_fast(points, max_dist=1.2, max_dim=2)
        for a, b in zip(s1, s2, strict=True):
            np.testing.assert_array_equal(a, b)


class TestSimplexBoundary:
    """Correctness tests for simplex_boundary_fast."""

    def test_edge_boundary(self) -> None:
        simplex = np.array([3, 7], dtype=np.int64)
        boundary = simplex_boundary_fast(simplex)
        assert boundary.shape == (2, 1)
        assert boundary[0, 0] in {3, 7}
        assert boundary[1, 0] in {3, 7}
        assert boundary[0, 0] != boundary[1, 0]

    def test_triangle_boundary(self) -> None:
        simplex = np.array([1, 2, 3], dtype=np.int64)
        boundary = simplex_boundary_fast(simplex)
        assert boundary.shape == (3, 2)
        faces = {tuple(sorted(row)) for row in boundary}
        assert faces == {(2, 3), (1, 3), (1, 2)}

    def test_tetrahedron_boundary(self) -> None:
        simplex = np.array([0, 1, 2, 3], dtype=np.int64)
        boundary = simplex_boundary_fast(simplex)
        assert boundary.shape == (4, 3)
        assert len({tuple(sorted(row)) for row in boundary}) == 4

    def test_single_vertex(self) -> None:
        simplex = np.array([5], dtype=np.int64)
        boundary = simplex_boundary_fast(simplex)
        assert boundary.shape == (1, 0)

    def test_all_vertices_present(self) -> None:
        simplex = np.array([9, 3, 7, 1], dtype=np.int64)
        boundary = simplex_boundary_fast(simplex)
        for row in boundary:
            for v in row:
                assert int(v) in {1, 3, 7, 9}

    def test_empty_simplex_raises(self) -> None:
        with pytest.raises(ValueError):
            simplex_boundary_fast(np.array([], dtype=np.int64))

    def test_duplicate_vertices_raises(self) -> None:
        with pytest.raises(ValueError):
            simplex_boundary_fast(np.array([1, 1], dtype=np.int64))

    def test_negative_indices_raises(self) -> None:
        with pytest.raises(ValueError):
            simplex_boundary_fast(np.array([1, -1], dtype=np.int64))

    def test_non_integer_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            simplex_boundary_fast(np.array([1.5, 2.0]))

    def test_not_1d_raises(self) -> None:
        with pytest.raises(ValueError):
            simplex_boundary_fast(np.array([[0], [1]], dtype=np.int64))


# Graph functions


class TestConnectedComponents:
    """Correctness tests for connected_components_fast."""

    def test_single_component_two_nodes(self) -> None:
        edges = np.array([[0, 1]], dtype=np.int64)
        labels = connected_components_fast(edges, n_vertices=2)
        assert labels[0] == labels[1]
        assert len(np.unique(labels)) == 1

    def test_two_components(self) -> None:
        edges = np.array([[0, 1], [2, 3]], dtype=np.int64)
        labels = connected_components_fast(edges, n_vertices=4)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]
        assert len(np.unique(labels)) == 2

    def test_three_components(self) -> None:
        edges = np.array([[0, 1], [3, 4]], dtype=np.int64)
        labels = connected_components_fast(edges, n_vertices=6)
        # 0-1: comp A; 2: comp B alone; 3-4: comp C; 5: comp D alone
        assert labels[0] == labels[1]
        assert labels[3] == labels[4]
        assert labels[0] != labels[2]
        assert labels[0] != labels[3]
        assert labels[0] != labels[5]
        assert labels[2] != labels[5]
        assert len(np.unique(labels)) == 4

    def test_singleton_nodes(self) -> None:
        edges = np.array([[1, 2]], dtype=np.int64)
        labels = connected_components_fast(edges, n_vertices=5)
        # Vertex 0 should be alone
        assert labels[0] != labels[1]
        assert labels[3] != labels[1]
        assert labels[4] != labels[1]

    def test_chain_graph(self) -> None:
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
        labels = connected_components_fast(edges, n_vertices=5)
        assert len(np.unique(labels)) == 1

    def test_empty_edges_all_singletons(self) -> None:
        edges = np.empty((0, 2), dtype=np.int64)
        labels = connected_components_fast(edges, n_vertices=5)
        assert len(np.unique(labels)) == 5
        # Each vertex has its own label
        for i in range(5):
            for j in range(i + 1, 5):
                assert labels[i] != labels[j]

    def test_single_vertex_no_edges(self) -> None:
        edges = np.empty((0, 2), dtype=np.int64)
        labels = connected_components_fast(edges, n_vertices=1)
        assert len(labels) == 1
        assert labels[0] == 0

    def test_zero_vertices(self) -> None:
        edges = np.empty((0, 2), dtype=np.int64)
        labels = connected_components_fast(edges, n_vertices=0)
        assert len(labels) == 0

    def test_complete_graph_single_component(self) -> None:
        edges = np.array(
            [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
            dtype=np.int64,
        )
        labels = connected_components_fast(edges, n_vertices=5)
        assert len(np.unique(labels)) == 1

    def test_disconnected_chain(self) -> None:
        edges = np.array([[0, 1], [3, 4]], dtype=np.int64)
        labels = connected_components_fast(edges, n_vertices=5)
        # Two connected components plus vertex 2 isolated
        assert labels[0] == labels[1]
        assert labels[3] == labels[4]
        assert labels[0] != labels[3]
        assert labels[0] != labels[2]
        assert labels[3] != labels[2]
        assert len(np.unique(labels)) == 3

    def test_labels_are_contiguous(self) -> None:
        rng = np.random.default_rng(61)
        n = 20
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        chosen = rng.choice(len(all_pairs), size=25, replace=False)
        edges = np.array([all_pairs[i] for i in chosen], dtype=np.int64)
        labels = connected_components_fast(edges, n_vertices=n)
        unique = np.unique(labels)
        assert unique[0] == 0
        assert np.all(unique == np.arange(len(unique)))

    def test_determinism(self) -> None:
        rng = np.random.default_rng(62)
        n = 15
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        chosen = rng.choice(len(all_pairs), size=12, replace=False)
        edges = np.array([all_pairs[i] for i in chosen], dtype=np.int64)
        l1 = connected_components_fast(edges, n_vertices=n)
        l2 = connected_components_fast(edges, n_vertices=n)
        np.testing.assert_array_equal(l1, l2)

    def test_out_of_bounds_index_raises(self) -> None:
        edges = np.array([[0, 5]], dtype=np.int64)
        with pytest.raises(ValueError):
            connected_components_fast(edges, n_vertices=4)

    def test_negative_index_raises(self) -> None:
        edges = np.array([[-1, 2]], dtype=np.int64)
        with pytest.raises(ValueError):
            connected_components_fast(edges, n_vertices=3)


class TestMinimumSpanningTree:
    """Correctness tests for minimum_spanning_tree_fast."""

    def test_single_edge(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 0.0]])
        mst = minimum_spanning_tree_fast(points)
        assert len(mst) == 1
        assert tuple(sorted(mst[0])) == (0, 1)

    def test_three_collinear_points(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        mst = minimum_spanning_tree_fast(points)
        assert len(mst) == 2
        edges_sorted = {tuple(sorted(e)) for e in mst}
        assert edges_sorted == {(0, 1), (1, 2)}

    def test_square(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        mst = minimum_spanning_tree_fast(points)
        # MST of a square has 3 edges
        assert len(mst) == 3
        # Total MST weight: 3 edges of length 1 = 3.0 (not counting diagonal)
        total = 0.0
        for a, b in mst:
            total += np.linalg.norm(points[a] - points[b])
        assert math.isclose(total, 3.0, rel_tol=1e-10)

    def test_square_with_diagonal_avoided(self) -> None:
        """MST must not pick the diagonal (sqrt(2)) when cheaper edges exist."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        mst = minimum_spanning_tree_fast(points)
        diag_edges = {(0, 2), (1, 3)}
        mst_set = {tuple(sorted(e)) for e in mst}
        for d in diag_edges:
            assert d not in mst_set

    def test_mst_connects_all_vertices(self) -> None:
        rng = np.random.default_rng(63)
        points = rng.uniform(-5, 5, (10, 2))
        mst = minimum_spanning_tree_fast(points)
        # Compute components from MST edges
        from pynerve._fast_graph import connected_components_fast as _cc

        labels = _cc(mst, n_vertices=10)
        assert len(np.unique(labels)) == 1

    def test_mst_has_n_minus_1_edges(self) -> None:
        for n in [2, 3, 5, 10]:
            rng = np.random.default_rng(64 + n)
            points = rng.uniform(-5, 5, (n, 3))
            mst = minimum_spanning_tree_fast(points)
            assert len(mst) == n - 1

    def test_single_node(self) -> None:
        points = np.array([[0.0, 0.0]])
        mst = minimum_spanning_tree_fast(points)
        assert len(mst) == 0
        assert mst.shape == (0, 2)

    def test_empty_input_zero_mst(self) -> None:
        """minimum_spanning_tree_fast with 0 points produces (0,2) array."""
        mst = minimum_spanning_tree_fast(np.empty((0, 2)))
        assert mst.shape == (0, 2)

    def test_no_self_loops(self) -> None:
        rng = np.random.default_rng(65)
        points = rng.uniform(-3, 3, (8, 2))
        mst = minimum_spanning_tree_fast(points)
        for e in mst:
            assert e[0] != e[1]

    def test_custom_edges(self) -> None:
        """MST computed from provided edge set matches full edge set result."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        # Provide only the 4 boundary edges (no diagonals)
        boundary_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
        mst_custom = minimum_spanning_tree_fast(points, edges=boundary_edges)
        assert len(mst_custom) == 3
        total_custom = 0.0
        for a, b in mst_custom:
            total_custom += np.linalg.norm(points[a] - points[b])
        assert math.isclose(total_custom, 3.0, rel_tol=1e-10)

    def test_custom_edges_subset(self) -> None:
        """With a restricted edge set, MST uses only those edges."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        # Only allow edges (0,1) and (1,2) -- not (0,2) directly
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        mst = minimum_spanning_tree_fast(points, edges=edges)
        assert len(mst) == 2
        mst_set = {tuple(sorted(e)) for e in mst}
        assert mst_set == {(0, 1), (1, 2)}

    def test_empty_edges_no_mst(self) -> None:
        points = np.array([[0.0], [1.0], [2.0]])
        mst = minimum_spanning_tree_fast(points, edges=np.empty((0, 2), dtype=np.int64))
        assert len(mst) == 0

    def test_determinism(self) -> None:
        rng = np.random.default_rng(66)
        points = rng.uniform(-5, 5, (15, 2))
        mst1 = minimum_spanning_tree_fast(points)
        mst2 = minimum_spanning_tree_fast(points)
        np.testing.assert_array_equal(mst1, mst2)

    def test_inf_points_raises(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, np.inf]])
        with pytest.raises(ValidationError):
            minimum_spanning_tree_fast(points)

    def test_nan_points_raises(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, np.nan]])
        with pytest.raises(ValidationError):
            minimum_spanning_tree_fast(points)
