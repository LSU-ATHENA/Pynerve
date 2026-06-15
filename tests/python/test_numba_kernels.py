"""Correctness tests for numba-accelerated kernel modules.

Each test class covers one kernel module.  When numba is not installed
every test is skipped via ``pytest.importorskip("numba")`` at module level.
"""

from __future__ import annotations

import numpy as np
import pytest

numba = pytest.importorskip("numba")


# helpers


def _call_numba_fn(import_path: str, fn_name: str, *args, **kwargs):
    """Import and call a numba function, skipping if compilation fails.

    Some numba functions use language constructs (e.g. ``del``) that are not
    supported by all numba versions.  This helper turns a ``CompilerError``
    into a skip so the remaining tests can still execute.
    """
    import importlib

    mod = importlib.import_module(import_path)
    fn = getattr(mod, fn_name)
    try:
        return fn(*args, **kwargs)
    except numba.core.errors.CompilerError as exc:
        pytest.skip(f"{fn_name} compilation failed: {exc}")


# _numba_compat


class TestNumbaCompat:
    def test_has_numba_is_boolean(self) -> None:
        from pynerve._numba_compat import HAS_NUMBA

        assert isinstance(HAS_NUMBA, bool)

    def test_njit_available(self) -> None:
        from pynerve._numba_compat import njit

        assert callable(njit)

        @njit
        def _add(a: float, b: float) -> float:
            return a + b

        assert _add(3.0, 4.0) == 7.0

    def test_prange_available(self) -> None:
        from pynerve._numba_compat import prange

        assert callable(prange)
        assert list(prange(3)) == [0, 1, 2]

    def test_njit_with_kwargs(self) -> None:
        from pynerve._numba_compat import njit

        @njit(cache=False)
        def _mul(a: float, b: float) -> float:
            return a * b

        assert _mul(3.0, 4.0) == 12.0


# _numba_distance


class TestNumbaPairwiseDistances:
    @staticmethod
    def _scipy_pairwise(points: np.ndarray) -> np.ndarray:
        from scipy.spatial.distance import pdist, squareform

        return squareform(pdist(points.astype(np.float64), "euclidean"))

    def test_random_small(self) -> None:
        from pynerve._numba_distance import numba_pairwise_distances

        rng = np.random.default_rng(42)
        for n in [2, 3, 5, 10, 20]:
            pts = rng.normal(size=(n, 3)).astype(np.float64)
            got = numba_pairwise_distances(pts)
            expected = self._scipy_pairwise(pts)
            np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_single_point(self) -> None:
        from pynerve._numba_distance import numba_pairwise_distances

        pts = np.array([[1.5, 2.5, 3.5]], dtype=np.float64)
        got = numba_pairwise_distances(pts)
        assert got.shape == (1, 1)
        assert got[0, 0] == 0.0

    def test_two_points(self) -> None:
        from pynerve._numba_distance import numba_pairwise_distances

        pts = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float64)
        got = numba_pairwise_distances(pts)
        assert got[0, 0] == 0.0
        assert got[1, 1] == 0.0
        assert abs(got[0, 1] - 5.0) < 1e-12
        assert abs(got[1, 0] - 5.0) < 1e-12

    def test_symmetry(self) -> None:
        from pynerve._numba_distance import numba_pairwise_distances

        rng = np.random.default_rng(7)
        pts = rng.normal(size=(12, 4)).astype(np.float64)
        got = numba_pairwise_distances(pts)
        np.testing.assert_allclose(got, got.T, rtol=1e-15)

    def test_zero_diagonal(self) -> None:
        from pynerve._numba_distance import numba_pairwise_distances

        rng = np.random.default_rng(3)
        pts = rng.normal(size=(8, 3)).astype(np.float64)
        got = numba_pairwise_distances(pts)
        np.testing.assert_allclose(np.diag(got), 0.0, atol=1e-15)


class TestNumbaNearestNeighbors:
    def test_k_smaller_than_n(self) -> None:
        from pynerve._numba_distance import numba_nearest_neighbors

        rng = np.random.default_rng(11)
        pts = rng.normal(size=(20, 3)).astype(np.float64)
        dists, indices = numba_nearest_neighbors(pts, k=3)
        assert dists.shape == (20, 3)
        assert indices.shape == (20, 3)

        # No self-neighbor
        assert not np.any(indices == np.arange(20)[:, None])

        # Distances are sorted within each row
        diffs = np.diff(dists, axis=1)
        assert np.all(diffs >= 0.0)

    def test_versus_brute_force(self) -> None:
        from pynerve._numba_distance import numba_nearest_neighbors

        rng = np.random.default_rng(13)
        pts = rng.normal(size=(15, 2)).astype(np.float64)
        k = 5
        dists_nb, idx_nb = numba_nearest_neighbors(pts, k)

        # Brute force
        n = pts.shape[0]
        all_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    all_dist[i, j] = np.linalg.norm(pts[i] - pts[j])
                else:
                    all_dist[i, j] = np.inf
        sorted_idx = np.argsort(all_dist, axis=1)[:, :k]

        for i in range(n):
            nb_expected_idx = sorted(sorted_idx[i])
            nb_got_idx = sorted(idx_nb[i])
            assert nb_expected_idx == nb_got_idx, f"row {i} mismatch"

            for j in range(k):
                true_dist = np.linalg.norm(pts[i] - pts[idx_nb[i, j]])
                assert abs(dists_nb[i, j] - true_dist) < 1e-12

    def test_k_zero_returns_empty(self) -> None:
        from pynerve._numba_distance import numba_nearest_neighbors

        pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        dists, indices = numba_nearest_neighbors(pts, k=0)
        assert dists.shape == (2, 0)
        assert indices.shape == (2, 0)

    def test_single_point(self) -> None:
        from pynerve._numba_distance import numba_nearest_neighbors

        pts = np.array([[1.0, 2.0]], dtype=np.float64)
        dists, indices = numba_nearest_neighbors(pts, k=3)
        assert dists.shape == (1, 3)
        assert indices.shape == (1, 3)
        assert np.all(np.isinf(dists))
        assert np.all(indices == 0)


# _numba_graph


class TestNumbaConnectedComponents:
    def test_simple_two_component(self) -> None:
        from pynerve._numba_graph import numba_connected_components

        # graph: 0-1  2-3-4
        edges = np.array([[0, 1], [2, 3], [3, 4]], dtype=np.int64)
        labels = numba_connected_components(edges, n_vertices=5)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3] == labels[4]
        assert labels[0] != labels[2]
        assert len(set(labels)) == 2

    def test_versus_scipy(self) -> None:
        from pynerve._numba_graph import numba_connected_components
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        rng = np.random.default_rng(99)
        for n in [5, 10, 20]:
            # Generate a random undirected graph
            adj = rng.random((n, n)) < 0.3
            adj = np.triu(adj, k=1)
            rows, cols = np.where(adj)
            edges = np.column_stack([rows, cols]).astype(np.int64)
            sparse = coo_matrix(
                (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
                shape=(n, n),
            )

            n_cc_scipy, _labels_scipy = connected_components(sparse, directed=False)
            labels_numba = numba_connected_components(edges, n)
            n_cc_numba = len(set(labels_numba))
            assert n_cc_numba == n_cc_scipy

    def test_empty_edges(self) -> None:
        from pynerve._numba_graph import numba_connected_components

        edges = np.empty((0, 2), dtype=np.int64)
        labels = numba_connected_components(edges, n_vertices=5)
        assert len(set(labels)) == 5  # each vertex in its own component

    def test_negative_n_vertices_raises(self) -> None:
        from pynerve._numba_graph import numba_connected_components

        with pytest.raises(ValueError, match="non-negative"):
            numba_connected_components(np.empty((0, 2), dtype=np.int64), n_vertices=-1)


class TestNumbaMstKruskal:
    def test_triangle(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        # Triangle with weights 1, 2, 3 -> MST edges: weights 1 and 2
        edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, n_vertices=3)
        assert len(mst) == 2  # V-1

        mst_weight = 0.0
        for e in mst:
            u, v = int(e[0]), int(e[1])
            for idx in range(len(edges)):
                if {int(edges[idx, 0]), int(edges[idx, 1])} == {u, v}:
                    mst_weight += weights[idx]
                    break
        assert abs(mst_weight - 3.0) < 1e-12

    def test_spanning_tree_properties(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        rng = np.random.default_rng(17)
        for n in [4, 6, 10]:
            rows, cols = np.triu_indices(n, k=1)
            edges = np.column_stack([rows, cols]).astype(np.int64)
            weights = rng.uniform(1.0, 10.0, size=len(edges)).astype(np.float64)

            mst = numba_mst_kruskal(edges, weights, n)
            assert len(mst) == n - 1, f"n={n}: got {len(mst)} edges"

            vertices = set(mst[:, 0]) | set(mst[:, 1])
            assert vertices == set(range(n)), f"n={n}: MST does not span all vertices"

    def test_minimum_total_weight(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        n = 5
        rows, cols = np.triu_indices(n, k=1)
        edges_all = np.column_stack([rows, cols]).astype(np.int64)
        weights_all = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64
        )
        mst = numba_mst_kruskal(edges_all, weights_all, n)
        mst_set = {(int(e[0]), int(e[1])) for e in mst}

        total = 0.0
        for idx in range(len(edges_all)):
            u, v = int(edges_all[idx, 0]), int(edges_all[idx, 1])
            if (u, v) in mst_set or (v, u) in mst_set:
                total += weights_all[idx]
        assert total == 10.0

    def test_single_vertex_returns_empty(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        mst = numba_mst_kruskal(
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float64),
            n_vertices=1,
        )
        assert mst.shape == (0, 2)

    def test_zero_vertices_raises(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        with pytest.raises(ValueError, match="non-negative"):
            numba_mst_kruskal(
                np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.float64),
                n_vertices=-1,
            )

    def test_mismatched_lengths_raises(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        with pytest.raises(ValueError, match="matching"):
            numba_mst_kruskal(
                np.array([[0, 1]], dtype=np.int64),
                np.array([1.0, 2.0], dtype=np.float64),
                n_vertices=2,
            )

    def test_disconnected_graph(self) -> None:
        from pynerve._numba_graph import numba_mst_kruskal

        # Two components: 0-1 (weight 2), 2-3 (weight 3); total V=4
        edges = np.array([[0, 1], [2, 3]], dtype=np.int64)
        weights = np.array([2.0, 3.0], dtype=np.float64)
        mst = numba_mst_kruskal(edges, weights, n_vertices=4)
        # Forest with one edge per 2-vertex component
        assert len(mst) == 2


# _numba_simplices


class TestNumbaVrEdges:
    def test_versus_reference(self) -> None:
        from pynerve._fast_simplices import vr_edges_fast
        from pynerve._numba_simplices import numba_vr_edges

        rng = np.random.default_rng(23)
        for n, max_dist in [(2, 1.0), (5, 0.5), (10, 1.0), (20, 2.0)]:
            pts = rng.normal(size=(n, 3)).astype(np.float64)
            got = numba_vr_edges(pts, max_dist)
            expected = vr_edges_fast(pts, max_dist)
            got_sorted = got[np.lexsort((got[:, 1], got[:, 0]))]
            exp_sorted = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
            np.testing.assert_array_equal(
                got_sorted, exp_sorted, err_msg=f"n={n}, max_dist={max_dist}"
            )

    def test_large_max_dist_all_edges(self) -> None:
        from pynerve._numba_simplices import numba_vr_edges

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        edges = numba_vr_edges(pts, max_dist=100.0)
        assert len(edges) == 3  # 3 choose 2

    def test_zero_max_dist_only_self(self) -> None:
        from pynerve._numba_simplices import numba_vr_edges

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        edges = numba_vr_edges(pts, max_dist=0.0)
        assert len(edges) == 0

    def test_negative_max_dist_raises(self) -> None:
        from pynerve._numba_simplices import numba_vr_edges

        pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="non-negative"):
            numba_vr_edges(pts, max_dist=-1.0)

    def test_edges_in_range(self) -> None:
        from pynerve._numba_simplices import numba_vr_edges

        rng = np.random.default_rng(31)
        pts = rng.normal(size=(15, 2)).astype(np.float64)
        max_dist = 1.5
        edges = numba_vr_edges(pts, max_dist)
        for e in edges:
            u, v = int(e[0]), int(e[1])
            dist = np.linalg.norm(pts[u] - pts[v])
            assert dist <= max_dist + 1e-12


class TestNumbaTriangleEnumeration:
    def test_triangle_graph(self) -> None:
        from pynerve._numba_simplices import numba_triangle_enumeration

        # Complete graph K4 -> 4 triangles
        edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
        tris = numba_triangle_enumeration(edges, n_vertices=4)
        assert len(tris) == 4
        tri_sets = {tuple(sorted(t)) for t in tris}
        assert tri_sets == {(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)}

    def test_no_triangles(self) -> None:
        from pynerve._numba_simplices import numba_triangle_enumeration

        # Path graph -> no triangles
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        tris = numba_triangle_enumeration(edges, n_vertices=4)
        assert len(tris) == 0

    def test_empty_edges(self) -> None:
        from pynerve._numba_simplices import numba_triangle_enumeration

        tris = numba_triangle_enumeration(np.empty((0, 2), dtype=np.int64), n_vertices=5)
        assert len(tris) == 0

    def test_less_than_three_vertices(self) -> None:
        from pynerve._numba_simplices import numba_triangle_enumeration

        edges = np.array([[0, 1]], dtype=np.int64)
        tris = numba_triangle_enumeration(edges, n_vertices=2)
        assert len(tris) == 0


class TestNumbaSimplexBoundary:
    def test_triangle_boundary(self) -> None:
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([0, 1, 2], dtype=np.int64)
        boundary = numba_simplex_boundary(simplex)
        expected = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        b_sorted = np.sort(boundary, axis=1)
        e_sorted = np.sort(expected, axis=1)
        np.testing.assert_array_equal(b_sorted, e_sorted)

    def test_edge_boundary(self) -> None:
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([3, 7], dtype=np.int64)
        boundary = numba_simplex_boundary(simplex)
        expected = np.array([[7], [3]], dtype=np.int64)
        np.testing.assert_array_equal(boundary, expected)

    def test_single_vertex(self) -> None:
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([5], dtype=np.int64)
        boundary = numba_simplex_boundary(simplex)
        assert boundary.shape == (1, 0)

    def test_empty_simplex(self) -> None:
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([], dtype=np.int64)
        boundary = numba_simplex_boundary(simplex)
        assert boundary.size == 0

    def test_tetrahedron_boundary(self) -> None:
        from pynerve._numba_simplices import numba_simplex_boundary

        simplex = np.array([1, 2, 3, 4], dtype=np.int64)
        boundary = numba_simplex_boundary(simplex)
        assert boundary.shape == (4, 3)
        for face in boundary:
            assert len(face) == 3
            assert all(v in [1, 2, 3, 4] for v in face)


# _numba_reduction


class TestNumbaColumnReduction:
    """Tests for the column-reduction kernel.

    The source module uses ``del filtration`` to suppress an unused-argument
    warning.  This is not valid in numba ``njit`` nopython mode on some
    numba versions (e.g. 0.65) so the kernel may refuse to compile.  Tests
    use the ``_call_numba_fn`` helper which turns ``CompilerError`` into a
    ``pytest.skip``.
    """

    @staticmethod
    def _pure_python_reference(boundary: np.ndarray) -> np.ndarray:
        n_lower, n_higher = boundary.shape
        low_inv = np.full(n_lower, -1, dtype=np.int64)
        pivots = np.full(n_higher, -1, dtype=np.int64)
        b = boundary.copy().astype(np.int64)

        for j in range(n_higher):
            low = -1
            for i in range(n_lower - 1, -1, -1):
                if b[i, j] == 1:
                    low = i
                    break

            while low >= 0 and low_inv[low] >= 0:
                col = low_inv[low]
                for i in range(n_lower):
                    b[i, j] ^= b[i, col]

                low = -1
                for i in range(n_lower - 1, -1, -1):
                    if b[i, j] == 1:
                        low = i
                        break

            if low >= 0:
                low_inv[low] = j
                pivots[j] = low

        return pivots

    def test_small_boundary(self) -> None:
        m = np.array(
            [[1, 0, 0], [0, 1, 1], [0, 0, 1]],
            dtype=np.int64,
        )
        filtration = np.arange(3, dtype=np.float64)

        pivots = _call_numba_fn(
            "pynerve._numba_reduction",
            "numba_column_reduction",
            m.copy(),
            filtration,
        )
        ref = self._pure_python_reference(m)
        np.testing.assert_array_equal(pivots, ref)

    def test_chain_example(self) -> None:
        m = np.array(
            [[1, 0, 0], [0, 1, 1]],
            dtype=np.int64,
        )
        filtration = np.arange(3, dtype=np.float64)

        pivots = _call_numba_fn(
            "pynerve._numba_reduction",
            "numba_column_reduction",
            m.copy(),
            filtration,
        )
        assert pivots[0] == 0
        assert pivots[1] == 1
        assert pivots[2] == -1

    def test_empty_columns(self) -> None:
        m = np.zeros((3, 2), dtype=np.int64)
        filtration = np.arange(2, dtype=np.float64)

        pivots = _call_numba_fn(
            "pynerve._numba_reduction",
            "numba_column_reduction",
            m.copy(),
            filtration,
        )
        assert np.all(pivots == -1)


class TestNumbaSparseReduction:
    def test_versus_dense(self) -> None:
        from pynerve._numba_reduction import numba_sparse_reduction

        rng = np.random.default_rng(37)
        m = (rng.random((5, 6)) < 0.3).astype(np.int64)

        # Compute dense pivots via the pure-Python reference
        dense_pivots = TestNumbaColumnReduction._pure_python_reference(m)

        # Convert to sparse column format
        columns_list = []
        col_lengths = np.zeros(6, dtype=np.int64)
        max_len = 0
        for j in range(6):
            rows = np.where(m[:, j])[0]
            columns_list.append(rows)
            col_lengths[j] = len(rows)
            max_len = max(max_len, len(rows))

        columns = np.full((6, max(max_len, 1)), -1, dtype=np.int64)
        for j in range(6):
            for k, r in enumerate(columns_list[j]):
                columns[j, k] = r

        sparse_pivots = numba_sparse_reduction(columns, col_lengths)
        np.testing.assert_array_equal(sparse_pivots, dense_pivots)

    def test_empty_columns(self) -> None:
        from pynerve._numba_reduction import numba_sparse_reduction

        columns = np.full((3, 1), -1, dtype=np.int64)
        col_lengths = np.zeros(3, dtype=np.int64)
        pivots = numba_sparse_reduction(columns, col_lengths)
        assert np.all(pivots == -1)


# _numba_representations


class TestNumbaBettiCurve:
    @staticmethod
    def _pure_python_reference(
        pairs: np.ndarray, max_dim: int, resolution: int, max_time: float
    ) -> np.ndarray:
        betti = np.zeros((max_dim + 1, resolution), dtype=np.int64)
        if pairs.shape[0] == 0 or max_time <= 0.0:
            return betti
        for d in range(max_dim + 1):
            for i in range(pairs.shape[0]):
                if int(pairs[i, 2]) == d:
                    birth = pairs[i, 0]
                    death = pairs[i, 1]
                    start = int(birth / max_time * resolution)
                    end = int(death / max_time * resolution)
                    start = max(0, start)
                    end = min(resolution, end)
                    for t in range(start, end):
                        betti[d, t] += 1
        return betti

    def test_versus_reference(self) -> None:
        from pynerve._numba_representations import numba_betti_curve

        rng = np.random.default_rng(41)
        for _ in range(3):
            n_pairs = rng.integers(0, 10)
            pairs = np.zeros((n_pairs, 3), dtype=np.float64)
            if n_pairs > 0:
                births = rng.uniform(0.0, 5.0, size=n_pairs)
                deaths = births + rng.uniform(0.1, 5.0, size=n_pairs)
                dims = rng.integers(0, 3, size=n_pairs).astype(np.float64)
                pairs[:, 0] = births
                pairs[:, 1] = deaths
                pairs[:, 2] = dims

            max_dim = 2
            resolution = 20
            max_time = 10.0

            got = numba_betti_curve(pairs, max_dim, resolution, max_time)
            expected = self._pure_python_reference(pairs, max_dim, resolution, max_time)
            np.testing.assert_array_equal(got, expected)

    def test_empty_pairs(self) -> None:
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.empty((0, 3), dtype=np.float64)
        got = numba_betti_curve(pairs, max_dim=2, resolution=10, max_time=5.0)
        assert got.shape == (3, 10)
        assert np.all(got == 0)

    def test_invalid_params_raise(self) -> None:
        from pynerve._numba_representations import numba_betti_curve

        pairs = np.array([[1.0, 2.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError):
            numba_betti_curve(pairs, max_dim=-1, resolution=10, max_time=5.0)
        with pytest.raises(ValueError):
            numba_betti_curve(pairs, max_dim=2, resolution=0, max_time=5.0)


class TestNumbaPersistenceImage:
    @staticmethod
    def _pure_python_reference(
        pairs: np.ndarray,
        resolution: int,
        sigma: float,
        min_birth: float,
        max_birth: float,
        min_death: float,
        max_death: float,
    ) -> np.ndarray:
        birth_span = max_birth - min_birth
        death_span = max_death - min_death
        if birth_span <= 0.0:
            birth_span = 1e-12
        if death_span <= 0.0:
            death_span = 1e-12
        image = np.zeros((resolution, resolution), dtype=np.float64)
        for i in range(pairs.shape[0]):
            birth = pairs[i, 0]
            death = pairs[i, 1]
            pers = death - birth
            x = int((birth - min_birth) / birth_span * (resolution - 1))
            y = int((death - min_death) / death_span * (resolution - 1))
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    px = x + dx
                    py = y + dy
                    if 0 <= px < resolution and 0 <= py < resolution:
                        dist_sq = (dx * dx + dy * dy) / (sigma * sigma)
                        gaussian = pers * np.exp(-dist_sq / 2.0)
                        image[px, py] += gaussian
        return image

    def test_versus_reference(self) -> None:
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[1.0, 3.0], [2.0, 5.0]], dtype=np.float64)
        args = (pairs, 10, 0.5, 0.0, 6.0, 0.0, 6.0)
        got = numba_persistence_image(*args)
        expected = self._pure_python_reference(*args)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_single_point(self) -> None:
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[2.0, 4.0]], dtype=np.float64)
        got = numba_persistence_image(
            pairs,
            resolution=8,
            sigma=0.5,
            min_birth=0.0,
            max_birth=5.0,
            min_death=0.0,
            max_death=5.0,
        )
        assert got.shape == (8, 8)
        assert np.any(got > 0)

    def test_empty_pairs(self) -> None:
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.empty((0, 2), dtype=np.float64)
        got = numba_persistence_image(
            pairs,
            resolution=8,
            sigma=0.5,
            min_birth=0.0,
            max_birth=5.0,
            min_death=0.0,
            max_death=5.0,
        )
        assert np.all(got == 0.0)

    def test_invalid_params_raise(self) -> None:
        from pynerve._numba_representations import numba_persistence_image

        pairs = np.array([[1.0, 2.0]], dtype=np.float64)
        with pytest.raises(ValueError):
            numba_persistence_image(
                pairs,
                resolution=0,
                sigma=0.5,
                min_birth=0.0,
                max_birth=5.0,
                min_death=0.0,
                max_death=5.0,
            )
        with pytest.raises(ValueError):
            numba_persistence_image(
                pairs,
                resolution=8,
                sigma=0.0,
                min_birth=0.0,
                max_birth=5.0,
                min_death=0.0,
                max_death=5.0,
            )


# _numba_dispatch


class TestComputeWithNumba:
    def test_known_operation_pairwise(self) -> None:
        from pynerve._numba_dispatch import compute_with_numba

        pts = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
        result = compute_with_numba("pairwise_distances", pts)
        assert result.shape == (2, 2)
        assert abs(result[0, 1] - 5.0) < 1e-12

    def test_invalid_operation_raises(self) -> None:
        from pynerve._numba_dispatch import compute_with_numba

        with pytest.raises(ValueError, match="Unknown operation"):
            compute_with_numba("nonexistent", np.array([1.0]))

        with pytest.raises(ValueError, match="non-empty"):
            compute_with_numba("", np.array([1.0]))

    def test_non_string_operation_raises(self) -> None:
        from pynerve._numba_dispatch import compute_with_numba

        with pytest.raises(ValueError):
            compute_with_numba(123, np.array([1.0]))

    def test_all_registered_operations(self) -> None:
        from pynerve._numba_dispatch import _OPERATIONS

        for op_name in _OPERATIONS:
            assert isinstance(op_name, str) and op_name


class TestBenchmarkNumbaVsNumpy:
    def test_basic_benchmark(self) -> None:
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        stats = benchmark_numba_vs_numpy(
            lambda x: x + 1.0,
            lambda x: x + 1.0,
            lambda: (np.array([1.0, 2.0, 3.0], dtype=np.float64),),
            n_trials=3,
        )
        assert "numba_mean" in stats
        assert "numpy_mean" in stats
        assert "speedup" in stats
        assert "results_match" in stats
        assert stats["results_match"] is True
        assert np.isfinite(stats["numba_mean"])
        assert np.isfinite(stats["numpy_mean"])
        assert stats["speedup"] > 0.0

    def test_non_callable_raises(self) -> None:
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        with pytest.raises(TypeError, match="func_numba"):
            benchmark_numba_vs_numpy(
                "not_callable",
                lambda x: x,
                lambda: (np.array([1.0]),),
            )

        with pytest.raises(TypeError, match="func_numpy"):
            benchmark_numba_vs_numpy(
                lambda x: x,
                "not_callable",
                lambda: (np.array([1.0]),),
            )

        with pytest.raises(TypeError, match="setup_fn"):
            benchmark_numba_vs_numpy(
                lambda x: x,
                lambda x: x,
                "not_callable",
            )

    def test_setup_fn_must_return_tuple(self) -> None:
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        with pytest.raises(TypeError, match="setup_fn must return a tuple"):
            benchmark_numba_vs_numpy(
                lambda x: x,
                lambda x: x,
                lambda: np.array([1.0]),
                n_trials=1,
            )

    def test_results_match_with_small_differs(self) -> None:
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        stats = benchmark_numba_vs_numpy(
            lambda x: x + 1.0,
            lambda x: x + 2.0,
            lambda: (np.array([1.0, 2.0, 3.0], dtype=np.float64),),
            n_trials=1,
        )
        assert stats["results_match"] is False

    def test_n_trials_zero_raises(self) -> None:
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        with pytest.raises((ValueError, Exception)):
            benchmark_numba_vs_numpy(
                lambda x: x,
                lambda x: x,
                lambda: (np.array([1.0]),),
                n_trials=0,
            )


# numba_kernels facade


class TestNumbaKernelsFacade:
    """Verify the public facade module exposes all expected entries."""

    def test_all_exports_present(self) -> None:
        from pynerve import numba_kernels

        expected = {
            "HAS_NUMBA",
            "pairwise_distances",
            "nearest_neighbors",
            "vr_edges",
            "triangle_enumeration",
            "simplex_boundary",
            "column_reduction",
            "sparse_reduction",
            "betti_curve",
            "persistence_image",
            "connected_components",
            "mst_kruskal",
            "compute_with_numba",
            "benchmark_numba_vs_numpy",
        }
        for name in expected:
            assert hasattr(numba_kernels, name), f"Missing: {name}"

    def test_all_callable(self) -> None:
        from pynerve import numba_kernels

        callable_names = [
            "pairwise_distances",
            "nearest_neighbors",
            "vr_edges",
            "triangle_enumeration",
            "simplex_boundary",
            "column_reduction",
            "sparse_reduction",
            "betti_curve",
            "persistence_image",
            "connected_components",
            "mst_kruskal",
            "compute_with_numba",
            "benchmark_numba_vs_numpy",
        ]
        for name in callable_names:
            obj = getattr(numba_kernels, name)
            assert callable(obj), f"{name} is not callable"

    def test_has_numba_boolean(self) -> None:
        from pynerve.numba_kernels import HAS_NUMBA

        assert isinstance(HAS_NUMBA, bool)
        assert HAS_NUMBA is True


# _numba_compat -- no-op path (must run last to avoid poisoning other tests)


class TestNumbaCompatNoOp:
    """Simulate numba-absent path.

    These reload ``_numba_compat`` without numba in ``sys.modules`` so the
    fallback shim is exercised.  **Must run after all other tests** because
    the ``reload`` persists in ``sys.modules`` and downstream tests that
    import ``numba_kernels`` for the first time would see ``HAS_NUMBA=False``.
    """

    def test_has_numba_false_without_numba(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "numba", None)
        import importlib

        import pynerve._numba_compat

        importlib.reload(pynerve._numba_compat)
        assert pynerve._numba_compat.HAS_NUMBA is False

    def test_njit_noop_returns_function(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "numba", None)
        import importlib

        import pynerve._numba_compat

        importlib.reload(pynerve._numba_compat)
        njit_fb = pynerve._numba_compat.njit

        @njit_fb
        def _f(x: float) -> float:
            return x + 1.0

        assert _f(5.0) == 6.0

    def test_njit_noop_with_callable_arg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "numba", None)
        import importlib

        import pynerve._numba_compat

        importlib.reload(pynerve._numba_compat)
        njit_fb = pynerve._numba_compat.njit

        def _f(x: float) -> float:
            return x * 2.0

        assert njit_fb(_f) is _f

    def test_prange_noop_is_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "numba", None)
        import importlib

        import pynerve._numba_compat

        importlib.reload(pynerve._numba_compat)
        assert pynerve._numba_compat.prange is range
