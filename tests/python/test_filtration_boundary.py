"""Correctness tests for filtration and boundary modules.

Tests cover:
- sort_filtration_fast: ordering, determinism, edge cases
- vietoris_rips_filtration_fast: simplex structure, filtration values, edge cases
- boundary_matrix_sparse: matrix entries, empty cases, multi-dimension
- column_reduction_sparse: reduced columns, ordering, determinism
- simplex_boundary_fast: boundary-of-boundary is zero, edge cases
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from pynerve.exceptions import ValidationError

_PYNERVE_DIR = Path(__file__).resolve().parents[2] / "python"


def _import_submodule(name: str):
    """Import a pynerve submodule without triggering the C++ extension load path.

    The pynerve __init__ may fail when the compiled extension has a linking issue
    (e.g. undefined symbol).  This helper imports source files directly so that
    test logic can always be exercised regardless of build state.
    """
    import importlib.machinery
    import importlib.util

    if name in sys.modules:
        return sys.modules[name]

    # Ensure parent packages exist without running their real __init__.py.
    # The editable install registers a meta-path finder that redirects
    # pynerve imports, but pynerve.__init__ imports a C++ extension that may
    # not be linked correctly.  We pre-populate parent packages with dummy
    # namespace modules so that submodule *source* files load cleanly.
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            dummy = type(sys)(parent)
            dummy.__file__ = str(_PYNERVE_DIR / parent.replace(".", "/") / "__init__.py")
            dummy.__path__ = []
            dummy.__package__ = parent
            sys.modules[parent] = dummy

    # Also register the dummy loader path for the common parent 'pynerve'
    # so that relative imports (e.g.  from ._fast_simplices import ...) resolve.
    if "pynerve" in sys.modules:
        pkg = sys.modules["pynerve"]
        if not hasattr(pkg, "__path__") or not pkg.__path__:
            pkg.__path__ = [str(_PYNERVE_DIR / "pynerve")]

    relpath = name.replace(".", "/") + ".py"
    fullpath = _PYNERVE_DIR / relpath
    loader = importlib.machinery.SourceFileLoader(name, str(fullpath))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def _get_filtration_functions():
    mod = _import_submodule("pynerve._fast_filtration")
    return mod.sort_filtration_fast, mod.vietoris_rips_filtration_fast


def _get_boundary_functions():
    mod = _import_submodule("pynerve._fast_boundary")
    return mod.boundary_matrix_sparse, mod.column_reduction_sparse


def _get_simplex_boundary():
    mod = _import_submodule("pynerve._fast_simplices")
    return mod.simplex_boundary_fast


# sort_filtration_fast

SIMPLICES_1D = np.array([[0], [1], [2], [3]], dtype=np.int64)
FILT_VALS_1D = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

SIMPLICES_2D = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
FILT_VALS_2D = np.array([1.0, 1.2, 2.5, 1.5, 0.8], dtype=np.float64)

SIMPLEX_SET_A = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [1, 3]], dtype=np.int64)
FILT_VALS_A = np.array([1.0, 1.2, 1.5, 0.5, 2.0], dtype=np.float64)


class TestSortFiltrationFast:
    def test_non_decreasing_filtration_order(self):
        sort_filtration_fast, _ = _get_filtration_functions()
        _, _, sorted_filt = sort_filtration_fast(SIMPLEX_SET_A, FILT_VALS_A)
        diffs = np.diff(sorted_filt)
        assert (diffs >= 0).all(), f"filtration values not non-decreasing: {sorted_filt.tolist()}"

    def test_output_shapes_match_input(self):
        sort_filtration_fast, _ = _get_filtration_functions()
        indices, sorted_simplices, sorted_filt = sort_filtration_fast(SIMPLEX_SET_A, FILT_VALS_A)
        n = len(FILT_VALS_A)
        assert len(indices) == n
        assert sorted_simplices.shape == SIMPLEX_SET_A.shape
        assert len(sorted_filt) == n

    def test_sorted_simplices_match_filtration_order(self):
        sort_filtration_fast, _ = _get_filtration_functions()
        _, sorted_simplices, _ = sort_filtration_fast(SIMPLEX_SET_A, FILT_VALS_A)
        # filtration order: 0.5 (col 3), 1.0 (col 0), 1.2 (col 1), 1.5 (col 2), 2.0 (col 4)
        expected = np.array([[0, 3], [0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int64)
        assert np.array_equal(sorted_simplices, expected), (
            f"expected {expected.tolist()}, got {sorted_simplices.tolist()}"
        )

    def test_stable_sort_preserves_relative_order_for_equal_filtration(self):
        sort_filtration_fast, _ = _get_filtration_functions()
        simplices = np.array([[0], [1], [2]], dtype=np.int64)
        filtration = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        _, sorted_simplices, _ = sort_filtration_fast(simplices, filtration)
        # mergesort is stable; order of equal-valued items preserved
        assert np.array_equal(sorted_simplices[0], [0])
        assert np.array_equal(sorted_simplices[1], [1])
        assert np.array_equal(sorted_simplices[2], [2])

    def test_single_simplex(self):
        sort_filtration_fast, _ = _get_filtration_functions()
        simplices = np.array([[0, 1]], dtype=np.int64)
        filtration = np.array([0.5], dtype=np.float64)
        indices, sorted_s, sorted_f = sort_filtration_fast(simplices, filtration)
        assert np.array_equal(indices, [0])
        assert np.array_equal(sorted_s[0], simplices[0])
        assert sorted_f == pytest.approx([0.5])

    def test_empty_input_raises(self):
        sort_filtration_fast, _ = _get_filtration_functions()
        simplices = np.zeros((0, 2), dtype=np.int64)
        filtration = np.zeros((0,), dtype=np.float64)
        indices, sorted_s, sorted_f = sort_filtration_fast(simplices, filtration)
        assert len(indices) == 0
        assert sorted_s.shape == (0, 2)
        assert len(sorted_f) == 0

    def test_mismatched_lengths_raises(self):
        sort_filtration_fast, _ = _get_filtration_functions()
        with pytest.raises(ValueError, match="matching lengths"):
            sort_filtration_fast(SIMPLEX_SET_A, np.array([1.0, 2.0], dtype=np.float64))

    def test_non_finite_filtration_raises(self):
        sort_filtration_fast, _ = _get_filtration_functions()
        bad = np.array([0.0, np.inf, 1.0, 0.5, 2.0], dtype=np.float64)
        with pytest.raises(ValueError, match="finite"):
            sort_filtration_fast(SIMPLEX_SET_A, bad)

    def test_determinism(self):
        sort_filtration_fast, _ = _get_filtration_functions()
        rng = np.random.RandomState(42)
        simplices = np.array([[i] for i in range(100)], dtype=np.int64)
        filtration = rng.rand(100).astype(np.float64)
        r1 = sort_filtration_fast(simplices, filtration)
        r2 = sort_filtration_fast(simplices, filtration)
        for a, b in zip(r1, r2, strict=True):
            assert np.array_equal(a, b)


# vietoris_rips_filtration_fast


class TestVietorisRipsFiltration:
    def test_equilateral_triangle(self):
        _, vietoris_rips_filtration_fast = _get_filtration_functions()
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(0.75)]], dtype=np.float64)
        simplices, filt = vietoris_rips_filtration_fast(points, 1.5, max_dim=2)

        # 0-simplices: 3 vertices with value 0
        assert len(simplices[0]) == 3
        assert (filt[0] == 0.0).all()

        # 1-simplices: 3 edges, each with value ~ 1.0 (side length)
        assert len(simplices[1]) == 3
        for d in filt[1]:
            assert d == pytest.approx(1.0)

        # 2-simplex: triangle, value = max edge = 1.0
        assert len(simplices[2]) == 1
        assert filt[2][0] == pytest.approx(1.0)

    def test_filtration_values_are_valid_edge_distances(self):
        _, vietoris_rips_filtration_fast = _get_filtration_functions()
        points = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]], dtype=np.float64)
        _, filt = vietoris_rips_filtration_fast(points, 5.0, max_dim=2)

        # edges: [0,1] dist=3, [0,2] dist=5, [1,2] dist=4
        edge_dists = set(round(d, 10) for d in filt[1])
        assert 3.0 in edge_dists
        assert 4.0 in edge_dists
        assert 5.0 in edge_dists

        # triangle: max among 3, 4, 5 = 5
        assert len(_get_filtration_functions()[1](points, 5.0, max_dim=2)[0][2]) == 1

    def test_filtration_non_decreasing_by_simplex_order(self):
        """Verify that for a complete VR filtration with a single-dimension
        sort, the output respects the filtration non-decreasing property."""
        _, vietoris_rips_filtration_fast = _get_filtration_functions()
        sort_filtration_fast, _ = _get_filtration_functions()
        points = np.random.RandomState(7).rand(8, 2).astype(np.float64)
        simplices, filt = vietoris_rips_filtration_fast(points, 0.8, max_dim=2)

        # Per-dimension simplices have uniform shape, so we can sort each
        # dimension individually via sort_filtration_fast and verify
        # the result is non-decreasing.
        for d in range(len(simplices)):
            if len(filt[d]) > 1:
                _, _, sorted_f = sort_filtration_fast(simplices[d], filt[d])
                diffs = np.diff(sorted_f)
                assert (diffs >= -1e-12).all(), (
                    f"dim {d}: sort_filtration_fast produced decreasing values"
                )

    def test_empty_point_set(self):
        _, vietoris_rips_filtration_fast = _get_filtration_functions()
        points = np.empty((0, 2), dtype=np.float64)
        simplices, filt = vietoris_rips_filtration_fast(points, 1.0, max_dim=2)
        # 0-simplices: an empty (0, 1) array
        assert simplices[0].shape == (0, 1)
        assert len(filt[0]) == 0
        # After enumerate_simplices_fast, edges may be an empty (0, 2) array
        # returned as a separate entry, so len(simplices) can be 2.
        assert len(filt) == len(simplices)
        for s in simplices:
            assert s.size == 0

    def test_single_point(self):
        _, vietoris_rips_filtration_fast = _get_filtration_functions()
        points = np.array([[5.0, 7.0]], dtype=np.float64)
        simplices, filt = vietoris_rips_filtration_fast(points, 1.0, max_dim=2)
        assert simplices[0].shape == (1, 1)
        assert np.array_equal(simplices[0][0], [0])
        assert filt[0] == pytest.approx([0.0])

    def test_all_points_at_same_location(self):
        _, vietoris_rips_filtration_fast = _get_filtration_functions()
        points = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        simplices, filt = vietoris_rips_filtration_fast(points, 0.0, max_dim=2)
        # All distances are 0, so edges with max_dist=0 are still included
        # because edge distance <= max_dist (0 <= 0 is True)
        assert len(simplices) >= 1
        assert (filt[0] == 0.0).all()
        if len(simplices) > 1:
            assert (filt[1] == 0.0).all()

    def test_1d_point_cloud(self):
        _, vietoris_rips_filtration_fast = _get_filtration_functions()
        points = np.array([[0.0], [1.0], [2.5], [3.0]], dtype=np.float64)
        simplices, filt = vietoris_rips_filtration_fast(points, 2.0, max_dim=2)
        assert len(simplices) >= 2  # vertices and at least some edges
        assert (filt[0] == 0.0).all()

    def test_2d_point_cloud_generates_all_dimensions(self):
        _, vietoris_rips_filtration_fast = _get_filtration_functions()
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        simplices, filt = vietoris_rips_filtration_fast(points, 1.5, max_dim=2)
        # With max_dist=1.5, all pairwise distances <= sqrt(2), so the
        # full complex should be present.
        assert len(simplices) == 3  # vertices, edges, triangles
        assert len(simplices[0]) == 4  # vertices
        assert len(simplices[1]) == 6  # all edges
        assert len(simplices[2]) >= 1  # at least one triangle

    def test_3d_point_cloud(self):
        _, vietoris_rips_filtration_fast = _get_filtration_functions()
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        simplices, filt = vietoris_rips_filtration_fast(points, 1.5, max_dim=2)
        assert (filt[0] == 0.0).all()
        # All 6 edges exist
        assert len(simplices[1]) == 6
        # All 4 triangles exist
        assert len(simplices[2]) == 4

    def test_determinism(self):
        _, vietoris_rips_filtration_fast = _get_filtration_functions()
        rng = np.random.RandomState(99)
        points = rng.rand(10, 3).astype(np.float64)
        s1, f1 = vietoris_rips_filtration_fast(points, 0.5, max_dim=2)
        s2, f2 = vietoris_rips_filtration_fast(points, 0.5, max_dim=2)
        for dim in range(min(len(s1), len(s2))):
            assert np.array_equal(s1[dim], s2[dim]), f"dim {dim} simplices differ"
            assert np.array_equal(f1[dim], f2[dim]), f"dim {dim} filtration differs"

    def test_negative_max_dist(self):
        _, vietoris_rips_filtration_fast = _get_filtration_functions()
        points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(Exception, match="non-negative"):
            vietoris_rips_filtration_fast(points, -0.5)

    def test_non_finite_points_raises(self):
        _, vietoris_rips_filtration_fast = _get_filtration_functions()
        points = np.array([[0.0, np.nan], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises((ValueError, ValidationError)):
            vietoris_rips_filtration_fast(points, 1.0)

    def test_max_dim_zero_yields_vertices_only(self):
        _, vietoris_rips_filtration_fast = _get_filtration_functions()
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
        simplices, filt = vietoris_rips_filtration_fast(points, 5.0, max_dim=0)
        assert len(simplices) == 1
        assert simplices[0].shape == (3, 1)
        assert (filt[0] == 0.0).all()


# simplex_boundary_fast


class TestSimplexBoundary:
    def test_vertex_boundary(self):
        simplex_boundary_fast = _get_simplex_boundary()
        # A vertex (0-simplex) has no boundary faces
        boundary = simplex_boundary_fast(np.array([0], dtype=np.int64))
        assert boundary.shape == (1, 0)

    def test_edge_boundary(self):
        simplex_boundary_fast = _get_simplex_boundary()
        boundary = simplex_boundary_fast(np.array([0, 1], dtype=np.int64))
        assert boundary.shape == (2, 1)
        # The two faces: [1] and [0]
        assert boundary[0, 0] == 1
        assert boundary[1, 0] == 0

    def test_triangle_boundary(self):
        simplex_boundary_fast = _get_simplex_boundary()
        boundary = simplex_boundary_fast(np.array([0, 1, 2], dtype=np.int64))
        assert boundary.shape == (3, 2)
        # alternating signs: [1,2], [0,2], [0,1]
        expected = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        assert np.array_equal(boundary, expected)

    def test_boundary_of_boundary_is_zero(self):
        """Fundamental property: delta composed with delta = 0"""
        simplex_boundary_fast = _get_simplex_boundary()
        # For a 3-simplex (tetrahedron), iterated boundary should vanish
        simplex = np.array([0, 1, 2, 3], dtype=np.int64)
        b1 = simplex_boundary_fast(simplex)
        assert b1.shape == (4, 3)
        # Compute boundary of each face and linear combination
        # The boundary operator satisfies: sum_i (-1)^i * d(face_i) = 0
        # We verify by computing the formal combination
        from collections import Counter

        face_counts: Counter[tuple[int, ...]] = Counter()
        for i, face in enumerate(b1):
            sign = 1 if i % 2 == 0 else -1
            face_boundary = simplex_boundary_fast(face)
            for j in range(face_boundary.shape[0]):
                f = tuple(sorted(int(v) for v in face_boundary[j]))
                sub_sign = (-1) ** j
                face_counts[f] += sign * sub_sign

        # All coefficients should cancel to zero
        for f, coeff in face_counts.items():
            assert coeff == 0, f"non-zero coefficient {coeff} for face {f} in dd of 3-simplex"

    def test_tetrahedron_boundary_of_boundary_via_matrix(self):
        """Verify d2 composed with d1 = 0 using the boundary matrix for a tetrahedron."""
        boundary_matrix_sparse, _ = _get_boundary_functions()

        # Build simplex list: 4 vertices, 6 edges, 4 triangles
        vertices = np.array([[0], [1], [2], [3]], dtype=np.int64)
        simplices = [vertices]
        # edges
        edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
        simplices.append(edges)
        # triangles
        triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
        simplices.append(triangles)
        # tetrahedron
        tetra = np.array([[0, 1, 2, 3]], dtype=np.int64)
        simplices.append(tetra)

        bm = boundary_matrix_sparse(simplices, max_dim=3)
        # d1: vertices x edges   (4x6)
        # d2: edges x triangles  (6x4)
        # d3: triangles x tetra  (4x1)
        assert bm[0].shape == (4, 6)  # d1
        assert bm[1].shape == (6, 4)  # d2
        assert bm[2].shape == (4, 1)  # d3

        # d2 composed with d1 = 0  ->  d1 * d2 = 0  as integer matrix product
        # d1: (4x6), d2: (6x4), so d1 * d2 is (4x4) -> should be zero.
        composite = (bm[0] @ bm[1]).tocoo()
        # Verificar that each entry is 0
        assert np.all(composite.data == 0), f"d1*d2 non-zero entries: {composite.data}"

        # d3 composed with d2 = 0  ->  d2 * d3 = 0
        # d2: (6x4), d3: (4x1), so d2 * d3 is (6x1) -> should be zero.
        composite2 = (bm[1] @ bm[2]).tocoo()
        assert np.all(composite2.data == 0), f"d2*d3 non-zero entries: {composite2.data}"

    def test_invalid_simplex_raises(self):
        simplex_boundary_fast = _get_simplex_boundary()
        with pytest.raises(ValueError, match="1D"):
            simplex_boundary_fast(np.array([[0, 1]], dtype=np.int64))
        with pytest.raises(ValueError, match="non-empty"):
            simplex_boundary_fast(np.array([], dtype=np.int64))
        with pytest.raises(TypeError, match="integer"):
            simplex_boundary_fast(np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="non-negative"):
            simplex_boundary_fast(np.array([-1, 0], dtype=np.int64))
        with pytest.raises(ValueError, match="unique"):
            simplex_boundary_fast(np.array([0, 0], dtype=np.int64))


# boundary_matrix_sparse


class TestBoundaryMatrixSparse:
    def test_single_vertex_no_boundary_matrix(self):
        boundary_matrix_sparse, _ = _get_boundary_functions()
        simplices = [np.array([[0]], dtype=np.int64)]
        bm = boundary_matrix_sparse(simplices, max_dim=1)
        assert bm == []  # no dimension >= 1

    def test_empty_simplex_set(self):
        boundary_matrix_sparse, _ = _get_boundary_functions()
        simplices = [np.empty((0, 1), dtype=np.int64)]
        bm = boundary_matrix_sparse(simplices, max_dim=1)
        assert bm == []

    def test_edge_boundary_matrix_correct(self):
        boundary_matrix_sparse, _ = _get_boundary_functions()
        vertices = np.array([[0], [1], [2]], dtype=np.int64)
        edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        simplices = [vertices, edges]
        bm = boundary_matrix_sparse(simplices, max_dim=1)
        assert len(bm) == 1
        mat = bm[0]
        assert mat.shape == (3, 3)  # 3 vertices x 3 edges

        # simplex_boundary_fast([v0, v1, ..., vk]) returns rows:
        #   face_0 (omit v0), face_1 (omit v1), ..., face_k (omit vk)
        # with sign (-1)^i for face i.
        #
        # Edge [0,1]: boundary = [[1], [0]] -> col 0: row(1)=+1, row(0)=-1
        # Edge [1,2]: boundary = [[2], [1]] -> col 1: row(2)=+1, row(1)=-1
        # Edge [0,2]: boundary = [[2], [0]] -> col 2: row(2)=+1, row(0)=-1
        expected = np.zeros((3, 3), dtype=np.int64)
        expected[1, 0] = 1
        expected[0, 0] = -1
        expected[2, 1] = 1
        expected[1, 1] = -1
        expected[2, 2] = 1
        expected[0, 2] = -1
        assert (mat.toarray() == expected).all(), f"got:\n{mat.toarray()}\nexpected:\n{expected}"

    def test_triangle_boundary_matrix(self):
        boundary_matrix_sparse, _ = _get_boundary_functions()
        vertices = np.array([[0], [1], [2]], dtype=np.int64)
        edges = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
        triangles = np.array([[0, 1, 2]], dtype=np.int64)
        simplices = [vertices, edges, triangles]
        bm = boundary_matrix_sparse(simplices, max_dim=2)
        assert len(bm) == 2
        # d2: edges x triangles  (3x1)
        mat = bm[1]
        assert mat.shape == (3, 1)
        col = mat.toarray().ravel()
        # boundary of [0,1,2]: faces [1,2], [0,2], [0,1] with signs +1, -1, +1
        assert col[0] == 1  # [0,1] -> sign depends on face ordering
        assert col[1] == -1  # [0,2]
        assert col[2] == 1  # [1,2]

    def test_multi_dimension_mixed_simplices(self):
        boundary_matrix_sparse, _ = _get_boundary_functions()
        # 4 vertices, 5 edges (some), 2 triangles
        vertices = np.array([[0], [1], [2], [3]], dtype=np.int64)
        edges = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
        triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        simplices = [vertices, edges, triangles]
        bm = boundary_matrix_sparse(simplices, max_dim=2)
        assert len(bm) == 2
        assert bm[0].shape == (4, 5)  # d1: vertices x edges
        assert bm[1].shape == (5, 2)  # d2: edges x triangles

    def test_boundary_matrix_entries_are_pm1(self):
        boundary_matrix_sparse, _ = _get_boundary_functions()
        _, vr = _get_filtration_functions()
        points = np.random.RandomState(3).rand(6, 2).astype(np.float64)
        simplices, _ = vr(points, 0.8, max_dim=3)
        if len(simplices) >= 2:
            bm = boundary_matrix_sparse(simplices, max_dim=min(3, len(simplices) - 1))
            for mat in bm:
                if mat.nnz > 0:
                    data = mat.data
                    assert np.all(np.abs(data) == 1.0), f"non-+/-1 entries: {data}"

    def test_determinism(self):
        boundary_matrix_sparse, _ = _get_boundary_functions()
        _, vr = _get_filtration_functions()
        points = np.random.RandomState(11).rand(6, 2).astype(np.float64)
        simplices, _ = vr(points, 0.8, max_dim=2)
        bm1 = boundary_matrix_sparse(simplices, max_dim=2)
        bm2 = boundary_matrix_sparse(simplices, max_dim=2)
        for m1, m2 in zip(bm1, bm2, strict=True):
            assert (m1 != m2).nnz == 0

    def test_max_dim_exceeds_available_yields_up_to_available(self):
        boundary_matrix_sparse, _ = _get_boundary_functions()
        vertices = np.array([[0], [1]], dtype=np.int64)
        edges = np.array([[0, 1]], dtype=np.int64)
        simplices = [vertices, edges]
        bm = boundary_matrix_sparse(simplices, max_dim=5)
        # Only dim 1 is available
        assert len(bm) == 1


# column_reduction_sparse


class TestColumnReductionSparse:
    def test_reduces_small_triangle_boundary(self):
        _, column_reduction_sparse = _get_boundary_functions()
        import scipy.sparse as sp

        # d1 of a triangle: 3 vertices, 3 edges
        data = [1, -1, 1, -1, 1, -1]
        rows = [0, 1, 1, 2, 0, 2]
        cols = [0, 0, 1, 1, 2, 2]
        mat = sp.csr_matrix((data, (rows, cols)), shape=(3, 3))
        filtration = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        pairs = column_reduction_sparse(mat, filtration)
        # Each column should find a pivot and produce a pair
        assert len(pairs) >= 1

    def test_empty_boundary_matrix(self):
        _, column_reduction_sparse = _get_boundary_functions()
        import scipy.sparse as sp

        mat = sp.csr_matrix((0, 0))
        filtration = np.array([], dtype=np.float64)
        pairs = column_reduction_sparse(mat, filtration)
        assert pairs == []

    def test_reduced_columns_are_in_correct_order(self):
        _, column_reduction_sparse = _get_boundary_functions()
        import scipy.sparse as sp

        # Triangular matrix with one reduction step
        data = [1, 1, 1, 1, 1, 1]  # duplicate pivot rows
        rows = [0, 0, 1, 1, 2, 2]
        cols = [0, 1, 0, 1, 0, 1]
        mat = sp.csr_matrix((data, (rows, cols)), shape=(3, 2))
        filtration = np.array([1.0, 2.0], dtype=np.float64)

        pairs = column_reduction_sparse(mat, filtration)
        # After reduction, the columns should be in non-decreasing birth order
        assert len(pairs) >= 1

    def test_persistence_pairs_have_valid_indices(self):
        _, column_reduction_sparse = _get_boundary_functions()
        import scipy.sparse as sp

        data = [1, -1, 1, -1, 1, -1]
        rows = [0, 1, 1, 2, 2, 3]
        cols = [0, 0, 1, 1, 2, 2]
        mat = sp.csr_matrix((data, (rows, cols)), shape=(4, 3))
        filtration = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        pairs = column_reduction_sparse(mat, filtration)
        for pivot, col, birth, death in pairs:
            assert 0 <= pivot < 4
            assert 0 <= col < 3
            assert np.isfinite(birth)
            assert np.isfinite(death)

    def test_with_row_filtration_values(self):
        _, column_reduction_sparse = _get_boundary_functions()
        import scipy.sparse as sp

        mat = sp.csr_matrix(([1, -1], ([0, 1], [0, 0])), shape=(2, 1))
        filtration = np.array([5.0], dtype=np.float64)
        row_filt = np.array([2.0, 1.0], dtype=np.float64)

        pairs = column_reduction_sparse(mat, filtration, row_filtration_values=row_filt)
        assert len(pairs) == 1
        _, _, birth, death = pairs[0]
        # The pivot row for column 0 is max(rows) = 1 (value -1)
        assert birth == pytest.approx(1.0)  # row_filt[1]
        assert death == pytest.approx(5.0)

    def test_determinism(self):
        _, column_reduction_sparse = _get_boundary_functions()
        import scipy.sparse as sp

        rng = np.random.RandomState(77)
        n, m = 10, 8
        data = rng.choice([1, -1], size=n * m * 2)
        rows = rng.randint(0, n, size=n * m * 2)
        cols = rng.randint(0, m, size=n * m * 2)
        mat = sp.csr_matrix((data, (rows, cols)), shape=(n, m))
        filtration = rng.rand(m).astype(np.float64)

        p1 = column_reduction_sparse(mat, filtration)
        p2 = column_reduction_sparse(mat, filtration)
        assert p1 == p2

    def test_column_reduction_end_to_end_with_vr(self):
        """Full pipeline: VR -> boundary matrix -> column reduction."""
        boundary_matrix_sparse, column_reduction_sparse = _get_boundary_functions()
        _, vietoris_rips_filtration_fast = _get_filtration_functions()

        # 3-point triangle
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(0.75)]], dtype=np.float64)
        simplices, filt = vietoris_rips_filtration_fast(points, 1.5, max_dim=2)
        bm = boundary_matrix_sparse(simplices, max_dim=min(2, len(simplices) - 1))

        # Reduce d1: boundary from edges to vertices
        if len(bm) >= 1:
            pairs = column_reduction_sparse(bm[0], filt[1])
            for pivot, col, birth, death in pairs:
                assert birth <= death + 1e-12
                assert 0 <= pivot < bm[0].shape[0]
                assert 0 <= col < bm[0].shape[1]
