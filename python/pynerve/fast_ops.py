"""Compatibility facade for vectorized NumPy topology operations.

This module re-exports fast NumPy-based implementations of core topology
operations including distance computation, simplex enumeration, filtration
construction, persistence representations, and graph algorithms.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ._fast_boundary import (
    boundary_matrix_sparse as _boundary_matrix_sparse,
)
from ._fast_boundary import (
    column_reduction_sparse as _column_reduction_sparse,
)
from ._fast_distance import (
    nearest_neighbors_fast as _nearest_neighbors_fast,
)
from ._fast_distance import (
    pairwise_distances_broadcast as _pairwise_distances_broadcast,
)
from ._fast_distance import (
    pairwise_distances_fast as _pairwise_distances_fast,
)
from ._fast_distance import (
    sparse_distance_matrix as _sparse_distance_matrix,
)
from ._fast_filtration import (
    sort_filtration_fast as _sort_filtration_fast,
)
from ._fast_filtration import (
    vietoris_rips_filtration_fast as _vietoris_rips_filtration_fast,
)
from ._fast_graph import (
    connected_components_fast as _connected_components_fast,
)
from ._fast_graph import (
    minimum_spanning_tree_fast as _minimum_spanning_tree_fast,
)
from ._fast_representations import (
    betti_curve_fast as _betti_curve_fast,
)
from ._fast_representations import (
    persistence_image_fast as _persistence_image_fast,
)
from ._fast_representations import (
    persistence_landscape_fast as _persistence_landscape_fast,
)
from ._fast_simplices import (
    enumerate_simplices_fast as _enumerate_simplices_fast,
)
from ._fast_simplices import (
    simplex_boundary_fast as _simplex_boundary_fast,
)
from ._fast_simplices import (
    vr_edges_fast as _vr_edges_fast,
)


def pairwise_distances(points: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distances for a point cloud.

    :param points: Input point cloud of shape ``(n_points, n_dims)``.
    :param metric: Distance metric passed to ``scipy.spatial.distance.pdist``
        (default: ``"euclidean"``).
    :returns: Symmetric distance matrix of shape ``(n_points, n_points)``.
    :raises ValueError: If ``points`` is invalid or ``metric`` is empty.
    """
    return _pairwise_distances_fast(points, metric)


def pairwise_distances_broadcast(points: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances using broadcast vectorization.

    :param points: Input point cloud of shape ``(n_points, n_dims)``.
    :returns: Symmetric Euclidean distance matrix of shape
        ``(n_points, n_points)``.
    :raises ValueError: If ``points`` is not a 2-D array with finite values.
    """
    return _pairwise_distances_broadcast(points)


def nearest_neighbors(
    points: np.ndarray, k: int = 5, metric: str = "euclidean"
) -> tuple[np.ndarray, np.ndarray]:
    """Find k-nearest neighbors for each point.

    :param points: Input point cloud of shape ``(n_points, n_dims)``.
    :param k: Number of nearest neighbors (default: 5).
    :param metric: Distance metric (default: ``"euclidean"``).
    :returns: A tuple ``(distances, indices)`` where ``distances`` has shape
        ``(n_points, k)`` and ``indices`` has shape ``(n_points, k)``.
    :raises ValueError: If ``k`` is not positive, ``k > n_points``, or
        validation of ``points`` or ``metric`` fails.
    """
    return _nearest_neighbors_fast(points, k, metric)


def sparse_distance_matrix(
    points: np.ndarray, max_dist: float, output_type: str = "dense"
) -> np.ndarray:
    """Compute a sparse distance matrix with values below a threshold.

    :param points: Input point cloud of shape ``(n_points, n_dims)``.
    :param max_dist: Maximum distance threshold. Entries above this are
        excluded from the sparse output.
    :param output_type: Output format (default: ``"dense"``). One of
        ``"dense"``, ``"sparse"``, ``"coo"``, or ``"csr"``.
    :returns: Distance matrix. Shape ``(n_points, n_points)`` for ``"dense"``
        output; ``scipy.sparse.coo_matrix`` or ``scipy.sparse.csr_matrix``
        for sparse formats.
    :raises ValueError: If ``points`` is invalid, ``max_dist`` is negative
        or infinite, or ``output_type`` is unrecognized.
    """
    return _sparse_distance_matrix(points, max_dist, output_type)


def vr_edges(
    points: np.ndarray, max_dist: float, return_dists: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute Vietoris-Rips edges within a distance threshold.

    :param points: Input point cloud of shape ``(n_points, n_dims)``.
    :param max_dist: Maximum edge length. Edges longer than this are excluded.
    :param return_dists: If ``True``, also return edge distances
        (default: ``False``).
    :returns: If ``return_dists`` is ``False``, an ndarray of shape
        ``(n_edges, 2)`` with vertex index pairs. If ``True``, a tuple
        ``(edges, distances)``.
    :raises ValueError: If ``points`` is invalid or ``max_dist`` is negative.
    """
    return _vr_edges_fast(points, max_dist, return_dists)


def enumerate_simplices(points: np.ndarray, max_dist: float, max_dim: int = 2) -> list[np.ndarray]:
    """Enumerate all simplices up to a given dimension within a distance threshold.

    :param points: Input point cloud of shape ``(n_points, n_dims)``.
    :param max_dist: Maximum edge length for the Vietoris-Rips complex.
    :param max_dim: Maximum simplex dimension (default: 2).
    :returns: List of ndarrays, where element ``i`` contains simplices of
        dimension ``i``, each of shape ``(n_simplices_i, i + 1)``.
    :raises ValueError: If ``points`` is invalid or ``max_dim`` is negative.
    """
    return _enumerate_simplices_fast(points, max_dist, max_dim)


def simplex_boundary(simplex: np.ndarray) -> np.ndarray:
    """Compute the boundary of a simplex.

    :param simplex: Vertex indices of the simplex, shape ``(dim + 1,)``.
    :returns: Boundary faces of shape ``(dim + 1, dim)``, one face per row.
    :raises ValueError: If ``simplex`` is empty, not 1-D, contains negative
        indices, or has duplicate vertex entries.
    """
    return _simplex_boundary_fast(simplex)


def boundary_matrix_sparse(simplices: list[np.ndarray], max_dim: int = 2) -> list[csr_matrix]:
    """Build sparse boundary matrices for a filtered complex.

    :param simplices: List of ndarrays indexed by dimension, where element
        ``d`` has shape ``(n_simplices_d, d + 1)``.
    :param max_dim: Maximum dimension for which to build boundary matrices
        (default: 2). Must not exceed ``len(simplices) - 1``.
    :returns: List of ``scipy.sparse.csr_matrix`` objects, one per dimension
        from 1 up to ``max_dim``. The matrix for dimension ``d`` has shape
        ``(n_simplices_{d-1}, n_simplices_d)``.
    """
    return _boundary_matrix_sparse(simplices, max_dim)


def column_reduction_sparse(
    boundary_matrix: csr_matrix,
    filtration_values: np.ndarray,
    row_filtration_values: np.ndarray | None = None,
) -> list[tuple[int, int, float, float]]:
    """Perform column reduction on a sparse boundary matrix.

    Implements the standard persistence algorithm using the
    column-reduction (matrix) approach on a sparse boundary matrix.

    :param boundary_matrix: Sparse boundary matrix as a
        ``scipy.sparse.csr_matrix`` of shape ``(n_rows, n_cols)``.
    :param filtration_values: Filtration values for each column (simplex),
        shape ``(n_cols,)``.
    :param row_filtration_values: Optional filtration values for each row
        (default: ``None``). Used to compute birth times from pivot rows.
    :returns: List of persistence pairs, each as a tuple
        ``(pivot_row, column, birth_time, death_time)``.
    """
    return _column_reduction_sparse(boundary_matrix, filtration_values, row_filtration_values)


def sort_filtration(
    simplices: np.ndarray, filtration_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort simplices by filtration value.

    :param simplices: Input simplices of shape ``(n_simplices, dim + 1)``.
    :param filtration_values: Filtration values of shape ``(n_simplices,)``.
    :returns: A tuple ``(sorted_indices, sorted_simplices, sorted_filtration)``
        where ``sorted_indices`` are the indices that sort the filtration,
        ``sorted_simplices`` are the simplices in sorted order, and
        ``sorted_filtration`` are the corresponding filtration values.
    :raises ValueError: If the lengths do not match or filtration values
        contain non-finite entries.
    """
    return _sort_filtration_fast(simplices, filtration_values)


def vietoris_rips_filtration(
    points: np.ndarray, max_dist: float, max_dim: int = 2
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build a Vietoris-Rips filtration.

    :param points: Input point cloud of shape ``(n_points, n_dims)``.
    :param max_dist: Maximum edge length.
    :param max_dim: Maximum simplex dimension (default: 2).
    :returns: A tuple ``(simplices, filtration_values)`` where each element is
        a list of ndarrays indexed by dimension (0-simplices, 1-simplices, ...).
    :raises ValueError: If validation of ``points`` or ``max_dim`` fails.
    """
    return _vietoris_rips_filtration_fast(points, max_dist, max_dim)


def persistence_image(
    pairs: np.ndarray,
    resolution: int = 64,
    sigma: float = 0.1,
    weight_fn: str | None = None,
) -> np.ndarray:
    """Compute a persistence image from birth-death pairs.

    .. note::
        The canonical NumPy implementation is in :func:`pynerve._image_utils.persistence_image`
        which supports additional parameters (``birth_range``, ``persistence_range``, ``weight``).
        This variant uses ``weight_fn`` instead of ``weight`` -- same semantics, different name.

    :param pairs: Birth-death pairs of shape ``(n_pairs, N)`` where ``N >= 2``.
    :param resolution: Image resolution in pixels (default: 64).
    :param sigma: Gaussian kernel bandwidth (default: 0.1).
    :param weight_fn: Weighting function: ``None`` (squared persistence,
        default), ``"linear"``, or ``"constant"``.
    :returns: Persistence image of shape ``(resolution, resolution)``.
    :raises ValueError: If ``pairs`` is invalid, ``resolution`` is not positive,
        ``sigma`` is not finite and positive, or ``weight_fn`` is unrecognized.
    """
    return _persistence_image_fast(pairs, resolution, sigma, weight_fn)


def betti_curve(
    pairs: np.ndarray,
    max_dim: int = 3,
    resolution: int = 100,
    max_time: float | None = None,
) -> np.ndarray:
    """Compute Betti curves from persistence pairs.

    :param pairs: Persistence pairs of shape ``(n_pairs, N)`` where ``N >= 3``.
        Columns are birth, death, and homology dimension.
    :param max_dim: Maximum homology dimension (default: 3).
    :param resolution: Number of time steps (default: 100).
    :param max_time: Maximum time value (default: inferred from data).
    :returns: Betti numbers over time of shape ``(max_dim + 1, resolution)``,
        one row per homology dimension.
    :raises ValueError: If ``pairs`` is invalid, ``max_dim`` is negative,
        ``resolution`` is not positive, or ``max_time`` is invalid.
    """
    return _betti_curve_fast(pairs, max_dim, resolution, max_time)


def persistence_landscape(
    pairs: np.ndarray, n_layers: int = 5, resolution: int = 100
) -> np.ndarray:
    """Compute persistence landscapes from birth-death pairs.

    :param pairs: Birth-death pairs of shape ``(n_pairs, N)`` where ``N >= 2``.
    :param n_layers: Number of landscape layers (default: 5).
    :param resolution: Number of time steps (default: 100).
    :returns: Persistence landscape functions of shape
        ``(n_layers, resolution)``.
    :raises ValueError: If ``pairs`` is invalid, ``n_layers`` is not positive,
        or ``resolution`` is not positive.
    """
    return _persistence_landscape_fast(pairs, n_layers, resolution)


def connected_components(edges: np.ndarray, n_vertices: int) -> np.ndarray:
    """Compute connected components from an edge list.

    :param edges: Edge list of shape ``(n_edges, 2)`` with integer vertex
        indices.
    :param n_vertices: Total number of vertices.
    :returns: Component label for each vertex of shape ``(n_vertices,)``.
    :raises ValueError: If ``n_vertices`` is negative, ``edges`` do not have
        shape ``(m, 2)``, or edge indices are out of bounds.
    """
    return _connected_components_fast(edges, n_vertices)


def minimum_spanning_tree(points: np.ndarray, edges: np.ndarray | None = None) -> np.ndarray:
    """Compute the minimum spanning tree of a point cloud.

    :param points: Input point cloud of shape ``(n_points, n_dims)``.
    :param edges: Pre-computed edge list of shape ``(n_edges, 2)``, or
        ``None`` to compute all pairwise edges (default: ``None``).
    :returns: Edges in the minimum spanning tree of shape
        ``(n_mst_edges, 2)``.
    :raises ValueError: If validation of ``points`` or ``edges`` fails.
    """
    return _minimum_spanning_tree_fast(points, edges)


__all__ = [
    "_persistence_image_fast",
    "_persistence_landscape_fast",
    "pairwise_distances",
    "pairwise_distances_broadcast",
    "nearest_neighbors",
    "sparse_distance_matrix",
    "vr_edges",
    "enumerate_simplices",
    "simplex_boundary",
    "boundary_matrix_sparse",
    "column_reduction_sparse",
    "sort_filtration",
    "vietoris_rips_filtration",
    "persistence_image",
    "betti_curve",
    "persistence_landscape",
    "connected_components",
    "minimum_spanning_tree",
]
