"""Compatibility facade for optional Numba kernels."""

from __future__ import annotations

from ._numba_compat import HAS_NUMBA
from ._numba_dispatch import benchmark_numba_vs_numpy, compute_with_numba
from ._numba_distance import (
    numba_nearest_neighbors as _numba_nearest_neighbors,
)
from ._numba_distance import (
    numba_pairwise_distances as _numba_pairwise_distances,
)
from ._numba_graph import (
    numba_connected_components as _numba_connected_components,
)
from ._numba_graph import (
    numba_mst_kruskal as _numba_mst_kruskal,
)
from ._numba_reduction import (
    numba_column_reduction as _numba_column_reduction,
)
from ._numba_reduction import (
    numba_sparse_reduction as _numba_sparse_reduction,
)
from ._numba_representations import (
    numba_betti_curve as _numba_betti_curve,
)
from ._numba_representations import (
    numba_persistence_image as _numba_persistence_image,
)
from ._numba_simplices import (
    numba_simplex_boundary as _numba_simplex_boundary,
)
from ._numba_simplices import (
    numba_triangle_enumeration as _numba_triangle_enumeration,
)
from ._numba_simplices import (
    numba_vr_edges as _numba_vr_edges,
)

pairwise_distances = _numba_pairwise_distances
nearest_neighbors = _numba_nearest_neighbors
vr_edges = _numba_vr_edges
triangle_enumeration = _numba_triangle_enumeration
simplex_boundary = _numba_simplex_boundary
column_reduction = _numba_column_reduction
sparse_reduction = _numba_sparse_reduction
betti_curve = _numba_betti_curve
persistence_image = _numba_persistence_image
connected_components = _numba_connected_components
mst_kruskal = _numba_mst_kruskal

__all__ = [
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
]
