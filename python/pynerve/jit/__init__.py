"""Numba JIT compilation helpers for topology operations."""

from __future__ import annotations

import numpy as np

from .._validation import (
    validate_nonnegative_finite,
    validate_nonnegative_int,
    validate_positive_finite,
    validate_positive_int,
)
from ._cache import JITCache, cached_jit
from ._cpu_kernels import (
    _jit_batch_betti_curves_impl,
    _jit_betti_curve_impl,
    _jit_filter_pairs_impl,
    _jit_pairwise_distances_impl,
    _jit_persistence_image_impl,
    _jit_vietoris_rips_edges_impl,
)
from ._device import _resolve_device
from ._gpu_kernels import _gpu_pairwise_distances, _jit_persistence_image_gpu
from ._setup import cuda
from ._validate import _validate_pairs, _validate_points


def _jit_pairwise_distances(points: np.ndarray) -> np.ndarray:
    return _jit_pairwise_distances_impl(_validate_points(points))  # type: ignore[no-any-return]


def pairwise_distances(points: np.ndarray, device: str | None = None) -> np.ndarray:
    """Compute a pairwise Euclidean distance matrix.

    :param points: Point cloud of shape ``(n_points, dim)``.
    :param device: ``"cpu"`` (default) for CPU JIT, ``"cuda"`` for GPU
        (requires Numba CUDA). *None* defaults to CPU.
    :returns: Symmetric distance matrix of shape ``(n_points, n_points)``.
    :raises InvalidArgumentError: If points are not a valid 2-D array.
    :raises RuntimeError: If *device* is ``"cuda"`` but CUDA is unavailable.
    """
    points = _validate_points(points)
    _use_gpu = _resolve_device(device)
    if _use_gpu:
        assert cuda is not None
        n = points.shape[0]
        dists = np.zeros((n, n), dtype=np.float32)
        d_points = cuda.to_device(points)
        d_dists = cuda.to_device(dists)
        threadsperblock = (16, 16)
        blockspergrid = (
            (n + threadsperblock[0] - 1) // threadsperblock[0],
            (n + threadsperblock[1] - 1) // threadsperblock[1],
        )
        _gpu_pairwise_distances[blockspergrid, threadsperblock](d_points, d_dists)
        return d_dists.copy_to_host()  # type: ignore[no-any-return]
    return _jit_pairwise_distances_impl(points)  # type: ignore[no-any-return]


def filter_pairs(pairs: np.ndarray, threshold: float) -> np.ndarray:
    """Filter persistence pairs by a persistence (death - birth) threshold.

    :param pairs: Array of shape ``(n_pairs, at_least_2)`` with columns
        ``[birth, death, ...]``.
    :param threshold: Minimum persistence value to keep.
    :returns: Boolean mask of shape ``(n_pairs,)``.
    :raises InvalidArgumentError: If *pairs* is malformed.
    """
    result = _jit_filter_pairs_impl(
        _validate_pairs(pairs), validate_nonnegative_finite(threshold, "threshold")
    )
    return result  # type: ignore[no-any-return]


def betti_curve(pairs: np.ndarray, max_dim: int, resolution: int) -> np.ndarray:
    """Compute Betti numbers over an evenly sampled filtration.

    :param pairs: Array of shape ``(n_pairs, at_least_3)`` with columns
        ``[birth, death, dimension]``.
    :param max_dim: Maximum homology dimension to include.
    :param resolution: Number of filtration steps.
    :returns: Betti curve array of shape ``(max_dim + 1, resolution)``.
    :raises InvalidArgumentError: If *pairs* is malformed.
    """
    pairs = _validate_pairs(pairs, require_dim=True)
    max_dim = validate_nonnegative_int(max_dim, "max_dim")
    resolution = validate_positive_int(resolution, "resolution")
    if pairs.shape[0] == 0:
        return np.zeros((max_dim + 1, resolution), dtype=np.int32)
    return _jit_betti_curve_impl(pairs, max_dim, resolution)  # type: ignore[no-any-return]


def persistence_image(
    pairs: np.ndarray,
    resolution: int = 64,
    sigma: float = 0.1,
    device: str | None = None,
) -> np.ndarray:
    """Compute a persistence image on a square grid.

    .. note::
        The canonical NumPy implementation is in :func:`pynerve._image_utils.persistence_image`
        which supports additional parameters (``birth_range``, ``persistence_range``, ``weight``).
        This variant adds ``device`` support for GPU-offloaded computation via Numba CUDA.

    :param pairs: Persistence pairs of shape ``(n_pairs, at_least_2)`` with
        columns ``[birth, death, ...]``.
    :param resolution: Grid resolution (both axes).
    :param sigma: Gaussian bandwidth for the point-spread function.
    :param device: ``"cpu"`` (default) or ``"cuda"`` (requires Numba CUDA).
        *None* defaults to CPU.
    :returns: Persistence image of shape ``(resolution, resolution)``.
    :raises InvalidArgumentError: If *pairs* is malformed.
    :raises RuntimeError: If *device* is ``"cuda"`` but CUDA is unavailable.
    """
    pairs = _validate_pairs(pairs)
    resolution = validate_positive_int(resolution, "resolution")
    sigma = validate_positive_finite(sigma, "sigma")
    if pairs.shape[0] == 0:
        return np.zeros((resolution, resolution), dtype=np.float32)
    if _resolve_device(device):
        return _jit_persistence_image_gpu(pairs, resolution, sigma)
    return _jit_persistence_image_impl(pairs, resolution, sigma)  # type: ignore[no-any-return]


def vietoris_rips_edges(points: np.ndarray, max_dist: float) -> np.ndarray:
    """Enumerate Vietoris-Rips edges within a distance cutoff.

    :param points: Point cloud of shape ``(n_points, dim)``.
    :param max_dist: Maximum edge length to include.
    :returns: Integer array of shape ``(n_edges, 2)`` with vertex indices.
    :raises InvalidArgumentError: If *points* is malformed.
    """
    result = _jit_vietoris_rips_edges_impl(
        _validate_points(points), validate_nonnegative_finite(max_dist, "max_dist")
    )
    return result  # type: ignore[no-any-return]


def batch_betti_curves(diagrams: np.ndarray, max_dim: int, resolution: int) -> np.ndarray:
    """Compute Betti curves for a batch of padded diagrams.

    Each diagram in the batch should be padded with zero rows to a uniform
    length.

    :param diagrams: Array of shape ``(batch, n_pairs, at_least_3)`` with
        columns ``[birth, death, dimension]``.
    :param max_dim: Maximum homology dimension to include.
    :param resolution: Number of filtration steps.
    :returns: Array of shape ``(batch, max_dim + 1, resolution)``.
    :raises ValueError: If *diagrams* has invalid shape or is empty.
    :raises InvalidArgumentError: If any diagram is malformed.
    """
    diagrams = np.asarray(diagrams, dtype=np.float32)
    if diagrams.ndim != 3 or diagrams.shape[2] < 3:
        raise ValueError("diagrams must have shape (batch, n_pairs, at least 3)")
    if diagrams.shape[0] == 0:
        raise ValueError("diagrams batch must be non-empty")
    for diagram in diagrams:
        _validate_pairs(diagram, require_dim=True)
    max_dim = validate_nonnegative_int(max_dim, "max_dim")
    resolution = validate_positive_int(resolution, "resolution")
    return _jit_batch_betti_curves_impl(diagrams, max_dim, resolution)  # type: ignore[no-any-return]


__all__ = [
    "pairwise_distances",
    "filter_pairs",
    "betti_curve",
    "persistence_image",
    "vietoris_rips_edges",
    "batch_betti_curves",
    "JITCache",
    "cached_jit",
]
