"""CuPy integration for zero-copy GPU operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._cupy_api import batch_diagrams_cupy, compute_diagrams_cupy

#: ``True`` when the CuPy library is installed, ``False`` otherwise.
from ._cupy_compat import HAS_CUPY
from ._cupy_convert import from_cupy, to_cupy
from ._cupy_memory import CudaStream, GPUBuffer, UnifiedMemoryBuffer
from ._cupy_persistence import CuPyPersistence


def compute_diagrams(
    points: np.ndarray | Any,
    max_radius: float | None = None,
    max_dim: int = 2,
    device_id: int = 0,
) -> dict[str, Any]:
    """Compute persistence diagrams, optionally using CuPy for GPU acceleration.

    :param points: 2D point cloud as a NumPy or CuPy array.
    :param max_radius: Maximum radius for the Vietoris-Rips complex.
        ``None`` means no limit.
    :param max_dim: Maximum homology dimension (default 2).
    :param device_id: GPU device ID (default 0).
    :returns: A dict with ``"pairs"``, ``"betti_numbers"``, and other
        persistence result fields.
    :raises ValidationError: If *points* or parameters are invalid.
    :raises RuntimeError: If the operation fails.
    """
    return compute_diagrams_cupy(points, max_radius, max_dim, device_id)


def batch_diagrams(point_clouds: list[np.ndarray | Any], **kwargs: Any) -> list[dict[str, Any]]:
    """Compute persistence diagrams for a batch of point clouds.

    :param point_clouds: Iterable of 2D point clouds (NumPy or CuPy).
    :param kwargs: Additional keyword arguments forwarded to the
        underlying implementation (``max_radius``, ``max_dim``,
        ``device_id``, etc.).
    :returns: A list of persistence result dicts, one per input cloud.
    :raises ValidationError: If any point cloud or parameter is invalid.
    """
    return batch_diagrams_cupy(point_clouds, **kwargs)


__all__ = [
    "HAS_CUPY",
    "CuPyPersistence",
    "GPUBuffer",
    "CudaStream",
    "UnifiedMemoryBuffer",
    "to_cupy",
    "from_cupy",
    "compute_diagrams",
    "batch_diagrams",
]
