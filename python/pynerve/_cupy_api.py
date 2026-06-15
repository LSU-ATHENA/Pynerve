"""CuPy-backed persistence API."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from ._cupy_compat import HAS_CUPY, cp
from ._cupy_convert import to_cupy
from ._cupy_persistence import CuPyPersistence, _compute_core_persistence
from ._validation import validate_max_radius, validate_nonnegative_int
from .exceptions import ValidationError

if TYPE_CHECKING:
    import cupy


def _validate_max_dim(max_dim: int) -> int:
    return validate_nonnegative_int(max_dim, "max_dim")


def _validate_point_cloud(points: Any) -> Any:
    if HAS_CUPY and hasattr(points, "ndim"):
        if points.ndim != 2:
            raise ValidationError("points must be a 2D point cloud")
        if points.shape[0] == 0 or points.shape[1] == 0:
            raise ValidationError("points must be non-empty")
        if hasattr(points, "dtype") and not np.issubdtype(
            points.dtype if hasattr(points.dtype, "char") else np.dtype(points.dtype), np.number
        ):
            raise ValidationError("points must have a numeric dtype")
        if hasattr(points, "get"):
            if not bool(cp.isfinite(points).all().item()):
                raise ValidationError("points must contain only finite coordinates")
        elif not bool(cp.isfinite(cp.asarray(points)).all().item()):
            raise ValidationError("points must contain only finite coordinates")
        return points
    host = np.asarray(points)
    if host.ndim != 2:
        raise ValidationError("points must be a 2D point cloud")
    if host.shape[0] == 0 or host.shape[1] == 0:
        raise ValidationError("points must be non-empty")
    if not np.issubdtype(host.dtype, np.number):
        raise ValidationError("points must have a numeric dtype")
    if not np.isfinite(host).all():
        raise ValidationError("points must contain only finite coordinates")
    return points


def _validate_point_clouds(point_clouds: list[np.ndarray | cupy.ndarray]) -> list[Any]:
    if isinstance(point_clouds, (str, bytes)) or not isinstance(point_clouds, Iterable):
        raise ValidationError("point_clouds must be an iterable of point clouds")
    clouds = list(point_clouds)
    for point_cloud in clouds:
        _validate_point_cloud(point_cloud)
    return clouds


def compute_diagrams_cupy(
    points: np.ndarray | cupy.ndarray,
    max_radius: float | None = None,
    max_dim: int = 2,
    device_id: int = 0,
) -> dict[str, Any]:
    """Compute persistence diagrams for a single point cloud.

    If CuPy is available and *points* is a CuPy array, the computation
    stays on the GPU where possible.

    :param points: 2D point cloud (NumPy or CuPy array).
    :param max_radius: Maximum radius for the Vietoris-Rips complex.
        ``None`` means no limit.
    :param max_dim: Maximum homology dimension (default 2).
    :param device_id: GPU device ID (default 0).
    :returns: A dict with ``"pairs"``, ``"betti_numbers"``, and other
        persistence result fields.
    :raises ValidationError: If *points* or parameters are invalid.
    """
    device_id = validate_nonnegative_int(device_id, "device_id")
    max_dim = _validate_max_dim(max_dim)
    max_radius = validate_max_radius(max_radius)
    points = _validate_point_cloud(points)
    if not HAS_CUPY:
        return _compute_core_persistence(np.asarray(points), max_radius, max_dim)

    if isinstance(points, np.ndarray):
        points = to_cupy(points, device_id=device_id)

    computer = CuPyPersistence(device_id=device_id)
    return computer.compute_persistence_cupy(points, max_radius, max_dim)


def batch_diagrams_cupy(
    point_clouds: list[np.ndarray | cupy.ndarray], **kwargs: Any
) -> list[dict[str, Any]]:
    """Compute persistence diagrams for a batch of point clouds.

    Each point cloud is processed independently. Accepted keyword
    arguments include ``max_radius``, ``max_dim``, and ``device_id``.

    :param point_clouds: List of 2D point clouds (NumPy or CuPy arrays).
    :param kwargs: Additional keyword arguments forwarded to the
        per-cloud persistence computation.
    :returns: A list of persistence result dicts, one per input cloud.
    :raises ValidationError: If any point cloud or parameter is invalid.
    """
    device_id = validate_nonnegative_int(kwargs.pop("device_id", 0), "device_id")
    if "max_dim" in kwargs:
        kwargs["max_dim"] = _validate_max_dim(kwargs["max_dim"])
    if "max_radius" in kwargs:
        kwargs["max_radius"] = validate_max_radius(kwargs["max_radius"])
    if "max_dist" in kwargs:
        kwargs["max_radius"] = validate_max_radius(kwargs.pop("max_dist"))
    point_clouds = _validate_point_clouds(point_clouds)

    if not HAS_CUPY:
        return [
            _compute_core_persistence(
                np.asarray(point_cloud),
                kwargs.get("max_radius"),
                kwargs.get("max_dim", 2),
            )
            for point_cloud in point_clouds
        ]

    cupy_clouds = [to_cupy(pc, device_id=device_id) for pc in point_clouds]

    computer = CuPyPersistence(device_id=device_id)
    return computer.batch_diagrams_cupy(cupy_clouds, **kwargs)
