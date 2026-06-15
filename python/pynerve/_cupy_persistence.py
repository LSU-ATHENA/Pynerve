"""CuPy persistence operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._compute_api import compute_persistence
from ._cupy_compat import HAS_CUPY, cp
from ._fallback_classes import PersistenceOptions
from ._validation import validate_max_radius

if TYPE_CHECKING:
    import cupy


def _validate_core_points(points: np.ndarray) -> np.ndarray:
    host = np.asarray(points)
    if host.ndim != 2:
        raise ValueError("points must be a 2D point cloud")
    if host.shape[0] == 0 or host.shape[1] == 0:
        raise ValueError("points must be non-empty")
    if not np.issubdtype(host.dtype, np.number) or np.issubdtype(host.dtype, np.complexfloating):
        raise TypeError("points must have a real numeric dtype")
    if not np.isfinite(host).all():
        raise ValueError("points must contain only finite coordinates")
    return host


def _compute_core_persistence(
    points: np.ndarray, max_radius: float | None, max_dim: int
) -> dict[str, Any]:
    if max_dim < 0:
        raise ValueError("max_dim must be non-negative")
    max_radius = validate_max_radius(max_radius)
    points = _validate_core_points(points)

    options = PersistenceOptions(
        max_dim=int(max_dim),
        max_radius=max_radius,
    )
    result = compute_persistence(points, options)
    from ._compute_core import PersistenceResult as PR  # noqa: PLC0415, N817

    if isinstance(result, PR):
        return {
            "pairs": result.pairs,
            "betti_numbers": result.betti_numbers,
            "max_dim": result.max_dim,
            "max_radius": result.max_radius,
            "diagnostics": result.diagnostics,
        }
    return result


def _validate_persistence_image_diagrams(diagrams: cupy.ndarray) -> None:
    if not hasattr(diagrams, "ndim") or not hasattr(diagrams, "dtype"):
        raise TypeError("diagrams must be an array")
    if diagrams.ndim != 2 or diagrams.shape[1] < 2:
        raise ValueError("diagrams must have shape (n_pairs, at least 2)")
    if not np.issubdtype(diagrams.dtype, np.number) or np.issubdtype(
        diagrams.dtype, np.complexfloating
    ):
        raise TypeError("diagrams must have a real numeric dtype")
    if diagrams.size > 0 and not bool(cp.isfinite(diagrams).all().item()):
        raise ValueError("diagrams must contain only finite birth/death pairs")
    if diagrams.size > 0 and not bool((diagrams[:, 1] >= diagrams[:, 0]).all().item()):
        raise ValueError("diagram deaths must be greater than or equal to births")


class CuPyPersistence:
    """CuPy helpers for distance, complex, and persistence workflows."""

    def __init__(self, device_id: int = 0):
        """Create a CuPy persistence helper bound to a specific GPU device.

        :param device_id: GPU device ID (default 0).
        :raises ValueError: If *device_id* is negative.
        """
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        self.device_id = device_id
        if HAS_CUPY:
            cp.cuda.Device(device_id).use()

    def pairwise_distances_cupy(self, points: cupy.ndarray) -> cupy.ndarray:
        """Compute a GPU pairwise Euclidean distance matrix.

        :param points: 2D CuPy array of shape ``(n_points, n_dims)``.
        :returns: A square CuPy array of shape ``(n_points, n_points)``
            with pairwise Euclidean distances.
        :raises RuntimeError: If CuPy is not installed.
        :raises ValueError: If *points* is not 2D or contains non-finite
            values.
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy required for GPU operations. Install with: pip install cupy")
        if points.ndim != 2:
            raise ValueError("points must be a 2D array")
        if points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("points cannot be empty")
        if not bool(cp.isfinite(points).all().item()):
            raise ValueError("points must contain only finite values")

        diff = points[:, None, :] - points[None, :, :]
        return cp.sqrt(cp.sum(diff**2, axis=2))

    def build_vr_complex_cupy(
        self, points: cupy.ndarray, max_dist: float, max_dim: int = 2
    ) -> tuple[cupy.ndarray, list[cupy.ndarray]]:
        """Build edges and triangles for a Vietoris-Rips complex on the GPU.

        :param points: 2D CuPy array of shape ``(n_points, n_dims)``.
        :param max_dist: Maximum distance for edge inclusion.
        :param max_dim: Maximum simplex dimension (1 or 2, default 2).
        :returns: A tuple ``(edges, simplices)`` where *edges* is a
            2-column CuPy array and *simplices* is a list of CuPy arrays
            per dimension.
        :raises RuntimeError: If CuPy is not installed.
        :raises ValueError: If parameters are out of range.
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy required. Install with: pip install cupy")
        if max_dist is None or max_dist < 0 or not np.isfinite(float(max_dist)):
            raise ValueError("max_dist must be finite and non-negative")
        max_dist = float(max_dist)
        if max_dim < 1 or max_dim > 2:
            raise ValueError("max_dim must be 1 or 2")

        dists = self.pairwise_distances_cupy(points)
        edges = cp.argwhere(dists <= max_dist)
        edges = edges[edges[:, 0] < edges[:, 1]]

        simplices = [edges]

        if max_dim >= 2:
            n = points.shape[0]
            adj = cp.zeros((n, n), dtype=cp.int32)
            adj[edges[:, 0], edges[:, 1]] = 1
            adj[edges[:, 1], edges[:, 0]] = 1

            for i, j in edges.get():
                common = cp.where(adj[i] & adj[j])[0]
                common = common[common > j]
                if len(common) > 0:
                    triangles = cp.stack(
                        [cp.full(len(common), i), cp.full(len(common), j), common],
                        axis=1,
                    )
                    simplices.append(triangles)

        return edges, simplices

    def compute_persistence_cupy(
        self, points: cupy.ndarray, max_radius: float | None = None, max_dim: int = 2
    ) -> dict[str, Any]:
        """Compute persistence by transferring GPU data to the host core engine.

        :param points: 2D CuPy point cloud.
        :param max_radius: Maximum radius for the VR complex.
            ``None`` means no limit.
        :param max_dim: Maximum homology dimension (default 2).
        :returns: A persistence result dict with ``"pairs"``,
            ``"betti_numbers"``, and diagnostics.
        :raises RuntimeError: If CuPy is not installed.
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy required. Install with: pip install cupy")
        host_points = points.get() if hasattr(points, "get") else np.asarray(points)
        return _compute_core_persistence(host_points, max_radius, max_dim)

    def persistence_image_cupy(
        self, diagrams: cupy.ndarray, resolution: int = 64, sigma: float = 0.1
    ) -> cupy.ndarray:
        """Compute a persistence image from finite birth/death pairs.

        :param diagrams: CuPy array of shape ``(n_pairs, 2+)`` with birth
            and death columns.
        :param resolution: Image resolution (number of bins per axis,
            default 64).
        :param sigma: Gaussian kernel bandwidth (default 0.1).
        :returns: A CuPy array of shape ``(resolution, resolution)``.
        :raises RuntimeError: If CuPy is not installed.
        :raises ValueError: If *resolution* or *sigma* are non-positive,
            or if *diagrams* is invalid.
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy required. Install with: pip install cupy")
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if sigma <= 0 or not np.isfinite(float(sigma)):
            raise ValueError("sigma must be finite and positive")

        _validate_persistence_image_diagrams(diagrams)
        if diagrams.size == 0:
            return cp.zeros((resolution, resolution))

        births = diagrams[:, 0]
        deaths = diagrams[:, 1]
        persistence = deaths - births

        min_birth = cp.min(births)
        max_birth = cp.max(births)
        min_death = cp.min(deaths)
        max_death = cp.max(deaths)

        b_range = cp.linspace(min_birth, max_birth, resolution)
        d_range = cp.linspace(min_death, max_death, resolution)
        B, D = cp.meshgrid(b_range, d_range)  # noqa: N806

        image = cp.zeros((resolution, resolution))

        for i in range(len(diagrams)):
            weight = persistence[i]
            dist_sq = ((B - births[i]) ** 2 + (D - deaths[i]) ** 2) / (sigma**2)
            gaussian = weight * cp.exp(-dist_sq / 2)
            image += gaussian

        return image

    def batch_diagrams_cupy(
        self,
        point_clouds: list[cupy.ndarray],
        max_radius: float | None = None,
        max_dim: int = 2,
    ) -> list[dict[str, Any]]:
        """Compute persistence for a batch of CuPy point clouds.

        :param point_clouds: List of 2D CuPy point cloud arrays.
        :param max_radius: Maximum radius for the VR complex.
            ``None`` means no limit.
        :param max_dim: Maximum homology dimension (default 2).
        :returns: A list of persistence result dicts, one per input cloud.
        :raises RuntimeError: If CuPy is not installed.
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy required. Install with: pip install cupy")

        return [
            self.compute_persistence_cupy(points, max_radius, max_dim) for points in point_clouds
        ]
