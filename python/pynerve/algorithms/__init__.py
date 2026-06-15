"""Algorithmic primitives for TDA  --  distances, neighbors, kernels, and diagram vectorization."""

from __future__ import annotations

from typing import cast

import numpy as np

from .._image_utils import persistence_image as _persistence_image_native
from .._validation import validate_diagram_array
from ..exceptions import InvalidArgumentError, ValidationError


def pairwise_distances(
    points: np.ndarray,
    metric: str = "euclidean",
    _use_simd: bool = True,
) -> np.ndarray:
    """Compute the pairwise distance matrix for a set of points.

    Args:
        points: Array of shape ``(n, d)``.
        metric: Distance metric (``"euclidean"``, ``"manhattan"``, ``"cosine"``).
        _use_simd: Ignored; preserved for API compatibility.

    Returns:
        Symmetric distance matrix of shape ``(n, n)``.
    """
    from scipy.spatial.distance import pdist, squareform  # noqa: PLC0415

    if not isinstance(points, np.ndarray) or points.ndim != 2:
        raise ValidationError("points must be a 2-D numpy array", parameter="points")
    distances = pdist(points, metric=metric)  # type: ignore[call-overload]
    return squareform(distances)  # type: ignore[no-any-return]


def knn(
    points: np.ndarray,
    k: int = 5,
    metric: str = "euclidean",
    algorithm: str = "brute_force",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute k-nearest neighbors for a set of points.

    Args:
        points: Array of shape ``(n, d)``.
        k: Number of neighbors (excluding the point itself).
        metric: Distance metric.
        algorithm: ``"brute_force"`` or ``"kd_tree"``.

    Returns:
        ``(distances, indices)``  --  each of shape ``(n, k)``.
    """
    from scipy.spatial import KDTree  # noqa: PLC0415

    if not isinstance(points, np.ndarray) or points.ndim != 2:
        raise ValidationError("points must be a 2-D numpy array", parameter="points")
    if k < 1:
        raise InvalidArgumentError("k must be >= 1", parameter="k")
    if algorithm == "kd_tree":
        tree = KDTree(points)
        dist, idx = tree.query(points, k=k + 1)
        return cast(np.ndarray, dist)[:, 1:], cast(np.ndarray, idx)[:, 1:]
    if algorithm == "brute_force":
        distances = pairwise_distances(points, metric=metric)
        np.fill_diagonal(distances, np.inf)
        idx = np.argpartition(distances, k, axis=1)[:, :k]
        rows = np.arange(len(points))[:, None]
        dst = distances[rows, idx]
        sort = np.argsort(dst, axis=1)
        return np.take_along_axis(dst, sort, axis=1), np.take_along_axis(idx, sort, axis=1)
    raise InvalidArgumentError(
        "algorithm must be 'brute_force' or 'kd_tree'", parameter="algorithm"
    )


def persistence_landscape(
    diagram: np.ndarray,
    num_levels: int = 5,
    resolution: int = 100,
) -> np.ndarray:
    """Compute persistence landscapes from a persistence diagram.

    Args:
        diagram: Array of shape ``(n, >=2)`` with ``(birth, death)`` columns.
        num_levels: Number of landscape levels.
        resolution: Number of sample points in the landscape.

    Returns:
        Landscapes of shape ``(num_levels, resolution)``.
    """
    validated = validate_diagram_array(diagram, require_dims=False)
    births = validated[:, 0]
    deaths = validated[:, 1]
    finite = np.isfinite(deaths) & (deaths > births)
    births = births[finite]
    deaths = deaths[finite]
    if births.size == 0:
        return np.zeros((num_levels, resolution))

    t_min = float(np.min(births))
    t_max = float(np.max(deaths[np.isfinite(deaths)]))
    if t_max <= t_min:
        t_max = t_min + 1.0
    t = np.linspace(t_min, t_max, resolution)

    landscapes = np.zeros((num_levels, resolution))
    for i in range(len(births)):
        b, d = float(births[i]), float(deaths[i])
        mid = (b + d) / 2.0
        for step in range(resolution):
            x = float(t[step])
            if x <= b or x >= d:
                continue
            val = x - b if x <= mid else d - x  # noqa: SIM108
            for level in range(num_levels):
                if val > landscapes[level, step]:
                    landscapes[level:, step] = np.roll(landscapes[level:, step], -1)
                    landscapes[level, step] = val
                    break
    return landscapes


def persistence_image(
    diagram: np.ndarray,
    resolution: int = 64,
    sigma: float = 0.1,
) -> np.ndarray:
    """Compute a persistence image. Delegates to :func:`pynerve.persistence_image`.

    Args:
        diagram: Array of shape ``(n, >=2)``.
        resolution: Image resolution.
        sigma: Gaussian kernel width.

    Returns:
        Persistence image of shape ``(resolution, resolution)``.
    """
    return _persistence_image_native(diagram, resolution=resolution, sigma=sigma)  # type: ignore[arg-type]


def persistence_silhouette(
    diagram: np.ndarray,
    resolution: int = 100,
    weight_power: float = 1.0,
) -> np.ndarray:
    """Compute persistence silhouettes.

    Args:
        diagram: Array of shape ``(n, >=2)``.
        resolution: Number of sample points.
        weight_power: Exponent for persistence weighting.

    Returns:
        Silhouette array of shape ``(resolution,)``.
    """
    validated = validate_diagram_array(diagram, require_dims=False)
    births = validated[:, 0]
    deaths = validated[:, 1]
    finite = np.isfinite(deaths) & (deaths > births)
    births = births[finite]
    deaths = deaths[finite]
    if births.size == 0:
        return np.zeros(resolution)

    t_min = float(np.min(births))
    t_max = float(np.max(deaths[np.isfinite(deaths)]))
    if t_max <= t_min:
        t_max = t_min + 1.0
    t = np.linspace(t_min, t_max, resolution)
    silhouette = np.zeros(resolution)

    for i in range(len(births)):
        b, d = float(births[i]), float(deaths[i])
        w = (d - b) ** weight_power
        mid = (b + d) / 2.0
        for step in range(resolution):
            x = float(t[step])
            if x <= b or x >= d:
                continue
            silhouette[step] += w * (x - b) if x <= mid else w * (d - x)

    return silhouette / len(births)


def persistence_heat_vector(
    diagram: np.ndarray,
    resolution: int = 100,
    sigma: float = 1.0,
    t: float = 1.0,
) -> np.ndarray:
    """Compute persistence-based heat kernel vector.

    Args:
        diagram: Array of shape ``(n, >=2)``.
        resolution: Number of Dirac deltas along the line.
        sigma: Gaussian kernel width for heat diffusion.
        t: Diffusion time parameter.

    Returns:
        Heat vector of shape ``(resolution,)``.
    """
    validated = validate_diagram_array(diagram, require_dims=False)
    births = validated[:, 0]
    deaths = validated[:, 1]
    finite = np.isfinite(deaths) & (deaths > births)
    births = births[finite]
    deaths = deaths[finite]
    if births.size == 0:
        return np.zeros(resolution)

    t_min = float(np.min(births))
    t_max = float(np.max(deaths[np.isfinite(deaths)]))
    if t_max <= t_min:
        t_max = t_min + 1.0
    x = np.linspace(t_min, t_max, resolution)
    vec = np.zeros(resolution)

    for i in range(len(births)):
        b, d = float(births[i]), float(deaths[i])
        w = d - b
        mid = (b + d) / 2.0
        vec += w * np.exp(-((x - mid) ** 2) / (2.0 * sigma * sigma))

    return np.exp(-t) * vec


def gaussian_kernel_matrix(
    d1: np.ndarray,
    d2: np.ndarray | None = None,
    sigma: float = 1.0,
) -> np.ndarray:
    """Compute a persistence Gaussian kernel matrix between two diagrams.

    Args:
        d1: First diagram of shape ``(n1, >=2)``.
        d2: Second diagram (optional). If ``None``, uses ``d1``.
        sigma: Kernel bandwidth.

    Returns:
        Kernel matrix of shape ``(n1, n2)``.
    """
    d1 = validate_diagram_array(d1, require_dims=False)
    d2 = d1 if d2 is None else validate_diagram_array(d2, require_dims=False)

    valid1 = np.isfinite(d1[:, 1]) & (d1[:, 1] > d1[:, 0])
    valid2 = np.isfinite(d2[:, 1]) & (d2[:, 1] > d2[:, 0])
    births1, deaths1 = d1[valid1, 0], d1[valid1, 1]
    births2, deaths2 = d2[valid2, 0], d2[valid2, 1]

    if births1.size == 0 or births2.size == 0:
        return np.zeros((len(births1), len(births2)))

    b1 = births1[:, None]
    d1_ = deaths1[:, None]
    b2 = births2[None, :]
    d2_ = deaths2[None, :]
    diff_b = (b1 - b2) ** 2
    diff_d = (d1_ - d2_) ** 2
    return np.exp(-(diff_b + diff_d) / (2.0 * sigma * sigma))


__all__ = [
    "gaussian_kernel_matrix",
    "knn",
    "pairwise_distances",
    "persistence_heat_vector",
    "persistence_image",
    "persistence_landscape",
    "persistence_silhouette",
]
