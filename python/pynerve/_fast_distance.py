"""Vectorized distance operations used by :mod:`pynerve.fast_ops`."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist, pdist, squareform

from ._validation import validate_nonnegative_finite, validate_points, validate_positive_int


def _validate_metric(metric: str) -> str:
    if not isinstance(metric, str) or not metric:
        raise ValueError(f"metric must be a non-empty string, got {metric!r}")
    return metric


def pairwise_distances_fast(points: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    points = validate_points(points)
    metric = _validate_metric(metric)
    dists = pdist(points, metric=metric)  # pyright: ignore[reportCallIssue, reportArgumentType]
    return squareform(dists)  # type: ignore[no-any-return]


def pairwise_distances_broadcast(points: np.ndarray) -> np.ndarray:
    points = validate_points(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))  # type: ignore[no-any-return]


def nearest_neighbors_fast(
    points: np.ndarray, k: int = 5, metric: str = "euclidean"
) -> tuple[np.ndarray, np.ndarray]:
    points = validate_points(points)
    k = validate_positive_int(k, "k")
    metric = _validate_metric(metric)
    if k > points.shape[0]:
        raise ValueError(f"k must be in the range [1, n_points], got {k}")
    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")
    nn.fit(points)
    distances, indices = nn.kneighbors(points)

    return distances, indices


def sparse_distance_matrix(
    points: np.ndarray, max_dist: float, output_type: str = "dense"
) -> np.ndarray:
    points = validate_points(points)
    max_dist = validate_nonnegative_finite(max_dist, "max_dist")
    if output_type not in {"dense", "sparse", "coo", "csr"}:
        raise ValueError(f"Unknown output_type: {output_type}")
    n = points.shape[0]

    if output_type == "dense":
        dists = pairwise_distances_fast(points)
        dists[dists > max_dist] = 0
        return dists

    if output_type in ["sparse", "coo", "csr"]:
        row_list, col_list, data_list = [], [], []

        chunk_size = 1000
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            chunk_dists = cdist(points[i:end_i], points, metric="euclidean")

            mask = (chunk_dists <= max_dist) & (chunk_dists > 0)
            rows, cols = np.where(mask)
            row_list.append(i + rows)
            col_list.append(cols.astype(rows.dtype))
            data_list.append(chunk_dists[rows, cols])

        row = np.concatenate(row_list) if row_list else np.array([], dtype=int)
        col = np.concatenate(col_list) if col_list else np.array([], dtype=int)
        data = np.concatenate(data_list) if data_list else np.array([], dtype=float)
        coo = coo_matrix((data, (row, col)), shape=(n, n))

        if output_type == "csr":
            return coo.tocsr()  # type: ignore[no-any-return]
        return coo  # type: ignore[no-any-return]

    raise RuntimeError(f"invalid output_type after validation: {output_type}")
