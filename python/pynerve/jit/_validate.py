"""Validation helpers for JIT kernel inputs."""

from __future__ import annotations

import numpy as np

from ..exceptions import InvalidArgumentError


def _validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise InvalidArgumentError("points must be a 2D array", parameter="points")
    if points.shape[0] == 0 or points.shape[1] == 0:
        raise InvalidArgumentError("points must be non-empty", parameter="points")
    if not np.isfinite(points).all():
        raise InvalidArgumentError(
            "points must contain only finite coordinates", parameter="points"
        )
    return points


def _validate_pairs(pairs: np.ndarray, *, require_dim: bool = False) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.float32)
    min_cols = 3 if require_dim else 2
    if pairs.ndim != 2 or pairs.shape[1] < min_cols:
        raise InvalidArgumentError(
            f"pairs must have shape (n_pairs, at least {min_cols})", parameter="pairs"
        )
    if pairs.size == 0:
        return pairs.reshape(0, max(pairs.shape[1], min_cols))
    births = pairs[:, 0]
    deaths = pairs[:, 1]
    if not np.isfinite(births).all():
        raise InvalidArgumentError("pair births must be finite", parameter="pairs")
    if np.isnan(deaths).any() or np.isneginf(deaths).any():
        raise InvalidArgumentError("pair deaths must be finite or +inf", parameter="pairs")
    finite_deaths = np.isfinite(deaths)
    if finite_deaths.any() and np.any(deaths[finite_deaths] < births[finite_deaths]):
        raise InvalidArgumentError(
            "pair finite deaths must be greater than or equal to births", parameter="pairs"
        )
    if require_dim:
        dims = pairs[:, 2]
        if (
            not np.isfinite(dims).all()
            or (dims < 0).any()
            or not np.equal(dims, np.floor(dims)).all()
        ):
            raise InvalidArgumentError(
                "pair dimensions must be finite non-negative integers", parameter="pairs"
            )
    return pairs
