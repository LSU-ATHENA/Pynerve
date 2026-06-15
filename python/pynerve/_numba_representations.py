"""Numba persistence-representation kernels."""

from __future__ import annotations

import numpy as np

from ._numba_compat import njit, prange


@njit(parallel=True, cache=True)
def numba_betti_curve(
    pairs: np.ndarray, max_dim: int, resolution: int, max_time: float
) -> np.ndarray:
    if max_dim < 0 or resolution <= 0:
        raise ValueError("max_dim must be non-negative and resolution positive")
    betti = np.zeros((max_dim + 1, resolution), dtype=np.int64)
    if pairs.shape[0] == 0 or max_time <= 0.0:
        return betti

    for d in prange(max_dim + 1):
        for i in range(pairs.shape[0]):
            if int(pairs[i, 2]) == d:
                birth = pairs[i, 0]
                death = pairs[i, 1]

                start_idx = int(birth / max_time * resolution)
                end_idx = int(death / max_time * resolution)

                start_idx = max(0, start_idx)
                end_idx = min(resolution, end_idx)

                for t in range(start_idx, end_idx):
                    betti[d, t] += 1

    return betti


@njit(cache=True)
def numba_persistence_image(
    pairs: np.ndarray,
    resolution: int,
    sigma: float,
    min_birth: float,
    max_birth: float,
    min_death: float,
    max_death: float,
) -> np.ndarray:
    if resolution <= 0 or sigma <= 0.0:
        raise ValueError("resolution and sigma must be positive")
    birth_span = max_birth - min_birth
    death_span = max_death - min_death
    if birth_span <= 0.0:
        birth_span = 1e-12
    if death_span <= 0.0:
        death_span = 1e-12

    image = np.zeros((resolution, resolution), dtype=np.float64)

    for i in range(pairs.shape[0]):
        birth = pairs[i, 0]
        death = pairs[i, 1]
        pers = death - birth

        x = int((birth - min_birth) / birth_span * (resolution - 1))
        y = int((death - min_death) / death_span * (resolution - 1))

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                px = x + dx
                py = y + dy

                if 0 <= px < resolution and 0 <= py < resolution:
                    dist_sq = (dx * dx + dy * dy) / (sigma * sigma)
                    gaussian = pers * np.exp(-dist_sq / 2.0)
                    image[px, py] += gaussian

    return image
