"""Vectorized persistence representations."""

from __future__ import annotations

import numpy as np


def _birth_death_pairs(pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=float)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] < 2:
        raise ValueError("pairs must have shape (n_pairs, at least 2)")
    birth_death = pairs[:, :2]
    if not np.isfinite(birth_death).all():
        raise ValueError("pair birth/death values must be finite")
    if not (birth_death[:, 1] >= birth_death[:, 0]).all():
        raise ValueError("pair death values must be greater than or equal to births")
    return birth_death


def _positive_resolution(resolution: int) -> None:
    if resolution <= 0:
        raise ValueError("resolution must be positive")


def persistence_image_fast(
    pairs: np.ndarray,
    resolution: int = 64,
    sigma: float = 0.1,
    weight_fn: str | None = None,
) -> np.ndarray:
    _positive_resolution(resolution)
    sigma = float(sigma)
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    if weight_fn not in {None, "linear", "constant"}:
        raise ValueError("weight_fn must be None, 'linear', or 'constant'")

    pairs = _birth_death_pairs(pairs)
    if pairs.shape[0] == 0:
        return np.zeros((resolution, resolution), dtype=float)

    births = pairs[:, 0]
    deaths = pairs[:, 1]
    persistence = deaths - births

    min_birth, max_birth = births.min(), births.max()
    min_death, max_death = deaths.min(), deaths.max()
    if max_birth <= min_birth:
        max_birth = min_birth + 1e-12
    if max_death <= min_death:
        max_death = min_death + 1e-12

    if weight_fn == "linear":
        weights = persistence
    elif weight_fn == "constant":
        weights = np.ones(len(pairs))
    else:
        weights = persistence**2

    x = np.linspace(min_birth, max_birth, resolution)
    y = np.linspace(min_death, max_death, resolution)
    X, Y = np.meshgrid(x, y)  # noqa: N806

    image = np.zeros((resolution, resolution), dtype=float)

    for w, b, d in zip(weights, births, deaths, strict=False):
        dx = (X - b) / sigma
        dy = (Y - d) / sigma
        image += w * np.exp(-(dx**2 + dy**2) / 2)

    return image


def betti_curve_fast(
    pairs: np.ndarray,
    max_dim: int = 3,
    resolution: int = 100,
    max_time: float | None = None,
) -> np.ndarray:
    _positive_resolution(resolution)
    if max_dim < 0:
        raise ValueError("max_dim must be non-negative")

    pairs = np.asarray(pairs, dtype=float)
    if pairs.size == 0:
        return np.zeros((max_dim + 1, resolution), dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] < 3:
        raise ValueError("pairs must have shape (n_pairs, at least 3)")
    if not np.isfinite(pairs[:, :3]).all():
        raise ValueError("pairs must contain only finite birth, death, and dimension values")
    if not (pairs[:, 1] >= pairs[:, 0]).all():
        raise ValueError("pair death values must be greater than or equal to births")
    if max_time is None:
        max_time = pairs[:, 1].max()
    max_time = float(max_time)  # pyright: ignore[reportArgumentType]  # pyright: ignore[reportArgumentType]
    if max_time < 0 or not np.isfinite(max_time):
        raise ValueError("max_time must be finite and non-negative")

    time_grid = np.linspace(0, max_time, resolution)
    betti = np.zeros((max_dim + 1, resolution), dtype=int)

    for d in range(max_dim + 1):
        mask = pairs[:, 2] == d
        d_pairs = pairs[mask]

        if len(d_pairs) == 0:
            continue

        births = d_pairs[:, 0]
        deaths = d_pairs[:, 1]

        active = (births[:, None] <= time_grid[None, :]) & (deaths[:, None] > time_grid[None, :])
        betti[d] = active.sum(axis=0)

    return betti


def persistence_landscape_fast(
    pairs: np.ndarray, n_layers: int = 5, resolution: int = 100
) -> np.ndarray:
    _positive_resolution(resolution)
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")

    pairs = _birth_death_pairs(pairs)
    if pairs.shape[0] == 0:
        return np.zeros((n_layers, resolution), dtype=float)

    births = pairs[:, 0]
    deaths = pairs[:, 1]
    midpoints = (births + deaths) / 2
    heights = (deaths - births) / 2

    t_min = births.min()
    t_max = deaths.max()
    if t_max <= t_min:
        t_max = t_min + 1e-12
    t = np.linspace(t_min, t_max, resolution)

    landscape_rows = []

    for m, h in zip(midpoints, heights, strict=False):
        tent = h - np.abs(t - m)
        tent = np.maximum(tent, 0)
        landscape_rows.append(tent)

    landscapes = np.sort(np.array(landscape_rows), axis=0)[::-1]

    if len(landscapes) >= n_layers:
        return landscapes[:n_layers]
    padding = np.zeros((n_layers - len(landscapes), resolution))
    return np.vstack([landscapes, padding])
