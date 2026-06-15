"""Example datasets for testing and demonstration.

Each function returns an (N, D) numpy array of point coordinates.
All datasets can be used directly with :func:`pynerve.compute_persistence`.
If ``as_tensor=True`` and PyTorch is installed, returns a ``torch.Tensor``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from ._validation import (
    validate_nonnegative_finite,
    validate_nonnegative_int,
    validate_positive_int,
)


def _maybe_tensor(data: np.ndarray, as_tensor: bool) -> Any:
    if as_tensor and torch is not None:
        return torch.from_numpy(data)
    return data


def load_swiss_roll(
    n_samples: int = 100, seed: int | None = None, as_tensor: bool = False
) -> np.ndarray | Any:
    """Generate a 3D Swiss roll dataset.

    :param n_samples: Number of points to generate. Default 100.
    :param seed: Random seed for reproducibility. Default None.
    :param as_tensor: If True and PyTorch is installed, return a
        ``torch.Tensor`` instead of ``np.ndarray``. Default False.
    :returns: Array of shape ``(n_samples, 3)`` with x, y, z coordinates.
    """
    n_samples = validate_positive_int(n_samples, "n_samples")
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=n_samples))
    x = t * np.cos(t)
    y = 20 * rng.uniform(size=n_samples)
    z = t * np.sin(t)
    data = np.column_stack([x, y, z])
    return _maybe_tensor(data, as_tensor)


def load_mobius_strip(
    n_samples: int = 100, seed: int | None = None, as_tensor: bool = False
) -> np.ndarray | Any:
    """Generate a 3D Moebius strip dataset.

    :param n_samples: Number of points to generate. Default 100.
    :param seed: Random seed for reproducibility. Default None.
    :param as_tensor: If True and PyTorch is installed, return a
        ``torch.Tensor`` instead of ``np.ndarray``. Default False.
    :returns: Array of shape ``(n_samples, 3)`` with x, y, z coordinates.
    """
    n_samples = validate_positive_int(n_samples, "n_samples")
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, size=n_samples)
    v = rng.uniform(-0.5, 0.5, size=n_samples)
    x = (1 + v * np.cos(u / 2)) * np.cos(u)
    y = (1 + v * np.cos(u / 2)) * np.sin(u)
    z = v * np.sin(u / 2)
    data = np.column_stack([x, y, z])
    return _maybe_tensor(data, as_tensor)


def load_klein_bottle(
    n_samples: int = 100, seed: int | None = None, as_tensor: bool = False
) -> np.ndarray | Any:
    """Generate a 4D Klein bottle dataset.

    :param n_samples: Number of points to generate. Default 100.
    :param seed: Random seed for reproducibility. Default None.
    :param as_tensor: If True and PyTorch is installed, return a
        ``torch.Tensor`` instead of ``np.ndarray``. Default False.
    :returns: Array of shape ``(n_samples, 4)`` with x, y, z, w coordinates.
    """
    n_samples = validate_positive_int(n_samples, "n_samples")
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, np.pi, size=n_samples)
    v = rng.uniform(0, 2 * np.pi, size=n_samples)
    x = (2.5 + 1.5 * np.cos(v)) * np.cos(u)
    y = (2.5 + 1.5 * np.cos(v)) * np.sin(u)
    z = 1.5 * np.sin(v) * np.cos(u / 2)
    w = 1.5 * np.sin(v) * np.sin(u / 2)
    data = np.column_stack([x, y, z, w])
    return _maybe_tensor(data, as_tensor)


def load_torus(
    n_samples: int = 100,
    major_radius: float = 2.0,
    minor_radius: float = 1.0,
    seed: int | None = None,
    as_tensor: bool = False,
) -> np.ndarray | Any:
    """Generate a 3D torus dataset.

    :param n_samples: Number of points to generate. Default 100.
    :param major_radius: Major radius of the torus. Default 2.0.
    :param minor_radius: Minor radius (tube radius). Default 1.0.
    :param seed: Random seed for reproducibility. Default None.
    :param as_tensor: If True and PyTorch is installed, return a
        ``torch.Tensor`` instead of ``np.ndarray``. Default False.
    :returns: Array of shape ``(n_samples, 3)`` with x, y, z coordinates.
    """
    n_samples = validate_positive_int(n_samples, "n_samples")
    major_radius = validate_nonnegative_finite(major_radius, "major_radius")
    minor_radius = validate_nonnegative_finite(minor_radius, "minor_radius")
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n_samples)
    phi = rng.uniform(0, 2 * np.pi, size=n_samples)
    x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
    y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
    z = minor_radius * np.sin(phi)
    data = np.column_stack([x, y, z])
    return _maybe_tensor(data, as_tensor)


def load_sphere(
    n_samples: int = 100,
    dim: int = 2,
    radius: float = 1.0,
    seed: int | None = None,
    as_tensor: bool = False,
) -> np.ndarray | Any:
    """Generate an n-dimensional sphere dataset.

    :param n_samples: Number of points to generate. Default 100.
    :param dim: Dimension of the sphere surface (e.g. 2 for a 2-sphere
        embedded in 3D). Default 2.
    :param radius: Radius of the sphere. Default 1.0.
    :param seed: Random seed for reproducibility. Default None.
    :param as_tensor: If True and PyTorch is installed, return a
        ``torch.Tensor`` instead of ``np.ndarray``. Default False.
    :returns: Array of shape ``(n_samples, dim + 1)``.
    """
    n_samples = validate_positive_int(n_samples, "n_samples")
    dim = validate_nonnegative_int(dim, "dim")
    radius = validate_nonnegative_finite(radius, "radius")
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_samples, dim + 1))
    points = radius * points / np.linalg.norm(points, axis=1, keepdims=True)
    return _maybe_tensor(points, as_tensor)


def load_circle(
    n_samples: int = 100,
    radius: float = 1.0,
    noise: float = 0.0,
    seed: int | None = None,
    as_tensor: bool = False,
) -> np.ndarray | Any:
    """Generate a noisy 2D circle dataset.

    :param n_samples: Number of points to generate. Default 100.
    :param radius: Radius of the circle. Default 1.0.
    :param noise: Standard deviation of Gaussian noise added to each
        coordinate. Default 0.0 (noiseless).
    :param seed: Random seed for reproducibility. Default None.
    :param as_tensor: If True and PyTorch is installed, return a
        ``torch.Tensor`` instead of ``np.ndarray``. Default False.
    :returns: Array of shape ``(n_samples, 2)`` with x, y coordinates.
    """
    n_samples = validate_positive_int(n_samples, "n_samples")
    radius = validate_nonnegative_finite(radius, "radius")
    noise = validate_nonnegative_finite(noise, "noise")
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n_samples)
    x = radius * np.cos(theta) + noise * rng.normal(size=n_samples)
    y = radius * np.sin(theta) + noise * rng.normal(size=n_samples)
    data = np.column_stack([x, y])
    return _maybe_tensor(data, as_tensor)


__all__ = [
    "load_swiss_roll",
    "load_mobius_strip",
    "load_klein_bottle",
    "load_torus",
    "load_sphere",
    "load_circle",
]
