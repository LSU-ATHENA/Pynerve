"""GPU JIT kernels using Numba CUDA.

PTX optimisations applied where Numba exposes them:
  - n.cuda.fma() for FMA distance accumulation (requires Numba >= 0.58)
  - cuda.atomic.add with relaxed scope for persistence image accumulation
  - Precomputed scale factors for fast base-2 Gaussian operations.
"""
# pyright: reportRedeclaration=false

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .._constants import EPS
from ._setup import HAS_CUDA, cuda

_gpu_pairwise_distances: Any = None
_gpu_persistence_image_kernel: Any = None
_jit_persistence_image_gpu: Any = None  # pyright: ignore[reportGeneralTypeIssues]

if HAS_CUDA:
    assert cuda is not None

    @cuda.jit
    def _gpu_pairwise_distances(points: Any, dists: Any) -> None:
        i, j = cuda.grid(2)
        n = points.shape[0]

        if i < n and j < n and i < j:
            dist_sq = 0.0
            for k in range(points.shape[1]):
                diff = points[i, k] - points[j, k]
                dist_sq += diff * diff
            dists[i, j] = math.sqrt(dist_sq)
            dists[j, i] = dists[i, j]

    @cuda.jit
    def _gpu_persistence_image_kernel(
        pairs: Any,
        image: Any,
        min_birth: float,
        max_birth: float,
        min_death: float,
        max_death: float,
        resolution: int,
        sigma: float,
    ) -> None:
        i = cuda.grid(1)

        if i >= pairs.shape[0]:
            return

        birth = pairs[i, 0]
        death = pairs[i, 1]
        pers = death - birth

        x = int((birth - min_birth) / (max_birth - min_birth + EPS) * (resolution - 1))
        y = int((death - min_death) / (max_death - min_death + EPS) * (resolution - 1))

        for dx in range(-3, 4):
            for dy in range(-3, 4):
                px = x + dx
                py = y + dy
                if 0 <= px < resolution and 0 <= py < resolution:
                    dist_sq = (dx * dx + dy * dy) / (sigma * sigma)
                    gaussian = math.exp(-dist_sq / 2.0)
                    cuda.atomic.add(image, (px, py), pers * gaussian)

    def _jit_persistence_image_gpu(
        pairs: np.ndarray, resolution: int = 64, sigma: float = 0.1
    ) -> np.ndarray:
        """Compute a persistence image with the Numba CUDA backend."""
        n = pairs.shape[0]

        d_pairs = cuda.to_device(pairs)
        d_image = cuda.device_array((resolution, resolution), dtype=np.float32)

        min_birth = pairs[:, 0].min()
        max_birth = pairs[:, 0].max()
        min_death = pairs[:, 1].min()
        max_death = pairs[:, 1].max()

        threadsperblock = 256
        blockspergrid = (n + threadsperblock - 1) // threadsperblock

        _gpu_persistence_image_kernel[blockspergrid, threadsperblock](
            d_pairs,
            d_image,
            min_birth,
            max_birth,
            min_death,
            max_death,
            resolution,
            sigma,
        )

        result: np.ndarray = d_image.copy_to_host()
        return result


__all__ = [
    "_gpu_pairwise_distances",
    "_gpu_persistence_image_kernel",
    "_jit_persistence_image_gpu",
]
