"""CPU JIT-compiled topology kernels."""

from __future__ import annotations

import numpy as np

from ._setup import jit, prange


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _jit_pairwise_distances_impl(points: np.ndarray) -> np.ndarray:
    """Compute a pairwise Euclidean distance matrix."""
    n = points.shape[0]
    dim = points.shape[1]
    dists = np.zeros((n, n), dtype=np.float32)

    for i in prange(n):
        for j in range(i + 1, n):
            dist_sq = 0.0
            for k in range(dim):
                diff = points[i, k] - points[j, k]
                dist_sq += diff * diff
            dists[i, j] = np.sqrt(dist_sq)
            dists[j, i] = dists[i, j]

    return dists


@jit(nopython=True, parallel=True, cache=True)
def _jit_filter_pairs_impl(pairs: np.ndarray, threshold: float) -> np.ndarray:
    """Return a mask for pairs whose persistence exceeds threshold."""
    n = pairs.shape[0]
    mask = np.zeros(n, dtype=bool)

    for i in prange(n):
        persistence = pairs[i, 1] - pairs[i, 0]
        mask[i] = persistence > threshold

    return mask


@jit(nopython=True, parallel=True, cache=True)
def _jit_betti_curve_impl(pairs: np.ndarray, max_dim: int, resolution: int) -> np.ndarray:
    """Compute Betti numbers over an evenly sampled filtration."""
    n = pairs.shape[0]
    betti = np.zeros((max_dim + 1, resolution), dtype=np.int32)

    max_death = np.max(pairs[:, 1])

    for r in range(resolution):
        threshold = max_death * r / resolution

        for i in range(n):
            birth = pairs[i, 0]
            death = pairs[i, 1]
            dim = int(pairs[i, 2])

            if dim <= max_dim and birth <= threshold < death:
                betti[dim, r] += 1

    return betti


@jit(nopython=True, parallel=True, cache=True)
def _jit_persistence_image_impl(pairs: np.ndarray, resolution: int, sigma: float) -> np.ndarray:
    """Compute a persistence image on a square grid."""
    n = pairs.shape[0]
    image = np.zeros((resolution, resolution), dtype=np.float32)

    min_birth = np.min(pairs[:, 0])
    max_birth = np.max(pairs[:, 0])
    min_death = np.min(pairs[:, 1])
    max_death = np.max(pairs[:, 1])

    for i in prange(n):
        birth = pairs[i, 0]
        death = pairs[i, 1]
        pers = death - birth

        weight = pers

        x = int((birth - min_birth) / (max_birth - min_birth + 1e-9) * (resolution - 1))
        y = int((death - min_death) / (max_death - min_death + 1e-9) * (resolution - 1))

        for dx in range(-3, 4):
            for dy in range(-3, 4):
                px = x + dx
                py = y + dy
                if 0 <= px < resolution and 0 <= py < resolution:
                    dist_sq = (dx * dx + dy * dy) / (sigma * sigma)
                    gaussian = np.exp(-dist_sq / 2.0)
                    image[px, py] += weight * gaussian

    return image


@jit(nopython=True, parallel=True, cache=True)
def _jit_vietoris_rips_edges_impl(points: np.ndarray, max_dist: float) -> np.ndarray:
    """Enumerate Vietoris-Rips edges within the distance cutoff."""
    n = points.shape[0]
    dim = points.shape[1]
    max_dist_sq = max_dist * max_dist

    edge_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = 0.0
            for k in range(dim):
                diff = points[i, k] - points[j, k]
                dist_sq += diff * diff
            if dist_sq <= max_dist_sq:
                edge_dists.append((i, j))

    edges = np.zeros((len(edge_dists), 2), dtype=np.int32)
    for idx in range(len(edge_dists)):
        edges[idx, 0] = edge_dists[idx][0]
        edges[idx, 1] = edge_dists[idx][1]

    return edges


@jit(nopython=True, parallel=True, cache=True)
def _jit_batch_betti_curves_impl(diagrams: np.ndarray, max_dim: int, resolution: int) -> np.ndarray:
    """Compute Betti curves for a batch of padded diagrams."""
    batch = diagrams.shape[0]
    curves = np.zeros((batch, max_dim + 1, resolution), dtype=np.int32)

    for b in prange(batch):
        pairs = diagrams[b]

        for r in range(resolution):
            threshold = r / resolution

            for i in range(pairs.shape[0]):
                if pairs[i, 0] == 0 and pairs[i, 1] == 0:
                    break

                birth = pairs[i, 0]
                death = pairs[i, 1]
                dim = int(pairs[i, 2])

                if dim <= max_dim and birth <= threshold < death:
                    curves[b, dim, r] += 1

    return curves
