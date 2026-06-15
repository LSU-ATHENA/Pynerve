"""Sparse distance matrix building block."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np

import torch
from torch import Tensor

torch.sparse.check_sparse_tensor_invariants.disable()

_DISTANCE_ALGORITHMS = {"faiss_gpu", "faiss_cpu", "pytorch", "brute"}


class SparseDistanceMatrix:
    def __init__(
        self,
        k_neighbors: int = 50,
        algorithm: Literal["faiss_gpu", "faiss_cpu", "pytorch", "brute"] = "pytorch",
        approximate: bool = True,
        nprobe: int = 10,
    ):
        if k_neighbors <= 0 or nprobe <= 0:
            raise ValueError("k_neighbors and nprobe must be positive")
        if algorithm not in _DISTANCE_ALGORITHMS:
            raise ValueError("unknown sparse distance algorithm")

        self.k_neighbors = k_neighbors
        self.algorithm = algorithm
        self.approximate = approximate
        self.nprobe = nprobe

    def __call__(
        self, points: Tensor, return_numpy: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[np.ndarray, np.ndarray]:
        if points.dim() not in {2, 3}:
            raise ValueError(f"Expected 2D or 3D tensor, got {points.dim()}D")
        if points.shape[-2] == 0 or points.shape[-1] == 0:
            raise ValueError("points must contain at least one point and coordinate")
        if points.dim() == 3 and points.shape[0] == 0:
            raise ValueError("batched points must contain at least one batch item")

        return self._call_python(points, return_numpy)

    def _call_python(
        self, points: Tensor, return_numpy: bool
    ) -> tuple[Tensor, Tensor] | tuple[np.ndarray, np.ndarray]:
        if points.dim() == 2:
            n_points = points.shape[0]
            k = min(self.k_neighbors, n_points - 1)

            pairwise = torch.cdist(points, points)
            distances, indices = torch.topk(pairwise, k + 1, largest=False, dim=1)
            distances = distances[:, 1:]
            indices = indices[:, 1:]

        elif points.dim() == 3:
            batch_size, n_points, _ = points.shape
            k = min(self.k_neighbors, n_points - 1)

            all_distances = []
            all_indices = []

            for b in range(batch_size):
                pairwise = torch.cdist(points[b], points[b])
                dist, idx = torch.topk(pairwise, k + 1, largest=False, dim=1)
                all_distances.append(dist[:, 1:])
                all_indices.append(idx[:, 1:])

            distances = torch.stack(all_distances)
            indices = torch.stack(all_indices)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {points.dim()}D")

        if return_numpy:
            return distances.detach().cpu().numpy(), indices.detach().cpu().numpy()
        return distances, indices

    def epsilon_neighborhood(self, points: Tensor, epsilon: float) -> torch.Tensor:
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if points.dim() != 2:
            raise ValueError("epsilon_neighborhood expects a 2D point cloud")
        distances, indices = self(points)

        assert isinstance(distances, Tensor)
        assert isinstance(indices, Tensor)

        mask = distances < epsilon
        n = points.shape[0]

        row_idx = torch.arange(n, device=points.device).unsqueeze(1).expand(-1, indices.shape[1])
        row_idx = row_idx[mask]
        col_idx = indices[mask]
        values = distances[mask]

        return cast(
            Tensor,
            torch.sparse_coo_tensor(torch.stack([row_idx, col_idx]), values, (n, n)).coalesce(),
        )
