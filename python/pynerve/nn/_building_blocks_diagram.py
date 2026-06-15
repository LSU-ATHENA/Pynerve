"""Persistence diagram container for neural building blocks."""

from __future__ import annotations

import numpy as np

from torch import Tensor


class PersistenceDiagram:
    """Persistence diagram values returned by NN building blocks."""

    def __init__(self, births: Tensor, deaths: Tensor, dimensions: Tensor):
        if births.dim() != 1 or deaths.dim() != 1 or dimensions.dim() != 1:
            raise ValueError("births, deaths, and dimensions must be 1D tensors")
        if births.shape != deaths.shape or births.shape != dimensions.shape:
            raise ValueError("births, deaths, and dimensions must have the same shape")
        self.births = births
        self.deaths = deaths
        self.dimensions = dimensions

    def persistence_values(self) -> Tensor:
        """Get persistence (death - birth) for each point."""
        return self.deaths - self.births

    def get_dimension(self, dim: int) -> PersistenceDiagram:
        """Get diagram for specific dimension."""
        mask = self.dimensions == dim
        return PersistenceDiagram(self.births[mask], self.deaths[mask], self.dimensions[mask])

    def __len__(self) -> int:
        return len(self.births)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to numpy arrays."""
        return (
            self.births.detach().cpu().numpy(),
            self.deaths.detach().cpu().numpy(),
            self.dimensions.detach().cpu().numpy(),
        )
