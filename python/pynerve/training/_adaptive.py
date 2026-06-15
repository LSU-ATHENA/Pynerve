"""Topology-adaptive batch size calculator."""

from __future__ import annotations

import numpy as np

import torch

from ._sampler_utils import _persistence, _validate_diagram


class TopologyAdaptiveBatchSize:
    """Dynamically adjust batch size based on topological complexity.

    Larger batch sizes are used for simpler diagrams; smaller batches for
    more complex ones to maintain consistent memory usage.

    :param base_batch_size: Default batch size.
    :param min_batch_size: Minimum allowed batch size.
    :param max_batch_size: Maximum allowed batch size.
    :param complexity_measure: Measure for complexity (``"num_features"``, ``"total_persistence"``, ``"num_pairs"``).
    :raises ValueError: If batch size constraints are violated or measure is unsupported.
    """

    def __init__(
        self,
        base_batch_size: int = 32,
        min_batch_size: int = 8,
        max_batch_size: int = 128,
        complexity_measure: str = "num_features",
    ):
        """Initialise the topology-adaptive batch size calculator.

        See the class docstring for parameter descriptions.
        """
        if min_batch_size <= 0 or base_batch_size <= 0 or max_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if min_batch_size > max_batch_size:
            raise ValueError("min_batch_size cannot exceed max_batch_size")
        if complexity_measure not in {"num_features", "total_persistence", "num_pairs"}:
            raise ValueError("unsupported complexity_measure")

        self.base_batch = min(max(base_batch_size, min_batch_size), max_batch_size)
        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
        self.complexity_measure = complexity_measure

    def compute_batch_size(self, diagrams: list[torch.Tensor]) -> int:
        """Compute the optimal batch size based on diagram complexities.

        :param diagrams: Persistence diagrams for the batch.
        :returns: The recommended batch size (clamped between min and max).
        """
        if not diagrams:
            return self.base_batch

        complexities = []
        for diagram in diagrams:
            _validate_diagram(diagram)
            if diagram.shape[0] == 0:
                complexities.append(0.0)
                continue

            persistence = _persistence(diagram)
            if self.complexity_measure == "num_features":
                complexities.append(float((persistence > 0.01).sum().item()))
            elif self.complexity_measure == "total_persistence":
                complexities.append(float(persistence.sum().item()))
            else:
                complexities.append(float(diagram.shape[0]))

        avg_complexity = float(np.mean(complexities))
        if avg_complexity > 50:
            return self.min_batch
        if avg_complexity > 20:
            return max(self.min_batch, self.base_batch // 2)
        if avg_complexity < 5:
            return self.max_batch
        return self.base_batch
