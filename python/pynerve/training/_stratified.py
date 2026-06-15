"""Persistence-stratified sampler."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

import torch

from .._validation import validate_finite_scalar as _finite_scalar
from ._sampler_utils import _persistence, _shuffle, _validate_diagram


class PersistenceStratifiedSampler(Sampler[int]):
    """A PyTorch Sampler that stratifies samples by persistence-based complexity.

    Divides samples into strata by total persistence, then draws batches
    that evenly mix samples from all strata (round-robin).

    :param diagrams: Persistence diagrams for each sample.
    :param num_strata: Number of complexity strata.
    :param batch_size: Desired batch size.
    :param persistence_threshold: Threshold below which persistence is ignored.
    :param drop_last: If ``True``, drop the last incomplete batch.
    :param seed: Random seed for reproducibility.
    :raises ValueError: If ``num_strata`` or ``batch_size`` is not positive.
    """

    def __init__(
        self,
        diagrams: list[torch.Tensor],
        num_strata: int = 5,
        batch_size: int = 32,
        persistence_threshold: float = 0.01,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        """Initialise the persistence stratified sampler.

        See the class docstring for parameter descriptions.
        """
        if num_strata <= 0 or batch_size <= 0:
            raise ValueError("num_strata and batch_size must be positive")
        persistence_threshold = _finite_scalar(persistence_threshold, "persistence_threshold")
        if persistence_threshold < 0:
            raise ValueError("persistence_threshold must be non-negative")

        self.diagrams = diagrams
        self.num_strata = num_strata
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        self.persistence_stats = self._compute_persistence_stats(persistence_threshold)
        self.strata_indices = self._create_strata()

    def _compute_persistence_stats(self, threshold: float) -> list[dict[str, float]]:
        stats = []
        for diagram in self.diagrams:
            _validate_diagram(diagram)
            if diagram.shape[0] == 0:
                stats.append({"total": 0.0, "max": 0.0, "count": 0.0, "mean": 0.0})
                continue

            persistence = _persistence(diagram)
            significant = persistence > threshold
            stats.append(
                {
                    "total": float(persistence[significant].sum().item()),
                    "max": float(persistence.max().item()),
                    "count": float(significant.sum().item()),
                    "mean": float(persistence[significant].mean().item())
                    if significant.any()
                    else 0.0,
                }
            )
        return stats

    def _create_strata(self) -> list[list[int]]:
        sorted_indices = sorted(
            range(len(self.persistence_stats)),
            key=lambda idx: self.persistence_stats[idx]["total"],
        )
        return [
            [int(idx) for idx in chunk] for chunk in np.array_split(sorted_indices, self.num_strata)
        ]

    def __iter__(self) -> Iterator[int]:
        """Yield sample indices batch by batch, stratified by complexity.

        :yields: Sample indices drawn round-robin across strata.
        """
        strata = [_shuffle(stratum, self.rng) for stratum in self.strata_indices]
        active = [idx for idx, stratum in enumerate(strata) if stratum]
        cursor = 0

        while active:
            batch: list[int] = []
            while active and len(batch) < self.batch_size:
                stratum_idx = active[cursor % len(active)]
                batch.append(strata[stratum_idx].pop())
                if not strata[stratum_idx]:
                    active.remove(stratum_idx)
                    cursor = 0
                else:
                    cursor += 1

            if len(batch) < self.batch_size and self.drop_last:
                break
            yield from _shuffle(batch, self.rng)

    def __len__(self) -> int:
        """Return the total number of samples.

        :returns: Total sample count, or a multiple of *batch_size*
            when *drop_last* is ``True``.
        """
        total = len(self.diagrams)
        if self.drop_last:
            return (total // self.batch_size) * self.batch_size
        return total
