"""Topology importance sampler based on novelty-weighted sampling."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

import torch

from .._validation import validate_finite_scalar as _finite_scalar
from ._sampler_utils import _persistence, _validate_diagram


class TopologyImportanceSampler(Sampler[int]):
    """A PyTorch Sampler based on novelty-weighted importance sampling.

    Computes a topological embedding for each diagram, then samples
    with higher probability for samples that are far from recently seen ones.

    :param diagrams: Persistence diagrams for each sample.
    :param batch_size: Desired batch size.
    :param history_length: Number of recently seen samples for novelty computation.
    :param novelty_threshold: Distance threshold for novelty saturation.
    :param seed: Random seed for reproducibility.
    :raises ValueError: If ``batch_size``, ``history_length``, or ``novelty_threshold`` is not positive.
    """

    def __init__(
        self,
        diagrams: list[torch.Tensor],
        batch_size: int = 32,
        history_length: int = 100,
        novelty_threshold: float = 0.5,
        seed: int | None = None,
    ):
        """Initialise the topology importance sampler.

        See the class docstring for parameter descriptions.
        """
        if batch_size <= 0 or history_length <= 0:
            raise ValueError("batch_size and history_length must be positive")
        novelty_threshold = _finite_scalar(novelty_threshold, "novelty_threshold")
        if novelty_threshold <= 0:
            raise ValueError("novelty_threshold must be positive")

        self.diagrams = diagrams
        self.batch_size = batch_size
        self.history_length = history_length
        self.novelty_threshold = novelty_threshold
        self.rng = np.random.default_rng(seed)
        self.embeddings = self._compute_embeddings()
        self.weights = np.ones(len(diagrams), dtype=float)
        if len(self.weights) > 0:
            self.weights /= self.weights.sum()
        self.seen_samples: list[int] = []

    def _compute_embeddings(self) -> np.ndarray:
        embeddings = []
        for diagram in self.diagrams:
            _validate_diagram(diagram)
            if diagram.shape[0] == 0:
                embeddings.append(np.zeros(8, dtype=float))
                continue

            persistence = _persistence(diagram)
            stats = [
                persistence.sum().item(),
                persistence.mean().item(),
                persistence.std(unbiased=False).item(),
                persistence.max().item(),
                float(persistence.numel()),
            ]
            for dim in range(3):
                stats.append(float((diagram[:, 2] == dim).sum().item()))
            embeddings.append(np.array(stats, dtype=float))
        return np.array(embeddings, dtype=float)

    def _compute_novelty(self, idx: int) -> float:
        if not self.seen_samples:
            return 1.0

        embedding = self.embeddings[idx]
        seen_embeddings = self.embeddings[self.seen_samples[-self.history_length :]]
        min_distance = np.linalg.norm(seen_embeddings - embedding, axis=1).min()
        return float(min(min_distance / self.novelty_threshold, 1.0))

    def update_weights(self, sampled_indices: list[int]) -> None:
        """Update sampling weights after a batch has been drawn.

        Recomputes novelty for all samples given the recently seen indices.

        :param sampled_indices: Indices of samples just drawn.
        """
        if len(self.weights) == 0:
            return
        self.seen_samples.extend(sampled_indices)
        for idx in range(len(self.diagrams)):
            self.weights[idx] = 0.5 + 0.5 * self._compute_novelty(idx)

        total = self.weights.sum()
        if total <= 0:
            self.weights[:] = 1.0 / len(self.weights)
        else:
            self.weights /= total

    def __iter__(self) -> Iterator[int]:
        """Yield sample indices via novelty-weighted importance sampling.

        :yields: Sample indices drawn according to novelty weights.
        """
        available = list(range(len(self.diagrams)))
        while available:
            size = min(self.batch_size, len(available))
            probabilities = self.weights[available]
            probabilities = probabilities / probabilities.sum()
            positions = self.rng.choice(len(available), size=size, replace=False, p=probabilities)
            batch_list = [available[int(pos)] for pos in positions]
            for pos in sorted(positions, reverse=True):
                del available[int(pos)]
            self.update_weights(batch_list)
            yield from batch_list

    def __len__(self) -> int:
        """Return the total number of samples.

        :returns: Total number of diagrams.
        """
        return len(self.diagrams)
