"""Betti-balanced sampler."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

import torch

from ._sampler_utils import _persistence, _shuffle, _validate_diagram


class BettiBalancedSampler(Sampler[int]):
    """A PyTorch Sampler that balances samples by Betti number signatures.

    Groups samples with similar Betti numbers and draws batches that are
    balanced across these groups.

    :param diagrams: Persistence diagrams for each sample.
    :param max_betti: Maximum Betti number considered (clamped above).
    :param batch_size: Desired batch size.
    :param balance_dimensions: Homology dimensions to balance over (default: ``[0, 1, 2]``).
    :param seed: Random seed for reproducibility.
    :raises ValueError: If ``max_betti`` is negative or ``batch_size`` is not positive.
    """

    def __init__(
        self,
        diagrams: list[torch.Tensor],
        max_betti: int = 10,
        batch_size: int = 32,
        balance_dimensions: list[int] | None = None,
        seed: int | None = None,
    ):
        """Initialise the Betti-balanced sampler.

        See the class docstring for parameter descriptions.
        """
        if max_betti < 0 or batch_size <= 0:
            raise ValueError("max_betti must be non-negative and batch_size positive")
        balance_dimensions = balance_dimensions or [0, 1, 2]
        if any(dim < 0 for dim in balance_dimensions):
            raise ValueError("balance_dimensions must be non-negative")

        self.diagrams = diagrams
        self.max_betti = max_betti
        self.batch_size = batch_size
        self.balance_dimensions = balance_dimensions
        self.rng = np.random.default_rng(seed)
        self.betti_numbers = self._compute_betti_numbers()
        self.betti_groups = self._create_betti_groups()

    def _compute_betti_numbers(self) -> list[dict[int, int]]:
        bettis = []
        for diagram in self.diagrams:
            _validate_diagram(diagram)
            if diagram.shape[0] == 0:
                bettis.append({dim: 0 for dim in self.balance_dimensions})
                continue

            persistence = _persistence(diagram)
            significant = persistence > 0.01
            dimensions = diagram[:, 2].long()
            bettis.append(
                {
                    dim: int(((dimensions == dim) & significant).sum().item())
                    for dim in self.balance_dimensions
                }
            )
        return bettis

    def _create_betti_groups(self) -> dict[tuple[int, ...], list[int]]:
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for idx, betti_dict in enumerate(self.betti_numbers):
            signature = tuple(
                min(betti_dict[dim], self.max_betti) for dim in self.balance_dimensions
            )
            groups[signature].append(idx)
        return dict(groups)

    def __iter__(self) -> Iterator[int]:
        """Yield sample indices batch by batch, balanced across Betti groups.

        :yields: Sample indices drawn round-robin across groups.
        """
        groups = {
            signature: _shuffle(indices, self.rng)
            for signature, indices in self.betti_groups.items()
        }
        signatures = list(groups)

        while signatures:
            batch: list[int] = []
            for signature in list(signatures):
                if len(batch) >= self.batch_size:
                    break
                if groups[signature]:
                    batch.append(groups[signature].pop())
                if not groups[signature]:
                    signatures.remove(signature)

            if not batch:
                break
            yield from _shuffle(batch, self.rng)

    def __len__(self) -> int:
        """Return the total number of samples.

        :returns: Total number of diagrams.
        """
        return len(self.diagrams)
