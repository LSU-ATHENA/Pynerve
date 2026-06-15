"""Multi-scale topology sampler."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

import torch

from .._constants import EPS
from .._validation import validate_finite_scalar as _finite_scalar
from ._sampler_utils import _persistence, _shuffle, _validate_diagram


class MultiScaleTopologySampler(Sampler[int]):
    """A PyTorch Sampler that draws batches across multiple persistence scales.

    Groups samples by their mean persistence scale (log-space matching) and
    ensures each scale is represented in every batch.

    :param diagrams: Persistence diagrams for each sample.
    :param scales: Persistence scales defining the groups (default: ``[0.01, 0.1, 0.5, 1.0]``).
    :param batch_size: Desired batch size.
    :param samples_per_scale: Number of samples to draw per scale per batch.
    :param seed: Random seed for reproducibility.
    :raises ValueError: If scales are not positive or batch sizes are invalid.
    """

    def __init__(
        self,
        diagrams: list[torch.Tensor],
        scales: list[float] | None = None,
        batch_size: int = 32,
        samples_per_scale: int = 8,
        seed: int | None = None,
    ):
        """Initialise the multi-scale topology sampler.

        See the class docstring for parameter descriptions.
        """
        self.scales = scales or [0.01, 0.1, 0.5, 1.0]
        self.scales = [_finite_scalar(scale, "scales") for scale in self.scales]
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("scales must be positive")
        if batch_size <= 0 or samples_per_scale <= 0:
            raise ValueError("batch_size and samples_per_scale must be positive")

        self.diagrams = diagrams
        self.batch_size = batch_size
        self.samples_per_scale = samples_per_scale
        self.rng = np.random.default_rng(seed)
        self.scale_groups = self._create_scale_groups()

    def _create_scale_groups(self) -> dict[float, list[int]]:
        groups: dict[float, list[int]] = {scale: [] for scale in self.scales}
        for idx, diagram in enumerate(self.diagrams):
            _validate_diagram(diagram)
            if diagram.shape[0] == 0:
                groups[self.scales[0]].append(idx)
                continue

            mean_persistence = float(_persistence(diagram).mean().item())
            closest_scale = min(
                self.scales,
                key=lambda scale: abs(np.log(scale) - np.log(mean_persistence + EPS)),
            )
            groups[closest_scale].append(idx)
        return groups

    def __iter__(self) -> Iterator[int]:
        """Yield sample indices balanced across persistence scales.

        :yields: Sample indices with equal representation per scale.
        """
        groups = {
            scale: _shuffle(indices, self.rng) for scale, indices in self.scale_groups.items()
        }
        pointers = {scale: 0 for scale in self.scales}

        while True:
            batch = []
            for scale in self.scales:
                group = groups[scale]
                start = pointers[scale]
                end = min(start + self.samples_per_scale, len(group))
                batch.extend(group[start:end])
                pointers[scale] = end

            if not batch:
                break
            yield from _shuffle(batch, self.rng)

            if all(pointers[scale] >= len(groups[scale]) for scale in self.scales):
                break

    def __len__(self) -> int:
        """Return the total number of samples.

        :returns: Total number of diagrams.
        """
        return len(self.diagrams)
