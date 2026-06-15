"""Data augmentation strategies for persistence diagrams."""

from __future__ import annotations

import torch

from .._constants import EPS_1e_6
from .._validation import validate_finite_scalar as _finite_scalar
from ._validation import _validate_ssl_diagram


class TopologyAugmentation:
    """Data augmentation strategies for persistence diagrams.

    Provides a collection of random transformations that preserve
    topological meaning while introducing controlled variation:
    birth/death perturbation, persistence scaling, small uniform
    shifts, feature dropout, and dimension shuffling.  Intended for
    self-supervised contrastive learning pipelines.
    """

    def __init__(self, min_death_gap: float = EPS_1e_6):
        """Initialise augmentation handler.

        :param min_death_gap: Minimum allowed death-minus-birth
            separation.  Must be strictly positive.
        """
        min_death_gap = _finite_scalar(min_death_gap, "min_death_gap")
        if min_death_gap <= 0:
            raise ValueError("min_death_gap must be positive")
        self.min_death_gap = min_death_gap

    def birth_death_perturbation(self, diagram: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise to birth and death coordinates.

        :param diagram: Persistence diagram tensor of shape
            ``(N, >=2)`` where columns 0/1 are birth/death.
        :param sigma: Standard deviation of the additive Gaussian
            noise.  Must be non-negative.
        :returns: Augmented diagram with the same shape and device.
        :raises ValueError: If ``diagram`` contains non-finite
            values or deaths < births.
        :raises ValueError: If ``sigma`` is negative.
        """
        _validate_ssl_diagram(diagram)
        sigma = _finite_scalar(sigma, "sigma")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")

        augmented = diagram.clone()
        augmented[:, :2] += torch.randn_like(diagram[:, :2]) * sigma
        invalid = augmented[:, 1] <= augmented[:, 0]
        augmented[invalid, 1] = augmented[invalid, 0] + self.min_death_gap
        return augmented

    def persistence_scaling(
        self, diagram: torch.Tensor, scale_range: tuple[float, float] = (0.9, 1.1)
    ) -> torch.Tensor:
        """Scale persistence (death minus birth) by a random factor.

        Each pair's midpoint is kept fixed while its persistence is
        multiplied by a scalar uniformly sampled from ``scale_range``.

        :param diagram: Persistence diagram tensor of shape
            ``(N, >=2)``.
        :param scale_range: ``(low, high)`` tuple bounding the
            uniform scaling factor.  Must satisfy ``0 < low <= high``.
        :returns: Augmented diagram with the same shape and device.
        :raises ValueError: If ``diagram`` fails validation.
        :raises ValueError: If ``scale_range`` bounds are invalid.
        """
        _validate_ssl_diagram(diagram)
        low = _finite_scalar(scale_range[0], "scale_range")
        high = _finite_scalar(scale_range[1], "scale_range")
        if low <= 0 or high < low:
            raise ValueError("scale_range must satisfy 0 < low <= high")

        scale = torch.empty((), device=diagram.device, dtype=diagram.dtype).uniform_(low, high)
        augmented = diagram.clone()
        midpoints = (diagram[:, 0] + diagram[:, 1]) / 2
        persistence = (diagram[:, 1] - diagram[:, 0]) * scale
        augmented[:, 0] = midpoints - persistence / 2
        augmented[:, 1] = midpoints + persistence / 2
        return augmented

    def small_birth_death_shift(
        self, diagram: torch.Tensor, max_shift: float = 0.05
    ) -> torch.Tensor:
        """Apply a small uniform translation to birth/death coordinates.

        Shifts all birth and death values by the same random offset,
        then clamps so that births remain non-negative.

        :param diagram: Persistence diagram tensor of shape
            ``(N, >=2)``.
        :param max_shift: Maximum absolute shift magnitude.  Must be
            non-negative.
        :returns: Augmented diagram with the same shape and device.
        :raises ValueError: If ``diagram`` fails validation.
        :raises ValueError: If ``max_shift`` is negative.
        """
        _validate_ssl_diagram(diagram)
        max_shift = _finite_scalar(max_shift, "max_shift")
        if max_shift < 0:
            raise ValueError("max_shift must be non-negative")

        shift = torch.empty((), device=diagram.device, dtype=diagram.dtype).uniform_(
            -max_shift, max_shift
        )
        augmented = diagram.clone()
        augmented[:, :2] += shift
        min_birth = augmented[:, 0].min()
        if min_birth < 0:
            augmented[:, :2] -= min_birth
        return augmented

    def feature_dropout(self, diagram: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
        """Drop low-persistence features with a given probability.

        Features below the ``drop_prob``-quantile of persistence are
        candidates; each is independently dropped with probability
        ``drop_prob``.  At least one feature is always retained.

        :param diagram: Persistence diagram tensor of shape
            ``(N, >=2)``.
        :param drop_prob: Dropout probability in ``[0, 1)``.
        :returns: Diagram with a subset of rows, same number of
            columns and device.
        :raises ValueError: If ``diagram`` fails validation.
        :raises ValueError: If ``drop_prob`` is not in ``[0, 1)``.
        """
        _validate_ssl_diagram(diagram)
        drop_prob = _finite_scalar(drop_prob, "drop_prob")
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob must be in [0, 1)")
        if diagram.shape[0] == 0:
            return diagram

        persistence = diagram[:, 1] - diagram[:, 0]
        threshold = persistence.quantile(drop_prob)
        low_persistence = persistence <= threshold
        drop = low_persistence & (torch.rand(diagram.shape[0], device=diagram.device) < drop_prob)
        keep = ~drop
        if not keep.any():
            keep[persistence.argmax()] = True
        return diagram[keep]

    def dimension_shuffle(self, diagram: torch.Tensor, shuffle_prob: float = 0.1) -> torch.Tensor:
        """Randomly flip topological dimensions of a subset of pairs.

        With probability ``shuffle_prob``, up to 10% of pairs have
        their dimension column incremented or decremented by 1
        (clamped to non-negative values).

        :param diagram: Persistence diagram tensor of shape
            ``(N, >=3)`` where column 2 holds the dimension.
        :param shuffle_prob: Probability of applying the shuffle,
            in ``[0, 1]``.
        :returns: Augmented diagram (possibly unchanged) with the
            same shape and device.
        :raises ValueError: If ``diagram`` fails validation.
        :raises ValueError: If ``shuffle_prob`` is not in ``[0, 1]``.
        """
        _validate_ssl_diagram(diagram)
        shuffle_prob = _finite_scalar(shuffle_prob, "shuffle_prob")
        if not 0 <= shuffle_prob <= 1:
            raise ValueError("shuffle_prob must be in [0, 1]")
        if diagram.shape[0] == 0 or diagram.shape[1] < 3:
            return diagram
        if torch.rand((), device=diagram.device) >= shuffle_prob:
            return diagram

        augmented = diagram.clone()
        n_swap = max(1, int(diagram.shape[0] * 0.1))
        indices = torch.randperm(diagram.shape[0], device=diagram.device)[:n_swap]
        direction = torch.randint(0, 2, (n_swap,), device=diagram.device, dtype=torch.long) * 2 - 1
        new_dims = (augmented[indices, 2].long() + direction).clamp(min=0)
        augmented[indices, 2] = new_dims.to(dtype=diagram.dtype)
        return augmented

    def __call__(self, diagram: torch.Tensor) -> torch.Tensor:
        """Randomly apply one of the available augmentations.

        :param diagram: Persistence diagram tensor of shape
            ``(N, >=2)``.
        :returns: Augmented diagram.
        """
        augmentations = (
            lambda d: self.birth_death_perturbation(d, sigma=0.02),
            self.persistence_scaling,
            self.small_birth_death_shift,
            lambda d: self.feature_dropout(d, drop_prob=0.1),
            self.dimension_shuffle,
        )
        idx = int(torch.randint(len(augmentations), (1,), device=diagram.device).item())
        return augmentations[idx](diagram)
