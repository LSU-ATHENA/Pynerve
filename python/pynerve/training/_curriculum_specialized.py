"""Specialized curriculum strategies: Betti and Filtration."""

from __future__ import annotations

import torch

from .._validation import validate_finite_scalar as _finite_scalar


class BettiCurriculum:
    """Curriculum that progressively introduces higher homology dimensions.

    :param max_dim: Maximum homology dimension to reach.
    :param epochs_per_dim: Number of epochs before advancing to the next dimension.
    :raises ValueError: If ``max_dim`` is negative or ``epochs_per_dim`` is not positive.
    """

    def __init__(self, max_dim: int = 3, epochs_per_dim: int = 10):
        """Initialize the Betti curriculum.

        :param max_dim: Maximum homology dimension to reach.
        :param epochs_per_dim: Number of epochs before advancing to the next dimension.
        :raises ValueError: If ``max_dim`` is negative or ``epochs_per_dim`` is not
            positive.
        """
        if max_dim < 0 or epochs_per_dim <= 0:
            raise ValueError("max_dim must be non-negative and epochs_per_dim positive")
        self.max_dim = max_dim
        self.epochs_per_dim = epochs_per_dim
        self.current_max_dim = 0

    def get_active_dimensions(self, epoch: int) -> list[int]:
        """Get homology dimensions active at a given epoch.

        :param epoch: The training epoch number.
        :returns: List of active dimension indices.
        :raises ValueError: If ``epoch`` is negative.
        """
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        stage = epoch // self.epochs_per_dim
        self.current_max_dim = min(stage, self.max_dim)
        return list(range(self.current_max_dim + 1))

    def filter_diagram_by_dim(self, diagram: torch.Tensor, epoch: int) -> torch.Tensor:
        """Filter a persistence diagram to only keep active homology dimensions.

        :param diagram: A persistence diagram tensor with at least 3 columns.
        :param epoch: The training epoch number.
        :returns: A filtered diagram containing only active dimensions.
        """
        from .curriculum import _validate_curriculum_diagram  # noqa: PLC0415

        _validate_curriculum_diagram(diagram)
        if diagram.shape[0] == 0:
            return diagram

        active_dims = torch.tensor(
            self.get_active_dimensions(epoch),
            device=diagram.device,
            dtype=diagram[:, 2].dtype,
        )
        return diagram[torch.isin(diagram[:, 2], active_dims)]


class FiltrationCurriculum:
    """Curriculum that progressively increases the Vietoris-Rips filtration radius.

    :param max_radius: Maximum filtration radius.
    :param num_stages: Number of curriculum stages.
    :param complexity_measure: How to scale the radius (``"num_simplices"`` or ``"radius"``).
    :raises ValueError: If any parameter is invalid.
    """

    def __init__(
        self,
        max_radius: float = 2.0,
        num_stages: int = 5,
        complexity_measure: str = "num_simplices",
    ):
        """Initialize the filtration curriculum.

        :param max_radius: Maximum filtration radius.
        :param num_stages: Number of curriculum stages.
        :param complexity_measure: How to scale the radius: ``"num_simplices"``
            (quadratic) or ``"radius"`` (linear).
        :raises ValueError: If any parameter is invalid.
        """
        max_radius = _finite_scalar(max_radius, "max_radius")
        if max_radius <= 0 or num_stages <= 0:
            raise ValueError("max_radius and num_stages must be positive")
        if complexity_measure not in {"num_simplices", "radius"}:
            raise ValueError("complexity_measure must be 'num_simplices' or 'radius'")

        self.max_radius = max_radius
        self.num_stages = num_stages
        self.complexity_measure = complexity_measure

    def get_stage_radius(self, stage: int) -> float:
        """Get the filtration radius for a given curriculum stage.

        :param stage: The curriculum stage index.
        :returns: The radius value for the stage.
        :raises ValueError: If ``stage`` is negative.
        """
        if stage < 0:
            raise ValueError("stage must be non-negative")
        stage = min(stage, self.num_stages - 1)
        progress = (stage + 1) / self.num_stages
        if self.complexity_measure == "num_simplices":
            return self.max_radius * progress**2
        return self.max_radius * progress

    def get_stage_threshold(self, stage: int) -> float:
        """Get the persistence threshold for a given curriculum stage.

        :param stage: The curriculum stage index.
        :returns: A threshold value that decreases with stage progression.
        :raises ValueError: If ``stage`` is negative.
        """
        if stage < 0:
            raise ValueError("stage must be non-negative")
        stage = min(stage, self.num_stages - 1)
        progress = stage / max(self.num_stages - 1, 1)
        return 0.2 * (1.0 - progress) + 0.01
