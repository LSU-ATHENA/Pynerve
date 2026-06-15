"""Topology-based curriculum learning utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from torch.utils.data import Dataset, Sampler

import torch

from .._constants import EPS
from .._torch_diagrams import persistence as _persistence
from .._torch_diagrams import validate_diagram as _validate_diagram
from .._validation import validate_finite_scalar as _finite_scalar
from ._curriculum_specialized import BettiCurriculum, FiltrationCurriculum  # noqa: F401
from ._curriculum_trainer import TopologicalCurriculumTrainer  # noqa: F401


class ComplexityMeasure(Enum):
    """Measures of topological complexity for curriculum learning.

    Each value defines a strategy for ordering data by difficulty based on
    persistence diagram properties.
    """

    TOTAL_PERSISTENCE = "total_persistence"
    NUM_FEATURES = "num_features"
    MAX_PERSISTENCE = "max_persistence"
    PERSISTENCE_ENTROPY = "persistence_entropy"
    BETTI_TOTAL = "betti_total"
    HOMOLOGY_DIMENSION = "max_homology_dim"
    MORSE_COMPLEXITY = "morse_complexity"


@dataclass
class CurriculumConfig:
    """Configuration for topology-based curriculum learning.

    :param complexity_measure: Measure used to rank samples by difficulty.
    :param num_stages: Number of curriculum stages.
    :param schedule: Stage progression schedule (``"linear"``, ``"exponential"``, ``"step"``).
    :param stage_ratio: Ratio determining the shape of the schedule.
    :param persistence_threshold: Threshold below which persistence is ignored.
    :param warmup_epochs: Number of epochs before stage advancement is considered.
    :param evaluation_frequency: How often evaluation runs during training.
    :raises ValueError: If any configuration value is invalid.
    """

    complexity_measure: ComplexityMeasure = ComplexityMeasure.TOTAL_PERSISTENCE
    num_stages: int = 5
    schedule: str = "linear"
    stage_ratio: float = 0.2
    persistence_threshold: float = 0.1
    warmup_epochs: int = 1
    evaluation_frequency: int = 1

    def __post_init__(self) -> None:
        """Validate configuration values after dataclass initialization.

        :raises ValueError: If any parameter is outside its valid range.
        """
        self.stage_ratio = _finite_scalar(self.stage_ratio, "stage_ratio")
        self.persistence_threshold = _finite_scalar(
            self.persistence_threshold, "persistence_threshold"
        )
        if self.num_stages <= 0:
            raise ValueError("num_stages must be positive")
        if self.schedule not in {"linear", "exponential", "step"}:
            raise ValueError("schedule must be 'linear', 'exponential', or 'step'")
        if not 0 < self.stage_ratio <= 1:
            raise ValueError("stage_ratio must be in (0, 1]")
        if self.persistence_threshold < 0:
            raise ValueError("persistence_threshold must be non-negative")
        if self.warmup_epochs <= 0 or self.evaluation_frequency <= 0:
            raise ValueError("warmup_epochs and evaluation_frequency must be positive")


def _validate_curriculum_diagram(diagram: torch.Tensor) -> None:
    _validate_diagram(diagram, min_cols=3, name="diagram")
    if diagram.shape[0] == 0:
        return
    if not torch.isfinite(diagram[:, :2]).all().item():
        raise ValueError("diagram birth/death coordinates must be finite")
    if not (diagram[:, 1] >= diagram[:, 0]).all().item():
        raise ValueError("diagram deaths must be greater than or equal to births")
    dimensions = diagram[:, 2]
    if not torch.isfinite(dimensions).all().item() or (dimensions < 0).any().item():
        raise ValueError("diagram dimensions must be finite and non-negative")
    if not torch.allclose(dimensions, dimensions.round()):
        raise ValueError("diagram dimensions must be integer-valued")


def _compute_total_persistence(
    persistence: torch.Tensor, diagram: torch.Tensor, threshold: float
) -> float:
    del diagram
    significant = persistence > threshold
    return float(persistence[significant].sum().item())


def _compute_num_features(
    persistence: torch.Tensor, diagram: torch.Tensor, threshold: float
) -> float:
    del diagram
    return float((persistence > threshold).sum().item())


def _compute_max_persistence(
    persistence: torch.Tensor, diagram: torch.Tensor, threshold: float
) -> float:
    del diagram
    significant = persistence > threshold
    return float(persistence[significant].max().item()) if significant.any() else 0.0


def _compute_persistence_entropy(
    persistence: torch.Tensor, diagram: torch.Tensor, threshold: float
) -> float:
    del diagram
    significant = persistence > threshold
    if not significant.any():
        return 0.0
    values = persistence[significant]
    probabilities = values / values.sum().clamp_min(1e-12)
    return float(-(probabilities * torch.log(probabilities + EPS)).sum().item())


def _compute_betti_total(
    persistence: torch.Tensor, diagram: torch.Tensor, threshold: float
) -> float:
    del persistence
    dimensions = diagram[:, 2].long()
    significant = (diagram[:, 1] - diagram[:, 0]) > threshold
    return float(
        sum(
            ((dimensions == dim) & significant).sum().item()
            for dim in range(int(dimensions.max().item()) + 1)
        )
    )


def _compute_homology_dimension(
    persistence: torch.Tensor, diagram: torch.Tensor, threshold: float
) -> float:
    del persistence
    significant = (diagram[:, 1] - diagram[:, 0]) > threshold
    return float(diagram[significant, 2].max().item()) if significant.any() else 0.0


def _compute_morse_complexity(
    persistence: torch.Tensor, diagram: torch.Tensor, threshold: float
) -> float:
    del persistence, threshold
    return float(diagram.shape[0])


_COMPUTE_COMPLEXITY_DISPATCH: dict[
    ComplexityMeasure, Callable[[torch.Tensor, torch.Tensor, float], float]
] = {
    ComplexityMeasure.TOTAL_PERSISTENCE: _compute_total_persistence,
    ComplexityMeasure.NUM_FEATURES: _compute_num_features,
    ComplexityMeasure.MAX_PERSISTENCE: _compute_max_persistence,
    ComplexityMeasure.PERSISTENCE_ENTROPY: _compute_persistence_entropy,
    ComplexityMeasure.BETTI_TOTAL: _compute_betti_total,
    ComplexityMeasure.HOMOLOGY_DIMENSION: _compute_homology_dimension,
    ComplexityMeasure.MORSE_COMPLEXITY: _compute_morse_complexity,
}


class TopologicalComplexityCalculator:
    """Compute topological complexity of persistence diagrams.

    Supports multiple complexity measures via the :class:`ComplexityMeasure` enum.

    :param persistence_threshold: Threshold below which persistence values are ignored.
    :raises ValueError: If ``persistence_threshold`` is negative.
    """

    def __init__(self, persistence_threshold: float = 0.1):
        """Initialize the complexity calculator.

        :param persistence_threshold: Threshold below which persistence values
            are ignored.
        :raises ValueError: If ``persistence_threshold`` is negative.
        """
        persistence_threshold = _finite_scalar(persistence_threshold, "persistence_threshold")
        if persistence_threshold < 0:
            raise ValueError("persistence_threshold must be non-negative")
        self.threshold = persistence_threshold

    def compute_complexity(self, diagram: torch.Tensor, measure: ComplexityMeasure) -> float:
        """Compute the complexity of a single persistence diagram.

        :param diagram: A persistence diagram tensor with at least 3 columns.
        :param measure: The complexity measure to apply.
        :returns: The computed complexity as a float.
        :raises ValueError: If the diagram is invalid.
        """
        _validate_curriculum_diagram(diagram)
        if diagram.shape[0] == 0:
            return 0.0
        persistence = _persistence(diagram)
        return _COMPUTE_COMPLEXITY_DISPATCH[measure](persistence, diagram, self.threshold)

    def compute_batch_complexity(
        self, diagrams: list[torch.Tensor], measure: ComplexityMeasure
    ) -> list[float]:
        """Compute complexity for a batch of diagrams.

        :param diagrams: List of persistence diagram tensors.
        :param measure: The complexity measure to apply.
        :returns: List of complexity values, one per diagram.
        """
        return [self.compute_complexity(diagram, measure) for diagram in diagrams]


class TopologicalCurriculumSampler(Sampler[int]):
    """A PyTorch Sampler that orders data by topological complexity per curriculum stage.

    Each stage exposes a progressively larger subset of the data, ordered from
    least to most complex according to the configured complexity measure.

    :param dataset: The dataset to sample from.
    :param diagrams: Persistence diagrams for each dataset sample.
    :param config: Curriculum configuration.
    :param current_stage: Starting curriculum stage.
    :param seed: Random seed for reproducibility.
    :raises ValueError: If ``current_stage`` is negative.
    """

    def __init__(
        self,
        dataset: Dataset[Any],
        diagrams: list[torch.Tensor],
        config: CurriculumConfig,
        current_stage: int = 0,
        seed: int | None = None,
    ):
        """Initialize the curriculum sampler.

        :param dataset: The PyTorch dataset to sample from.
        :param diagrams: Persistence diagrams for each dataset sample.
        :param config: Curriculum configuration controlling stage progression.
        :param current_stage: Starting curriculum stage (0-indexed).
        :param seed: Random seed for reproducibility.
        :raises ValueError: If ``current_stage`` is negative.
        """
        if current_stage < 0:
            raise ValueError("current_stage must be non-negative")

        self.dataset = dataset
        self.diagrams = diagrams
        self.config = config
        self.current_stage = min(current_stage, config.num_stages - 1)
        self.rng = np.random.default_rng(seed)
        self.calculator = TopologicalComplexityCalculator(config.persistence_threshold)
        self.complexities = self.calculator.compute_batch_complexity(
            diagrams, config.complexity_measure
        )
        self.indices_by_complexity = sorted(
            range(len(diagrams)), key=lambda idx: self.complexities[idx]
        )
        self.stage_boundaries = self._compute_stage_boundaries()

    def _compute_stage_boundaries(self) -> list[int]:
        n_samples = len(self.diagrams)
        if n_samples == 0:
            return [0] * self.config.num_stages

        boundaries = []
        for stage in range(self.config.num_stages):
            if self.config.schedule == "linear":
                progress = (stage + 1) / self.config.num_stages
            elif self.config.schedule == "exponential":
                start = self.config.stage_ratio
                exponent = stage / max(self.config.num_stages - 1, 1)
                progress = min(1.0, start ** (1.0 - exponent))
            else:
                progress = 0.5 if stage < self.config.num_stages - 1 else 1.0

            boundaries.append(max(1, min(n_samples, int(np.ceil(n_samples * progress)))))

        boundaries[-1] = n_samples
        return boundaries

    def set_stage(self, stage: int) -> None:
        """Set the current curriculum stage.

        :param stage: The stage index (clamped to ``[0, num_stages-1]``).
        :raises ValueError: If ``stage`` is negative.
        """
        if stage < 0:
            raise ValueError("stage must be non-negative")
        self.current_stage = min(stage, self.config.num_stages - 1)

    def __iter__(self) -> Iterator[int]:
        """Iterate over shuffled indices for the current curriculum stage.

        :returns: Iterator yielding dataset indices in curriculum order.
        """
        max_idx = self.stage_boundaries[self.current_stage]
        indices = list(self.indices_by_complexity[:max_idx])
        self.rng.shuffle(indices)
        yield from indices

    def __len__(self) -> int:
        """Return the number of samples available in the current curriculum stage.

        :returns: Sample count for the current stage.
        """
        return self.stage_boundaries[self.current_stage]


__all__ = [
    "BettiCurriculum",
    "ComplexityMeasure",
    "CurriculumConfig",
    "FiltrationCurriculum",
    "TopologicalComplexityCalculator",
    "TopologicalCurriculumSampler",
    "TopologicalCurriculumTrainer",
]
