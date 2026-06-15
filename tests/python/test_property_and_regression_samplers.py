from __future__ import annotations

import pytest
from pynerve.exceptions import ValidationError

try:
    import hypothesis
    from hypothesis import strategies
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent.
    hypothesis = None
    strategies = None


def test_topology_samplers_reject_invalid_numeric_inputs(torch) -> None:
    from pynerve.training.topology_sampler import (
        BettiBalancedSampler,
        MultiScaleTopologySampler,
        PersistenceStratifiedSampler,
        TopologyImportanceSampler,
    )

    diagrams = [
        torch.tensor([[0.0, 1.0, 0.0], [0.25, 0.75, 1.0]], dtype=torch.float32),
        torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float32),
    ]

    assert len(PersistenceStratifiedSampler(diagrams, batch_size=1)) == 2, (
        f"expected 2, got {len(PersistenceStratifiedSampler(diagrams, batch_size=1))}"
    )
    assert len(BettiBalancedSampler(diagrams, batch_size=1)) == 2, (
        f"expected 2, got {len(BettiBalancedSampler(diagrams, batch_size=1))}"
    )
    assert len(MultiScaleTopologySampler(diagrams, batch_size=1)) == 2, (
        f"expected 2, got {len(MultiScaleTopologySampler(diagrams, batch_size=1))}"
    )
    empty_importance = TopologyImportanceSampler([], batch_size=1)
    empty_importance.update_weights([])
    assert len(empty_importance) == 0, f"expected 0, got {len(empty_importance)}"
    with pytest.raises(ValidationError, match="persistence_threshold"):
        PersistenceStratifiedSampler(diagrams, persistence_threshold=float("nan"))
    with pytest.raises(ValidationError, match="birth"):
        PersistenceStratifiedSampler(
            [torch.tensor([[float("nan"), 1.0, 0.0]], dtype=torch.float32)]
        )
    with pytest.raises(ValidationError, match="deaths"):
        BettiBalancedSampler([torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)])
    with pytest.raises(ValueError, match="dimensions"):
        BettiBalancedSampler([torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float32)])
    with pytest.raises(ValidationError, match="novelty_threshold"):
        TopologyImportanceSampler(diagrams, novelty_threshold=float("nan"))
    with pytest.raises(ValidationError, match="scales"):
        MultiScaleTopologySampler(diagrams, scales=[float("inf")])


def test_topological_curriculum_rejects_invalid_numeric_inputs(torch) -> None:
    from pynerve.training.curriculum import (
        BettiCurriculum,
        ComplexityMeasure,
        CurriculumConfig,
        FiltrationCurriculum,
        TopologicalComplexityCalculator,
        TopologicalCurriculumSampler,
        TopologicalCurriculumTrainer,
    )
    from torch.utils.data import TensorDataset

    diagrams = [
        torch.tensor([[0.0, 1.0, 0.0], [0.25, 0.75, 1.0]], dtype=torch.float32),
        torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float32),
    ]
    dataset = TensorDataset(torch.ones(2, 1), torch.tensor([0, 1]))
    config = CurriculumConfig(num_stages=2, persistence_threshold=0.1)
    calculator = TopologicalComplexityCalculator(config.persistence_threshold)

    assert calculator.compute_complexity(diagrams[0], ComplexityMeasure.TOTAL_PERSISTENCE) > 0, (
        f"expected positive complexity, got {calculator.compute_complexity(diagrams[0], ComplexityMeasure.TOTAL_PERSISTENCE)}"
    )
    assert len(TopologicalCurriculumSampler(dataset, diagrams, config)) >= 1, (
        f"expected >= 1, got {len(TopologicalCurriculumSampler(dataset, diagrams, config))}"
    )
    assert BettiCurriculum(max_dim=1).filter_diagram_by_dim(diagrams[0], 1).shape[1] == 3, (
        f"expected column count 3, got {BettiCurriculum(max_dim=1).filter_diagram_by_dim(diagrams[0], 1).shape[1]}"
    )
    assert FiltrationCurriculum(max_radius=2.0, num_stages=2).get_stage_radius(1) == 2.0, (
        f"expected 2.0, got {FiltrationCurriculum(max_radius=2.0, num_stages=2).get_stage_radius(1)}"
    )
    with pytest.raises(ValidationError, match="stage_ratio"):
        CurriculumConfig(stage_ratio=float("nan"))
    with pytest.raises(ValidationError, match="persistence_threshold"):
        TopologicalComplexityCalculator(float("inf"))
    with pytest.raises(ValueError, match="birth"):
        calculator.compute_complexity(
            torch.tensor([[float("nan"), 1.0, 0.0]], dtype=torch.float32),
            ComplexityMeasure.TOTAL_PERSISTENCE,
        )
    with pytest.raises(ValueError, match="deaths"):
        calculator.compute_complexity(
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            ComplexityMeasure.TOTAL_PERSISTENCE,
        )
    with pytest.raises(ValueError, match="integer"):
        calculator.compute_complexity(
            torch.tensor([[0.0, 1.0, 0.5]], dtype=torch.float32),
            ComplexityMeasure.TOTAL_PERSISTENCE,
        )
    trainer = TopologicalCurriculumTrainer(
        torch.nn.Linear(1, 1), config, stage_advancement_criterion="performance"
    )
    with pytest.raises(ValidationError, match="validation_score"):
        trainer.should_advance_stage(float("nan"))
    with pytest.raises(ValueError, match="num_workers"):
        trainer.create_dataloader(dataset, diagrams, num_workers=-1)
    with pytest.raises(ValidationError, match="max_radius"):
        FiltrationCurriculum(max_radius=float("nan"))
