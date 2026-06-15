"""Training utilities for topological learning."""

from __future__ import annotations

try:
    from .curriculum import (
        BettiCurriculum,
        ComplexityMeasure,
        CurriculumConfig,
        FiltrationCurriculum,
        TopologicalComplexityCalculator,
        TopologicalCurriculumSampler,
        TopologicalCurriculumTrainer,
    )
    from .stability_reg import (
        CoherentPerturbationSampler,
        InterleavingRegularizer,
        PersistenceStabilityLoss,
        RobustTopologyTraining,
        StabilityRegularizer,
    )
    from .topology_sampler import (
        BettiBalancedSampler,
        MultiScaleTopologySampler,
        PersistenceStratifiedSampler,
        TopologyAdaptiveBatchSize,
        TopologyImportanceSampler,
    )
except ImportError as _exc:
    _torch_missing = _exc.name == "torch" if hasattr(_exc, "name") else False
    if _torch_missing or "torch" in str(_exc):
        raise ImportError(
            "pynerve.training requires PyTorch. Install it with: pip install pynerve[torch]"
        ) from _exc
    raise

__all__ = [
    "BettiBalancedSampler",
    "BettiCurriculum",
    "CoherentPerturbationSampler",
    "ComplexityMeasure",
    "CurriculumConfig",
    "FiltrationCurriculum",
    "InterleavingRegularizer",
    "MultiScaleTopologySampler",
    "PersistenceStabilityLoss",
    "PersistenceStratifiedSampler",
    "RobustTopologyTraining",
    "StabilityRegularizer",
    "TopologicalComplexityCalculator",
    "TopologicalCurriculumSampler",
    "TopologicalCurriculumTrainer",
    "TopologyAdaptiveBatchSize",
    "TopologyImportanceSampler",
]
