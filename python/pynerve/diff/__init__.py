"""Differentiable persistence homology layers.

Provides end-to-end differentiable Vietoris-Rips, Alpha, and cubical
complex constructions, persistence landscape computation, and
topology-aware loss functions for deep learning.
"""

from __future__ import annotations

try:
    from .ph_layer import (
        DifferentiableAlphaComplex,
        DifferentiableCubical,
        DifferentiableVietorisRips,
        FiltrationLearningLayer,
        LearnableFiltrationPersistence,
        compute_persistence_landscape,
        persistence_image,
    )
    from .ph_layer_module import (
        DifferentiablePersistentHomology,
        DifferentiablePHFunction,
        TopologyLoss,
        persistence_penalty,
        topology_regularizer,
    )
    from .topology_loss import (
        BettiNumberLoss,
        DiagramComplexityLoss,
        LandscapeLoss,
        MultiScaleTopologyLoss,
        PersistenceLoss,
        StabilityLoss,
        TopologicalComplexityLoss,
    )
    from .topology_loss import (
        TopologyLoss as CombinedTopologyLoss,
    )
except ImportError as _exc:
    _torch_missing = _exc.name == "torch" if hasattr(_exc, "name") else False
    if _torch_missing or "torch" in str(_exc):
        raise ImportError(
            "pynerve.diff requires PyTorch. Install it with: pip install pynerve[torch]"
        ) from _exc
    raise

__all__ = [
    "BettiNumberLoss",
    "CombinedTopologyLoss",
    "DiagramComplexityLoss",
    "DifferentiableAlphaComplex",
    "DifferentiableCubical",
    "DifferentiablePHFunction",
    "DifferentiablePersistentHomology",
    "DifferentiableVietorisRips",
    "FiltrationLearningLayer",
    "LandscapeLoss",
    "LearnableFiltrationPersistence",
    "MultiScaleTopologyLoss",
    "PersistenceLoss",
    "StabilityLoss",
    "TopologicalComplexityLoss",
    "TopologyLoss",
    "compute_persistence_landscape",
    "persistence_image",
    "persistence_penalty",
    "topology_regularizer",
]
