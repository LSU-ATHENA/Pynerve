"""Topology-aware neural network regularizers.

Provides persistent-homology-based dropout, batch normalization,
Betti number constraints, homotopy regularization, and topological
smoothness penalties for training neural networks.
"""

from __future__ import annotations

try:
    from .persistent_dropout import (
        AdaptivePersistentDropout,
        CurricularPersistentDropout,
        FeaturePersistenceTracker,
        MultiScalePersistentDropout,
        StructuredPersistentDropout,
    )
    from .topology_constraints import (
        BettiConstraintLayer,
        HomotopyRegularizer,
        MorseRegularizer,
        PersistentBatchNorm,
        PersistentDropout,
        TopologicalSmoothness,
        TopologyPreservingDropout,
    )
except ImportError as _exc:
    _torch_missing = _exc.name == "torch" if hasattr(_exc, "name") else False
    if _torch_missing or "torch" in str(_exc):
        raise ImportError(
            "pynerve.regularization requires PyTorch. Install it with: pip install pynerve[torch]"
        ) from _exc
    raise

__all__ = [
    "AdaptivePersistentDropout",
    "BettiConstraintLayer",
    "CurricularPersistentDropout",
    "FeaturePersistenceTracker",
    "HomotopyRegularizer",
    "MorseRegularizer",
    "MultiScalePersistentDropout",
    "PersistentBatchNorm",
    "PersistentDropout",
    "StructuredPersistentDropout",
    "TopologicalSmoothness",
    "TopologyPreservingDropout",
]
