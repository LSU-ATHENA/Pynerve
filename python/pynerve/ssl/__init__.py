"""Self-supervised learning for persistence diagrams."""

from __future__ import annotations

try:
    from .contrastive_learning import (
        BYOLTopology,
        PersistencePredictionTask,
        SimCLRTopology,
        TopologyAugmentation,
    )
    from .topology_completion import (
        BettiNumberPrediction,
        FiltrationOrderingTask,
        MultiTaskTopologySSL,
        TopologyCompletionModel,
        TopologyDenoising,
    )
except ImportError as _exc:
    _torch_missing = _exc.name == "torch" if hasattr(_exc, "name") else False
    if _torch_missing or "torch" in str(_exc):
        raise ImportError(
            "pynerve.ssl requires PyTorch. Install it with: pip install pynerve[torch]"
        ) from _exc
    raise

__all__ = [
    "BYOLTopology",
    "BettiNumberPrediction",
    "FiltrationOrderingTask",
    "MultiTaskTopologySSL",
    "PersistencePredictionTask",
    "SimCLRTopology",
    "TopologyAugmentation",
    "TopologyCompletionModel",
    "TopologyDenoising",
]
