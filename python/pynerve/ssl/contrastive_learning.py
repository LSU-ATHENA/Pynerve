"""Self-supervised learning tasks for persistence diagrams."""

from __future__ import annotations

from ._augmentation import TopologyAugmentation
from ._byol import BYOLTopology
from ._persistence_prediction import PersistencePredictionTask
from ._simclr import SimCLRTopology

__all__ = [
    "BYOLTopology",
    "PersistencePredictionTask",
    "SimCLRTopology",
    "TopologyAugmentation",
]
