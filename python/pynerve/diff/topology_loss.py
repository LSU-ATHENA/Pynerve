"""Differentiable topology losses for diagram-level optimization.

This module is a re-export facade. The implementation lives in submodules:
  - ``_loss_helpers``       --  validation helpers
  - ``_diagram_distances``  --  ``PersistenceLoss`` class
  - ``_loss_modules``       --  ``BettiNumberLoss``, ``DiagramComplexityLoss``,
    ``StabilityLoss``, ``MultiScaleTopologyLoss``, ``LandscapeLoss``
  - ``_composite_loss``     --  ``TopologyLoss`` combined loss
"""

from ._composite_loss import TopologyLoss
from ._diagram_distances import PersistenceLoss
from ._loss_modules import (
    BettiNumberLoss,
    DiagramComplexityLoss,
    LandscapeLoss,
    MultiScaleTopologyLoss,
    StabilityLoss,
    TopologicalComplexityLoss,
)

__all__ = [
    "BettiNumberLoss",
    "DiagramComplexityLoss",
    "LandscapeLoss",
    "MultiScaleTopologyLoss",
    "PersistenceLoss",
    "StabilityLoss",
    "TopologicalComplexityLoss",
    "TopologyLoss",
]
