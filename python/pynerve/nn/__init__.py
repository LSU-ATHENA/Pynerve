"""Neural-network components for topological learning."""

from __future__ import annotations

try:
    from .building_blocks import (
        PersistenceDiagram,
        PersistenceSketch,
        SparseDistanceMatrix,
        SparseRipsPersistence,
        WitnessComplexPersistence,
    )
    from .diagram_conv import (
        DiagramConv1D,
        DiagramConvNet,
        DiagramDeepSet,
        DiagramMultiHeadAttention,
        DiagramPooling,
        DiagramTransformerBlock,
    )
    from .persistent_homology import (
        PersistentHomology,
        PersistentHomologyFunction,
        compute_persistence_diagrams,
    )
    from .sparse_ph import (
        SparsePH,
        TopologyAttention,
        WindowedPH,
        compute_witness_persistence,
        farthest_point_sampling,
    )
    from .topo_regularization import (
        DiagramMatchingLoss,
        PersistenceEntropyLoss,
        TopologicalComplexityLoss,
        TopologicalRegularizationLoss,
    )
except ImportError as _exc:
    _torch_missing = _exc.name == "torch" if hasattr(_exc, "name") else False
    if _torch_missing or "torch" in str(_exc):
        raise ImportError(
            "pynerve.nn requires PyTorch. Install it with: pip install pynerve[torch]"
        ) from _exc
    raise

__all__ = [
    "PersistentHomology",
    "PersistentHomologyFunction",
    "compute_persistence_diagrams",
    "SparsePH",
    "WindowedPH",
    "farthest_point_sampling",
    "compute_witness_persistence",
    "SparseDistanceMatrix",
    "SparseRipsPersistence",
    "WitnessComplexPersistence",
    "PersistenceSketch",
    "PersistenceDiagram",
    "TopologyAttention",
    "TopologicalRegularizationLoss",
    "PersistenceEntropyLoss",
    "TopologicalComplexityLoss",
    "DiagramMatchingLoss",
    "DiagramConv1D",
    "DiagramDeepSet",
    "DiagramMultiHeadAttention",
    "DiagramTransformerBlock",
    "DiagramPooling",
    "DiagramConvNet",
]
