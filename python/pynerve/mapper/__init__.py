"""Topological Mapper graph components.

Provides differentiable Mapper constructions, soft clustering,
learnable lens functions, and Mapper-based graph neural network
layers for topological feature learning.
"""

from __future__ import annotations

try:
    from .learnable_mapper import (
        AdaptiveCover,
        DifferentiableMapper,
        LensFunction,
        MapperAutoencoder,
        MapperGraphEncoder,
        SoftClusterAssignment,
    )
    from .mapper_gnn import (
        HierarchicalMapperPooling,
        MapperGNNClassifier,
        MapperGraphConv,
        MapperNodeEncoder,
        TopologyAwareReadout,
    )
except ImportError as _exc:
    _torch_missing = _exc.name == "torch" if hasattr(_exc, "name") else False
    if _torch_missing or "torch" in str(_exc):
        raise ImportError(
            "pynerve.mapper requires PyTorch. Install it with: pip install pynerve[torch]"
        ) from _exc
    raise

__all__ = [
    "AdaptiveCover",
    "DifferentiableMapper",
    "HierarchicalMapperPooling",
    "LensFunction",
    "MapperAutoencoder",
    "MapperGNNClassifier",
    "MapperGraphConv",
    "MapperGraphEncoder",
    "MapperNodeEncoder",
    "SoftClusterAssignment",
    "TopologyAwareReadout",
]
