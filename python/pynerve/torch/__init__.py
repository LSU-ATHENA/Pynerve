"""
PyTorch-native Nerve API.

This package exposes persistence computation primitives and ML-oriented helper
modules while keeping the top-level surface close to idiomatic PyTorch usage.
"""

# pyright: reportUnsupportedDunderAll=false

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch
from torch import Tensor

try:
    import pynerve_internal as _C  # noqa: N812 # pyright: ignore[reportUnusedImport]
except ImportError:  # pragma: no cover - optional dependency
    _C = None

try:
    import pynerve_torch_internal as torch_C  # noqa: N816, N812 # pyright: ignore[reportUnusedImport]
except ImportError:  # pragma: no cover - optional dependency
    torch_C = None  # noqa: N816

from ._diagram import PersistenceDiagram  # noqa: E402


def diagram_wasserstein(
    d1: PersistenceDiagram | Tensor,
    d2: PersistenceDiagram | Tensor,
    p: float = 2.0,
    q: float = 2.0,
) -> Tensor:
    """Compute Wasserstein distance between persistence diagrams.

    Supports batched input: if both diagrams are 3D (B, N, C), returns
    a 1D tensor of B distances.
    """
    if isinstance(d1, Tensor) and d1.dim() == 3 and isinstance(d2, Tensor) and d2.dim() == 3:
        _d1_batch: Tensor = d1
        _d2_batch: Tensor = d2
        if _d1_batch.shape[0] != _d2_batch.shape[0]:
            raise ValueError("batch dimensions must match")
        return torch.stack(
            [
                diagram_wasserstein(_d1_batch[i], _d2_batch[i], p, q)
                for i in range(_d1_batch.shape[0])
            ]
        ).squeeze(-1)
    from ._distance_core import diagram_wasserstein as _wasserstein_impl  # noqa: PLC0415

    return _wasserstein_impl(d1, d2, p, q)  # type: ignore[arg-type]


def diagram_bottleneck(
    d1: PersistenceDiagram | Tensor,
    d2: PersistenceDiagram | Tensor,
) -> Tensor:
    """Compute bottleneck distance between persistence diagrams.

    Supports batched input: if both diagrams are 3D (B, N, C), returns
    a 1D tensor of B distances.
    """
    if isinstance(d1, Tensor) and d1.dim() == 3 and isinstance(d2, Tensor) and d2.dim() == 3:
        _d1_batch: Tensor = d1
        _d2_batch: Tensor = d2
        if _d1_batch.shape[0] != _d2_batch.shape[0]:
            raise ValueError("batch dimensions must match")
        return torch.stack(
            [diagram_bottleneck(_d1_batch[i], _d2_batch[i]) for i in range(_d1_batch.shape[0])]
        ).squeeze(-1)
    from ._distance_core import diagram_bottleneck as _bottleneck_impl  # noqa: PLC0415

    return _bottleneck_impl(d1, d2)  # type: ignore[arg-type]


_SUBMODULES = frozenset(
    {
        "data",
        "mapper",
        "preprocessing",
        "vectorization",
        "statistics",
        "kernels",
        "sklearn_transformers",
        "nn_layers",
        "training_utils",
        "viz",
        "tensorboard",
    }
)

_ATTR_FROM_MODULE: dict[str, str] = {
    "batch_diagrams": "pynerve.torch._diagram",
    "unbatch_diagrams": "pynerve.torch._diagram",
    "PersistenceDataset": "pynerve.torch.data",
    "PointCloudDataset": "pynerve.torch.data",
    "collate_diagrams": "pynerve.torch.data",
    "collate_point_clouds": "pynerve.torch.data",
    "MapperTransformer": "pynerve.torch.mapper",
    "visualize_mapper_graph": "pynerve.torch.mapper",
    "alpha_persistence": "pynerve.torch._persistence_api",
    "persistence_from_matrix": "pynerve.torch._persistence_api",
    "persistence_image": "pynerve.torch._persistence_api",
    "vr_persistence": "pynerve.torch._persistence_api",
    "witness_persistence": "pynerve.torch._persistence_api",
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        try:
            module = import_module(f"{__name__}.{name}")
        except ImportError as exc:
            raise ImportError(
                f"Cannot import pynerve.torch.{name}. "
                f"Ensure pynerve[torch] extras are installed. "
                f"Original error: {exc}"
            ) from exc
        globals()[name] = module
        return module
    if name in _ATTR_FROM_MODULE:
        try:
            module = import_module(_ATTR_FROM_MODULE[name])
        except ImportError as exc:
            raise ImportError(
                f"Cannot import {name} from {_ATTR_FROM_MODULE[name]}. "
                f"Ensure pynerve[torch] extras are installed. "
                f"Original error: {exc}"
            ) from exc
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "PersistenceDiagram",
    "PersistenceDataset",
    "PointCloudDataset",
    "vr_persistence",
    "witness_persistence",
    "alpha_persistence",
    "persistence_from_matrix",
    "mapper",
    "MapperTransformer",
    "visualize_mapper_graph",
    "diagram_wasserstein",
    "diagram_bottleneck",
    "persistence_image",
    "batch_diagrams",
    "unbatch_diagrams",
    "collate_diagrams",
    "collate_point_clouds",
    "preprocessing",
    "vectorization",
    "statistics",
    "kernels",
    "sklearn_transformers",
    "nn_layers",
    "training_utils",
    "data",
    "viz",
    "tensorboard",
]
