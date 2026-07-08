"""Pynerve Python API for persistence and topology computations."""

from __future__ import annotations

import logging
import os
from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._compute_api import (
    compute_persistence,
    compute_persistence_ph0,
    compute_persistence_ph3,
    compute_persistence_up_to_dim_4,
    compute_persistence_up_to_dim_5,
    compute_persistence_up_to_dim_6,
    update_persistence,
)
from ._compute_core import PersistenceResult
from ._error_codes import (
    SUCCESS,
    UNKNOWN,
    ErrorCategory,
    ErrorSeverity,
)
from ._error_translation import translate_cpp_exception
from ._fallback_classes import (
    EventType,
    PersistenceBackend,
    PersistenceEngine,
    PersistenceMode,
    PersistenceOptions,
    PH5PH6Config,
    PH5PH6Engine,
    PH5PH6Metrics,
)
from ._image_utils import persistence_image
from ._types import (
    ClusteringAlgorithm,
    DistanceMatrix,
    DistanceMetric,
    FilterFunction,
    PersistenceComputer,
    PersistenceDiagramLike,
    PersistencePair,
    PointCloud,
    VectorizationMethod,
)
from .exceptions import (  # noqa: F401
    AllocationError,
    BackendRequiredError,
    BettiError,
    BudgetExceededError,
    ConvergenceError,
    DeterminismError,
    DeviceError,
    DimensionError,
    DtypeError,
    GPUError,
    GPULaunchError,
    GPUMemoryError,
    InvalidArgumentError,
    InvalidSimplexError,
    MatrixStructureError,
    NerveError,
    NerveIOError,
    NerveMemoryError,
    NUMAError,
    NumericalError,
    NumericalInstabilityError,
    OutOfMemoryError,
    PersistenceError,
    PrecisionError,
    ShapeError,
    ShapeMismatchError,
    TypeMismatchError,
    ValidationError,
)
from .formats import Diagram, DiagramLike

logger = logging.getLogger(__name__)

try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError):
    try:
        from importlib.metadata import version as _importlib_version

        __version__ = _importlib_version("pynerve")
    except Exception:
        __version__ = "0.0.0+unknown"

_local_core: Any = None
_core_import_error: Any = None
try:
    import pynerve_internal as _local_core  # type: ignore[no-redef]
except (ModuleNotFoundError, ImportError):
    try:
        import nerve_internal as _local_core  # type: ignore[no-redef]
    except (ModuleNotFoundError, ImportError) as exc:
        _core_import_error = exc
        logger.debug(
            "C++ extension not found. "
            "Only pure-Python sub-packages (nn, torch, formats, etc.) are available. "
            "compute_persistence() will raise BackendRequiredError at call time. "
            "Install with: pip install -e ./python"
        )

_pytorch: Any = None
try:
    import torch as _torch_module

    _pytorch = _torch_module
except (ImportError, OSError):
    _pytorch = None

# All Python-side classes from _fallback_classes are used directly.
# The C++ bridge (_to_internal_options in _compute_core.py) handles
# conversion to C++ PersistenceOptions when the extension is available.
# This avoids maintaining duplicate class hierarchies in two places.

_core: Any = _local_core

_CUBLAS_CONFIGURED = False


def _ensure_cublas_config() -> None:
    """Set CUBLAS_WORKSPACE_CONFIG on first CUDA use instead of at import time."""
    if _CUBLAS_CONFIGURED:
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    globals()["_CUBLAS_CONFIGURED"] = True


_PUBLIC_MODULES = {
    "algorithms",
    "async_api",
    "benchmark",
    "cache",
    "cupy_ops",
    "datasets",
    "diagnostics",
    "diff",
    "fast_ops",
    "formats",
    "jit",
    "mapper",
    "merge",
    "mp_shared",
    "nn",
    "numba_kernels",
    "pipeline",
    "random",
    "regularization",
    "ssl",
    "torch",
    "training",
}

if TYPE_CHECKING:
    from . import algorithms as algorithms
    from . import async_api as async_api
    from . import benchmark as benchmark
    from . import cache as cache
    from . import cupy_ops as cupy_ops
    from . import datasets as datasets
    from . import diagnostics as diagnostics
    from . import diff as diff
    from . import fast_ops as fast_ops
    from . import formats as formats
    from . import jit as jit
    from . import mapper as mapper
    from . import merge as merge
    from . import mp_shared as mp_shared
    from . import nn as nn
    from . import numba_kernels as numba_kernels
    from . import pipeline as pipeline
    from . import random as random
    from . import regularization as regularization
    from . import ssl as ssl
    from . import torch as torch
    from . import training as training


def __getattr__(name: str) -> Any:
    if name in _PUBLIC_MODULES:
        try:
            if name in ("cupy_ops", "torch"):
                _ensure_cublas_config()
            module = import_module(f"{__name__}.{name}")
        except ImportError as exc:
            raise ImportError(
                f"Cannot import pynerve.{name}: the submodule or its dependencies "
                f"may not be installed. Run 'pip install pynerve[all]' for full features. "
                f"Original error: {exc}"
            ) from exc
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(object.__dir__(__import__(__name__))))


__all__ = [
    "__version__",
    "PersistenceMode",
    "PersistenceBackend",
    "PersistenceEngine",
    "PH5PH6Config",
    "PH5PH6Metrics",
    "PH5PH6Engine",
    "EventType",
    "PersistenceOptions",
    "PersistenceResult",
    "Diagram",
    "DiagramLike",
    "compute_persistence",
    "compute_persistence_ph0",
    "compute_persistence_ph3",
    "compute_persistence_up_to_dim_4",
    "compute_persistence_up_to_dim_5",
    "compute_persistence_up_to_dim_6",
    "persistence_image",
    "update_persistence",
    "PersistenceDiagramLike",
    "PersistenceComputer",
    "FilterFunction",
    "ClusteringAlgorithm",
    "DistanceMetric",
    "VectorizationMethod",
    "PointCloud",
    "DistanceMatrix",
    "PersistencePair",
    "ErrorCategory",
    "ErrorSeverity",
    "SUCCESS",
    "UNKNOWN",
    "NerveError",
    "ValidationError",
    "ShapeError",
    "DtypeError",
    "DeviceError",
    "BackendRequiredError",
    "PersistenceError",
    "GPUError",
    "GPUMemoryError",
    "GPULaunchError",
    "NerveMemoryError",
    "OutOfMemoryError",
    "AllocationError",
    "NumericalError",
    "ConvergenceError",
    "PrecisionError",
    "NumericalInstabilityError",
    "InvalidArgumentError",
    "BudgetExceededError",
    "NerveIOError",
    "DeterminismError",
    "NUMAError",
    "BettiError",
    "ShapeMismatchError",
    "DimensionError",
    "TypeMismatchError",
    "InvalidSimplexError",
    "MatrixStructureError",
    "translate_cpp_exception",
    "algorithms",
    "async_api",
    "benchmark",
    "cache",
    "cupy_ops",
    "datasets",
    "diagnostics",
    "diff",
    "fast_ops",
    "formats",
    "jit",
    "mapper",
    "merge",
    "mp_shared",
    "nn",
    "numba_kernels",
    "pipeline",
    "random",
    "regularization",
    "ssl",
    "torch",
    "training",
]
