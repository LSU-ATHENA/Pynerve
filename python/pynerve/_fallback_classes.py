"""Fallback classes used when pynerve_internal is not available."""

from __future__ import annotations

import dataclasses
import math
import numbers
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._constants import EPS_1e_9
from .exceptions import ValidationError


class PersistenceMode(Enum):
    EXACT = "EXACT"
    APPROX = "APPROX"


class PersistenceBackend(Enum):
    CPU_EXACT = "CPU_EXACT"
    CPU_ADAPTIVE_ACCELERATION = "CPU_ADAPTIVE_ACCELERATION"
    CUDA_HYBRID = "CUDA_HYBRID"


class PersistenceEngine(Enum):
    AUTO = "auto"
    PH0 = "ph0"
    PH3 = "ph3"
    PH4 = "ph4"
    PH5 = "ph5"
    PH6 = "ph6"


class EventType(Enum):
    ADD = "add"
    REMOVE = "remove"


@dataclass(frozen=True)
class PersistenceOptions:
    """Configuration for persistence computation.

    Immutable after construction. Use :meth:`replace` or
    :func:`dataclasses.replace` to create modified copies::

        opts = PersistenceOptions(max_dim=3)
        opts2 = opts.replace(max_radius=1.0)

    ``max_radius`` is ``None`` by default, meaning the engine
    auto-selects a value based on the data. Pass an explicit
    finite value to override.

    .. note::

        When the C++ extension (``pynerve_internal``) is loaded,
        this class is replaced by the C++ implementation at import time.
        Both versions expose the same attributes and validation,
        but the C++ version delegates validation to the native layer.
    """

    mode: PersistenceMode = PersistenceMode.EXACT
    backend: PersistenceBackend = PersistenceBackend.CPU_ADAPTIVE_ACCELERATION
    max_dim: int = 2
    max_radius: float | None = None
    threads: int = 0
    error_tolerance: float = 0.0

    def replace(self, **kwargs: Any) -> PersistenceOptions:
        """Return a new instance with the given fields replaced.

        This is a convenience wrapper around :func:`dataclasses.replace`
        that preserves immutability.

        Args:
            **kwargs: Fields to replace (``max_dim``, ``max_radius``,
                ``mode``, ``backend``, ``threads``, ``error_tolerance``).

        Returns:
            A new :class:`PersistenceOptions` instance with the applied changes.

        Example:
            >>> opts = PersistenceOptions(max_dim=2, max_radius=None)
            >>> opts.replace(max_dim=3)
            PersistenceOptions(mode=..., max_dim=3, ...)
        """
        return dataclasses.replace(self, **kwargs)

    def __post_init__(self) -> None:
        raw_max_dim = self.max_dim
        if isinstance(raw_max_dim, bool) or not isinstance(raw_max_dim, numbers.Integral):
            raise ValidationError("max_dim must be an integer", parameter="max_dim")
        if raw_max_dim < 0:
            raise ValidationError("max_dim must be non-negative", parameter="max_dim")
        object.__setattr__(self, "max_dim", int(raw_max_dim))
        if self.max_radius is not None:
            if not isinstance(self.max_radius, (int, float)):
                raise ValidationError("max_radius must be a number or None", parameter="max_radius")
            if not math.isfinite(self.max_radius) and self.max_radius != float("inf"):
                raise ValidationError(
                    "max_radius must be finite and non-negative (inf allowed)",
                    parameter="max_radius",
                )
            if self.max_radius < 0:
                raise ValidationError("max_radius must be non-negative", parameter="max_radius")
        raw_threads = self.threads
        if isinstance(raw_threads, bool) or not isinstance(raw_threads, numbers.Integral):
            raise ValidationError("threads must be an integer", parameter="threads")
        if raw_threads < 0:
            raise ValidationError("threads must be non-negative", parameter="threads")
        object.__setattr__(self, "threads", int(raw_threads))
        if not isinstance(self.error_tolerance, (int, float)):
            raise ValidationError("error_tolerance must be a number", parameter="error_tolerance")
        if not math.isfinite(self.error_tolerance) or self.error_tolerance < 0:
            raise ValidationError(
                "error_tolerance must be finite and non-negative", parameter="error_tolerance"
            )


@dataclass(frozen=True)
class PH5PH6Config:
    """Configuration for PH5/PH6 persistence engines.

    Defaults are tuned for double-precision point clouds up to ~10^5 points.
    For single-precision or larger datasets, relax ``numerical_tolerance``
    and reduce ``max_iterations``.
    """

    numerical_tolerance: float = EPS_1e_9
    max_iterations: int = 1000
    enable_stability_checks: bool = True
    validate_results: bool = True
    require_bitwise_reproducibility: bool = False
    enable_checksum_validation: bool = True
    computation_id: str = ""

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in dataclasses.asdict(self).items() if v)
        return f"PH5PH6Config({fields})" if fields else "PH5PH6Config()"


@dataclass
class PH5PH6Metrics:
    computation_time_ms: float = 0.0
    peak_memory_bytes: int = 0
    original_simplices: int = 0
    final_simplices: int = 0
    compression_ratio: float = 1.0
    quality_score: float = 0.0
    passed_stability_checks: bool = False
    numerical_errors: int = 0
    checksum_validation_passed: bool = False

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in dataclasses.asdict(self).items() if v)
        return f"PH5PH6Metrics({fields})" if fields else "PH5PH6Metrics()"


class PH5PH6Engine:
    def __init__(self, config: PH5PH6Config | None = None) -> None:
        self.config = config or PH5PH6Config()

    def __repr__(self) -> str:
        return f"PH5PH6Engine(config={self.config!r})"
