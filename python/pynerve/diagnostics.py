"""Diagnostics, profiling, and debugging helpers."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._validation import (
    validate_nonempty_string as _validate_nonempty_string,
)
from ._validation import (
    validate_nonnegative_finite as _validate_finite_nonnegative,
)
from ._validation import (
    validate_optional_finite as _validate_optional_finite,
)
from ._validation import (
    validate_optional_nonnegative_int,
)
from ._validation import (
    validate_optional_string as _validate_optional_string,
)

logger = logging.getLogger("pynerve")


@dataclass
class DiagnosticInfo:
    """Record of a single instrumented operation.

    :param operation: Name of the operation being tracked.
    :param duration: Wall-clock duration in seconds.
    :param memory_delta: Change in host memory (MB), or ``None`` if not measured.
    :param gpu_memory_delta: Change in GPU memory (MB), or ``None`` if not measured.
    :param n_points: Number of data points processed, or ``None``.
    :param n_simplices: Number of simplices produced, or ``None``.
    :param backend: Backend used (e.g. ``"cpu"``, ``"cuda"``), or ``None``.
    :param error: Error message if the operation failed, or ``None``.
    """

    operation: str
    duration: float
    memory_delta: float | None = None
    gpu_memory_delta: float | None = None
    n_points: int | None = None
    n_simplices: int | None = None
    backend: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        self.operation = _validate_nonempty_string(self.operation, "operation")
        self.duration = _validate_finite_nonnegative(self.duration, "duration")
        self.memory_delta = _validate_optional_finite(self.memory_delta, "memory_delta")
        self.gpu_memory_delta = _validate_optional_finite(self.gpu_memory_delta, "gpu_memory_delta")
        self.n_points = validate_optional_nonnegative_int(self.n_points, "n_points")
        self.n_simplices = validate_optional_nonnegative_int(self.n_simplices, "n_simplices")
        self.backend = _validate_optional_string(self.backend, "backend")
        self.error = _validate_optional_string(self.error, "error")

    def __repr__(self) -> str:
        status = "ERROR" if self.error else "OK"
        return f"[{status}] {self.operation}: {self.duration * 1000:.1f}ms"


class DiagnosticsCollector:
    """Collect timing and memory diagnostics across operations.

    Use the :meth:`track` context manager to instrument code blocks.
    Call :meth:`report` to get a formatted summary or :meth:`summary`
    for structured data.

    Example:
        >>> from pynerve.diagnostics import DiagnosticsCollector
        >>> import pynerve, numpy as np
        >>> dc = DiagnosticsCollector()
        >>> rng = np.random.default_rng(42)
        >>> points = rng.random((500, 3))
        >>> with dc.track("persistence", n_points=500):
        ...     result = pynerve.compute_persistence(points, max_dim=2)  # doctest: +SKIP
        >>> print(dc.report())  # doctest: +SKIP
    """

    def __init__(self) -> None:
        """Initialize an empty diagnostics collector.

        :returns: A new ``DiagnosticsCollector`` instance.
        """
        self._diagnostics: list[DiagnosticInfo] = []
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        n = len(self._diagnostics)
        errors = sum(1 for d in self._diagnostics if d.error)
        return f"DiagnosticsCollector(operations={n}, errors={errors})"

    @property
    def diagnostics(self) -> list[DiagnosticInfo]:
        """Return a snapshot of all collected diagnostics.

        :returns: A shallow copy of the internal diagnostics list.
        """
        with self._lock:
            return list(self._diagnostics)

    @contextmanager
    def track(self, operation: str, **kwargs: Any) -> Generator[DiagnosticInfo, None, None]:
        """Context manager that instruments a code block.

        Yields a :class:`DiagnosticInfo` object whose fields can be updated
        inside the block.  On exit the duration is recorded and the info is
        appended to the internal list.

        :param operation: A name for the operation being tracked.
        Additional keyword arguments are used to pre-populate fields on the
        ``DiagnosticInfo`` (``memory_delta``, ``gpu_memory_delta``,
            ``n_points``, ``n_simplices``, ``backend``, ``error``).
        :yields: A :class:`DiagnosticInfo` instance for in-block updates.
        :raises Exception: Any exception from the wrapped block is re-raised
            after capture.
        """
        operation = _validate_nonempty_string(operation, "operation")
        _valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in ("memory_delta", "gpu_memory_delta", "n_points", "n_simplices", "backend", "error")
        }
        _unexpected = set(kwargs) - set(_valid_kwargs)
        if _unexpected:
            import warnings as _warnings  # noqa: PLC0415

            _warnings.warn(
                f"Unexpected keyword arguments for DiagnosticInfo: {sorted(_unexpected)}. "
                f"Accepted: memory_delta, gpu_memory_delta, n_points, n_simplices, backend, error.",
                UserWarning,
                stacklevel=2,
            )
        info = DiagnosticInfo(operation=operation, duration=0.0, **_valid_kwargs)
        start_time = time.perf_counter()

        logger.debug("Starting %s", operation)
        try:
            yield info
        except Exception as e:
            info.error = str(e)
            info.duration = time.perf_counter() - start_time
            logger.error("%s failed after %.2fms: %s", operation, info.duration * 1000, e)
            with self._lock:
                self._diagnostics.append(info)
            raise
        else:
            info.duration = time.perf_counter() - start_time
            logger.debug("%s completed in %.2fms", operation, info.duration * 1000)
            with self._lock:
                self._diagnostics.append(info)

    def report(self) -> str:
        """Return a human-readable summary of all diagnostics.

        :returns: A multi-line string with status, timing, and error info
            for each recorded operation.
        """
        lines = ["Diagnostic Report", "=" * 50]

        total_time = sum(d.duration for d in self.diagnostics)

        for diag in self.diagnostics:
            status = "OK" if not diag.error else "ERROR"
            lines.append(f"{status} {diag.operation:20s} {diag.duration * 1000:8.2f}ms")

            if diag.error:
                lines.append(f"  Error: {diag.error}")

        lines.append("=" * 50)
        lines.append(f"Total time: {total_time * 1000:.2f}ms")

        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Return aggregate statistics across all diagnostics.

        Fields returned: ``n_operations``, ``n_errors``, ``total_time``,
        ``mean_time``, ``max_time``, ``error_rate``.

        :returns: A dictionary of summary metrics.  Empty dict if no
            diagnostics have been recorded.
        """
        if not self.diagnostics:
            return {}

        times = [d.duration for d in self.diagnostics if not d.error]
        errors = [d for d in self.diagnostics if d.error]

        return {
            "n_operations": len(self.diagnostics),
            "n_errors": len(errors),
            "total_time": sum(times),
            "mean_time": float(np.mean(times)) if times else 0.0,
            "max_time": float(np.max(times)) if times else 0.0,
            "error_rate": len(errors) / len(self.diagnostics) if self.diagnostics else 0,
        }


_verbose_enabled: ContextVar[bool] = ContextVar("_verbose_enabled", default=False)
_verbose_level: ContextVar[str] = ContextVar("_verbose_level", default="info")


@contextmanager
def verbose(enabled: bool = True, level: str = "info") -> Any:
    """Context manager that controls logger verbosity.

    :param enabled: When ``True`` (default) the logger level is raised to the
        requested ``level``; when ``False`` the context is a no-op.
    :param level: One of ``"info"``, ``"debug"``, or ``"trace"``.
    :yields: ``None``.
    :raises TypeError: If ``enabled`` is not a boolean.
    :raises ValueError: If ``level`` is not one of the supported values.
    """
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a boolean")
    if level not in {"info", "debug", "trace"}:
        raise ValueError("level must be 'info', 'debug', or 'trace'")
    token_enabled = _verbose_enabled.set(bool(enabled))
    token_level = _verbose_level.set(level)
    _level = logging.DEBUG if level == "debug" else logging.INFO
    logger.setLevel(_level)
    try:
        yield
    finally:
        _verbose_enabled.reset(token_enabled)
        _verbose_level.reset(token_level)


from ._diagnostics_data import DataQualityReport, check_data_quality  # noqa: E402
from ._diagnostics_failure import FailureDiagnosis, diagnose_failure  # noqa: E402
from ._diagnostics_system import (  # noqa: E402
    DebugMode,
    check_gpu_availability,
    profile_memory,
    system_info,
)

__all__ = [
    "DiagnosticsCollector",
    "DiagnosticInfo",
    "DataQualityReport",
    "FailureDiagnosis",
    "verbose",
    "diagnose_failure",
    "check_data_quality",
    "profile_memory",
    "DebugMode",
    "check_gpu_availability",
    "system_info",
]
