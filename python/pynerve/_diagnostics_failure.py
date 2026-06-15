"""Failure diagnosis helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._validation import (
    validate_nonempty_string as _validate_nonempty_string,
)
from .exceptions import (
    BackendRequiredError,
    ConvergenceError,
    DeviceError,
    GPUMemoryError,
    NerveIOError,
    NumericalError,
    OutOfMemoryError,
    ShapeError,
)


class FailureDiagnosis(str):
    """A diagnosis string with structured attributes for programmatic access.

    Behaves like a ``str`` (for display) but also has ``cause_category``,
    ``suggestions``, ``data_info``, and ``context`` attributes.
    """

    def __new__(
        cls,
        message: str,
        cause_category: str = "unknown",
        suggestions: list[str] | None = None,
        data_info: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> FailureDiagnosis:
        instance = super().__new__(cls, message)
        instance.cause_category = cause_category  # type: ignore[attr-defined]
        instance.suggestions = suggestions or []  # type: ignore[attr-defined]
        instance.data_info = data_info  # type: ignore[attr-defined]
        instance.context = context  # type: ignore[attr-defined]
        return instance


_CAUSE_SUGGESTIONS: list[tuple[type, str, list[str]]] = [
    (
        MemoryError,
        "out_of_memory",
        [
            "Reduce number of points (n_samples)",
            "Reduce max_dim (dimension)",
            "Use streaming mode for large datasets",
            "Try GPU mode if available",
        ],
    ),
    (
        OutOfMemoryError,
        "nerve_out_of_memory",
        [
            "Reduce number of points or max_dim",
            "Use streaming mode (pynerve.async_api.stream_persistence)",
            "Free GPU memory or switch to CPU",
            "Increase system swap or use memory-mapped persistence",
        ],
    ),
    (
        GPUMemoryError,
        "gpu_out_of_memory",
        [
            "Reduce batch size",
            "Free GPU memory with torch.cuda.empty_cache()",
            "Use a smaller max_radius or max_dim",
            "Fall back to CPU implementation",
        ],
    ),
    (
        BackendRequiredError,
        "backend_missing",
        [
            "Install pynerve from source: pip install -e .",
            "Install a pre-built wheel from PyPI",
            "Ensure C++ compiler and CMake are available",
        ],
    ),
    (
        ShapeError,
        "shape_mismatch",
        [
            "Check input dimensions (expected [n, dim] for point clouds)",
            "Verify distance matrix is square",
            "Ensure batch dimensions are consistent",
        ],
    ),
    (
        ConvergenceError,
        "no_convergence",
        [
            "Check for NaN or Inf in input data",
            "Reduce max_dim or max_radius",
            "Try a different persistence engine (ph5 vs ph6)",
        ],
    ),
    (
        DeviceError,
        "device_error",
        [
            "Verify CUDA device is available: torch.cuda.is_available()",
            "Check device ID is valid",
            "Ensure tensor is on the correct device",
        ],
    ),
    (
        NerveIOError,
        "io_failure",
        [
            "Verify file path exists and is readable",
            "Check file format matches expected format",
            "Ensure sufficient disk space",
        ],
    ),
    (
        NumericalError,
        "numerical_error",
        [
            "Check input data for extreme values",
            "Reduce max_radius or max_dim",
            "Use higher precision (float64)",
        ],
    ),
    (
        ValueError,
        "invalid_input",
        [
            "Check data shape (should be [n, dim] for point clouds)",
            "Check for NaN or Inf values",
            "Verify max_radius > 0",
        ],
    ),
    (
        RuntimeError,
        "runtime_failure",
        [
            "Check GPU availability if using GPU mode",
            "Try CPU implementation",
            "Update Pynerve to latest version",
        ],
    ),
]


def _classify(exception: Exception) -> tuple[str, list[str]]:
    for exc_type, category, suggestions in _CAUSE_SUGGESTIONS:
        if isinstance(exception, exc_type):
            return category, suggestions
    return "unknown", [
        "Check input data format",
        "Try smaller dataset first",
        "Report issue with full traceback",
    ]


def _build_message(
    operation: str,
    exception: Exception,
    cause_category: str,
    suggestions: list[str],
    data_info: dict[str, Any] | None,
    context_dict: dict[str, Any] | None,
) -> str:
    parts = [f"Failure in: {operation}", f"Error: {str(exception)}"]
    parts.append(f"\nPossible cause: {cause_category.replace('_', ' ').capitalize()}")
    parts.append("Suggestions:")
    for s in suggestions:
        parts.append(f"  - {s}")
    if data_info:
        parts.append("\nData info:")
        for k, v in data_info.items():
            parts.append(f"  {k}: {v}")
    if context_dict:
        parts.append("\nContext:")
        for k, v in sorted(context_dict.items()):
            parts.append(f"  {k}: {v}")
    return "\n".join(parts)


def diagnose_failure(
    operation: str,
    exception: Exception,
    data: Any | None = None,
    context: dict[str, Any] | None = None,
) -> FailureDiagnosis:
    """Classify an exception and produce a structured diagnosis.

    Matches the exception against known category patterns and returns a
    :class:`FailureDiagnosis` with human-readable suggestions, data info,
    and ambient context.

    :param operation: Name of the operation that failed.
    :param exception: The exception to diagnose.
    :param data: Optional input data; if provided, ``shape``, ``dtype``,
        and ``size`` attributes are introspected for the report.
    :param context: Optional free-form key-value pairs attached to the
        diagnosis.
    :returns: A :class:`FailureDiagnosis` string with ``cause_category``,
        ``suggestions``, ``data_info``, and ``context`` attributes.
    :raises TypeError: If ``exception`` is not an ``Exception`` or if
        ``context`` is provided but is not a mapping.
    """
    operation = _validate_nonempty_string(operation, "operation")
    if not isinstance(exception, Exception):
        raise TypeError("exception must be an Exception")
    if context is not None and not isinstance(context, Mapping):
        raise TypeError("context must be a mapping")

    cause_category, suggestions = _classify(exception)

    data_info: dict[str, Any] | None = None
    if data is not None:
        data_info = {}
        if hasattr(data, "shape"):
            data_info["shape"] = data.shape
        if hasattr(data, "dtype"):
            data_info["dtype"] = str(data.dtype)
        if hasattr(data, "size"):
            data_info["size"] = int(data.size)

    context_dict = dict(context) if context else None

    return FailureDiagnosis(
        message=_build_message(
            operation, exception, cause_category, suggestions, data_info, context_dict
        ),
        cause_category=cause_category,
        suggestions=suggestions,
        data_info=data_info,
        context=context_dict,
    )
