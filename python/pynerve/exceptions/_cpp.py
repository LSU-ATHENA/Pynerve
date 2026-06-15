"""Remaining C++-facing exception classes."""

from __future__ import annotations

from typing import Any

from .._error_codes import (
    E00_IO_TIMEOUT,
    E30_DET_MISMATCH,
    E50_PH_ABORT,
    E53_PH4_BUDGET_EXCEEDED,
    E54_PH4_INVALID_INPUT,
    E60_NUMA_BIND_FAIL,
    E85_MATRIX_STRUCTURE,
    E87_INVALID_BETTI_NUMBERS,
    E88_INVALID_SIMPLICES,
    E91_SHAPE_ERROR,
    ErrorCategory,
)
from .._validation import validate_optional_string as _validate_optional_string
from ._base import NerveError


class PersistenceError(NerveError):
    """Persistence computation error."""

    error_code = E50_PH_ABORT
    error_category = ErrorCategory.ALGORITHMIC

    def __init__(
        self, message: str, backend: str | None = None, operation: str | None = None, **kwargs: Any
    ) -> None:
        backend = _validate_optional_string(backend, "backend")
        operation = _validate_optional_string(operation, "operation")
        super().__init__(message, **kwargs)
        self.backend = backend
        self.operation = operation

    def __repr__(self) -> str:
        base = NerveError.__repr__(self)
        parts = []
        if self.backend is not None:
            parts.append(f"backend={self.backend!r}")
        if self.operation is not None:
            parts.append(f"operation={self.operation!r}")
        if not parts:
            return base
        return f"{base} {', '.join(parts)}"


class ShapeMismatchError(NerveError):
    """Shape mismatch error from C++ ShapeMismatchError."""

    error_code = E91_SHAPE_ERROR
    error_category = ErrorCategory.ALGORITHMIC


class DimensionError(NerveError):
    """Dimension error from C++ DimensionError."""

    error_code = E91_SHAPE_ERROR
    error_category = ErrorCategory.ALGORITHMIC


class TypeMismatchError(NerveError):
    """Type mismatch error from C++ TypeError."""

    error_code = E54_PH4_INVALID_INPUT
    error_category = ErrorCategory.PH4_RESEARCH


class InvalidSimplexError(NerveError):
    """Invalid simplex error from C++ InvalidSimplexError."""

    error_code = E88_INVALID_SIMPLICES
    error_category = ErrorCategory.ALGORITHMIC


class MatrixStructureError(NerveError):
    """Matrix structure error from C++ MatrixStructureError."""

    error_code = E85_MATRIX_STRUCTURE
    error_category = ErrorCategory.ALGORITHMIC


class InvalidArgumentError(NerveError):
    """Invalid argument error from C++ InvalidArgumentError."""

    error_code = E54_PH4_INVALID_INPUT
    error_category = ErrorCategory.PH4_RESEARCH

    def __init__(
        self,
        message: str,
        parameter: str | None = None,
        expected: str | None = None,
        actual: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.parameter = parameter
        self.expected = expected
        self.actual = actual

    def __repr__(self) -> str:
        base = NerveError.__repr__(self)
        parts = []
        if self.parameter is not None:
            parts.append(f"parameter={self.parameter!r}")
        if self.expected is not None:
            parts.append(f"expected={self.expected!r}")
        if self.actual is not None:
            parts.append(f"actual={self.actual!r}")
        if not parts:
            return base
        return f"{base} {', '.join(parts)}"


class BudgetExceededError(NerveError):
    """Budget exceeded error from C++ BudgetExceededError."""

    error_code = E53_PH4_BUDGET_EXCEEDED
    error_category = ErrorCategory.PH4_RESEARCH


class NerveIOError(NerveError):
    """IO error from C++ IOError."""

    error_code = E00_IO_TIMEOUT
    error_category = ErrorCategory.IO_INFRA


class DeterminismError(NerveError):
    """Determinism error from C++ DeterminismError."""

    error_code = E30_DET_MISMATCH
    error_category = ErrorCategory.DETERMINISM


class NUMAError(NerveError):
    """NUMA error from C++ NUMAError."""

    error_code = E60_NUMA_BIND_FAIL
    error_category = ErrorCategory.NUMA_AFFINITY


class BettiError(NerveError):
    """Betti number error from C++ BettiError."""

    error_code = E87_INVALID_BETTI_NUMBERS
    error_category = ErrorCategory.ALGORITHMIC
