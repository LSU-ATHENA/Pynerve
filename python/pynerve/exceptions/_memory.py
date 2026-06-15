"""Memory-related exception hierarchy (C++-facing)."""

from __future__ import annotations

from .._error_codes import (
    E41_RESOURCE_LIMIT,
    ErrorCategory,
)
from ._base import NerveError


class NerveMemoryError(NerveError):
    """Memory error base class from C++ MemoryError."""

    error_code = E41_RESOURCE_LIMIT
    error_category = ErrorCategory.CAPACITY


class OutOfMemoryError(NerveMemoryError):
    """Out of memory error from C++ OutOfMemoryError."""

    error_code = E41_RESOURCE_LIMIT


class AllocationError(NerveMemoryError):
    """Allocation error from C++ AllocationError."""

    error_code = E41_RESOURCE_LIMIT
