"""Base Nerve exception class and shared helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._error_codes import (
    UNKNOWN,
    ErrorCategory,
    ErrorSeverity,
)
from .._validation import validate_nonempty_string as _validate_nonempty_string  # noqa: PLC0415
from .._validation import validate_optional_string as _validate_optional_string  # noqa: PLC0415


def _validate_message(message: str) -> str:
    return _validate_nonempty_string(message, "message")


def _validate_details(details: dict[str, Any] | None) -> dict[str, Any]:
    if details is None:
        return {}
    if not isinstance(details, Mapping):
        raise TypeError("details must be a mapping")
    return dict(details)


_CATEGORY_NAMES: dict[int, str] = {
    ErrorCategory.SUCCESS: "success",
    ErrorCategory.IO_INFRA: "io",
    ErrorCategory.GPU_COMPUTE: "gpu",
    ErrorCategory.NUMERICAL: "numerical",
    ErrorCategory.DETERMINISM: "determinism",
    ErrorCategory.CAPACITY: "capacity",
    ErrorCategory.ALGORITHMIC: "algorithm",
    ErrorCategory.OPERATIONAL: "operational",
    ErrorCategory.PH4_RESEARCH: "research",
    ErrorCategory.PH5_PH6_HIGHDIM: "highdim",
    ErrorCategory.NUMA_AFFINITY: "numa",
    ErrorCategory.PRECISION: "precision",
    ErrorCategory.UNKNOWN_CATEGORY: "unknown",
}


class NerveError(ValueError):
    """Base exception for all Nerve errors."""

    error_code: int = UNKNOWN
    error_category: int = ErrorCategory.UNKNOWN_CATEGORY
    severity: int = ErrorSeverity.ERROR

    def __init__(
        self,
        message: str = "",
        details: dict[str, Any] | None = None,
        cpp_message: str | None = None,
    ):
        message = _validate_message(message) if message else "Nerve error"
        cpp_validated: str | None = None
        if cpp_message:
            cpp_validated = _validate_optional_string(cpp_message, "cpp_message")
        self.cpp_message: str = cpp_validated or message
        self.details = _validate_details(details)
        category_name: str = _CATEGORY_NAMES.get(int(self.error_category), "unknown")
        formatted = (
            f"[{self.__class__.__name__}] {message} "
            f"(category={category_name}, code=0x{self.error_code:08X})"
        )
        super().__init__(formatted)

    def __repr__(self) -> str:
        category_name: str = _CATEGORY_NAMES.get(int(self.error_category), "unknown")
        return f"{self.__class__.__name__}(category={category_name}, code=0x{self.error_code:08X})"


ERROR_CODE_MAP: dict[int, type[NerveError]] = {}
