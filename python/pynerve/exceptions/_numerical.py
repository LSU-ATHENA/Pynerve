"""Numerical computation exception hierarchy (C++-facing)."""

from __future__ import annotations

from .._error_codes import (
    E20_NUM_NAN,
    E21_NUM_NO_CONVERGE,
    E71_PRECISION_LOSS,
    E73_PRECISION_CATASTROPHIC,
    ErrorCategory,
)
from ._base import NerveError


class NumericalError(NerveError):
    """Numerical error base class from C++ NumericalError."""

    error_code = E20_NUM_NAN
    error_category = ErrorCategory.NUMERICAL


class ConvergenceError(NumericalError):
    """Convergence error from C++ ConvergenceError."""

    error_code = E21_NUM_NO_CONVERGE


class PrecisionError(NumericalError):
    """Precision loss error from C++ PrecisionError."""

    error_code = E71_PRECISION_LOSS
    error_category = ErrorCategory.PRECISION


class NumericalInstabilityError(NumericalError):
    """Numerical instability error from C++ NumericalInstabilityError."""

    error_code = E73_PRECISION_CATASTROPHIC
    error_category = ErrorCategory.PRECISION
