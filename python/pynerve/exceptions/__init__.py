"""Nerve exception hierarchy.

Adding a new error:
    1. Define a new error code in _error_codes.py
    2. Create an exception class in the appropriate submodule
    3. Set ``error_code`` and ``error_category`` class attributes
    4. Import the class here for the public API
    5. The class is auto-registered in ``ERROR_CODE_MAP``
"""

from __future__ import annotations

from .._error_codes import UNKNOWN
from ._base import ERROR_CODE_MAP, NerveError
from ._cpp import (
    BettiError,
    BudgetExceededError,
    DeterminismError,
    DimensionError,
    InvalidArgumentError,
    InvalidSimplexError,
    MatrixStructureError,
    NerveIOError,
    NUMAError,
    PersistenceError,
    ShapeMismatchError,
    TypeMismatchError,
)
from ._gpu import GPUError, GPULaunchError, GPUMemoryError
from ._memory import AllocationError, NerveMemoryError, OutOfMemoryError
from ._numerical import ConvergenceError, NumericalError, NumericalInstabilityError, PrecisionError
from ._validation import BackendRequiredError, DeviceError, DtypeError, ShapeError, ValidationError

__all__ = [
    "NerveError",
    "ValidationError",
    "ShapeError",
    "DtypeError",
    "DeviceError",
    "BackendRequiredError",
    "PersistenceError",
    "ShapeMismatchError",
    "DimensionError",
    "InvalidSimplexError",
    "MatrixStructureError",
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
    "TypeMismatchError",
    "BettiError",
    "ERROR_CODE_MAP",
]

# Register all NerveError subclasses into ERROR_CODE_MAP.
# Iterates module-level names directly rather than depending on __all__,
# so adding a new exception class auto-registers it without updating __all__.
import sys as _sys  # noqa: E402

_this_module = _sys.modules[__name__]
for _value in list(vars(_this_module).values()):
    if isinstance(_value, type) and issubclass(_value, NerveError) and _value is not NerveError:
        _code = getattr(_value, "error_code", None)
        if _code is not None and _code != UNKNOWN:
            ERROR_CODE_MAP.setdefault(_code, _value)
del _sys, _this_module
