"""User-facing input validation error hierarchy."""

from __future__ import annotations

from typing import Any

from .._error_codes import (
    E90_VALIDATION_ERROR,
    E91_SHAPE_ERROR,
    E92_DTYPE_ERROR,
    E93_DEVICE_ERROR,
    E94_BACKEND_REQUIRED,
    ErrorCategory,
)
from .._validation import (
    validate_optional_nonnegative_int as _validate_optional_nonnegative_int,
)
from .._validation import validate_shape_tuple as _validate_shape_tuple
from .._validation import validate_string_list as _validate_string_list
from ._base import NerveError


class ValidationError(NerveError):
    """Input validation error."""

    error_code = E90_VALIDATION_ERROR
    error_category = ErrorCategory.OPERATIONAL

    def __init__(
        self,
        message: str,
        parameter: str | None = None,
        expected: str | None = None,
        actual: str | None = None,
    ):
        if parameter is not None and (not isinstance(parameter, str) or not parameter):
            raise TypeError("parameter must be a non-empty string")
        if expected is not None and (not isinstance(expected, str) or not expected):
            raise TypeError("expected must be a non-empty string")
        if actual is not None and (not isinstance(actual, str) or not actual):
            raise TypeError("actual must be a non-empty string")
        super().__init__(message)
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


class ShapeError(ValidationError):
    """Tensor shape validation error."""

    error_code = E91_SHAPE_ERROR

    def __init__(
        self,
        message: str,
        parameter: str | None = None,
        expected_shape: tuple[Any, ...] | None = None,
        actual_shape: tuple[Any, ...] | None = None,
        expected_ndim: int | None = None,
        actual_ndim: int | None = None,
    ):
        expected_shape = _validate_shape_tuple(expected_shape, "expected_shape")
        actual_shape = _validate_shape_tuple(actual_shape, "actual_shape")
        expected_ndim = _validate_optional_nonnegative_int(expected_ndim, "expected_ndim")
        actual_ndim = _validate_optional_nonnegative_int(actual_ndim, "actual_ndim")
        super().__init__(message, parameter)
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.expected_ndim = expected_ndim
        self.actual_ndim = actual_ndim

    def __repr__(self) -> str:
        base = NerveError.__repr__(self)
        parts = []
        if self.expected_shape is not None:
            parts.append(f"expected_shape={self.expected_shape}")
        if self.actual_shape is not None:
            parts.append(f"actual_shape={self.actual_shape}")
        if self.expected_ndim is not None:
            parts.append(f"expected_ndim={self.expected_ndim}")
        if self.actual_ndim is not None:
            parts.append(f"actual_ndim={self.actual_ndim}")
        if not parts:
            return base
        return f"{base} {', '.join(parts)}"


class DtypeError(ValidationError):
    """Tensor dtype validation error."""

    error_code = E92_DTYPE_ERROR

    def __init__(
        self,
        message: str,
        parameter: str | None = None,
        expected_dtypes: list[str] | None = None,
        actual_dtype: str | None = None,
    ):
        expected_dtypes_tmp = _validate_string_list(expected_dtypes, "expected_dtypes")
        super().__init__(message, parameter)
        self.expected_dtypes: list[str] = expected_dtypes_tmp
        self.actual_dtype = actual_dtype

    def __repr__(self) -> str:
        base = NerveError.__repr__(self)
        parts = []
        if self.expected_dtypes:
            parts.append(f"expected_dtypes={self.expected_dtypes}")
        if self.actual_dtype is not None:
            parts.append(f"actual_dtype={self.actual_dtype!r}")
        if not parts:
            return base
        return f"{base} {', '.join(parts)}"


class DeviceError(ValidationError):
    """Device placement or availability error."""

    error_code = E93_DEVICE_ERROR

    def __init__(
        self,
        message: str,
        requested_device: str | None = None,
        available_devices: list[str] | None = None,
    ):
        if requested_device is not None and (
            not isinstance(requested_device, str) or not requested_device
        ):
            raise TypeError("requested_device must be a non-empty string")
        available_devices_lst = _validate_string_list(available_devices, "available_devices")
        super().__init__(message)
        self.requested_device = requested_device
        self.available_devices = available_devices_lst

    def __repr__(self) -> str:
        base = NerveError.__repr__(self)
        parts = []
        if self.requested_device is not None:
            parts.append(f"requested_device={self.requested_device!r}")
        if self.available_devices:
            parts.append(f"available_devices={self.available_devices}")
        if not parts:
            return base
        return f"{base} {', '.join(parts)}"


class BackendRequiredError(ValidationError):
    """Requested backend is not loaded."""

    error_code = E94_BACKEND_REQUIRED

    def __init__(
        self,
        message: str,
        backend: str | None = None,
        installation_hint: str | None = None,
        **kwargs: Any,
    ) -> None:
        if backend is not None and (not isinstance(backend, str) or not backend):
            raise ValidationError("backend must be a non-empty string", parameter="backend")
        if installation_hint is not None and (
            not isinstance(installation_hint, str) or not installation_hint
        ):
            raise ValidationError(
                "installation_hint must be a non-empty string", parameter="installation_hint"
            )
        super().__init__(message, **kwargs)
        self.backend = backend
        self.installation_hint = installation_hint

    def __repr__(self) -> str:
        base = NerveError.__repr__(self)
        parts = []
        if self.backend is not None:
            parts.append(f"backend={self.backend!r}")
        if self.installation_hint is not None:
            parts.append(f"installation_hint={self.installation_hint!r}")
        if not parts:
            return base
        return f"{base} {', '.join(parts)}"
