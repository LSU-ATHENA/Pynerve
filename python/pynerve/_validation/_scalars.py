"""Scalar parameter validators."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

from ._helpers import _validation_error


def validate_positive_int(value: int, name: str) -> int:
    """Validate that a value is a positive integer.

    :param value: The value to validate.
    :param name: The parameter name for error messages.
    :returns: The validated integer value.
    :raises ValidationError: If the value is not a positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        _validation_error(f"{name} must be an integer", param=name)
    result = int(value)
    if result <= 0:
        _validation_error(f"{name} must be positive", param=name)
    return result


def validate_nonnegative_int(value: int, name: str) -> int:
    """Validate that a value is a non-negative integer.

    :param value: The value to validate.
    :param name: The parameter name for error messages.
    :returns: The validated integer value.
    :raises ValidationError: If the value is not a non-negative integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        _validation_error(f"{name} must be an integer", param=name)
    result = int(value)
    if result < 0:
        _validation_error(f"{name} must be non-negative", param=name)
    return result


def validate_positive_finite(value: float, name: str) -> float:
    """Validate that a value is a finite positive float.

    :param value: The value to validate.
    :param name: The parameter name for error messages.
    :returns: The validated float value.
    :raises ValidationError: If the value is not finite or not positive.
    """
    result = float(value)
    if result <= 0 or not math.isfinite(result):
        _validation_error(f"{name} must be finite and positive", param=name)
    return result


def validate_nonnegative_finite(value: float, name: str) -> float:
    """Validate that a value is a finite non-negative float.

    :param value: The value to validate.
    :param name: The parameter name for error messages.
    :returns: The validated float value.
    :raises ValidationError: If the value is not finite or is negative.
    """
    result = float(value)
    if result < 0 or not math.isfinite(result):
        _validation_error(f"{name} must be finite and non-negative", param=name)
    return result


def validate_nonempty_string(value: str, name: str) -> str:
    """Validate that a value is a non-empty string.

    :param value: The value to validate.
    :param name: The parameter name for error messages.
    :returns: The validated string value.
    :raises ValidationError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        _validation_error(f"{name} must be a non-empty string", param=name)
    return value


def validate_bool(value: bool, name: str) -> bool:
    """Validate that a value is a boolean.

    :param value: The value to validate.
    :param name: The parameter name for error messages.
    :returns: The validated boolean value.
    :raises ValidationError: If the value is not a boolean.
    """
    if not isinstance(value, bool):
        _validation_error(f"{name} must be a boolean", param=name)
    return value


def validate_probability(value: float, name: str) -> float:
    """Validate that a value is a probability in [0, 1].

    :param value: The value to validate.
    :param name: The parameter name for error messages.
    :returns: The validated float value.
    :raises ValidationError: If the value is outside [0, 1].
    """
    result = float(value)
    if result < 0 or result > 1:
        _validation_error(f"{name} must be in [0, 1]", param=name)
    return result


def validate_finite_scalar(value: float, name: str) -> float:
    """Validate that a scalar value is finite.

    :param value: The value to validate.
    :param name: The parameter name for error messages.
    :returns: The validated float value.
    :raises ValidationError: If the value is not finite.
    """
    result = float(value)
    if not math.isfinite(result):
        _validation_error(f"{name} must be finite", param=name)
    return result


def validate_device_id(device_id: int, name: str = "device_id") -> int:
    """Validate that a value is a non-negative integer device id.

    :param device_id: The device index to validate.
    :param name: The parameter name for error messages.
    :returns: The validated integer device id.
    :raises ValidationError: If the value is not a non-negative integer.
    """
    if isinstance(device_id, bool) or not isinstance(device_id, Integral):
        _validation_error(f"{name} must be an integer", param=name)
    device_id = int(device_id)
    if device_id < 0:
        _validation_error(f"{name} must be non-negative", param=name)
    return device_id


def validate_optional_positive_int(value: int | None, name: str) -> int | None:
    """Validate that an optional value is a positive integer or None.

    :param value: The value to validate.
    :param name: The parameter name for error messages.
    :returns: The validated integer value or None.
    :raises ValidationError: If the value is not None and not a positive integer.
    """
    if value is None:
        return None
    return validate_positive_int(value, name)


def validate_optional_nonnegative_int(value: int | None, name: str) -> int | None:
    """Validate that an optional value is a non-negative integer or None.

    :param value: The value to validate.
    :param name: The parameter name for error messages.
    :returns: The validated integer value or None.
    :raises ValidationError: If the value is not None and not a non-negative integer.
    """
    if value is None:
        return None
    return validate_nonnegative_int(value, name)


def validate_optional_finite(value: float | None, name: str) -> float | None:
    """Validate that an optional value is a finite float or None.

    :param value: The value to validate.
    :param name: The parameter name for error messages.
    :returns: The validated float value or None.
    :raises ValidationError: If the value is not None and not finite.
    """
    if value is None:
        return None
    return validate_finite_scalar(value, name)


def validate_seed(seed: int | None, name: str = "seed") -> int | None:
    """Validate that a random seed is a non-negative integer or None.

    :param seed: The seed value to validate.
    :param name: The parameter name for error messages.
    :returns: The validated integer seed or None.
    :raises ValidationError: If the value is not None and not a non-negative integer.
    """
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        _validation_error(f"{name} must be an integer", param=name)
    seed = int(seed)
    if seed < 0:
        _validation_error(f"{name} must be non-negative", param=name)
    return seed


def validate_optional_string(value: str | None, name: str) -> str | None:
    """Validate that an optional value is a non-empty string or None.

    :param value: The value to validate.
    :param name: The parameter name for error messages.
    :returns: The validated string value or None.
    :raises ValidationError: If the value is not None and not a non-empty string.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        _validation_error(f"{name} must be a non-empty string", param=name)
    return value


def validate_string_list(values: list[str] | None, name: str) -> list[str]:
    """Validate a list of non-empty strings.

    Returns an empty list if the input is None.

    :param values: A list of strings or None.
    :param name: The parameter name for error messages.
    :returns: The validated list of non-empty strings.
    :raises ValidationError: If any value is not a non-empty string.
    """
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, list):
        _validation_error(f"{name} must be a sequence of strings", param=name)
    if not all(isinstance(value, str) and value for value in values):
        _validation_error(f"{name} must contain non-empty strings", param=name)
    return list(values)


def validate_max_dist(max_dist: float | None, name: str = "max_dist") -> float | None:
    """Validate an optional maximum distance parameter.

    :param max_dist: A finite non-negative float or None.
    :param name: The parameter name for error messages.
    :returns: The validated float value or None.
    :raises ValidationError: If the value is not None, not finite, or negative.
    """
    if max_dist is None:
        return None
    result = float(max_dist)
    if result < 0 or not math.isfinite(result):
        _validation_error(f"{name} must be finite and non-negative", param=name)
    return result


def validate_max_radius(max_radius: float | None, name: str = "max_radius") -> float:
    """Validate an optional maximum radius parameter.

    Returns positive infinity if None.

    :param max_radius: A finite non-negative float or None.
    :param name: The parameter name for error messages.
    :returns: The validated float value or ``float("inf")`` if None.
    :raises ValidationError: If the value is not None, not finite, or negative.
    """
    if max_radius is None:
        return float("inf")
    result = float(max_radius)
    if result < 0 or not math.isfinite(result):
        _validation_error(f"{name} must be finite and non-negative", param=name)
    return result


def parse_nonnegative_int(value: Any, name: str) -> int:
    """Parse and validate a non-negative integer from a string or int.

    :param value: An ``int`` or ``str`` representation of an integer.
    :param name: The parameter name for error messages.
    :returns: The parsed non-negative integer.
    :raises ValidationError: If the value cannot be parsed as a non-negative integer.
    """
    if isinstance(value, bool):
        _validation_error(f"{name} must be an integer", param=name)
        raise RuntimeError("unreachable")
    if isinstance(value, (int, str)):
        return validate_nonnegative_int(int(value), name)
    _validation_error(f"{name} must be an integer", param=name)
    raise RuntimeError("unreachable")
