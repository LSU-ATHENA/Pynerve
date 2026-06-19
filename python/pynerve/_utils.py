"""Internal validation and backend helpers."""

from __future__ import annotations

import warnings
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, TypeVar

import numpy as np

from ._validation import validate_finite_scalar, validate_positive_int
from .exceptions import DeviceError, DtypeError, ShapeError, ValidationError

T = TypeVar("T")

_has_torch: bool = False
_torch_module: Any = None
try:
    import torch as _torch_module

    _has_torch = True
except ImportError:
    pass


def _get_torch() -> Any:
    if not _has_torch:
        raise ImportError("PyTorch is required for this operation but is not installed")
    return _torch_module


def _validate_tensor(tensor: Any, name: str = "tensor") -> Any:
    if _has_torch:
        if isinstance(tensor, _torch_module.Tensor):
            return tensor
    elif str(type(tensor)).endswith(".Tensor'>"):
        raise ImportError("PyTorch is required for this tensor operation but is not installed")
    raise TypeError(f"{name} must be a torch.Tensor")


def _validate_name(name: str, label: str = "name") -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label} must be a non-empty string")
    return name


def _get_dtype(tensor: Any) -> Any:
    return tensor.dtype


def ensure_batch_dim(tensor: Any, expected_ndim: int) -> tuple[Any, bool]:
    """Ensure *tensor* has a batch dimension, adding one if necessary.

    :param tensor: A ``torch.Tensor`` with ``expected_ndim`` or ``expected_ndim - 1`` dimensions.
    :param expected_ndim: Target number of dimensions including batch.
    :returns: A tuple ``(tensor_with_batch, was_single)`` where *was_single* is
        ``True`` if a batch dimension was inserted.
    :raises TypeError: If *tensor* is not a ``torch.Tensor``.
    :raises ValueError: If *expected_ndim* is not a positive integer.
    :raises ShapeError: If *tensor* dimensionality is incompatible.
    """
    tensor = _validate_tensor(tensor)
    expected_ndim = validate_positive_int(expected_ndim, "expected_ndim")
    if tensor.dim() == expected_ndim - 1:
        return tensor.unsqueeze(0), True
    if tensor.dim() == expected_ndim:
        return tensor, False
    raise ShapeError(
        f"Expected {expected_ndim - 1}D or {expected_ndim}D input, "
        f"got {tensor.dim()}D with shape {tuple(tensor.shape)}",
        expected_ndim=expected_ndim,
        actual_ndim=tensor.dim(),
        actual_shape=tuple(tensor.shape),
    )


def remove_batch_dim(tensor: Any, was_single: bool) -> Any:
    """Remove a previously-inserted batch dimension if *was_single* is ``True``.

    :param tensor: The ``torch.Tensor`` from which to remove the batch dimension.
    :param was_single: If ``True``, squeeze the first dimension when it has size 1.
    :returns: The tensor with or without the batch dimension removed.
    :raises TypeError: If *tensor* is not a ``torch.Tensor`` or *was_single* is not ``bool``.
    """
    tensor = _validate_tensor(tensor)
    if not isinstance(was_single, bool):
        raise TypeError("was_single must be a boolean")
    if was_single and tensor.dim() > 0 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def validate_tensor_shape(tensor: Any, expected_ndim: int, name: str = "tensor") -> None:
    """Validate that *tensor* has exactly *expected_ndim* dimensions.

    :param tensor: The ``torch.Tensor`` to validate.
    :param expected_ndim: Expected number of dimensions.
    :param name: Human-readable name for the tensor used in error messages.
    :raises TypeError: If *tensor* is not a ``torch.Tensor``.
    :raises ValueError: If *expected_ndim* is not a positive integer or *name* is empty.
    :raises ShapeError: If *tensor* does not have *expected_ndim* dimensions.
    """
    tensor = _validate_tensor(tensor, name)
    expected_ndim = validate_positive_int(expected_ndim, "expected_ndim")
    name = _validate_name(name)
    if tensor.dim() != expected_ndim:
        raise ShapeError(
            f"Expected {name} to be {expected_ndim}D, "
            f"got {tensor.dim()}D with shape {tuple(tensor.shape)}",
            parameter=name,
            expected_ndim=expected_ndim,
            actual_ndim=tensor.dim(),
            actual_shape=tuple(tensor.shape),
        )


def validate_dtype(tensor: Any, supported_dtypes: set[Any], name: str = "tensor") -> None:
    """Validate that *tensor* has a dtype in *supported_dtypes*.

    :param tensor: The ``torch.Tensor`` to validate.
    :param supported_dtypes: Set of allowable ``torch.dtype`` values.
    :param name: Human-readable name for the tensor used in error messages.
    :raises TypeError: If *tensor* is not a ``torch.Tensor`` or *supported_dtypes* contains non-dtype items.
    :raises ValueError: If *supported_dtypes* is empty or *name* is empty.
    :raises DtypeError: If *tensor* dtype is not in *supported_dtypes*.
    """
    tensor = _validate_tensor(tensor, name)
    name = _validate_name(name)
    torch = _get_torch()
    if not isinstance(supported_dtypes, set) or not supported_dtypes:
        raise ValueError("supported_dtypes must be a non-empty set")
    if not all(isinstance(dtype, torch.dtype) for dtype in supported_dtypes):
        raise TypeError("supported_dtypes must contain torch.dtype values")
    if tensor.dtype not in supported_dtypes:
        raise DtypeError(
            f"Unsupported dtype {tensor.dtype} for {name}. "
            f"Supported: {[str(d) for d in supported_dtypes]}",
            parameter=name,
            actual_dtype=str(tensor.dtype),
            expected_dtypes=[str(d) for d in supported_dtypes],
        )


def validate_positive(value: float, name: str = "value") -> None:
    """Validate that *value* is a finite positive number.

    :param value: The scalar value to validate.
    :param name: Human-readable name used in error messages.
    :raises ValidationError: If *value* is not greater than zero.
    :raises ValueError: If *value* is not finite or *name* is empty.
    """
    value = validate_finite_scalar(value, name)
    name = _validate_name(name)
    if value <= 0:
        raise ValidationError(f"Expected {name} to be positive, got {value}", parameter=name)


def validate_range(value: float, min_val: float, max_val: float, name: str = "value") -> None:
    """Validate that *value* lies within the inclusive range ``[min_val, max_val]``.

    :param value: The scalar value to validate.
    :param min_val: Minimum allowable value (inclusive).
    :param max_val: Maximum allowable value (inclusive).
    :param name: Human-readable name used in error messages.
    :raises ValidationError: If *value* is not in range.
    :raises ValueError: If any argument is not finite, *name* is empty,
        or *min_val* exceeds *max_val*.
    """
    value = validate_finite_scalar(value, name)
    min_val = validate_finite_scalar(min_val, "min_val")
    max_val = validate_finite_scalar(max_val, "max_val")
    name = _validate_name(name)
    if min_val > max_val:
        raise ValueError("min_val must be less than or equal to max_val")
    if not (min_val <= value <= max_val):
        raise ValidationError(
            f"Expected {name} to be in range [{min_val}, {max_val}], got {value}",
            parameter=name,
        )


def validate_devices_match(tensors: list[Any], names: list[str]) -> None:
    """Validate that all tensors reside on the same device.

    :param tensors: List of ``torch.Tensor`` objects to check.
    :param names: List of human-readable names, one per tensor.
    :raises TypeError: If any tensor is not a ``torch.Tensor``.
    :raises ValueError: If *tensors* and *names* have different lengths or any *name* is empty.
    :raises DeviceError: If any tensor is on a different device than the first.
    """
    if not tensors:
        return
    if len(tensors) != len(names):
        raise ValueError("tensors and names must have matching lengths")
    for tensor, name in zip(tensors, names, strict=False):
        _validate_tensor(tensor, name)
        _validate_name(name)

    first_device = tensors[0].device
    for tensor, name in zip(tensors[1:], names[1:], strict=False):
        if tensor.device != first_device:
            raise DeviceError(
                f"Device mismatch: {names[0]} is on {first_device}, "
                f"but {name} is on {tensor.device}",
                requested_device=str(first_device),
                available_devices=[str(first_device), str(tensor.device)],
            )


@contextmanager
def suppress_warnings(category: type[Warning] = Warning) -> Generator[None, None, None]:
    """Context manager that temporarily suppresses warnings of the given *category*.

    :param category: A ``Warning`` subclass to suppress (default: all warnings).
    :returns: A context manager usable with ``with``.
    :raises TypeError: If *category* is not a ``Warning`` subclass.
    """
    if not isinstance(category, type) or not issubclass(category, Warning):
        raise TypeError("category must be a Warning subclass")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category)
        yield


def is_tensor(obj: Any) -> bool:
    """Return ``True`` if *obj* is a ``torch.Tensor``.

    :param obj: Any Python object.
    :returns: ``True`` if PyTorch is installed and *obj* is a ``torch.Tensor``,
        ``False`` otherwise.
    """
    if not _has_torch:
        return False
    return isinstance(obj, _torch_module.Tensor)


def is_numpy_array(obj: Any) -> bool:
    """Return ``True`` if *obj* is a ``numpy.ndarray``.

    :param obj: Any Python object.
    :returns: ``True`` if *obj* is a ``numpy.ndarray``, ``False`` otherwise.
    """
    return isinstance(obj, np.ndarray)


__all__ = [
    "ensure_batch_dim",
    "remove_batch_dim",
    "validate_tensor_shape",
    "validate_dtype",
    "validate_positive",
    "validate_range",
    "validate_devices_match",
    "suppress_warnings",
    "is_tensor",
    "is_numpy_array",
]
