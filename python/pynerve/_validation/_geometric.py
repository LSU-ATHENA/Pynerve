"""Tensor and geometric parameter validators."""

from __future__ import annotations

from collections import abc as collections_abc
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

try:
    import torch as _torch_mod
except ImportError:
    _torch_mod = None  # type: ignore[assignment]

from ._helpers import _dtype_error, _shape_error, _validation_error


def validate_finite_deaths(deaths: np.ndarray, name: str = "deaths") -> None:
    """Validate that an array contains only finite values or positive infinity.

    The input is typically used for death values in persistence diagrams, where
    positive infinity represents essential (infinite) homology classes.

    :param deaths: NumPy array of death values to validate.
    :param name: The parameter name for error messages.
    :returns: None.
    :raises ValidationError: If the array contains NaN or negative infinity.
    """
    if np.isnan(deaths).any():
        _validation_error(f"{name} must not be NaN", param=name)
    invalid = ~(np.isfinite(deaths) | np.isposinf(deaths))
    if invalid.any():
        _validation_error(f"{name} must be finite or positive infinity", param=name)


def validate_finite_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Validate that a tensor contains only finite values.

    :param tensor: The PyTorch tensor to validate.
    :param name: The parameter name for error messages.
    :raises DtypeError: If the input is not a PyTorch tensor.
    :raises ValidationError: If the tensor contains non-finite values.
    """
    assert _torch_mod is not None, "torch is required for this function"
    if not isinstance(tensor, _torch_mod.Tensor):
        _dtype_error(f"{name} must be a torch.Tensor")
    if not _torch_mod.isfinite(tensor).all().item():
        _validation_error(f"{name} must contain only finite values", param=name)


def validate_diagram(
    diagram: torch.Tensor, name: str = "diagram", min_cols: int = 2
) -> torch.Tensor:
    """Validate a persistence diagram tensor.

    Checks that the diagram is 2D, has enough columns, uses floating-point dtype,
    has finite births, and that finite deaths are not less than births.

    :param diagram: A 2D tensor representing a persistence diagram.
    :param name: The parameter name for error messages.
    :param min_cols: Minimum number of columns required (default: 2).
    :returns: The validated diagram tensor.
    :raises ShapeError: If the tensor is not 2D or has too few columns.
    :raises DtypeError: If the tensor does not use a floating-point dtype.
    :raises ValidationError: If births or deaths are invalid.
    """
    assert _torch_mod is not None, "torch is required for this function"
    if diagram.dim() != 2:
        _shape_error(
            f"{name} must be a 2D tensor, got {diagram.dim()}D",
            param=name,
            expected_ndim=2,
            actual_ndim=diagram.dim(),
        )
    if diagram.shape[-1] < min_cols:
        _shape_error(
            f"{name} must have at least {min_cols} columns",
            param=name,
        )
    if not _torch_mod.is_floating_point(diagram):
        _dtype_error(f"{name} must use a floating-point dtype")
    if diagram.numel() == 0:
        return diagram

    work = diagram[..., :2].to(dtype=_torch_mod.float64)
    births = work[:, 0]
    deaths = work[:, 1]
    if not _torch_mod.isfinite(births).all().item():
        _validation_error(f"{name} births must be finite", param=name)
    if _torch_mod.isnan(deaths).any().item():
        _validation_error(f"{name} deaths must not be NaN", param=name)
    finite_death_mask = _torch_mod.isfinite(deaths)
    if (
        finite_death_mask.any().item()
        and not (deaths[finite_death_mask] >= births[finite_death_mask]).all().item()
    ):
        _validation_error(f"{name} finite deaths must be >= births", param=name)
    return diagram


def validate_points(points: np.ndarray, name: str = "points") -> np.ndarray:
    """Validate that a point cloud is a 2D array with finite coordinates.

    Accepts both NumPy arrays and PyTorch tensors; tensors are converted to NumPy.

    :param points: A 2D array of points as a NumPy array or PyTorch tensor.
    :param name: The parameter name for error messages.
    :returns: The validated points as a NumPy array.
    :raises ShapeError: If the array is not 2D or has zero coordinates.
    :raises ValidationError: If the array contains non-finite coordinates.
    """
    if _torch_mod is not None and isinstance(points, _torch_mod.Tensor):
        if points.ndim != 2:
            _shape_error(
                f"{name} must be a 2D array",
                param=name,
                expected_ndim=2,
                actual_ndim=points.ndim,
            )
        if points.shape[1] == 0:
            _shape_error(f"{name} must contain at least one coordinate", param=name)
        if points.numel() > 0 and not _torch_mod.isfinite(points).all():
            _validation_error(f"{name} must contain only finite coordinates", param=name)
        return points.cpu().numpy()
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        _shape_error(
            f"{name} must be a 2D array", param=name, expected_ndim=2, actual_ndim=points.ndim
        )
    if points.shape[1] == 0:
        _shape_error(f"{name} must contain at least one coordinate", param=name)
    if points.size > 0 and not np.isfinite(points).all():
        _validation_error(f"{name} must contain only finite coordinates", param=name)
    return points


def validate_floating_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Validate that a tensor uses a floating-point dtype.

    :param tensor: The PyTorch tensor to validate.
    :param name: The parameter name for error messages.
    :raises DtypeError: If the tensor is not a PyTorch tensor or uses a non-floating dtype.
    """
    assert _torch_mod is not None, "torch is required for this function"
    if not isinstance(tensor, _torch_mod.Tensor):
        _dtype_error(f"{name} must be a torch.Tensor")
    if not _torch_mod.is_floating_point(tensor):
        _dtype_error(f"{name} must use a floating-point dtype")


def validate_shape(
    shape: collections_abc.Sequence[int] | int, *, name: str = "shape", allow_infer: bool = False
) -> tuple[int, ...]:
    """Validate a shape specification.

    Accepts either a single integer or a sequence of integers. When ``allow_infer`` is
    True, at most one dimension may be -1 (inferred).

    :param shape: A shape as an integer or sequence of integers.
    :param name: The parameter name for error messages.
    :param allow_infer: Whether to allow a single -1 dimension for inference.
    :returns: The validated shape as a tuple of integers.
    :raises ShapeError: If the shape specification is invalid.
    """
    if isinstance(shape, Integral) and not isinstance(shape, bool):
        values: tuple[int, ...] = (int(shape),)
    elif isinstance(shape, (str, bytes)) or not isinstance(shape, collections_abc.Sequence):
        _shape_error(f"{name} must be an integer or sequence of integers", param=name)
        raise RuntimeError("unreachable")
    else:
        values = tuple(shape)
    if len(values) == 0:
        _shape_error(f"{name} must contain at least one dimension", param=name)

    infer_count = 0
    result: list[int] = []
    for dim in values:
        if isinstance(dim, bool) or not isinstance(dim, Integral):
            _shape_error(f"{name} dimensions must be integers", param=name)
        dim = int(dim)
        if dim == -1 and allow_infer:
            infer_count += 1
        elif dim < 0:
            _shape_error(f"{name} dimensions must be non-negative", param=name)
        result.append(dim)
    if infer_count > 1:
        _shape_error(f"{name} may contain at most one inferred dimension", param=name)
    return tuple(result)


_VALID_DEVICE_PREFIXES = {
    "cpu": "cpu",
    "cuda": "cuda",
    "mps": "mps",
    "hip": "hip",
    "xpu": "xpu",
    "rocm": "hip",
}


def validate_shape_tuple(value: tuple[int, ...] | None, name: str) -> tuple[int, ...] | None:
    """Validate an optional shape tuple of non-negative integers.

    :param value: A tuple of integers or None.
    :param name: The parameter name for error messages.
    :returns: The validated tuple or None.
    :raises ShapeError: If any dimension is not a non-negative integer.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, collections_abc.Sequence):
        _shape_error(f"{name} must be a sequence of dimensions", param=name)
    result: list[int] = []
    for dim in value:
        if isinstance(dim, bool) or not isinstance(dim, Integral):
            _shape_error(f"{name} dimensions must be integers", param=name)
        dim = int(dim)
        if dim < 0:
            _shape_error(f"{name} dimensions must be non-negative", param=name)
        result.append(dim)
    return tuple(result)


def validate_diagram_array(
    array: np.ndarray,
    name: str = "diagram",
    require_dims: bool = False,
) -> np.ndarray:
    """Validate a persistence diagram stored as a NumPy array.

    Checks that the array is 2D with at least 2 columns, births are finite,
    deaths are finite or +inf, and dimension columns (if present) are valid integers.

    :param array: A NumPy array of shape ``(n, >=2)``.
    :param name: The parameter name for error messages.
    :param require_dims: Whether to require at least 3 columns (birth, death, dim).
    :returns: The validated NumPy array (guaranteed C-contiguous).
    :raises ShapeError: If the shape is invalid.
    :raises ValidationError: If birth, death, or dimension values are invalid.
    """
    if array.size == 0:
        return np.empty((0, 3), dtype=array.dtype)
    if array.ndim != 2 or array.shape[1] < 2:
        _shape_error(
            f"{name} must have shape (n, at least 2), got {tuple(array.shape)}",
            param=name,
            expected_ndim=2,
        )
    births = array[:, 0]
    deaths = array[:, 1]
    if not np.isfinite(births).all():
        _validation_error(f"{name} births must be finite", param=name)
    if np.isnan(deaths).any() or np.isneginf(deaths).any():
        _validation_error(f"{name} deaths must be finite or +inf", param=name)
    finite_deaths = np.isfinite(deaths)
    if finite_deaths.any() and np.any(deaths[finite_deaths] < births[finite_deaths]):
        _validation_error(f"{name} finite deaths must be >= births", param=name)
    if array.shape[1] >= 3:
        dims = array[:, 2]
        if (
            not np.isfinite(dims).all()
            or (dims < 0).any()
            or not np.equal(dims, np.floor(dims)).all()
        ):
            _validation_error(f"{name} dimensions must be finite non-negative integers", param=name)
    elif require_dims:
        _shape_error(
            f"{name} must have at least 3 columns (birth, death, dim), got {array.shape[1]}",
            param=name,
        )
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def validate_device_spec(device: str, name: str = "device") -> None:
    """Validate a device specification string.

    Accepts ``"cpu"`` and ``"cuda:N"`` style device strings with
    supported prefixes including ``cuda``, ``mps``, ``hip``, ``xpu``, ``rocm``.

    :param device: A device string.
    :param name: The parameter name for error messages.
    :raises ValidationError: If the device string is invalid.
    """
    if not isinstance(device, str) or not device:
        _validation_error(f"{name} must be a non-empty string", param=name)
    if device == "cpu":
        return
    for prefix in _VALID_DEVICE_PREFIXES:
        if device == prefix:
            return
        if device.startswith(f"{prefix}:"):
            try:
                idx = int(device.split(":", 1)[1])
                if idx < 0:
                    raise ValueError
            except (ValueError, IndexError):
                _validation_error(
                    f"Invalid {name}: {device!r}. Use '{prefix}:N' with N >= 0.",
                    param=name,
                )
            return
    _validation_error(
        f"Unknown {name}: {device!r}. Supported: {', '.join(sorted(_VALID_DEVICE_PREFIXES))}:N",
        param=name,
    )
