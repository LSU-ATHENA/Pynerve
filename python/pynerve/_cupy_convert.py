"""Array conversion helpers for CuPy interop."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ._cupy_compat import HAS_CUPY, cp
from ._validation import validate_device_id

if TYPE_CHECKING:
    import cupy
    import tensorflow as tf  # pyright: ignore[reportMissingModuleSource]

    import torch

TargetArray = Literal["numpy", "torch", "tensorflow", "jax"]
_TARGET_ARRAYS = {"numpy", "torch", "tensorflow", "jax"}


def _validate_target_type(target_type: TargetArray) -> TargetArray:
    if target_type not in _TARGET_ARRAYS:
        valid = ", ".join(sorted(_TARGET_ARRAYS))
        raise ValueError(f"Unknown target type '{target_type}', expected one of {valid}")
    return target_type


def _require_cupy() -> None:
    if not HAS_CUPY:
        raise RuntimeError("CuPy required")


def _is_cupy_array(array: Any) -> bool:
    return HAS_CUPY and cp is not None and isinstance(array, cp.ndarray)


def _as_dlpack_source(array: Any) -> Any:
    if hasattr(array, "__dlpack__"):
        return array

    return _dlpack_capsule(array)


def _dlpack_capsule(array: Any) -> Any:
    to_dlpack = getattr(array, "toDlpack", None) or getattr(array, "to_dlpack", None)
    if callable(to_dlpack):
        return to_dlpack()

    if hasattr(array, "__dlpack__"):
        return array.__dlpack__()

    raise TypeError(f"{type(array).__name__} does not expose DLPack")


def _torch_to_cupy(array: Any) -> Any:
    return cp.from_dlpack(_as_dlpack_source(array))


def _tensorflow_to_cupy(array: Any) -> Any:
    import tensorflow as tf  # noqa: PLC0415 # pyright: ignore[reportMissingModuleSource]

    return cp.from_dlpack(tf.experimental.dlpack.to_dlpack(array))


def _jax_to_cupy(array: Any) -> Any:
    return cp.from_dlpack(_as_dlpack_source(array))


def _try_torch_to_cupy(array: Any) -> Any | None:
    with suppress(ImportError):
        import torch  # noqa: PLC0415

        if isinstance(array, torch.Tensor):
            return _torch_to_cupy(array)
    return None


def _try_tensorflow_to_cupy(array: Any) -> Any | None:
    with suppress(ImportError):
        import tensorflow as tf  # noqa: PLC0415 # pyright: ignore[reportMissingModuleSource]

        if isinstance(array, tf.Tensor):
            return _tensorflow_to_cupy(array)
    return None


def _try_jax_to_cupy(array: Any) -> Any | None:
    with suppress(ImportError):
        import jax  # noqa: PLC0415

        if isinstance(array, jax.Array):
            return _jax_to_cupy(array)
    return None


def to_cupy(array: np.ndarray | cupy.ndarray | torch.Tensor, device_id: int = 0) -> cupy.ndarray:
    """Convert an array from any supported framework to a CuPy array.

    :param array: Input array (NumPy, CuPy, PyTorch, TensorFlow, or JAX).
    :param device_id: Target GPU device ID (default 0).
    :returns: A CuPy ndarray on the specified device.
    :raises RuntimeError: If CuPy is not installed.
    :raises TypeError: If *array* uses object dtype or is not convertible.
    """
    device_id = validate_device_id(device_id)
    if isinstance(array, np.ndarray) and array.dtype.hasobject:
        raise TypeError("array cannot use object dtype")
    _require_cupy()

    if _is_cupy_array(array):
        return array

    with cp.cuda.Device(device_id):
        if isinstance(array, np.ndarray):
            return cp.asarray(array)

        result = _try_torch_to_cupy(array)
        if result is not None:
            return result

        result = _try_tensorflow_to_cupy(array)
        if result is not None:
            return result

        result = _try_jax_to_cupy(array)
        if result is not None:
            return result

        return cp.asarray(array)


def _cupy_to_torch(cupy_array: Any) -> Any:
    from torch.utils.dlpack import from_dlpack  # noqa: PLC0415

    return from_dlpack(_as_dlpack_source(cupy_array))


def _cupy_to_tensorflow(cupy_array: Any) -> Any:
    import tensorflow as tf  # noqa: PLC0415 # pyright: ignore[reportMissingModuleSource]

    return tf.experimental.dlpack.from_dlpack(_dlpack_capsule(cupy_array))


def _cupy_to_jax(cupy_array: Any) -> Any:
    import jax  # noqa: PLC0415

    return jax.dlpack.from_dlpack(_dlpack_capsule(cupy_array))


def from_cupy(
    cupy_array: cupy.ndarray, target_type: TargetArray = "numpy"
) -> np.ndarray | torch.Tensor | tf.Tensor:
    """Convert a CuPy array to a target framework array.

    :param cupy_array: A CuPy ndarray.
    :param target_type: Target framework. One of ``"numpy"``,
        ``"torch"``, ``"tensorflow"``, ``"jax"`` (default ``"numpy"``).
    :returns: An array in the target framework.
    :raises RuntimeError: If CuPy is not installed.
    :raises ValueError: If *target_type* is unknown.
    """
    target_type = _validate_target_type(target_type)
    _require_cupy()

    converters = {
        "numpy": lambda array: array.get(),
        "torch": _cupy_to_torch,
        "tensorflow": _cupy_to_tensorflow,
        "jax": _cupy_to_jax,
    }
    return converters[target_type](cupy_array)
