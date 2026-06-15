"""Pipeline helpers: validation, tensor conversion, option resolution."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ._fallback_classes import PersistenceBackend, PersistenceMode, PersistenceOptions
from ._persistence_result import _MAX_RADIUS_CAP, _nerve_state, _warn_large_max_radius_cap
from ._validation import validate_device_spec
from .exceptions import InvalidArgumentError, ShapeMismatchError


def _is_likely_distance_matrix(array: np.ndarray) -> bool:
    """Detect if a 2D array is likely a precomputed distance matrix."""
    if array.ndim != 2:
        return False
    n, m = array.shape
    if n != m or n < 2:
        return False
    if not np.allclose(np.diag(array), 0.0):
        return False
    return bool(np.allclose(array, array.T))


def _clone_options(options: PersistenceOptions | None) -> PersistenceOptions:
    return dataclasses.replace(options or PersistenceOptions())


def _validate_max_dim(max_dim: int) -> int:
    max_dim = int(max_dim)
    if max_dim < 0:
        raise InvalidArgumentError(
            "max_dim must be non-negative",
            parameter="max_dim",
            expected=">= 0",
            actual=str(max_dim),
        )
    return max_dim


def _validate_max_radius(max_radius: float, cap: float | None = None) -> float:
    radius_value = float(max_radius)
    effective_cap = cap if cap is not None else _MAX_RADIUS_CAP
    if radius_value == float("inf"):
        _warn_large_max_radius_cap()
        radius_value = effective_cap
    if not np.isfinite(radius_value) or radius_value < 0:
        raise InvalidArgumentError(
            "max_radius must be finite and non-negative",
            parameter="max_radius",
            expected="finite and >= 0",
            actual=str(radius_value),
        )
    return radius_value


def _validate_threads(threads: int) -> int:
    threads = int(threads)
    if threads <= 0:
        raise InvalidArgumentError(
            "threads must be positive",
            parameter="threads",
            expected="> 0",
            actual=str(threads),
        )
    return threads


def _validate_error_tolerance(error_tolerance: float) -> float:
    tolerance_value = float(error_tolerance)
    if not np.isfinite(tolerance_value) or tolerance_value < 0:
        raise InvalidArgumentError(
            "error_tolerance must be finite and non-negative",
            parameter="error_tolerance",
            expected="finite and >= 0",
            actual=str(tolerance_value),
        )
    return tolerance_value


def _apply_option_overrides(
    options: PersistenceOptions,
    *,
    max_dim: int | None = None,
    max_radius: float | None = None,
    mode: PersistenceMode | None = None,
    backend: PersistenceBackend | None = None,
    threads: int | None = None,
    device: str | None = None,
    seed: int | None = None,
    error_tolerance: float | None = None,
    max_radius_cap: float | None = None,
) -> PersistenceOptions:
    from ._compute_backend import _resolve_device_to_backend, _seed_rng  # noqa: PLC0415

    kwargs: dict[str, Any] = {}

    if max_dim is not None:
        kwargs["max_dim"] = _validate_max_dim(max_dim)
    if max_radius is not None:
        kwargs["max_radius"] = _validate_max_radius(max_radius, max_radius_cap)
    if mode is not None:
        kwargs["mode"] = mode
    if backend is not None:
        kwargs["backend"] = backend
    if device is not None:
        validate_device_spec(device)
        kwargs["backend"] = _resolve_device_to_backend(device)
    if seed is not None:
        _seed_rng(seed)
    if threads is not None:
        kwargs["threads"] = _validate_threads(threads)
    if error_tolerance is not None:
        kwargs["error_tolerance"] = _validate_error_tolerance(error_tolerance)

    return dataclasses.replace(options, **kwargs)


def _validate_array(array: np.ndarray) -> np.ndarray:
    if array.ndim != 2:
        raise ShapeMismatchError(
            f"points must be a 2D array, got {array.ndim}D with shape {tuple(array.shape)}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise InvalidArgumentError("points cannot be empty")
    if not np.isfinite(array).all():
        raise InvalidArgumentError("points contain NaN or infinite values")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def _tensor_to_array(tensor: Any, pytorch_mod: Any, dtype: str | None = None) -> np.ndarray:
    tensor = tensor.detach()
    if tensor.ndim != 2:
        raise ShapeMismatchError(
            f"points must be a 2D array, got {tensor.ndim}D with shape {tuple(tensor.shape)}"
        )
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise InvalidArgumentError("points cannot be empty")
    if tensor.is_cuda:
        import warnings as _warnings  # noqa: PLC0415

        _warnings.warn(
            "GPU tensor moved to CPU for persistence computation. "
            "Pass device='cuda' to use GPU acceleration instead.",
            UserWarning,
            stacklevel=3,
        )
        tensor = tensor.to(device="cpu")
    target_dtype = getattr(pytorch_mod, dtype) if dtype else pytorch_mod.float64
    if tensor.dtype != target_dtype:
        tensor = tensor.to(dtype=target_dtype)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return _validate_array(tensor.numpy())


def _to_point_array(points: Any, dtype: str | None = None) -> np.ndarray:
    _, _, _PYTORCH = _nerve_state()  # noqa: N806

    if _PYTORCH is not None and isinstance(points, _PYTORCH.Tensor):
        return _tensor_to_array(points, _PYTORCH, dtype)

    np_dtype = np.dtype(dtype) if dtype else np.float64
    if isinstance(points, (list, tuple)):
        points = np.asarray(points, dtype=np_dtype)
        if points.dtype.kind == "O":
            raise InvalidArgumentError(
                "points must be a 2D array-like of numeric values; "
                "jagged or non-numeric lists are not supported",
                parameter="points",
            )
    else:
        points = np.asarray(points, dtype=np_dtype)

    return _validate_array(points)


def _resolve_options(
    points: Any,
    options: PersistenceOptions | None,
    **overrides: Any,
) -> PersistenceOptions:
    _, _, _PYTORCH = _nerve_state()  # noqa: N806

    resolved = _apply_option_overrides(_clone_options(options), **overrides)
    if (
        _PYTORCH is not None
        and isinstance(points, _PYTORCH.Tensor)
        and points.is_cuda
        and resolved.backend == PersistenceBackend.CPU_EXACT
    ):
        resolved = dataclasses.replace(resolved, backend=PersistenceBackend.CUDA_HYBRID)
    return resolved
