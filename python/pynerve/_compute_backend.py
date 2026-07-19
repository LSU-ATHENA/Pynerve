"""Backend, device, and RNG helpers + top-level compute entry point."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from ._fallback_classes import EventType, PersistenceBackend
from ._persistence_result import (
    _MAX_RADIUS_CAP,
    PersistenceResult,
    _nerve_state,
    _warn_large_max_radius_cap,
)
from .exceptions import InvalidArgumentError, ValidationError


def _to_events_list(
    events: Iterable[tuple[EventType | str, Sequence[int]]],
) -> list[tuple[str, list[int]]]:
    out: list[tuple[str, list[int]]] = []
    for event_type, simplex in events:
        if isinstance(event_type, EventType):
            type_str = event_type.value
        elif event_type in ("add", "remove"):
            type_str = event_type
        else:
            raise ValidationError(
                f"event type must be EventType, 'add', or 'remove', got '{event_type}'",
                parameter="event_type",
            )
        out.append((type_str, [int(v) for v in simplex]))
    return out


def _to_internal_options(py_opts: Any) -> Any:
    _core = _nerve_state()[0]
    if _core is None:
        return py_opts
    int_opts = _core.PersistenceOptions()
    mode_name = (
        py_opts.mode.name if hasattr(py_opts.mode, "name") else str(py_opts.mode)
    )
    int_opts.mode = getattr(_core.PersistenceMode, mode_name)
    backend_name = (
        py_opts.backend.name
        if hasattr(py_opts.backend, "name")
        else str(py_opts.backend)
    )
    int_opts.backend = getattr(_core.PersistenceBackend, backend_name)
    int_opts.max_dim = int(py_opts.max_dim)
    mr = py_opts.max_radius
    if mr is None:
        int_opts.max_radius = float("inf")
    else:
        mr = float(mr)
        if mr == float("inf"):
            _warn_large_max_radius_cap()
            int_opts.max_radius = _MAX_RADIUS_CAP
        else:
            int_opts.max_radius = mr
    int_opts.threads = int(py_opts.threads)
    int_opts.error_tolerance = float(py_opts.error_tolerance)
    return int_opts


def _warn_device_overrides_backend(device: str, backend: Any) -> None:
    backend_name = backend.name if hasattr(backend, "name") else str(backend)
    import warnings as _warnings  # noqa: PLC0415

    _warnings.warn(
        f"Both device={device!r} and backend={backend_name!r} provided. "
        "The device argument takes precedence - backend will be overridden "
        "to the backend associated with the specified device.",
        UserWarning,
        stacklevel=3,
    )


_DEVICE_TO_BACKEND = {
    "cpu": "CPU_ADAPTIVE_ACCELERATION",
    "cuda": "CUDA_HYBRID",
}


def _resolve_device_to_backend(device: str) -> Any:
    base = device.split(":", 1)[0]
    backend_name = _DEVICE_TO_BACKEND.get(base)
    if backend_name is not None:
        return getattr(PersistenceBackend, backend_name)
    raise ValueError(
        f"Unknown device: {device!r}. Supported devices: "
        f"{', '.join(f'{p}[:N]' for p in sorted(_DEVICE_TO_BACKEND))}."
    )


def _seed_rng(seed_value: int) -> None:
    seed_int = int(seed_value)
    if seed_int < 0:
        raise InvalidArgumentError(
            "seed must be non-negative",
            parameter="seed",
            expected=">= 0",
            actual=str(seed_int),
        )
    np.random.seed(seed_int)
    _core = _nerve_state()[0]
    if _core is not None:
        _core.determinism_seed(seed_int)
    _, _, _pytorch = _nerve_state()
    if _pytorch is not None:
        _pytorch.manual_seed(seed_int)
        if hasattr(_pytorch.cuda, "is_available") and _pytorch.cuda.is_available():
            _pytorch.cuda.manual_seed_all(seed_int)


def _compute_with_options(
    core_func: Callable[[np.ndarray, Any], dict[str, Any]],
    points: Any,
    options: Any | None,
    **overrides: Any,
) -> PersistenceResult:
    from ._compute_engine import _require_core  # noqa: PLC0415
    from ._compute_pipeline import (  # noqa: PLC0415
        _is_likely_distance_matrix,
        _resolve_options,
        _to_point_array,
    )

    maybe_dm = _try_as_ndarray(points)
    if maybe_dm is not None and _is_likely_distance_matrix(maybe_dm):
        _, _, pytorch = _nerve_state()
        if pytorch is not None:
            return _persistence_from_distance_matrix(
                maybe_dm,
                overrides.get("max_dim", 2),
            )

    _require_core()
    dtype = overrides.get("dtype")
    max_radius_cap = overrides.get("max_radius_cap")
    point_array = _to_point_array(points, dtype)
    filtered_overrides = {
        k: v for k, v in overrides.items() if k not in ("dtype", "max_radius_cap")
    }
    py_opts = _resolve_options(
        points, options, **filtered_overrides, max_radius_cap=max_radius_cap
    )
    opts = _to_internal_options(py_opts)
    return PersistenceResult.from_dict(core_func(point_array, opts))


def _try_as_ndarray(points: Any) -> np.ndarray | None:
    """Try to convert input to a numpy array for inspection. Returns None on failure."""
    try:
        if isinstance(points, np.ndarray):
            return points
        _, _, pytorch = _nerve_state()
        if pytorch is not None and isinstance(points, pytorch.Tensor):
            arr = points.detach().cpu().numpy()
            if arr.ndim == 2:
                return arr
            return None
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim == 2:
            return arr
        return None
    except (TypeError, ValueError):
        return None


def _persistence_from_distance_matrix(
    distance_matrix: np.ndarray, max_dim: int
) -> PersistenceResult:
    """Compute persistence from a distance matrix using torch."""
    _, _, pytorch = _nerve_state()
    from pynerve.torch import persistence_from_matrix  # noqa: PLC0415

    dm_tensor = pytorch.as_tensor(distance_matrix, dtype=pytorch.float64)
    diagram = persistence_from_matrix(dm_tensor, max_dim=max_dim)

    d = diagram.diagrams.squeeze(0)
    pairs: list[tuple[float, float, int]] = []
    for i in range(d.shape[0]):
        b = d[i, 0].item()
        de = d[i, 1].item()
        dim = int(d[i, 2].item())
        pairs.append((float(b), float(de), dim))

    betti = [
        sum(1 for _, de, dim in pairs if dim == i and not np.isfinite(de))
        for i in range(max_dim + 1)
    ]

    return PersistenceResult(
        pairs=pairs,
        betti_numbers=betti,
        max_dim=max_dim,
        max_radius=float("inf"),
        diagnostics={},
    )
