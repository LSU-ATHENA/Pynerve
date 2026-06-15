"""Engine selection and resolution helpers."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, cast

from ._fallback_classes import PersistenceBackend, PersistenceEngine, PersistenceMode
from ._persistence_result import _nerve_state
from .exceptions import BackendRequiredError

_ENGINE_TO_METHOD: dict[PersistenceEngine, str] = {
    PersistenceEngine.PH4: "compute_persistence_ph4",
    PersistenceEngine.PH5: "compute_persistence_ph5",
    PersistenceEngine.PH6: "compute_persistence_ph6",
    PersistenceEngine.PH3: "compute_persistence_cohomology",
}


def _auto_select_engine(
    n_points: int,
    dim: int,
    device: str | None,
    mode: PersistenceMode | str | None,
    _backend: PersistenceBackend | str | None,
) -> PersistenceEngine:
    """Select the best persistence engine based on input characteristics.

    The selection rule is:

    - n < 1K: PH0 (standard homology, fastest for tiny data)
    - n < 10K: PH3 (basic cohomology)
    - n < 100K: PH4 (cohomology + clearing)
    - n >= 100K, < 1M: PH5 (unified adaptive engine)
    - n >= 1M: PH6 (block-sparse speculative engine)
    - GPU device: PH5
    - APPROX mode: PH5
    """
    if device and device.startswith("cuda"):
        return PersistenceEngine.PH5

    mode_str = mode.value if isinstance(mode, Enum) else str(mode or "")
    if mode_str.upper() == "APPROX":
        return PersistenceEngine.PH5

    if n_points < 1_000:
        return PersistenceEngine.PH0 if dim <= 3 else PersistenceEngine.PH3
    elif n_points < 10_000:
        return PersistenceEngine.PH3
    elif n_points < 100_000:
        return PersistenceEngine.PH4
    else:
        return PersistenceEngine.PH6 if n_points >= 1_000_000 else PersistenceEngine.PH5


def _require_core() -> Any:
    _core, _core_import_error = _nerve_state()[:2]
    if _core is not None:
        return _core
    raise BackendRequiredError(
        "pynerve_internal is required for core persistence operations.",
        backend="pynerve_internal",
        installation_hint=str(_core_import_error),
    )


def _resolve_engine_func(
    core: Any,
    engine: PersistenceEngine,
    n_points: int = 0,
    dim: int = 3,
    device: str | None = None,
    mode: PersistenceMode | str | None = None,
    backend: PersistenceBackend | str | None = None,
) -> Callable[..., Any]:
    engine_use = engine
    if engine_use == PersistenceEngine.AUTO:
        engine_use = _auto_select_engine(n_points, dim, device, mode, backend)

    if engine_use in _ENGINE_TO_METHOD:
        method_name = _ENGINE_TO_METHOD[engine_use]
        if hasattr(core, method_name):
            return cast(Callable[..., Any], getattr(core, method_name))
    return cast(Callable[..., Any], core.compute_persistence)
