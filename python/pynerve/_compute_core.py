"""Re-export facade for the compute core subpackage."""

from __future__ import annotations

from ._compute_backend import (
    _compute_with_options,
    _to_events_list,
    _warn_device_overrides_backend,
)
from ._compute_engine import (
    _auto_select_engine,
    _require_core,
    _resolve_engine_func,
)
from ._compute_pipeline import (
    _apply_option_overrides,
    _clone_options,
)
from ._fallback_classes import (
    EventType,
    PersistenceBackend,
    PersistenceEngine,
    PersistenceMode,
    PersistenceOptions,
)
from ._persistence_result import (
    PersistenceResult,
    _estimate_n_points,
    _nerve_state,
)

__all__ = [
    "PersistenceResult",
    "PersistenceEngine",
    "PersistenceBackend",
    "PersistenceMode",
    "PersistenceOptions",
    "EventType",
    "_apply_option_overrides",
    "_auto_select_engine",
    "_clone_options",
    "_compute_with_options",
    "_estimate_n_points",
    "_nerve_state",
    "_require_core",
    "_resolve_engine_func",
    "_to_events_list",
    "_warn_device_overrides_backend",
]
