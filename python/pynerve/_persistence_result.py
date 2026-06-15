"""PersistenceResult data class and module-level state.

Environment variables:
    NERVE_MAX_RADIUS_CAP: Upper bound for ``max_radius=inf`` (default: 1e15).
        Set to your data's max pairwise distance to prevent OOM.
    NERVE_COORD_RANGE_WARN: Threshold for coordinate range warning (default: 1e6).
    NERVE_NPOINTS_WARN: Threshold for large point count warning (default: 1e6).
"""

from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _estimate_n_points(points: Any) -> int:
    """Safely estimate the number of points in an array-like input."""
    try:
        if hasattr(points, "shape"):
            return points.shape[0] if len(points.shape) >= 1 else 0
        return len(points)
    except (TypeError, AttributeError):
        return 0


_MAX_RADIUS_CAP = float(os.environ.get("NERVE_MAX_RADIUS_CAP", "1e15"))

_CAP_WARNED = [False]  # mutable to avoid global statement
_CAP_WARN_LOCK = threading.Lock()


def _warn_large_max_radius_cap() -> None:
    """Emit a one-time warning when the large default max_radius_cap is activated."""
    if _CAP_WARNED[0] or _MAX_RADIUS_CAP < 1e14:
        return
    with _CAP_WARN_LOCK:
        if _CAP_WARNED[0]:
            return
        _CAP_WARNED[0] = True
    import warnings  # noqa: PLC0415

    warnings.warn(
        f"NERVE_MAX_RADIUS_CAP is {_MAX_RADIUS_CAP:.0e} (default 1e15). "
        "This very large default can cause out-of-memory errors. "
        "Set NERVE_MAX_RADIUS_CAP to a smaller value (e.g. your data's max pairwise distance) "
        "or pass max_radius_cap explicitly.",
        UserWarning,
        stacklevel=2,
    )


# Thread-safe lazy import of pynerve package state
_CORE: Any = None
_CORE_IMPORT_ERROR: Any = None
_PYTORCH: Any = None
_NERVE_STATE_LOCK: Any = threading.Lock()


def _nerve_state() -> tuple[Any, Any, Any]:
    global _CORE, _CORE_IMPORT_ERROR, _PYTORCH  # noqa: PLW0603
    if _CORE is not None or _CORE_IMPORT_ERROR is not None:
        return _CORE, _CORE_IMPORT_ERROR, _PYTORCH
    with _NERVE_STATE_LOCK:
        if _CORE is None and _CORE_IMPORT_ERROR is None:
            import sys as _sys  # noqa: PLC0415

            _nerve_mod = _sys.modules.get("pynerve")
            if _nerve_mod is not None:
                _CORE = getattr(_nerve_mod, "_core", None)
                _CORE_IMPORT_ERROR = getattr(_nerve_mod, "_core_import_error", None)
                _PYTORCH = getattr(_nerve_mod, "_pytorch", None)
    return _CORE, _CORE_IMPORT_ERROR, _PYTORCH


@dataclass
class PersistenceResult:
    """Result of a persistence computation.

    Returned by :func:`pynerve.compute_persistence` and related functions.
    All fields are populated after a successful computation.

    Attributes:
        pairs: List of ``(birth, death, dimension)`` tuples. Each tuple
            represents one persistent homology feature with its birth time,
            death time, and homology dimension. Infinite deaths (essential
            classes) have ``death=inf``.
        betti_numbers: Betti numbers ``[b0, b1, ..., b_{max_dim}]``, one
            per homology dimension. ``b0`` is the number of connected
            components, ``b1`` the number of loops, ``b2`` the number of
            voids, etc.
        max_dim: Maximum homology dimension that was computed.
        max_radius: Filtration radius cutoff used in the computation.
        diagnostics: Optional diagnostic information. When computation
            diagnostics are enabled, this dict may contain keys such as
            ``num_simplices``, ``peak_memory_bytes``, ``computation_time_ms``,
            and backend-specific metrics.

    Example:

        >>> import pynerve
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> points = rng.random((100, 3))
        >>> result = pynerve.compute_persistence(points, max_dim=2, max_radius=1.0)
        >>> result.betti_numbers  # doctest: +SKIP
        [1, 3, 0]
        >>> len(result.pairs)  # doctest: +SKIP
        5
        >>> result.pairs_array.shape  # doctest: +SKIP
        (5, 3)
    """

    pairs: list[tuple[float, float, int]] = field(default_factory=list)
    betti_numbers: list[int] = field(default_factory=list)
    max_dim: int = 0
    max_radius: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    _pairs_array_cache: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def num_pairs(self) -> int:
        """Number of persistence pairs in the result."""
        return len(self.pairs)

    @property
    def pairs_array(self) -> np.ndarray:
        """Persistence pairs as an (N, 3) numpy array with columns (birth, death, dimension).

        More convenient than ``.pairs`` for vectorized operations and plotting.
        The result is cached; access is O(1) after the first call.
        """
        if self._pairs_array_cache is not None:
            return self._pairs_array_cache
        arr = np.array(self.pairs, dtype=np.float64)
        self._pairs_array_cache = arr
        return arr

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersistenceResult:
        """Create a PersistenceResult from a dictionary of attributes."""
        known_fields = {"pairs", "betti_numbers", "max_dim", "max_radius", "diagnostics"}
        unknown = set(data) - known_fields
        if unknown:
            warnings.warn(
                f"Unknown fields in persistence result: {sorted(unknown)}. "
                "This may indicate a version mismatch between Python and C++ extension.",
                UserWarning,
                stacklevel=2,
            )
        pairs = data.get("pairs", [])
        if pairs:
            arr = np.array(pairs, dtype=np.float64)
            clamp = (arr[:, 1] < arr[:, 0]) & np.isfinite(arr[:, 1])
            if clamp.any():
                arr[clamp, 1] = arr[clamp, 0]
            pairs = [(float(arr[i, 0]), float(arr[i, 1]), int(arr[i, 2])) for i in range(len(arr))]
        return cls(
            pairs=pairs,
            betti_numbers=data.get("betti_numbers", []),
            max_dim=data.get("max_dim", 0),
            max_radius=data.get("max_radius", 0.0),
            diagnostics=data.get("diagnostics", {}),
        )

    def __repr__(self) -> str:
        """Return a concise string representation of the result."""
        parts = [
            f"pairs={len(self.pairs)}",
            f"betti_numbers={self.betti_numbers}",
            f"max_dim={self.max_dim}",
            f"max_radius={self.max_radius}",
        ]
        if self.diagnostics:
            parts.append(f"diagnostics={list(self.diagnostics.keys())}")
        return f"PersistenceResult({', '.join(parts)})"

    def __str__(self) -> str:
        return self.__repr__()
