"""Optional Numba import shim for JIT kernels."""

from __future__ import annotations

from typing import Any

try:
    from numba import njit, prange  # noqa: F811

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any) -> Any:
        if (
            args
            and callable(args[0])
            and len(args) == 1
            and not kwargs
            and not isinstance(args[0], bool)
        ):
            return args[0]

        def _decorator(func: Any) -> Any:
            return func

        return _decorator

    prange = range


__all__ = [
    "HAS_NUMBA",
    "njit",
    "prange",
]
