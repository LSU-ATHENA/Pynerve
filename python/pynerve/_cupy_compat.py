"""Optional CuPy import shim."""

from __future__ import annotations

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


__all__ = [
    "HAS_CUPY",
    "cp",
    "require_cupy",
]


def require_cupy() -> None:
    if not HAS_CUPY:
        raise RuntimeError("CuPy required")
