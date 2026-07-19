"""Optional CuPy import shim."""

from __future__ import annotations

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    HAS_CUPY = False


def _cupy_runtime_available() -> bool:
    if not HAS_CUPY:
        return False
    try:
        assert cp is not None  # type guard: HAS_CUPY ensures cp is imported
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


__all__ = [
    "HAS_CUPY",
    "cp",
    "require_cupy",
    "_cupy_runtime_available",
]


def require_cupy() -> None:
    if not HAS_CUPY:
        raise RuntimeError("CuPy required")
    if not _cupy_runtime_available():
        raise RuntimeError("CuPy installed but CUDA runtime is not available")
