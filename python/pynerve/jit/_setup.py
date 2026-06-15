"""Numba availability detection and fallback decorators."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

try:
    from numba import cuda as _cuda_module  # type: ignore[assignment]
    from numba import jit, prange

    HAS_NUMBA = True
    HAS_CUDA = True
except ImportError:
    HAS_NUMBA = False
    HAS_CUDA = False
    _cuda_module = None

    def jit(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            "Numba is not installed. JIT-decorated functions will run as pure Python "
            "and may be significantly slower. Install with: pip install pynerve[ml]",
            RuntimeWarning,
            stacklevel=2,
        )
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return _decorator

    prange = range

cuda: Any = _cuda_module
if not HAS_CUDA:
    warnings.warn("Numba CUDA JIT is not active; CPU JIT helpers remain usable", stacklevel=2)

__all__ = [
    "HAS_CUDA",
    "HAS_NUMBA",
    "cuda",
    "jit",
    "prange",
]
