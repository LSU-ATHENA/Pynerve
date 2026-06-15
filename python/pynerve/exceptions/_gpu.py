"""GPU-related exception hierarchy (C++-facing)."""

from __future__ import annotations

from .._error_codes import (
    E10_GPU_OOM,
    E11_GPU_LAUNCH_FAIL,
    ErrorCategory,
)
from ._base import NerveError


class GPUError(NerveError):
    """GPU error base class from C++ GPUError."""

    error_code = E10_GPU_OOM
    error_category = ErrorCategory.GPU_COMPUTE


class GPUMemoryError(GPUError):
    """GPU out of memory error from C++ GPUMemoryError."""

    error_code = E10_GPU_OOM


class GPULaunchError(GPUError):
    """GPU kernel launch error from C++ GPULaunchError."""

    error_code = E11_GPU_LAUNCH_FAIL
