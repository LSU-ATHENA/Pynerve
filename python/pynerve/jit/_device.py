"""Compute device resolution for JIT kernels."""

from __future__ import annotations

from ..exceptions import InvalidArgumentError
from ._setup import HAS_CUDA


def _resolve_device(device: str | None) -> bool:
    if device is None or device == "cpu":
        return False
    if device == "cuda":
        if not HAS_CUDA:
            raise RuntimeError("CUDA device requested but Numba CUDA is not available")
        return True
    raise InvalidArgumentError(
        f"unsupported device: {device!r} (expected 'cpu' or 'cuda')", parameter="device"
    )
