"""Triton GPU kernels for Nerve.

Structure:
  _distance.py     -- Pairwise distance, norms
  _persistence.py  -- Persistence image rasterisation
  _laplacian.py    -- CSR SpMV, AXPY, scale, orthogonalise
  _wasserstein.py  -- Wasserstein/Sinkhorn distance
  _mapper.py       -- Mapper filter, k-means, cover, nerve edges
  _nn_ops.py       -- Activation fusion for diagram conv
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

_TRITON_AVAILABLE: bool | None = None


def _check_triton() -> bool:
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        try:
            importlib.import_module("triton")
            importlib.import_module("triton.language")
            _TRITON_AVAILABLE = True
        except ImportError:
            _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


def _warn_cpu_fallback(name: str) -> None:
    warnings.warn(
        f"Triton is unavailable or tensor is on CPU; {name} will use PyTorch fallback.",
        RuntimeWarning,
        stacklevel=3,
    )


def _use_triton(tensor: Any) -> bool:
    import torch

    return bool(
        tensor.device.type == "cuda" and torch.cuda.is_available() and _check_triton()
    )


__all__ = ["_check_triton", "_use_triton"]
