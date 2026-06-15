"""Kernel matrix utilities for persistence-diagram learning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import Tensor

from ._kernels_pairwise import (
    gaussian_kernel,
    linear_kernel,
    persistence_fisher_kernel,
    persistence_scale_space_kernel,
    sliced_wasserstein_kernel,
)

_KERNEL_FUNCTIONS: dict[str, Callable[..., Tensor]] = {
    "gaussian": gaussian_kernel,
    "pss": persistence_scale_space_kernel,
    "sliced_wasserstein": sliced_wasserstein_kernel,
    "fisher": persistence_fisher_kernel,
    "linear": linear_kernel,
}


def _validate_kernel_matrix(
    K: Tensor,  # noqa: N803
    *,
    require_positive_diagonal: bool = False,
) -> Tensor:
    if K.dim() != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")
    if not torch.is_floating_point(K):
        raise TypeError("K must use a floating-point dtype")
    if K.numel() > 0 and not torch.isfinite(K).all().item():
        raise ValueError("K must be finite")
    if require_positive_diagonal and K.shape[0] > 0 and not (torch.diag(K) > 0).all().item():
        raise ValueError("kernel matrix diagonal must be positive")
    return K


def compute_kernel_matrix(
    diagrams: list[Tensor],
    kernel: Literal["gaussian", "pss", "sliced_wasserstein", "fisher", "linear"] = "gaussian",
    **kwargs: Any,
) -> Tensor:
    """Compute a symmetric kernel matrix over a list of diagrams."""
    if kernel not in _KERNEL_FUNCTIONS:
        raise ValueError(f"Unknown kernel: {kernel}")
    kernel_fn = _KERNEL_FUNCTIONS[kernel]

    n = len(diagrams)
    if n == 0:
        return torch.empty((0, 0))
    first = diagrams[0]
    if not isinstance(first, Tensor):
        raise TypeError("diagrams must contain torch.Tensor objects")
    if not torch.is_floating_point(first):
        raise TypeError("diagrams must use a floating-point dtype")
    K = torch.zeros((n, n), dtype=first.dtype, device=first.device)  # noqa: N806
    for i in range(n):
        for j in range(i, n):
            k_val = kernel_fn(diagrams[i], diagrams[j], **kwargs)
            K[i, j] = k_val
            if i != j:
                K[j, i] = k_val
    return K


def normalize_kernel_matrix(K: Tensor) -> Tensor:  # noqa: N803
    """Normalize to unit diagonal."""
    K = _validate_kernel_matrix(K, require_positive_diagonal=True)  # noqa: N806
    diag = torch.sqrt(torch.diag(K))
    diag = diag.clamp_min(torch.finfo(K.dtype).eps)
    K_norm = K / (diag.unsqueeze(0) * diag.unsqueeze(1))  # noqa: N806
    return torch.clamp(K_norm, -1.0, 1.0)


def center_kernel_matrix(K: Tensor) -> Tensor:  # noqa: N803
    """Center kernel matrix for kernel PCA and related methods."""
    K = _validate_kernel_matrix(K)  # noqa: N806
    n = K.shape[0]
    if n == 0:
        return K.clone()
    one_n = torch.ones((n, n), device=K.device, dtype=K.dtype) / n
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n


__all__ = [
    "gaussian_kernel",
    "persistence_scale_space_kernel",
    "sliced_wasserstein_kernel",
    "persistence_fisher_kernel",
    "linear_kernel",
    "compute_kernel_matrix",
    "normalize_kernel_matrix",
    "center_kernel_matrix",
]
