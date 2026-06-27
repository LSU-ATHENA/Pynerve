"""Triton persistent-Laplacian kernels."""

from __future__ import annotations

import triton
import triton.language as tl

import torch

from . import _use_triton, _warn_cpu_fallback


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def _csr_spmv_kernel(
    row_offsets_ptr,
    col_indices_ptr,
    values_ptr,
    x_ptr,
    y_ptr,
    n: int,
    nnz: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < n
    row = pid

    row_start = tl.load(row_offsets_ptr + row, mask=mask, other=0)
    row_end = tl.load(row_offsets_ptr + row + 1, mask=mask, other=0)

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    for j in range(nnz):
        j_idx = tl.arange(0, BLOCK_SIZE) + j
        col = tl.load(col_indices_ptr + j_idx, mask=j_idx < nnz, other=0)
        val = tl.load(values_ptr + j_idx, mask=j_idx < nnz, other=0.0).to(tl.float64)
        x_val = tl.load(x_ptr + col, mask=col < n, other=0.0).to(tl.float64)
        acc += val * x_val

    tl.store(y_ptr + pid, acc.to(y_ptr.dtype.element_ty), mask=mask)


def csr_spmv(
    row_offsets: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """CSR sparse-matrix × dense-vector: y = A · x."""
    n = row_offsets.size(0) - 1
    if _use_triton(x):
        y = torch.empty(n, dtype=x.dtype, device=x.device)
        nnz = values.size(0)
        grid = (triton.cdiv(n, 512),)
        _csr_spmv_kernel[grid](row_offsets, col_indices, values, x, y, n, nnz)
        return y
    _warn_cpu_fallback("csr_spmv")
    y = torch.zeros(n, dtype=x.dtype, device=x.device)
    for i in range(n):
        start = int(row_offsets[i])
        end = int(row_offsets[i + 1])
        for j in range(start, end):
            col = int(col_indices[j])
            y[i] += values[j] * x[col]
    return y


def axpy(alpha: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """y += alpha * x  (in-place)."""
    y.add_(x, alpha=alpha)
    return y


def scale(alpha: float, x: torch.Tensor) -> torch.Tensor:
    """x *= alpha  (in-place)."""
    x.mul_(alpha)
    return x


def orthogonalize(v: torch.Tensor, w: torch.Tensor, dot: float) -> torch.Tensor:
    """w -= dot * v  (in-place)."""
    w.sub_(v, alpha=dot)
    return w
