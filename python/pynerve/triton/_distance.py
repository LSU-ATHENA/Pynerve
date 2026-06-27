"""Triton pairwise-distance kernels."""

from __future__ import annotations

import torch

from . import _check_triton, _use_triton, _warn_cpu_fallback

if _check_triton():
    import triton
    import triton.language as tl
else:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 8}, num_warps=8),
    ],
    key=["dim", "stride_d"],
)
@triton.jit
def _pairwise_sqeuclidean_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_a: int,
    n_b: int,
    dim: int,
    stride_a_m: int,
    stride_a_d: int,
    stride_b_n: int,
    stride_b_d: int,
    stride_out_m: int,
    stride_out_n: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)

    for k in range(0, dim, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rm[:, None] * stride_a_m + rk[None, :] * stride_a_d
        b_ptrs = b_ptr + rk[:, None] * stride_b_d + rn[None, :] * stride_b_n
        a_mask = (rm[:, None] < n_a) & (rk[None, :] < dim)
        b_mask = (rk[:, None] < dim) & (rn[None, :] < n_b)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float64)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float64)
        a_sq = a * a
        acc += -2.0 * tl.dot(a, b.to(tl.float64), allow_tf32=False)
        acc += tl.sum(a_sq, axis=1)[:, None]

    b_sq_accum = tl.zeros((BLOCK_N,), dtype=tl.float64)
    for k in range(0, dim, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        b_ptrs = b_ptr + rk[:, None] * stride_b_d + rn[None, :] * stride_b_n
        b_mask = (rk[:, None] < dim) & (rn[None, :] < n_b)
        b_val = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float64)
        b_sq_accum += tl.sum(b_val * b_val, axis=0)
    acc += b_sq_accum[None, :]

    acc = tl.maximum(acc, 0.0)
    acc = tl.sqrt(acc)
    out_ptrs = out_ptr + rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
    out_mask = (rm[:, None] < n_a) & (rn[None, :] < n_b)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def _pairwise_euclidean_triton(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    n_a, dim = a.shape
    n_b = b.shape[0]
    a_c = a.contiguous()
    b_c = b.contiguous()
    out = torch.empty(n_a, n_b, dtype=a.dtype, device=a.device)
    stride_a_m, stride_a_d = a_c.stride()
    stride_b_d, stride_b_n = b_c.stride()
    stride_out_m, stride_out_n = out.stride()
    grid = (triton.cdiv(n_a, 128), triton.cdiv(n_b, 128))
    _pairwise_sqeuclidean_kernel[grid](
        a_c,
        b_c,
        out,
        n_a,
        n_b,
        dim,
        stride_a_m,
        stride_a_d,
        stride_b_d,
        stride_b_n,
        stride_out_m,
        stride_out_n,
    )
    return out


def pairwise_euclidean(
    a: torch.Tensor,
    b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise Euclidean distance between *a* and *b*.

    If *b* is None, computes the self-distance matrix of *a*.

    shape: a = [n_a, dim], b = [n_b, dim]  ->  out = [n_a, n_b]
    """
    b_t = a if b is None else b
    if b_t.shape[1] != a.shape[1]:
        raise ValueError("dimension mismatch")
    if _use_triton(a) and _use_triton(b_t):
        return _pairwise_euclidean_triton(a, b_t)
    _warn_cpu_fallback("pairwise_euclidean")
    return torch.cdist(a, b_t)


@triton.jit
def _compute_norms_kernel(
    points_ptr,
    norms_ptr,
    n_points: int,
    dim: int,
    stride_m: int,
    stride_d: int,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    row = points_ptr + pid * stride_m
    acc = tl.zeros((1,), dtype=tl.float64)
    for d in range(0, dim, BLOCK_D):
        rd = d + tl.arange(0, BLOCK_D)
        mask = rd < dim
        vals = tl.load(row + rd * stride_d, mask=mask, other=0.0).to(tl.float64)
        acc += tl.sum(vals * vals)
    tl.store(norms_ptr + pid, acc.to(norms_ptr.dtype.element_ty))


def compute_norms(points: torch.Tensor) -> torch.Tensor:
    """Row-wise squared L2 norms."""
    if _use_triton(points):
        n_points, dim = points.shape
        points_c = points.contiguous()
        stride_m, stride_d = points_c.stride()
        norms = torch.empty(n_points, dtype=points.dtype, device=points.device)
        _compute_norms_kernel[(n_points,)](
            points_c, norms, n_points, dim, stride_m, stride_d, BLOCK_D=128
        )
        return norms
    return torch.sum(points * points, dim=1)
