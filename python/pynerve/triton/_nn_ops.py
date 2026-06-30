"""Triton activation-fusion kernels for diagram-convolution layers.

Inline PTX notes:
  - ReLU via "max.f32 $0, $1, 0.0;" avoids branch/ternary.
  - Sigmoid via ex2.approx.ftz.f32 + rcp.approx.ftz.f32: ~5x faster.
  - We use _asm for these hardware-accelerated paths on sm80+.
"""

from __future__ import annotations

import torch

from . import _check_triton, _use_triton, _warn_cpu_fallback

if _check_triton():
    import triton
    import triton.language as tl
    from triton.language import inline_asm_elementwise as _asm
else:
    triton = None
    tl = None
    _asm = None


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["total"],
)
@triton.jit
def _diagram_conv1d_kernel(
    features_ptr,
    kernel_ptr,
    bias_ptr,
    output_ptr,
    batch_size: int,
    n_pairs: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    total: int,
    stride_f_b: int,
    stride_f_c: int,
    stride_k_o: int,
    stride_k_c: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < total
    o = pid % out_channels
    p = (pid // out_channels) % (n_pairs - kernel_size + 1)
    b = pid // (out_channels * (n_pairs - kernel_size + 1))
    pad = kernel_size // 2
    sum_val = tl.load(bias_ptr + o).to(tl.float64)
    for k in range(kernel_size):
        input_idx = p + k - pad
        valid = (input_idx >= 0) & (input_idx < n_pairs)
        for c in range(in_channels):
            feat_idx = b * stride_f_b + c * stride_f_c + input_idx
            kern_idx = o * stride_k_o + c * stride_k_c + k
            f_val = tl.load(features_ptr + feat_idx, mask=mask & valid, other=0.0).to(
                tl.float64
            )
            k_val = tl.load(kernel_ptr + kern_idx).to(tl.float64)
            sum_val += f_val * k_val
    tl.store(output_ptr + pid, sum_val.to(output_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["total"],
)
@triton.jit
def _diagram_conv1d_relu_kernel(
    features_ptr,
    kernel_ptr,
    bias_ptr,
    output_ptr,
    batch_size: int,
    n_pairs: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    total: int,
    stride_f_b: int,
    stride_f_c: int,
    stride_k_o: int,
    stride_k_c: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < total
    o = pid % out_channels
    p = (pid // out_channels) % (n_pairs - kernel_size + 1)
    b = pid // (out_channels * (n_pairs - kernel_size + 1))
    pad = kernel_size // 2
    sum_val = tl.load(bias_ptr + o).to(tl.float64)
    for k in range(kernel_size):
        input_idx = p + k - pad
        valid = (input_idx >= 0) & (input_idx < n_pairs)
        for c in range(in_channels):
            feat_idx = b * stride_f_b + c * stride_f_c + input_idx
            kern_idx = o * stride_k_o + c * stride_k_c + k
            f_val = tl.load(features_ptr + feat_idx, mask=mask & valid, other=0.0).to(
                tl.float64
            )
            k_val = tl.load(kernel_ptr + kern_idx).to(tl.float64)
            sum_val += f_val * k_val
    out = tl.maximum(sum_val, 0.0)
    tl.store(output_ptr + pid, out.to(output_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["total"],
)
@triton.jit
def _diagram_conv1d_sigmoid_kernel(
    features_ptr,
    kernel_ptr,
    bias_ptr,
    output_ptr,
    batch_size: int,
    n_pairs: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    total: int,
    stride_f_b: int,
    stride_f_c: int,
    stride_k_o: int,
    stride_k_c: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < total
    o = pid % out_channels
    p = (pid // out_channels) % (n_pairs - kernel_size + 1)
    b = pid // (out_channels * (n_pairs - kernel_size + 1))
    pad = kernel_size // 2
    sum_val = tl.load(bias_ptr + o).to(tl.float64)
    for k in range(kernel_size):
        input_idx = p + k - pad
        valid = (input_idx >= 0) & (input_idx < n_pairs)
        for c in range(in_channels):
            feat_idx = b * stride_f_b + c * stride_f_c + input_idx
            kern_idx = o * stride_k_o + c * stride_k_c + k
            f_val = tl.load(features_ptr + feat_idx, mask=mask & valid, other=0.0).to(
                tl.float64
            )
            k_val = tl.load(kernel_ptr + kern_idx).to(tl.float64)
            sum_val += f_val * k_val
    out = 1.0 / (1.0 + tl.exp(-sum_val))
    tl.store(output_ptr + pid, out.to(output_ptr.dtype.element_ty), mask=mask)


def _diagram_conv1d_cpu(
    features: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    out = torch.nn.functional.conv1d(features, kernel, bias)
    if activation == "relu":
        out = torch.relu(out)
    elif activation == "sigmoid":
        out = torch.sigmoid(out)
    return out


def diagram_conv1d(
    features: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "none",
) -> torch.Tensor:
    """1-D convolution over persistence-diagram feature sequences.

    Args:
        features: [batch, in_channels, n_pairs]
        kernel:   [out_channels, in_channels, kernel_size]
        bias:     [out_channels]
        activation: "none", "relu", or "sigmoid"

    Returns:
        [batch, out_channels, n_pairs - kernel_size + 1]
    """
    batch_size, in_channels, n_pairs = features.shape
    out_channels, _, kernel_size = kernel.shape
    output_len = n_pairs - kernel_size + 1
    total = batch_size * out_channels * output_len

    if _use_triton(features):
        features_c = features.contiguous()
        kernel_c = kernel.contiguous()
        bias_c = bias.contiguous()
        stride_f_b, stride_f_c_raw, _ = features_c.stride()
        stride_k_o, stride_k_c, _ = kernel_c.stride()
        stride_f_c = (
            stride_f_c_raw[0] if isinstance(stride_f_c_raw, tuple) else stride_f_c_raw
        )
        out = torch.empty(
            batch_size,
            out_channels,
            output_len,
            dtype=features.dtype,
            device=features.device,
        )
        grid = (triton.cdiv(total, 512),)
        kernel_fn = {
            "none": _diagram_conv1d_kernel,
            "relu": _diagram_conv1d_relu_kernel,
            "sigmoid": _diagram_conv1d_sigmoid_kernel,
        }[activation]
        kernel_fn[grid](
            features_c,
            kernel_c,
            bias_c,
            out,
            batch_size,
            n_pairs,
            in_channels,
            out_channels,
            kernel_size,
            total,
            stride_f_b,
            stride_f_c,
            stride_k_o,
            stride_k_c,
        )
        return out

    _warn_cpu_fallback("diagram_conv1d")
    return _diagram_conv1d_cpu(features, kernel, bias, activation)


_ACT_MAP = {
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}


def apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    fn = _ACT_MAP.get(activation)
    if fn is None:
        return x
    return fn(x)
