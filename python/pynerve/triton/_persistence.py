"""Triton persistence-image kernels.

Two strategies:
  persistence_image_pixel   -- iterate pairs per pixel (many pairs, moderate resolution)
  persistence_image_pair    -- iterate pixels per pair   (few pairs, high resolution)
The auto-select picks whichever is expected to touch fewer global-memory elements.

Inline PTX notes:
  - Gaussian exp(-dist²/(2σ²)) -> ex2.approx.ftz.f32(dist_sq * scale)
    scale = -1/(2σ²·ln(2)) = -0.72134752/σ². ~4x faster with <0.5% error.
  - FMA available as tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", ...)
  - atom.add.f32 can use relaxed memory scope for image accumulation.
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


def _select_strategy(n_pairs: int, resolution: int) -> str:
    if n_pairs > resolution * resolution:
        return "pixel"
    return "pair"


@triton.jit
def _pixel_kernel(
    births_ptr,
    deaths_ptr,
    image_ptr,
    n_pairs: int,
    resolution: int,
    sigma: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    stride_img_y: int,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid_x = tl.program_id(0) * BLOCK_X
    pid_y = tl.program_id(1) * BLOCK_Y
    lx = pid_x + tl.arange(0, BLOCK_X)
    ly = pid_y + tl.arange(0, BLOCK_Y)
    mask = (lx < resolution) & (ly < resolution)
    x_range = tl.maximum(x_max - x_min, 1.0)
    y_range = tl.maximum(y_max - y_min, 1.0)
    pixel_x = x_min + (lx.to(tl.float64) + 0.5) * x_range / resolution
    pixel_y = y_min + (ly.to(tl.float64) + 0.5) * y_range / resolution
    sigma_sq2 = 2.0 * sigma * sigma
    neg_inv = -1.0 / sigma_sq2
    ln2 = 0.69314718
    ex2_scale = neg_inv / ln2
    weight_sum = tl.zeros((BLOCK_X,), dtype=tl.float64)
    for i in range(n_pairs):
        birth = tl.load(births_ptr + i).to(tl.float64)
        death = tl.load(deaths_ptr + i).to(tl.float64)
        persistence = death - birth
        dx = pixel_x - birth
        dy = pixel_y - persistence
        dist2 = dx * dx + dy * dy
        w = tl.exp(ex2_scale * dist2)
        weight_sum += w
    img_ptr = image_ptr + ly * stride_img_y + lx
    tl.atomic_add(img_ptr, weight_sum.to(image_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _pair_kernel(
    births_ptr,
    deaths_ptr,
    image_ptr,
    n_pairs: int,
    resolution: int,
    sigma: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    stride_img_y: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < n_pairs
    birth = tl.load(births_ptr + pid, mask=mask, other=0.0).to(tl.float64)
    death = tl.load(deaths_ptr + pid, mask=mask, other=0.0).to(tl.float64)
    persistence = death - birth
    x_range = tl.maximum(x_max - x_min, 1.0)
    y_range = tl.maximum(y_max - y_min, 1.0)
    sigma_sq2 = 2.0 * sigma * sigma
    neg_inv = -1.0 / sigma_sq2
    bx_rel = (birth - x_min) / x_range * (resolution - 1)
    py_rel = (persistence - y_min) / y_range * (resolution - 1)
    radius_x = 3.0 * sigma * resolution / x_range
    radius_y = 3.0 * sigma * resolution / y_range
    x0 = tl.maximum(0, tl.cast(tl.floor(bx_rel - radius_x), tl.int32))
    x1 = tl.minimum(resolution - 1, tl.cast(tl.ceil(bx_rel + radius_x), tl.int32))
    y0 = tl.maximum(0, tl.cast(tl.floor(py_rel - radius_y), tl.int32))
    y1 = tl.minimum(resolution - 1, tl.cast(tl.ceil(py_rel + radius_y), tl.int32))
    for y in range(resolution):
        pixel_y = y_min + (y + 0.5) * y_range / resolution
        dy = pixel_y - persistence
        for x in range(resolution):
            in_bounds = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            if not in_bounds:
                continue
            pixel_x = x_min + (x + 0.5) * x_range / resolution
            dx = pixel_x - birth
            dist2 = dx * dx + dy * dy
            w = tl.exp(neg_inv * dist2)
            tl.atomic_add(
                image_ptr + y * stride_img_y + x,
                w.to(image_ptr.dtype.element_ty),
                mask=mask,
            )


def _bounds_and_valid(
    births_f: torch.Tensor,
    deaths_f: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, float, float, float, float, float, float, float, float
]:
    finite_mask = torch.isfinite(deaths_f) & (deaths_f > births_f)
    valid = finite_mask.nonzero(as_tuple=False).squeeze(-1)
    b_valid = births_f[valid]
    d_valid = deaths_f[valid]
    pers = d_valid - b_valid
    b_min = float(b_valid.min())
    b_max = float(b_valid.max())
    p_min = float(pers.min())
    p_max = float(pers.max())
    if b_max <= b_min:
        b_max = b_min + 1.0
    if p_max <= p_min:
        p_max = p_min + 1.0
    x_range = b_max - b_min
    y_range = p_max - p_min
    x_min = b_min - 0.1 * x_range
    x_max = b_max + 0.1 * x_range
    y_min = p_min - 0.1 * y_range
    y_max = p_max + 0.1 * y_range
    return b_valid, d_valid, x_min, x_max, y_min, y_max, x_range, y_range, b_min, p_min


def _persistence_image_cpu(
    b_valid: torch.Tensor,
    d_valid: torch.Tensor,
    resolution: int,
    sigma: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> torch.Tensor:
    n = b_valid.size(0)
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    sigma_sq2 = 2.0 * sigma * sigma
    xs = (
        x_min
        + (torch.arange(resolution, dtype=torch.float32, device=b_valid.device) + 0.5)
        * x_range
        / resolution
    )
    ys = (
        y_min
        + (torch.arange(resolution, dtype=torch.float32, device=b_valid.device) + 0.5)
        * y_range
        / resolution
    )
    image = torch.zeros(
        resolution, resolution, dtype=torch.float32, device=b_valid.device
    )
    for i in range(n):
        b = b_valid[i]
        p = d_valid[i] - b
        dx = xs - b
        dy = ys[:, None] - p
        dist2 = dx**2 + dy**2
        w = torch.exp(-dist2 / sigma_sq2)
        image += w
    return image


def persistence_image_from_diagram(
    births: torch.Tensor,
    deaths: torch.Tensor,
    resolution: int = 64,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Rasterise a persistence diagram to a weighted Gaussian image."""
    n = min(births.size(0), deaths.size(0))
    if n == 0:
        return torch.zeros(
            resolution, resolution, dtype=torch.float32, device=births.device
        )
    births_f = births[:n].float().contiguous()
    deaths_f = deaths[:n].float().contiguous()
    b_valid, d_valid, x_min, x_max, y_min, y_max, _xr, _yr, _bm, _pm = (
        _bounds_and_valid(births_f, deaths_f)
    )
    if b_valid.numel() == 0:
        return torch.zeros(
            resolution, resolution, dtype=torch.float32, device=births.device
        )

    if _use_triton(births):
        image = torch.zeros(
            resolution, resolution, dtype=torch.float32, device=births.device
        )
        stride_img_y = image.stride(0)
        strategy = _select_strategy(int(b_valid.numel()), resolution)
        if strategy == "pixel":
            grid = (triton.cdiv(resolution, 16), triton.cdiv(resolution, 16))
            _pixel_kernel[grid](
                b_valid,
                d_valid,
                image,
                int(b_valid.numel()),
                resolution,
                sigma,
                x_min,
                x_max,
                y_min,
                y_max,
                stride_img_y,
                BLOCK_X=16,
                BLOCK_Y=16,
            )
        else:
            grid = (triton.cdiv(int(b_valid.numel()), 256),)
            _pair_kernel[grid](
                b_valid,
                d_valid,
                image,
                int(b_valid.numel()),
                resolution,
                sigma,
                x_min,
                x_max,
                y_min,
                y_max,
                stride_img_y,
                BLOCK_SIZE=256,
            )
        return image

    _warn_cpu_fallback("persistence_image_from_diagram")
    return _persistence_image_cpu(
        b_valid, d_valid, resolution, sigma, x_min, x_max, y_min, y_max
    )
