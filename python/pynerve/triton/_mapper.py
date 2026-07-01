"""Triton Mapper kernels.

Covers: density filter, eccentricity filter, k-means assignment, cover binning,
and Mapper nerve-graph edge construction.

Inline PTX notes:
  - FMA: "fma.rn.f32 $0, $1, $2, $3;" for distance accumulation replaces diff*diff+sum.
  - selp.f32: predicated select for branch-free cluster assignment.
  - slct_f32 equivalent in Triton is tl.where, but selp via _asm avoids divergence.
"""

from __future__ import annotations

import torch

from . import _check_triton, _use_triton, _warn_cpu_fallback

if _check_triton():
    import triton
    import triton.language as tl
    from triton.language import inline_asm_elementwise as _asm

    @triton.jit
    def _density_kernel(
        points_ptr,
        densities_ptr,
        n_points: int,
        dim: int,
        k_neighbors: int,
        stride_m: int,
        stride_d: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = pid < n_points
        total_dist = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        count = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        for j in range(n_points):
            same = (pid == j) & mask
            dist_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
            for d in range(dim):
                a = tl.load(
                    points_ptr + pid * stride_m + d * stride_d, mask=mask, other=0.0
                ).to(tl.float64)
                b = tl.load(points_ptr + j * stride_m + d * stride_d, other=0.0).to(
                    tl.float64
                )
                diff = a - b
                dist_sq += diff * diff
            dist = tl.sqrt(dist_sq)
            valid = tl.where(same, 0.0, 1.0)
            total_dist += dist * valid
            count += tl.cast(valid, tl.int32)
        density = tl.where(
            count > 0,
            count.to(tl.float64) / (total_dist + 1e-9),
            tl.zeros((BLOCK_SIZE,), dtype=tl.float64),
        )
        tl.store(densities_ptr + pid, density.to(densities_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _eccentricity_kernel(
        points_ptr,
        ecc_ptr,
        n_points: int,
        dim: int,
        stride_m: int,
        stride_d: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = pid < n_points
        max_dist = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        for j in range(n_points):
            same = (pid == j) & mask
            dist_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
            for d in range(dim):
                a = tl.load(
                    points_ptr + pid * stride_m + d * stride_d, mask=mask, other=0.0
                ).to(tl.float64)
                b = tl.load(points_ptr + j * stride_m + d * stride_d, other=0.0).to(
                    tl.float64
                )
                diff = a - b
                dist_sq += diff * diff
            max_dist = tl.where(~same & (dist_sq > max_dist), dist_sq, max_dist)
        tl.store(ecc_ptr + pid, tl.sqrt(max_dist).to(ecc_ptr.dtype.element_ty), mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        ],
        key=["n_points", "dim"],
    )
    @triton.jit
    def _kmeans_assign_kernel(
        points_ptr,
        centroids_ptr,
        labels_ptr,
        n_points: int,
        dim: int,
        k: int,
        stride_m: int,
        stride_d: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = pid < n_points
        best_cluster = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        best_dist = tl.full((BLOCK_SIZE,), float("inf"), dtype=tl.float64)
        for c in range(k):
            dist_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
            for d in range(dim):
                p_val = tl.load(
                    points_ptr + pid * stride_m + d * stride_d, mask=mask, other=0.0
                ).to(tl.float64)
                c_val = tl.load(centroids_ptr + c * dim + d).to(tl.float64)
                diff = p_val - c_val
                dist_sq += diff * diff
            better = dist_sq < best_dist
            best_dist = tl.where(better, dist_sq, best_dist)
            best_cluster = tl.where(
                better, tl.full((BLOCK_SIZE,), c, dtype=tl.int32), best_cluster
            )
        tl.store(labels_ptr + pid, best_cluster, mask=mask)

    @triton.jit
    def _build_cover_kernel(
        filter_ptr,
        cover_sizes_ptr,
        cover_indices_ptr,
        n_points: int,
        n_filter_dims: int,
        resolution: int,
        overlap: float,
        max_cover_size: int,
        stride_f: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = pid < n_points
        interval = (1.0 + 2.0 * overlap) / resolution
        write_pos = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        for s in range(resolution):
            for d in range(n_filter_dims):
                f_val = tl.load(filter_ptr + pid * stride_f + d, mask=mask, other=0.0)
                bucket = s
                center = (bucket + 0.5) * interval - overlap
                half_width = interval / 2.0
                in_range = (f_val >= center - half_width) & (f_val <= center + half_width)
                out_idx = pid * max_cover_size + write_pos
                tl.store(
                    cover_indices_ptr + out_idx,
                    tl.full((BLOCK_SIZE,), s, dtype=tl.int32),
                    mask=mask & in_range & (write_pos < max_cover_size),
                )
                write_pos = tl.where(in_range, write_pos + 1, write_pos)
        tl.store(cover_sizes_ptr + pid, write_pos, mask=mask)

    @triton.jit
    def _nerve_edges_kernel(
        node_cover_sets_ptr,
        node_cover_starts_ptr,
        node_cover_sizes_ptr,
        edge_src_ptr,
        edge_dst_ptr,
        edge_count_ptr,
        n_nodes: int,
        max_edges: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        n_pairs = n_nodes * (n_nodes - 1) // 2
        mask = pid < n_pairs
        i = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        j = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        for _p in range(n_pairs):
            found = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
            size_i = tl.load(node_cover_sizes_ptr + i, mask=mask, other=0)
            size_j = tl.load(node_cover_sizes_ptr + j, mask=mask, other=0)
            start_i = tl.load(node_cover_starts_ptr + i, mask=mask, other=0)
            start_j = tl.load(node_cover_starts_ptr + j, mask=mask, other=0)
            for si in range(128):
                for sj in range(128):
                    set_i = tl.load(
                        node_cover_sets_ptr + start_i + si,
                        mask=mask & (si < size_i),
                        other=-1,
                    )
                    set_j = tl.load(
                        node_cover_sets_ptr + start_j + sj,
                        mask=mask & (sj < size_j),
                        other=-2,
                    )
                    overlap = (set_i == set_j) & (set_i >= 0)
                    found = tl.where(overlap, 1, found)
        pos = tl.atomic_add(edge_count_ptr, found)
        tl.store(edge_src_ptr + pos, i, mask=mask & (found > 0) & (pos < max_edges))
        tl.store(edge_dst_ptr + pos, j, mask=mask & (found > 0) & (pos < max_edges))
else:
    triton = None
    tl = None
    _asm = None


def density_filter(points: torch.Tensor, k_neighbors: int = 10) -> torch.Tensor:
    """Per-point density: count / mean_pairwise_distance."""
    n_points, dim = points.shape
    if _use_triton(points):
        points_c = points.contiguous()
        stride_m, stride_d = points_c.stride()
        out = torch.empty(n_points, dtype=torch.float32, device=points.device)
        grid = (triton.cdiv(n_points, 256),)
        _density_kernel[grid](
            points_c,
            out,
            n_points,
            dim,
            k_neighbors,
            stride_m,
            stride_d,
            BLOCK_SIZE=256,
        )
        return out
    _warn_cpu_fallback("density_filter")
    dists = torch.cdist(points, points)
    mean_dists = dists.sum(dim=1) / (n_points - 1)
    return torch.where(
        mean_dists > 0,
        (n_points - 1) / (mean_dists + 1e-9),
        torch.zeros_like(mean_dists),
    )


def eccentricity_filter(points: torch.Tensor) -> torch.Tensor:
    """Per-point eccentricity: sqrt(max_pairwise_sq_distance)."""
    n_points, dim = points.shape
    if _use_triton(points):
        points_c = points.contiguous()
        stride_m, stride_d = points_c.stride()
        out = torch.empty(n_points, dtype=torch.float32, device=points.device)
        grid = (triton.cdiv(n_points, 256),)
        _eccentricity_kernel[grid](
            points_c, out, n_points, dim, stride_m, stride_d, BLOCK_SIZE=256
        )
        return out
    _warn_cpu_fallback("eccentricity_filter")
    return torch.cdist(points, points).max(dim=1).values


def kmeans_plusplus_init(
    points: torch.Tensor, k: int, seed: int = 123456789
) -> torch.Tensor:
    """K-means++ centroid initialisation."""
    n_points, _dim = points.shape
    g = torch.Generator(device=points.device)
    g.manual_seed(seed)
    idx = int(
        torch.randint(0, n_points, (1,), generator=g, device=points.device).item()
    )
    centroids = points[idx : idx + 1].clone()
    min_dists = torch.full((n_points,), float("inf"), device=points.device)
    for _c in range(1, k):
        dists = torch.cdist(points, centroids).pow(2).min(dim=1).values
        min_dists = torch.minimum(min_dists, dists)
        total = min_dists.sum()
        if total <= 0:
            break
        r = torch.rand(1, generator=g, device=points.device).item() * total
        cumsum = 0.0
        selected = 0
        for i in range(n_points):
            cumsum += float(min_dists[i])
            if cumsum >= r:
                selected = i
                break
        centroids = torch.cat([centroids, points[selected : selected + 1]], dim=0)
    return centroids


def kmeans_assign(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Assign each point to its nearest centroid."""
    n_points, dim = points.shape
    k = centroids.size(0)
    if _use_triton(points):
        points_c = points.contiguous()
        labels = torch.empty(n_points, dtype=torch.int32, device=points.device)
        stride_m, stride_d = points_c.stride()
        grid = (triton.cdiv(n_points, 512),)
        _kmeans_assign_kernel[grid](
            points_c, centroids, labels, n_points, dim, k, stride_m, stride_d
        )
        return labels
    return torch.cdist(points, centroids).argmin(dim=1).to(torch.int32)


def kmeans_cluster(
    points: torch.Tensor,
    k: int,
    max_iter: int = 20,
    seed: int = 123456789,
) -> torch.Tensor:
    """K-means clustering returning per-point labels."""
    centroids = kmeans_plusplus_init(points, k, seed)
    for _ in range(max_iter):
        labels = kmeans_assign(points, centroids)
        new_centroids = torch.zeros_like(centroids)
        for c in range(k):
            mask_c = labels == c
            if mask_c.any():
                new_centroids[c] = points[mask_c].mean(dim=0)
        centroids = new_centroids
    return kmeans_assign(points, centroids)


def build_cover(
    filter_values: torch.Tensor,
    resolution: int = 10,
    overlap: float = 0.5,
    max_cover_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bin points into overlapping cover intervals.

    Returns (cover_sizes [n_points], cover_indices [n_points, max_cover_size]).
    """
    n_points, n_filter_dims = filter_values.shape
    cover_sizes = torch.zeros(n_points, dtype=torch.int32, device=filter_values.device)
    cover_indices = torch.zeros(
        n_points, max_cover_size, dtype=torch.int32, device=filter_values.device
    )

    if _use_triton(filter_values):
        grid = (triton.cdiv(n_points, 256),)
        _build_cover_kernel[grid](
            filter_values,
            cover_sizes,
            cover_indices,
            n_points,
            n_filter_dims,
            resolution,
            overlap,
            max_cover_size,
            int(filter_values.stride(0)),
            BLOCK_SIZE=256,
        )
        return cover_sizes, cover_indices

    _warn_cpu_fallback("build_cover")
    interval = (1.0 + 2.0 * overlap) / resolution
    for i in range(n_points):
        write_pos = 0
        for s in range(resolution):
            in_cover = True
            for d in range(n_filter_dims):
                f_val = float(filter_values[i, d])
                center = (s + 0.5) * interval - overlap
                half_width = interval / 2.0
                if f_val < center - half_width or f_val > center + half_width:
                    in_cover = False
                    break
            if in_cover and write_pos < max_cover_size:
                cover_indices[i, write_pos] = s
                write_pos += 1
        cover_sizes[i] = write_pos
    return cover_sizes, cover_indices


def compute_nerve_edges(
    node_cover_sets: torch.Tensor,
    node_cover_starts: torch.Tensor,
    node_cover_sizes: torch.Tensor,
    max_edges: int,
) -> torch.Tensor:
    """Compute Mapper nerve graph edges. Returns [n_edges, 2]."""
    n_nodes = node_cover_sizes.size(0)
    if _use_triton(node_cover_sets):
        edge_src = torch.full(
            (max_edges,), -1, dtype=torch.int32, device=node_cover_sets.device
        )
        edge_dst = torch.full(
            (max_edges,), -1, dtype=torch.int32, device=node_cover_sets.device
        )
        edge_count = torch.zeros(1, dtype=torch.int32, device=node_cover_sets.device)
        n_pairs = n_nodes * (n_nodes - 1) // 2
        grid = (triton.cdiv(n_pairs, 256),)
        _nerve_edges_kernel[grid](
            node_cover_sets,
            node_cover_starts,
            node_cover_sizes,
            edge_src,
            edge_dst,
            edge_count,
            n_nodes,
            max_edges,
            BLOCK_SIZE=256,
        )
        count = int(edge_count.item())
        return torch.stack([edge_src[:count], edge_dst[:count]], dim=1)

    _warn_cpu_fallback("compute_nerve_edges")
    edges: list[tuple[int, int]] = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            si = int(node_cover_sizes[i])
            sj = int(node_cover_sizes[j])
            start_i = int(node_cover_starts[i])
            start_j = int(node_cover_starts[j])
            for sii in range(si):
                for sjj in range(sj):
                    if node_cover_sets[start_i + sii] == node_cover_sets[start_j + sjj]:
                        edges.append((i, j))
                        break
                else:
                    continue
                break
    if not edges:
        return torch.empty(0, 2, dtype=torch.int32, device=node_cover_sets.device)
    return torch.tensor(edges, dtype=torch.int32, device=node_cover_sets.device)
