#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>

namespace nerve::gpu::tensorcore
{

template <int TILE_M = 16, int TILE_N = 16, int TILE_K = 16>
__global__ __launch_bounds__(256)
    pairwiseDistanceWmma(const half *__restrict__ points, float *__restrict__ normsq,
                         float *__restrict__ distances, int n, int dim)
{
    using namespace nvcuda::wmma;

    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    if (block_row >= n || block_col >= n)
        return;

    fragment<matrix_a, TILE_M, TILE_N, TILE_K, half, row_major> a_frag;
    fragment<matrix_b, TILE_M, TILE_N, TILE_K, half, row_major> b_frag;
    fragment<accumulator, TILE_M, TILE_N, TILE_K, float> dot_frag;
    fill_fragment(dot_frag, 0.0f);

    for (int k = 0; k < dim; k += TILE_K)
    {
        int k_end = min(k + TILE_K, dim);
        int actual_k = k_end - k;

        if (actual_k == TILE_K)
        {
            load_matrix_sync(a_frag, &points[static_cast<size_t>(block_row) * dim + k], dim);
            load_matrix_sync(b_frag, &points[static_cast<size_t>(block_col) * dim + k], dim);
            mma_sync(dot_frag, a_frag, b_frag, dot_frag);
        }
    }

    float dot_val[TILE_M * TILE_N];
    store_matrix_sync(dot_val, dot_frag, TILE_M, mem_row_major);

    for (int i = 0; i < TILE_M && (block_row + i) < n; ++i)
    {
        for (int j = 0; j < TILE_N && (block_col + j) < n; ++j)
        {
            int global_i = block_row + i;
            int global_j = block_col + j;
            float dist_sq = normsq[global_i] + normsq[global_j] - 2.0f * dot_val[i * TILE_M + j];
            dist_sq = fmaxf(dist_sq, 0.0f);
            distances[static_cast<size_t>(global_i) * n + global_j] = sqrtf(dist_sq);
        }
    }
}

cudaError_t launchWmmaDistanceMatrix(const void *d_points, void *d_normsq, void *d_distances, int n,
                                     int dim, cudaStream_t stream = nullptr);

} // namespace nerve::gpu::tensorcore

#ifdef __CUDACC__
namespace nerve::gpu::tensorcore
{

cudaError_t launchWmmaDistanceMatrix(const void *d_points, void *d_normsq, void *d_distances, int n,
                                     int dim, cudaStream_t stream)
{
    if (n <= 0 || dim <= 0)
        return cudaErrorInvalidValue;
    if (d_points == nullptr || d_normsq == nullptr || d_distances == nullptr)
        return cudaErrorInvalidValue;

    cudaDeviceProp prop;
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
        return err;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
        return err;

    if (prop.major < 7)
        return cudaErrorNotSupported;

    cudaGetLastError();

    constexpr int TILE = 16;
    dim3 block(32, 1, 1);
    dim3 grid(static_cast<unsigned>((n + TILE - 1) / TILE),
              static_cast<unsigned>((n + TILE - 1) / TILE));

    pairwiseDistanceWmma<TILE, TILE, TILE><<<grid, block, 0, stream>>>(
        static_cast<const half *>(d_points), static_cast<float *>(d_normsq),
        static_cast<float *>(d_distances), n, dim);

    return cudaGetLastError();
}

} // namespace nerve::gpu::tensorcore
#endif // __CUDACC__
