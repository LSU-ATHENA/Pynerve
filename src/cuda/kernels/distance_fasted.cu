#include "nerve/gpu/distance_fasted.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cuda/pipeline>

namespace nerve::gpu
{
namespace
{

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__device__ __forceinline__ float half_to_float(half h)
{
    return __half2float(h);
}

__device__ __forceinline__ half float_to_half(float f)
{
    return __float2half(f);
}

template <int WARP_TILE_M, int WARP_TILE_N, int WARP_TILE_K>
__global__ void fastedDistanceKernel(const half *__restrict__ points, int n_points, int dim_padded,
                                     float *__restrict__ distances, int dist_stride,
                                     const int *__restrict__ block_queue, int n_block_tiles)
{
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int block_tile_id = block_queue ? block_queue[blockIdx.x] : blockIdx.x;

    if (block_tile_id >= n_block_tiles)
        return;

    int tile_row = (block_tile_id / ((n_points + WARP_TILE_M * 2 - 1) / (WARP_TILE_M * 2))) *
                   (WARP_TILE_M * 2);
    int tile_col = (block_tile_id % ((n_points + WARP_TILE_M * 2 - 1) / (WARP_TILE_M * 2))) *
                   (WARP_TILE_N * 2);

    __shared__ half shmem_points_a[WARP_TILE_M * 2 * WARP_TILE_K * 2];
    __shared__ half shmem_points_b[WARP_TILE_N * 2 * WARP_TILE_K * 2];

    // Use cuda::pipeline for async copy
    auto pipeline = cuda::make_pipeline();

    // Warp computes its sub-tile within the block tile
    int warp_offset_m = (warp_id & 1) * WARP_TILE_M;
    int warp_offset_n = (warp_id >> 1) * WARP_TILE_N;

    nvcuda::wmma::fragment<nvcuba::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        nvcuda::wmma::fill_fragment(acc[i], 0.0f);
    }

    auto *a_shmem = &shmem_points_a[warp_offset_m * WARP_TILE_K];
    auto *b_shmem = &shmem_points_b[warp_offset_n * WARP_TILE_K];

    for (int k = 0; k < dim_padded; k += WARP_TILE_K)
    {
        // Stage 1: Async copy A tile from global to shared memory
        if (tile_row + warp_offset_m + lane_id < n_points)
        {
            cuda::memcpy_async(&shmem_points_a[(warp_offset_m + lane_id) * WARP_TILE_K],
                               &points[(tile_row + warp_offset_m + lane_id) * dim_padded + k],
                               WARP_TILE_K * sizeof(half), pipeline);
        }
        __syncwarp();

        if (tile_col + warp_offset_n + lane_id < n_points)
        {
            cuda::memcpy_async(&shmem_points_b[(warp_offset_n + lane_id) * WARP_TILE_K],
                               &points[(tile_col + warp_offset_n + lane_id) * dim_padded + k],
                               WARP_TILE_K * sizeof(half), pipeline);
        }
        __syncwarp();

        pipeline.commit();
        __syncwarp();
        pipeline.wait();
        __syncwarp();
        __syncthreads();

        nvcuba::wmma::fragment<nvcuba::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                               nvcuba::wmma::row_major>
            frag_a[4];
        nvcuba::wmma::fragment<nvcuba::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                               nvcuba::wmma::col_major>
            frag_b[4];

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int a_row = (i >> 1) * WMMA_M;
            int a_col = 0;
            nvcuba::wmma::load_matrix_sync(frag_a[i], &shmem_points_a[a_row * WARP_TILE_K + a_col],
                                           WARP_TILE_K);

            int b_row = (i & 1) * WMMA_N;
            int b_col = 0;
            nvcuba::wmma::load_matrix_sync(frag_b[i], &shmem_points_b[b_row * WARP_TILE_K + b_col],
                                           WARP_TILE_K);

            nvcuba::wmma::mma_sync(acc[i], frag_a[i], frag_b[i], acc[i]);
        }
        __syncthreads();
    }

// Combine 4 sub-accumulators for final 64x64 output
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int out_row = tile_row + warp_offset_m + ((i >> 1) * WMMA_M);
        int out_col = tile_col + warp_offset_n + ((i & 1) * WMMA_N);

        float *out_ptr = &distances[out_row * dist_stride + out_col];
        nvcuba::wmma::store_matrix_sync(out_ptr, acc[i], dist_stride, nvcuba::wmma::mem_row_major);
    }
}

__global__ void blockQueueInitKernel(int *queue, int n_tiles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_tiles)
        queue[tid] = tid;
}

} // namespace

cudaError_t launchDistanceFastEd(const float *points, int n_points, int dim, float *distances,
                                 int distance_stride, FastedConfig config)
{
    return launchDistanceFastEdAsync(points, n_points, dim, distances, distance_stride, nullptr,
                                     config);
}

cudaError_t launchDistanceFastEdAsync(const float *points, int n_points, int dim, float *distances,
                                      int distance_stride, cudaStream_t stream, FastedConfig config)
{
    if (!points || n_points <= 0 || dim <= 0 || !distances)
    {
        return cudaErrorInvalidValue;
    }

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess)
        return err;

    int cc = prop.major * 10 + prop.minor;
    if (cc < 80)
    {
        return cudaErrorNotSupported;
    }

    int dim_padded = ((dim + WMMA_K - 1) / WMMA_K) * WMMA_K;
    half *d_points_half = nullptr;
    int *d_block_queue = nullptr;

    err = cudaMalloc(&d_points_half, static_cast<size_t>(n_points) * dim_padded * sizeof(half));
    if (err != cudaSuccess)
        return err;

    // Convert float input to half
    std::vector<half> h_points_half(static_cast<size_t>(n_points) * dim_padded);
    for (int i = 0; i < n_points; ++i)
    {
        for (int d = 0; d < dim; ++d)
        {
            h_points_half[static_cast<size_t>(i) * dim_padded + d] =
                __float2half(points[static_cast<size_t>(i) * dim + d]);
        }
        for (int d = dim; d < dim_padded; ++d)
        {
            h_points_half[static_cast<size_t>(i) * dim_padded + d] = __float2half(0.0f);
        }
    }
    err = cudaMemcpyAsync(d_points_half, h_points_half.data(),
                          static_cast<size_t>(n_points) * dim_padded * sizeof(half),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_points_half);
        return err;
    }

    dim3 block_dim(32, 4);
    int n_tiles_row = (n_points + config.block_tile_m - 1) / config.block_tile_m;
    int n_tiles_col = (n_points + config.block_tile_n - 1) / config.block_tile_n;
    int n_block_tiles = n_tiles_row * n_tiles_col;
    dim3 grid_dim((n_block_tiles + 65535) / 65535, 1);

    if (config.use_l2_optimization)
    {
        err = cudaMalloc(&d_block_queue, n_block_tiles * sizeof(int));
        if (err != cudaSuccess)
        {
            cudaFree(d_points_half);
            return err;
        }
        int tile_threads = 256;
        blockQueueInitKernel<<<(n_block_tiles + tile_threads - 1) / tile_threads, tile_threads, 0,
                               stream>>>(d_block_queue, n_block_tiles);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFree(d_points_half);
            cudaFree(d_block_queue);
            return err;
        }
    }

    fastedDistanceKernel<64, 64, 16>
        <<<grid_dim, block_dim, 0, stream>>>(d_points_half, n_points, dim_padded, distances,
                                             distance_stride, d_block_queue, n_block_tiles);

    err = cudaGetLastError();
    if (d_points_half)
        cudaFree(d_points_half);
    if (d_block_queue)
        cudaFree(d_block_queue);
    return err;
}

} // namespace nerve::gpu
