#include "nerve/gpu/device_array.hpp"
#include "nerve/gpu/distance_tedjoin.cuh"

#include <cuda_runtime.h>
#include <mma.h>

#include <stdexcept>

namespace nerve::gpu::tedjoin
{

namespace
{

constexpr int WMMA_M = 8;
constexpr int WMMA_K = 4;
constexpr int BLOCK_DIM = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_DIM / WARP_SIZE;

__global__ void __launch_bounds__(256)
    padPointsKernel(const double *__restrict__ src, double *__restrict__ dst, int n_points, int dim,
                    int dim_padded)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points)
        return;
    const double *row_src = src + static_cast<size_t>(idx) * dim;
    double *row_dst = dst + static_cast<size_t>(idx) * dim_padded;
    for (int d = 0; d < dim; ++d)
    {
        row_dst[d] = row_src[d];
    }
    for (int d = dim; d < dim_padded; ++d)
    {
        row_dst[d] = 0.0;
    }
}

__global__ void __launch_bounds__(256)
    computeNormsKernel(const double *__restrict__ points, int n_points, int dim_padded,
                       double *__restrict__ norms)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points)
        return;
    double sum = 0.0;
    const double *row = points + static_cast<size_t>(idx) * dim_padded;
    for (int d = 0; d < dim_padded; ++d)
    {
        sum += row[d] * row[d];
    }
    norms[idx] = sum;
}

template <int BLOCK_DIM>
__global__ void __launch_bounds__(BLOCK_DIM)
    fp64TensorDistanceKernel(const double *__restrict__ points, int n_points, int dim_padded,
                             const double *__restrict__ norms, double *__restrict__ distances,
                             int dist_stride)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int WMMA_N = 8;
    constexpr int warps_per_block = BLOCK_DIM / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int tile_i = blockIdx.x * warps_per_block + warp_id;
    int tile_j = blockIdx.y;

    int row_start = tile_i * WMMA_M;
    int col_start = tile_j * WMMA_N;

    if (row_start + WMMA_M > n_points || col_start + WMMA_N > n_points)
        return;

    using namespace nvcuda::wmma;
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, double, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, double, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, double> c_frag;

    fill_fragment(c_frag, 0.0);

    for (int k = 0; k < dim_padded; k += WMMA_K)
    {
        load_matrix_sync(a_frag, points + static_cast<size_t>(row_start) * dim_padded + k,
                         dim_padded);
        load_matrix_sync(b_frag, points + static_cast<size_t>(col_start) * dim_padded + k,
                         dim_padded);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __shared__ double tile_dot[warps_per_block][WMMA_M * WMMA_N];
    store_matrix_sync(tile_dot[warp_id], c_frag, WMMA_N, mem_row_major);
    __syncthreads();

    for (int elem = lane_id; elem < warps_per_block * WMMA_M * WMMA_N; elem += WARP_SIZE)
    {
        int w = elem / (WMMA_M * WMMA_N);
        int local_idx = elem % (WMMA_M * WMMA_N);
        int i = local_idx / WMMA_N;
        int j = local_idx % WMMA_N;
        int global_i = (blockIdx.x * warps_per_block + w) * WMMA_M + i;
        int global_j = tile_j * WMMA_N + j;

        if (global_i < n_points && global_j < n_points)
        {
            double dot = tile_dot[w][local_idx];
            double dist_sq = norms[global_i] + norms[global_j] - 2.0 * dot;
            dist_sq = fmax(dist_sq, 0.0);
            distances[static_cast<size_t>(global_i) * dist_stride + global_j] = sqrt(dist_sq);
        }
    }
#endif
}

__global__ void __launch_bounds__(256)
    fringeDistanceKernel(const double *__restrict__ points, int n_points, int dim_padded,
                         const double *__restrict__ norms, double *__restrict__ distances,
                         int dist_stride, int aligned)
{
    int fringe_rows = n_points - aligned;
    if (fringe_rows == 0)
        return;

    int total_bottom = fringe_rows * n_points;
    int total_right = aligned * fringe_rows;
    int total_fringe = total_bottom + total_right;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_fringe)
        return;

    int i, j;
    if (idx < total_bottom)
    {
        i = aligned + idx / n_points;
        j = idx % n_points;
    }
    else
    {
        int r_idx = idx - total_bottom;
        i = r_idx / fringe_rows;
        j = aligned + r_idx % fringe_rows;
    }

    double dot = 0.0;
    const double *row_i = points + static_cast<size_t>(i) * dim_padded;
    const double *row_j = points + static_cast<size_t>(j) * dim_padded;
    for (int d = 0; d < dim_padded; ++d)
    {
        dot += row_i[d] * row_j[d];
    }
    double dist_sq = norms[i] + norms[j] - 2.0 * dot;
    dist_sq = fmax(dist_sq, 0.0);
    distances[static_cast<size_t>(i) * dist_stride + j] = sqrt(dist_sq);
}

} // anonymous namespace

cudaError_t launchFp64TensorDistance(const double *points, int n_points, int dim, double *distances,
                                     int dist_stride, cudaStream_t stream)
{
    if (points == nullptr || distances == nullptr || n_points <= 0 || dim <= 0)
    {
        return cudaErrorInvalidValue;
    }

    cudaDeviceProp prop;
    int device_id;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess)
        return err;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess)
        return err;
    if (prop.major < 8)
    {
        return cudaErrorNotSupported;
    }

    int dim_padded = ((dim + WMMA_K - 1) / WMMA_K) * WMMA_K;

    DeviceArray<double> padded_points;
    DeviceArray<double> norms;
    try
    {
        padded_points = DeviceArray<double>(static_cast<size_t>(n_points) * dim_padded);
        norms = DeviceArray<double>(static_cast<size_t>(n_points));
    }
    catch (const std::runtime_error &)
    {
        return cudaErrorMemoryAllocation;
    }

    int pad_blocks = (n_points + BLOCK_DIM - 1) / BLOCK_DIM;
    padPointsKernel<<<pad_blocks, BLOCK_DIM, 0, stream>>>(points, padded_points.get(), n_points,
                                                          dim, dim_padded);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    int norm_blocks = (n_points + BLOCK_DIM - 1) / BLOCK_DIM;
    computeNormsKernel<<<norm_blocks, BLOCK_DIM, 0, stream>>>(padded_points.get(), n_points,
                                                              dim_padded, norms.get());
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    int aligned = (n_points / WMMA_M) * WMMA_M;
    if (aligned > 0)
    {
        int n_tiles = aligned / WMMA_M;
        int grid_x = (n_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        int grid_y = n_tiles;
        dim3 grid(grid_x, grid_y);

        fp64TensorDistanceKernel<BLOCK_DIM><<<grid, BLOCK_DIM, 0, stream>>>(
            padded_points.get(), n_points, dim_padded, norms.get(), distances, dist_stride);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return err;
    }

    if (aligned < n_points)
    {
        int fringe_rows = n_points - aligned;
        long long total_fringe = static_cast<long long>(fringe_rows) * n_points +
                                 static_cast<long long>(aligned) * fringe_rows;
        int fringe_blocks = static_cast<int>((total_fringe + BLOCK_DIM - 1) / BLOCK_DIM);

        fringeDistanceKernel<<<fringe_blocks, BLOCK_DIM, 0, stream>>>(
            padded_points.get(), n_points, dim_padded, norms.get(), distances, dist_stride,
            aligned);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return err;
    }

    return cudaSuccess;
}

} // namespace nerve::gpu::tedjoin
