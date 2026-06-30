#include "nerve/persistence/cuda/cuda_distance_matrix.hpp"
#include "nerve/persistence/cuda/cuda_error_handling.hpp"
#include "nerve/gpu/gpu_ptx_ops.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nerve::persistence::accelerated
{
using namespace nerve::gpu::ptx;

constexpr int DISTANCE_SHARED_TILE = 16;

cudaError_t computeDistanceMatrixOptimized(const double *points, double *distances, Size n_points,
                                           Size point_dim, double max_radius, cudaStream_t stream,
                                           Size shared_memory_budget_bytes);

__device__ inline void writeSymmetricDistance(double *distances, Size n_points, Size row, Size col,
                                              double value)
{
    const Size idx = row * n_points + col;
    distances[idx] = value;
    if (row != col)
    {
        distances[col * n_points + row] = value;
    }
}

__device__ inline bool accumulateSquaredDifference(double diff, double &distance_sq)
{
    const double next_distance_sq = ptx::fma_f64(diff, diff, distance_sq);
    if (!isfinite(diff) || !isfinite(next_distance_sq))
    {
        distance_sq = INFINITY;
        return false;
    }
    distance_sq = next_distance_sq;
    return true;
}

__global__ void __launch_bounds__(256)
    computeDistanceMatrixKernel(const double *__restrict__ points, double *__restrict__ distances,
                                Size n_points, Size point_dim, double max_radius_sq)
{
    const Size total = n_points * n_points;
    const Size start = static_cast<Size>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Size stride = static_cast<Size>(gridDim.x) * blockDim.x;

    for (Size global_idx = start; global_idx < total; global_idx += stride)
    {
        const Size row = global_idx / n_points;
        const Size col = global_idx % n_points;

        if (row >= n_points || col >= n_points || row > col)
        {
            continue;
        }
        if (row == col)
        {
            distances[global_idx] = 0.0;
            continue;
        }
        double distance_sq = 0.0;
        if (point_dim <= 4)
        {
            if (point_dim >= 1)
            {
                double diff = points[row * point_dim + 0] - points[col * point_dim + 0];
                if (!accumulateSquaredDifference(diff, distance_sq) || distance_sq > max_radius_sq)
                {
                    writeSymmetricDistance(distances, n_points, row, col, -1.0);
                    continue;
                }
            }
            if (point_dim >= 2)
            {
                double diff = points[row * point_dim + 1] - points[col * point_dim + 1];
                if (!accumulateSquaredDifference(diff, distance_sq) || distance_sq > max_radius_sq)
                {
                    writeSymmetricDistance(distances, n_points, row, col, -1.0);
                    continue;
                }
            }
            if (point_dim >= 3)
            {
                double diff = points[row * point_dim + 2] - points[col * point_dim + 2];
                if (!accumulateSquaredDifference(diff, distance_sq) || distance_sq > max_radius_sq)
                {
                    writeSymmetricDistance(distances, n_points, row, col, -1.0);
                    continue;
                }
            }
            if (point_dim >= 4)
            {
                double diff = points[row * point_dim + 3] - points[col * point_dim + 3];
                if (!accumulateSquaredDifference(diff, distance_sq) || distance_sq > max_radius_sq)
                {
                    writeSymmetricDistance(distances, n_points, row, col, -1.0);
                    continue;
                }
            }
        }
        else
        {
            for (Size i = 0; i < point_dim; ++i)
            {
                double diff = points[row * point_dim + i] - points[col * point_dim + i];
                if (!accumulateSquaredDifference(diff, distance_sq) || distance_sq > max_radius_sq)
                {
                    writeSymmetricDistance(distances, n_points, row, col, -1.0);
                    break;
                }
            }
        }

        if (distance_sq > max_radius_sq)
        {
            continue;
        }
        writeSymmetricDistance(distances, n_points, row, col, sqrt(distance_sq));
    }
}
__global__ void __launch_bounds__(256)
    computeDistanceMatrixSimdKernel(const double *__restrict__ points,
                                    double *__restrict__ distances, Size n_points, Size point_dim,
                                    double max_radius_sq)
{
    const Size total = n_points * n_points;
    const Size start = static_cast<Size>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Size stride = static_cast<Size>(gridDim.x) * blockDim.x;

    for (Size global_idx = start; global_idx < total; global_idx += stride)
    {
        const Size row = global_idx / n_points;
        const Size col = global_idx % n_points;

        if (row >= n_points || col >= n_points || row > col)
        {
            continue;
        }
        if (row == col)
        {
            distances[global_idx] = 0.0;
            continue;
        }
        const double *row_ptr = points + row * point_dim;
        const double *col_ptr = points + col * point_dim;

        double distance_sq = 0.0;
        const Size simd_chunks = point_dim / 4;
        for (Size chunk = 0; chunk < simd_chunks; ++chunk)
        {
            const Size base_idx = chunk * 4;
            const double r0 = row_ptr[base_idx + 0];
            const double r1 = row_ptr[base_idx + 1];
            const double r2 = row_ptr[base_idx + 2];
            const double r3 = row_ptr[base_idx + 3];

            const double c0 = col_ptr[base_idx + 0];
            const double c1 = col_ptr[base_idx + 1];
            const double c2 = col_ptr[base_idx + 2];
            const double c3 = col_ptr[base_idx + 3];
            const double d0 = r0 - c0;
            const double d1 = r1 - c1;
            const double d2 = r2 - c2;
            const double d3 = r3 - c3;
            const bool chunk_valid = accumulateSquaredDifference(d0, distance_sq) &&
                                     accumulateSquaredDifference(d1, distance_sq) &&
                                     accumulateSquaredDifference(d2, distance_sq) &&
                                     accumulateSquaredDifference(d3, distance_sq);
            if (!chunk_valid || distance_sq > max_radius_sq)
            {
                writeSymmetricDistance(distances, n_points, row, col, -1.0);
                break;
            }
        }
        if (distance_sq > max_radius_sq)
        {
            continue;
        }
        for (Size i = simd_chunks * 4; i < point_dim; ++i)
        {
            const double diff = row_ptr[i] - col_ptr[i];
            if (!accumulateSquaredDifference(diff, distance_sq) || distance_sq > max_radius_sq)
            {
                writeSymmetricDistance(distances, n_points, row, col, -1.0);
                break;
            }
        }
        if (distance_sq > max_radius_sq)
        {
            continue;
        }
        writeSymmetricDistance(distances, n_points, row, col, sqrt(distance_sq));
    }
}
__global__ void __launch_bounds__(256)
    computeDistanceMatrixSharedKernel(const double *__restrict__ points,
                                      double *__restrict__ distances, Size n_points, Size point_dim,
                                      double max_radius_sq)
{
    Size row = blockIdx.y * blockDim.y + threadIdx.y;
    Size col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_points || col >= n_points || row > col)
    {
        return;
    }
    if (row == col)
    {
        distances[row * n_points + col] = 0.0;
        return;
    }

    __shared__ double row_tile[DISTANCE_SHARED_TILE];
    __shared__ double col_tile[DISTANCE_SHARED_TILE];
    double distance_sq = 0.0;

    for (Size base = 0; base < point_dim; base += DISTANCE_SHARED_TILE)
    {
        const Size local_dim = static_cast<Size>(threadIdx.x);
        if (threadIdx.y == 0 && local_dim < DISTANCE_SHARED_TILE)
        {
            const Size dim = base + local_dim;
            row_tile[local_dim] = (dim < point_dim) ? points[row * point_dim + dim] : 0.0;
            col_tile[local_dim] = (dim < point_dim) ? points[col * point_dim + dim] : 0.0;
        }
        __syncthreads();

        const Size remaining = point_dim - base;
        const Size limit = (remaining < static_cast<Size>(DISTANCE_SHARED_TILE))
                               ? remaining
                               : static_cast<Size>(DISTANCE_SHARED_TILE);
#pragma unroll
        for (Size d = 0; d < limit; ++d)
        {
            const double diff = row_tile[d] - col_tile[d];
            if (!accumulateSquaredDifference(diff, distance_sq) || distance_sq > max_radius_sq)
            {
                writeSymmetricDistance(distances, n_points, row, col, -1.0);
                return;
            }
        }
        __syncthreads();
    }
    writeSymmetricDistance(distances, n_points, row, col, sqrt(distance_sq));
}
__global__ void __launch_bounds__(256)
    computeDistanceMatrixStreamingKernel(const double *__restrict__ points,
                                         double *__restrict__ distances, Size n_points,
                                         Size point_dim, double max_radius_sq, Size stream_offset,
                                         Size stream_size)
{
    const Size start = static_cast<Size>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Size stride = static_cast<Size>(gridDim.x) * blockDim.x;

    for (Size global_idx = start; global_idx < stream_size; global_idx += stride)
    {
        const Size absolute_idx = stream_offset + global_idx;
        const Size row = absolute_idx / n_points;
        const Size col = absolute_idx % n_points;

        if (row >= n_points || col >= n_points || row > col)
        {
            continue;
        }
        if (row == col)
        {
            distances[absolute_idx] = 0.0;
            continue;
        }

        double distance_sq = 0.0;
        const double *row_ptr = points + row * point_dim;
        const double *col_ptr = points + col * point_dim;
        for (Size i = 0; i < point_dim; ++i)
        {
            const double diff = row_ptr[i] - col_ptr[i];
            if (!accumulateSquaredDifference(diff, distance_sq) || distance_sq > max_radius_sq)
            {
                writeSymmetricDistance(distances, n_points, row, col, -1.0);
                break;
            }
        }
        if (distance_sq > max_radius_sq)
        {
            continue;
        }
        writeSymmetricDistance(distances, n_points, row, col, sqrt(distance_sq));
    }
}

namespace cuda_host
{

errors::ErrorResult<void> launchDistanceMatrixKernel(const double *points, double *distances,
                                                     Size n_points, Size point_dim,
                                                     double max_radius,
                                                     const CUDADistanceMatrixConfig &config,
                                                     Size stream_offset, Size stream_size)
{
    auto launch_valid = validateLaunchParams(n_points, point_dim, config);
    if (launch_valid.isError())
    {
        return launch_valid;
    }
    if (!points || !distances)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Null pointer arguments");
    }
    if (!std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Invalid radius threshold");
    }

    Size total_elements = 0;
    if (!detail::checkedSizeProduct(n_points, n_points, total_elements))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Distance matrix size overflow");
    }
    if (stream_offset >= total_elements && stream_size > 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Stream offset out of bounds");
    }
    if (!config.enable_streaming && (stream_offset > 0 || stream_size > 0))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG,
                                                "Streaming is off by configuration");
    }

    const double max_radius_sq = max_radius * max_radius;
    if (!std::isfinite(max_radius_sq))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Radius threshold square overflows");
    }

    int device_id = 0;
    cudaError_t result = cudaGetDevice(&device_id);
    if (result != cudaSuccess)
    {
        return cuda_error_handling::check_cuda_operation(result, "cudaGetDevice",
                                                         std::source_location::current());
    }

    cudaDeviceProp prop{};
    result = cudaGetDeviceProperties(&prop, device_id);
    if (result != cudaSuccess)
    {
        return cuda_error_handling::check_cuda_operation(result, "cudaGetDeviceProperties",
                                                         std::source_location::current());
    }

    const bool use_streaming =
        config.enable_streaming && stream_size > 0 && stream_offset < total_elements;
    const Size effective_stream_size =
        use_streaming ? std::min(stream_size, total_elements - stream_offset) : total_elements;
    const bool use_shared_tiled =
        config.enable_shared_memory && !use_streaming && point_dim >= 8 && point_dim <= 4096;
    const bool use_unrolled_chunks = config.enable_simd && prop.major >= 6 && point_dim >= 4;

    if (use_shared_tiled)
    {
        result = computeDistanceMatrixOptimized(points, distances, n_points, point_dim, max_radius,
                                                0, config.shared_memory_size);
        if (result != cudaSuccess)
        {
            return cuda_error_handling::check_cuda_operation(
                result, "computeDistanceMatrixOptimized", std::source_location::current());
        }
    }
    else
    {
        const Size requested_block = std::clamp<Size>(config.max_block_size, 32, 1024);
        const unsigned int block_x = static_cast<unsigned int>(requested_block);
        const Size required_blocks = (effective_stream_size / requested_block) +
                                     ((effective_stream_size % requested_block) == 0 ? 0 : 1);
        const Size capped_blocks =
            std::max<Size>(1, std::min(required_blocks, config.max_grid_size));

        dim3 block_size(block_x, 1, 1);
        dim3 grid_size(static_cast<unsigned int>(capped_blocks), 1, 1);

        if (use_streaming)
        {
            computeDistanceMatrixStreamingKernel<<<grid_size, block_size>>>(
                points, distances, n_points, point_dim, max_radius_sq, stream_offset,
                effective_stream_size);
        }
        else if (use_unrolled_chunks)
        {
            computeDistanceMatrixSimdKernel<<<grid_size, block_size>>>(points, distances, n_points,
                                                                       point_dim, max_radius_sq);
        }
        else
        {
            computeDistanceMatrixKernel<<<grid_size, block_size>>>(points, distances, n_points,
                                                                   point_dim, max_radius_sq);
        }

        result = cudaGetLastError();
        if (result != cudaSuccess)
        {
            return cuda_error_handling::validateKernelLaunch("distance_matrix_kernel",
                                                             std::source_location::current());
        }
    }

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess)
    {
        return cuda_error_handling::check_cuda_operation(result, "cudaDeviceSynchronize",
                                                         std::source_location::current());
    }
    return errors::ErrorResult<void>::ok();
}

errors::ErrorResult<void> launchDistanceMatrixKernel(const double *points, double *distances,
                                                     Size n_points, Size point_dim,
                                                     double max_radius, Size stream_offset,
                                                     Size stream_size)
{
    const CUDADistanceMatrixConfig config = getOptimalConfig(n_points, point_dim);
    return launchDistanceMatrixKernel(points, distances, n_points, point_dim, max_radius, config,
                                      stream_offset, stream_size);
}

} // namespace cuda_host

} // namespace nerve::persistence::accelerated
