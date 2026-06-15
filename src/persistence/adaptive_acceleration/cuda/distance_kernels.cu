
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_system_capabilities.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace nerve::persistence::adaptive_acceleration::gpu
{

template <int ComputeCapability>
struct EnhancedKernelParams
{
    static constexpr int TILE_SIZE = 16;
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 32;
    static constexpr int SHARED_MEM_SIZE = 48 * 1024; // 48 KB per SM

    static constexpr bool USE_TENSOR_CORES = (ComputeCapability >= 70);
    static constexpr bool USE_WMMA = (ComputeCapability >= 70);
    static constexpr bool USE_MMA = (ComputeCapability >= 80);
    static constexpr bool USE_FP64_TENSOR_CORES = (ComputeCapability >= 80);

    static constexpr bool USE_FP16 = (ComputeCapability >= 70);
    static constexpr bool USE_BF16 = (ComputeCapability >= 86);
    static constexpr bool USE_TF32 = (ComputeCapability >= 80);

    static constexpr bool ENABLE_MIXED_PRECISION = true;
    static constexpr bool ENABLE_SHARED_MEM_OPTIMIZATION = true;
    static constexpr bool ENABLE_COALESCED_ACCESS = true;
    static constexpr bool ENABLE_PREFETCHING = (ComputeCapability >= 80);
};

// Compile-time compute capability exposed as a device function.
__device__ __forceinline__ int getComputeCapability()
{
#if defined(__CUDA_ARCH__)
    return __CUDA_ARCH__ / 10;
#else
    return 0;
#endif
}

__device__ __forceinline__ bool canUseTensorCores()
{
    return getComputeCapability() >= 70;
}

template <typename T>
__device__ __forceinline__ T deviceInfinity();

template <>
__device__ __forceinline__ float deviceInfinity<float>()
{
    return __int_as_float(0x7f800000);
}

template <>
__device__ __forceinline__ double deviceInfinity<double>()
{
    return __longlong_as_double(0x7ff0000000000000ULL);
}

template <typename T>
__device__ __forceinline__ bool accumulateKahanSquaredDiff(T diff, T &dist_sq, T &compensation)
{
    const T contribution = diff * diff;
    const T y = contribution - compensation;
    const T next = dist_sq + y;
    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(y) || !isfinite(next))
    {
        return false;
    }
    compensation = (next - dist_sq) - y;
    dist_sq = next;
    return true;
}

template <typename T>
__device__ __forceinline__ bool accumulateSquaredDiff(T diff, T &dist_sq)
{
    const T contribution = diff * diff;
    const T next = dist_sq + contribution;
    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next))
    {
        return false;
    }
    dist_sq = next;
    return true;
}

template <typename T>
__device__ __forceinline__ T distanceFromSquared(T dist_sq, bool valid_distance)
{
    if (!valid_distance || !isfinite(dist_sq))
    {
        return deviceInfinity<T>();
    }
    return (dist_sq > static_cast<T>(0)) ? static_cast<T>(sqrt(static_cast<double>(dist_sq)))
                                         : static_cast<T>(0);
}

template <typename T>
__device__ __forceinline__ T computeDistanceSquaredSimd(const T *__restrict__ point_a,
                                                        const T *__restrict__ point_b, int dim)
{
    T dist_sq = static_cast<T>(0);
    T compensation = static_cast<T>(0);

    for (int i = 0; i < dim; ++i)
    {
        const T diff = point_a[i] - point_b[i];
        if (!accumulateKahanSquaredDiff(diff, dist_sq, compensation))
        {
            return deviceInfinity<T>();
        }
    }
    return dist_sq;
}

template <typename T>
__global__ __launch_bounds__(256) void distanceMatrixKernelEnhanced(const T *__restrict__ points,
                                                                    T *__restrict__ distances,
                                                                    size_t num_points,
                                                                    size_t point_dim,
                                                                    const T max_radius)
{
    const size_t global_pos = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    const size_t total = num_points * num_points;
    if (global_pos >= total)
        return;

    const size_t row = global_pos / num_points;
    const size_t col = global_pos % num_points;

    // Compute only upper triangle; mirror below.
    if (row > col)
        return;

    // Diagonal is zero distance.
    if (row == col)
    {
        distances[global_pos] = static_cast<T>(0);
        return;
    }

    // Compute squared distance directly from global memory (no shared-memory
    // race). Kahan summation for numerical stability.
    const T *pa = points + row * point_dim;
    const T *pb = points + col * point_dim;

    T dist_sq = static_cast<T>(0);
    T compensation = static_cast<T>(0);
    bool valid_distance = true;

    for (int k = 0; k < (int)point_dim; ++k)
    {
        const T diff = pa[k] - pb[k];
        if (!accumulateKahanSquaredDiff(diff, dist_sq, compensation))
        {
            valid_distance = false;
            break;
        }
    }

    T dist = distanceFromSquared(dist_sq, valid_distance);

    distances[row * num_points + col] = dist;
    distances[col * num_points + row] = dist; // symmetric
}

template <typename T>
__global__ __launch_bounds__(256) void distanceMatrixKernelTensorCoreEnhanced(
    const T *__restrict__ points, T *__restrict__ distances, size_t num_points, size_t point_dim,
    const T max_radius)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    return;
#else
    const size_t global_pos = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    const size_t total = num_points * num_points;
    if (global_pos >= total)
        return;

    const size_t row = global_pos / num_points;
    const size_t col = global_pos % num_points;

    if (row > col)
        return;

    if (row == col)
    {
        distances[global_pos] = static_cast<T>(0);
        return;
    }

    const T *pa = points + row * point_dim;
    const T *pb = points + col * point_dim;

    T dist_sq = static_cast<T>(0);
    bool valid_distance = true;
    for (int k = 0; k < (int)point_dim; ++k)
    {
        const T diff = pa[k] - pb[k];
        if (!accumulateSquaredDiff(diff, dist_sq))
        {
            valid_distance = false;
            break;
        }
    }

    T dist = distanceFromSquared(dist_sq, valid_distance);

    distances[row * num_points + col] = dist;
    distances[col * num_points + row] = dist;
#endif
}

__global__ __launch_bounds__(256) void distanceMatrixKernelMixedPrecisionEnhanced(
    const double *__restrict__ points, double *__restrict__ distances, size_t num_points,
    size_t point_dim, const double max_radius)
{
    const size_t global_pos = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    const size_t total = num_points * num_points;
    if (global_pos >= total)
        return;

    const size_t row = global_pos / num_points;
    const size_t col = global_pos % num_points;

    if (row > col)
        return;

    if (row == col)
    {
        distances[global_pos] = 0.0;
        return;
    }

    const double *pa = points + row * point_dim;
    const double *pb = points + col * point_dim;

    // Accumulate in FP32 for speed, store in FP64.
    float dist_sq_fp32 = 0.0f;
    float compensation = 0.0f;
    bool valid_distance = true;

    for (int k = 0; k < (int)point_dim; ++k)
    {
        float a = static_cast<float>(pa[k]);
        float b = static_cast<float>(pb[k]);
        float diff = a - b;
        if (!accumulateKahanSquaredDiff(diff, dist_sq_fp32, compensation))
        {
            valid_distance = false;
            break;
        }
    }

    double dist = valid_distance ? static_cast<double>(distanceFromSquared(dist_sq_fp32, true))
                                 : deviceInfinity<double>();

    distances[row * num_points + col] = dist;
    distances[col * num_points + row] = dist;
}

template <typename T>
cudaError_t launchDistanceMatrixKernelEnhanced(const T *points, T *distances, size_t num_points,
                                               size_t point_dim, T max_radius,
                                               cudaStream_t stream = nullptr)
{
    if (num_points == 0)
        return cudaSuccess;
    if (points == nullptr || distances == nullptr || point_dim == 0 ||
        !std::isfinite(static_cast<double>(max_radius)) || max_radius < T(0))
    {
        return cudaErrorInvalidValue;
    }
    if (num_points > std::numeric_limits<size_t>::max() / num_points)
    {
        return cudaErrorInvalidConfiguration;
    }

    const size_t total_pairs = num_points * num_points;
    const int block_size = 256;

    size_t grid_size_raw = (total_pairs + block_size - 1) / block_size;
    if (grid_size_raw > static_cast<size_t>(2147483647u))
    {
        // Cannot launch  --  too many blocks. Caller should tile.
        return cudaErrorInvalidConfiguration;
    }
    unsigned int grid_size = static_cast<unsigned int>(grid_size_raw);

    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
        return err;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
        return err;

    const int compute_cap = prop.major * 10 + prop.minor;

    if (compute_cap >= 80)
    {
        // No dynamic shared memory needed after removing the broken pattern.
        distanceMatrixKernelTensorCoreEnhanced<T><<<grid_size, block_size, 0, stream>>>(
            points, distances, num_points, point_dim, max_radius);
    }
    else if (compute_cap >= 70)
    {
        if constexpr (std::is_same_v<T, double>)
        {
            distanceMatrixKernelMixedPrecisionEnhanced<<<grid_size, block_size, 0, stream>>>(
                points, distances, num_points, point_dim, max_radius);
        }
        else
        {
            distanceMatrixKernelEnhanced<T><<<grid_size, block_size, 0, stream>>>(
                points, distances, num_points, point_dim, max_radius);
        }
    }
    else
    {
        distanceMatrixKernelEnhanced<T><<<grid_size, block_size, 0, stream>>>(
            points, distances, num_points, point_dim, max_radius);
    }

    return cudaGetLastError();
}

// Explicit instantiations for float and double
template cudaError_t launchDistanceMatrixKernelEnhanced<float>(const float *, float *, size_t,
                                                               size_t, float, cudaStream_t);
template cudaError_t launchDistanceMatrixKernelEnhanced<double>(const double *, double *, size_t,
                                                                size_t, double, cudaStream_t);

} // namespace nerve::persistence::adaptive_acceleration::gpu
