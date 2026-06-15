// Hopper/Blackwell-oriented distance-matrix kernels.
// This translation unit provides a stable runtime API that compiles on all
// CUDA builds and applies architecture-specific hints via runtime checks.

#include "nerve/gpu/gpu_error.hpp"
#include "nerve/persistence/cuda/cuda_blackwell_tma.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <string_view>

namespace nerve
{
namespace gpu
{
namespace blackwell
{

namespace
{

constexpr int kTileSize = 32;
constexpr int kThreads = 256;
constexpr int kPersistentBlocks = 256;

inline bool checkedSizeProduct(size_t lhs, size_t rhs, size_t &out)
{
    if (lhs != 0 && rhs > (std::numeric_limits<size_t>::max() / lhs))
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

inline bool checkedIntProduct(int lhs, int rhs, int &out)
{
    if (lhs < 0 || rhs < 0)
    {
        return false;
    }
    const int max_int = std::numeric_limits<int>::max();
    if (lhs != 0 && rhs > max_int / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

inline bool radiusSquareFloat(double radius, float &out)
{
    const long double radius_sq =
        static_cast<long double>(radius) * static_cast<long double>(radius);
    if (!std::isfinite(radius_sq) ||
        radius_sq > static_cast<long double>(std::numeric_limits<float>::max()))
    {
        return false;
    }
    out = static_cast<float>(radius_sq);
    return std::isfinite(out);
}

inline bool getDeviceProp(cudaDeviceProp &prop, int device = 0)
{
    return cudaGetDeviceProperties(&prop, device) == cudaSuccess;
}

inline bool hasComputeCapabilityAtLeast(int major, int minor = 0)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
    {
        return false;
    }
    for (int device = 0; device < device_count; ++device)
    {
        cudaDeviceProp prop{};
        if (!getDeviceProp(prop, device))
        {
            continue;
        }
        if (prop.major > major || (prop.major == major && prop.minor >= minor))
        {
            return true;
        }
    }
    return false;
}

inline bool isBlackwellName(std::string_view name)
{
    return name.find("Blackwell") != std::string_view::npos ||
           name.find("RTX 50") != std::string_view::npos ||
           name.find("GB20") != std::string_view::npos ||
           name.find("GB10") != std::string_view::npos;
}

__device__ inline bool accumulateSquaredDifference(float diff, float &dist_sq)
{
    const float contribution = diff * diff;
    const float next_dist_sq = dist_sq + contribution;
    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
    {
        dist_sq = INFINITY;
        return false;
    }
    dist_sq = next_dist_sq;
    return true;
}

__global__ void __launch_bounds__(1024)
    distanceMatrixTileKernel(const double *__restrict__ points, float *__restrict__ distances,
                             int n_points, int point_dim, float max_radius_sq)
{
    const int tile_row = blockIdx.y * kTileSize;
    const int tile_col = blockIdx.x * kTileSize;
    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x;
    const int global_row = tile_row + local_row;
    const int global_col = tile_col + local_col;

    extern __shared__ double shared_mem[];
    double *points_a = shared_mem;
    double *points_b = shared_mem + static_cast<size_t>(kTileSize) * point_dim;

    const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;
    const int tile_elems = kTileSize * point_dim;

    for (int idx = linear_tid; idx < tile_elems; idx += block_threads)
    {
        const int point_idx = idx / point_dim;
        const int dim = idx % point_dim;
        const int row_idx = tile_row + point_idx;
        const int col_idx = tile_col + point_idx;

        points_a[idx] = (row_idx < n_points) ? points[row_idx * point_dim + dim] : 0.0;
        points_b[idx] = (col_idx < n_points) ? points[col_idx * point_dim + dim] : 0.0;
    }
    __syncthreads();

    if (global_row >= n_points || global_col >= n_points || global_row > global_col)
    {
        return;
    }
    if (global_row == global_col)
    {
        distances[global_row * n_points + global_col] = 0.0f;
        return;
    }

    const int row_base = local_row * point_dim;
    const int col_base = local_col * point_dim;
    float dist_sq = 0.0f;
    for (int d = 0; d < point_dim; ++d)
    {
        const float diff = static_cast<float>(points_a[row_base + d] - points_b[col_base + d]);
        if (!accumulateSquaredDifference(diff, dist_sq) || dist_sq > max_radius_sq)
        {
            distances[global_row * n_points + global_col] = -1.0f;
            distances[global_col * n_points + global_row] = -1.0f;
            return;
        }
    }

    const float dist = sqrtf(dist_sq);
    distances[global_row * n_points + global_col] = dist;
    distances[global_col * n_points + global_row] = dist;
}

__global__ void __launch_bounds__(256)
    persistentDistanceKernel(const double *__restrict__ points, float *__restrict__ distances,
                             int n_points, int point_dim, float max_radius_sq, int tiles_per_dim,
                             int tile_count, int *work_counter)
{
    while (true)
    {
        const int tile_idx = atomicAdd(work_counter, 1);
        if (tile_idx >= tile_count)
        {
            return;
        }

        const int tile_x = tile_idx % tiles_per_dim;
        const int tile_y = tile_idx / tiles_per_dim;
        for (int linear = threadIdx.x; linear < kTileSize * kTileSize; linear += blockDim.x)
        {
            const int local_row = linear / kTileSize;
            const int local_col = linear % kTileSize;
            const int row = tile_y * kTileSize + local_row;
            const int col = tile_x * kTileSize + local_col;
            if (row >= n_points || col >= n_points || row > col)
            {
                continue;
            }
            if (row == col)
            {
                distances[row * n_points + col] = 0.0f;
                continue;
            }

            float dist_sq = 0.0f;
            for (int d = 0; d < point_dim; ++d)
            {
                const float diff =
                    static_cast<float>(points[row * point_dim + d] - points[col * point_dim + d]);
                if (!accumulateSquaredDifference(diff, dist_sq) || dist_sq > max_radius_sq)
                {
                    break;
                }
            }

            const float out = (dist_sq > max_radius_sq) ? -1.0f : sqrtf(dist_sq);
            distances[row * n_points + col] = out;
            distances[col * n_points + row] = out;
        }
    }
}

} // namespace

void launchBlackwellDistanceMatrix(const double *d_points, float *d_distances, int n_points,
                                   int point_dim, double max_radius, cudaStream_t stream)
{
    if (d_points == nullptr || d_distances == nullptr || n_points <= 0 || point_dim <= 0)
    {
        return;
    }
    if (!std::isfinite(max_radius) || max_radius <= 0.0)
    {
        return;
    }
    float max_radius_sq = 0.0f;
    if (!radiusSquareFloat(max_radius, max_radius_sq))
    {
        return;
    }

    cudaDeviceProp prop{};
    if (!getDeviceProp(prop))
    {
        return;
    }

    const int tiles_per_dim = (n_points + kTileSize - 1) / kTileSize;
    int tile_count = 0;
    if (!checkedIntProduct(tiles_per_dim, tiles_per_dim, tile_count))
    {
        return;
    }
    if (tiles_per_dim > prop.maxGridSize[0] || tiles_per_dim > prop.maxGridSize[1])
    {
        return;
    }

    const dim3 block(kTileSize, kTileSize);
    const dim3 grid(static_cast<unsigned>(tiles_per_dim), static_cast<unsigned>(tiles_per_dim));

    size_t staged_values = 0;
    size_t shared_bytes = 0;
    if (!checkedSizeProduct(static_cast<size_t>(2 * kTileSize), static_cast<size_t>(point_dim),
                            staged_values) ||
        !checkedSizeProduct(staged_values, sizeof(double), shared_bytes))
    {
        return;
    }

    if (shared_bytes > static_cast<size_t>(prop.sharedMemPerBlock))
    {
        launchPersistentDistanceMatrix(d_points, d_distances, n_points, point_dim, max_radius,
                                       stream);
        return;
    }

    distanceMatrixTileKernel<<<grid, block, shared_bytes, stream>>>(d_points, d_distances, n_points,
                                                                    point_dim, max_radius_sq);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchPersistentDistanceMatrix(const double *d_points, float *d_distances, int n_points,
                                    int point_dim, double max_radius, cudaStream_t stream)
{
    if (d_points == nullptr || d_distances == nullptr || n_points <= 0 || point_dim <= 0)
    {
        return;
    }
    if (!std::isfinite(max_radius) || max_radius <= 0.0)
    {
        return;
    }
    float max_radius_sq = 0.0f;
    if (!radiusSquareFloat(max_radius, max_radius_sq))
    {
        return;
    }

    const int tiles_per_dim = (n_points + kTileSize - 1) / kTileSize;
    int tile_count = 0;
    if (!checkedIntProduct(tiles_per_dim, tiles_per_dim, tile_count))
    {
        return;
    }

    int *d_work_counter = nullptr;
    if (cudaMalloc(&d_work_counter, sizeof(int)) != cudaSuccess)
    {
        return;
    }
    if (cudaMemsetAsync(d_work_counter, 0, sizeof(int), stream) != cudaSuccess)
    {
        cudaFree(d_work_counter);
        return;
    }

    persistentDistanceKernel<<<kPersistentBlocks, kThreads, 0, stream>>>(
        d_points, d_distances, n_points, point_dim, max_radius_sq, tiles_per_dim, tile_count,
        d_work_counter);
    GPU_CHECK(cudaPeekAtLastError());

    cudaFree(d_work_counter);
}

bool isBlackwellAvailable()
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
    {
        return false;
    }
    for (int device = 0; device < device_count; ++device)
    {
        cudaDeviceProp prop{};
        if (!getDeviceProp(prop, device))
        {
            continue;
        }
        if (prop.major >= 10 || isBlackwellName(prop.name))
        {
            return true;
        }
    }
    return false;
}

bool isHopperAvailable()
{
    return hasComputeCapabilityAtLeast(9, 0);
}

BlackwellInfo getBlackwellInfo()
{
    BlackwellInfo info{};
    cudaDeviceProp prop{};
    if (!getDeviceProp(prop))
    {
        info.compute_capability_major = 0;
        info.compute_capability_minor = 0;
        info.generation = "No CUDA device";
        info.supports_tma = false;
        info.supports_wgmma = false;
        info.supports_persistent = false;
        info.supports_clusters = false;
        return info;
    }

    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.supports_tma = prop.major >= 9;
    info.supports_wgmma = prop.major >= 9;
    info.supports_persistent = true;
    info.supports_clusters = prop.major >= 9;

    if (prop.major >= 10 || isBlackwellName(prop.name))
    {
        info.generation = "Blackwell";
    }
    else if (prop.major >= 9)
    {
        info.generation = "Hopper";
    }
    else
    {
        info.generation = "Pre-Hopper";
    }
    return info;
}

} // namespace blackwell
} // namespace gpu
} // namespace nerve

extern "C"
{
    int isHopperOrBlackwell()
    {
        return nerve::gpu::blackwell::isHopperAvailable() ? 1 : 0;
    }
}
