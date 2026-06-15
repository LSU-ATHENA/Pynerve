
#include "nerve/persistence/cuda/cuda_multi_gpu.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nerve
{
namespace gpu
{
namespace multi
{

// Multi-GPU Kernel Constants
constexpr int MULTI_GPU_BLOCK_SIZE = 256; // Threads per block for multi-GPU kernels

// GPU topology and peer access detection
struct MultiGPUContext
{
    int num_gpus = 0;
    std::vector<int> device_ids;
    std::vector<bool> p2p_accessible; // P2P[i][j] = GPU i can access GPU j
    bool nvlink_available = false;

    void initialize()
    {
        if (cudaGetDeviceCount(&num_gpus) != cudaSuccess)
        {
            num_gpus = 0;
            return;
        }
        nvlink_available = false;
        if (num_gpus <= 0)
        {
            return;
        }

        device_ids.resize(num_gpus);
        for (int i = 0; i < num_gpus; ++i)
        {
            device_ids[i] = i;
        }

        // Detect P2P accessibility
        p2p_accessible.resize(num_gpus * num_gpus);
        for (int i = 0; i < num_gpus; ++i)
        {
            for (int j = 0; j < num_gpus; ++j)
            {
                if (i == j)
                {
                    p2p_accessible[i * num_gpus + j] = true;
                }
                else
                {
                    int can_access = 0;
                    if (cudaDeviceCanAccessPeer(&can_access, i, j) != cudaSuccess)
                    {
                        can_access = 0;
                    }
                    p2p_accessible[i * num_gpus + j] = (can_access == 1);

                    if (can_access)
                    {
                        cudaSetDevice(i);
                        cudaDeviceEnablePeerAccess(j, 0);
                    }
                }
            }
        }

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
        {
            return;
        }
        if (prop.major >= 7)
        {
            // Check if any P2P is available
            for (int i = 0; i < num_gpus; ++i)
            {
                for (int j = i + 1; j < num_gpus; ++j)
                {
                    if (p2p_accessible[i * num_gpus + j])
                    {
                        nvlink_available = true;
                        break;
                    }
                }
            }
        }
    }

    bool canAccessPeer(int from, int to) const
    {
        if (from < 0 || to < 0 || from >= num_gpus || to >= num_gpus)
        {
            return false;
        }
        return p2p_accessible[from * num_gpus + to];
    }
};

// Distributed point cloud across multiple GPUs
class DistributedPointCloud
{
public:
    struct GPUPartition
    {
        int device_id = 0;
        double *d_points = nullptr; // Points on this GPU
        size_t start_idx = 0;       // Global start index
        size_t local_count = 0;     // Number of points on this GPU
        cudaStream_t stream = nullptr;
    };

    std::vector<GPUPartition> partitions;
    size_t total_points = 0;
    size_t point_dim = 0;

    void distribute(const std::vector<double> &points, size_t dim, const MultiGPUContext &context)
    {
        total_points = points.size() / dim;
        point_dim = dim;

        int num_gpus = context.num_gpus;
        partitions.resize(num_gpus);

        // Calculate partition sizes (load balancing)
        size_t base_count = total_points / num_gpus;
        size_t remainder = total_points % num_gpus;

        size_t current_start = 0;

        for (int gpu = 0; gpu < num_gpus; ++gpu)
        {
            partitions[gpu].device_id = gpu;
            partitions[gpu].start_idx = current_start;
            partitions[gpu].local_count = base_count + (gpu < remainder ? 1 : 0);

            // Create stream
            cudaSetDevice(gpu);
            cudaStreamCreate(&partitions[gpu].stream);

            // Allocate memory
            size_t alloc_size = partitions[gpu].local_count * point_dim * sizeof(double);
            cudaMalloc(&partitions[gpu].d_points, alloc_size);

            // Copy data
            cudaMemcpyAsync(partitions[gpu].d_points, points.data() + current_start * point_dim,
                            alloc_size, cudaMemcpyHostToDevice, partitions[gpu].stream);

            current_start += partitions[gpu].local_count;
        }

        // Synchronize all transfers
        for (int gpu = 0; gpu < num_gpus; ++gpu)
        {
            cudaSetDevice(gpu);
            cudaStreamSynchronize(partitions[gpu].stream);
        }
    }

    ~DistributedPointCloud()
    {
        for (auto &part : partitions)
        {
            cudaSetDevice(part.device_id);
            if (part.d_points != nullptr)
            {
                cudaFree(part.d_points);
            }
            if (part.stream != nullptr)
            {
                cudaStreamDestroy(part.stream);
            }
        }
    }
};

__device__ inline bool accumulateSquaredDifference(double diff, double &dist_sq)
{
    const double contribution = diff * diff;
    const double next_dist_sq = dist_sq + contribution;
    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
    {
        dist_sq = INFINITY;
        return false;
    }
    dist_sq = next_dist_sq;
    return true;
}

// Multi-GPU distance matrix computation
__global__ void __launch_bounds__(256)
    multiGpuDistanceKernel(const double *__restrict__ d_points_local, size_t local_count,
                           size_t total_points, size_t point_dim, float *__restrict__ d_distances,
                           double max_radius_sq, size_t global_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= local_count)
        return;

    const double *p_i = &d_points_local[idx * point_dim];

    // Compute distances to all local points
    for (size_t j = idx + 1; j < local_count; ++j)
    {
        const double *p_j = &d_points_local[j * point_dim];

        double dist_sq = 0.0;
        for (size_t d = 0; d < point_dim; ++d)
        {
            double diff = p_i[d] - p_j[d];
            if (!accumulateSquaredDifference(diff, dist_sq))
            {
                break;
            }
        }

        if (isfinite(dist_sq) && dist_sq <= max_radius_sq)
        {
            size_t global_i = global_offset + idx;
            size_t global_j = global_offset + j;
            const auto dist = static_cast<float>(sqrt(dist_sq));
            d_distances[global_i * total_points + global_j] = dist;
            d_distances[global_j * total_points + global_i] = dist;
        }
    }
}

// Launch multi-GPU distance computation
void launchMultiGpuDistanceMatrix(DistributedPointCloud &cloud, const MultiGPUContext &context,
                                  float *d_distances, // Unified memory or host-pinned
                                  double max_radius)
{
    if (d_distances == nullptr || !std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return;
    }
    double max_radius_sq = max_radius * max_radius;
    if (!std::isfinite(max_radius_sq))
    {
        return;
    }

    // Launch kernels on all GPUs
    for (int gpu = 0; gpu < context.num_gpus; ++gpu)
    {
        cudaSetDevice(gpu);

        auto &part = cloud.partitions[gpu];

        int threads = MULTI_GPU_BLOCK_SIZE;
        int blocks = (part.local_count + threads - 1) / threads;

        multiGpuDistanceKernel<<<blocks, threads, 0, part.stream>>>(
            part.d_points, part.local_count, cloud.total_points, cloud.point_dim, d_distances,
            max_radius_sq, part.start_idx);
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            return;
        }
    }

    // Synchronize all GPUs
    for (int gpu = 0; gpu < context.num_gpus; ++gpu)
    {
        cudaSetDevice(gpu);
        cudaStreamSynchronize(cloud.partitions[gpu].stream);
    }
}

// Multi-GPU Flood Complex computation
MultiGpuResult computeMultiGpuFloodComplex(const std::vector<double> &points, size_t point_dim,
                                           const persistence::FloodComplexConfig &config)
{
    MultiGpuResult result{};
    if (point_dim == 0)
    {
        return result;
    }
    for (const double value : points)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("Point coordinates must be finite");
        }
    }
    if (!std::isfinite(config.max_radius) || config.max_radius < 0.0 ||
        !std::isfinite(config.subset_ratio) || config.subset_ratio < 0.0 ||
        config.subset_ratio > 1.0 || !std::isfinite(config.flooding_tolerance) ||
        config.flooding_tolerance < 0.0)
    {
        throw std::invalid_argument("FloodComplexConfig values must be finite and in range");
    }

    MultiGPUContext context;
    context.initialize();

    result.num_gpus_used = context.num_gpus;
    result.nvlink_enabled = context.nvlink_available;
    result.total_points_processed = points.size() / point_dim;

    auto start = std::chrono::high_resolution_clock::now();
    if (context.num_gpus < 2)
    {
        auto single_result = persistence::computeFloodComplex(
            points, point_dim, result.total_points_processed, config);
        result.pairs = std::move(single_result.pairs);
        result.total_time_ms = single_result.total_time_ms;
        return result;
    }

    DistributedPointCloud cloud;
    cloud.distribute(points, point_dim, context);

    std::vector<std::vector<persistence::Pair>> gpu_results(context.num_gpus);

#pragma omp parallel for num_threads(context.num_gpus)
    for (int gpu = 0; gpu < context.num_gpus; ++gpu)
    {
        cudaSetDevice(gpu);

        std::vector<double> local_points;
        local_points.resize(cloud.partitions[gpu].local_count * point_dim);

        cudaMemcpy(local_points.data(), cloud.partitions[gpu].d_points,
                   local_points.size() * sizeof(double), cudaMemcpyDeviceToHost);

        auto flood_result = persistence::computeFloodComplex(
            local_points, point_dim, cloud.partitions[gpu].local_count, config);

        gpu_results[gpu] = flood_result.pairs;
    }

    result.pairs.clear();
    for (const auto &gpu_pairs : gpu_results)
    {
        result.pairs.insert(result.pairs.end(), gpu_pairs.begin(), gpu_pairs.end());
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

// Check if multi-GPU is beneficial
bool shouldUseMultiGpu(size_t num_points, size_t point_dim)
{
    if (num_points == 0 || point_dim == 0)
    {
        return false;
    }

    int num_gpus = 0;
    if (cudaGetDeviceCount(&num_gpus) != cudaSuccess || num_gpus < 2)
    {
        return false;
    }

    if (point_dim > std::numeric_limits<size_t>::max() / num_points)
    {
        return true;
    }
    const size_t elements = num_points * point_dim;
    if (elements > std::numeric_limits<size_t>::max() / sizeof(double))
    {
        return true;
    }

    const size_t data_size = elements * sizeof(double);
    return data_size > DEFAULT_SINGLE_GPU_MEMORY / 2;
}

// Get multi-GPU info
MultiGpuInfo getMultiGpuInfo()
{
    MultiGpuInfo info{};

    if (cudaGetDeviceCount(&info.num_gpus) != cudaSuccess || info.num_gpus <= 0)
    {
        info.num_gpus = 0;
        return info;
    }

    for (int i = 0; i < info.num_gpus; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        GpuInfo gpu;
        gpu.id = i;
        gpu.name = prop.name;
        gpu.memory_mb = prop.totalGlobalMem / BYTES_PER_MB;
        gpu.compute_capability = std::to_string(prop.major) + "." + std::to_string(prop.minor);

        info.gpus.push_back(gpu);
    }

    // Detect NVLink
    MultiGPUContext context;
    context.initialize();
    info.nvlink_available = context.nvlink_available;
    info.p2p_matrix = context.p2p_accessible;

    return info;
}

} // namespace multi
} // namespace gpu
} // namespace nerve

// C-linkage wrappers
extern "C"
{
    int getNumGpus()
    {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess)
        {
            return 0;
        }
        return count;
    }

    int isNvlinkAvailable()
    {
        nerve::gpu::multi::MultiGPUContext context;
        context.initialize();
        return context.nvlink_available ? 1 : 0;
    }
}
