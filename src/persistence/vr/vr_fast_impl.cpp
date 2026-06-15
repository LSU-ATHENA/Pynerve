
#include "nerve/config.hpp"
#include "nerve/persistence/accelerated/accelerated_interface.hpp"
#include "nerve/persistence/detail/fast_vr_accelerated_detail.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/runtime/hardware_probe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#if defined(NERVE_HAS_CUDA_RUNTIME) || defined(NERVE_HAS_CUDA)
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif

namespace nerve::persistence::accelerated::detail
{
namespace
{

#if defined(NERVE_HAS_CUDA_RUNTIME) || defined(NERVE_HAS_CUDA)
constexpr std::size_t kKernelTileWidth = 16;
#endif

[[maybe_unused]] std::size_t
countPivotHitsFromAdjacency(const std::vector<std::vector<std::uint8_t>> &adjacency)
{
    if (adjacency.empty())
    {
        return 0;
    }

    std::vector<std::vector<std::uint8_t>> matrix = adjacency;
    const std::size_t rows = matrix.size();
    const std::size_t cols = matrix.front().size();
    std::size_t pivot_row = 0;
    std::size_t pivot_hits = 0;

    for (std::size_t col = 0; col < cols && pivot_row < rows; ++col)
    {
        std::size_t selected = pivot_row;
        while (selected < rows && matrix[selected][col] == 0U)
        {
            ++selected;
        }
        if (selected == rows)
        {
            continue;
        }

        if (selected != pivot_row)
        {
            std::swap(matrix[selected], matrix[pivot_row]);
        }

        for (std::size_t row = 0; row < rows; ++row)
        {
            if (row == pivot_row || matrix[row][col] == 0U)
            {
                continue;
            }
            for (std::size_t k = col; k < cols; ++k)
            {
                matrix[row][k] ^= matrix[pivot_row][k];
            }
        }

        ++pivot_hits;
        ++pivot_row;
    }

    return pivot_hits;
}

} // namespace

double bytesToMb(std::size_t bytes)
{
    constexpr double kBytesPerMb = 1024.0 * 1024.0;
    return static_cast<double>(bytes) / kBytesPerMb;
}

double estimateProblemOps(std::size_t n_points, std::size_t point_dim, std::size_t max_dim)
{
    if (n_points < 2 || point_dim == 0)
    {
        return 0.0;
    }
    const double edges = static_cast<double>(n_points) * static_cast<double>(n_points - 1U) * 0.5;
    const double distance_ops = edges * (2.0 * static_cast<double>(point_dim) + 1.0);
    const double topology_ops = edges * std::max<double>(1.0, static_cast<double>(max_dim));
    return distance_ops + topology_ops;
}

SystemCapabilities detectSystemCapabilitiesImpl()
{
    SystemCapabilities capabilities;
    capabilities.cuda_available = false;
    capabilities.available_memory = 0;
    capabilities.compute_capability = 0.0;
    capabilities.num_multiprocessors = 0;
    capabilities.max_threads_per_block = 0;
    capabilities.shared_memory_per_block = 0;
    capabilities.supported_features = {"cpu_exact"};

    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    if (snapshot.available_memory_bytes.ok())
    {
        capabilities.available_memory = snapshot.available_memory_bytes.value;
    }

    if (runtime::has_cuda_gpu(snapshot))
    {
        const runtime::GpuDeviceInfo &gpu = snapshot.gpus.value.front();
        capabilities.cuda_available = true;
        capabilities.gpu_info.device_id = gpu.device_id;
        capabilities.gpu_info.name = gpu.name;
        capabilities.gpu_info.total_memory = gpu.total_memory_bytes;
        capabilities.gpu_info.available_memory = gpu.free_memory_bytes;
        capabilities.gpu_info.compute_capability =
            gpu.compute_capability_major * 10 + gpu.compute_capability_minor;
        capabilities.compute_capability = static_cast<double>(gpu.compute_capability_major) +
                                          static_cast<double>(gpu.compute_capability_minor) / 10.0;
        capabilities.supported_features.push_back("cuda_runtime");
        capabilities.supported_features.push_back("cuda_hybrid");
    }

    return capabilities;
}

errors::ErrorResult<StagePointsResult>
stagePointsOnGpu([[maybe_unused]] const core::BufferView<const double> &points,
                 [[maybe_unused]] std::size_t point_dim, [[maybe_unused]] double max_radius,
                 [[maybe_unused]] std::size_t chunk_elements)
{
#if defined(NERVE_HAS_CUDA_RUNTIME) || defined(NERVE_HAS_CUDA)
    StagePointsResult transfer;
    transfer.staged_points.resize(points.size());
    if (points.empty())
    {
        return errors::ErrorResult<StagePointsResult>::success(std::move(transfer));
    }
    if (point_dim == 0 || (points.size() % point_dim) != 0)
    {
        return errors::ErrorResult<StagePointsResult>::error(errors::ErrorCode::E51_PH_INPUT);
    }
    if (!std::isfinite(max_radius) || max_radius < 0.0)
    {
        return errors::ErrorResult<StagePointsResult>::error(errors::ErrorCode::E51_PH_INPUT);
    }
    const double max_radius_sq = max_radius * max_radius;
    if (!std::isfinite(max_radius_sq))
    {
        return errors::ErrorResult<StagePointsResult>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }

    const std::size_t total_elements = points.size();
    const std::size_t chunk = std::max<std::size_t>(
        kMinGpuChunkElements,
        chunk_elements == 0 ? total_elements : std::min(chunk_elements, total_elements));

    double *device_chunk = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&device_chunk), chunk * sizeof(double)) != cudaSuccess)
    {
        return errors::ErrorResult<StagePointsResult>::error(errors::ErrorCode::E10_GPU_OOM);
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t offset = 0; offset < total_elements; offset += chunk)
    {
        const std::size_t count = std::min(chunk, total_elements - offset);
        const std::size_t bytes = count * sizeof(double);

        if (cudaMemcpy(device_chunk, points.data() + offset, bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess)
        {
            cudaFree(device_chunk);
            return errors::ErrorResult<StagePointsResult>::error(
                errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
        if (cudaMemcpy(transfer.staged_points.data() + offset, device_chunk, bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cudaFree(device_chunk);
            return errors::ErrorResult<StagePointsResult>::error(
                errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
        transfer.bytes_moved += 2 * bytes;
    }

    const std::size_t n_points = points.size() / point_dim;
    const std::size_t sample_points = std::min<std::size_t>(n_points, 64);
    if (sample_points >= 2)
    {
        const std::size_t sample_elements = sample_points * point_dim;
        for (std::size_t idx = 0; idx < sample_elements; ++idx)
        {
            if (!std::isfinite(points[idx]))
            {
                ++transfer.kernel_diagnostics.invalid_distance_inputs;
            }
        }

        double *device_sample = nullptr;
        if (cudaMalloc(reinterpret_cast<void **>(&device_sample),
                       sample_elements * sizeof(double)) != cudaSuccess)
        {
            cudaFree(device_chunk);
            return errors::ErrorResult<StagePointsResult>::error(errors::ErrorCode::E10_GPU_OOM);
        }
        if (cudaMemcpy(device_sample, points.data(), sample_elements * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cudaFree(device_sample);
            cudaFree(device_chunk);
            return errors::ErrorResult<StagePointsResult>::error(
                errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        cublasHandle_t handle = nullptr;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
        {
            cudaFree(device_sample);
            cudaFree(device_chunk);
            return errors::ErrorResult<StagePointsResult>::error(
                errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        std::vector<double> normsSq(sample_points, 0.0);
        for (std::size_t i = 0; i < sample_points; ++i)
        {
            double l2_norm = 0.0;
            if (cublasDnrm2(handle, static_cast<int>(point_dim), device_sample + i * point_dim, 1,
                            &l2_norm) != CUBLAS_STATUS_SUCCESS)
            {
                cublasDestroy(handle);
                cudaFree(device_sample);
                cudaFree(device_chunk);
                return errors::ErrorResult<StagePointsResult>::error(
                    errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
            }
            normsSq[i] = l2_norm * l2_norm;
        }

        std::size_t edge_count = 0;
        std::vector<std::vector<std::uint8_t>> adjacency(
            sample_points, std::vector<std::uint8_t>(sample_points, 0U));
        const std::size_t tiles_per_pair =
            std::max<std::size_t>(1, (point_dim + (kKernelTileWidth - 1)) / kKernelTileWidth);
        for (std::size_t i = 0; i < sample_points; ++i)
        {
            for (std::size_t j = i + 1; j < sample_points; ++j)
            {
                transfer.kernel_diagnostics.dimension_tiles_processed += tiles_per_pair;
                double dot = 0.0;
                if (cublasDdot(handle, static_cast<int>(point_dim), device_sample + i * point_dim,
                               1, device_sample + j * point_dim, 1, &dot) != CUBLAS_STATUS_SUCCESS)
                {
                    cublasDestroy(handle);
                    cudaFree(device_sample);
                    cudaFree(device_chunk);
                    return errors::ErrorResult<StagePointsResult>::error(
                        errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
                }
                if (!std::isfinite(normsSq[i]) || !std::isfinite(normsSq[j]) || !std::isfinite(dot))
                {
                    ++transfer.kernel_diagnostics.invalid_distance_inputs;
                    ++transfer.kernel_diagnostics.dropped_invalid_distances;
                    continue;
                }

                const double raw_dist_sq = normsSq[i] + normsSq[j] - 2.0 * dot;
                if (!std::isfinite(raw_dist_sq))
                {
                    ++transfer.kernel_diagnostics.dropped_invalid_distances;
                    continue;
                }

                double dist_sq = raw_dist_sq;
                if (dist_sq < 0.0)
                {
                    ++transfer.kernel_diagnostics.dropped_invalid_distances;
                    dist_sq = 0.0;
                }
                if (dist_sq <= max_radius_sq)
                {
                    ++edge_count;
                    adjacency[i][j] = 1U;
                    adjacency[j][i] = 1U;
                }
            }
        }
        transfer.kernel_diagnostics.pivot_hits = countPivotHitsFromAdjacency(adjacency);

        cublasDestroy(handle);
        cudaFree(device_sample);
        const std::size_t total_pairs = (sample_points * (sample_points - 1)) / 2;
        transfer.graph_phase_pairs_evaluated = total_pairs;
        transfer.graph_phase_ops =
            static_cast<double>(total_pairs) * static_cast<double>(point_dim) * 3.0;
        transfer.graph_phase_edge_density =
            total_pairs > 0 ? static_cast<double>(edge_count) / static_cast<double>(total_pairs)
                            : 0.0;
        transfer.graph_phase_sampled_points = sample_points;
        transfer.graph_phase_executed = true;
    }

    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        cudaFree(device_chunk);
        return errors::ErrorResult<StagePointsResult>::error(
            errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    const auto end = std::chrono::steady_clock::now();
    cudaFree(device_chunk);

    transfer.gpu_time_ms = std::max(std::chrono::duration<double, std::milli>(end - start).count(),
                                    std::numeric_limits<double>::epsilon());
    return errors::ErrorResult<StagePointsResult>::success(std::move(transfer));
#else
    return errors::ErrorResult<StagePointsResult>::error(errors::ErrorCode::E10_GPU_OOM);
#endif
}

} // namespace nerve::persistence::accelerated::detail
