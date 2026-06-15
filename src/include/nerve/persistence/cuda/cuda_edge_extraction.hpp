
#pragma once
#include "nerve/core.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/cuda/cuda_edge_types.hpp"
#include "nerve/persistence/cuda/cuda_safe_arithmetic.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
namespace nerve::persistence::accelerated
{

struct EdgeExtractionConfig
{
    Size max_edges = 1000000;             /// Maximum number of edges to extract
    bool enable_early_termination = true; /// Enable early termination when max_edges reached
    bool enable_filtering = false;        /// Enable degree/weight filtering
    bool sort_by_weight = false;          /// Sort edges by weight
    double min_edge_weight = 0.0;         /// Minimum edge weight threshold
    Size max_degree = 1000;               /// Maximum degree per vertex
    bool use_shared_memory = true;        /// Use shared memory optimization
    bool enable_streaming = true;         /// Enable streaming for large problems
    Size streaming_threshold = 1000000;   /// Threshold for streaming
    errors::ErrorResult<void> validate() const;
};
struct EdgeExtractionStats
{
    double total_time_ms = 0.0;       /// Total extraction time
    double gpu_time_ms = 0.0;         /// GPU computation time
    double cpu_time_ms = 0.0;         /// CPU processing time
    Size edges_extracted = 0;         /// Number of edges extracted
    Size edges_filtered = 0;          /// Number of edges filtered out
    double edge_density = 0.0;        /// Edge density (edges / possible edges)
    double average_edge_weight = 0.0; /// Average edge weight
    double max_edge_weight = 0.0;     /// Maximum edge weight
    double min_edge_weight = 0.0;     /// Minimum edge weight
    double getExtractionRate() const
    {
        return total_time_ms > 0 ? (static_cast<double>(edges_extracted) * 1000.0) / total_time_ms
                                 : 0.0;
    }
    double getFilteringRate() const
    {
        const Size total_edges = detail::saturatingAdd(edges_extracted, edges_filtered);
        return total_edges > 0
                   ? static_cast<double>(edges_filtered) / static_cast<double>(total_edges)
                   : 0.0;
    }
    double getEfficiencyScore() const
    {
        if (total_time_ms <= 0.0)
        {
            return 1.0;
        }
        double expected_time = estimateExpectedTime();
        return expected_time > 0 ? expected_time / total_time_ms : 1.0;
    }

private:
    double estimateExpectedTime() const
    {
        double base_time = static_cast<double>(edges_extracted) * 0.001; // 1us per edge
        return base_time / (edge_density + 0.001);                       // Adjust for density
    }
};
class CUDAEgdeExtractor
{
public:
    static errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>
    create(const EdgeExtractionConfig &config = {});
    ~CUDAEgdeExtractor();
    errors::ErrorResult<void> extractEdges(const core::BufferView<const double> &distances,
                                           core::BufferView<Edge> edges, Size n_points,
                                           double max_radius);
    errors::ErrorResult<void> extractEdgesStreaming(const core::BufferView<const double> &distances,
                                                    core::BufferView<Edge> edges, Size n_points,
                                                    double max_radius, Size stream_size);
    errors::ErrorResult<void>
    extractEdgesBatch(const std::vector<core::BufferView<const double>> &distances_batch,
                      std::vector<core::BufferView<Edge>> &edges_batch,
                      const std::vector<Size> &n_points_batch, double max_radius);
    const EdgeExtractionStats &getPerformanceStats() const;
    const EdgeExtractionConfig &getConfig() const;
    Size estimateEdgeCount(Size n_points, double max_radius, double density_factor = 0.1);
    errors::ErrorResult<void> validateParameters(Size n_points, double max_radius) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
namespace cuda_kernels
{
void extractEdgesKernel(const double *__restrict__ distances, Edge *__restrict__ edges,
                        Size *__restrict__ edge_count, Size n_points, double max_radius,
                        Size max_edges);
void extractEdgesEarlyTerminationKernel(const double *__restrict__ distances,
                                        Edge *__restrict__ edges, Size *__restrict__ edge_count,
                                        Size n_points, double max_radius, Size max_edges,
                                        bool *__restrict__ early_termination_flag);
void extractEdgesSharedKernel(const double *__restrict__ distances, Edge *__restrict__ edges,
                              Size *__restrict__ edge_count, Size n_points, double max_radius,
                              Size max_edges);
void extractEdgesFilteredKernel(const double *__restrict__ distances, Edge *__restrict__ edges,
                                Size *__restrict__ edge_count, Size n_points, double max_radius,
                                Size max_edges, double min_edge_weight, Size max_degree,
                                Size *__restrict__ degree_count);
void extractEdgesSortedKernel(const double *__restrict__ distances, Edge *__restrict__ edges,
                              Size *__restrict__ edge_count, Size n_points, double max_radius,
                              Size max_edges, bool sort_by_weight);
} // namespace cuda_kernels
namespace cuda_host
{
errors::ErrorResult<void> launchEdgeExtractionKernel(const double *distances, Edge *edges,
                                                     Size n_points, double max_radius,
                                                     const EdgeExtractionConfig &config);
EdgeExtractionConfig getOptimalConfig(Size n_points, double max_radius,
                                      const EdgeExtractionConfig &base_config = {});
errors::ErrorResult<void> validateLaunchParams(Size n_points, double max_radius,
                                               const EdgeExtractionConfig &config);
Size estimateMemoryUsage(Size n_points, Size max_edges, const EdgeExtractionConfig &config);
} // namespace cuda_host
namespace factory
{
inline errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>
createAcceleratedEdgeExtractor(Size n_points = 0, double max_radius = 1.0,
                               double density_factor = 0.1)
{
    if (!std::isfinite(max_radius) || !(max_radius > 0.0) || !std::isfinite(density_factor) ||
        density_factor < 0.0)
    {
        return errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>::error(
            errors::ErrorCode::E51_PH_INPUT, "Invalid edge extractor factory parameters");
    }
    EdgeExtractionConfig config = cuda_host::getOptimalConfig(n_points, max_radius);
    if (density_factor > 0.5)
    {
        config.enable_filtering = true;
        config.max_degree = std::max<Size>(1, detail::saturatingScale(n_points, 0.1));
    }
    return CUDAEgdeExtractor::create(config);
}
inline errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>
createBatchEdgeExtractor(Size batch_size, Size avg_points = 1000, double max_radius = 1.0)
{
    if (batch_size == 0 || avg_points == 0 || !std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>::error(
            errors::ErrorCode::E51_PH_INPUT, "Invalid batch edge extractor parameters");
    }
    EdgeExtractionConfig config = cuda_host::getOptimalConfig(avg_points, max_radius);
    config.enable_streaming = true;
    const Size matrix_entries = detail::saturatingProduct(avg_points, avg_points);
    config.streaming_threshold = detail::saturatingProduct(batch_size, matrix_entries) / 4;
    config.max_edges = detail::saturatingCompleteGraphEdgeCount(avg_points);
    return CUDAEgdeExtractor::create(config);
}
inline errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>
createStreamingEdgeExtractor(Size problem_size, Size n_points, double max_radius = 1.0)
{
    if (problem_size == 0 || n_points == 0 || !std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>::error(
            errors::ErrorCode::E51_PH_INPUT, "Invalid streaming edge extractor parameters");
    }
    EdgeExtractionConfig config = cuda_host::getOptimalConfig(n_points, max_radius);
    config.enable_streaming = true;
    config.streaming_threshold = std::max<Size>(1, problem_size / 4);
    config.enable_early_termination = true;
    config.max_edges = detail::saturatingCompleteGraphEdgeCount(n_points);
    return CUDAEgdeExtractor::create(config);
}
inline errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>
createHighDensityEdgeExtractor(Size n_points, double max_radius = 1.0, double density_factor = 0.5)
{
    if (n_points == 0 || !std::isfinite(max_radius) || !(max_radius > 0.0) ||
        !std::isfinite(density_factor) || density_factor < 0.0)
    {
        return errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>::error(
            errors::ErrorCode::E51_PH_INPUT, "Invalid high-density edge extractor parameters");
    }
    EdgeExtractionConfig config;
    config.enable_filtering = true;
    config.sort_by_weight = true;
    config.min_edge_weight = max_radius * 0.01;
    config.max_degree = std::max<Size>(1, detail::saturatingScale(n_points, 0.2));
    const Size max_possible = detail::saturatingCompleteGraphEdgeCount(n_points);
    config.max_edges = detail::saturatingScale(max_possible, std::min(density_factor, 1.0));
    return CUDAEgdeExtractor::create(config);
}
inline errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>
createSparseEdgeExtractor(Size n_points, double max_radius = 1.0, double density_factor = 0.05)
{
    if (n_points == 0 || !std::isfinite(max_radius) || !(max_radius > 0.0) ||
        !std::isfinite(density_factor) || density_factor < 0.0)
    {
        return errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>::error(
            errors::ErrorCode::E51_PH_INPUT, "Invalid sparse edge extractor parameters");
    }
    EdgeExtractionConfig config = cuda_host::getOptimalConfig(n_points, max_radius);
    config.enable_early_termination = true;
    config.enable_filtering =
        (max_radius > 0.0 && max_radius < 0.25) || density_factor > 0.3 || n_points > 10000;
    const Size max_possible = detail::saturatingCompleteGraphEdgeCount(n_points);
    const double radius_density = max_radius > 0.0 ? max_radius / (1.0 + max_radius) : 0.0;
    const double effective_density = std::clamp(density_factor * (0.5 + radius_density), 0.0, 1.0);
    config.max_edges = detail::saturatingScale(max_possible, effective_density);
    return CUDAEgdeExtractor::create(config);
}
} // namespace factory
namespace utils
{
inline Size estimateEdgeExtractionMemoryUsage(Size n_points, Size max_edges,
                                              const EdgeExtractionConfig &config)
{
    const Size distance_entries = detail::saturatingProduct(n_points, n_points);
    Size total = detail::saturatingProduct(distance_entries, sizeof(double));
    total = detail::saturatingAdd(total, detail::saturatingProduct(max_edges, sizeof(Edge)));
    Size auxiliary_memory = 0;
    if (config.enable_filtering)
    {
        auxiliary_memory = detail::saturatingAdd(auxiliary_memory,
                                                 detail::saturatingProduct(n_points, sizeof(Size)));
    }
    if (config.enable_early_termination)
    {
        auxiliary_memory = detail::saturatingAdd(auxiliary_memory, sizeof(bool));
    }
    return detail::saturatingAdd(total, auxiliary_memory);
}
inline double estimateEdgeExtractionTime(Size n_points, Size max_edges, bool enable_gpu = true)
{
    double base_time = std::pow(static_cast<double>(n_points), 2.0); // O(n^2) distance matrix
    base_time += static_cast<double>(max_edges) * 0.001;             // Edge extraction overhead
    if (enable_gpu && n_points >= 1000)
    {
        const double acceleration =
            std::clamp(std::sqrt(static_cast<double>(n_points) / 256.0), 1.0, 64.0);
        base_time /= acceleration;
    }
    return base_time;
}
inline errors::ErrorResult<void> validateEdgeExtractionParams(Size n_points, double max_radius,
                                                              const EdgeExtractionConfig &config)
{
    if (n_points == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT,
                                                "Number of points must be greater than 0");
    }
    if (!std::isfinite(max_radius) || max_radius <= 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Maximum radius must be greater than 0");
    }
    if (n_points > 1000000)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT,
                                                "Number of points too large (>1M)");
    }
    if (config.max_edges == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG,
                                                "Maximum edges must be greater than 0");
    }
    Size max_possible_edges = 0;
    if (!detail::checkedCompleteGraphEdgeCount(n_points, max_possible_edges))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Possible edge count overflows size_t");
    }
    if (config.max_edges > max_possible_edges)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG,
                                                "Maximum edges exceeds possible edges");
    }
    if (!std::isfinite(config.min_edge_weight) || config.min_edge_weight < 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG,
                                                "Minimum edge weight must be non-negative");
    }
    if (config.max_degree == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG,
                                                "Maximum degree must be greater than 0");
    }
    return errors::ErrorResult<void>::ok();
}
inline bool shouldUseStreaming(Size n_points, Size max_edges, Size available_gpu_memory)
{
    Size memory_needed =
        estimateEdgeExtractionMemoryUsage(n_points, max_edges, EdgeExtractionConfig{});
    return memory_needed > available_gpu_memory / 2; // Use streaming if > 50% of GPU memory
}
inline Size getOptimalMaxEdges(Size n_points, double max_radius, double density_factor = 0.1)
{
    const Size max_possible = detail::saturatingCompleteGraphEdgeCount(n_points);
    if (!std::isfinite(max_radius) || !std::isfinite(density_factor) || max_radius <= 0.0 ||
        density_factor <= 0.0)
    {
        return 0;
    }
    const double radius_density = max_radius / (1.0 + max_radius);
    const double effective_density = std::clamp(density_factor * (0.5 + radius_density), 0.0, 1.0);
    const Size estimated = detail::saturatingScale(max_possible, effective_density);
    const Size max_reasonable = std::min(Size(1000000), max_possible);
    return std::min(estimated, max_reasonable);
}
inline bool shouldEnableFiltering(Size n_points, double max_radius, double density_factor = 0.1)
{
    const bool radius_selective = max_radius > 0.0 && max_radius < 0.25;
    return radius_selective || density_factor > 0.3 || n_points > 10000;
}
inline bool shouldEnableSorting(Size max_edges, bool enable_gpu = true)
{
    return enable_gpu && max_edges > 10000 && max_edges < 1000000;
}
inline bool shouldUseSharedMemory(Size n_points, Size max_edges)
{
    const Size total_elements = detail::saturatingProduct(n_points, n_points);
    return total_elements <= 65536 && max_edges <= 10000;
}
inline bool shouldEnableEarlyTermination(Size max_edges, bool enable_gpu = true)
{
    return max_edges > 100000 || (enable_gpu && max_edges > 10000);
}
} // namespace utils
} // namespace nerve::persistence::accelerated
