
#include "nerve/persistence/cuda/cuda_edge_extraction.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace nerve::persistence::accelerated
{

namespace
{

Size estimateEdgeCountLocal(Size n_points, double max_radius, double density_factor)
{
    return utils::getOptimalMaxEdges(n_points, max_radius, density_factor);
}

} // namespace

errors::ErrorResult<void> EdgeExtractionConfig::validate() const
{
    if (max_edges == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG);
    }
    if (max_degree == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG);
    }
    if (!std::isfinite(min_edge_weight) || min_edge_weight < 0.0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG);
    }
    return errors::ErrorResult<void>::success();
}

class CUDAEgdeExtractor::Impl
{
public:
    explicit Impl(EdgeExtractionConfig config)
        : config_(config)
    {}

    errors::ErrorResult<void> extractEdges(core::BufferView<const double>distances,
                                           core::BufferView<Edge> edges, Size n_points,
                                           double max_radius)
    {
        auto validation = utils::validateEdgeExtractionParams(n_points, max_radius, config_);
        if (validation.isError())
        {
            return validation;
        }

        Size required = 0;
        if (!detail::checkedSizeProduct(n_points, n_points, required))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        if (distances.size() < required)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
        }

        const auto start = std::chrono::steady_clock::now();
        stats_ = EdgeExtractionStats{};
        std::vector<Size> degree(n_points, 0);
        Size edge_count = 0;

        for (Size i = 0; i < n_points; ++i)
        {
            for (Size j = i + 1; j < n_points; ++j)
            {
                const double distance = distances[i * n_points + j];
                if (!std::isfinite(distance))
                {
                    return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN);
                }
                if (distance > max_radius || distance < config_.min_edge_weight)
                {
                    continue;
                }
                if (config_.enable_filtering &&
                    (degree[i] >= config_.max_degree || degree[j] >= config_.max_degree))
                {
                    ++stats_.edges_filtered;
                    continue;
                }
                if (edge_count >= std::min(config_.max_edges, edges.size()))
                {
                    if (config_.enable_early_termination)
                    {
                        break;
                    }
                    continue;
                }

                edges[edge_count++] = Edge(static_cast<int>(i), static_cast<int>(j), distance);
                ++degree[i];
                ++degree[j];
            }
            if (config_.enable_early_termination &&
                edge_count >= std::min(config_.max_edges, edges.size()))
            {
                break;
            }
        }

        if (config_.sort_by_weight && edge_count > 1)
        {
            std::sort(edges.data(), edges.data() + edge_count,
                      [](const Edge &lhs, const Edge &rhs) { return lhs.w < rhs.w; });
        }

        stats_.edges_extracted = edge_count;
        const Size max_possible = detail::saturatingCompleteGraphEdgeCount(n_points);
        stats_.edge_density =
            max_possible == 0 ? 0.0
                              : static_cast<double>(edge_count) / static_cast<double>(max_possible);

        if (edge_count > 0)
        {
            double total = 0.0;
            stats_.min_edge_weight = edges[0].w;
            stats_.max_edge_weight = edges[0].w;
            for (Size index = 0; index < edge_count; ++index)
            {
                const double value = edges[index].w;
                total += value;
                stats_.min_edge_weight = std::min(stats_.min_edge_weight, value);
                stats_.max_edge_weight = std::max(stats_.max_edge_weight, value);
            }
            stats_.average_edge_weight = total / static_cast<double>(edge_count);
        }

        const auto elapsed = std::chrono::steady_clock::now() - start;
        stats_.total_time_ms = std::chrono::duration<double, std::milli>(elapsed).count();
        stats_.cpu_time_ms = stats_.total_time_ms;
        stats_.gpu_time_ms = 0.0;
        return errors::ErrorResult<void>::success();
    }

    errors::ErrorResult<void> extractEdgesStreaming(core::BufferView<const double>distances,
                                                    core::BufferView<Edge> edges, Size n_points,
                                                    double max_radius, Size /*stream_size*/
    )
    {
        return extractEdges(distances, edges, n_points, max_radius);
    }

    errors::ErrorResult<void>
    extractEdgesBatch(const std::vector<core::BufferView<const double>> &distances_batch,
                      std::vector<core::BufferView<Edge>> &edges_batch,
                      const std::vector<Size> &n_points_batch, double max_radius)
    {
        if (distances_batch.size() != edges_batch.size() ||
            distances_batch.size() != n_points_batch.size())
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
        }
        for (Size index = 0; index < distances_batch.size(); ++index)
        {
            auto result = extractEdges(distances_batch[index], edges_batch[index],
                                       n_points_batch[index], max_radius);
            if (result.isError())
            {
                return result;
            }
        }
        return errors::ErrorResult<void>::success();
    }

    Size estimateEdgeCount(Size n_points, double max_radius, double density_factor) const
    {
        return estimateEdgeCountLocal(n_points, max_radius, density_factor);
    }

    errors::ErrorResult<void> validateParameters(Size n_points, double max_radius) const
    {
        return utils::validateEdgeExtractionParams(n_points, max_radius, config_);
    }

    const EdgeExtractionStats &stats() const { return stats_; }

    const EdgeExtractionConfig &config() const { return config_; }

private:
    EdgeExtractionConfig config_;
    EdgeExtractionStats stats_;
};

errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>
CUDAEgdeExtractor::create(const EdgeExtractionConfig &config)
{
    auto validation = config.validate();
    if (validation.isError())
    {
        return errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>::error(
            validation.errorCode());
    }
    auto extractor = std::make_unique<CUDAEgdeExtractor>();
    extractor->impl_ = std::make_unique<Impl>(config);
    return errors::ErrorResult<std::unique_ptr<CUDAEgdeExtractor>>::success(std::move(extractor));
}

CUDAEgdeExtractor::~CUDAEgdeExtractor() = default;

errors::ErrorResult<void>
CUDAEgdeExtractor::extractEdges(core::BufferView<const double>distances,
                                core::BufferView<Edge> edges, Size n_points, double max_radius)
{
    if (!impl_)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
    }
    return impl_->extractEdges(distances, edges, n_points, max_radius);
}

errors::ErrorResult<void>
CUDAEgdeExtractor::extractEdgesStreaming(core::BufferView<const double>distances,
                                         core::BufferView<Edge> edges, Size n_points,
                                         double max_radius, Size stream_size)
{
    if (!impl_)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
    }
    return impl_->extractEdgesStreaming(distances, edges, n_points, max_radius,
                                        stream_size);
}

errors::ErrorResult<void> CUDAEgdeExtractor::extractEdgesBatch(
    const std::vector<core::BufferView<const double>> &distances_batch,
    std::vector<core::BufferView<Edge>> &edges_batch, const std::vector<Size> &n_points_batch,
    double max_radius)
{
    if (!impl_)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
    }
    return impl_->extractEdgesBatch(distances_batch, edges_batch, n_points_batch, max_radius);
}

const EdgeExtractionStats &CUDAEgdeExtractor::getPerformanceStats() const
{
    static const EdgeExtractionStats kEmptyStats{};
    return impl_ ? impl_->stats() : kEmptyStats;
}

const EdgeExtractionConfig &CUDAEgdeExtractor::getConfig() const
{
    static const EdgeExtractionConfig kDefaultConfig{};
    return impl_ ? impl_->config() : kDefaultConfig;
}

Size CUDAEgdeExtractor::estimateEdgeCount(Size n_points, double max_radius, double density_factor)
{
    return impl_ ? impl_->estimateEdgeCount(n_points, max_radius, density_factor)
                 : estimateEdgeCountLocal(n_points, max_radius, density_factor);
}

errors::ErrorResult<void> CUDAEgdeExtractor::validateParameters(Size n_points,
                                                                double max_radius) const
{
    if (!impl_)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
    }
    return impl_->validateParameters(n_points, max_radius);
}

namespace cuda_host
{

errors::ErrorResult<void> launchEdgeExtractionKernel(const double *distances, Edge *edges,
                                                     Size n_points, double max_radius,
                                                     const EdgeExtractionConfig &config)
{
    if (distances == nullptr || edges == nullptr)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    Size required = 0;
    if (!detail::checkedSizeProduct(n_points, n_points, required))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    core::BufferView<const double> distanceView(distances, required);
    core::BufferView<Edge> edgeView(edges, config.max_edges);
    auto extractor_result = CUDAEgdeExtractor::create(config);
    if (extractor_result.isError())
    {
        return errors::ErrorResult<void>::error(extractor_result.errorCode());
    }
    return extractor_result.value()->extractEdges(distanceView, edgeView, n_points,
                                                  max_radius);
}

EdgeExtractionConfig getOptimalConfig(Size n_points, double max_radius,
                                      const EdgeExtractionConfig &base_config)
{
    EdgeExtractionConfig config = base_config;
    config.max_edges = utils::getOptimalMaxEdges(n_points, max_radius, 0.1);
    config.enable_filtering = utils::shouldEnableFiltering(n_points, max_radius, 0.1);
    config.sort_by_weight = utils::shouldEnableSorting(config.max_edges, false);
    config.use_shared_memory = utils::shouldUseSharedMemory(n_points, config.max_edges);
    config.enable_early_termination = utils::shouldEnableEarlyTermination(config.max_edges, false);
    return config;
}

errors::ErrorResult<void> validateLaunchParams(Size n_points, double max_radius,
                                               const EdgeExtractionConfig &config)
{
    return utils::validateEdgeExtractionParams(n_points, max_radius, config);
}

Size estimateMemoryUsage(Size n_points, Size max_edges, const EdgeExtractionConfig &config)
{
    return utils::estimateEdgeExtractionMemoryUsage(n_points, max_edges, config);
}

} // namespace cuda_host

} // namespace nerve::persistence::accelerated
