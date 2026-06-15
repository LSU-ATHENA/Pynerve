#include "nerve/algorithms/knn_hnsw.hpp"
#include "nerve/persistence/approximate/approximate_nearest_neighbor.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <queue>
#include <random>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nerve::persistence
{

// HNSW (Hierarchical Navigable Small World) algorithm constants
constexpr size_t SMALL_DATASET_THRESHOLD = 10000;
constexpr size_t MEDIUM_DATASET_THRESHOLD = 100000;

// Recall target constants (accuracy vs speed trade-off)
constexpr double RECALL_HIGH = 0.98;
constexpr double RECALL_BALANCED = 0.95;
constexpr double RECALL_FAST = 0.90;
constexpr double RECALL_DEFAULT = 0.95;

// HNSW construction parameters
constexpr int EF_CONSTRUCTION_HIGH = 200;
constexpr int EF_CONSTRUCTION_BALANCED = 150;
constexpr int EF_CONSTRUCTION_FAST = 100;

// HNSW search parameters
constexpr int EF_SEARCH_HIGH = 100;
constexpr int EF_SEARCH_BALANCED = 50;
constexpr int EF_SEARCH_FAST = 30;

// Speedup estimates
constexpr double SPEEDUP_NONE = 1.0;
constexpr double SPEEDUP_SMALL = 2.0;
constexpr double SPEEDUP_MEDIUM = 10.0;
constexpr double SPEEDUP_LARGE = 50.0;

namespace
{

double finiteBenchmarkSpeedup(double baseline, double accelerated)
{
    if (!std::isfinite(baseline) || baseline < 0.0 || !std::isfinite(accelerated) ||
        accelerated <= 0.0)
    {
        return 1.0;
    }
    const double speedup = baseline / accelerated;
    return std::isfinite(speedup) && speedup >= 0.0 ? speedup : 1.0;
}

} // namespace

ANNResult fastEdgeDetectionANN(const std::vector<double> &points, size_t point_dim,
                               size_t num_points, double max_radius, const ANNConfig &config)
{
    ANNResult result{};
    result.max_radius = max_radius;
    result.recall_target = config.recall_target;

    auto start_total = std::chrono::high_resolution_clock::now();
    const bool shape_overflow = point_dim != 0 && num_points > points.size() / point_dim;
    const size_t required_values = shape_overflow ? 0 : num_points * point_dim;
    if (point_dim == 0 || num_points == 0 || shape_overflow || !std::isfinite(max_radius) ||
        max_radius < 0.0 || num_points > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        !std::all_of(points.begin(),
                     std::next(points.begin(),
                               static_cast<std::vector<double>::difference_type>(required_values)),
                     [](double value) { return std::isfinite(value); }))
    {
        result.total_time_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() - start_total)
                                   .count();
        return result;
    }

    auto start_build = std::chrono::high_resolution_clock::now();

    nerve::algorithms::HNSWIndex<double> index(static_cast<int>(point_dim));
    index.build(std::span<const double>(points.data(), points.size()), num_points);

    auto end_build = std::chrono::high_resolution_clock::now();
    result.index_build_time_ms =
        std::chrono::duration<double, std::milli>(end_build - start_build).count();

    auto start_search = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<int, int>> edges;
    std::vector<double> edge_weights;

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points; ++i)
    {
        auto neighbors =
            index.searchRadius(std::span<const double>(&points[i * point_dim], point_dim),
                               static_cast<double>(max_radius), config.ef_search);

#pragma omp critical
        {
            for (const auto &[j, dist] : neighbors)
            {
                if (j > i)
                {
                    edges.push_back({static_cast<int>(i), static_cast<int>(j)});
                    edge_weights.push_back(dist);
                }
            }
        }
    }

    auto end_search = std::chrono::high_resolution_clock::now();
    result.search_time_ms =
        std::chrono::duration<double, std::milli>(end_search - start_search).count();

    result.edges = std::move(edges);
    result.edge_weights = std::move(edge_weights);
    result.num_edges = result.edges.size();

    // Estimate speedup
    double brute_force_time = static_cast<double>(num_points) * static_cast<double>(num_points) *
                              static_cast<double>(point_dim) * 1e-9; // Rough estimate
    result.estimated_speedup = finiteBenchmarkSpeedup(brute_force_time, result.index_build_time_ms +
                                                                            result.search_time_ms);

    auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();

    return result;
}

ANNConfig getOptimalANNConfig(size_t num_points, size_t point_dim)
{
    ANNConfig config;
    if (num_points < SMALL_DATASET_THRESHOLD)
    {
        config.recall_target = RECALL_HIGH; // High accuracy for small datasets
        config.ef_construction = EF_CONSTRUCTION_HIGH;
        config.ef_search = EF_SEARCH_HIGH;
    }
    else if (num_points < MEDIUM_DATASET_THRESHOLD)
    {
        config.recall_target = RECALL_BALANCED; // Good balance
        config.ef_construction = EF_CONSTRUCTION_BALANCED;
        config.ef_search = EF_SEARCH_BALANCED;
    }
    else
    {
        config.recall_target = RECALL_FAST; // Fast for large datasets
        config.ef_construction = EF_CONSTRUCTION_FAST;
        config.ef_search = EF_SEARCH_FAST;
    }
    if (point_dim >= 32)
    {
        config.ef_construction = static_cast<int>(config.ef_construction * 1.25);
        config.ef_search = static_cast<int>(config.ef_search * 1.25);
    }
    return config;
}

ANNSpeedupEstimate estimateANNSpeedup(size_t num_points, size_t point_dim)
{
    ANNSpeedupEstimate estimate{};
    if (num_points < 2 || point_dim == 0)
    {
        estimate.estimated_recall = RECALL_DEFAULT;
        return estimate;
    }

    double brute_force_ops = static_cast<double>(num_points) * static_cast<double>(num_points) *
                             static_cast<double>(point_dim);

    double ann_build_ops = static_cast<double>(num_points) *
                           std::log2(static_cast<double>(num_points)) *
                           static_cast<double>(point_dim);
    double ann_search_ops =
        static_cast<double>(num_points) * std::log2(static_cast<double>(num_points));

    estimate.theoretical_speedup =
        finiteBenchmarkSpeedup(brute_force_ops, ann_build_ops + ann_search_ops);

    // Empirical estimates based on benchmarks
    if (num_points < 1000)
    {
        estimate.expected_speedup = SPEEDUP_NONE; // Not worth overhead
        estimate.recommended = false;
    }
    else if (num_points < SMALL_DATASET_THRESHOLD)
    {
        estimate.expected_speedup = SPEEDUP_SMALL;
        estimate.recommended = true;
    }
    else if (num_points < MEDIUM_DATASET_THRESHOLD)
    {
        estimate.expected_speedup = SPEEDUP_MEDIUM;
        estimate.recommended = true;
    }
    else
    {
        estimate.expected_speedup = SPEEDUP_LARGE;
        estimate.recommended = true;
    }

    estimate.estimated_recall = RECALL_DEFAULT;

    return estimate;
}

} // namespace nerve::persistence
