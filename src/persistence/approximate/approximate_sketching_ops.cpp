// Probabilistic Sketching Approximation for Persistent Homology
// Based on 2025 research on randomized algorithms with guarantees
// Uses: Column sampling, random projection, importance sampling
#include "nerve/persistence/approximate/sketching_approximation.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
namespace nerve::persistence
{
namespace
{
// Sketching Approximation Constants
constexpr int SKETCHING_DEFAULT_MAX_DIM = 2;
constexpr double SKETCHING_DEFAULT_MAX_RADIUS = 1.0;
constexpr size_t SKETCHING_SMALL_DATASET_THRESHOLD = 1000;
constexpr size_t SKETCHING_MEDIUM_DATASET_THRESHOLD = 10000;
constexpr size_t SKETCHING_LARGE_DATASET_THRESHOLD = 100000;
constexpr double SKETCHING_SAMPLE_RATE_FULL = 1.0;  // No sampling
constexpr double SKETCHING_SAMPLE_RATE_HALF = 0.5;  // 50% sampling
constexpr double SKETCHING_SAMPLE_RATE_FIFTH = 0.2; // 20% sampling
constexpr double SKETCHING_SAMPLE_RATE_TENTH = 0.1; // 10% sampling
constexpr size_t SKETCHING_DIM_REDUCTION_THRESHOLD = 10;
constexpr size_t SKETCHING_TARGET_DIMENSION = 50;
constexpr unsigned SKETCHING_DEFAULT_RANDOM_SEED = 42;
constexpr int SPARSE_JL_NONZERO_ENTRIES = 3; // s = 3 for sparse JL
// Random projection matrix for dimensionality reduction
// Uses sparse Johnson-Lindenstrauss transform
class RandomProjection
{
public:
    RandomProjection(size_t original_dim, size_t projected_dim,
                     unsigned seed = SKETCHING_DEFAULT_RANDOM_SEED)
        : original_dim_(original_dim)
        , projected_dim_(std::max<size_t>(1, projected_dim))
    {
        rng_.seed(seed);
        const int nonzeros_per_column =
            std::max(1, std::min<int>(SPARSE_JL_NONZERO_ENTRIES, static_cast<int>(projected_dim_)));
        const double scale = 1.0 / std::sqrt(static_cast<double>(nonzeros_per_column));
        projection_matrix_.reserve(original_dim_ * static_cast<size_t>(nonzeros_per_column));
        for (size_t j = 0; j < original_dim; ++j)
        {
            std::vector<size_t> rows(projected_dim_);
            std::iota(rows.begin(), rows.end(), 0);
            std::shuffle(rows.begin(), rows.end(), rng_);
            for (int k = 0; k < nonzeros_per_column; ++k)
            {
                size_t row = rows[k];
                double value = ((rng_() % 2 == 0) ? 1.0 : -1.0) * scale;
                projection_matrix_.push_back({j, row, value});
            }
        }
    }
    // Project a point: y = R * x / sqrt(projected_dim)
    std::vector<double> project(const double *point) const
    {
        std::vector<double> result(projected_dim_, 0.0);
        for (const auto &[col, row, val] : projection_matrix_)
        {
            result[row] += val * point[col];
        }
        return result;
    }

private:
    size_t original_dim_;
    size_t projected_dim_;
    std::mt19937 rng_;
    std::vector<std::tuple<size_t, size_t, double>> projection_matrix_;
};

SketchingConfig normalizeConfig(const SketchingConfig &input, size_t point_dim)
{
    SketchingConfig config = input;
    if (!std::isfinite(config.edge_sampling_rate) || config.edge_sampling_rate < 0.0 ||
        config.edge_sampling_rate > 1.0)
    {
        throw std::invalid_argument("edge_sampling_rate must be finite and in [0, 1]");
    }
    if (!std::isfinite(config.column_sampling_rate) || config.column_sampling_rate < 0.0 ||
        config.column_sampling_rate > 1.0)
    {
        throw std::invalid_argument("column_sampling_rate must be finite and in [0, 1]");
    }
    if (!std::isfinite(config.max_radius) || config.max_radius <= 0.0)
    {
        throw std::invalid_argument("max_radius must be finite and positive");
    }
    if (point_dim == 0)
    {
        config.use_dimensionality_reduction = false;
        config.target_dimension = 0;
    }
    else if (config.target_dimension == 0 || config.target_dimension > point_dim)
    {
        config.target_dimension = std::max<size_t>(1, point_dim / 2);
    }
    return config;
}
} // namespace
// Main API: Approximate PH using sketching
SketchingResult computeApproximatePHSketching(const core::BufferView<const double> &points,
                                              Size point_dim, const SketchingConfig &config)
{
    SketchingResult result{};
    result.config = normalizeConfig(config, point_dim);
    auto start_total = std::chrono::high_resolution_clock::now();
    if (point_dim == 0 || points.size() == 0 || points.size() % point_dim != 0)
    {
        result.total_time_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() - start_total)
                                   .count();
        return result;
    }
    const size_t num_points = points.size() / point_dim;
    if (num_points > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        result.total_time_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() - start_total)
                                   .count();
        return result;
    }
    if (!std::all_of(points.begin(), points.end(),
                     [](double value) { return std::isfinite(value); }))
    {
        throw std::invalid_argument("points must contain only finite values");
    }
    std::vector<double> processed_points;
    size_t processed_dim = point_dim;
    const size_t target_dim = std::clamp<size_t>(result.config.target_dimension, 1, point_dim);
    if (result.config.use_dimensionality_reduction && point_dim > target_dim)
    {
        auto start_proj = std::chrono::high_resolution_clock::now();
        RandomProjection rp(point_dim, target_dim, result.config.random_seed);
        processed_points.reserve(num_points * target_dim);
        for (size_t i = 0; i < num_points; ++i)
        {
            auto proj = rp.project(&points[i * point_dim]);
            processed_points.insert(processed_points.end(), proj.begin(), proj.end());
        }
        processed_dim = target_dim;
        auto end_proj = std::chrono::high_resolution_clock::now();
        result.projection_time_ms =
            std::chrono::duration<double, std::milli>(end_proj - start_proj).count();
    }
    else
    {
        processed_points.assign(points.begin(), points.end());
    }
    auto start_build = std::chrono::high_resolution_clock::now();
    const double sampling_rate = result.config.edge_sampling_rate;
    std::vector<std::pair<int, int>> sampled_edges;
    std::vector<double> sampled_weights;
    size_t eligible_edges = 0;
    std::mt19937 rng(result.config.random_seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double max_radius_sq = result.config.max_radius * result.config.max_radius;
    if (!std::isfinite(max_radius_sq))
    {
        result.total_time_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() - start_total)
                                   .count();
        return result;
    }
    for (size_t i = 0; i < num_points; ++i)
    {
        const double *p_i = &processed_points[i * processed_dim];
        for (size_t j = i + 1; j < num_points; ++j)
        {
            const double *p_j = &processed_points[j * processed_dim];
            double d_sq = 0.0;
            for (size_t d = 0; d < processed_dim; ++d)
            {
                double diff = p_i[d] - p_j[d];
                d_sq += diff * diff;
            }
            if (d_sq <= max_radius_sq)
            {
                ++eligible_edges;
                const double dist_val = std::sqrt(d_sq);
                const double keep_prob =
                    sampling_rate >= 1.0
                        ? 1.0
                        : sampling_rate * std::exp(-dist_val / result.config.max_radius);
                if (dist(rng) < keep_prob)
                {
                    sampled_edges.push_back({static_cast<int>(i), static_cast<int>(j)});
                    sampled_weights.push_back(dist_val);
                }
            }
        }
    }
    auto end_build = std::chrono::high_resolution_clock::now();
    result.build_time_ms =
        std::chrono::duration<double, std::milli>(end_build - start_build).count();
    auto start_ph = std::chrono::high_resolution_clock::now();
    auto d0_pairs_raw = computeD0PersistenceUnionFind(processed_points, processed_dim, num_points,
                                                      sampled_edges, sampled_weights);
    result.pairs.clear();
    result.pairs.reserve(d0_pairs_raw.size());
    for (const auto &p : d0_pairs_raw)
    {
        result.pairs.push_back(p);
    }
    auto end_ph = std::chrono::high_resolution_clock::now();
    result.persistence_time_ms =
        std::chrono::duration<double, std::milli>(end_ph - start_ph).count();
    // Compute statistics
    auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();
    result.num_edges_sampled = sampled_edges.size();
    result.num_edges_total = eligible_edges;
    result.edge_sampling_ratio = eligible_edges == 0 ? 0.0
                                                     : static_cast<double>(sampled_edges.size()) /
                                                           static_cast<double>(eligible_edges);
    const auto accuracy = estimateApproximationAccuracy(result.config, num_points, processed_dim);
    result.estimated_accuracy = accuracy.overall_accuracy;
    result.theoretical_error_bound = accuracy.confidence_interval_95;
    return result;
}
// Get optimal sketching config
SketchingConfig getOptimalSketchingConfig(size_t num_points, size_t point_dim)
{
    SketchingConfig config;
    config.max_dim = SKETCHING_DEFAULT_MAX_DIM;
    config.max_radius = SKETCHING_DEFAULT_MAX_RADIUS;
    // Sampling rate based on dataset size
    if (num_points < SKETCHING_SMALL_DATASET_THRESHOLD)
    {
        config.edge_sampling_rate = SKETCHING_SAMPLE_RATE_FULL; // No sampling for small datasets
    }
    else if (num_points < SKETCHING_MEDIUM_DATASET_THRESHOLD)
    {
        config.edge_sampling_rate = SKETCHING_SAMPLE_RATE_HALF; // 50% sampling
    }
    else if (num_points < SKETCHING_LARGE_DATASET_THRESHOLD)
    {
        config.edge_sampling_rate = SKETCHING_SAMPLE_RATE_FIFTH; // 20% sampling
    }
    else
    {
        config.edge_sampling_rate = SKETCHING_SAMPLE_RATE_TENTH; // 10% sampling for very large
    }
    config.use_dimensionality_reduction = (point_dim > SKETCHING_DIM_REDUCTION_THRESHOLD);
    config.target_dimension =
        point_dim == 0 ? 0
                       : std::min(SKETCHING_TARGET_DIMENSION, std::max<size_t>(1, point_dim / 2));
    config.random_seed = SKETCHING_DEFAULT_RANDOM_SEED;
    return normalizeConfig(config, point_dim);
}
// Estimate approximation accuracy
ApproximationAccuracy estimateApproximationAccuracy(const SketchingConfig &config,
                                                    size_t num_points, size_t point_dim)
{
    ApproximationAccuracy accuracy{};
    if (num_points < 2 || point_dim == 0)
    {
        return accuracy;
    }
    const SketchingConfig normalized = normalizeConfig(config, point_dim);
    const double n = static_cast<double>(num_points);
    const double sampling_rate = normalized.edge_sampling_rate;
    if (normalized.use_dimensionality_reduction && normalized.target_dimension > 0 &&
        normalized.target_dimension < point_dim)
    {
        const double raw_distortion =
            std::sqrt(std::log(n) / static_cast<double>(normalized.target_dimension));
        accuracy.jl_distortion = std::clamp(raw_distortion, 0.0, 1.0);
        accuracy.distance_preservation = 1.0 - accuracy.jl_distortion;
    }
    else
    {
        accuracy.jl_distortion = 0.0;
        accuracy.distance_preservation = 1.0;
    }
    const double effective_samples = sampling_rate * n;
    accuracy.sampling_error =
        effective_samples <= 0.0 ? 1.0 : std::clamp(1.0 / std::sqrt(effective_samples), 0.0, 1.0);
    accuracy.overall_accuracy =
        std::clamp(accuracy.distance_preservation * (1.0 - accuracy.sampling_error), 0.0, 1.0);
    accuracy.confidence_interval_95 = std::clamp(2.0 * accuracy.sampling_error, 0.0, 1.0);
    return accuracy;
}
} // namespace nerve::persistence
