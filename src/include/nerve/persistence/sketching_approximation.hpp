#pragma once

#include "nerve/persistence/core/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace nerve::persistence
{

// Configuration for sketching approximation.
struct SketchingConfig
{
    std::size_t sketch_size = 1000;
    double approximation_factor = 0.1;
    bool use_random_projection = true;
    int max_dim = 2;                 // Maximum dimension.
    double max_radius = 2.0;         // Maximum VR radius.
    double edge_sampling_rate = 0.5; // Edge sampling rate.
    double error_tolerance = 0.1;    // Error tolerance.
    bool use_dimensionality_reduction = false;
    unsigned int random_seed = 42;
    std::size_t target_dimension = 50; // Target dimension for JL projection.
};

// Result of sketching computation.
struct SketchingResult
{
    std::vector<Pair> persistence_pairs;
    double computation_time_ms = 0.0;
    double approximation_error = 0.0;
    std::size_t num_columns_sketched = 0;
    double projection_time_ms = 0.0;
    double build_time_ms = 0.0;
    double persistence_time_ms = 0.0;
    double total_time_ms = 0.0;
    std::size_t num_edges_sampled = 0;
    std::size_t num_edges_total = 0;
    double edge_sampling_ratio = 0.0;
    double estimated_accuracy = 0.0;
    double theoretical_error_bound = 0.0;
};

// Approximation accuracy metrics.
struct ApproximationAccuracy
{
    double overall_accuracy = 0.0;
    double distance_preservation = 0.0;
    double jl_distortion = 0.0;
    double sampling_error = 0.0;
    double confidence_interval_95 = 0.0;
};

// Compute D0 persistence using union-find.
std::vector<Pair> computeD0PersistenceUnionFind(const std::vector<double> &points,
                                                std::size_t point_dim, std::size_t num_points,
                                                double max_distance);

// Random projection sketch for distance matrix approximation.
class DistanceMatrixSketch
{
public:
    explicit DistanceMatrixSketch(const SketchingConfig &config = {})
        : config_(config)
    {}

    // Build sketch from point cloud.
    void build(const std::vector<double> &points, std::size_t point_dim, std::size_t num_points)
    {
        projected_points_.clear();
        projected_dim_ = 0;
        point_dim_ = point_dim;
        num_points_ = num_points;

        if (point_dim == 0 || num_points == 0 || points.size() != point_dim * num_points)
        {
            return;
        }

        bool reduce_dim = config_.use_random_projection && config_.use_dimensionality_reduction &&
                          config_.target_dimension > 0 && config_.target_dimension < point_dim;
        if (!reduce_dim)
        {
            projected_dim_ = point_dim;
            projected_points_ = points;
            return;
        }

        projected_dim_ = config_.target_dimension;
        projected_points_.assign(num_points * projected_dim_, 0.0);

        std::mt19937_64 rng(config_.random_seed);
        const double scale = 1.0 / std::sqrt(static_cast<double>(projected_dim_));
        std::normal_distribution<double> normal(0.0, scale);

        std::vector<double> projection(projected_dim_ * point_dim, 0.0);
        for (double &x : projection)
        {
            x = normal(rng);
        }

        for (std::size_t i = 0; i < num_points; ++i)
        {
            const std::size_t src = i * point_dim;
            const std::size_t dst = i * projected_dim_;
            for (std::size_t r = 0; r < projected_dim_; ++r)
            {
                const std::size_t row = r * point_dim;
                double sum = 0.0;
                for (std::size_t c = 0; c < point_dim; ++c)
                {
                    sum += projection[row + c] * points[src + c];
                }
                projected_points_[dst + r] = sum;
            }
        }
    }

    // Approximate distance between two points.
    [[nodiscard]] double approximateDistance(std::size_t i, std::size_t j) const
    {
        if (projected_dim_ == 0 || i >= num_points_ || j >= num_points_)
        {
            return std::numeric_limits<double>::infinity();
        }
        if (i == j)
        {
            return 0.0;
        }

        const std::size_t bi = i * projected_dim_;
        const std::size_t bj = j * projected_dim_;
        double dist2 = 0.0;
        for (std::size_t d = 0; d < projected_dim_; ++d)
        {
            const double diff = projected_points_[bi + d] - projected_points_[bj + d];
            dist2 += diff * diff;
        }
        return std::sqrt(dist2);
    }

private:
    SketchingConfig config_;
    std::vector<double> projected_points_;
    std::size_t projected_dim_ = 0;
    std::size_t point_dim_ = 0;
    std::size_t num_points_ = 0;
};

} // namespace nerve::persistence
