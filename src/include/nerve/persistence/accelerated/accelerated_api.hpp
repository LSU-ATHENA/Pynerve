
#pragma once
#include "nerve/core.hpp"
#include "nerve/persistence/accelerated/heterogeneous_fast_vr.hpp"
#include "nerve/persistence/accelerated/thread_safe_allocator.hpp"
#include "nerve/persistence/accelerated/work_distributor.hpp"
#include "nerve/persistence/cuda/gpu_reduction_engine.hpp"
#include "nerve/persistence/hybrid_reduction_engine.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace nerve::persistence::accelerated
{

// Re-export canonical performance types from nerve/common/accelerated_types.hpp.
using ::nerve::common::AcceleratedPerformanceStats;
using ::nerve::common::PerformanceMetrics;

#ifndef NERVE_ACCELERATED_MEMORY_USAGE_STATS_DEFINED
#define NERVE_ACCELERATED_MEMORY_USAGE_STATS_DEFINED
struct MemoryUsageStats
{
    size_t total_allocated = 0;
    size_t peak_allocated = 0;
    size_t active_allocations = 0;
    size_t pool_bytes = 0;
    size_t pool_free_bytes = 0;
    double fragmentation_ratio = 0.0;
};
#endif
namespace factory
{
inline errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>
createOptimalVrEngine(size_t n_points = 0, size_t point_dim = 0)
{
    VRConfig config;
    if (n_points > 0)
    {
        config.acceleration.threshold = std::min<size_t>(1000, n_points / 2);
    }
    if (point_dim > 3)
    {
        config.max_dim = std::min<size_t>(point_dim - 1, 5);
    }
    return AcceleratedVREngine::create(config);
}
} // namespace factory
namespace utils
{
inline errors::ErrorResult<void> validateVrInput(core::BufferView<const double> points,
                                                 size_t point_dim,
                                                 const core::DeterminismContract &contract)
{
    if (points.size() == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT,
                                                "Empty input points");
    }
    if (point_dim == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT,
                                                "Invalid point dimension");
    }
    if (points.size() % point_dim != 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT,
                                                "Point data size not divisible by dimension");
    }
    const bool strict = contract.level == core::DeterminismLevel::STRICT;
    for (size_t i = 0; i < points.size(); ++i)
    {
        if (!std::isfinite(points[i]))
        {
            return errors::ErrorResult<void>::error(
                errors::ErrorCode::E50_PH_ABORT,
                strict ? "Point coordinates must be finite under strict determinism"
                       : "Point coordinates must be finite");
        }
    }
    return errors::ErrorResult<void>::ok();
}

inline void validateEstimatorRadius(double max_radius)
{
    if (!std::isfinite(max_radius) || max_radius <= 0.0)
    {
        throw std::invalid_argument("max_radius must be finite and positive");
    }
}

inline double estimateComputationTime(size_t n_points, size_t point_dim, double max_radius,
                                      bool enable_gpu = false)
{
    validateEstimatorRadius(max_radius);
    double base_time =
        std::pow(static_cast<double>(n_points), 1.5) * static_cast<double>(point_dim) * max_radius;
    if (enable_gpu && n_points >= 1000)
    {
        const double acceleration = std::clamp(std::log2(static_cast<double>(n_points)), 1.0, 32.0);
        base_time /= acceleration;
    }
    return base_time;
}

inline double estimateVrEdgeDensity(size_t n_points, size_t point_dim, double max_radius)
{
    validateEstimatorRadius(max_radius);
    if (n_points < 2 || point_dim == 0)
    {
        return 0.0;
    }
    const double exponent = static_cast<double>(std::clamp<size_t>(point_dim, 1, 4));
    const double radius_density = std::pow(max_radius, exponent);
    const double minimum_density = 1.0 / static_cast<double>(n_points);
    return std::clamp(radius_density, minimum_density, 1.0);
}

inline size_t estimateMemoryUsage(size_t n_points, size_t point_dim, double max_radius,
                                  bool enable_gpu = false)
{
    validateEstimatorRadius(max_radius);
    if (n_points == 0 || point_dim == 0)
    {
        return 0;
    }

    const long double point_count = static_cast<long double>(n_points);
    const long double pair_count = point_count * static_cast<long double>(n_points - 1) / 2.0L;
    const long double density = estimateVrEdgeDensity(n_points, point_dim, max_radius);
    const long double estimated_edges = std::ceil(pair_count * density);

    const long double points_memory =
        point_count * static_cast<long double>(point_dim) * sizeof(double);
    const long double edge_records =
        estimated_edges * (sizeof(std::uint32_t) * 2.0L + sizeof(double));
    const long double filtration_columns =
        estimated_edges * (sizeof(std::uint32_t) + sizeof(double));

    long double total_memory = points_memory + edge_records + filtration_columns;
    if (enable_gpu && n_points >= 1000)
    {
        total_memory += points_memory + edge_records;
    }
    constexpr long double bytes_per_mb = 1024.0L * 1024.0L;
    const long double memory_mb = std::ceil(total_memory / bytes_per_mb);
    if (memory_mb >= static_cast<long double>(std::numeric_limits<size_t>::max()))
    {
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(memory_mb);
}
template <typename T>
inline core::BufferView<const T> createBufferView(const std::vector<T> &data)
{
    return core::BufferView<const T>(data.data(), data.size());
}
template <typename T>
inline core::BufferView<const T> createBufferView(const T *data, size_t size)
{
    return core::BufferView<const T>(data, size);
}
} // namespace utils
} // namespace nerve::persistence::accelerated
