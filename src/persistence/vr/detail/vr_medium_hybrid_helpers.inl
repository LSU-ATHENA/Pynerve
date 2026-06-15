#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/vr/vr_medium_hybrid_ops.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <thread>
#include <unordered_set>
#include <vector>

namespace nerve::persistence
{
namespace
{

constexpr size_t FAST_PATH_THRESHOLD = 1024;
constexpr size_t EXACT_PATH_THRESHOLD = 10000;

struct ExecutionPlan
{
    bool prefer_numa_tiling;
    int numa_nodes;
};

bool checkedProduct(size_t lhs, size_t rhs, size_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        out = 0;
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checkedSquareCount(size_t count, size_t &out)
{
    return checkedProduct(count, count, out);
}

ExecutionPlan computeExecutionPlan(size_t n_points, size_t point_dim, const VRConfig &config)
{
    (void)config;

    size_t point_values = 0;
    size_t matrix_values = 0;
    size_t points_bytes = 0;
    size_t matrix_bytes = 0;
    size_t total_bytes = std::numeric_limits<size_t>::max();
    if (checkedProduct(n_points, point_dim, point_values) &&
        checkedProduct(point_values, sizeof(double), points_bytes) &&
        checkedSquareCount(n_points, matrix_values) &&
        checkedProduct(matrix_values, sizeof(double), matrix_bytes) &&
        matrix_bytes <= std::numeric_limits<size_t>::max() - points_bytes)
    {
        total_bytes = points_bytes + matrix_bytes;
    }
    const bool large_working_set = total_bytes >= (32ULL * 1024ULL * 1024ULL);

    const bool prefer_numa_tiling = large_working_set;
    unsigned hw_threads = std::thread::hardware_concurrency();
    int numa_nodes = hw_threads > 16 ? 2 : 1;
    if (!prefer_numa_tiling)
    {
        numa_nodes = 1;
    }

    return ExecutionPlan{prefer_numa_tiling, numa_nodes};
}

struct SimplexKeyHash
{
    std::size_t operator()(const std::vector<int> &vertices) const noexcept
    {
        std::size_t seed = vertices.size();
        for (int vertex : vertices)
        {
            seed ^= std::hash<int>{}(vertex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

using SimplexSet = std::unordered_set<std::vector<int>, SimplexKeyHash>;

bool hasValidShape(const core::BufferView<const double> &points, Size point_dim)
{
    return point_dim > 0 && !points.empty() && (points.size() % point_dim) == 0;
}

bool hasValidMediumHybridInput(const core::BufferView<const double> &points, Size point_dim,
                               const VRConfig &config)
{
    if (!hasValidShape(points, point_dim))
    {
        return false;
    }
    if (!std::isfinite(config.max_radius) || config.max_radius < 0.0)
    {
        return false;
    }
    if (config.max_dim > static_cast<Size>(std::numeric_limits<Dimension>::max()))
    {
        return false;
    }
    const Size num_points = points.size() / point_dim;
    if (num_points > static_cast<Size>(std::numeric_limits<Index>::max()))
    {
        return false;
    }
    const long double safe_abs =
        std::sqrt(static_cast<long double>(std::numeric_limits<double>::max()) /
                  static_cast<long double>(point_dim)) /
        4.0L;
    for (const double value : points)
    {
        if (!std::isfinite(value) || std::abs(static_cast<long double>(value)) > safe_abs)
        {
            return false;
        }
    }
    return true;
}

} // namespace
} // namespace nerve::persistence
