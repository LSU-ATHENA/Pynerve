
// Unified dispatcher for exact per-dimension H0-H6 kernels.

#include "nerve/persistence/core/per_dimension_exact.hpp"
#include "nerve/persistence/kernels/kernel_h1_ops.hpp"
#include "nerve/persistence/kernels/kernel_h2_alpha_ops.hpp"
#include "nerve/persistence/kernels/kernel_h3_tetrahedra_ops.hpp"
#include "nerve/persistence/kernels/kernel_h4_chunked_ops.hpp"
#include "nerve/persistence/kernels/kernel_h5_prefetch_ops.hpp"
#include "nerve/persistence/kernels/kernel_h6_streaming_ops.hpp"
#include "nerve/persistence/reduction/reduction_union_find_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>

namespace nerve::persistence::perdim
{

namespace
{

double pointDistance(const std::vector<double> &points, size_t point_dim, size_t lhs, size_t rhs)
{
    double squared = 0.0;
    const size_t lhs_offset = lhs * point_dim;
    const size_t rhs_offset = rhs * point_dim;
    for (size_t axis = 0; axis < point_dim; ++axis)
    {
        const double delta = points[lhs_offset + axis] - points[rhs_offset + axis];
        const double contribution = delta * delta;
        if (!std::isfinite(contribution) ||
            squared > std::numeric_limits<double>::max() - contribution)
        {
            throw std::overflow_error("point distance overflow");
        }
        squared += contribution;
    }
    const double distance = std::sqrt(squared);
    if (!std::isfinite(distance))
    {
        throw std::overflow_error("point distance overflow");
    }
    return distance;
}

void validatePerDimensionInputs(const std::vector<std::vector<int>> &simplices,
                                const std::vector<double> &filtration_values,
                                const std::vector<int> &dimensions)
{
    if (simplices.size() != filtration_values.size() || simplices.size() != dimensions.size())
    {
        throw std::invalid_argument(
            "simplices, filtration values, and dimensions must have matching sizes");
    }
    for (size_t i = 0; i < simplices.size(); ++i)
    {
        if (simplices[i].empty())
        {
            throw std::invalid_argument("simplices must be non-empty");
        }
        for (int vertex : simplices[i])
        {
            if (vertex < 0)
            {
                throw std::invalid_argument("simplex vertices must be non-negative");
            }
        }
        if (!std::isfinite(filtration_values[i]))
        {
            throw std::invalid_argument("filtration values must be finite");
        }
        if (dimensions[i] < 0 || dimensions[i] != static_cast<int>(simplices[i].size()) - 1)
        {
            throw std::invalid_argument("simplex dimensions must match simplex cardinality");
        }
    }
}

void buildVietorisRipsComplex(const std::vector<double> &points, size_t point_dim,
                              size_t num_points, const PerDimensionConfig &config,
                              std::vector<std::vector<int>> *simplices,
                              std::vector<double> *filtration_values, std::vector<int> *dimensions)
{
    if (point_dim == 0 && num_points > 0)
    {
        throw std::invalid_argument("point dimension must be positive");
    }
    if (!std::isfinite(config.max_radius) || config.max_radius < 0.0)
    {
        throw std::invalid_argument("max_radius must be finite and non-negative");
    }
    if (point_dim != 0 && num_points > std::numeric_limits<size_t>::max() / point_dim)
    {
        throw std::invalid_argument("point buffer dimensions overflow");
    }
    if (points.size() != point_dim * num_points)
    {
        throw std::invalid_argument("point buffer size does not match point count and dimension");
    }
    if (num_points > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("point count exceeds per-dimension vertex index range");
    }
    for (double value : points)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("point coordinates must be finite");
        }
    }

    simplices->clear();
    filtration_values->clear();
    dimensions->clear();
    if (num_points == 0)
    {
        return;
    }
    const int point_bound = num_points > static_cast<size_t>(std::numeric_limits<int>::max())
                                ? std::numeric_limits<int>::max()
                                : static_cast<int>(num_points) - 1;
    const int max_dim = std::min({config.max_dim, 6, point_bound});
    if (max_dim < 0)
    {
        return;
    }

    if (num_points > std::numeric_limits<size_t>::max() / num_points)
    {
        throw std::length_error("distance matrix size overflows");
    }
    const size_t distance_count = num_points * num_points;
    if (distance_count > std::vector<double>().max_size())
    {
        throw std::length_error("distance matrix size exceeds vector capacity");
    }
    std::vector<double> distances(distance_count, 0.0);
    for (size_t i = 0; i < num_points; ++i)
    {
        for (size_t j = i + 1; j < num_points; ++j)
        {
            const double distance = pointDistance(points, point_dim, i, j);
            distances[i * num_points + j] = distance;
            distances[j * num_points + i] = distance;
        }
        simplices->push_back({static_cast<int>(i)});
        filtration_values->push_back(0.0);
        dimensions->push_back(0);
    }

    std::vector<int> current;
    current.reserve(static_cast<size_t>(max_dim) + 1);

    std::function<void(size_t, double)> extend = [&](size_t next_vertex, double max_edge) {
        if (current.size() > 1)
        {
            simplices->push_back(current);
            filtration_values->push_back(max_edge);
            dimensions->push_back(static_cast<int>(current.size()) - 1);
        }
        if (current.size() == static_cast<size_t>(max_dim) + 1)
        {
            return;
        }

        for (size_t candidate = next_vertex; candidate < num_points; ++candidate)
        {
            double next_max = max_edge;
            bool is_clique = true;
            for (int existing : current)
            {
                const double edge =
                    distances[static_cast<size_t>(existing) * num_points + candidate];
                if (edge > config.max_radius)
                {
                    is_clique = false;
                    break;
                }
                next_max = std::max(next_max, edge);
            }
            if (!is_clique)
            {
                continue;
            }
            current.push_back(static_cast<int>(candidate));
            extend(candidate + 1, next_max);
            current.pop_back();
        }
    };

    for (size_t vertex = 0; vertex < num_points; ++vertex)
    {
        current.push_back(static_cast<int>(vertex));
        extend(vertex + 1, 0.0);
        current.pop_back();
    }
}

} // namespace

// Main dispatcher: computes each dimension with the selected exact kernel.
PerDimensionResult computePerDimension(const std::vector<std::vector<int>> &simplices,
                                       const std::vector<double> &filtration_values,
                                       const std::vector<int> &dimensions,
                                       const PerDimensionConfig &config)
{
    validatePerDimensionInputs(simplices, filtration_values, dimensions);
    PerDimensionResult result;
    result.config = config;

    auto start_total = std::chrono::high_resolution_clock::now();

    // H0: Union-Find
    if (config.compute_h0)
    {
        auto h0_result = computeH0UnionFind(simplices, filtration_values);
        result.all_pairs.insert(result.all_pairs.end(), h0_result.pairs.begin(),
                                h0_result.pairs.end());
        result.h0_time_ms = h0_result.time_ms;
        result.h0_pairs = h0_result.num_pairs;
    }

    // H1: exact Z2 reduction
    if (config.compute_h1)
    {
        auto h1_result = computeH1ReducedVR(simplices, filtration_values, dimensions);
        result.all_pairs.insert(result.all_pairs.end(), h1_result.pairs.begin(),
                                h1_result.pairs.end());
        result.h1_time_ms = h1_result.time_ms;
        result.h1_pairs = h1_result.num_pairs;
    }

    // H2: exact Z2 reduction
    if (config.compute_h2)
    {
        auto h2_result = computeH2AlphaComplex(simplices, filtration_values, dimensions);
        result.all_pairs.insert(result.all_pairs.end(), h2_result.pairs.begin(),
                                h2_result.pairs.end());
        result.h2_time_ms = h2_result.time_ms;
        result.h2_pairs = h2_result.num_pairs;
    }

    // H3: exact Z2 reduction over tetrahedra and cofaces
    if (config.compute_h3)
    {
        auto h3_config =
            h3::getOptimalH3Config(std::count(dimensions.begin(), dimensions.end(), 3));

        auto h3_result =
            h3::computeH3Tetrahedra(simplices, filtration_values, dimensions, h3_config);

        for (const auto &pair : h3_result.pairs)
        {
            result.all_pairs.push_back({pair.birth_time, pair.death_time, 3});
        }

        result.h3_time_ms = h3_result.total_time_ms;
        result.h3_pairs = static_cast<int>(h3_result.pairs.size());
        result.h3_used_tetrahedra_opt = false;
    }

    // H4: exact Z2 reduction
    if (config.compute_h4)
    {
        auto h4_config =
            h4::getOptimalH4Config(std::count(dimensions.begin(), dimensions.end(), 4));

        auto h4_result = h4::computeH4Chunked(simplices, filtration_values, dimensions, h4_config);

        for (const auto &pair : h4_result.pairs)
        {
            result.all_pairs.push_back({pair.birth_time, pair.death_time, 4});
        }

        result.h4_time_ms = h4_result.computation_time_ms;
        result.h4_pairs = static_cast<int>(h4_result.pairs.size());
        result.h4_used_chunking = h4_result.used_chunking;
    }

    // H5: exact Z2 reduction with input prefetch
    if (config.compute_h5)
    {
        auto h5_config =
            h5::getOptimalH5Config(std::count(dimensions.begin(), dimensions.end(), 5));

        auto h5_result = h5::computeH5Prefetch(simplices, filtration_values, dimensions, h5_config);

        for (const auto &pair : h5_result.pairs)
        {
            result.all_pairs.push_back({pair.birth_time, pair.death_time, 5});
        }

        result.h5_time_ms = h5_result.computation_time_ms;
        result.h5_pairs = static_cast<int>(h5_result.pairs.size());
        result.h5_used_prefetch = h5_result.used_prefetching;
    }

    // H6: exact Z2 reduction
    if (config.compute_h6)
    {
        auto h6_config =
            h6::getOptimalH6Config(std::count(dimensions.begin(), dimensions.end(), 6));

        auto h6_result =
            h6::computeH6Streaming(simplices, filtration_values, dimensions, h6_config);

        for (const auto &pair : h6_result.pairs)
        {
            result.all_pairs.push_back({pair.birth_time, pair.death_time, 6});
        }

        result.h6_time_ms = h6_result.computation_time_ms;
        result.h6_pairs = static_cast<int>(h6_result.pairs.size());
        result.h6_used_streaming = h6_result.used_streaming;
    }

    // Compute totals
    result.total_time_ms = result.h0_time_ms + result.h1_time_ms + result.h2_time_ms +
                           result.h3_time_ms + result.h4_time_ms + result.h5_time_ms +
                           result.h6_time_ms;

    auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();

    return result;
}

// Convenience function: compute 0-6D from point cloud
PerDimensionResult compute0To6DPerDimension(const std::vector<double> &points, size_t point_dim,
                                            size_t num_points, const PerDimensionConfig &config)
{
    std::vector<std::vector<int>> simplices;
    std::vector<double> filtration_values;
    std::vector<int> dimensions;
    buildVietorisRipsComplex(points, point_dim, num_points, config, &simplices, &filtration_values,
                             &dimensions);
    return computePerDimension(simplices, filtration_values, dimensions, config);
}

// Get optimal per-dimension config
PerDimensionConfig getOptimalPerDimensionConfig(size_t num_points, size_t point_dim, int max_dim)
{
    PerDimensionConfig config;
    config.max_dim = std::clamp(max_dim, 0, 6);

    // Enable all dimensions up to max_dim
    config.compute_h0 = true;
    config.compute_h1 = (config.max_dim >= 1);
    config.compute_h2 = (config.max_dim >= 2);
    config.compute_h3 = (config.max_dim >= 3);
    config.compute_h4 = (config.max_dim >= 4);
    config.compute_h5 = (config.max_dim >= 5);
    config.compute_h6 = (config.max_dim >= 6);

    // Configure each dimension
    config.h1 = h1::getOptimalH1Config(num_points, config.max_radius);
    config.h2 = h2::getOptimalH2Config(num_points, point_dim, config.max_radius);
    // H3-H6 configured automatically in their compute functions

    return config;
}

// Benchmark per-dimension
PerDimensionBenchmark benchmarkPerDimension(const std::vector<std::vector<int>> &simplices,
                                            const std::vector<double> &filtration_values,
                                            const std::vector<int> &dimensions)
{
    PerDimensionBenchmark bench;

    // Run per-dimension
    PerDimensionConfig config;
    config.compute_h0 = true;
    config.compute_h1 = true;
    config.compute_h2 = true;
    config.compute_h3 = true;
    config.compute_h4 = true;
    config.compute_h5 = true;
    config.compute_h6 = true;

    auto result = computePerDimension(simplices, filtration_values, dimensions, config);

    // Record times
    bench.h0_time_ms = result.h0_time_ms;
    bench.h1_time_ms = result.h1_time_ms;
    bench.h2_time_ms = result.h2_time_ms;
    bench.h3_time_ms = result.h3_time_ms;
    bench.h4_time_ms = result.h4_time_ms;
    bench.h5_time_ms = result.h5_time_ms;
    bench.h6_time_ms = result.h6_time_ms;
    bench.total_time_ms = result.total_time_ms;

    // Count pairs
    bench.h0_pairs = result.h0_pairs;
    bench.h1_pairs = result.h1_pairs;
    bench.h2_pairs = result.h2_pairs;
    bench.h3_pairs = result.h3_pairs;
    bench.h4_pairs = result.h4_pairs;
    bench.h5_pairs = result.h5_pairs;
    bench.h6_pairs = result.h6_pairs;

    return bench;
}

} // namespace nerve::persistence::perdim
