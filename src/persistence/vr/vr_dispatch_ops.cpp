
#include "nerve/persistence/reduction/reduction_edge_collapse_ops.hpp"
#include "nerve/persistence/vr/vr_dispatch_ops.hpp"

using nerve::persistence::collapseEdges;

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace nerve::persistence
{

namespace
{

constexpr double kPairTolerance = 1e-9;

VRConfig buildFastConfig(const VRDispatchConfig &dispatch_config, size_t num_points,
                         size_t point_dim)
{
    VRConfig fast_config = getOptimalFastvrConfig(num_points, point_dim);
    fast_config.max_dim = dispatch_config.max_dim;
    fast_config.max_radius = dispatch_config.max_radius;
    fast_config.num_threads = dispatch_config.num_threads;
    fast_config.algorithm = VRAlgorithmSelection::AUTO;

    // Use dispatch config optimization flags instead of hardcoding false
    fast_config.use_acceleration = true;
    fast_config.use_accelerated_runtime = true;
    fast_config.use_adaptive_acceleration = true;
    fast_config.enable_lockfree_multicore = dispatch_config.use_lockfree_reduction;
    fast_config.acceleration.enable_streaming = false;
    fast_config.enable_approximation = false;
    fast_config.enable_matrix_multiplication = false;
    fast_config.enable_sparsification = false;
    fast_config.prefer_gpu = false;
    fast_config.prefer_multicore = false;
    fast_config.acceleration.threshold = dispatch_config.min_points_for_collapse;
    fast_config.acceleration.mode = AccelerationMode::CPU_ONLY;

    return fast_config;
}

bool arePairsEquivalent(const std::vector<Pair> &lhs, const std::vector<Pair> &rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        if (lhs[i].dimension != rhs[i].dimension)
        {
            return false;
        }
        if (std::abs(lhs[i].birth - rhs[i].birth) > kPairTolerance)
        {
            return false;
        }
        const bool lhs_inf = std::isinf(lhs[i].death);
        const bool rhs_inf = std::isinf(rhs[i].death);
        if (lhs_inf != rhs_inf)
        {
            return false;
        }
        if (!lhs_inf && std::abs(lhs[i].death - rhs[i].death) > kPairTolerance)
        {
            return false;
        }
    }
    return true;
}

} // namespace

namespace
{
std::vector<std::vector<int>> buildNeighborGraph(const double *points, size_t n, size_t dim,
                                                 double max_radius)
{
    std::vector<std::vector<int>> neighbors(n);
    const double max_radius_sq = max_radius * max_radius;
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = i + 1; j < n; ++j)
        {
            double dist_sq = 0.0;
            for (size_t d = 0; d < dim; ++d)
            {
                double diff = points[i * dim + d] - points[j * dim + d];
                dist_sq += diff * diff;
            }
            if (dist_sq <= max_radius_sq)
            {
                neighbors[i].push_back(static_cast<int>(j));
                neighbors[j].push_back(static_cast<int>(i));
            }
        }
    }
    return neighbors;
}

std::vector<double> buildEdgeWeights(const double *points, size_t n, size_t dim,
                                     const std::vector<std::vector<int>> &neighbors)
{
    std::vector<double> weights;
    for (size_t i = 0; i < n; ++i)
    {
        for (int j : neighbors[i])
        {
            if (static_cast<size_t>(j) <= i)
                continue;
            double dist_sq = 0.0;
            for (size_t d = 0; d < dim; ++d)
            {
                double diff = points[i * dim + d] - points[static_cast<size_t>(j) * dim + d];
                dist_sq += diff * diff;
            }
            weights.push_back(std::sqrt(dist_sq));
        }
    }
    return weights;
}
} // namespace

std::vector<Pair> computeVrPersistenceDispatch(core::BufferView<const double> points,
                                               Size point_dim, const VRDispatchConfig &config)
{
    const Size point_count = point_dim == 0 ? 0 : points.size() / point_dim;
    VRConfig fast_config = buildFastConfig(config, point_count, point_dim);

    // Apply edge collapse preprocessing if enabled and beneficial
    if (config.use_edge_collapse && point_count >= config.min_points_for_collapse)
    {
        const double *pt_data = points.data();
        auto neighbors = buildNeighborGraph(pt_data, point_count, point_dim, config.max_radius);

        auto edge_weights = buildEdgeWeights(pt_data, point_count, point_dim, neighbors);

        auto collapse_result = collapseEdges(neighbors, edge_weights, config.max_radius);

        double vertex_reduction = 1.0 - static_cast<double>(collapse_result.remaining_vertices) /
                                            static_cast<double>(collapse_result.original_vertices);

        if (vertex_reduction >= config.collapse_min_reduction &&
            collapse_result.remaining_vertices > 0)
        {
            // Build reduced point cloud from alive vertices
            std::vector<double> reduced_points;
            std::vector<size_t> alive_to_original;
            reduced_points.reserve(collapse_result.remaining_vertices * point_dim);

            for (size_t i = 0; i < point_count; ++i)
            {
                if (collapse_result.vertex_alive[i])
                {
                    alive_to_original.push_back(i);
                    for (size_t d = 0; d < point_dim; ++d)
                    {
                        reduced_points.push_back(pt_data[i * point_dim + d]);
                    }
                }
            }

            core::BufferView<const double> reduced_view(reduced_points.data(),
                                                        reduced_points.size());

            auto reduced_pairs = computeVrPersistenceFast(reduced_view, point_dim, fast_config);

            // Map pairs back to original vertex indices
            for (auto &pair : reduced_pairs)
            {
                if (pair.dimension == 0 && pair.birth >= 0 &&
                    static_cast<size_t>(pair.birth) < alive_to_original.size())
                {
                    pair.birth =
                        static_cast<double>(alive_to_original[static_cast<size_t>(pair.birth)]);
                }
                if (pair.dimension == 0 && std::isfinite(pair.death) && pair.death >= 0 &&
                    static_cast<size_t>(pair.death) < alive_to_original.size())
                {
                    pair.death =
                        static_cast<double>(alive_to_original[static_cast<size_t>(pair.death)]);
                }
            }
            return reduced_pairs;
        }
    }

    return computeVrPersistenceFast(points, point_dim, fast_config);
}

VRDispatchConfig getVrDispatchConfig(size_t num_points, size_t point_dim)
{
    VRDispatchConfig config;
    (void)point_dim;
    config.max_dim = 2;
    config.max_radius = 1.0;
    config.use_union_find_d0 = true;
    config.use_union_find_top = true;
    config.use_lockfree_reduction = true;
    config.use_edge_collapse = (num_points >= 100);
    config.use_discrete_morse = false;
    config.min_points_for_collapse = 100;
    config.collapse_min_reduction = 0.15;
    return config;
}

VRDispatchBenchmark benchmarkVrDispatch(core::BufferView<const double> points, Size point_dim,
                                        const VRDispatchConfig &config)
{
    VRDispatchBenchmark result{};
    result.num_points = point_dim == 0 ? 0 : points.size() / point_dim;
    result.point_dim = point_dim;

    auto start_standard = std::chrono::high_resolution_clock::now();
    VRConfig standard_config = getOptimalFastvrConfig(result.num_points, point_dim);
    standard_config.max_dim = config.max_dim;
    standard_config.max_radius = config.max_radius;
    auto standard_pairs = computeVrPersistenceFast(points, point_dim, standard_config);
    auto end_standard = std::chrono::high_resolution_clock::now();
    result.standard_time_ms =
        std::chrono::duration<double, std::milli>(end_standard - start_standard).count();
    result.standard_num_pairs = standard_pairs.size();

    auto start_dispatch = std::chrono::high_resolution_clock::now();
    auto dispatch_pairs = computeVrPersistenceDispatch(points, point_dim, config);
    auto end_dispatch = std::chrono::high_resolution_clock::now();
    result.dispatch_time_ms =
        std::chrono::duration<double, std::milli>(end_dispatch - start_dispatch).count();
    result.dispatch_num_pairs = dispatch_pairs.size();

    result.speedup =
        result.dispatch_time_ms > 0.0 ? result.standard_time_ms / result.dispatch_time_ms : 1.0;
    result.correct = arePairsEquivalent(standard_pairs, dispatch_pairs);
    return result;
}

} // namespace nerve::persistence
