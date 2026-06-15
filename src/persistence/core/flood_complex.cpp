// Flood complex orchestration and public helpers.

#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/persistence/core/flood_complex.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nerve::persistence
{

std::vector<int> farthestPointSampling(const std::vector<double> &points, size_t point_dim,
                                       size_t n_points, size_t subset_size);

bool isSimplexFlooded(const std::vector<int> &simplex_vertices,
                      const std::vector<int> &candidate_points, const std::vector<double> &points,
                      size_t point_dim, double radius);

double simplexCircumradius(const std::vector<int> &vertices, const std::vector<double> &points,
                           size_t point_dim);

namespace
{

using FiltrationMap = std::unordered_map<algebra::Simplex, double, algebra::Simplex::Hash>;

constexpr size_t FLOOD_MIN_SUBSET_POINTS = 4;
constexpr size_t FLOOD_MAX_FLOODING_CANDIDATES = 20000;
constexpr size_t FLOOD_COMPLETE_GRAPH_LIMIT = 512;
constexpr double FLOOD_EPS = 1e-12;

void validateFloodConfig(const FloodComplexConfig &config)
{
    if (!std::isfinite(config.max_radius) || config.max_radius < 0.0)
    {
        throw std::invalid_argument(
            "FloodComplexConfig max_radius must be finite and non-negative");
    }
    if (!std::isfinite(config.subset_ratio) || config.subset_ratio < 0.0 ||
        config.subset_ratio > 1.0)
    {
        throw std::invalid_argument("FloodComplexConfig subset_ratio must be finite and in [0, 1]");
    }
    if (!std::isfinite(config.flooding_tolerance) || config.flooding_tolerance < 0.0)
    {
        throw std::invalid_argument(
            "FloodComplexConfig flooding_tolerance must be finite and non-negative");
    }
}

size_t chooseClamped(size_t n, size_t k, size_t cap)
{
    if (k > n)
    {
        return 0;
    }
    if (k == 0 || k == n)
    {
        return 1;
    }
    k = std::min(k, n - k);
    long double value = 1.0L;
    for (size_t i = 1; i <= k; ++i)
    {
        value *= static_cast<long double>(n - k + i);
        value /= static_cast<long double>(i);
        if (value >= static_cast<long double>(cap))
        {
            return cap;
        }
    }
    return static_cast<size_t>(std::llround(value));
}

size_t checkedRequiredPointValues(size_t num_points, size_t point_dim)
{
    if (point_dim != 0 && num_points > std::numeric_limits<size_t>::max() / point_dim)
    {
        throw std::length_error("Flood Complex point buffer dimensions overflow");
    }
    return num_points * point_dim;
}

void validatePointValues(const std::vector<double> &points, size_t required_values)
{
    for (size_t i = 0; i < required_values; ++i)
    {
        if (!std::isfinite(points[i]))
        {
            throw std::invalid_argument("Flood Complex point coordinates must be finite");
        }
    }
}

double estimateVrSimplices(size_t num_points, Size max_dim)
{
    if (num_points == 0)
    {
        return 1.0;
    }
    const size_t dim_cap = static_cast<size_t>(std::max<Size>(0, max_dim));
    const size_t k_max = std::min(num_points, dim_cap + 1);
    double total = 0.0;
    for (size_t k = 0; k <= k_max; ++k)
    {
        total += static_cast<double>(
            chooseClamped(num_points, k + 1, std::numeric_limits<size_t>::max() / 4));
    }
    return std::max(1.0, total);
}

std::vector<int> floodCandidates(size_t num_points)
{
    std::vector<int> candidates;
    if (num_points == 0)
    {
        return candidates;
    }
    if (num_points <= FLOOD_MAX_FLOODING_CANDIDATES)
    {
        candidates.resize(num_points);
        std::iota(candidates.begin(), candidates.end(), 0);
        return candidates;
    }

    candidates.reserve(FLOOD_MAX_FLOODING_CANDIDATES);
    const size_t stride = std::max<size_t>(1, num_points / FLOOD_MAX_FLOODING_CANDIDATES);
    for (size_t i = 0; i < num_points && candidates.size() < FLOOD_MAX_FLOODING_CANDIDATES;
         i += stride)
    {
        candidates.push_back(static_cast<int>(i));
    }
    return candidates;
}

void insertWithClosure(const algebra::Simplex &simplex, double filtration, Size max_dim,
                       FiltrationMap *filtrations)
{
    if (filtrations == nullptr)
    {
        return;
    }
    if (simplex.dimension() > max_dim)
    {
        for (const auto &face : simplex.faces(core::DeterminismContract{}))
        {
            insertWithClosure(face, filtration, max_dim, filtrations);
        }
        return;
    }

    const auto [it, inserted] = filtrations->insert({simplex, filtration});
    if (!inserted && filtration < it->second)
    {
        it->second = filtration;
    }
    if (simplex.dimension() == 0)
    {
        return;
    }
    for (const auto &face : simplex.faces(core::DeterminismContract{}))
    {
        insertWithClosure(face, filtration, max_dim, filtrations);
    }
}

algebra::Simplex makeSimplex(const std::vector<int> &vertices)
{
    std::vector<Index> converted;
    converted.reserve(vertices.size());
    for (const int v : vertices)
    {
        if (v >= 0)
        {
            converted.push_back(static_cast<Index>(v));
        }
    }
    return algebra::Simplex(converted);
}

} // namespace

FloodComplexResult computeFloodComplex(const std::vector<double> &points, size_t point_dim,
                                       size_t num_points, const FloodComplexConfig &config)
{
    FloodComplexResult result{};
    result.original_points = num_points;

    const auto total_start = std::chrono::high_resolution_clock::now();
    validateFloodConfig(config);
    const size_t required_values = checkedRequiredPointValues(num_points, point_dim);
    if (point_dim == 0 || num_points == 0 || points.size() < required_values)
    {
        return result;
    }
    validatePointValues(points, required_values);

    const size_t effective_points = std::min(num_points, points.size() / point_dim);
    const Size max_dim = std::max<Size>(0, config.max_dim);
    const double radius_limit = config.max_radius;

    const auto subset_start = std::chrono::high_resolution_clock::now();
    const size_t min_subset =
        std::min(effective_points, std::max(FLOOD_MIN_SUBSET_POINTS, point_dim + 1));
    size_t subset_size =
        static_cast<size_t>(std::ceil(config.subset_ratio * static_cast<double>(effective_points)));
    subset_size = std::max(min_subset, subset_size);
    if (config.max_subset_size > 0)
    {
        subset_size = std::min(subset_size, config.max_subset_size);
    }
    subset_size = std::min(subset_size, effective_points);

    std::vector<int> subset_indices;
    if (subset_size == effective_points)
    {
        subset_indices.resize(effective_points);
        std::iota(subset_indices.begin(), subset_indices.end(), 0);
    }
    else
    {
        subset_indices = farthestPointSampling(points, point_dim, effective_points, subset_size);
    }
    result.subset_points = subset_indices.size();
    const auto subset_end = std::chrono::high_resolution_clock::now();
    result.subset_selection_time_ms =
        std::chrono::duration<double, std::milli>(subset_end - subset_start).count();
    if (subset_indices.empty())
    {
        result.total_time_ms = result.subset_selection_time_ms;
        return result;
    }

    const auto delaunay_start = std::chrono::high_resolution_clock::now();
    std::vector<IndexedPoint> subset_points;
    subset_points.reserve(subset_indices.size());
    for (const int idx : subset_indices)
    {
        subset_points.emplace_back(
            std::vector<double>(points.begin() + static_cast<std::ptrdiff_t>(idx * point_dim),
                                points.begin() +
                                    static_cast<std::ptrdiff_t>((idx + 1) * point_dim)),
            idx);
    }

    std::vector<std::vector<int>> top_simplices;
    /*
     * Flood complex filtration assembly:
     * - Build top simplices from 3D Delaunay tetrahedra on the sampled subset.
     * - Assign filtration values from geometric circumradius.
     * - Enforce closure so every accepted simplex contributes its full face lattice.
     */
    if (point_dim == 3 && subset_points.size() >= FLOOD_MIN_SUBSET_POINTS)
    {
        Delaunay3D triangulation;
        const auto tetrahedra = triangulation.compute(subset_points);
        top_simplices.reserve(tetrahedra.size());
        for (const auto &tet : tetrahedra)
        {
            top_simplices.push_back({
                subset_points[static_cast<size_t>(tet.v[0])].original_index,
                subset_points[static_cast<size_t>(tet.v[1])].original_index,
                subset_points[static_cast<size_t>(tet.v[2])].original_index,
                subset_points[static_cast<size_t>(tet.v[3])].original_index,
            });
        }
    }
    const auto delaunay_end = std::chrono::high_resolution_clock::now();
    result.delaunay_time_ms =
        std::chrono::duration<double, std::milli>(delaunay_end - delaunay_start).count();

    const auto flooding_start = std::chrono::high_resolution_clock::now();
    const std::vector<int> candidates =
        config.use_flooding ? floodCandidates(effective_points) : std::vector<int>{};

    FiltrationMap filtrations;
    filtrations.reserve(subset_indices.size() * 8);
    for (const int vertex : subset_indices)
    {
        insertWithClosure(algebra::Simplex({static_cast<Index>(vertex)}), 0.0, max_dim,
                          &filtrations);
    }

    for (const auto &simplex_vertices : top_simplices)
    {
        const double circumradius = simplexCircumradius(simplex_vertices, points, point_dim);
        if (!std::isfinite(circumradius) || circumradius > radius_limit + config.flooding_tolerance)
        {
            continue;
        }
        if (config.use_flooding && !candidates.empty() &&
            !isSimplexFlooded(simplex_vertices, candidates, points, point_dim, radius_limit))
        {
            continue;
        }
        insertWithClosure(makeSimplex(simplex_vertices), circumradius, max_dim, &filtrations);
    }

    if (top_simplices.empty() && subset_indices.size() >= 2)
    {
        const size_t edge_limit = std::min(subset_indices.size(), FLOOD_COMPLETE_GRAPH_LIMIT);
        for (size_t i = 0; i < edge_limit; ++i)
        {
            for (size_t j = i + 1; j < edge_limit; ++j)
            {
                std::vector<int> edge{subset_indices[i], subset_indices[j]};
                const double edge_radius = simplexCircumradius(edge, points, point_dim);
                if (!std::isfinite(edge_radius) || edge_radius > radius_limit + FLOOD_EPS)
                {
                    continue;
                }
                insertWithClosure(makeSimplex(edge), edge_radius, max_dim, &filtrations);
            }
        }
    }
    const auto flooding_end = std::chrono::high_resolution_clock::now();
    result.flooding_time_ms =
        std::chrono::duration<double, std::milli>(flooding_end - flooding_start).count();

    result.num_simplices = filtrations.size();
    if (filtrations.empty())
    {
        const auto total_end = std::chrono::high_resolution_clock::now();
        result.total_time_ms =
            std::chrono::duration<double, std::milli>(total_end - total_start).count();
        return result;
    }

    const auto persistence_start = std::chrono::high_resolution_clock::now();
    algebra::SimplicialComplex complex;
    for (const auto &[simplex, filtration] : filtrations)
    {
        complex.addSimplexWithFiltration(simplex, filtration);
    }
    auto exact = computeExactPersistenceZ2(complex, max_dim);
    result.pairs = std::move(exact.pairs);
    const auto persistence_end = std::chrono::high_resolution_clock::now();
    result.persistence_time_ms =
        std::chrono::duration<double, std::milli>(persistence_end - persistence_start).count();

    const auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start).count();
    result.simplex_reduction_ratio =
        static_cast<double>(result.num_simplices) / estimateVrSimplices(effective_points, max_dim);
    result.estimated_approximation_error =
        1.0 - static_cast<double>(result.subset_points) / static_cast<double>(effective_points);
    result.estimated_approximation_error =
        std::clamp(result.estimated_approximation_error, 0.0, 1.0);

    return result;
}

FloodComplexConfig getOptimalFloodConfig(size_t num_points, size_t point_dim)
{
    FloodComplexConfig config;
    config.max_dim = point_dim >= 3 ? 2 : 1;
    config.max_radius = 1.0;
    config.use_flooding = true;
    config.flooding_tolerance = 1e-6;

    if (num_points >= 1'000'000)
    {
        config.subset_ratio = 0.003;
        config.max_subset_size = 20000;
    }
    else if (num_points >= 250'000)
    {
        config.subset_ratio = 0.01;
        config.max_subset_size = 15000;
    }
    else if (num_points >= 50'000)
    {
        config.subset_ratio = 0.03;
        config.max_subset_size = 10000;
    }
    else
    {
        config.subset_ratio = 0.08;
        config.max_subset_size = 6000;
    }
    return config;
}

size_t estimateFloodComplexMemory(size_t num_points, size_t point_dim,
                                  const FloodComplexConfig &config)
{
    validateFloodConfig(config);
    if (num_points == 0 || point_dim == 0)
    {
        return 0;
    }

    size_t subset_points =
        static_cast<size_t>(std::ceil(config.subset_ratio * static_cast<double>(num_points)));
    subset_points = std::max(subset_points, std::min(num_points, FLOOD_MIN_SUBSET_POINTS));
    if (config.max_subset_size > 0)
    {
        subset_points = std::min(subset_points, config.max_subset_size);
    }
    subset_points = std::min(subset_points, num_points);

    const size_t point_bytes = subset_points * point_dim * sizeof(double);
    const size_t delaunay_tets = std::max<size_t>(subset_points, 1) * 6;
    const size_t simplex_estimate = delaunay_tets * 15;
    const size_t boundary_bytes = simplex_estimate * 12 * sizeof(Index);
    const size_t aux_bytes = simplex_estimate * (sizeof(double) + sizeof(Index) * 2);
    return point_bytes + boundary_bytes + aux_bytes;
}

} // namespace nerve::persistence
