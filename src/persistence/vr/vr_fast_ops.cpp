#include "nerve/algebra/complex.hpp"
#include "nerve/persistence/detail/fast_vr_accelerated_detail.hpp"
#include "nerve/persistence/utils/backend_selector.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"
#include "nerve/persistence/utils/exact_engine_fast.hpp"
#include "nerve/persistence/vr/detail/vr_detail.hpp"
#include "nerve/persistence/vr/vr_cohomology_ops.hpp"
#include "nerve/persistence/vr/vr_dispatch_ops.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/persistence/vr/vr_fast_simd_ops.hpp"
#include "nerve/persistence/vr/vr_large_witness_ops.hpp"
#include "nerve/persistence/vr/vr_medium_hybrid_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef BUILD_CUDA
#include <cuda_runtime_api.h>
#endif
namespace nerve::persistence
{
namespace
{
constexpr size_t FAST_SIMD_DISPATCH_MAX_POINTS = 1024;
constexpr size_t FAST_SIMD_DISPATCH_MAX_DIM = 16;
constexpr size_t MEDIUM_HYBRID_DISPATCH_MIN_POINTS = 512;
constexpr size_t MEDIUM_HYBRID_DISPATCH_MAX_POINTS = 10000;
constexpr size_t LARGE_WITNESS_DISPATCH_MIN_POINTS = 5000;
struct SimplexKey
{
    std::vector<int> verts;
    bool operator==(const SimplexKey &other) const noexcept { return verts == other.verts; }
};
struct SimplexKeyHash
{
    std::size_t operator()(const SimplexKey &key) const noexcept
    {
        std::size_t h = 0;
        for (int v : key.verts)
        {
            h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};
using EdgeKey = std::uint64_t;
using EdgeWeights = std::unordered_map<EdgeKey, double>;
EdgeKey makeEdgeKey(int a, int b)
{
    if (a > b)
    {
        std::swap(a, b);
    }
    return (static_cast<EdgeKey>(static_cast<std::uint32_t>(a)) << 32) |
           static_cast<EdgeKey>(static_cast<std::uint32_t>(b));
}
double euclideanDistance(const std::vector<double> &points, Size point_dim, Size i, Size j)
{
    double s = 0.0;
    for (Size d = 0; d < point_dim; ++d)
    {
        const double diff = points[i * point_dim + d] - points[j * point_dim + d];
        const double contribution = diff * diff;
        if (!std::isfinite(contribution) || s > std::numeric_limits<double>::max() - contribution)
        {
            return std::numeric_limits<double>::infinity();
        }
        s += contribution;
    }
    const double distance = std::sqrt(s);
    return std::isfinite(distance) ? distance : std::numeric_limits<double>::infinity();
}
bool hasFiniteSafePointCoordinates(const core::BufferView<const double> &points, Size point_dim)
{
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
bool isValidFastVrInput(const core::BufferView<const double> &points, Size point_dim,
                        const VRConfig &config)
{
    if (point_dim == 0 || points.empty() || (points.size() % point_dim) != 0)
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
    if (num_points > std::vector<std::vector<int>>().max_size())
    {
        return false;
    }
    if (!hasFiniteSafePointCoordinates(points, point_dim))
    {
        return false;
    }
    return true;
}
void buildRadiusGraph(const std::vector<double> &points, Size point_dim, Size num_points,
                      double radius, std::vector<std::vector<int>> &neighbors,
                      EdgeWeights &edge_weights)
{
    neighbors.assign(num_points, {});
    edge_weights.clear();
    edge_weights.reserve(num_points * 4);
    for (Size i = 0; i < num_points; ++i)
    {
        for (Size j = i + 1; j < num_points; ++j)
        {
            const double d = euclideanDistance(points, point_dim, i, j);
            if (!std::isfinite(d) || d > radius)
            {
                continue;
            }
            neighbors[i].push_back(static_cast<int>(j));
            neighbors[j].push_back(static_cast<int>(i));
            edge_weights.emplace(makeEdgeKey(static_cast<int>(i), static_cast<int>(j)), d);
        }
    }
    for (auto &adjacency : neighbors)
    {
        std::ranges::sort(adjacency);
    }
}
double simplexFiltration(const std::vector<int> &verts, const EdgeWeights &edge_weights)
{
    double w = 0.0;
    for (Size i = 0; i < verts.size(); ++i)
    {
        for (Size j = i + 1; j < verts.size(); ++j)
        {
            const auto it = edge_weights.find(makeEdgeKey(verts[i], verts[j]));
            if (it == edge_weights.end())
            {
                return std::numeric_limits<double>::infinity();
            }
            const double d = it->second;
            if (d > w)
            {
                w = d;
            }
        }
    }
    return w;
}
void addSimplexIfNew(const std::vector<int> &verts, const EdgeWeights &edge_weights,
                     algebra::SimplicialComplex &complex,
                     std::unordered_set<SimplexKey, SimplexKeyHash> &seen)
{
    std::vector<int> sorted = verts;
    std::ranges::sort(sorted);
    SimplexKey key{sorted};
    if (!seen.insert(key).second)
    {
        return;
    }
    std::vector<Index> simplex_vertices;
    simplex_vertices.reserve(sorted.size());
    for (int v : sorted)
    {
        simplex_vertices.push_back(static_cast<Index>(v));
    }
    const double filt = simplexFiltration(sorted, edge_weights);
    if (!std::isfinite(filt))
    {
        return;
    }
    complex.addSimplexWithFiltration(algebra::Simplex(simplex_vertices), filt);
}
void enumerateCliquesRec(const std::vector<std::vector<int>> &neighbors,
                         const EdgeWeights &edge_weights, std::vector<int> &current,
                         std::vector<int> &candidates, Size target_size,
                         algebra::SimplicialComplex &complex,
                         std::unordered_set<SimplexKey, SimplexKeyHash> &seen)
{
    if (current.size() == target_size)
    {
        addSimplexIfNew(current, edge_weights, complex, seen);
        return;
    }
    while (!candidates.empty())
    {
        const int next = candidates.back();
        candidates.pop_back();
        std::vector<int> next_candidates;
        next_candidates.reserve(candidates.size());
        const auto &nset = neighbors[static_cast<Size>(next)];
        for (int c : candidates)
        {
            if (std::binary_search(nset.begin(), nset.end(), c))
            {
                next_candidates.push_back(c);
            }
        }
        current.push_back(next);
        enumerateCliquesRec(neighbors, edge_weights, current, next_candidates, target_size, complex,
                            seen);
        current.pop_back();
    }
}
std::vector<Pair> computeVrPersistenceExact(const core::BufferView<const double> &points,
                                            Size point_dim, const VRConfig &config)
{
    if (!isValidFastVrInput(points, point_dim, config))
    {
        return {};
    }
    std::vector<double> pointData(points.begin(), points.end());
    const Size num_points = pointData.size() / point_dim;
    if (num_points == 0)
    {
        return {};
    }
    if (num_points > static_cast<Size>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    std::vector<std::vector<int>> neighbors(num_points);
    EdgeWeights edge_weights;
    buildRadiusGraph(pointData, point_dim, num_points, config.max_radius, neighbors, edge_weights);
    algebra::SimplicialComplex complex;
    std::unordered_set<SimplexKey, SimplexKeyHash> seen;
    seen.reserve(num_points * 8);
    for (Size i = 0; i < num_points; ++i)
    {
        std::vector<Index> v{static_cast<Index>(i)};
        complex.addSimplexWithFiltration(algebra::Simplex(v), 0.0);
    }
    for (Size i = 0; i < num_points; ++i)
    {
        for (int j : neighbors[i])
        {
            if (static_cast<Size>(j) <= i)
            {
                continue;
            }
            std::vector<int> edge{static_cast<int>(i), j};
            addSimplexIfNew(edge, edge_weights, complex, seen);
        }
    }
    // For cohomology/accelerated: use fast engine (lexicographic coboundary enumeration)
    // which avoids SimplicialComplex entirely.
    bool use_cohomology = config.use_accelerated_runtime || config.use_adaptive_acceleration;
    if (use_cohomology)
    {
        auto exact = computeExactCohomologyZ2Fast(static_cast<int>(num_points),
                                                  static_cast<int>(config.max_dim),
                                                  config.max_radius, neighbors, edge_weights);
        const auto &diagram = exact.pairs;
        std::vector<Pair> pairs;
        pairs.reserve(diagram.size());
        for (const auto &pair : diagram)
        {
            if (pair.dimension <= static_cast<Dimension>(config.max_dim))
            {
                pairs.push_back(pair);
            }
        }
        std::ranges::sort(pairs, {}, &Pair::dimension);
        return pairs;
    }

    const Size max_simplex_size = std::min(num_points, config.max_dim + 2);
    for (Size simplex_size = 3; simplex_size <= max_simplex_size; ++simplex_size)
    {
        // Tetrahedra (simplex_size=4): use neighbor intersection for O(n^3), not recursion O(n^4)
        if (simplex_size == 4)
        {
            for (Size a = 0; a < num_points; ++a)
            {
                for (int b : neighbors[a])
                {
                    if ((Size)b <= a)
                        continue;
                    double dab = 0.0;
                    {
                        auto it = edge_weights.find(makeEdgeKey((int)a, b));
                        if (it == edge_weights.end())
                            continue;
                        dab = it->second;
                    }
                    for (int c : neighbors[b])
                    {
                        if ((Size)c <= (Size)b)
                            continue;
                        auto it_ac = edge_weights.find(makeEdgeKey((int)a, c));
                        if (it_ac == edge_weights.end())
                            continue;
                        double dmax = std::max({dab, it_ac->second});
                        auto it_bc = edge_weights.find(makeEdgeKey((int)b, c));
                        if (it_bc == edge_weights.end())
                            continue;
                        dmax = std::max(dmax, it_bc->second);
                        // Find d > c that connects to a, b, c
                        for (int d : neighbors[a])
                        {
                            if ((Size)d <= (Size)c)
                                continue;
                            if (std::find(neighbors[b].begin(), neighbors[b].end(), d) ==
                                neighbors[b].end())
                                continue;
                            if (std::find(neighbors[c].begin(), neighbors[c].end(), d) ==
                                neighbors[c].end())
                                continue;
                            double dd = dmax;
                            auto it_d = edge_weights.find(makeEdgeKey((int)a, d));
                            if (it_d != edge_weights.end())
                                dd = std::max(dd, it_d->second);
                            it_d = edge_weights.find(makeEdgeKey((int)b, d));
                            if (it_d != edge_weights.end())
                                dd = std::max(dd, it_d->second);
                            it_d = edge_weights.find(makeEdgeKey((int)c, d));
                            if (it_d != edge_weights.end())
                                dd = std::max(dd, it_d->second);
                            if (!std::isfinite(dd))
                                continue;
                            std::vector<int> verts = {(int)a, b, c, d};
                            std::ranges::sort(verts);
                            SimplexKey key{verts};
                            if (seen.insert(key).second)
                            {
                                std::vector<Index> iv;
                                for (int v : verts)
                                    iv.push_back(v);
                                complex.addSimplexWithFiltration(algebra::Simplex(iv), dd);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            for (Size i = 0; i < num_points; ++i)
            {
                std::vector<int> current{static_cast<int>(i)};
                std::vector<int> candidates;
                candidates.reserve(neighbors[i].size());
                for (int c : neighbors[i])
                {
                    if (c > static_cast<int>(i))
                    {
                        candidates.push_back(c);
                    }
                }
                enumerateCliquesRec(neighbors, edge_weights, current, candidates, simplex_size,
                                    complex, seen);
            }
        }
    }
    auto exact = use_cohomology ? computeExactCohomologyZ2(complex, config.max_dim)
                                : computeExactPersistenceZ2(complex, config.max_dim);
    const auto &diagram = exact.pairs;
    std::vector<Pair> pairs;
    pairs.reserve(diagram.size());
    for (const auto &pair : diagram)
    {
        if (pair.dimension <= static_cast<Dimension>(config.max_dim))
        {
            pairs.push_back(pair);
        }
    }
    std::ranges::sort(pairs, {}, &Pair::dimension);
    return pairs;
}
} // namespace
errors::ErrorResult<std::vector<Pair>>
computeVrPersistenceAdaptiveAcceleration(const core::BufferView<const double> &points,
                                         Size point_dim, const VRConfig &config)
{
    if (!isValidFastVrInput(points, point_dim, config))
    {
        return errors::ErrorResult<std::vector<Pair>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    if (!is_cuda_available())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E10_GPU_OOM);
    }
    const bool force_adaptive_acceleration =
        config.use_accelerated_runtime || config.use_adaptive_acceleration;
    if (force_adaptive_acceleration && !isAdaptiveAccelerationAvailable())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E10_GPU_OOM);
    }
    // Route to the optimal algorithm based on dataset characteristics
    const Size num_points = points.size() / point_dim;
    auto pairs =
        computeVrPersistenceFast(points, point_dim, getOptimalFastvrConfig(num_points, point_dim));
    return errors::ErrorResult<std::vector<Pair>>::success(std::move(pairs));
}
std::vector<Pair> computeVrPersistenceFast(const core::BufferView<const double> &points,
                                           Size point_dim, const VRConfig &config)
{
    if (!isValidFastVrInput(points, point_dim, config))
    {
        return {};
    }

    const Size num_points = points.size() / point_dim;

    // Route explicit algorithm requests to their concrete implementations.
    switch (config.algorithm)
    {
        case VRAlgorithmSelection::FAST_SIMD:
            if (num_points <= FAST_SIMD_DISPATCH_MAX_POINTS &&
                point_dim <= FAST_SIMD_DISPATCH_MAX_DIM)
            {
                return computeVrPersistenceFastSimd(points, point_dim, config);
            }
            return computeVrPersistenceExact(points, point_dim, config);
        case VRAlgorithmSelection::MEDIUM_HYBRID:
            if (num_points >= MEDIUM_HYBRID_DISPATCH_MIN_POINTS &&
                num_points <= MEDIUM_HYBRID_DISPATCH_MAX_POINTS)
            {
                return computeVrPersistenceMediumHybrid(points, point_dim, config);
            }
            return computeVrPersistenceExact(points, point_dim, config);
        case VRAlgorithmSelection::LARGE_WITNESS:
            if (num_points >= LARGE_WITNESS_DISPATCH_MIN_POINTS)
            {
                return computeVrPersistenceLargeWitness(points, point_dim, config);
            }
            return computeVrPersistenceExact(points, point_dim, config);
        case VRAlgorithmSelection::EXACT_STANDARD:
            return computeVrPersistenceExact(points, point_dim, config);
        case VRAlgorithmSelection::AUTO:
        case VRAlgorithmSelection::ACCELERATED:
            break;
    }

    // Route to the explicitly selected fast pipeline.
    if (config.algorithm == VRAlgorithmSelection::ACCELERATED)
    {
        VRDispatchConfig dispatch_config;
        dispatch_config.max_dim = config.max_dim;
        dispatch_config.max_radius = config.max_radius;
        dispatch_config.num_threads =
            config.num_threads > static_cast<Size>(std::numeric_limits<int>::max())
                ? std::numeric_limits<int>::max()
                : static_cast<int>(config.num_threads);
        return computeVrPersistenceDispatch(points, point_dim, dispatch_config);
    }
    // AUTO/ACCELERATED path: SIMD for small, cohomology for medium+
    if (config.algorithm == VRAlgorithmSelection::AUTO ||
        config.algorithm == VRAlgorithmSelection::ACCELERATED)
    {
        if (num_points <= 200 && point_dim <= FAST_SIMD_DISPATCH_MAX_DIM)
        {
            return computeVrPersistenceFastSimd(points, point_dim, config);
        }
        // For >200 points, use exact homology engine (proven correct, 5/5 MATCH, 261s)
        return computeVrPersistenceExact(points, point_dim, config);
    }
    return computeVrPersistenceExact(points, point_dim, config);
}

errors::ErrorResult<std::vector<Pair>>
computeVrPersistenceFastResult(const core::BufferView<const double> &points, Size point_dim,
                               const VRConfig &config)
{
    if (!isValidFastVrInput(points, point_dim, config))
    {
        return errors::ErrorResult<std::vector<Pair>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    if (!is_cuda_available())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E10_GPU_OOM);
    }
    if ((config.use_accelerated_runtime || config.use_adaptive_acceleration) &&
        !isAdaptiveAccelerationAvailable())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E10_GPU_OOM);
    }

    // Preserve the existing algorithm selection behavior while exposing
    // a typed status result for callers that need explicit error handling.
    auto pairs = computeVrPersistenceFast(points, point_dim, config);
    return errors::ErrorResult<std::vector<Pair>>::success(std::move(pairs));
}
} // namespace nerve::persistence
