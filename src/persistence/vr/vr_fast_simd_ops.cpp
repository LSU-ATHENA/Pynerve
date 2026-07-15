#include "nerve/algebra/complex.hpp"
#include "nerve/persistence/utils/exact_engine_fast.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/persistence/vr/vr_fast_simd_ops.hpp"
#include "nerve/platform.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __AVX512F__
#include "nerve/cpu/x86_intrinsics.hpp"
#endif
namespace nerve::persistence
{
namespace
{

// Cache-friendly data structures for small point sets
struct alignas(64) PointCache
{
    static constexpr size_t MAX_POINTS = 1024;
    static constexpr size_t MAX_DIM = 16;
    static constexpr size_t COORDS_SIZE = MAX_POINTS * MAX_DIM;
    double stack_coords[COORDS_SIZE];
    std::vector<double> heap_coords;
    double *coords = stack_coords;
    size_t n_points;
    size_t point_dim;

    void load(const std::vector<double> &points, size_t dim)
    {
        n_points = points.size() / dim;
        point_dim = dim;
        if (points.size() > COORDS_SIZE)
        {
            heap_coords.assign(points.begin(), points.end());
            coords = heap_coords.data();
        }
        else
        {
            std::copy(points.begin(), points.end(), stack_coords);
            coords = stack_coords;
        }
    }

    const double *getPoint(size_t idx) const { return &coords[idx * point_dim]; }
};

// AVX-512 distance calculation for small point sets with prefetching
#ifdef __AVX512F__
inline double euclideanDistanceAvx512(const double *p1, const double *p2, size_t dim)
{
    __m512d sum_vec = _mm512_setzero_pd();
    size_t i = 0;

    // Prefetch next cache line
    nerve_prefetch_read(p1 + 32, NervePrefetchLevel::L1);
    nerve_prefetch_read(p2 + 32, NervePrefetchLevel::L1);

    // Process 8 doubles at a time with FMA
    for (; i + 8 <= dim; i += 8)
    {
        __m512d a = _mm512_loadu_pd(p1 + i);
        __m512d b = _mm512_loadu_pd(p2 + i);
        __m512d diff = _mm512_sub_pd(a, b);
        // FMA: sum += diff * diff (single instruction, better precision)
        sum_vec = _mm512_fmadd_pd(diff, diff, sum_vec);
    }

    double sum = _mm512_reduce_add_pd(sum_vec);

    // Handle remaining elements with scalar FMA
    for (; i < dim; ++i)
    {
        double diff = p1[i] - p2[i];
        sum = std::fma(diff, diff, sum); // FMA instead of multiply-add
    }

    return std::sqrt(sum);
}
#endif

// Scalar implementation distance calculation with FMA
inline double euclideanDistanceScalar(const double *p1, const double *p2, size_t dim)
{
    double sum = 0.0;
    // Prefetch next cache line
    nerve_prefetch_read(p1 + 8, NervePrefetchLevel::L1);
    nerve_prefetch_read(p2 + 8, NervePrefetchLevel::L1);

    for (size_t i = 0; i < dim; ++i)
    {
        double diff = p1[i] - p2[i];
        sum = std::fma(diff, diff, sum); // FMA: better precision and performance
    }
    return std::sqrt(sum);
}

bool canUseAvx512Distance()
{
#if defined(__AVX512F__) &&                                                                        \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
    static const bool available = nerve::cpu::CpuFeatureFlags::detect().has_avx512f;
    return available;
#else
    return false;
#endif
}

// Select best distance function based on CPU capabilities
double euclideanDistanceOptimized(const double *p1, const double *p2, size_t dim)
{
#ifdef __AVX512F__
    if (dim >= 8 && canUseAvx512Distance())
    {
        return euclideanDistanceAvx512(p1, p2, dim);
    }
#endif
    return euclideanDistanceScalar(p1, p2, dim);
}

// Optimized edge key for small point sets (32-bit indices)
using EdgeKey32 = uint32_t;

inline EdgeKey32 makeEdgeKey32(int a, int b)
{
    if (a > b)
        std::swap(a, b);
    return (static_cast<uint32_t>(a) << 16) | static_cast<uint32_t>(b);
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

bool isValidFastSimdInput(core::BufferView<const double> points, Size point_dim,
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

// Pre-allocated adjacency structure for cache efficiency
struct FastAdjacency
{
    static constexpr size_t MAX_NEIGHBORS = 64; // Typical max for small point sets
    int neighbors[PointCache::MAX_POINTS][MAX_NEIGHBORS];
    uint8_t neighbor_counts[PointCache::MAX_POINTS];
    bool overflowed_;

    void clear(size_t n_points)
    {
        std::fill(neighbor_counts, neighbor_counts + n_points, 0);
        overflowed_ = false;
    }

    bool addEdge(int a, int b)
    {
        if (a < 0 || static_cast<size_t>(a) >= PointCache::MAX_POINTS || b < 0 ||
            static_cast<size_t>(b) >= PointCache::MAX_POINTS ||
            neighbor_counts[a] >= MAX_NEIGHBORS || neighbor_counts[b] >= MAX_NEIGHBORS)
        {
            overflowed_ = true;
            return false;
        }
        neighbors[a][neighbor_counts[a]++] = b;
        neighbors[b][neighbor_counts[b]++] = a;
        return true;
    }

    void sortNeighbors(size_t n_points)
    {
        for (size_t i = 0; i < n_points; ++i)
        {
            std::ranges::sort(std::span(neighbors[i], neighbor_counts[i]));
        }
    }

    bool overflowed() const { return overflowed_; }
};

// Fixed-size clique enumerator for small graphs.
class FastCliqueEnumerator
{
public:
    FastCliqueEnumerator(const FastAdjacency &adj,
                         const std::unordered_map<EdgeKey32, double> &weights, size_t max_dim,
                         double max_radius)
        : adj_(adj)
        , weights_(weights)
        , max_dim_(max_dim)
        , max_radius_(max_radius)
        , result_complex_(nullptr)
        , seen_(nullptr)
    {}

    void enumerate(size_t n_points, algebra::SimplicialComplex &complex, SimplexSet &seen)
    {
        result_complex_ = &complex;
        seen_ = &seen;

        // Add vertices (dimension 0)
        for (size_t i = 0; i < n_points; ++i)
        {
            std::vector<Index> v{static_cast<Index>(i)};
            complex.addSimplexWithFiltration(algebra::Simplex(v), 0.0);
        }

        // Add edges (dimension 1)
        for (size_t i = 0; i < n_points; ++i)
        {
            for (uint8_t ni = 0; ni < adj_.neighbor_counts[i]; ++ni)
            {
                int j = adj_.neighbors[i][ni];
                if (static_cast<size_t>(j) > i)
                {
                    addEdge(static_cast<int>(i), j);
                }
            }
        }

        // To compute H_k correctly, include simplices up to dimension k + 1.
        if (n_points >= 3)
        {
            const size_t max_simplex_size = max_dim_ > n_points - 2 ? n_points : max_dim_ + 2;
            for (size_t simplex_size = 3; simplex_size <= max_simplex_size; ++simplex_size)
            {
                for (size_t i = 0; i < n_points; ++i)
                {
                    std::vector<int> current{static_cast<int>(i)};
                    std::vector<int> candidates;
                    candidates.reserve(adj_.neighbor_counts[i]);
                    for (uint8_t ni = 0; ni < adj_.neighbor_counts[i]; ++ni)
                    {
                        int j = adj_.neighbors[i][ni];
                        if (j > static_cast<int>(i))
                        {
                            candidates.push_back(j);
                        }
                    }
                    expandCliques(current, candidates, simplex_size);
                }
            }
        }
    }

private:
    const FastAdjacency &adj_;
    const std::unordered_map<EdgeKey32, double> &weights_;
    size_t max_dim_;
    double max_radius_;
    algebra::SimplicialComplex *result_complex_;
    SimplexSet *seen_;

    void addEdge(int a, int b)
    {
        auto it = weights_.find(makeEdgeKey32(a, b));
        if (it != weights_.end() && it->second <= max_radius_)
        {
            std::vector<Index> verts{static_cast<Index>(a), static_cast<Index>(b)};
            result_complex_->addSimplexWithFiltration(algebra::Simplex(verts), it->second);
        }
    }

    double simplexFiltration(const std::vector<int> &verts)
    {
        double w = 0.0;
        for (size_t i = 0; i < verts.size(); ++i)
        {
            for (size_t j = i + 1; j < verts.size(); ++j)
            {
                auto it = weights_.find(makeEdgeKey32(verts[i], verts[j]));
                if (it == weights_.end())
                {
                    return std::numeric_limits<double>::infinity();
                }
                w = std::max(w, it->second);
            }
        }
        return w;
    }

    void expandCliques(std::vector<int> &current, std::vector<int> &candidates, size_t target_size)
    {
        if (current.size() == target_size)
        {
            std::vector<int> key = current;
            std::ranges::sort(key);
            if (seen_->insert(key).second)
            {
                double filt = simplexFiltration(current);
                if (std::isfinite(filt) && filt <= max_radius_)
                {
                    std::vector<Index> verts;
                    verts.reserve(current.size());
                    for (int v : current)
                    {
                        verts.push_back(static_cast<Index>(v));
                    }
                    result_complex_->addSimplexWithFiltration(algebra::Simplex(verts), filt);
                }
            }
            return;
        }

        while (!candidates.empty())
        {
            int v = candidates.back();
            candidates.pop_back();

            if (v < 0 || static_cast<size_t>(v) >= PointCache::MAX_POINTS)
                continue;

            std::vector<int> new_candidates;
            new_candidates.reserve(candidates.size());

            const auto &v_neighbors = adj_.neighbors[v];
            uint8_t v_count = adj_.neighbor_counts[v];

            for (int u : candidates)
            {
                if (u < 0 || static_cast<size_t>(u) >= PointCache::MAX_POINTS)
                    continue;

                // Binary search in v's neighbors (since they're sorted)
                if (std::binary_search(v_neighbors, v_neighbors + v_count, u))
                {
                    new_candidates.push_back(u);
                }
            }

            current.push_back(v);
            expandCliques(current, new_candidates, target_size);
            current.pop_back();
        }
    }
};

} // namespace

// Public API: Optimized VR for small point sets (< 1K points)
std::vector<Pair> computeVrPersistenceFastSimd(core::BufferView<const double> points,
                                               Size point_dim, const VRConfig &config)
{
    if (!isValidFastSimdInput(points, point_dim, config))
    {
        return {};
    }

    const Size num_points = points.size() / point_dim;

    // Fall back to standard implementation for larger point sets
    if (num_points > PointCache::MAX_POINTS || point_dim > PointCache::MAX_DIM)
    {
        return computeVrPersistenceFast(points, point_dim, config);
    }

    // Load points into cache-friendly structure
    PointCache cache;
    std::vector<double> point_data(points.begin(), points.end());
    cache.load(point_data, point_dim);

    // Build radius graph with the best distance kernel available at runtime.
    FastAdjacency adj;
    adj.clear(num_points);

    std::unordered_map<EdgeKey32, double> edge_weights;
    edge_weights.reserve(num_points * 4);

    for (Size i = 0; i < num_points; ++i)
    {
        const double *p1 = cache.getPoint(i);
        for (Size j = i + 1; j < num_points; ++j)
        {
            const double *p2 = cache.getPoint(j);
            double d = euclideanDistanceOptimized(p1, p2, point_dim);
            if (std::isfinite(d) && d <= config.max_radius)
            {
                adj.addEdge(static_cast<int>(i), static_cast<int>(j));
                edge_weights[makeEdgeKey32(static_cast<int>(i), static_cast<int>(j))] = d;
            }
        }
    }
    if (adj.overflowed())
    {
        VRConfig exact_config = config;
        exact_config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
        return computeVrPersistenceFast(points, point_dim, exact_config);
    }

    adj.sortNeighbors(num_points);

    std::vector<std::vector<int>> neighbors(num_points);
    std::unordered_map<std::uint64_t, double> ew64;
    ew64.reserve(edge_weights.size());
    for (Size i = 0; i < num_points; ++i)
        neighbors[i].assign(adj.neighbors[i], adj.neighbors[i] + adj.neighbor_counts[i]);
    for (const auto &[k32, dist] : edge_weights)
    {
        int a = (int)(k32 >> 16), b = (int)(k32 & 0xFFFF);
        if (a > b)
            std::swap(a, b);
        ew64[((std::uint64_t)(std::uint32_t)a << 32) | (std::uint32_t)b] = dist;
    }
    auto exact = computeExactCohomologyZ2Fast((int)num_points, (int)config.max_dim,
                                              config.max_radius, neighbors, ew64);
    const auto &diagram = exact.pairs;
    std::vector<Pair> pairs;
    pairs.reserve(diagram.size());
    for (const auto &pair : diagram)
        if (pair.dimension <= (Dimension)config.max_dim)
            pairs.push_back(pair);
    std::ranges::sort(pairs, {},
                      [](const Pair &p) { return std::tuple(p.dimension, p.birth, p.death); });
    return pairs;
}

bool isAvx512Available()
{
    return canUseAvx512Distance();
}
size_t getOptimalSimdBlockSize(size_t point_dim)
{
    if (point_dim == 0)
        return 0;
    return point_dim <= 4 ? (isAvx512Available() ? 256 : 128) : (isAvx512Available() ? 128 : 64);
}

} // namespace nerve::persistence
