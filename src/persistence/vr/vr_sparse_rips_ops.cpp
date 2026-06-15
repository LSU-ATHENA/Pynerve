#include "nerve/algebra/complex.hpp"
#include "nerve/persistence/core/flood_complex.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"
#include "nerve/persistence/vr/vr_sparse_rips_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nerve::persistence
{

constexpr int SPARSE_RIPS_DEFAULT_MAX_DIM = 2;
constexpr double SPARSE_RIPS_DEFAULT_MAX_RADIUS = 1.0;
constexpr size_t SPARSE_RIPS_SMALL_DATASET_THRESHOLD = 10000;
constexpr size_t SPARSE_RIPS_MEDIUM_DATASET_THRESHOLD = 100000;
constexpr size_t SPARSE_RIPS_PRACTICAL_THRESHOLD = 1000;
constexpr size_t SPARSE_RIPS_OVERHEAD_THRESHOLD = 20;
constexpr double SPARSE_RIPS_EPSILON_FAST = 0.1;       // 10% error
constexpr double SPARSE_RIPS_EPSILON_BALANCED = 0.2;   // 20% error
constexpr double SPARSE_RIPS_EPSILON_AGGRESSIVE = 0.3; // 30% error
constexpr double SPARSE_RIPS_MIN_EPSILON = 0.05;
constexpr double SPARSE_RIPS_SPEEDUP_SMALL = 2.0;
constexpr double SPARSE_RIPS_SPEEDUP_DIVISOR = 10.0;

namespace
{

constexpr double SPARSE_RIPS_MAX_FINITE_ESTIMATE = 1.0e300;

double finiteNonnegativeEstimate(long double value)
{
    if (!std::isfinite(value))
    {
        return SPARSE_RIPS_MAX_FINITE_ESTIMATE;
    }
    if (value <= 0.0L)
    {
        return 0.0;
    }
    return static_cast<double>(
        std::min(value, static_cast<long double>(SPARSE_RIPS_MAX_FINITE_ESTIMATE)));
}

double denseVrSimplexEstimate(size_t num_points)
{
    return finiteNonnegativeEstimate(std::pow(2.0L, static_cast<long double>(num_points)));
}

double sparseRipsSimplexEstimate(size_t num_points, double epsilon)
{
    const long double inv_epsilon = 1.0L / static_cast<long double>(epsilon);
    return finiteNonnegativeEstimate(static_cast<long double>(num_points) * inv_epsilon *
                                     inv_epsilon);
}

bool hasValidPointBuffer(const std::vector<double> &points, size_t point_dim, size_t num_points)
{
    if (point_dim == 0 || num_points == 0)
    {
        return false;
    }
    if (num_points > std::numeric_limits<size_t>::max() / point_dim)
    {
        return false;
    }
    return points.size() >= num_points * point_dim;
}

bool hasValidEpsilon(double epsilon)
{
    return std::isfinite(epsilon) && epsilon >= SPARSE_RIPS_MIN_EPSILON;
}

void validateSparseRipsConfig(const SparseRipsConfig &config)
{
    if (!hasValidEpsilon(config.epsilon))
    {
        throw std::invalid_argument("epsilon must be finite and at least 0.05");
    }
    if (!std::isfinite(config.max_radius) || config.max_radius <= 0.0)
    {
        throw std::invalid_argument("max_radius must be finite and positive");
    }
}

// A point in the net with insertion time and parent
struct NetPoint
{
    int original_index;
    double insertion_time; // When this point was added to the net
    int parent;            // Parent in the net hierarchy
    std::vector<int> children;
    double weight; // Weight for filtration value computation
};

// Greedy permutation (farthest-point sampling) with net structure
// This is the foundation of sparse Rips
class GreedyPermutation
{
public:
    std::vector<NetPoint> compute(const std::vector<double> &points, size_t point_dim,
                                  size_t num_points)
    {
        std::vector<NetPoint> net;
        net.reserve(num_points);

        // Distance to nearest net point for each point
        std::vector<double> min_dists(num_points, std::numeric_limits<double>::infinity());
        std::vector<int> parents(num_points, -1);

        // Start with point 0
        int first = 0;
        min_dists[first] = 0;

        for (size_t i = 0; i < num_points; ++i)
        {
            // Find farthest point
            int farthest = -1;
            double max_dist = -1;

            for (size_t j = 0; j < num_points; ++j)
            {
                if (min_dists[j] > max_dist)
                {
                    max_dist = min_dists[j];
                    farthest = static_cast<int>(j);
                }
            }

            if (farthest < 0)
                break;

            // Add to net
            NetPoint np;
            np.original_index = farthest;
            np.insertion_time = max_dist; // Distance to previous net
            np.parent = parents[farthest];
            np.weight = np.insertion_time;

            net.push_back(np);

            // Update distances
            const double *new_pt = &points[farthest * point_dim];

            for (size_t j = 0; j < num_points; ++j)
            {
                if (min_dists[j] == 0)
                    continue; // Already in net

                const double *pt = &points[j * point_dim];
                double dist_sq = 0.0;
                for (size_t d = 0; d < point_dim; ++d)
                {
                    double diff = pt[d] - new_pt[d];
                    dist_sq += diff * diff;
                }

                double dist = std::sqrt(dist_sq);
                if (dist < min_dists[j])
                {
                    min_dists[j] = dist;
                    parents[j] = farthest;
                }
            }

            min_dists[farthest] = 0; // Mark as in net
        }

        // Build parent-child relationships
        for (size_t i = 0; i < net.size(); ++i)
        {
            if (net[i].parent >= 0)
            {
                for (size_t j = 0; j < i; ++j)
                {
                    if (net[j].original_index == net[i].parent)
                    {
                        net[j].children.push_back(static_cast<int>(i));
                        break;
                    }
                }
            }
        }

        return net;
    }
};

// Build sparse Rips filtration using the net
// Only adds simplices when vertices become "close enough"
class SparseRipsBuilder
{
public:
    struct Simplex
    {
        std::vector<int> vertices;
        double filtration_value;
        int dimension;
    };

    std::vector<Simplex> buildFiltration(const std::vector<double> &points, size_t point_dim,
                                         const std::vector<NetPoint> &net, double epsilon,
                                         double max_radius)
    {
        std::vector<Simplex> simplices;

        // Add vertices (always included)
        for (const auto &np : net)
        {
            Simplex s;
            s.vertices = {np.original_index};
            s.filtration_value = 0.0;
            s.dimension = 0;
            simplices.push_back(s);
        }

        // Add edges (1-simplices)
        // Only add edge (i,j) if they're "close" at scale determined by epsilon
        for (size_t i = 0; i < net.size(); ++i)
        {
            for (size_t j = i + 1; j < net.size(); ++j)
            {
                const auto &p1 = net[i];
                const auto &p2 = net[j];

                // Compute distance
                const double *pt1 = &points[p1.original_index * point_dim];
                const double *pt2 = &points[p2.original_index * point_dim];

                double dist_sq = 0.0;
                for (size_t d = 0; d < point_dim; ++d)
                {
                    double diff = pt1[d] - pt2[d];
                    dist_sq += diff * diff;
                }
                double dist = std::sqrt(dist_sq);

                if (dist > max_radius)
                    continue;

                // Sparse condition: add edge when both points are "old enough"
                double scale = std::max(p1.insertion_time, p2.insertion_time);
                double threshold = (1.0 + epsilon) * scale;

                if (dist <= threshold)
                {
                    Simplex s;
                    s.vertices = {p1.original_index, p2.original_index};
                    s.filtration_value = dist;
                    s.dimension = 1;
                    simplices.push_back(s);
                }
            }
        }

        // Add higher-dimensional simplices (cliques in the 1-skeleton)
        // Uses greedy clique expansion from edges

        if (max_radius > 0)
        {
            // Build adjacency from edges
            std::unordered_map<int, std::unordered_set<int>> adjacency;
            for (const auto &s : simplices)
            {
                if (s.dimension == 1)
                {
                    adjacency[s.vertices[0]].insert(s.vertices[1]);
                    adjacency[s.vertices[1]].insert(s.vertices[0]);
                }
            }

            // Find triangles
            for (const auto &[v, neighbors] : adjacency)
            {
                std::vector<int> neighs(neighbors.begin(), neighbors.end());

                for (size_t i = 0; i < neighs.size(); ++i)
                {
                    for (size_t j = i + 1; j < neighs.size(); ++j)
                    {
                        if (adjacency[neighs[i]].count(neighs[j]))
                        {
                            // Triangle found
                            Simplex s;
                            s.vertices = {v, neighs[i], neighs[j]};

                            // Filtration value is max edge
                            const double *p0 = &points[v * point_dim];
                            const double *p1 = &points[neighs[i] * point_dim];
                            const double *p2 = &points[neighs[j] * point_dim];

                            double max_edge = 0.0;
                            auto dist_sq = [&](const double *a, const double *b) {
                                double s = 0.0;
                                for (size_t d = 0; d < point_dim; ++d)
                                {
                                    double diff = a[d] - b[d];
                                    s += diff * diff;
                                }
                                return s;
                            };

                            max_edge = std::max(max_edge, std::sqrt(dist_sq(p0, p1)));
                            max_edge = std::max(max_edge, std::sqrt(dist_sq(p0, p2)));
                            max_edge = std::max(max_edge, std::sqrt(dist_sq(p1, p2)));

                            s.filtration_value = max_edge;
                            s.dimension = 2;
                            simplices.push_back(s);
                        }
                    }
                }
            }
        }

        // Sort by filtration value
        std::ranges::sort(simplices, {}, &Simplex::filtration_value);

        return simplices;
    }
};

} // namespace

SparseRipsResult computeSparseRips(const std::vector<double> &points, size_t point_dim,
                                   size_t num_points, const SparseRipsConfig &config)
{
    SparseRipsResult result;
    validateSparseRipsConfig(config);
    result.epsilon = config.epsilon;
    result.original_points = num_points;

    if (!hasValidPointBuffer(points, point_dim, num_points))
    {
        return result;
    }
    const size_t required_values = num_points * point_dim;
    const auto values_end =
        points.begin() + static_cast<std::vector<double>::difference_type>(required_values);
    if (!std::ranges::all_of(points.begin(), values_end,
                             [](double value) { return std::isfinite(value); }))
    {
        throw std::invalid_argument("points must contain only finite values");
    }

    result.approximation_factor = 1.0 + config.epsilon;
    result.theoretical_error_bound = config.epsilon;

    auto start_total = std::chrono::high_resolution_clock::now();

    auto start_perm = std::chrono::high_resolution_clock::now();

    GreedyPermutation perm;
    auto net = perm.compute(points, point_dim, num_points);

    auto end_perm = std::chrono::high_resolution_clock::now();
    result.permutation_time_ms =
        std::chrono::duration<double, std::milli>(end_perm - start_perm).count();

    result.net_size = net.size();

    auto start_build = std::chrono::high_resolution_clock::now();

    SparseRipsBuilder builder;
    auto simplices =
        builder.buildFiltration(points, point_dim, net, config.epsilon, config.max_radius);

    auto end_build = std::chrono::high_resolution_clock::now();
    result.build_time_ms =
        std::chrono::duration<double, std::milli>(end_build - start_build).count();

    result.num_simplices = simplices.size();

    auto start_ph = std::chrono::high_resolution_clock::now();

    // Convert to our complex format
    algebra::SimplicialComplex complex;
    for (const auto &s : simplices)
    {
        std::vector<Index> verts;
        verts.reserve(s.vertices.size());
        for (int v : s.vertices)
        {
            verts.push_back(static_cast<Index>(v));
        }
        complex.addSimplexWithFiltration(algebra::Simplex(verts), s.filtration_value);
    }

    auto exact = computeExactPersistenceZ2(complex, config.max_dim);
    result.pairs = exact.pairs;

    auto end_ph = std::chrono::high_resolution_clock::now();
    result.persistence_time_ms =
        std::chrono::duration<double, std::milli>(end_ph - start_ph).count();

    auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();

    const double dense_estimate = denseVrSimplexEstimate(num_points);
    result.sparse_ratio =
        dense_estimate > 0.0
            ? std::clamp(static_cast<double>(simplices.size()) / dense_estimate, 0.0, 1.0)
            : 0.0;
    result.compression_ratio = 1.0 - result.sparse_ratio;

    result.approximation_factor = 1.0 + config.epsilon;
    result.theoretical_error_bound = config.epsilon;

    return result;
}

// Get optimal sparse Rips configuration
SparseRipsConfig getOptimalSparseRipsConfig(size_t num_points, size_t point_dim)
{
    SparseRipsConfig config;

    config.max_dim = SPARSE_RIPS_DEFAULT_MAX_DIM;
    config.max_radius = SPARSE_RIPS_DEFAULT_MAX_RADIUS;

    // Epsilon based on desired approximation quality
    // Smaller epsilon = better approximation, more simplices
    if (num_points < SPARSE_RIPS_SMALL_DATASET_THRESHOLD)
    {
        config.epsilon = SPARSE_RIPS_EPSILON_FAST; // 10% error, fast
    }
    else if (num_points < SPARSE_RIPS_MEDIUM_DATASET_THRESHOLD)
    {
        config.epsilon = SPARSE_RIPS_EPSILON_BALANCED; // 20% error, good balance
    }
    else
    {
        config.epsilon = SPARSE_RIPS_EPSILON_AGGRESSIVE; // 30% error, aggressive compression
    }
    if (point_dim >= 32)
    {
        config.epsilon = std::min(config.epsilon, SPARSE_RIPS_EPSILON_BALANCED);
    }
    return config;
}

// Estimate memory savings
SparseRipsSavings estimateSparseRipsSavings(size_t num_points, double epsilon)
{
    SparseRipsSavings savings;
    if (!hasValidEpsilon(epsilon))
    {
        return savings;
    }

    // Theoretical analysis
    // Dense VR: O(2^n) simplices
    // Sparse Rips: O(n) simplices

    savings.dense_simplices = denseVrSimplexEstimate(num_points);
    savings.sparse_simplices = sparseRipsSimplexEstimate(num_points, epsilon);

    if (savings.dense_simplices > 0.0)
    {
        savings.memory_reduction_ratio =
            std::clamp(1.0 - (savings.sparse_simplices / savings.dense_simplices), 0.0, 1.0);
    }

    // For practical sizes
    if (num_points <= SPARSE_RIPS_OVERHEAD_THRESHOLD)
    {
        // For small n, sparse may actually be larger due to overhead
        savings.recommended = false;
    }
    else if (num_points <= SPARSE_RIPS_PRACTICAL_THRESHOLD)
    {
        savings.recommended = (epsilon >= SPARSE_RIPS_EPSILON_FAST);
        savings.expected_speedup = SPARSE_RIPS_SPEEDUP_SMALL;
    }
    else
    {
        savings.recommended = true;
        savings.expected_speedup = static_cast<double>(num_points) / SPARSE_RIPS_SPEEDUP_DIVISOR;
    }

    return savings;
}

// Check if sparse Rips should be used
bool shouldUseSparseRips(size_t num_points, double epsilon)
{
    return num_points >= SPARSE_RIPS_PRACTICAL_THRESHOLD && hasValidEpsilon(epsilon);
}

} // namespace nerve::persistence
