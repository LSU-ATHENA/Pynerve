// High-dimensional exact persistent homology (0-6D).
// The implementation combines union-find for H0 with cohomology and
// involution-enabled paths for higher dimensions when requested.
#include "nerve/persistence/core/high_dimensional_exact.hpp"
#include "nerve/persistence/reduction/reduction_union_find_ops.hpp"

#include <ranges>
#include <unordered_set>
// Enforce 0-6D limit at compile time where possible
constexpr int MAX_SUPPORTED_DIM = 6;
constexpr int MAX_DIM_ARRAY_SIZE = MAX_SUPPORTED_DIM + 1; // H0-H6
// Tolerances for high-dimensional exact persistence calculations
constexpr double H12_BIRTH_TOLERANCE = 1e-10;
constexpr double PERSISTENCE_COMPARISON_TOLERANCE = 1e-6;
// Algorithm Threshold Constants
constexpr size_t APPARENT_PAIRS_THRESHOLD = 1000; // Point count threshold for apparent pairs
constexpr double PERCENTAGE_MULTIPLIER = 100.0;   // Convert ratio to percentage
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace nerve::persistence
{

namespace
{

void validateHighDimensionalInput(const std::vector<std::vector<int>> &simplices,
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

} // namespace

// Main API: Unified high-dimensional exact computation
HighDimensionalExactResult computeHighDimensionalExact(
    const std::vector<std::vector<int>> &simplices, const std::vector<double> &filtration_values,
    const std::vector<int> &dimensions, const HighDimensionalExactConfig &config)
{
    validateHighDimensionalInput(simplices, filtration_values, dimensions);
    HighDimensionalExactResult result{};
    result.used_cohomology = config.use_cohomology;
    result.used_involution = config.use_involution;
    result.used_clearing = config.use_clearing;
    result.num_simplices = static_cast<int>(simplices.size());
    auto start_total = std::chrono::high_resolution_clock::now();
    // Split simplices by dimension for targeted processing
    // DIMENSION PRUNING: Use fixed-size arrays for 0-6D (not dynamic)
    std::vector<std::vector<int>> dim_simplices[MAX_DIM_ARRAY_SIZE]; // H0-H6
    std::vector<double> dim_filtrations[MAX_DIM_ARRAY_SIZE];
    std::vector<int> dim_indices[MAX_DIM_ARRAY_SIZE];
    for (size_t i = 0; i < simplices.size(); ++i)
    {
        int dim = dimensions[i];
        // HARD 0-6D LIMIT: Skip any simplices beyond dimension 6
        if (dim >= 0 && dim <= std::min(config.max_dim, MAX_SUPPORTED_DIM))
        {
            dim_simplices[dim].push_back(simplices[i]);
            dim_filtrations[dim].push_back(filtration_values[i]);
            dim_indices[dim].push_back(static_cast<int>(i));
        }
        // Note: Simplices with dim > 6 are discarded (dimension pruning)
    }
    // Track cleared indices for dimension cascade
    std::unordered_set<int> cleared_indices;
    result.num_columns_cleared = 0;
    result.num_apparent_pairs = 0;
    // H0: Use Union-Find (fastest possible)
    if (config.max_dim >= 0 && !dim_simplices[0].empty())
    {
        auto start_h0 = std::chrono::high_resolution_clock::now();
        // Build edge list from H0 and H1
        std::vector<std::pair<int, int>> edges;
        std::vector<double> edge_weights;
        for (size_t i = 0; i < dim_simplices[1].size(); ++i)
        {
            if (dim_simplices[1][i].size() == 2)
            {
                edges.push_back({dim_simplices[1][i][0], dim_simplices[1][i][1]});
                edge_weights.push_back(dim_filtrations[1][i]);
            }
        }
        // Get number of vertices
        int num_vertices = 0;
        for (const auto &s : dim_simplices[0])
        {
            if (!s.empty())
            {
                num_vertices = std::max(num_vertices, s[0] + 1);
            }
        }
        // H0 reduction only needs vertex cardinality and weighted edges.
        // We pass a zero-initialized coordinate buffer to satisfy the shared API.
        std::vector<double> vertex_coordinate_buffer(num_vertices, 0.0);
        auto h0_pairs = computeD0PersistenceUnionFind(vertex_coordinate_buffer, 1, num_vertices,
                                                      edges, edge_weights);
        result.pairs_h0 = h0_pairs;
        for (const auto &pair : h0_pairs)
        {
            result.pairs.push_back(pair);
        }
        // Mark vertices as cleared for H1
        for (const auto &pair : h0_pairs)
        {
            if (pair.death != std::numeric_limits<double>::infinity())
            {
                cleared_indices.insert(static_cast<int>(pair.birth));
                ++result.num_columns_cleared;
            }
        }
        auto end_h0 = std::chrono::high_resolution_clock::now();
        result.time_h0_h2_ms = std::chrono::duration<double, std::milli>(end_h0 - start_h0).count();
    }
    // H1-H2: Use Cohomology + Clearing
    if (config.max_dim >= 1)
    {
        auto start_h12 = std::chrono::high_resolution_clock::now();
        // Build combined H1-H2 simplices
        std::vector<std::vector<int>> h12_simplices;
        std::vector<double> h12_filtrations;
        std::vector<int> h12_dims;
        std::vector<int> h12_original_indices;
        for (int d = 1; d <= std::min(2, config.max_dim); ++d)
        {
            for (size_t i = 0; i < dim_simplices[d].size(); ++i)
            {
                h12_simplices.push_back(dim_simplices[d][i]);
                h12_filtrations.push_back(dim_filtrations[d][i]);
                h12_dims.push_back(d);
                h12_original_indices.push_back(dim_indices[d][i]);
            }
        }
        if (!h12_simplices.empty())
        {
            CohomologyConfig cohom_config;
            cohom_config.max_dimension = std::min(2, config.max_dim);
            cohom_config.use_cohomology = config.use_cohomology;
            cohom_config.use_clearing = config.use_clearing;
            cohom_config.use_apparent_pairs = config.use_apparent_pairs;
            cohom_config.use_dimension_cascade = true;
            auto cohom_result = computePersistentCohomology(
                h12_simplices, h12_filtrations, h12_dims, cohom_config.max_dimension, cohom_config);
            // Extract pairs by dimension
            for (const auto &pair : cohom_result.all_pairs)
            {
                result.pairs.push_back(pair);
                switch (pair.dimension)
                {
                    case 1:
                        result.pairs_h1.push_back(pair);
                        break;
                    case 2:
                        result.pairs_h2.push_back(pair);
                        break;
                }
            }
            // Update cleared count
            result.num_columns_cleared += static_cast<int>(cohom_result.num_cleared);
            // Pass clearing to H3+
            for (const auto &pair : cohom_result.all_pairs)
            {
                if (pair.death != std::numeric_limits<double>::infinity())
                {
                    // Find index
                    for (size_t i = 0; i < h12_simplices.size(); ++i)
                    {
                        if (std::abs(h12_filtrations[i] - pair.birth) < H12_BIRTH_TOLERANCE)
                        {
                            cleared_indices.insert(h12_original_indices[i]);
                            break;
                        }
                    }
                }
            }
        }
        auto end_h12 = std::chrono::high_resolution_clock::now();
        result.time_h0_h2_ms +=
            std::chrono::duration<double, std::milli>(end_h12 - start_h12).count();
    }
    // H3-H6: Use Involuted Homology (or Cohomology if involution disabled)
    if (config.max_dim >= 3)
    {
        auto start_h36 = std::chrono::high_resolution_clock::now();
        // Build combined H3-H6 simplices
        std::vector<std::vector<int>> h36_simplices;
        std::vector<double> h36_filtrations;
        std::vector<int> h36_dims;
        for (int d = 3; d <= config.max_dim && d <= 6; ++d)
        {
            for (size_t i = 0; i < dim_simplices[d].size(); ++i)
            {
                h36_simplices.push_back(dim_simplices[d][i]);
                h36_filtrations.push_back(dim_filtrations[d][i]);
                h36_dims.push_back(d);
            }
        }
        if (!h36_simplices.empty())
        {
            if (config.use_involution)
            {
                // Use involuted homology
                InvolutedConfig inv_config;
                inv_config.max_dim = config.max_dim;
                inv_config.use_involution = true;
                inv_config.involution_threshold_dim = 3;
                auto inv_result = computeInvolutedHomology(h36_simplices, h36_filtrations, h36_dims,
                                                           config.max_dim, inv_config);
                // Extract pairs
                for (const auto &pair : inv_result.all_pairs)
                {
                    result.pairs.push_back(pair);
                    switch (pair.dimension)
                    {
                        case 3:
                            result.pairs_h3.push_back(pair);
                            break;
                        case 4:
                            result.pairs_h4.push_back(pair);
                            break;
                        case 5:
                            result.pairs_h5.push_back(pair);
                            break;
                        case 6:
                            result.pairs_h6.push_back(pair);
                            break;
                    }
                }
            }
            else
            {
                // Fall back to cohomology
                CohomologyConfig cohom_config;
                cohom_config.max_dimension = config.max_dim;
                cohom_config.use_cohomology = true;
                cohom_config.use_clearing = config.use_clearing;
                auto cohom_result = computePersistentCohomology(
                    h36_simplices, h36_filtrations, h36_dims, config.max_dim, cohom_config);
                for (const auto &pair : cohom_result.all_pairs)
                {
                    result.pairs.push_back(pair);
                    switch (pair.dimension)
                    {
                        case 3:
                            result.pairs_h3.push_back(pair);
                            break;
                        case 4:
                            result.pairs_h4.push_back(pair);
                            break;
                        case 5:
                            result.pairs_h5.push_back(pair);
                            break;
                        case 6:
                            result.pairs_h6.push_back(pair);
                            break;
                    }
                }
            }
        }
        auto end_h36 = std::chrono::high_resolution_clock::now();
        result.time_h3_h6_ms =
            std::chrono::duration<double, std::milli>(end_h36 - start_h36).count();
    }
    auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();
    return result;
}
// Get optimal configuration
HighDimensionalExactConfig getOptimalHighDimensionalConfig(size_t num_points, int max_dim)
{
    HighDimensionalExactConfig config;
    config.max_dim = max_dim;
    // Always use cohomology and clearing
    config.use_cohomology = true;
    config.use_clearing = true;
    config.use_apparent_pairs = (num_points > APPARENT_PAIRS_THRESHOLD);
    // Use involution for H3+ if max_dim >= 3
    config.use_involution = (max_dim >= 3);
    config.involution_threshold = 3;
    return config;
}
// Benchmark all approaches with actual timing
HighDimensionalBenchmark benchmarkHighDimensional(const std::vector<std::vector<int>> &simplices,
                                                  const std::vector<double> &filtration_values,
                                                  const std::vector<int> &dimensions, int max_dim)
{
    HighDimensionalBenchmark bench{};
    bench.homology_time_ms = 0.0;
    bench.cohomology_time_ms = 0.0;
    bench.involuted_time_ms = 0.0;
    bench.speedup_vs_homology = 1.0;
    bench.speedup_vs_cohomology = 1.0;

    if (simplices.empty() || simplices.size() != filtration_values.size() ||
        simplices.size() != dimensions.size() || max_dim < 0)
    {
        return bench;
    }

    auto run_once = [&](const HighDimensionalExactConfig &cfg) -> double {
        const auto start = std::chrono::high_resolution_clock::now();
        const auto computed =
            computeHighDimensionalExact(simplices, filtration_values, dimensions, cfg);
        if (computed.num_simplices != static_cast<int>(simplices.size()))
        {
            throw std::runtime_error(
                "high-dimensional benchmark produced inconsistent simplex count");
        }
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    HighDimensionalExactConfig baseline{};
    baseline.max_dim = std::min(max_dim, MAX_SUPPORTED_DIM);
    baseline.use_cohomology = false;
    baseline.use_involution = false;
    baseline.use_clearing = false;
    baseline.use_apparent_pairs = false;
    bench.homology_time_ms = run_once(baseline);

    HighDimensionalExactConfig cohomology_only = baseline;
    cohomology_only.use_cohomology = true;
    cohomology_only.use_clearing = true;
    cohomology_only.use_apparent_pairs = true;
    bench.cohomology_time_ms = run_once(cohomology_only);

    HighDimensionalExactConfig involuted = cohomology_only;
    involuted.use_involution = (involuted.max_dim >= 3);
    bench.involuted_time_ms = run_once(involuted);

    if (bench.involuted_time_ms > 0.0)
    {
        bench.speedup_vs_homology = bench.homology_time_ms / bench.involuted_time_ms;
        bench.speedup_vs_cohomology = bench.cohomology_time_ms / bench.involuted_time_ms;
    }
    return bench;
}
// Analyze dimension breakdown
std::vector<DimensionBreakdown> analyzeDimensionBreakdown(const HighDimensionalExactResult &result)
{
    std::vector<DimensionBreakdown> breakdown;
    const std::vector<Pair> *dim_pairs[] = {&result.pairs_h0, &result.pairs_h1, &result.pairs_h2,
                                            &result.pairs_h3, &result.pairs_h4, &result.pairs_h5,
                                            &result.pairs_h6};
    const char *algorithms[] = {"Union-Find", "Cohomology", "Cohomology", "Involuted",
                                "Involuted",  "Involuted",  "Involuted"};
    double dim_times[] = {
        result.time_h0_h2_ms * 0.1,  // Approximate H0
        result.time_h0_h2_ms * 0.4,  // Approximate H1
        result.time_h0_h2_ms * 0.5,  // Approximate H2
        result.time_h3_h6_ms * 0.25, // H3
        result.time_h3_h6_ms * 0.25, // H4
        result.time_h3_h6_ms * 0.25, // H5
        result.time_h3_h6_ms * 0.25  // H6
    };
    for (int d = 0; d <= 6; ++d)
    {
        DimensionBreakdown db;
        db.dimension = d;
        db.num_simplices = static_cast<int>(dim_pairs[d]->size());
        db.num_pairs = static_cast<int>(dim_pairs[d]->size());
        db.computation_time_ms = dim_times[d];
        db.percent_of_total = (result.total_time_ms > 0)
                                  ? (dim_times[d] / result.total_time_ms * PERCENTAGE_MULTIPLIER)
                                  : 0;
        db.algorithm_used = algorithms[d];
        breakdown.push_back(db);
    }
    return breakdown;
}
// Validation
bool validateExactResults(const HighDimensionalExactResult &result,
                          const std::vector<Pair> &reference_homology)
{
    // Check that results match
    if (result.pairs.size() != reference_homology.size())
    {
        return false;
    }
    // Sort both for comparison
    auto sorted_result = result.pairs;
    auto sorted_ref = reference_homology;
    std::ranges::sort(sorted_result, {},
                      [](const Pair &p) { return std::tuple(p.dimension, p.birth, p.death); });
    std::ranges::sort(sorted_ref, {},
                      [](const Pair &p) { return std::tuple(p.dimension, p.birth, p.death); });
    // Compare
    for (size_t i = 0; i < sorted_result.size(); ++i)
    {
        if (sorted_result[i].dimension != sorted_ref[i].dimension ||
            std::abs(sorted_result[i].birth - sorted_ref[i].birth) >
                PERSISTENCE_COMPARISON_TOLERANCE ||
            std::abs(sorted_result[i].death - sorted_ref[i].death) >
                PERSISTENCE_COMPARISON_TOLERANCE)
        {
            return false;
        }
    }
    return true;
}
} // namespace nerve::persistence
