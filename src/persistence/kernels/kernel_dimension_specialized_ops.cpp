#include "nerve/persistence/cohomology/cohomology_involuted_ops.hpp"
#include "nerve/persistence/kernels/dimension_specialized_kernels.hpp"
#include "nerve/persistence/kernels/kernel_dimension_specialized_ops.hpp"

constexpr int MAX_DIM_SPECIALIZED = 6;

#include "nerve/persistence/cohomology/involuted_homology.hpp"
#include "nerve/persistence/reduction/reduction_union_find_ops.hpp"
#include "nerve/persistence/reduction/union_find_ph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace nerve::persistence::specialized
{

using namespace kernels;

namespace
{

bool hasValidInput(const std::vector<std::vector<int>> &simplices,
                   const std::vector<double> &filtration_values, const std::vector<int> &dimensions,
                   int max_dim)
{
    if (max_dim < 0 || simplices.size() != filtration_values.size() ||
        simplices.size() != dimensions.size())
    {
        return false;
    }
    for (size_t i = 0; i < simplices.size(); ++i)
    {
        if (!std::isfinite(filtration_values[i]) || dimensions[i] < 0 ||
            dimensions[i] > MAX_DIM_SPECIALIZED ||
            dimensions[i] != static_cast<int>(simplices[i].size()) - 1)
        {
            return false;
        }
        for (int vertex : simplices[i])
        {
            if (vertex < 0)
            {
                return false;
            }
        }
    }
    return true;
}

std::vector<std::pair<double, double>> computeH0Pairs(const std::vector<double> &points,
                                                      size_t point_dim, size_t num_points,
                                                      const std::vector<std::pair<int, int>> &edges,
                                                      const std::vector<double> &edge_weights)
{
    const auto raw_pairs = ::nerve::persistence::computeD0PersistenceUnionFind(
        points, point_dim, num_points, edges, edge_weights);
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(raw_pairs.size());
    for (const auto &pair : raw_pairs)
    {
        pairs.emplace_back(pair.birth, pair.death);
    }
    return pairs;
}

// H0: Union-Find.
H0Result computeH0Optimized(const std::vector<double> &points, size_t point_dim, size_t num_points,
                            const std::vector<std::pair<int, int>> &edges,
                            const std::vector<double> &edge_weights)
{
    H0Result result;

    auto start = std::chrono::high_resolution_clock::now();

    auto pairs = computeH0Pairs(points, point_dim, num_points, edges, edge_weights);
    result.persistence_pairs = pairs;
    result.used_bit_parallel = false;

    auto end = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.algorithm_used = "Union-Find";

    return result;
}

// H1-H2: Cohomology with clearing.
H12Result computeH12Optimized(const std::vector<std::vector<int>> &simplices,
                              const std::vector<double> &filtration_values,
                              const std::vector<int> &dimensions, const DimensionConfig &config)
{
    H12Result result;

    auto start = std::chrono::high_resolution_clock::now();

    // Separate H1 and H2 simplices
    std::vector<std::vector<int>> h1_simplices, h2_simplices;
    std::vector<double> h1_filt, h2_filt;
    std::vector<int> h1_dims, h2_dims;

    for (size_t i = 0; i < simplices.size(); ++i)
    {
        if (dimensions[i] == 1)
        {
            h1_simplices.push_back(simplices[i]);
            h1_filt.push_back(filtration_values[i]);
            h1_dims.push_back(1);
        }
        else if (dimensions[i] == 2)
        {
            h2_simplices.push_back(simplices[i]);
            h2_filt.push_back(filtration_values[i]);
            h2_dims.push_back(2);
        }
    }

    // Combine for cohomology computation
    std::vector<std::vector<int>> h12_simplices;
    std::vector<double> h12_filt;
    std::vector<int> h12_dims;

    h12_simplices.insert(h12_simplices.end(), h1_simplices.begin(), h1_simplices.end());
    h12_simplices.insert(h12_simplices.end(), h2_simplices.begin(), h2_simplices.end());
    h12_filt.insert(h12_filt.end(), h1_filt.begin(), h1_filt.end());
    h12_filt.insert(h12_filt.end(), h2_filt.begin(), h2_filt.end());
    h12_dims.insert(h12_dims.end(), h1_dims.begin(), h1_dims.end());
    h12_dims.insert(h12_dims.end(), h2_dims.begin(), h2_dims.end());

    if (config.use_cohomology)
    {
        // Use cohomology with clearing
        CohomologyConfig cohom_config;
        cohom_config.max_dimension = 2;
        cohom_config.use_cohomology = true;
        cohom_config.use_clearing = true;
        cohom_config.use_apparent_pairs = true;
        cohom_config.use_dimension_cascade = true;

        auto cohom_result =
            computePersistentCohomology(h12_simplices, h12_filt, h12_dims, 2, cohom_config);

        // Extract H1 and H2 pairs
        for (const auto &pair : cohom_result.all_pairs)
        {
            auto persistence_pair = std::make_pair(pair.birth, pair.death);
            if (pair.dimension == 1)
            {
                result.pairs_h1.push_back(persistence_pair);
                result.all_pairs.push_back(persistence_pair);
            }
            else if (pair.dimension == 2)
            {
                result.pairs_h2.push_back(persistence_pair);
                result.all_pairs.push_back(persistence_pair);
            }
        }

        result.used_cohomology = true;
        result.algorithm_used = "Cohomology + Clearing";
    }

    result.used_bit_parallel = false;
    result.used_clear_compress = false;

    auto end = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

// H3-H6: Involuted homology when valid dual structure exists.
H36Result computeH36Optimized(const std::vector<std::vector<int>> &simplices,
                              const std::vector<double> &filtration_values,
                              const std::vector<int> &dimensions, int max_dim,
                              const DimensionConfig &config)
{
    H36Result result;

    auto start = std::chrono::high_resolution_clock::now();

    // Separate by dimension
    std::vector<std::vector<int>> dim_simplices[4]; // H3, H4, H5, H6
    std::vector<double> dim_filt[4];

    for (size_t i = 0; i < simplices.size(); ++i)
    {
        int dim = dimensions[i];
        if (dim >= 3 && dim <= max_dim && dim <= 6)
        {
            int idx = dim - 3;
            dim_simplices[idx].push_back(simplices[i]);
            dim_filt[idx].push_back(filtration_values[i]);
        }
    }

    // Build combined H3-H6 list
    std::vector<std::vector<int>> h36_simplices;
    std::vector<double> h36_filt;
    std::vector<int> h36_dims;

    for (int d = 3; d <= max_dim && d <= 6; ++d)
    {
        int idx = d - 3;
        for (size_t i = 0; i < dim_simplices[idx].size(); ++i)
        {
            h36_simplices.push_back(dim_simplices[idx][i]);
            h36_filt.push_back(dim_filt[idx][i]);
            h36_dims.push_back(d);
        }
    }

    // Involuted homology for algebraic structure
    InvolutedConfig inv_config;
    inv_config.max_dim = max_dim;
    inv_config.use_involution = config.use_involution;
    inv_config.involution_threshold_dim = 3;

    auto inv_result =
        computeInvolutedHomology(h36_simplices, h36_filt, h36_dims, max_dim, inv_config);

    // Extract pairs by dimension
    for (const auto &pair : inv_result.all_pairs)
    {
        auto persistence_pair = std::make_pair(pair.birth, pair.death);
        result.all_pairs.push_back(persistence_pair);

        switch (pair.dimension)
        {
            case 3:
                result.pairs_h3.push_back(persistence_pair);
                break;
            case 4:
                result.pairs_h4.push_back(persistence_pair);
                break;
            case 5:
                result.pairs_h5.push_back(persistence_pair);
                break;
            case 6:
                result.pairs_h6.push_back(persistence_pair);
                break;
        }
    }

    result.used_involution = inv_result.used_involution;
    result.used_bit_parallel = false;
    result.used_clear_compress = false;
    result.bit_parallel_speedup = 1.0;
    result.algorithm_used =
        result.used_involution ? "Involuted Homology" : "High-Dimensional Homology";

    auto end = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

} // namespace

// Main API: Unified dimension-specialized computation
DimensionSpecializedResult computeDimensionSpecialized(
    const std::vector<std::vector<int>> &simplices, const std::vector<double> &filtration_values,
    const std::vector<int> &dimensions, int max_dim, const DimensionConfig &config)
{
    DimensionSpecializedResult result;
    result.max_dim = std::clamp(max_dim, 0, MAX_DIM_SPECIALIZED);
    result.config_used = config;
    if (!hasValidInput(simplices, filtration_values, dimensions, max_dim))
    {
        return result;
    }

    auto start_total = std::chrono::high_resolution_clock::now();

    // Extract edges for H0 if needed
    if (max_dim >= 0)
    {
        std::vector<std::pair<int, int>> edges;
        std::vector<double> edge_weights;

        for (size_t i = 0; i < simplices.size(); ++i)
        {
            if (dimensions[i] == 1 && simplices[i].size() == 2)
            {
                edges.push_back({simplices[i][0], simplices[i][1]});
                edge_weights.push_back(filtration_values[i]);
            }
        }

        // Get number of vertices from H0 simplices
        size_t num_points = 0;
        for (size_t i = 0; i < simplices.size(); ++i)
        {
            if (dimensions[i] == 0 && !simplices[i].empty())
            {
                num_points = std::max(num_points, static_cast<size_t>(simplices[i][0] + 1));
            }
        }

        // Build auxiliary points (not needed for Union-Find)
        std::vector<double> points(num_points, 0.0);

        // H0: Union-Find
        auto h0_result = computeH0Optimized(points, 1, num_points, edges, edge_weights);
        result.h0 = h0_result;

        for (const auto &pair : h0_result.pairs())
        {
            result.all_pairs.push_back(pair);
        }
    }

    // H1-H2: Cohomology with clearing
    if (max_dim >= 1)
    {
        auto h12_result = computeH12Optimized(simplices, filtration_values, dimensions, config);
        result.h12 = h12_result;

        for (const auto &pair : h12_result.all_pairs)
        {
            result.all_pairs.push_back(pair);
        }
    }

    // H3-H6: Involuted homology
    if (max_dim >= 3)
    {
        auto h36_result =
            computeH36Optimized(simplices, filtration_values, dimensions, max_dim, config);
        result.h36 = h36_result;

        for (const auto &pair : h36_result.all_pairs)
        {
            result.all_pairs.push_back(pair);
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();

    // This path does not execute a reference baseline in-process.
    result.total_speedup_estimate = 1.0;
    result.speedup_vs_standard = 1.0;

    return result;
}

// Policy/tuning helpers are split out to keep this translation unit under
// repository line-cap constraints.
#include "detail/kernel_dimension_specialized_tuning.inl"

} // namespace nerve::persistence::specialized
