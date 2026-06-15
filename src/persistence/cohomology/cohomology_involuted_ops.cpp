// Involuted/cohomology hybrid persistence operations.

#include "nerve/persistence/cohomology/cohomology_involuted_ops.hpp"
#include "nerve/persistence/cohomology/involuted_homology.hpp"
#include "nerve/persistence/kernels/simplex_reduction_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nerve::persistence
{

struct InvolutedSimplex
{
    std::vector<int> vertices;
    double filtration_value;
    int dimension;
    int index;

    int dual_index;
    bool is_essential;
};

namespace
{

constexpr int kMaxInvolutedDimension = 6;

class InvolutedCoboundaryOperator
{
public:
    void initialize(const std::vector<InvolutedSimplex> &simplices)
    {
        simplices_by_dimension_.clear();
        for (const auto &s : simplices)
        {
            simplices_by_dimension_[s.dimension].push_back(&s);
        }
    }

    std::vector<int> computeCoboundary(const InvolutedSimplex &simplex) const
    {
        std::vector<int> coboundary;
        const auto cofaces = simplices_by_dimension_.find(simplex.dimension + 1);
        if (cofaces == simplices_by_dimension_.end())
        {
            return coboundary;
        }
        for (const InvolutedSimplex *coface : cofaces->second)
        {
            if (std::ranges::includes(coface->vertices, simplex.vertices))
            {
                coboundary.push_back(coface->index);
            }
        }
        std::ranges::sort(coboundary);
        return coboundary;
    }

private:
    std::unordered_map<int, std::vector<const InvolutedSimplex *>> simplices_by_dimension_;
};

class HybridDimensionReducer
{
public:
    void reduceDimension(int dim, const std::vector<InvolutedSimplex> &simplices,
                         const std::vector<InvolutedSimplex> &all_simplices,
                         const std::vector<std::vector<int>> &coboundary_matrix,
                         const std::vector<std::vector<int>> &all_coboundaries,
                         std::vector<Pair> &pairs, bool use_involution)
    {
        reduceColumns(dim, simplices, all_simplices, coboundary_matrix, all_coboundaries, pairs,
                      dim >= 3 && use_involution);
    }

private:
    void reduceColumns(int dim, const std::vector<InvolutedSimplex> &simplices,
                       const std::vector<InvolutedSimplex> &all_simplices,
                       const std::vector<std::vector<int>> &coboundary_matrix,
                       const std::vector<std::vector<int>> &all_coboundaries,
                       std::vector<Pair> &pairs, bool use_involution)
    {
        std::unordered_map<int, std::vector<int>> reduced_by_pivot;

        for (size_t i = 0; i < coboundary_matrix.size(); ++i)
        {
            auto column = coboundary_matrix[i];

            // Involution is only folded for same-dimensional dual columns; cross-
            // dimension duals belong to different chain groups and must not share pivots.
            const int dual_idx = simplices[i].dual_index;
            if (use_involution && dual_idx >= 0 &&
                static_cast<size_t>(dual_idx) < all_simplices.size() &&
                all_simplices[dual_idx].dimension == simplices[i].dimension)
            {
                std::vector<int> combined;
                const auto &dual_col = all_coboundaries[static_cast<size_t>(dual_idx)];
                std::set_symmetric_difference(column.begin(), column.end(), dual_col.begin(),
                                              dual_col.end(), std::back_inserter(combined));
                column = std::move(combined);
                std::ranges::sort(column);
            }

            while (!column.empty())
            {
                int pivot = column.back();
                auto it = reduced_by_pivot.find(pivot);

                if (it != reduced_by_pivot.end())
                {
                    std::vector<int> result;
                    const auto &other = it->second;
                    std::set_symmetric_difference(column.begin(), column.end(), other.begin(),
                                                  other.end(), std::back_inserter(result));
                    column = std::move(result);
                }
                else
                {
                    reduced_by_pivot.emplace(pivot, column);
                    Pair pair;
                    pair.dimension = dim;
                    pair.birth = simplices[i].filtration_value;
                    pair.death = all_simplices[static_cast<size_t>(pivot)].filtration_value;
                    pairs.push_back(pair);
                    break;
                }
            }

            if (column.empty())
            {
                Pair pair;
                pair.dimension = dim;
                pair.birth = simplices[i].filtration_value;
                pair.death = std::numeric_limits<double>::infinity();
                pairs.push_back(pair);
            }
        }
    }
};

} // namespace

InvolutedResult computeInvolutedHomology(const std::vector<std::vector<int>> &simplices,
                                         const std::vector<double> &filtration_values,
                                         const std::vector<int> &dimensions, int max_dim,
                                         const InvolutedConfig &config)
{
    kernels::validateSimplicialInput(simplices, filtration_values, dimensions);
    if (max_dim < 0 || config.max_dim < 0 || config.involution_threshold_dim < 0)
    {
        throw std::invalid_argument("involuted dimensions must be non-negative");
    }

    const int effective_max_dim = std::min({max_dim, config.max_dim, kMaxInvolutedDimension});
    InvolutedResult result{};
    result.max_dim = effective_max_dim;
    result.used_involution = false;
    result.involution_threshold = config.involution_threshold_dim;

    auto start_total = std::chrono::high_resolution_clock::now();

    std::vector<InvolutedSimplex> inv_simplices;
    inv_simplices.reserve(simplices.size());

    for (size_t i = 0; i < simplices.size(); ++i)
    {
        InvolutedSimplex s;
        s.vertices = simplices[i];
        std::ranges::sort(s.vertices);
        s.filtration_value = filtration_values[i];
        s.dimension = dimensions[i];
        s.index = static_cast<int>(i);
        s.dual_index = -1;
        s.is_essential = false;
        inv_simplices.push_back(s);
    }

    std::ranges::sort(inv_simplices, {}, &InvolutedSimplex::filtration_value);

    for (size_t i = 0; i < inv_simplices.size(); ++i)
    {
        inv_simplices[i].index = static_cast<int>(i);
    }

    size_t num_foldable_duals = 0;
    if (config.use_involution)
    {
        for (auto &s : inv_simplices)
        {
            if (s.dimension >= config.involution_threshold_dim)
            {
                s.dual_index = computeDualIndex(s, inv_simplices);
                if (s.dual_index >= 0 &&
                    inv_simplices[static_cast<size_t>(s.dual_index)].dimension == s.dimension)
                {
                    ++num_foldable_duals;
                }
            }
        }
    }
    result.used_involution = config.use_involution && num_foldable_duals > 0;

    InvolutedCoboundaryOperator coboundary_op;
    coboundary_op.initialize(inv_simplices);
    std::vector<std::vector<int>> all_coboundaries(inv_simplices.size());
    for (const auto &s : inv_simplices)
    {
        all_coboundaries[static_cast<size_t>(s.index)] = coboundary_op.computeCoboundary(s);
    }

    HybridDimensionReducer reducer;

    for (int dim = 0; dim <= effective_max_dim; ++dim)
    {
        auto start_dim = std::chrono::high_resolution_clock::now();

        std::vector<InvolutedSimplex> dim_simplices;
        for (const auto &s : inv_simplices)
        {
            if (s.dimension == dim)
                dim_simplices.push_back(s);
        }

        if (dim_simplices.empty())
            continue;

        std::vector<std::vector<int>> coboundary_matrix;
        coboundary_matrix.reserve(dim_simplices.size());
        for (const auto &s : dim_simplices)
        {
            coboundary_matrix.push_back(all_coboundaries[static_cast<size_t>(s.index)]);
        }

        std::vector<Pair> dim_pairs;
        bool use_inv = config.use_involution && dim >= config.involution_threshold_dim;

        reducer.reduceDimension(dim, dim_simplices, inv_simplices, coboundary_matrix,
                                all_coboundaries, dim_pairs, use_inv);

        result.pairs_by_dimension[dim] = dim_pairs;
        for (const auto &pair : dim_pairs)
        {
            result.all_pairs.push_back(pair);
        }

        auto end_dim = std::chrono::high_resolution_clock::now();
        result.dimension_times_ms[dim] =
            std::chrono::duration<double, std::milli>(end_dim - start_dim).count();
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();

    result.num_simplices = static_cast<int>(simplices.size());

    return result;
}

int computeDualIndex(const InvolutedSimplex &simplex,
                     const std::vector<InvolutedSimplex> &all_simplices)
{
    std::vector<int> vertices;
    for (const auto &s : all_simplices)
    {
        vertices.insert(vertices.end(), s.vertices.begin(), s.vertices.end());
    }
    std::ranges::sort(vertices);
    vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());

    std::vector<int> complement;
    std::set_difference(vertices.begin(), vertices.end(), simplex.vertices.begin(),
                        simplex.vertices.end(), std::back_inserter(complement));
    for (const auto &candidate : all_simplices)
    {
        if (candidate.vertices == complement)
        {
            return candidate.index;
        }
    }
    return -1;
}

CohomologyResult computeHybridCohomologyInvoluted(const std::vector<std::vector<int>> &simplices,
                                                  const std::vector<double> &filtration_values,
                                                  const std::vector<int> &dimensions, int max_dim)
{
    kernels::validateSimplicialInput(simplices, filtration_values, dimensions);
    if (max_dim < 0)
    {
        throw std::invalid_argument("maximum dimension must be non-negative");
    }

    const int effective_max_dim = std::min(max_dim, kMaxInvolutedDimension);
    CohomologyResult result{};
    result.max_dim = effective_max_dim;
    result.used_cohomology = true;
    result.used_clearing = true;
    result.used_apparent_pairs = true;
    result.pairs_by_dimension.resize(static_cast<size_t>(effective_max_dim + 1));
    result.dimension_times_ms.resize(static_cast<size_t>(effective_max_dim + 1), 0.0);

    if (effective_max_dim <= 2)
    {
        CohomologyConfig config;
        config = getOptimalCohomologyConfig(effective_max_dim, simplices.size());
        return computePersistentCohomology(simplices, filtration_values, dimensions,
                                           effective_max_dim, config);
    }
    std::vector<std::vector<int>> low_dim_simplices;
    std::vector<double> low_dim_filtrations;
    std::vector<int> low_dim_dims;

    std::vector<std::vector<int>> high_dim_simplices;
    std::vector<double> high_dim_filtrations;
    std::vector<int> high_dim_dims;

    for (size_t i = 0; i < simplices.size(); ++i)
    {
        if (dimensions[i] <= 2)
        {
            low_dim_simplices.push_back(simplices[i]);
            low_dim_filtrations.push_back(filtration_values[i]);
            low_dim_dims.push_back(dimensions[i]);
        }
        else
        {
            high_dim_simplices.push_back(simplices[i]);
            high_dim_filtrations.push_back(filtration_values[i]);
            high_dim_dims.push_back(dimensions[i]);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    CohomologyConfig cohom_config;
    cohom_config = getOptimalCohomologyConfig(2, low_dim_simplices.size());
    auto low_result = computePersistentCohomology(low_dim_simplices, low_dim_filtrations,
                                                  low_dim_dims, 2, cohom_config);

    if (!high_dim_simplices.empty())
    {
        InvolutedConfig inv_config;
        inv_config.max_dim = effective_max_dim;
        inv_config.use_involution = true;
        inv_config.involution_threshold_dim = 3;

        auto high_result = computeInvolutedHomology(high_dim_simplices, high_dim_filtrations,
                                                    high_dim_dims, effective_max_dim, inv_config);

        result.all_pairs = low_result.all_pairs;
        for (const auto &pair : high_result.all_pairs)
        {
            result.all_pairs.push_back(pair);
        }

        for (int d = 0; d <= 2; ++d)
        {
            if (static_cast<size_t>(d) < low_result.dimension_times_ms.size())
            {
                result.dimension_times_ms[static_cast<size_t>(d)] =
                    low_result.dimension_times_ms[static_cast<size_t>(d)];
            }
            if (static_cast<size_t>(d) < low_result.pairs_by_dimension.size())
            {
                result.pairs_by_dimension[static_cast<size_t>(d)] =
                    low_result.pairs_by_dimension[static_cast<size_t>(d)];
            }
        }
        for (int d = 3; d <= effective_max_dim; ++d)
        {
            result.dimension_times_ms[static_cast<size_t>(d)] = high_result.dimension_times_ms[d];
            result.pairs_by_dimension[static_cast<size_t>(d)] = high_result.pairs_by_dimension[d];
        }
    }
    else
    {
        result = low_result;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

InvolutedConfig getOptimalInvolutedConfig(int max_dim, size_t num_simplices)
{
    InvolutedConfig config;

    config.use_involution = (max_dim >= 3 && num_simplices >= 1024);
    config.involution_threshold_dim = 3;

    return config;
}

InvolutedSpeedupEstimate estimateInvolutedSpeedup(int max_dim, size_t num_high_dim_simplices)
{
    InvolutedSpeedupEstimate estimate;

    if (max_dim >= 3)
    {
        const double size_factor = std::clamp(
            std::log2(static_cast<double>(std::max<size_t>(1, num_high_dim_simplices))) / 20.0,
            0.25, 1.0);
        estimate.involution_speedup = (1.5 + (max_dim - 3) * 0.3) * size_factor;
        estimate.involution_speedup = std::min(estimate.involution_speedup, 3.0);

        estimate.memory_reduction = 0.2 * size_factor;
        estimate.total_speedup_vs_homology = 2.0 * estimate.involution_speedup;
    }
    else
    {
        estimate.involution_speedup = 1.0;
        estimate.total_speedup_vs_homology = 2.0;
    }

    return estimate;
}

} // namespace nerve::persistence
