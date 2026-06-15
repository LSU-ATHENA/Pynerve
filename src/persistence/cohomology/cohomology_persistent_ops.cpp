// Persistent cohomology with clearing and apparent-pair optimizations.
// Extended for 0-6D with dimension-cascade execution order.

constexpr int MAX_DIM_COHOMOLOGY = 6;
#include "nerve/persistence/cohomology/persistent_cohomology.hpp"
#include "nerve/persistence/kernels/simplex_reduction_utils.hpp"

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cmath>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nerve::persistence
{
namespace
{
// Coboundary matrix column for cohomology
// In cohomology, we work with columns = coboundaries (reverse of homology)
struct CoboundaryColumn
{
    std::vector<int> coboundary; // Indices of cofaces (simplices containing this one)
    int pivot;                   // Lowest 1 in the column
    bool reduced;                // Whether fully reduced
    CoboundaryColumn()
        : pivot(-1)
        , reduced(false)
    {}
};
// Simplex representation for cohomology
struct CohomologySimplex
{
    std::vector<int> vertices; // Vertex indices (sorted)
    double filtration_value;   // Birth time
    int dimension;             // Simplex dimension
    int index;                 // Global index in filtration
    bool operator<(const CohomologySimplex &other) const
    {
        if (filtration_value != other.filtration_value)
            return filtration_value < other.filtration_value;
        return dimension < other.dimension;
    }
};
// Check if two simplices share a common face
bool areCobounding(const CohomologySimplex &a, const CohomologySimplex &b)
{
    // For cohomology, check if a is a face of b
    if (b.dimension != a.dimension + 1)
        return false;
    // Check if all vertices of a are in b
    size_t ai = 0, bi = 0;
    while (ai < a.vertices.size() && bi < b.vertices.size())
    {
        if (a.vertices[ai] == b.vertices[bi])
        {
            ++ai;
            ++bi;
        }
        else if (a.vertices[ai] > b.vertices[bi])
        {
            ++bi;
        }
        else
        {
            return false;
        }
    }
    return ai == a.vertices.size();
}
// Clearing optimization: mark birth indices that are already paired
// This is the key to fast's speed
class ClearingTracker
{
public:
    void markCleared(int simplex_index) { cleared_indices_.insert(simplex_index); }
    bool isCleared(int simplex_index) const { return cleared_indices_.count(simplex_index) > 0; }

private:
    std::unordered_set<int> cleared_indices_;
};
// Apparent pair detection (optimization used in fast)
// An apparent pair is (sigma, tau) where:
// - tau is the youngest cofacet of sigma
// - sigma is the oldest facet of tau
// These don't need matrix reduction
bool isApparentPair(const CohomologySimplex &sigma, const CohomologySimplex &tau,
                    const std::vector<CohomologySimplex> &simplices)
{
    if (tau.dimension != sigma.dimension + 1)
    {
        return false;
    }
    if (!areCobounding(sigma, tau))
    {
        return false;
    }

    // Tau must be the youngest cofacet of sigma under (filtration, index) order.
    for (const auto &s : simplices)
    {
        if (s.dimension != tau.dimension || !areCobounding(sigma, s))
        {
            continue;
        }
        if (s.filtration_value > tau.filtration_value)
        {
            return false;
        }
        if (s.filtration_value == tau.filtration_value && s.index > tau.index)
        {
            return false;
        }
    }

    // Sigma must be the oldest facet of tau under (filtration, index) order.
    for (const auto &s : simplices)
    {
        if (s.dimension != sigma.dimension || !areCobounding(s, tau))
        {
            continue;
        }
        if (s.filtration_value < sigma.filtration_value)
        {
            return false;
        }
        if (s.filtration_value == sigma.filtration_value && s.index < sigma.index)
        {
            return false;
        }
    }
    return true;
}
// Add column j to column i (in-place)
void addColumn(CoboundaryColumn &col_i, const CoboundaryColumn &col_j)
{
    // XOR the coboundaries (Z2 arithmetic)
    std::vector<int> result;
    result.reserve(col_i.coboundary.size() + col_j.coboundary.size());
    std::set_symmetric_difference(col_i.coboundary.begin(), col_i.coboundary.end(),
                                  col_j.coboundary.begin(), col_j.coboundary.end(),
                                  std::back_inserter(result));
    col_i.coboundary = std::move(result);
    col_i.pivot = col_i.coboundary.empty() ? -1 : col_i.coboundary.back();
}
} // namespace
// Main API: Persistent Cohomology with Clearing
CohomologyResult computePersistentCohomology(const std::vector<std::vector<int>> &simplices,
                                             const std::vector<double> &filtration_values,
                                             const std::vector<int> &dimensions, int max_dim,
                                             const CohomologyConfig &config)
{
    kernels::validateSimplicialInput(simplices, filtration_values, dimensions);
    if (max_dim < 0)
    {
        throw std::invalid_argument("maximum dimension must be non-negative");
    }
    CohomologyResult result{};
    const int effective_max_dim = std::clamp(max_dim, 0, MAX_DIM_COHOMOLOGY);
    result.max_dim = effective_max_dim;
    result.used_cohomology = true;
    result.used_clearing = config.use_clearing;
    result.used_apparent_pairs = config.use_apparent_pairs;
    auto start_total = std::chrono::high_resolution_clock::now();
    // Build simplex structures
    std::vector<CohomologySimplex> cohom_simplices;
    cohom_simplices.reserve(simplices.size());
    for (size_t i = 0; i < simplices.size(); ++i)
    {
        CohomologySimplex s;
        s.vertices = simplices[i];
        s.filtration_value = filtration_values[i];
        s.dimension = dimensions[i];
        s.index = static_cast<int>(i);
        cohom_simplices.push_back(s);
    }
    // Sort by filtration value
    std::ranges::sort(cohom_simplices, {}, &CohomologySimplex::filtration_value);
    // Update indices after sorting
    for (size_t i = 0; i < cohom_simplices.size(); ++i)
    {
        cohom_simplices[i].index = static_cast<int>(i);
    }
    std::unordered_map<int, const CohomologySimplex *> simplex_by_index;
    simplex_by_index.reserve(cohom_simplices.size());
    for (const auto &simplex : cohom_simplices)
    {
        simplex_by_index.emplace(simplex.index, &simplex);
    }
    result.pairs_by_dimension.resize(static_cast<size_t>(effective_max_dim + 1));
    result.dimension_times_ms.resize(static_cast<size_t>(effective_max_dim + 1), 0.0);

    // Dimension-cascading clearing
    // H0 -> clear for H1 -> clear for H2 -> ... -> H6
    ClearingTracker clearing;
    for (int dim = 0; dim <= effective_max_dim; ++dim)
    {
        auto start_dim = std::chrono::high_resolution_clock::now();
        // Get simplices of this dimension and dimension+1 (coboundaries)
        std::vector<CohomologySimplex> dim_simplices;
        std::vector<CohomologySimplex> dim_plus_1_simplices;
        for (const auto &s : cohom_simplices)
        {
            if (s.dimension == dim)
                dim_simplices.push_back(s);
            if (s.dimension == dim + 1)
                dim_plus_1_simplices.push_back(s);
        }
        // Build coboundary matrix
        std::vector<CoboundaryColumn> coboundary_matrix(dim_simplices.size());
        for (size_t i = 0; i < dim_simplices.size(); ++i)
        {
            // Compute coboundary: all (dim+1)-simplices containing this simplex
            for (const auto &s : dim_plus_1_simplices)
            {
                if (areCobounding(dim_simplices[i], s))
                {
                    coboundary_matrix[i].coboundary.push_back(s.index);
                }
            }
            std::ranges::sort(coboundary_matrix[i].coboundary);
            coboundary_matrix[i].pivot = coboundary_matrix[i].coboundary.empty()
                                             ? -1
                                             : coboundary_matrix[i].coboundary.back();
        }
        // Reduce coboundary matrix with clearing
        std::vector<Pair> dim_pairs;
        std::unordered_map<int, int> pivot_to_column;
        for (size_t i = 0; i < coboundary_matrix.size(); ++i)
        {
            // Skip if this index was cleared in previous dimension
            if (config.use_clearing && clearing.isCleared(dim_simplices[i].index))
            {
                continue;
            }
            auto &col = coboundary_matrix[i];
            // Check for apparent pair
            if (config.use_apparent_pairs && col.pivot >= 0)
            {
                const auto simplex_it = simplex_by_index.find(col.pivot);
                if (simplex_it != simplex_by_index.end() &&
                    isApparentPair(dim_simplices[i], *simplex_it->second, cohom_simplices))
                {
                    pivot_to_column[col.pivot] = static_cast<int>(i);
                    Pair pair;
                    pair.dimension = dim;
                    pair.birth = dim_simplices[i].filtration_value;
                    pair.death = simplex_it->second->filtration_value;
                    dim_pairs.push_back(pair);
                    clearing.markCleared(col.pivot);
                    col.coboundary.clear();
                    col.pivot = -1;
                    col.reduced = true;
                    continue;
                }
            }
            // Reduce column
            while (col.pivot >= 0)
            {
                auto it = pivot_to_column.find(col.pivot);
                if (it != pivot_to_column.end())
                {
                    // Add that column
                    addColumn(col, coboundary_matrix[it->second]);
                }
                else
                {
                    // New pivot found
                    pivot_to_column[col.pivot] = static_cast<int>(i);
                    // Record persistence pair
                    Pair pair;
                    pair.dimension = dim;
                    pair.birth = dim_simplices[i].filtration_value;
                    pair.death = cohom_simplices[col.pivot].filtration_value;
                    dim_pairs.push_back(pair);
                    // Mark death simplex as cleared (for next dimension)
                    clearing.markCleared(col.pivot);
                    break;
                }
            }
            if (col.pivot < 0)
            {
                // Essential class (infinite persistence)
                Pair pair;
                pair.dimension = dim;
                pair.birth = dim_simplices[i].filtration_value;
                pair.death = std::numeric_limits<double>::infinity();
                dim_pairs.push_back(pair);
            }
        }
        // Store pairs for this dimension
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
    // Compute statistics
    result.num_simplices = static_cast<int>(simplices.size());
    result.num_cleared = static_cast<int>(result.all_pairs.size()); // Approximate
    return result;
}
// Extract homology from cohomology
// The persistence diagrams are the same
std::vector<Pair> extractHomologyFromCohomology(const CohomologyResult &result)
{
    // For Z2 coefficients, persistent cohomology and homology give same barcodes
    // Universal Coefficient Theorem for persistence
    return result.all_pairs;
}
// Get optimal cohomology configuration
CohomologyConfig getOptimalCohomologyConfig(int max_dim, size_t num_simplices)
{
    CohomologyConfig config;
    config.max_dimension = max_dim;
    config.use_cohomology = true;
    config.use_clearing = true;                          // Always beneficial
    config.use_apparent_pairs = (num_simplices > 10000); // Worth overhead for large
    config.use_dimension_cascade = (max_dim >= 2);       // Beneficial for H2+
    return config;
}
// Estimate speedup vs homology
CohomologySpeedupEstimate estimateCohomologySpeedup(int max_dim, size_t num_simplices,
                                                    bool use_clearing, bool use_apparent_pairs)
{
    CohomologySpeedupEstimate estimate;
    // Base speedup from cohomology scales with workload size.
    double size_scaling = 1.0;
    if (num_simplices > 100000)
    {
        size_scaling = 1.25;
    }
    else if (num_simplices > 10000)
    {
        size_scaling = 1.1;
    }
    estimate.cohomology_speedup = 1.5 * size_scaling;
    // Additional from clearing
    if (use_clearing)
    {
        estimate.clearing_speedup = 2.0 + (max_dim * 0.5); // More dimensions = more benefit
    }
    // Additional from apparent pairs
    if (use_apparent_pairs)
    {
        // Typically 10-30% of pairs are apparent
        estimate.apparent_pairs_speedup = 1.3;
    }
    // Combined
    estimate.total_speedup =
        estimate.cohomology_speedup * estimate.clearing_speedup * estimate.apparent_pairs_speedup;
    // Memory benefit: cohomology often needs less memory
    estimate.memory_reduction_ratio = 0.3; // ~30% less memory
    return estimate;
}
// Check if cohomology should be used
bool shouldUseCohomology(int max_dim, size_t num_simplices, bool require_representatives)
{
    // Cohomology is faster but doesn't directly give homology representatives
    // If representatives needed, use homology or convert after
    if (require_representatives)
    {
        // Can still use cohomology but need conversion
        // Homology provides representatives directly
        // Or use: return true; // and convert cohomology reps to homology
    }
    // Cohomology beneficial for:
    // - Higher dimensions (H2+)
    // - Large simplex counts
    return max_dim >= 1 && num_simplices > 1000;
}
} // namespace nerve::persistence
