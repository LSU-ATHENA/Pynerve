#pragma once

#include "nerve/algebra/cellular.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace nerve::persistence
{

// Configuration for cohomology computation.
struct CohomologyConfig
{
    int max_dimension = 6;
    double max_filtration_value = std::numeric_limits<double>::infinity();
    bool use_clearing = true;
    bool use_apparent_pairs = true;
    int num_threads = 1;
    bool use_dimension_cascade = true;
    bool use_cohomology = true;
};

// Result of cohomology computation.
struct CohomologyResult
{
    std::vector<Pair> all_pairs;
    std::vector<std::vector<Pair>> pairs_by_dimension;
    double computation_time_ms = 0.0;
    std::size_t num_reductions = 0;
    std::size_t num_apparent_pairs = 0;
    bool used_clearing = false;
    bool used_apparent_pairs = false;
    double memory_usage_mb = 0.0;
    std::vector<double> dimension_times_ms;
    double total_time_ms = 0.0;
    bool used_cohomology = false;
    std::size_t num_simplices = 0;
    std::size_t num_cleared = 0;
    int max_dim = 6;
};

// Speedup estimation.
struct CohomologySpeedupEstimate
{
    double total_speedup = 1.0;
    double clearing_speedup = 1.0;
    double apparent_pairs_speedup = 1.0;
    double bit_parallel_speedup = 1.0;
    double memory_bandwidth_speedup = 1.0;
    double computational_complexity_speedup = 1.0;
    bool worthwhile = false;
    double cohomology_speedup = 1.0;
    double memory_reduction_ratio = 0.0;
};

// Persistent cohomology computation class.
class PersistentCohomologyComputer
{
public:
    PersistentCohomologyComputer() = default;
    explicit PersistentCohomologyComputer(const algebra::CellularComplex &complex);

    // Compute persistent cohomology from filtration.
    std::vector<Pair> computePersistentCohomology(
        const std::vector<std::pair<algebra::Cell, double>> &filtration) const;

    // Compute persistent cohomology for specific dimension.
    std::vector<Pair>
    computeForDimension(const std::vector<std::pair<algebra::Cell, double>> &filtration,
                        int dimension) const;

    // Get barcode representation.
    std::vector<std::tuple<int, double, double>> getBarcode() const;

private:
    algebra::CellularComplex complex_;
};

} // namespace nerve::persistence
