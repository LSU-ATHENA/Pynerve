
#pragma once

#include "nerve/core.hpp"

#include <chrono>
#include <compare>
#include <cstddef>
#include <vector>

namespace nerve
{
namespace persistence
{
namespace distilled
{

struct DistilledVRConfig
{
    int max_dim = 6;
    double max_radius = 1.0;
    bool use_bit_parallel = true;
    bool parallel_construction = true;
};

struct DistilledSimplex
{
    std::vector<int> vertices;
    int dimension = 0;
    double filtration_value = 0.0;
    int original_index = -1;

    [[nodiscard]] auto operator<=>(const DistilledSimplex &other) const
    {
        return filtration_value <=> other.filtration_value;
    }
    [[nodiscard]] bool operator==(const DistilledSimplex &other) const = default;
};

struct DistilledPair
{
    int birth_index = -1;
    int death_index = -1;
    double birth_time = 0.0;
    double death_time = 0.0;
    int dimension = 0;
};

struct DistilledFiltration
{
    std::vector<DistilledSimplex> simplices;
    int original_complex_size = 0;
    int distilled_size = 0;
    double reduction_ratio = 0.0;
    DistilledVRConfig config;

    double build_time_ms = 0.0;
    double morse_time_ms = 0.0;
    double total_time_ms = 0.0;
};

struct DistilledPersistenceResult
{
    std::vector<DistilledPair> pairs;
    double computation_time_ms = 0.0;
    bool used_bit_parallel = false;
};

struct DistilledSpeedupEstimate
{
    double memory_reduction = 1.0;
    double cache_efficiency_speedup = 1.0;
    double bit_parallel_speedup = 1.0;
    double total_speedup = 1.0;
};

DistilledFiltration buildDistilledFiltration(const std::vector<double> &points, size_t point_dim,
                                             size_t num_points, const DistilledVRConfig &config);

DistilledPersistenceResult computePersistenceDistilled(const DistilledFiltration &filtration,
                                                       const DistilledVRConfig &config);

DistilledVRConfig getOptimalDistilledVRConfig(size_t num_points, int max_dim, double max_radius);

DistilledSpeedupEstimate estimateDistilledSpeedup(size_t num_points, int max_dim,
                                                  double max_radius);

inline bool shouldUseDistilledVR(size_t num_points, int max_dim)
{
    return num_points > 500 || max_dim >= 3;
}

} // namespace distilled
} // namespace persistence
} // namespace nerve
