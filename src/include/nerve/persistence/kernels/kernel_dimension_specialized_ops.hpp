// Compatibility facade for dimension-specialized kernels.

#pragma once

#include "nerve/persistence/kernels/dimension_specialized_kernels.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace nerve::persistence::specialized
{

using DimensionConfig = kernels::DimensionConfig;
using H0Result = kernels::H0Result;
using H12Result = kernels::H12Result;
using H36Result = kernels::H36Result;
using DimensionSpecializedResult = kernels::DimensionSpecializedResult;

struct SpecializedBenchmark
{
    double standard_time_ms = 0.0;
    double cohomology_time_ms = 0.0;
    double involuted_time_ms = 0.0;
    double bit_parallel_time_ms = 0.0;
    double specialized_time_ms = 0.0;
    double speedup_vs_standard = 1.0;
};

struct DimensionSpeedupEstimate
{
    std::string algorithm;
    double speedup = 1.0;
};

DimensionSpecializedResult computeDimensionSpecialized(
    const std::vector<std::vector<int>> &simplices, const std::vector<double> &filtration_values,
    const std::vector<int> &dimensions, int max_dim, const DimensionConfig &config);

DimensionConfig getOptimalDimensionConfig(size_t num_simplices, int max_dim, size_t num_points);

SpecializedBenchmark benchmarkSpecialized(const std::vector<std::vector<int>> &simplices,
                                          const std::vector<double> &filtration_values,
                                          const std::vector<int> &dimensions, int max_dim);

DimensionSpeedupEstimate estimateDimensionSpeedup(int dim, size_t num_simplices, size_t num_points);

inline DimensionSpecializedResult
compute0To6DSpecialized(const std::vector<std::vector<int>> &simplices,
                        const std::vector<double> &filtration_values,
                        const std::vector<int> &dimensions)
{
    int max_dim = 0;
    for (int d : dimensions)
    {
        max_dim = std::max(max_dim, d);
    }
    max_dim = std::min(max_dim, 6);
    const auto config = getOptimalDimensionConfig(simplices.size(), max_dim, 0);
    return computeDimensionSpecialized(simplices, filtration_values, dimensions, max_dim, config);
}

} // namespace nerve::persistence::specialized
