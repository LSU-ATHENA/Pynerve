
#pragma once

#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"

#include <cstddef>
#include <vector>

namespace nerve::persistence::adaptive_acceleration
{

class SparseMatrix;
struct SystemCapabilities;

#ifndef NERVE_ADAPTIVE_ACCELERATION_PROBLEM_CHARACTERISTICS_DEFINED
#define NERVE_ADAPTIVE_ACCELERATION_PROBLEM_CHARACTERISTICS_DEFINED
struct ProblemCharacteristics
{
    std::size_t num_points = 0;
    std::size_t point_dim = 0;
    std::size_t max_simplex_size = 0;
    std::size_t estimated_columns = 0;

    double density = 0.0;
    double apparent_pair_ratio = 0.0;
    double sparsity_ratio = 0.0;
    bool is_dense = false;
    bool is_sparse = false;
    bool isHighDimensional = false;
    bool hasRegularStructure = false;

    double estimated_complexity = 0.0;
    double memory_requirement_mb = 0.0;

    bool suitable_for_matrix_multiplication = false;
    bool suitable_for_sparsification = false;
    bool suitable_for_lockfree_multicore = false;
    bool suitable_for_gpu = false;
    bool suitable_for_streaming = false;
};
#endif

class ProblemAnalyzer
{
public:
    static ProblemCharacteristics analyzeProblem(const core::BufferView<const double> &points,
                                                 std::size_t point_dim);

    static ProblemCharacteristics analyzeMatrix(const SparseMatrix &matrix);

    static double estimateComplexity(const ProblemCharacteristics &problem);
    static double estimateMemoryRequirement(const ProblemCharacteristics &problem);
    static double estimateApparentPairRatio(const ProblemCharacteristics &problem);

    static bool shouldUseMatrixMultiplication(const ProblemCharacteristics &problem);
    static bool shouldUseSparsification(const ProblemCharacteristics &problem);
    static bool should_use_lockfree_multicore(const ProblemCharacteristics &problem,
                                              const SystemCapabilities &system);
    static bool shouldUseGpu(const ProblemCharacteristics &problem,
                             const SystemCapabilities &system);

private:
    static double computeDensity(const core::BufferView<const double> &points,
                                 std::size_t point_dim);
    static std::size_t estimateMaxSimplexSize(std::size_t num_points, std::size_t point_dim,
                                              double max_radius);
    static std::size_t estimateNumColumns(std::size_t num_points, std::size_t max_simplex_size);
    static double estimateSparsityRatio(std::size_t num_points, std::size_t max_simplex_size);
    static bool isHighDimensional(std::size_t point_dim);
    static bool hasRegularStructure(const core::BufferView<const double> &points);
};

} // namespace nerve::persistence::adaptive_acceleration
