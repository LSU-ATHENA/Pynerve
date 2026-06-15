
#pragma once

#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace nerve::persistence::adaptive_acceleration
{

class SparseMatrix;
struct ProblemCharacteristics;

struct MatrixMultiplicationConfig
{
    bool use_strassen = false;
    bool use_coppersmith_winograd = false;
    size_t crossover_threshold = 1000;
    double sparsity_threshold = 0.1;
    bool enable_streaming = false;
    size_t memory_limit = 0;
};

struct Cycle
{
    std::vector<int> vertices;
    std::vector<double> coefficients;
    Dimension dimension = 0;
    double birth_time = 0.0;
    double death_time = 0.0;

    bool isValid() const
    {
        const bool finite_death = std::isfinite(death_time);
        const bool infinite_death = std::isinf(death_time) && death_time > 0.0;
        return !vertices.empty() && vertices.size() == coefficients.size() && dimension >= 0 &&
               std::isfinite(birth_time) && (finite_death || infinite_death) &&
               (!finite_death || death_time >= birth_time) &&
               std::all_of(coefficients.begin(), coefficients.end(),
                           [](double value) { return std::isfinite(value); });
    }
};

struct PerformanceStats
{
    double computation_time_ms = 0.0;
    size_t memory_used_bytes = 0;
    size_t operations_performed = 0;
    double sparsity_ratio = 0.0;
    AlgorithmType algorithm_used = AlgorithmType::STANDARD_CPU;
    std::string optimization_details;
};

class MatrixMultiplicationEngine
{
public:
    static errors::ErrorResult<std::unique_ptr<MatrixMultiplicationEngine>>
    create(const MatrixMultiplicationConfig &config);

    errors::ErrorResult<std::vector<Pair>> compute(const SparseMatrix &boundary_matrix,
                                                   const ProblemCharacteristics &problem);

    errors::ErrorResult<std::vector<Cycle>>
    computeRepresentativesFast(const SparseMatrix &reduced_matrix);

    errors::ErrorResult<SparseMatrix> fastMatrixMultiply(const SparseMatrix &a,
                                                         const SparseMatrix &b);

    const PerformanceStats &getPerformanceStats() const;

    ~MatrixMultiplicationEngine();

private:
    explicit MatrixMultiplicationEngine(const MatrixMultiplicationConfig &config);

    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nerve::persistence::adaptive_acceleration
