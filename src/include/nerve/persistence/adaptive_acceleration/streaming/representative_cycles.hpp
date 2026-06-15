
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/adaptive_acceleration/sparse_matrix.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace nerve::persistence::adaptive_acceleration::representative
{

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

    std::size_t size() const { return vertices.size(); }

    double persistence() const { return death_time - birth_time; }
};

struct RepresentativeConfig
{
    bool use_matrix_multiplication = true;
    bool use_dual_cohomology = false;
    bool enable_fast_computation = true;
    bool enable_compression = true;
    bool enable_parallel_computation = true;
    std::size_t max_cycles_per_dimension = 1000;
    double min_persistence = 1e-6;
    bool enable_visualization_data = true;
};

struct RepresentativeStats
{
    double computation_time_ms = 0.0;
    std::size_t total_cycles_computed = 0;
    std::size_t cycles_per_dimension[4] = {0, 0, 0, 0};
    double average_cycle_size = 0.0;
    double max_cycle_size = 0.0;
    double min_cycle_size = 0.0;
    bool used_matrix_multiplication = false;
    bool used_dual_cohomology = false;
    std::string computation_details;
};

class RepresentativeCycleComputer
{
public:
    ~RepresentativeCycleComputer();

    static errors::ErrorResult<std::unique_ptr<RepresentativeCycleComputer>>
    create(const RepresentativeConfig &config);

    errors::ErrorResult<std::vector<Cycle>>
    computeRepresentativesFast(const SparseMatrix &reduced_matrix);

    errors::ErrorResult<std::vector<Cycle>>
    computeWithDualCohomology(const SparseMatrix &boundary_matrix,
                              const SparseMatrix &reduced_matrix);

    errors::ErrorResult<std::vector<Cycle>> computeForDimension(const SparseMatrix &reduced_matrix,
                                                                Dimension dimension);

    errors::ErrorResult<std::vector<Cycle>> computeParallel(const SparseMatrix &reduced_matrix,
                                                            std::size_t num_threads);

    const RepresentativeStats &getComputationStats() const { return computation_stats_; }

private:
    explicit RepresentativeCycleComputer(const RepresentativeConfig &config);

    class Impl;
    std::unique_ptr<Impl> impl_;
    RepresentativeConfig config_;
    RepresentativeStats computation_stats_;

    errors::ErrorResult<std::vector<Cycle>>
    computeCyclesStandard(const SparseMatrix &reduced_matrix);

    errors::ErrorResult<std::vector<Cycle>>
    computeCyclesMatrixMultiplication(const SparseMatrix &reduced_matrix);

    errors::ErrorResult<std::vector<Cycle>>
    computeCyclesDualCohomology(const SparseMatrix &boundary_matrix,
                                const SparseMatrix &reduced_matrix);

    errors::ErrorResult<Cycle> extractSingleCycle(const SparseMatrix &matrix,
                                                  std::size_t column_index, Dimension dimension);

    errors::ErrorResult<void> validateCycle(const Cycle &cycle);
};

class DualCohomologyComputer
{
public:
    ~DualCohomologyComputer();

    static errors::ErrorResult<std::unique_ptr<DualCohomologyComputer>>
    create(const RepresentativeConfig &config);

    errors::ErrorResult<std::vector<Cycle>>
    computeCohomologyRepresentatives(const SparseMatrix &boundary_matrix,
                                     const SparseMatrix &reduced_matrix);

    errors::ErrorResult<std::vector<Cycle>>
    convertToHomology(const std::vector<Cycle> &cohomology_cycles,
                      const SparseMatrix &boundary_matrix);

private:
    explicit DualCohomologyComputer(const RepresentativeConfig &config);

    class Impl;
    std::unique_ptr<Impl> impl_;
    RepresentativeConfig config_;
};

struct CycleVisualizationData
{
    std::vector<std::vector<double>> vertices;
    std::vector<std::vector<int>> edges;
    std::vector<double> edge_weights;
    Dimension dimension = 0;
    double persistence = 0.0;
    std::string visualization_format;

    bool isValid() const
    {
        const bool finite_persistence = std::isfinite(persistence);
        const bool infinite_persistence = std::isinf(persistence) && persistence > 0.0;
        return !vertices.empty() && dimension >= 0 &&
               (finite_persistence || infinite_persistence) &&
               std::all_of(vertices.begin(), vertices.end(),
                           [](const auto &point) {
                               return std::all_of(point.begin(), point.end(), [](double value) {
                                   return std::isfinite(value);
                               });
                           }) &&
               std::all_of(edge_weights.begin(), edge_weights.end(),
                           [](double value) { return std::isfinite(value); });
    }
};

class CycleVisualizer
{
public:
    ~CycleVisualizer();

    static errors::ErrorResult<std::unique_ptr<CycleVisualizer>>
    create(const RepresentativeConfig &config);

    errors::ErrorResult<CycleVisualizationData>
    generateVisualizationData(const Cycle &cycle, const core::BufferView<const double> &points,
                              std::size_t point_dim);

    errors::ErrorResult<std::vector<CycleVisualizationData>>
    generateVisualizationData(const std::vector<Cycle> &cycles,
                              const core::BufferView<const double> &points, std::size_t point_dim);

private:
    explicit CycleVisualizer(const RepresentativeConfig &config);

    class Impl;
    std::unique_ptr<Impl> impl_;
    RepresentativeConfig config_;
};

class CycleValidator
{
public:
    static errors::ErrorResult<bool> validateCycle(const Cycle &cycle,
                                                   const SparseMatrix &boundary_matrix);

    static errors::ErrorResult<std::vector<bool>>
    validateCycles(const std::vector<Cycle> &cycles, const SparseMatrix &boundary_matrix);

    static bool isBoundaryCycle(const Cycle &cycle, const SparseMatrix &boundary_matrix);

    static bool isCycle(const Cycle &cycle, const SparseMatrix &boundary_matrix);

private:
    static bool checkBoundaryCondition(const Cycle &cycle, const SparseMatrix &boundary_matrix);

    static bool checkCycleCondition(const Cycle &cycle, const SparseMatrix &boundary_matrix);
};

class RepresentativeCycleFactory
{
public:
    static errors::ErrorResult<std::unique_ptr<RepresentativeCycleComputer>>
    createFast(const RepresentativeConfig &config);

    static errors::ErrorResult<std::unique_ptr<RepresentativeCycleComputer>>
    createDualCohomology(const RepresentativeConfig &config);

    static errors::ErrorResult<std::unique_ptr<RepresentativeCycleComputer>>
    createParallel(const RepresentativeConfig &config);

    static errors::ErrorResult<std::unique_ptr<RepresentativeCycleComputer>>
    createForVisualization(const RepresentativeConfig &config);
};

} // namespace nerve::persistence::adaptive_acceleration::representative
