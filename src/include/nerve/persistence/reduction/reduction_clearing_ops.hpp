
#pragma once

#include "nerve/algebra/boundary.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/reduction/reduction_ops.hpp"

#include <unordered_map>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Optimized reducer with cohomology optimization and clearing
 *
 * Implements advanced optimization techniques for persistence computation:
 * - Cohomological reduction (de Silva 2011)
 * - Clearing optimization (Chen & Kerber 2011)
 * - Column compression
 */
class OptimizedReducer : public Reducer
{
public:
    /**
     * @brief Configuration for reduction optimizations
     */
    struct Config
    {
        bool use_cohomology = true;  // de Silva 2011 optimization
        bool use_clearing = true;    // Chen & Kerber 2011
        bool use_compression = true; // Column height reduction
        int max_dimension = 2;

        /// When true, dispatches cohomology reduction to GpuPersistenceReducer
        /// (CUDA column operations). Falls back to CPU if GPU is unavailable.
        /// GPU availability is auto-detected at runtime; this flag gates the dispatch.
        bool enable_gpu = true;
        Size threshold = 10000;
    };

    OptimizedReducer() = delete;
    explicit OptimizedReducer(const algebra::BoundaryMatrix &boundary_matrix);
    explicit OptimizedReducer(const algebra::BoundaryMatrix &boundary_matrix, Config config);

    // Optimized computation methods
    errors::ErrorResult<void> computeOptimized(const core::DeterminismContract &contract = {});

    errors::ErrorResult<void> computeCohomology(const core::DeterminismContract &contract = {});

    errors::ErrorResult<void> cohomologyReductionStep(Size column,
                                                      const core::DeterminismContract &contract);

    errors::ErrorResult<void> applyClearingOptimization(const core::DeterminismContract &contract);

    Size getAcceleratedOperationsCount() const;
    double getOptimizationSpeedup() const;

    // Performance monitoring
    Size getAcceleratedOps() const { return accelerated_ops_; }
    double getSpeedup() const { return speedup_; }
    Size getOperationsSaved() const { return ops_saved_; }

    // Configuration
    void setConfig(const Config &config) { config_ = config; }
    Config getConfig() const { return config_; }

private:
    Config config_;

    // Cohomological reduction implementation
    errors::ErrorResult<void> cohomologyStep(Size column,
                                             const core::DeterminismContract &contract);

    // Clearing optimization
    errors::ErrorResult<void> applyClearing(const core::DeterminismContract &contract);

    // Emergent pair detection (de Silva 2011)
    bool isEmergentPair(Size column) const;
    bool isPositiveSimplex(Size column) const;

    // Performance tracking
    Size accelerated_ops_ = 0;
    double baseline_time_ = 0.0;
    double accelerated_time_ = 0.0;
    double optimized_time_ = 0.0;
    double speedup_ = 1.0;
    Size ops_saved_ = 0;

    const algebra::BoundaryMatrix *matrix_view_ = nullptr;

    // Internal state for optimization
    std::vector<bool> positive_simplices_;
    std::vector<bool> negative_simplices_;
    std::vector<bool> cleared_columns_;
};

} // namespace nerve::persistence
