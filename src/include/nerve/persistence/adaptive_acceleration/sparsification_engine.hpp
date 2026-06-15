
#pragma once

#include "nerve/core.hpp"
#include "nerve/errors/errors.hpp"

#include <atomic>
#include <memory>
#include <vector>

namespace nerve::persistence::adaptive_acceleration
{

// Forward declarations
class SparseMatrix;
struct ProblemCharacteristics;

/**
 * @brief Sparsification strategies for maintaining matrix sparsity
 */
enum class SparsificationStrategy
{
    SWAP_REDUCTION,          // O(n log n) operations, good for moderate sparsity
    EXHAUSTIVE_REDUCTION,    // Maximum sparsity, O(n^2) operations
    RETROSPECTIVE_REDUCTION, // Balance speed and sparsity
    ADAPTIVE_HYBRID          // Automatic strategy selection
};

/**
 * @brief Statistics for sparsification operations
 */
struct SparsificationStats
{
    double computation_time_ms = 0.0;
    size_t original_nonzero_count = 0;
    size_t sparsified_nonzero_count = 0;
    double sparsity_ratio = 0.0;
    double speedup_factor = 1.0;
    SparsificationStrategy strategy_used = SparsificationStrategy::ADAPTIVE_HYBRID;
    std::string optimization_details;
};

/**
 * @brief Configuration for sparsification optimization
 */
struct SparsificationConfig
{
    SparsificationStrategy strategy = SparsificationStrategy::ADAPTIVE_HYBRID;
    double target_sparsity = 0.5;     // Target sparsity ratio
    double max_sparsity_loss = 0.001; // Maximum mathematical accuracy loss
    bool preserve_diagonal = true;    // Keep diagonal elements
    bool enable_compression = true;   // Compress sparse storage
};

/**
 * @brief Sparsification engine implementing multiple reduction variants
 *
 * This engine implements multiple sparsification strategies that maintain
 * mathematical correctness while reducing matrix density for downstream kernels.
 *
 * Key innovations:
 * - Swap reduction with O(n log n) complexity
 * - Exhaustive reduction for maximum sparsity
 * - Retrospective reduction for balanced optimization
 * - Adaptive strategy selection
 */
class SparsificationEngine
{
public:
    /**
     * @brief Create a sparsification engine
     * @param config Configuration for sparsification
     * @return ErrorResult containing engine instance or error
     */
    static errors::ErrorResult<std::unique_ptr<SparsificationEngine>>
    create(const SparsificationConfig &config);

    ~SparsificationEngine();

    /**
     * @brief Sparsify a matrix using selected strategy
     * @param matrix Input matrix to sparsify
     * @param strategy Sparsification strategy to use
     * @param target_sparsity Target sparsity ratio
     * @return ErrorResult containing sparsified matrix or error
     */
    errors::ErrorResult<SparseMatrix>
    sparsify(const SparseMatrix &matrix, SparsificationStrategy strategy, double target_sparsity);

    /**
     * @brief Automatically select optimal sparsification strategy
     * @param problem Problem characteristics for optimization
     * @return Optimal sparsification strategy
     */
    static SparsificationStrategy selectOptimalStrategy(const ProblemCharacteristics &problem);

    /**
     * @brief Estimate sparsification effectiveness
     * @param matrix Input matrix
     *param strategy Sparsification strategy
     * @return Estimated speedup factor (1.0 = no improvement)
     */
    static double estimateSparsificationBenefit(const SparseMatrix &matrix,
                                                SparsificationStrategy strategy);

    /**
     * @brief Get performance statistics
     * @return Performance statistics from last sparsification
     */
    const SparsificationStats &getPerformanceStats() const { return performance_stats_; }

private:
    SparsificationEngine(const SparsificationConfig &config);

    // Implementation details
    class Impl;
    std::unique_ptr<Impl> impl_;
    SparsificationStats performance_stats_;

    // Sparsification algorithms
    static errors::ErrorResult<SparseMatrix> swapReduction(const SparseMatrix &matrix);

    static errors::ErrorResult<SparseMatrix> exhaustiveReduction(const SparseMatrix &matrix);

    static errors::ErrorResult<SparseMatrix> retrospectiveReduction(const SparseMatrix &matrix);

    static errors::ErrorResult<SparseMatrix> adaptiveHybrid(const SparseMatrix &matrix,
                                                            const ProblemCharacteristics &problem);

    // Helper methods
    static bool shouldPreserveElement(const SparseMatrix &matrix, size_t row, size_t col,
                                      SparsificationStrategy strategy);

    static double computeSparsificationLoss(const SparseMatrix &original,
                                            const SparseMatrix &sparsified);
};

} // namespace nerve::persistence::adaptive_acceleration
