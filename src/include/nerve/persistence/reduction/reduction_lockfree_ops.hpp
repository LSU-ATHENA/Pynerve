
#pragma once

#include "nerve/core.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Fine-grained profiling breakdown for lockfree reduction
 */
struct LockfreeProfile
{
    double column_init_ms = 0.0;      // Copy boundary_matrix into ReductionColumn
    double coboundary_build_ms = 0.0; // Build reverse index
    double atomics_init_ms = 0.0;     // Initialize pivot_to_column atomics
    double queue_setup_ms = 0.0;      // Create work queues + distribute work
    double worker_reduction_ms = 0.0; // Worker thread wall time (reduction)
    double pair_extract_ms = 0.0;     // Extract pairs from reduced columns
    size_t add_column_calls = 0;      // Number of addColumnLockfree calls
    double add_column_total_ms = 0.0; // Total time in addColumnLockfree
    size_t apparent_pairs = 0;        // Number of apparent pairs found
    int empty_columns = 0;            // Empty columns pre-reduced before workers
    int re_reduced_columns = 0;       // Columns re-reduced in post-pass
    int num_threads = 0;
    int num_columns = 0;
    int num_rows = 0;
    size_t nnz = 0; // Total nonzeros in boundary
};

/**
 * @brief Statistics about lockfree parallel reduction
 */
struct LockfreeStats
{
    int num_threads;
    size_t num_columns;
    double columns_per_thread;
    double computation_time_ms;
    double estimated_ideal_speedup;
    double estimated_real_speedup;
    size_t num_atomic_operations;
    size_t cache_line_bounces;
};

/**
 * @brief Lockfree parallel matrix reduction for persistence
 *
 * Based on Morozov & Nigmetov's lockfree fast implementation.
 * Uses work-stealing queues and atomic operations to scale
 * on multicore systems while minimizing lock contention.
 *
 * Key optimizations:
 * - Work-stealing queues for dynamic load balancing
 * - Atomic XOR for Z2 coefficient arithmetic
 * - Apparent pair detection to skip unnecessary work
 * - Lockfree pivot tracking
 *
 * @param boundary_matrix Sparse boundary matrix (columns = simplices, rows = faces)
 * @param filtration_values Filtration value for each simplex
 * @param simplex_dimensions Dimension of each simplex (column index -> dimension)
 * @param num_threads Number of threads (0 = auto-detect)
 * @return Persistence pairs (birth/death)
 */
std::vector<Pair> reduceMatrixLockfree(const std::vector<std::vector<int>> &boundary_matrix,
                                       const std::vector<double> &filtration_values,
                                       const std::vector<double> *row_filtration_values,
                                       const std::vector<Dimension> &simplex_dimensions,
                                       int num_threads = 0);

/**
 * @brief Profiled variant - fills LockfreeProfile with phase breakdown.
 */
std::vector<Pair> reduceMatrixLockfreeProfiled(const std::vector<std::vector<int>> &boundary_matrix,
                                               const std::vector<double> &filtration_values,
                                               const std::vector<double> *row_filtration_values,
                                               const std::vector<Dimension> &simplex_dimensions,
                                               int num_threads, LockfreeProfile *profile);

/**
 * @brief Get statistics about lockfree reduction
 */
LockfreeStats getLockfreeStats(int num_threads, size_t num_columns, double computation_time_ms);

/**
 * @brief Recommended number of threads for lockfree reduction
 *
 * Uses hardware concurrency but caps at practical limits
 * to avoid oversubscription.
 */
inline int recommendedThreadCount()
{
    int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
    // Cap at 32 threads - beyond this, overhead increases
    return std::min(hw_threads, 32);
}

} // namespace nerve::persistence
