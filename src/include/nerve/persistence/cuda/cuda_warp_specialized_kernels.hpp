
#pragma once

#include "nerve/core.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace nerve::persistence::gpu
{

/**
 * @brief Configuration for warp specialization
 */
struct WarpSpecializationConfig
{
    bool use_warp_specialization = true; // Enable warp-level parallelism
    bool use_tensor_cores = false;       // Use Tensor Cores (Ampere+)
    bool use_pipelining = true;          // Software pipelining
    bool use_async_copy = false;         // Async data movement (Hopper+)
    int warps_per_block = 8;             // Warps per thread block
    int pipeline_stages = 2;             // Number of pipeline stages
};

/**
 * @brief Benchmark results for warp specialization
 */
struct WarpSpecializationBenchmark
{
    double column_add_time_ms;
    double pivot_find_time_ms;
    double reduction_time_ms;
    double warp_speedup;
    double tensor_core_speedup;
    double pipelining_speedup;
    double total_speedup;
};

/**
 * @brief GPU Warp Specialization for Persistent Homology
 *
 * Provides warp-centric CUDA kernels used by the persistence reduction path.
 * The implementation focuses on:
 * - one-warp-per-column work decomposition,
 * - warp shuffle primitives for intra-warp coordination, and
 * - launch helpers for column add, pivot search, and pipelined reduction.
 *
 * Reported benchmark fields only reflect measurements that are executed in the
 * benchmark routine; optional comparative fields use neutral finite values when
 * no baseline run is performed in-process.
 */

/**
 * @brief Launch warp-specialized column addition
 *
 * Each warp handles one column, performing Z2 XOR in parallel.
 *
 * @param d_columns_a Target columns (in-place modification)
 * @param d_columns_b Source columns to add
 * @param d_col_sizes Size of each column
 * @param num_words Words per column
 * @param num_columns Number of columns
 * @param stream CUDA stream for async execution
 */
void launchWarpSpecializedColumnAdd(uint64_t *d_columns_a, const uint64_t *d_columns_b,
                                    const int *d_col_sizes, int num_words, int num_columns,
                                    cudaStream_t stream = 0);

/**
 * @brief Launch warp-specialized pivot finding
 *
 * Each warp finds the highest set bit in one column.
 *
 * @param d_columns Column data
 * @param d_col_sizes Column sizes
 * @param num_words Words per column
 * @param num_columns Number of columns
 * @param d_pivots Output pivot indices
 * @param stream CUDA stream
 */
void launchWarpSpecializedPivotFind(const uint64_t *d_columns, const int *d_col_sizes,
                                    int num_words, int num_columns, int *d_pivots,
                                    cudaStream_t stream = 0);

/**
 * @brief Launch pipelined reduction
 *
 * Overlaps computation with memory operations for maximum throughput.
 *
 * @param d_columns Column data (in-place)
 * @param d_col_pivots Current pivots
 * @param d_pivot_to_col Pivot to column mapping
 * @param num_words Words per column
 * @param num_columns Number of columns
 * @param d_new_pivots Output new pivots
 * @param stream CUDA stream
 */
void launchPipelinedReduction(uint64_t *d_columns, const int *d_col_pivots,
                              const int *d_pivot_to_col, int num_words, int num_columns,
                              int *d_new_pivots, cudaStream_t stream = 0,
                              bool use_async_copy = false);

/**
 * @brief Benchmark warp specialization
 *
 * Measures performance of different kernel configurations.
 *
 * @param num_columns Number of columns to test
 * @param num_words Words per column
 * @return Benchmark results
 */
WarpSpecializationBenchmark benchmarkWarpSpecialization(int num_columns, int num_words);

/**
 * @brief Check for Tensor Core support
 *
 * @return True if device has Tensor Cores (Volta+)
 */
bool hasTensorCoreSupport();

/**
 * @brief Check for async copy support (TMA)
 *
 * @return True if device supports TMA (Hopper/Blackwell)
 */
bool hasAsyncCopySupport();

/**
 * @brief Get optimal configuration
 *
 * Automatically selects best settings based on hardware and problem size.
 *
 * @param num_columns Number of columns
 * @param num_words Words per column
 * @param has_tensor_cores Whether Tensor Cores are available
 * @return Optimized configuration
 */
WarpSpecializationConfig getOptimalWarpSpecializationConfig(int num_columns, int num_words,
                                                            bool has_tensor_cores);

/**
 * @brief Estimate speedup
 *
 * Provides a coarse architecture weighting used for runtime heuristics.
 *
 * @param major CUDA compute capability major version
 * @param minor CUDA compute capability minor version
 * @return Heuristic architecture factor
 */
inline double estimateWarpSpecSpeedup(int major, int minor)
{
    (void)minor;
    if (major >= 10)
    {
        return 4.0; // Blackwell
    }
    else if (major >= 9)
    {
        return 3.5; // Hopper
    }
    else if (major >= 8)
    {
        return 2.5; // Ampere
    }
    else if (major >= 7)
    {
        return 1.5; // Turing/Volta
    }
    return 1.0; // Older
}

} // namespace nerve::persistence::gpu
