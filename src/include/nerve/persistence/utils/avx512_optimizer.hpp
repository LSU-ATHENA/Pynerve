
#pragma once

#include "nerve/core.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

namespace nerve::persistence::avx512
{

/**
 * @brief AVX-512 feature detection results
 */
struct AVX512Features
{
    bool has_avx512f = false;     // Foundation
    bool has_avx512vl = false;    // Vector length extensions
    bool has_avx512bw = false;    // Byte/word extensions
    bool has_avx512dq = false;    // Doubleword/quadword
    bool has_full_avx512 = false; // All above
};

/**
 * @brief AVX-512 benchmark results
 */
struct AVX512Benchmark
{
    double scalar_time_ms = 0.0;
    double avx512_time_ms = 0.0;
    double speedup = 1.0;             // Measured ratio when both timings are available
    double theoretical_speedup = 1.0; // Feature-based throughput factor
};

/**
 * @brief AVX-512 configuration
 */
struct AVX512Config
{
    bool use_avx512 = false;
    bool use_non_temporal = false;
    size_t alignment = 64; // 64-byte alignment for 512-bit
    size_t min_words_for_avx512 = 8;
};

/**
 * @brief Speedup estimate
 */
struct AVX512SpeedupEstimate
{
    double base_speedup = 1.0;             // Vector-lane throughput factor
    double memory_bandwidth_speedup = 1.0; // Cache/memory utilization factor
    double non_temporal_speedup = 1.0;     // Streaming-store factor
    double total_speedup = 1.0;
};

/**
 * @brief Vectorization mode
 */
enum class VectorizationMode
{
    SCALAR,         // No vectorization
    AVX2,           // 256-bit
    AVX512_PARTIAL, // Some AVX-512 features
    AVX512_FULL     // Full AVX-512
};

/**
 * @brief AVX-512 Optimizer for Persistent Homology
 *
 * Provides runtime feature detection and optional AVX-512 kernels for
 * bit-packed persistence operations. 512-bit instructions operate on eight
 * uint64_t lanes per vector operation when the binary is built with AVX-512
 * support and the current CPU exposes the required feature bits.
 *
 * Features:
 * - 512-bit XOR operations
 * - Optional non-temporal stores for streaming writes
 * - CPU feature detection and runtime dispatch
 * - 64-byte alignment-oriented configuration
 *
 * Supported Operations:
 * - Column XOR (Z2 addition)
 * - Column copy
 * - Column zeroing
 * - Pivot finding
 * - Batch operations
 *
 * Requirements:
 * - Compiler and CPU support for AVX-512 when AVX-512 kernels are enabled
 *
 * Baseline:
 * Runtime dispatch falls back to scalar operations when AVX-512 kernels are
 * missing in the binary or unsupported by the CPU.
 *
 * References:
 * - Intel AVX-512 Instruction Set Architecture
 * - "Keeping it sparse: Computing Persistent Homology revisited" - Bauer et al.
 */

/**
 * @brief Detect AVX-512 CPU features
 *
 * Uses CPUID to detect available AVX-512 instruction subsets.
 *
 * @return Feature detection results
 */
AVX512Features detectAVX512Features();

/**
 * @brief 512-bit column addition
 *
 * XORs two columns using 512-bit AVX-512 instructions.
 * Processes 8 uint64_t words (512 bits) per instruction.
 *
 * @param dest Destination column (modified in-place)
 * @param src Source column to XOR
 * @param num_words Number of 64-bit words
 */
#ifdef __AVX512F__
void addBitColumnsAVX512(uint64_t *__restrict__ dest, const uint64_t *__restrict__ src,
                         size_t num_words);

/**
 * @brief 512-bit column addition with non-temporal stores
 *
 * Uses streaming stores that bypass cache.
 * Best for large matrices where cache pollution hurts.
 *
 * @param dest Destination column
 * @param src Source column
 * @param num_words Number of words
 */
void addBitColumnsAVX512Streaming(uint64_t *__restrict__ dest, const uint64_t *__restrict__ src,
                                  size_t num_words);

/**
 * @brief Batch XOR multiple columns
 *
 * Matrix-multiplication-style batch XOR.
 *
 * @param dest Destination
 * @param src1 First source
 * @param src2 Second source
 * @param num_words Number of words
 */
void batchXORAVX512(uint64_t *__restrict__ dest, const uint64_t *__restrict__ src1,
                    const uint64_t *__restrict__ src2, size_t num_words);

/**
 * @brief Find pivot using AVX-512
 *
 * Searches for highest set bit using vectorized scan.
 *
 * @param words Column words
 * @param num_words Number of words
 * @return Pivot index or -1
 */
int findPivotAVX512(const uint64_t *words, size_t num_words);

/**
 * @brief Zero a column using AVX-512
 *
 * Fast column initialization.
 *
 * @param dest Destination column
 * @param num_words Number of words
 */
void zeroColumnAVX512(uint64_t *dest, size_t num_words);

/**
 * @brief Copy column using AVX-512
 *
 * Fast column copy with 512-bit moves.
 *
 * @param dest Destination
 * @param src Source
 * @param num_words Number of words
 */
void copyColumnAVX512(uint64_t *__restrict__ dest, const uint64_t *__restrict__ src,
                      size_t num_words);
#endif // __AVX512F__

/**
 * @brief Auto-dispatch optimized addition
 *
 * Selects the compiled AVX-512 kernel when both binary and CPU support are
 * present; otherwise uses scalar word-wise XOR.
 *
 * @param dest Destination column
 * @param src Source column
 * @param num_words Number of words
 */
void addBitColumnsOptimized(uint64_t *dest, const uint64_t *src, size_t num_words);

/**
 * @brief Benchmark AVX-512 vs scalar
 *
 * Measures actual speedup on current hardware.
 *
 * @param num_words Column size in words
 * @param iterations Number of iterations
 * @return Benchmark results
 */
AVX512Benchmark benchmarkAVX512(size_t num_words, int iterations = 1000);

/**
 * @brief Get optimal vectorization mode
 *
 * @return Best available vectorization mode
 */
VectorizationMode getOptimalVectorizationMode();

/**
 * @brief Get optimal AVX-512 configuration
 *
 * @param num_columns Number of columns
 * @param num_rows Number of rows
 * @return Optimized configuration
 */
AVX512Config getOptimalAVX512Config(size_t num_columns, int num_rows);

/**
 * @brief Estimate speedup for given problem
 *
 * @param num_words Column size
 * @return Speedup estimate
 */
AVX512SpeedupEstimate estimateAVX512Speedup(size_t num_words);

/**
 * @brief Check if AVX-512 should be used
 *
 * @param num_words Column size
 * @return True if AVX-512 beneficial
 */
inline bool shouldUseAVX512(size_t num_words)
{
#ifdef __AVX512F__
    return num_words >= 8 && detectAVX512Features().has_avx512f;
#else
    (void)num_words;
    return false;
#endif
}

} // namespace nerve::persistence::avx512
