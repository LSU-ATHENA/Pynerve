
#pragma once

#include "nerve/core.hpp"

#include <chrono>
#include <unordered_map>
#include <vector>

namespace nerve::persistence::clearcompress
{

/**
 * @brief Persistence pair from chunk reduction
 */
struct PersistencePair
{
    int birth_index = -1;
    int death_index = -1;
    double birth_time = 0.0;
    double death_time = 0.0;
};

/**
 * @brief Column within a chunk
 */
struct ChunkColumn
{
    int global_index = -1;    // Index in full matrix
    std::vector<int> indices; // Sparse column entries
    bool is_cleared = false;  // Whether already paired
};

/**
 * @brief Chunk of matrix columns
 */
struct Chunk
{
    int chunk_index = -1; // Chunk identifier
    int start_column = 0; // First column index (global)
    int end_column = 0;   // Last column index (global)
    std::vector<ChunkColumn> columns;
    bool is_compressed = false; // Whether compression applied
};

/**
 * @brief Chunk processing configuration
 */
struct ChunkConfig
{
    size_t chunk_size = 65536; // Columns per chunk
    bool compress_after_reduction = true;
    bool interleave_processing = false;
    int num_threads = 0; // 0 = auto
};

/**
 * @brief Result of single chunk reduction
 */
struct ChunkReductionResult
{
    int chunk_index = -1;
    std::vector<PersistencePair> pairs;
    double reduction_time_ms = 0.0;
    int xor_operations = 0;
    double memory_reduction = 0.0; // Fraction of memory saved
};

/**
 * @brief Result of clear-and-compress reduction
 */
struct ClearCompressResult
{
    std::vector<PersistencePair> all_pairs;
    double total_time_ms = 0.0;
    double chunk_build_time_ms = 0.0;
    int num_chunks = 0;
    int total_xor_operations = 0;
    double total_memory_reduction = 0.0; // Average across chunks
    std::vector<double> chunk_times_ms;
};

/**
 * @brief Speedup metrics
 */
struct ClearCompressSpeedup
{
    double memory_reduction = 0.0;         // Memory savings
    double cache_efficiency_speedup = 1.0; // From better cache use
    double compression_overhead = 1.0;     // Negative factor
    double total_speedup = 1.0;
};

/**
 * @brief Clear-and-Compress: Chunked Matrix Reduction
 *
 * **2-3x Memory Reduction + Better Cache Performance**
 *
 * Key Innovation:
 * Process matrix in cache-sized chunks, compress after each chunk.
 * This dramatically reduces memory and improves cache locality.
 *
 * How It Works:
 * ```
 * - Divide matrix into chunks (e.g., 65536 columns each)
 * - Load chunk into cache
 * - Reduce chunk fully
 * - Compress chunk (remove zeros/cleared columns)
 * - Store compressed chunk back
 * - Process next chunk
 * ```
 *
 * Benefits:
 * - **Memory**: 2-3x less (compression removes zeros)
 * - **Cache**: Chunks fit in L3 cache
 * - **Speed**: Better cache locality = faster access
 * - **Scalability**: Process larger matrices
 *
 * Clear + Compress:
 * The "clear" part tracks which simplices are already paired.
 * The "compress" part removes those columns entirely.
 *
 * Best For:
 * - Large matrices (100K+ columns)
 * - When memory is constrained
 * - High-dimensional computation (H5-H6)
 *
 * References:
 * - Bauer et al. "Clear and Compress: Computing Persistent Homology in Chunks"
 * - "Keeping it sparse: Computing Persistent Homology revisited"
 */

/**
 * @brief Build chunks from matrix columns
 */
std::vector<Chunk> buildChunks(const std::vector<std::vector<int>> &matrix_columns,
                               const ChunkConfig &config);

/**
 * @brief Reduce a single chunk
 */
ChunkReductionResult reduceChunk(Chunk &chunk, std::unordered_map<int, int> &global_pivot_map,
                                 const ChunkConfig &config);

/**
 * @brief Process all chunks sequentially
 */
ClearCompressResult reduceMatrixClearCompress(const std::vector<std::vector<int>> &matrix_columns,
                                              const ChunkConfig &config);

/**
 * @brief Parallel chunk processing
 */
ClearCompressResult
reduceMatrixClearCompressParallel(const std::vector<std::vector<int>> &matrix_columns,
                                  const ChunkConfig &config, int num_threads);

/**
 * @brief Interleaved chunk processing for better cache
 */
ClearCompressResult reduceMatrixInterleaved(const std::vector<std::vector<int>> &matrix_columns,
                                            const ChunkConfig &config);

/**
 * @brief Get optimal chunk configuration
 */
ChunkConfig getOptimalChunkConfig(size_t num_columns, int num_rows);

/**
 * @brief Estimate speedup vs standard reduction
 */
ClearCompressSpeedup estimateClearCompressSpeedup(size_t num_columns, int num_rows,
                                                  size_t avg_column_size);

/**
 * @brief Check if clear-compress should be used
 */
bool shouldUseClearCompress(size_t num_columns, size_t avg_column_size);

} // namespace nerve::persistence::clearcompress
