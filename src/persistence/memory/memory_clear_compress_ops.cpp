// Chunked clear-and-compress reduction for sparse boundary matrices.

#include "nerve/persistence/memory/clear_and_compress.hpp"
#include "nerve/persistence/utils/bit_parallel_z2.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace nerve::persistence::clearcompress
{

namespace
{

// Density-based chunk size configuration
constexpr double DENSITY_VERY_SPARSE_THRESHOLD = 0.01;
constexpr double DENSITY_SPARSE_THRESHOLD = 0.1;
constexpr size_t CHUNK_SIZE_VERY_SPARSE = 65536; // Very sparse matrices
constexpr size_t CHUNK_SIZE_SPARSE = 32768;      // Sparse matrices
constexpr size_t CHUNK_SIZE_DENSE = 16384;       // Dense matrices

// Cache and memory speedup factors
constexpr double MEMORY_REDUCTION_ESTIMATE = 2.5;
constexpr double CACHE_EFFICIENCY_BASE = 1.0;
constexpr double CACHE_EFFICIENCY_MULTIPLIER = 2.0;
constexpr double CACHE_EFFICIENCY_FIT = 1.2;
constexpr double COMPRESSION_OVERHEAD_FACTOR = 0.9; // 10% overhead

constexpr size_t BYTES_PER_MB = 1024ULL * 1024ULL;

// L3 Cache Size Constants (in bytes)
constexpr size_t L3_CACHE_SIZE_ESTIMATE = 8 * BYTES_PER_MB;      // 8MB L3 estimate
constexpr size_t L3_CACHE_SIZE_CONSERVATIVE = 16 * BYTES_PER_MB; // 16MB conservative estimate

void validateColumnIndices(const std::vector<int> &column)
{
    if (std::ranges::any_of(column, [](int row) { return row < 0; }))
    {
        throw std::invalid_argument("clear-compress boundary rows must be non-negative");
    }
}

void validateChunkConfig(const ChunkConfig &config)
{
    if (config.chunk_size == 0)
    {
        throw std::invalid_argument("clear-compress chunk_size must be positive");
    }
}

void validateChunk(const Chunk &chunk)
{
    if (chunk.chunk_index < 0 || chunk.start_column < 0 || chunk.end_column < chunk.start_column)
    {
        throw std::invalid_argument("clear-compress chunk metadata is invalid");
    }
    for (const auto &column : chunk.columns)
    {
        if (column.global_index < 0)
        {
            throw std::invalid_argument("clear-compress column index must be non-negative");
        }
        validateColumnIndices(column.indices);
    }
}

bool compressColumn(std::vector<int> &column)
{
    validateColumnIndices(column);
    std::ranges::sort(column);
    const auto [first, last] = std::ranges::unique(column);
    column.erase(first, last);

    return true;
}

size_t estimateChunkMemory(const Chunk &chunk)
{
    size_t total = 0;
    for (const auto &col : chunk.columns)
    {
        total += col.indices.size() * sizeof(int);
    }
    return total;
}

void compactChunk(Chunk &chunk)
{
    chunk.columns.erase(std::remove_if(chunk.columns.begin(), chunk.columns.end(),
                                       [](const ChunkColumn &col) { return col.indices.empty(); }),
                        chunk.columns.end());

    for (auto &col : chunk.columns)
    {
        compressColumn(col.indices);
        if (col.indices.empty())
        {
            col.is_cleared = true;
        }
    }
    chunk.is_compressed = true;
}

void addSparseColumns(std::vector<int> &a, const std::vector<int> &b)
{
    std::vector<int> result;
    result.reserve(a.size() + b.size());

    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(),
                                  std::back_inserter(result));

    a = std::move(result);
}

int findPivotSparse(const std::vector<int> &col)
{
    if (col.empty())
        return -1;
    return col.back(); // Highest index
}

} // namespace

std::vector<Chunk> buildChunks(const std::vector<std::vector<int>> &matrix_columns,
                               const ChunkConfig &config)
{
    validateChunkConfig(config);
    if (matrix_columns.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("clear-compress column count exceeds int range");
    }
    for (const auto &column : matrix_columns)
    {
        validateColumnIndices(column);
    }

    std::vector<Chunk> chunks;

    size_t columns_per_chunk = config.chunk_size;
    size_t num_chunks = (matrix_columns.size() + columns_per_chunk - 1) / columns_per_chunk;

    chunks.reserve(num_chunks);

    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
    {
        Chunk chunk;
        chunk.chunk_index = static_cast<int>(chunk_idx);
        chunk.start_column = static_cast<int>(chunk_idx * columns_per_chunk);
        chunk.end_column =
            static_cast<int>(std::min((chunk_idx + 1) * columns_per_chunk, matrix_columns.size()));
        chunk.is_compressed = false;

        for (int col_idx = chunk.start_column; col_idx < chunk.end_column; ++col_idx)
        {
            ChunkColumn col;
            col.global_index = col_idx;
            col.indices = matrix_columns[col_idx];
            col.is_cleared = false;
            chunk.columns.push_back(std::move(col));
        }

        chunks.push_back(std::move(chunk));
    }

    return chunks;
}

ChunkReductionResult reduceChunk(Chunk &chunk, std::unordered_map<int, int> &global_pivot_map,
                                 const ChunkConfig &config)
{
    validateChunkConfig(config);
    validateChunk(chunk);
    ChunkReductionResult result;
    result.chunk_index = chunk.chunk_index;

    auto start = std::chrono::high_resolution_clock::now();

    std::unordered_map<int, int> local_pivot_map;

    size_t memory_before = estimateChunkMemory(chunk);

    for (auto &col : chunk.columns)
    {
        if (col.is_cleared)
            continue;

        int pivot = findPivotSparse(col.indices);

        while (pivot >= 0)
        {
            auto local_it = local_pivot_map.find(pivot);
            if (local_it != local_pivot_map.end())
            {
                addSparseColumns(col.indices, chunk.columns[local_it->second].indices);
                pivot = findPivotSparse(col.indices);
                ++result.xor_operations;
                continue;
            }

            auto global_it = global_pivot_map.find(pivot);
            if (global_it != global_pivot_map.end())
            {
                break;
            }
            local_pivot_map[pivot] = col.global_index - chunk.start_column;
            PersistencePair pair;
            pair.birth_index = col.global_index;
            pair.death_index = pivot;
            result.pairs.push_back(pair);

            break;
        }

        if (pivot < 0)
        {
            PersistencePair pair;
            pair.birth_index = col.global_index;
            pair.death_index = -1;
            result.pairs.push_back(pair);
        }
    }

    if (config.compress_after_reduction)
    {
        compactChunk(chunk);
    }

    for (const auto &[pivot, local_idx] : local_pivot_map)
    {
        global_pivot_map[pivot] = chunk.columns[local_idx].global_index;
    }

    size_t memory_after = estimateChunkMemory(chunk);
    result.memory_reduction =
        (memory_before > 0)
            ? (1.0 - static_cast<double>(memory_after) / static_cast<double>(memory_before))
            : 0.0;

    auto end = std::chrono::high_resolution_clock::now();
    result.reduction_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

ClearCompressResult reduceMatrixClearCompress(const std::vector<std::vector<int>> &matrix_columns,
                                              const ChunkConfig &config)
{
    ClearCompressResult result;

    auto start_total = std::chrono::high_resolution_clock::now();

    auto start_build = std::chrono::high_resolution_clock::now();
    auto chunks = buildChunks(matrix_columns, config);
    auto end_build = std::chrono::high_resolution_clock::now();
    result.chunk_build_time_ms =
        std::chrono::duration<double, std::milli>(end_build - start_build).count();

    result.num_chunks = static_cast<int>(chunks.size());

    std::unordered_map<int, int> global_pivot_map;

    for (auto &chunk : chunks)
    {
        auto chunk_result = reduceChunk(chunk, global_pivot_map, config);
        for (const auto &pair : chunk_result.pairs)
        {
            result.all_pairs.push_back(pair);
        }

        result.chunk_times_ms.push_back(chunk_result.reduction_time_ms);
        result.total_xor_operations += chunk_result.xor_operations;
        result.total_memory_reduction += chunk_result.memory_reduction;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();

    if (result.num_chunks > 0)
    {
        result.total_memory_reduction /= result.num_chunks;
    }

    return result;
}

ClearCompressResult
reduceMatrixClearCompressParallel(const std::vector<std::vector<int>> &matrix_columns,
                                  const ChunkConfig &config, int num_threads)
{
    ClearCompressResult result;

    auto start_total = std::chrono::high_resolution_clock::now();

    auto chunks = buildChunks(matrix_columns, config);
    result.num_chunks = static_cast<int>(chunks.size());

    std::vector<ChunkReductionResult> chunk_results(chunks.size());

#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (size_t i = 0; i < chunks.size(); ++i)
    {
        std::unordered_map<int, int> local_pivot_map;
        chunk_results[i] = reduceChunk(chunks[i], local_pivot_map, config);
    }

    for (const auto &chunk_result : chunk_results)
    {
        for (const auto &pair : chunk_result.pairs)
        {
            result.all_pairs.push_back(pair);
        }
        result.total_xor_operations += chunk_result.xor_operations;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();

    return result;
}

ClearCompressResult reduceMatrixInterleaved(const std::vector<std::vector<int>> &matrix_columns,
                                            const ChunkConfig &config)
{
    ClearCompressResult result;

    auto start_total = std::chrono::high_resolution_clock::now();
    auto chunks = buildChunks(matrix_columns, config);

    size_t max_cols = 0;
    for (const auto &chunk : chunks)
    {
        max_cols = std::max(max_cols, chunk.columns.size());
    }

    std::unordered_map<int, int> global_pivot_map;

    for (size_t col_offset = 0; col_offset < max_cols; ++col_offset)
    {
        for (auto &chunk : chunks)
        {
            if (col_offset >= chunk.columns.size())
                continue;

            auto &col = chunk.columns[col_offset];
            if (col.is_cleared)
                continue;

            int pivot = findPivotSparse(col.indices);

            while (pivot >= 0)
            {
                auto it = global_pivot_map.find(pivot);
                if (it != global_pivot_map.end())
                {
                    break;
                }
                global_pivot_map[pivot] = col.global_index;

                PersistencePair pair;
                pair.birth_index = col.global_index;
                pair.death_index = pivot;
                result.all_pairs.push_back(pair);
                break;
            }
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();

    return result;
}

ChunkConfig getOptimalChunkConfig(size_t num_columns, int num_rows)
{
    ChunkConfig config;

    double density = static_cast<double>(num_rows) / static_cast<double>(num_columns + 1);

    if (density < DENSITY_VERY_SPARSE_THRESHOLD)
    {
        config.chunk_size = CHUNK_SIZE_VERY_SPARSE;
    }
    else if (density < DENSITY_SPARSE_THRESHOLD)
    {
        config.chunk_size = CHUNK_SIZE_SPARSE;
    }
    else
    {
        config.chunk_size = CHUNK_SIZE_DENSE;
    }

    config.compress_after_reduction = true;
    config.interleave_processing = (num_columns > 100000);

    return config;
}

ClearCompressSpeedup estimateClearCompressSpeedup(size_t num_columns, int num_rows,
                                                  size_t avg_column_size)
{
    ClearCompressSpeedup speedup;

    speedup.memory_reduction = MEMORY_REDUCTION_ESTIMATE;

    const size_t row_span = static_cast<size_t>(std::max(1, num_rows));
    size_t matrix_size_bytes = num_columns * std::min(avg_column_size, row_span) * sizeof(int);
    size_t l3_cache_size = L3_CACHE_SIZE_ESTIMATE;
    double cache_efficiency =
        std::min(1.0, static_cast<double>(matrix_size_bytes) / static_cast<double>(l3_cache_size));

    if (cache_efficiency < 1.0)
    {
        speedup.cache_efficiency_speedup =
            CACHE_EFFICIENCY_BASE +
            (CACHE_EFFICIENCY_BASE - cache_efficiency) * CACHE_EFFICIENCY_MULTIPLIER;
    }
    else
    {
        speedup.cache_efficiency_speedup = CACHE_EFFICIENCY_FIT;
    }

    speedup.compression_overhead = COMPRESSION_OVERHEAD_FACTOR;
    speedup.total_speedup = speedup.cache_efficiency_speedup * speedup.compression_overhead;

    return speedup;
}

bool shouldUseClearCompress(size_t num_columns, size_t avg_column_size)
{
    size_t matrix_memory = num_columns * avg_column_size * sizeof(int);
    size_t l3_cache = L3_CACHE_SIZE_CONSERVATIVE;

    return matrix_memory > l3_cache;
}

} // namespace nerve::persistence::clearcompress
