#include "nerve/cpu/x86_intrinsics.hpp"
#include "nerve/persistence/utils/avx512_optimizer.hpp"
#include "nerve/persistence/utils/bit_parallel_z2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace nerve::persistence::bitparallel
{

constexpr size_t BITS_PER_WORD = 64;
constexpr size_t BITS_PER_BYTE = 8;

constexpr double BIT_PARALLEL_PREFETCH_SPEEDUP = 1.3;
constexpr double BIT_PARALLEL_BRANCHLESS_SPEEDUP = 1.2;
constexpr double BIT_PARALLEL_MEMORY_REDUCTION = 0.5;
#ifdef __AVX512F__
constexpr double BIT_PARALLEL_AVX512_SPEEDUP = 8.0;
constexpr double BIT_PARALLEL_AVX512_BONUS = 1.5;
#endif

constexpr int BIT_PARALLEL_PREFETCH_THRESHOLD = 1000;
#ifdef __AVX512F__
constexpr int BIT_PARALLEL_AVX512_THRESHOLD = 10000;
#endif
constexpr int BIT_PARALLEL_PREFETCH_DISTANCE = 4;

size_t checkedMul(size_t lhs, size_t rhs, const char *context)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        throw std::overflow_error(context);
    }
    return lhs * rhs;
}

size_t checkedVectorCount(size_t lhs, size_t rhs, const char *context)
{
    const size_t count = checkedMul(lhs, rhs, context);
    if (count > std::vector<std::vector<std::tuple<int, int, uint64_t>>>().max_size())
    {
        throw std::length_error(context);
    }
    return count;
}

double filtrationValue(const std::vector<double> &filtration, int index, double default_value)
{
    if (index < 0 || static_cast<size_t>(index) >= filtration.size())
    {
        return default_value;
    }
    const double value = filtration[static_cast<size_t>(index)];
    if (!std::isfinite(value))
    {
        throw std::invalid_argument("bit-parallel filtration values must be finite");
    }
    return value;
}

size_t prepareColumnAddition(BitColumn &a, const BitColumn &b)
{
    if (b.words.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("bit column word count exceeds int range");
    }
    if (a.words.size() < b.words.size())
    {
        a.words.resize(b.words.size(), 0);
    }
    if (a.words.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("bit column word count exceeds int range");
    }
    a.num_rows = std::max(a.num_rows, b.num_rows);
    a.num_words = static_cast<int>(a.words.size());
    return b.words.size();
}

BitColumn buildBitColumn(const std::vector<int> &sparse_indices, int max_row)
{
    BitColumn col;
    col.num_rows = std::max(0, max_row);
    const size_t num_rows = static_cast<size_t>(col.num_rows);
    const size_t num_words = (num_rows + BITS_PER_WORD - 1) / BITS_PER_WORD;
    if (num_words > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("bit column word count exceeds int range");
    }
    if (num_words > col.words.max_size())
    {
        throw std::length_error("bit column word count exceeds vector capacity");
    }
    col.num_words = static_cast<int>(num_words);
    col.words.resize(col.num_words, 0);

    for (int idx : sparse_indices)
    {
        if (idx >= 0 && idx < col.num_rows)
        {
            const size_t word_idx = static_cast<size_t>(idx) / BITS_PER_WORD;
            const int bit_idx = idx % static_cast<int>(BITS_PER_WORD);
            col.words[word_idx] |= (1ULL << bit_idx);
        }
    }

    col.updatePivot();
    return col;
}

std::vector<int> bitColumnToSparse(const BitColumn &col)
{
    std::vector<int> sparse;
    sparse.reserve(static_cast<size_t>(std::max(0, col.num_rows)) / BITS_PER_BYTE);

    for (size_t w = 0; w < col.words.size(); ++w)
    {
        if (w > static_cast<size_t>(std::numeric_limits<int>::max()) / BITS_PER_WORD)
        {
            throw std::overflow_error("bit column sparse row index exceeds int range");
        }
        uint64_t word = col.words[w];
        while (word != 0)
        {
            const int bit = __builtin_ctzll(word);
            const int row = static_cast<int>(w * BITS_PER_WORD + static_cast<size_t>(bit));
            if (row < col.num_rows)
            {
                sparse.push_back(row);
            }
            word &= (word - 1);
        }
    }

    return sparse;
}

void addBitColumns(BitColumn &a, const BitColumn &b)
{
    const size_t word_count = prepareColumnAddition(a, b);
    for (size_t i = 0; i < word_count; ++i)
    {
        a.words[i] ^= b.words[i];
    }

    a.updatePivot();
}

#ifdef __AVX512F__
void addBitColumnsAVX512(BitColumn &a, const BitColumn &b)
{
    avx512::addBitColumnsAVX512(a.words.data(), b.words.data(),
                                std::min(a.words.size(), b.words.size()));
    a.updatePivot();
}
#endif

void addBitColumnsAuto(BitColumn &a, const BitColumn &b)
{
    const size_t word_count = prepareColumnAddition(a, b);
    avx512::addBitColumnsOptimized(a.words.data(), b.words.data(), word_count);
    a.updatePivot();
}

int findPivotBranchless(const BitColumn &col)
{
    if (col.words.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("bit column word count exceeds int range");
    }
    int pivot = -1;

    for (int i = static_cast<int>(col.words.size()) - 1; i >= 0; --i)
    {
        const uint64_t word = col.words[i];
        const int has_bit = (word != 0);
        const int local_pivot =
            has_bit ? (i * static_cast<int>(BITS_PER_WORD) + 63 - __builtin_clzll(word)) : -1;

        const int should_update = (pivot < 0) & has_bit;
        pivot = should_update ? local_pivot : pivot;

        if (pivot >= 0)
            break;
    }

    return pivot;
}

BitParallelReductionResult reduceMatrixBitParallel(std::vector<BitColumn> &columns,
                                                   const BitParallelConfig &config,
                                                   const std::vector<double> &filtration)
{
    BitParallelReductionResult result;
    if (columns.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("bit-parallel column count exceeds int range");
    }
    result.columns_processed = static_cast<int>(columns.size());

    auto start = std::chrono::high_resolution_clock::now();

    std::unordered_map<int, int> pivot_to_column;
    std::vector<bool> infinite_pair_candidates(columns.size(), false);
    result.pairs.reserve(columns.size());

    int xor_operations = 0;
    int apparent_pairs = 0;

    for (size_t col_idx = 0; col_idx < columns.size(); ++col_idx)
    {
        auto &col = columns[col_idx];

        if (col.pivot < 0 && config.use_clearing)
        {
            continue;
        }

        while (col.pivot >= 0)
        {
            auto it = pivot_to_column.find(col.pivot);
            if (it != pivot_to_column.end())
            {
                const auto &pivot_column = columns[it->second];
                if (config.use_avx512 && pivot_column.words.size() >= 8)
                {
                    addBitColumnsAuto(col, pivot_column);
                }
                else
                {
                    addBitColumns(col, pivot_column);
                }
                if (xor_operations == std::numeric_limits<int>::max())
                {
                    throw std::overflow_error("bit-parallel xor operation count exceeds int range");
                }
                ++xor_operations;
            }
            else
            {
                pivot_to_column[col.pivot] = static_cast<int>(col_idx);

                PersistencePair pair;
                pair.birth_index = col.pivot;
                pair.death_index = static_cast<int>(col_idx);
                pair.birth_time = filtrationValue(filtration, pair.birth_index, 0.0);
                pair.death_time = filtrationValue(filtration, pair.death_index, 0.0);
                result.pairs.push_back(pair);

                break;
            }
        }

        if (col.pivot < 0)
        {
            infinite_pair_candidates[col_idx] = true;
        }
    }

    for (size_t col_idx = 0; col_idx < infinite_pair_candidates.size(); ++col_idx)
    {
        if (!infinite_pair_candidates[col_idx] ||
            pivot_to_column.find(static_cast<int>(col_idx)) != pivot_to_column.end())
        {
            continue;
        }

        PersistencePair pair;
        pair.birth_index = static_cast<int>(col_idx);
        pair.death_index = -1;
        pair.birth_time = filtrationValue(filtration, pair.birth_index, 0.0);
        pair.death_time = std::numeric_limits<double>::infinity();
        result.pairs.push_back(pair);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.reduction_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.xor_operations = xor_operations;
    result.apparent_pairs = apparent_pairs;
    int max_rows = 0;
    for (const auto &column : columns)
    {
        max_rows = std::max(max_rows, column.num_rows);
    }
    result.speedup_estimate =
        estimateBitParallelSpeedup(max_rows, result.columns_processed).total_speedup;

    return result;
}

void convertToCSB(const std::vector<BitColumn> &columns, CompressedSparseBlockMatrix &csb,
                  int block_size)
{
    if (block_size <= 0)
    {
        throw std::invalid_argument("CSB block size must be positive");
    }
    if (columns.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("CSB column count exceeds int range");
    }
    csb.block_size = block_size;
    csb.num_cols = static_cast<int>(columns.size());
    const auto block_size_count = static_cast<size_t>(block_size);
    const size_t num_block_rows = (columns.size() + block_size_count - 1) / block_size_count;
    if (num_block_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("CSB block row count exceeds int range");
    }
    csb.num_block_rows = static_cast<int>(num_block_rows);

    const size_t block_count =
        checkedVectorCount(num_block_rows, num_block_rows, "CSB block count overflow");
    csb.blocks.clear();
    csb.blocks.resize(block_count);

    for (size_t col_idx = 0; col_idx < columns.size(); ++col_idx)
    {
        const size_t block_col = col_idx / block_size_count;

        for (size_t word_idx = 0; word_idx < columns[col_idx].words.size(); ++word_idx)
        {
            if (columns[col_idx].words[word_idx] == 0)
                continue;

            if (word_idx > static_cast<size_t>(std::numeric_limits<int>::max()) / BITS_PER_WORD)
            {
                throw std::overflow_error("CSB row index exceeds int range");
            }
            const size_t row_start = word_idx * BITS_PER_WORD;
            const size_t block_row = row_start / block_size_count;

            const size_t block_idx = block_row * num_block_rows + block_col;
            if (block_idx < csb.blocks.size())
            {
                csb.blocks[block_idx].push_back({static_cast<int>(col_idx),
                                                 static_cast<int>(word_idx),
                                                 columns[col_idx].words[word_idx]});
            }
        }
    }
}

void prefetchColumn(const BitColumn &col, int prefetch_distance)
{
    for (size_t i = 0; i < col.words.size() && i < static_cast<size_t>(prefetch_distance); ++i)
    {
        __builtin_prefetch(&col.words[i], 0, 3);
    }
}

BitParallelReductionResult reduceMatrixWithPrefetch(std::vector<BitColumn> &columns,
                                                    const BitParallelConfig &config)
{
    auto prefetched_config = config;
    prefetched_config.use_prefetching = true;
    for (size_t col_idx = 0; col_idx < columns.size(); ++col_idx)
    {
        if (col_idx + prefetched_config.prefetch_distance < columns.size())
        {
            prefetchColumn(columns[col_idx + prefetched_config.prefetch_distance],
                           prefetched_config.prefetch_distance);
        }
    }
    return reduceMatrixBitParallel(columns, prefetched_config, {});
}

BitParallelConfig getOptimalBitParallelConfig(size_t num_columns, int num_rows)
{
    BitParallelConfig config;

    config.use_bit_parallel = (num_rows >= 128);

#ifdef __AVX512F__
    config.use_avx512 = (num_columns >= BIT_PARALLEL_AVX512_THRESHOLD);
#else
    config.use_avx512 = false;
#endif

    config.use_prefetching = (num_columns >= BIT_PARALLEL_PREFETCH_THRESHOLD);
    config.prefetch_distance = BIT_PARALLEL_PREFETCH_DISTANCE;

    config.use_clearing = true;

    return config;
}

BitParallelSpeedup estimateBitParallelSpeedup(int num_rows, int num_cols)
{
    BitParallelSpeedup speedup{};
    speedup.base_speedup = 1.0;
    speedup.avx512_speedup = 1.0;
    speedup.prefetch_speedup = 1.0;
    speedup.branchless_speedup = 1.0;
    speedup.total_speedup = 1.0;
    speedup.memory_reduction = BIT_PARALLEL_MEMORY_REDUCTION;

    if (num_rows <= 0 || num_cols <= 0)
    {
        return speedup;
    }

    const int row_words =
        (num_rows + static_cast<int>(BITS_PER_WORD) - 1) / static_cast<int>(BITS_PER_WORD);

    /*
     * Packed columns amortize one XOR over up to 64 coefficients. The practical
     * factor is bounded by the number of populated row words and by the matrix
     * size where packing overhead becomes worthwhile.
     */
    double base_speedup =
        shouldUseBitParallel(num_rows, num_cols)
            ? std::min(static_cast<double>(BITS_PER_WORD), static_cast<double>(row_words))
            : 1.0;

#ifdef __AVX512F__
    if (num_cols >= BIT_PARALLEL_AVX512_THRESHOLD &&
        avx512::shouldUseAVX512(static_cast<size_t>(row_words)))
    {
        speedup.avx512_speedup = BIT_PARALLEL_AVX512_SPEEDUP;
        base_speedup *= BIT_PARALLEL_AVX512_BONUS;
    }
#endif

    if (num_cols >= BIT_PARALLEL_PREFETCH_THRESHOLD)
    {
        speedup.prefetch_speedup = BIT_PARALLEL_PREFETCH_SPEEDUP;
    }

    speedup.branchless_speedup = BIT_PARALLEL_BRANCHLESS_SPEEDUP;

    speedup.total_speedup = base_speedup * speedup.prefetch_speedup * speedup.branchless_speedup;

    speedup.base_speedup = base_speedup;
    return speedup;
}

} // namespace nerve::persistence::bitparallel
