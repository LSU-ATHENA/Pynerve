
#pragma once

#include "nerve/core.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace nerve::persistence::bitparallel
{

struct BitColumn
{
    std::vector<uint64_t> words;
    int num_rows;
    int num_words;
    int pivot;

    BitColumn()
        : num_rows(0)
        , num_words(0)
        , pivot(-1)
    {}

    void updatePivot()
    {
        if (words.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("bit column word count exceeds int range");
        }
        pivot = -1;
        for (int i = static_cast<int>(words.size()) - 1; i >= 0; --i)
        {
            if (words[i] != 0)
            {
                if (static_cast<size_t>(i) >
                    static_cast<size_t>(std::numeric_limits<int>::max()) / 64U)
                {
                    throw std::overflow_error("bit column pivot exceeds int range");
                }
                pivot = i * 64 + 63 - __builtin_clzll(words[i]);
                break;
            }
        }
    }

    [[nodiscard]] bool isEmpty() const { return pivot < 0; }
    int computePivot()
    {
        updatePivot();
        return pivot;
    }
    int getPivot() const { return pivot; }
    void xorInPlace(const BitColumn &other)
    {
        size_t min_size = std::min(words.size(), other.words.size());
        for (size_t i = 0; i < min_size; ++i)
        {
            words[i] ^= other.words[i];
        }
        updatePivot();
    }

    static BitColumn fromSparseIndices(const std::vector<int> &indices, int max_row)
    {
        BitColumn col;
        col.num_rows = std::max(0, max_row);
        const size_t num_rows = static_cast<size_t>(col.num_rows);
        const size_t num_words = (num_rows + 63U) / 64U;
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
        for (int idx : indices)
        {
            if (idx >= 0 && idx < max_row)
            {
                col.words[idx / 64] |= (1ULL << (idx % 64));
            }
        }
        col.updatePivot();
        return col;
    }

    [[nodiscard]] double sparsity() const
    {
        if (num_rows == 0)
            return 0.0;
        int non_zero = 0;
        for (uint64_t word : words)
        {
            non_zero += __builtin_popcountll(word);
        }
        return static_cast<double>(non_zero) / num_rows;
    }
};

struct PersistencePair
{
    int birth_index = -1;
    int death_index = -1;
    double birth_time = 0.0;
    double death_time = 0.0;
    int dimension = 0;
};

struct BitParallelReductionResult
{
    std::vector<PersistencePair> pairs;
    double reduction_time_ms = 0.0;
    int columns_processed = 0;
    int xor_operations = 0;
    int apparent_pairs = 0;
    double speedup_estimate = 0.0;
};

struct BitParallelConfig
{
    bool use_bit_parallel = true;
    bool use_avx512 = false;
    bool use_prefetching = true;
    bool use_clearing = true;
    bool use_branchless = true;
    int prefetch_distance = 4;
};

struct BitParallelSpeedup
{
    double base_speedup = 1.0;
    double avx512_speedup = 1.0;
    double prefetch_speedup = 1.0;
    double branchless_speedup = 1.0;
    double total_speedup = 1.0;
    double memory_reduction = 1.0;
};

struct CompressedSparseBlockMatrix
{
    int block_size;
    int num_cols;
    int num_block_rows;

    std::vector<std::vector<std::tuple<int, int, uint64_t>>> blocks;
};

BitColumn buildBitColumn(const std::vector<int> &sparse_indices, int max_row);

std::vector<int> bitColumnToSparse(const BitColumn &col);

void addBitColumns(BitColumn &a, const BitColumn &b);

#ifdef __AVX512F__
void addBitColumnsAVX512(BitColumn &a, const BitColumn &b);
void recomputePivotAVX512(BitColumn &col);
#endif

int findPivotBranchless(const BitColumn &col);

BitParallelReductionResult reduceMatrixBitParallel(std::vector<BitColumn> &columns,
                                                   const BitParallelConfig &config,
                                                   const std::vector<double> &filtration = {});

void convertToCSB(const std::vector<BitColumn> &columns, CompressedSparseBlockMatrix &csb,
                  int block_size);

void prefetchColumn(const BitColumn &col, int prefetch_distance);

BitParallelReductionResult reduceMatrixWithPrefetch(std::vector<BitColumn> &columns,
                                                    const BitParallelConfig &config);

BitParallelConfig getOptimalBitParallelConfig(size_t num_columns, int num_rows);

BitParallelSpeedup estimateBitParallelSpeedup(int num_rows, int num_cols);

inline bool shouldUseBitParallel(int num_rows, int num_cols)
{
    return num_rows >= 128 && num_cols >= 64;
}

} // namespace nerve::persistence::bitparallel
