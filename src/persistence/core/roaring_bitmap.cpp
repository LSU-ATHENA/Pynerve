#include "nerve/persistence/core/roaring_bitmap.hpp"
#include "nerve/persistence/core/roaring_bitmap_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nerve::persistence::roaring
{

namespace
{

constexpr size_t BENCHMARK_MAX_ROWS = 100000;
constexpr size_t PARALLEL_BUILD_THRESHOLD = 1000;
constexpr size_t BENCHMARK_MAX_COLUMNS = 2048;
constexpr size_t BITSET_BITS_PER_WORD = 64;
constexpr double MIN_RECOMMENDED_MEMORY_REDUCTION = 1.25;

double normalizeSparsity(double sparsity)
{
    if (!std::isfinite(sparsity))
    {
        throw std::invalid_argument("Roaring sparsity must be finite");
    }
    if (sparsity > 1.0)
    {
        sparsity /= 100.0;
    }
    return std::clamp(sparsity, 0.0, 1.0);
}

size_t denseWordCount(size_t num_rows)
{
    return (num_rows + BITSET_BITS_PER_WORD - 1) / BITSET_BITS_PER_WORD;
}

std::vector<int> deterministicSparseIndices(size_t column, size_t max_rows, size_t cardinality)
{
    std::vector<int> indices;
    if (max_rows == 0 || cardinality == 0)
    {
        return indices;
    }

    cardinality = std::min(cardinality, max_rows);
    indices.reserve(cardinality);
    const size_t stride = std::max<size_t>(1, max_rows / cardinality);
    size_t value = (column * 2654435761u) % max_rows;
    for (size_t i = 0; indices.size() < cardinality && i < max_rows * 2; ++i)
    {
        indices.push_back(static_cast<int>(value));
        value = (value + stride + 1) % max_rows;
        if (indices.size() > 1 && indices.back() == indices[indices.size() - 2])
        {
            value = (value + 1) % max_rows;
        }
    }

    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    for (size_t fill = 0; indices.size() < cardinality && fill < max_rows; ++fill)
    {
        const int candidate = static_cast<int>((fill + column) % max_rows);
        if (!std::binary_search(indices.begin(), indices.end(), candidate))
        {
            indices.insert(std::upper_bound(indices.begin(), indices.end(), candidate), candidate);
        }
    }
    return indices;
}

void xorDenseColumn(std::vector<uint64_t> &accumulator, const std::vector<int> &indices)
{
    for (int index : indices)
    {
        const size_t word = static_cast<size_t>(index) / BITSET_BITS_PER_WORD;
        const size_t bit = static_cast<size_t>(index) % BITSET_BITS_PER_WORD;
        accumulator[word] ^= (uint64_t{1} << bit);
    }
}

} // namespace

// Benchmark and estimation functions
RoaringBenchmark benchmarkRoaring(size_t num_columns, size_t sparsity_percent)
{
    RoaringBenchmark bench{};
    if (num_columns == 0)
    {
        return bench;
    }

    const double sparsity = normalizeSparsity(static_cast<double>(sparsity_percent));
    const double density = 1.0 - sparsity;
    const size_t max_rows = BENCHMARK_MAX_ROWS;
    const size_t cardinality =
        std::clamp(static_cast<size_t>(std::ceil(density * static_cast<double>(max_rows))),
                   size_t{1}, max_rows);
    const size_t sample_columns = std::min(num_columns, BENCHMARK_MAX_COLUMNS);
    const size_t num_words = denseWordCount(max_rows);

    std::vector<std::vector<int>> columns;
    columns.reserve(sample_columns);
    for (size_t column = 0; column < sample_columns; ++column)
    {
        columns.push_back(deterministicSparseIndices(column, max_rows, cardinality));
    }

    std::vector<uint64_t> dense_accumulator(num_words, 0);
    const auto dense_start = std::chrono::steady_clock::now();
    for (const auto &indices : columns)
    {
        xorDenseColumn(dense_accumulator, indices);
    }
    bench.bitvector_time_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dense_start)
            .count();
    bench.bitvector_memory_bytes = num_columns * num_words * sizeof(uint64_t);

    RoaringColumn roaring_accumulator;
    size_t sample_roaring_memory = 0;
    bool accumulator_initialized = false;
    const auto roaring_start = std::chrono::steady_clock::now();
    for (const auto &indices : columns)
    {
        RoaringColumn column =
            RoaringColumn::fromSparseIndices(indices, static_cast<int>(max_rows));
        sample_roaring_memory += column.memoryUsage();
        if (!accumulator_initialized)
        {
            roaring_accumulator = std::move(column);
            accumulator_initialized = true;
        }
        else
        {
            roaring_accumulator.xorInPlace(column);
        }
    }
    bench.roaring_time_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - roaring_start)
            .count();

    const double scale = static_cast<double>(num_columns) / static_cast<double>(sample_columns);
    bench.roaring_memory_bytes = static_cast<size_t>(
        std::ceil(static_cast<double>(std::max<size_t>(1, sample_roaring_memory)) * scale));
    bench.speedup =
        bench.roaring_time_ms > 0.0 ? bench.bitvector_time_ms / bench.roaring_time_ms : 0.0;
    bench.memory_reduction = bench.roaring_memory_bytes > 0
                                 ? static_cast<double>(bench.bitvector_memory_bytes) /
                                       static_cast<double>(bench.roaring_memory_bytes)
                                 : 0.0;
    return bench;
}

RoaringConfig getOptimalRoaringConfig(size_t num_columns, double avg_sparsity)
{
    RoaringConfig config{};
    const double sparsity = normalizeSparsity(avg_sparsity);
    config.sparsity_threshold = sparsity >= 0.98 ? 0.85 : 0.9;
    config.auto_convert = sparsity < 0.995;
    config.parallel_build = num_columns >= PARALLEL_BUILD_THRESHOLD;
    return config;
}

RoaringEstimate estimateRoaringBenefit(size_t num_rows, size_t expected_cardinality)
{
    RoaringEstimate estimate{};
    if (num_rows == 0)
    {
        estimate.memory_reduction = 1.0;
        estimate.speedup = 1.0;
        estimate.recommended = false;
        return estimate;
    }

    expected_cardinality = std::min(expected_cardinality, num_rows);
    const size_t dense_bytes = denseWordCount(num_rows) * sizeof(uint64_t);
    const size_t sparse_bytes =
        sizeof(RoaringBitmapImpl) + sizeof(Container) + expected_cardinality * sizeof(uint16_t);

    estimate.memory_reduction =
        sparse_bytes > 0 ? static_cast<double>(dense_bytes) / static_cast<double>(sparse_bytes)
                         : 1.0;

    const double sparsity =
        1.0 - (static_cast<double>(expected_cardinality) / static_cast<double>(num_rows));
    if (sparsity >= 0.99)
    {
        estimate.speedup = 1.35;
    }
    else if (sparsity >= 0.95)
    {
        estimate.speedup = 1.15;
    }
    else if (sparsity >= 0.90)
    {
        estimate.speedup = 1.0;
    }
    else
    {
        estimate.speedup = 0.75;
    }
    estimate.recommended = shouldUseRoaring(expected_cardinality, num_rows) &&
                           estimate.memory_reduction >= MIN_RECOMMENDED_MEMORY_REDUCTION &&
                           estimate.speedup >= 0.9;
    return estimate;
}

} // namespace nerve::persistence::roaring
