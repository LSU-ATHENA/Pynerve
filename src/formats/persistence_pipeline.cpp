#include "nerve/formats/packed_boundary_matrix.hpp"
#include "nerve/formats/packed_gpu_scan.hpp"
#include "nerve/formats/persistence_pipeline.hpp"
#include "nerve/platform.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nerve::formats
{

namespace
{

double wallClockMs()
{
    using Clock = std::chrono::high_resolution_clock;
    static const Clock::time_point t0 = Clock::now();
    auto now = Clock::now();
    return std::chrono::duration<double, std::milli>(now - t0).count();
}

struct PackedReductionState
{
    std::vector<PackedWord> packed_data;
    std::vector<Size> column_word_counts;
    size_t words_per_column;
    size_t num_columns;

    std::unordered_map<Index, Index> pivot_to_column;
    std::vector<Pair> pairs;
};

Index findPivotInPacked(const PackedWord *col, Size nw)
{
    for (Size w = nw; w > 0; --w)
    {
        if (col[w - 1] != 0)
        {
            int msb = nerve::bits::clz64(col[w - 1]);
            int bit = 63 - msb;
            return static_cast<Index>((w - 1) * kBitsPerPackedWord + static_cast<Size>(bit));
        }
    }
    return -1;
}

bool isColumnEmpty(const PackedWord *col, Size nw)
{
    for (Size w = 0; w < nw; ++w)
    {
        if (col[w] != 0)
            return false;
    }
    return true;
}

void xorColumnsPacked(PackedWord *dst, const PackedWord *src, Size nw)
{
    for (Size w = 0; w < nw; ++w)
    {
        dst[w] ^= src[w];
    }
}

void reduceOneColumnPacked(Index col_idx, PackedReductionState &state)
{
    PackedWord *col =
        state.packed_data.data() + static_cast<size_t>(col_idx) * state.words_per_column;
    Size nw = state.column_word_counts[static_cast<size_t>(col_idx)];

    Size iter_limit = nw * 4 + 10;
    for (Size it = 0; it < iter_limit; ++it)
    {
        if (isColumnEmpty(col, nw))
            break;

        Index pivot = findPivotInPacked(col, nw);
        if (pivot < 0)
            break;

        auto it_piv = state.pivot_to_column.find(pivot);
        if (it_piv == state.pivot_to_column.end())
        {
            state.pivot_to_column[pivot] = col_idx;
            return;
        }

        Index reducer_col = it_piv->second;
        const PackedWord *reducer =
            state.packed_data.data() + static_cast<size_t>(reducer_col) * state.words_per_column;
        Size reducer_nw = state.column_word_counts[static_cast<size_t>(reducer_col)];
        Size max_nw = std::max(nw, reducer_nw);
        xorColumnsPacked(col, reducer, max_nw);
        for (Size w = nw; w < max_nw; ++w)
        {
            if (col[w] != 0)
                nw = w + 1;
        }
    }
}

void extractPairsPacked(PackedReductionState &state, const std::vector<Field> &filtration_values)
{
    for (const auto &[pivot, col] : state.pivot_to_column)
    {
        Pair pair;
        pair.birth_index = pivot;
        pair.death_index = col;
        pair.dimension = 1;
        if (static_cast<Size>(pivot) < filtration_values.size())
            pair.birth = filtration_values[static_cast<Size>(pivot)];
        if (static_cast<Size>(col) < filtration_values.size())
            pair.death = filtration_values[static_cast<Size>(col)];
        if (pair.death == 0.0)
            pair.death = std::numeric_limits<Field>::infinity();
        state.pairs.push_back(pair);
    }
}

} // namespace

PersistencePipeline::PersistencePipeline(const PipelineConfig &config)
    : config_(config)
{}


errors::ErrorResult<PipelineResult>
PersistencePipeline::compute(const PackedBoundaryMatrix &boundary_matrix,
                             const std::vector<Field> &filtration_values)
{
    if (!boundary_matrix.isValid())
    {
        return errors::ErrorResult<PipelineResult>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    return errors::ErrorResult<PipelineResult>::success(
        runGpuScanAndReduce(boundary_matrix, filtration_values));
}

errors::ErrorResult<PipelineResult>
PersistencePipeline::computeFromColumns(Size n_rows,
                                        const std::vector<std::vector<Index>> &boundary_columns,
                                        const std::vector<Field> &filtration_values)
{
    if (boundary_columns.empty())
    {
        PipelineResult result;
        return errors::ErrorResult<PipelineResult>::success(std::move(result));
    }

    auto build_result = PackedBoundaryMatrix::fromBoundaryColumns(n_rows, boundary_columns);
    if (build_result.isError())
    {
        return errors::ErrorResult<PipelineResult>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    auto matrix = build_result.value();
    return errors::ErrorResult<PipelineResult>::success(
        runGpuScanAndReduce(matrix, filtration_values));
}

PipelineResult PersistencePipeline::runGpuScanAndReduce(const PackedBoundaryMatrix &matrix,
                                                        const std::vector<Field> &filtration_values)
{
    PipelineResult result;
    const double t_start = wallClockMs();

    result.total_columns = matrix.numCols();
    result.nnz = matrix.nnz();

    PackedBoundaryMatrix packed(matrix);
    packed.convertToPacked();

    Size n_cols = packed.numCols();
    Size nw = packed.numWordsPerColumn();

    auto layout_result = packed.buildGpuLayout();
    const double t_pack = wallClockMs();

    GpuScanResult scan_result;
    bool gpu_used = false;

    if (config_.enable_gpu_scan && !layout_result.isError())
    {
        auto scan_err = gpu::launchPackedScan(layout_result.value(), nullptr, config_.device_id);
        if (!scan_err.isError())
        {
            scan_result = scan_err.value();
            gpu_used = true;
        }
    }

    result.scan_time_ms = wallClockMs() - t_pack;

    Size n_stable = 0;
    Size n_unstable = n_cols;

    if (gpu_used && scan_result.stable_count > 0)
    {
        n_stable = scan_result.stable_count;
        n_unstable = scan_result.unstable_count;
    }
    else
    {
        n_stable = 0;
        n_unstable = n_cols;
    }

    result.stable_columns = n_stable;
    result.unstable_columns = n_unstable;
    result.compression_ratio =
        result.total_columns > 0
            ? static_cast<double>(n_stable) / static_cast<double>(result.total_columns)
            : 0.0;

    PackedReductionState state;
    state.words_per_column = nw;
    state.num_columns = n_cols;
    state.packed_data.resize(n_cols * nw, 0);
    state.column_word_counts.resize(n_cols, 0);

    const auto &src_columns = packed.packedColumns();
    const auto &src_sizes = packed.columnSizes();
    for (Size col = 0; col < n_cols; ++col)
    {
        Size col_nw = src_sizes.empty() ? nw : std::min(src_sizes[col], nw);
        state.column_word_counts[col] = col_nw;
        PackedWord *dst = state.packed_data.data() + col * nw;
        const PackedWord *src = src_columns.data() + col * nw;
        std::memcpy(dst, src, col_nw * kPackedWordBytes);
    }

    const double t_reduce_start = wallClockMs();

    if (gpu_used)
    {
        for (Index col : scan_result.stable_columns)
        {
            if (col < 0 || static_cast<Size>(col) >= n_cols)
                continue;
            PackedWord *c =
                state.packed_data.data() + static_cast<size_t>(col) * state.words_per_column;
            Size cnw = state.column_word_counts[static_cast<size_t>(col)];
            if (isColumnEmpty(c, cnw))
                continue;
            Index pivot = findPivotInPacked(c, cnw);
            if (pivot >= 0)
            {
                state.pivot_to_column[pivot] = col;
            }
        }

        for (Index col : scan_result.unstable_columns)
        {
            if (col < 0 || static_cast<Size>(col) >= n_cols)
                continue;
            reduceOneColumnPacked(col, state);
        }
    }
    else
    {
        for (Size col = 0; col < n_cols; ++col)
        {
            reduceOneColumnPacked(static_cast<Index>(col), state);
        }
    }

    extractPairsPacked(state, filtration_values);

    result.reduction_time_ms = wallClockMs() - t_reduce_start;
    result.total_time_ms = wallClockMs() - t_start;
    result.persistence_pairs = std::move(state.pairs);

    return result;
}

errors::ErrorResult<std::vector<Pair>>
computePackedPersistence(const std::vector<std::vector<Index>> &boundary_columns, Size n_rows,
                         const std::vector<Field> &filtration_values, int device_id)
{
    PipelineConfig config;
    config.device_id = device_id;
    config.enable_gpu_scan = true;

    PersistencePipeline pipeline(config);
    auto result = pipeline.computeFromColumns(n_rows, boundary_columns, filtration_values);
    if (result.isError())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    return errors::ErrorResult<std::vector<Pair>>::success(
        std::move(result.value().persistence_pairs));
}

} // namespace nerve::formats
