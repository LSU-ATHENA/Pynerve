// Tile-Based Out-of-Core Persistent Homology
// For datasets larger than RAM or GPU memory
// Processes data in tiles with overlap handling
// Scales to large datasets with bounded memory usage

#include "nerve/persistence/core/flood_complex.hpp"
#include "nerve/persistence/streaming/tile_streaming_ph.hpp"
#include "nerve/persistence/utils/backend_selector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nerve::persistence
{

namespace
{

// Default dedup tolerance when two tile-local diagrams contain the same pair.
constexpr double kDefaultTileMergeTolerance = 1e-6;

size_t checkedPointValueCount(size_t num_points, size_t point_dim)
{
    if (point_dim != 0 && num_points > std::numeric_limits<size_t>::max() / point_dim)
    {
        throw std::overflow_error("point buffer shape overflows size_t");
    }
    return num_points * point_dim;
}

size_t checkedPointBytes(size_t num_points, size_t point_dim)
{
    const size_t values = checkedPointValueCount(num_points, point_dim);
    if (values > std::numeric_limits<size_t>::max() / sizeof(double))
    {
        throw std::overflow_error("point buffer byte size overflows size_t");
    }
    return values * sizeof(double);
}

bool hasFinitePointBuffer(const std::vector<double> &points)
{
    return std::ranges::all_of(points, [](double value) { return std::isfinite(value); });
}

double validatedOverlapRatio(double overlap_ratio)
{
    if (!std::isfinite(overlap_ratio) || overlap_ratio < 0.0 || overlap_ratio > 0.5)
    {
        throw std::invalid_argument("StreamingConfig overlap_ratio must be finite and in [0, 0.5]");
    }
    return overlap_ratio;
}

double validatedMergeTolerance(double merge_tolerance)
{
    if (!std::isfinite(merge_tolerance) || merge_tolerance < 0.0)
    {
        throw std::invalid_argument(
            "StreamingConfig merge_tolerance must be finite and non-negative");
    }
    return merge_tolerance;
}

void validateStreamingConfig(const StreamingConfig &config)
{
    if (!std::isfinite(config.max_radius) || config.max_radius < 0.0)
    {
        throw std::invalid_argument("StreamingConfig max_radius must be finite and non-negative");
    }
    (void)validatedOverlapRatio(config.overlap_ratio);
    (void)validatedMergeTolerance(config.merge_tolerance);
}

#include "detail/tile_streaming_runtime.inl"
StreamingResult computeStreamingPH(const std::string &data_path, size_t point_dim,
                                   size_t num_points, const StreamingConfig &config)
{
    if (point_dim == 0)
    {
        throw std::invalid_argument("computeStreamingPH requires point_dim > 0");
    }
    if (config.tile_size == 0)
    {
        throw std::invalid_argument("computeStreamingPH requires tile_size > 0");
    }
    validateStreamingConfig(config);

    StreamingResult result{};
    result.total_points = num_points;
    result.point_dim = point_dim;

    if (num_points == 0)
    {
        return result;
    }

    const auto start_total = std::chrono::high_resolution_clock::now();

    const auto start_partition = std::chrono::high_resolution_clock::now();
    const auto tiles =
        TilePartitioner::partitionLinear(num_points, std::min(config.tile_size, num_points),
                                         validatedOverlapRatio(config.overlap_ratio));
    const auto end_partition = std::chrono::high_resolution_clock::now();
    result.partition_time_ms =
        std::chrono::duration<double, std::milli>(end_partition - start_partition).count();

    const auto start_process = std::chrono::high_resolution_clock::now();
    const size_t total_bytes = checkedPointBytes(num_points, point_dim);
    MemoryMappedFile memmap(data_path, 0, total_bytes);

    AsyncTileProcessor processor(resolveThreadCount(config.num_threads));
    std::vector<std::future<std::vector<Pair>>> futures;
    futures.reserve(tiles.size());

    FloodComplexConfig flood_config;
    flood_config.max_dim = config.max_dim;
    flood_config.max_radius = config.max_radius;

    for (const auto &tile : tiles)
    {
        auto tile_points = memmap.readChunk(tile.start_idx, tile.local_count, point_dim);
        futures.push_back(processor.submitTile(tile_points, point_dim, flood_config));
    }

    std::vector<std::vector<Pair>> tile_results;
    tile_results.reserve(futures.size());
    for (auto &future : futures)
    {
        tile_results.push_back(future.get());
    }

    const auto end_process = std::chrono::high_resolution_clock::now();
    result.processing_time_ms =
        std::chrono::duration<double, std::milli>(end_process - start_process).count();

    const auto start_merge = std::chrono::high_resolution_clock::now();
    result.pairs = TileResultMerger::mergeResults(tile_results, tiles,
                                                  validatedMergeTolerance(config.merge_tolerance));
    const auto end_merge = std::chrono::high_resolution_clock::now();
    result.merge_time_ms =
        std::chrono::duration<double, std::milli>(end_merge - start_merge).count();

    const auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();

    result.num_tiles = tiles.size();
    result.pairs_count = result.pairs.size();

    const size_t full_memory_bytes = total_bytes;
    size_t peak_tile_bytes = 0;
    for (const auto &tile : tiles)
    {
        peak_tile_bytes = std::max(peak_tile_bytes, checkedPointBytes(tile.local_count, point_dim));
    }

    result.memory_reduction_ratio =
        (full_memory_bytes == 0)
            ? 0.0
            : 1.0 - static_cast<double>(peak_tile_bytes) / static_cast<double>(full_memory_bytes);

    return result;
}

StreamingResult computeTiledPH(const std::vector<double> &points, size_t point_dim,
                               const StreamingConfig &config)
{
    if (point_dim == 0)
    {
        throw std::invalid_argument("computeTiledPH requires point_dim > 0");
    }
    if (points.size() % point_dim != 0)
    {
        throw std::invalid_argument("computeTiledPH received non-rectangular point buffer");
    }
    if (!hasFinitePointBuffer(points))
    {
        throw std::invalid_argument("computeTiledPH received non-finite point coordinates");
    }
    if (config.tile_size == 0)
    {
        throw std::invalid_argument("computeTiledPH requires tile_size > 0");
    }
    validateStreamingConfig(config);

    const auto start_total = std::chrono::high_resolution_clock::now();
    const size_t num_points = points.size() / point_dim;
    const auto start_partition = std::chrono::high_resolution_clock::now();
    auto tiles = TilePartitioner::partitionPoints(
        points, point_dim, num_points,
        std::max<size_t>(1, std::min(config.tile_size, std::max<size_t>(num_points, 1))),
        validatedOverlapRatio(config.overlap_ratio));
    const auto end_partition = std::chrono::high_resolution_clock::now();

    const auto start_process = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<Pair>> tile_results(tiles.size());

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (std::ptrdiff_t tile_idx = 0; tile_idx < static_cast<std::ptrdiff_t>(tiles.size()); ++tile_idx)
    {
        const auto &tile = tiles[tile_idx];

        std::vector<double> tile_points;
        tile_points.reserve(checkedPointValueCount(tile.local_count, point_dim));

        if (tile.start_idx >= num_points)
        {
            continue;
        }
        const size_t local_count = std::min(tile.local_count, num_points - tile.start_idx);
        const size_t end = tile.start_idx + local_count;
        for (size_t i = tile.start_idx; i < end; ++i)
        {
            const size_t offset = i * point_dim;
            const auto begin_offset = static_cast<std::vector<double>::difference_type>(offset);
            const auto end_offset =
                static_cast<std::vector<double>::difference_type>(offset + point_dim);
            tile_points.insert(tile_points.end(), std::next(points.begin(), begin_offset),
                               std::next(points.begin(), end_offset));
        }

        FloodComplexConfig flood_config;
        flood_config.max_dim = config.max_dim;
        flood_config.max_radius = config.max_radius;

        auto result = computeFloodComplex(tile_points, point_dim, tile_points.size() / point_dim,
                                          flood_config);
        tile_results[tile_idx] = std::move(result.pairs);
    }
    const auto end_process = std::chrono::high_resolution_clock::now();

    StreamingResult final_result{};
    final_result.total_points = num_points;
    final_result.point_dim = point_dim;
    final_result.num_tiles = tiles.size();
    const auto start_merge = std::chrono::high_resolution_clock::now();
    final_result.pairs = TileResultMerger::mergeResults(
        tile_results, tiles, validatedMergeTolerance(config.merge_tolerance));
    const auto end_merge = std::chrono::high_resolution_clock::now();
    final_result.pairs_count = final_result.pairs.size();
    const auto end_total = std::chrono::high_resolution_clock::now();

    final_result.partition_time_ms =
        std::chrono::duration<double, std::milli>(end_partition - start_partition).count();
    final_result.processing_time_ms =
        std::chrono::duration<double, std::milli>(end_process - start_process).count();
    final_result.merge_time_ms =
        std::chrono::duration<double, std::milli>(end_merge - start_merge).count();
    final_result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();

    size_t peak_tile_bytes = 0;
    for (const auto &tile : tiles)
    {
        peak_tile_bytes = std::max(peak_tile_bytes, checkedPointBytes(tile.local_count, point_dim));
    }
    const size_t full_memory_bytes = checkedPointBytes(num_points, point_dim);
    final_result.memory_reduction_ratio =
        full_memory_bytes == 0
            ? 0.0
            : 1.0 - static_cast<double>(peak_tile_bytes) / static_cast<double>(full_memory_bytes);
    return final_result;
}

StreamingConfig getOptimalStreamingConfig(size_t num_points, size_t point_dim,
                                          size_t available_memory_mb)
{
    if (point_dim == 0)
    {
        throw std::invalid_argument("getOptimalStreamingConfig requires point_dim > 0");
    }

    StreamingConfig config;
    config.max_dim = 2;
    config.max_radius = 1.0;

    if (point_dim > std::numeric_limits<size_t>::max() / sizeof(double))
    {
        throw std::overflow_error("streaming point size overflows size_t");
    }
    const size_t point_size = point_dim * sizeof(double);
    constexpr size_t kBytesPerMiB = 1024ull * 1024ull;
    const size_t available_bytes =
        available_memory_mb > std::numeric_limits<size_t>::max() / kBytesPerMiB
            ? std::numeric_limits<size_t>::max()
            : available_memory_mb * kBytesPerMiB;
    const size_t compute_budget_bytes = available_bytes / 2;
    const size_t budgeted_points =
        std::max<size_t>(1, compute_budget_bytes / std::max<size_t>(1, point_size));

    config.tile_size = std::clamp<size_t>(budgeted_points, 1000, 10000);
    if (num_points > 0)
    {
        config.tile_size = std::min(config.tile_size, num_points);
    }

    config.overlap_ratio = 0.1;
    config.num_threads = resolveThreadCount(0);
    config.merge_tolerance = kDefaultTileMergeTolerance;
    return config;
}

} // namespace nerve::persistence
