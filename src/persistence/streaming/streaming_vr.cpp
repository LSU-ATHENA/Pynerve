#include "nerve/core/buffer_view.hpp"
#include "nerve/persistence/streaming/streaming_vr.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace nerve::persistence
{

namespace
{

bool hasValidPointShape(const std::vector<double> &points, size_t point_dim, size_t num_points)
{
    if (point_dim == 0 || num_points == 0)
    {
        return false;
    }
    if (num_points > std::numeric_limits<size_t>::max() / point_dim)
    {
        return false;
    }
    return points.size() == point_dim * num_points;
}

bool hasFinitePoints(const std::vector<double> &points)
{
    return std::ranges::all_of(points, [](double value) { return std::isfinite(value); });
}

bool hasValidConfig(const VRConfig &config)
{
    return std::isfinite(config.max_radius) && config.max_radius >= 0.0;
}

} // namespace

errors::ErrorResult<std::vector<Pair>> FastVR::computeVRResult(const std::vector<double> &points,
                                                               size_t point_dim, size_t num_points,
                                                               const VRConfig &config)
{
    if (!hasValidPointShape(points, point_dim, num_points) || !hasFinitePoints(points) ||
        !hasValidConfig(config))
    {
        return errors::ErrorResult<std::vector<Pair>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    core::BufferView<const double> point_view(points.data(), points.size());
    auto result = computeVrPersistenceFastResult(point_view, point_dim, config);
    if (result.isError())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(result.errorCode());
    }
    return errors::ErrorResult<std::vector<Pair>>::ok(result.moveValue());
}

std::vector<Pair> FastVR::computeVR(const std::vector<double> &points, size_t point_dim,
                                    size_t num_points, const VRConfig &config)
{
    auto result = computeVRResult(points, point_dim, num_points, config);
    if (result.isError())
    {
        return {};
    }
    return result.moveValue();
}

StreamingVR::StreamingVR(size_t chunk_size, size_t overlap_size)
    : chunk_size_(chunk_size)
    , overlap_size_(overlap_size)
{}

errors::ErrorResult<std::vector<Pair>>
StreamingVR::computeStreamingResult(const std::vector<double> &points, size_t point_dim,
                                    size_t num_points, const VRConfig &config)
{
    // Streaming VR: process point cloud in overlapping chunks
    // This reduces memory usage while maintaining topological accuracy

    std::vector<Pair> all_pairs;

    if (!hasValidPointShape(points, point_dim, num_points) || !hasFinitePoints(points) ||
        !hasValidConfig(config))
    {
        return errors::ErrorResult<std::vector<Pair>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    if (chunk_size_ == 0 || chunk_size_ <= overlap_size_)
    {
        FastVR vr;
        return vr.computeVRResult(points, point_dim, num_points, config);
    }

    // Process in chunks
    size_t stride = chunk_size_ - overlap_size_;
    size_t num_chunks = (num_points + stride - 1) / stride;

    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
    {
        size_t start = chunk_idx * stride;
        size_t end = std::min(start + chunk_size_, num_points);
        size_t chunk_size = end - start;

        // Extract chunk
        std::vector<double> chunk_points;
        chunk_points.reserve(chunk_size * point_dim);
        for (size_t i = start; i < end; ++i)
        {
            for (size_t d = 0; d < point_dim; ++d)
            {
                chunk_points.push_back(points[i * point_dim + d]);
            }
        }

        // Compute VR for this chunk
        FastVR chunk_vr;
        auto chunk_config = config;
        auto chunk_result =
            chunk_vr.computeVRResult(chunk_points, point_dim, chunk_size, chunk_config);
        if (chunk_result.isError())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(chunk_result.errorCode());
        }
        auto chunk_pairs = chunk_result.moveValue();

        // Adjust indices for global coordinates
        for (auto &pair : chunk_pairs)
        {
            if (pair.birth_index >= 0)
            {
                pair.birth_index += static_cast<Index>(start);
            }
            if (pair.death_index >= 0)
            {
                pair.death_index += static_cast<Index>(start);
            }
        }

        // Merge pairs from overlap region
        // Only keep pairs that are "new" (not in previous chunk)
        if (chunk_idx > 0)
        {
            size_t overlap_end = std::min(start + overlap_size_, end);
            Index overlap_end_index = static_cast<Index>(overlap_end);

            // Filter: only add pairs that involve points in non-overlap region
            for (const auto &pair : chunk_pairs)
            {
                bool is_new = (pair.birth_index >= overlap_end_index ||
                               pair.death_index >= overlap_end_index);
                if (is_new)
                {
                    all_pairs.push_back(pair);
                }
            }
        }
        else
        {
            // First chunk: keep all pairs
            all_pairs.insert(all_pairs.end(), chunk_pairs.begin(), chunk_pairs.end());
        }
    }

    return errors::ErrorResult<std::vector<Pair>>::ok(std::move(all_pairs));
}

std::vector<Pair> StreamingVR::computeStreaming(const std::vector<double> &points, size_t point_dim,
                                                size_t num_points, const VRConfig &config)
{
    auto result = computeStreamingResult(points, point_dim, num_points, config);
    if (result.isError())
    {
        return {};
    }
    return result.moveValue();
}

} // namespace nerve::persistence
