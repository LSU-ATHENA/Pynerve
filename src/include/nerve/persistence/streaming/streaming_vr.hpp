#pragma once

/// @file streaming_vr.hpp

#include "nerve/errors/errors.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <cstddef>
#include <vector>

namespace nerve::persistence
{

class FastVR
{
public:
    /// @brief Compute VR persistence for one chunk and return typed error status.
    errors::ErrorResult<std::vector<Pair>> computeVRResult(const std::vector<double> &points,
                                                           size_t point_dim, size_t num_points,
                                                           const VRConfig &config);

    /// @brief Compute VR persistence for one chunk.
    /// @note API kept for compatibility; returns `{}` on error.
    std::vector<Pair> computeVR(const std::vector<double> &points, size_t point_dim,
                                size_t num_points, const VRConfig &config);
};

/// Processes large point clouds in overlapping chunks to reduce memory usage
/// while maintaining topological accuracy.
class StreamingVR
{
public:
    StreamingVR(size_t chunk_size, size_t overlap_size);

    /// @brief Compute VR persistence with typed error status.
    errors::ErrorResult<std::vector<Pair>> computeStreamingResult(const std::vector<double> &points,
                                                                  size_t point_dim,
                                                                  size_t num_points,
                                                                  const VRConfig &config);

    /// @brief Compute VR persistence using streaming approach
    /// @note API kept for compatibility; returns `{}` on error.
    std::vector<Pair> computeStreaming(const std::vector<double> &points, size_t point_dim,
                                       size_t num_points, const VRConfig &config);

private:
    size_t chunk_size_;
    size_t overlap_size_;
};

} // namespace nerve::persistence
