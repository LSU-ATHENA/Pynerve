#pragma once

#include "nerve/gpu/device_array.hpp"

#include <cuda_runtime.h>

namespace nerve::gpu
{

// Launches a full symmetric Euclidean distance matrix on device buffers and
// optionally clips entries greater than max_radius to +inf.
//
// points: row-major [n_points, dim] with leading dimension points_ld
// out:    row-major [n_points, n_points] with leading dimension out_ld
//
// If max_radius <= 0, clipping is disabled.
cudaError_t launch_pairwise_distance_radius_f32(const float *d_points, int points_ld, float *d_out,
                                                int out_ld, int n_points, int dim, float max_radius,
                                                void *stream_handle = nullptr);

cudaError_t launch_pairwise_distance_radius_f64(const double *d_points, int points_ld,
                                                double *d_out, int out_ld, int n_points, int dim,
                                                double max_radius, void *stream_handle = nullptr);

} // namespace nerve::gpu
