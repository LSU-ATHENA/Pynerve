#pragma once
#include <cuda_runtime.h>

#include <cstddef>

namespace nerve::gpu
{

struct FastedConfig
{
    int warp_tile_m = 64;
    int warp_tile_n = 64;
    int warp_tile_k = 16;
    int block_tile_m = 128;
    int block_tile_n = 128;
    int pipeline_stages = 2;
    bool use_l2_optimization = true;
    bool use_xor_swizzle = true;
};

cudaError_t launchDistanceFastEd(const float *points, int n_points, int dim, float *distances,
                                 int distance_stride, FastedConfig config = FastedConfig{});

cudaError_t launchDistanceFastEdAsync(const float *points, int n_points, int dim, float *distances,
                                      int distance_stride, cudaStream_t stream = nullptr,
                                      FastedConfig config = FastedConfig{});

} // namespace nerve::gpu
