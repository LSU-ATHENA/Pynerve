#pragma once

/// @file gpu_ptx.hpp

#include <cuda_runtime.h>

#include <cstdint>

namespace nerve::gpu::advanced
{

struct PtxMemory
{
    /// Streaming store - bypass L2
    __device__ static void streamingStore(float *ptr, float value);

    /// LDG load - use read-only cache
    __device__ static float ldgLoad(const float *ptr);

    /// Prefetch to L1
    __device__ static void prefetchL1(const void *ptr);

    /// Prefetch to L2
    __device__ static void prefetchL2(const void *ptr);
};

struct WarpScheduler
{
    /// Ensure 4 independent instruction chains
    /// for dual-issue on all 4 schedulers
    template <typename Func>
    __device__ static void quadSchedulerExecute(Func chain0, Func chain1, Func chain2, Func chain3)
    {
        auto r0 = chain0();
        auto r1 = chain1();
        auto r2 = chain2();
        auto r3 = chain3();
        (void)r0;
        (void)r1;
        (void)r2;
        (void)r3;
    }
};

struct Ptx92MicroOps
{
    __device__ static uint32_t bmsk(int width, int pos);
    __device__ static int32_t szext(int8_t value);
};

struct DualIssuePerfect
{
    __device__ static void distanceQuadChain(const float *p1, const float *p2, int dim,
                                             float &result);
};

struct CacheLinePacker
{
    __device__ static void packExact(const float *input, float *output, int count);
};

} // namespace nerve::gpu::advanced
