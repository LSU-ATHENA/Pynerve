// PTX helper implementations for GPU advanced utilities.
// The implementation favors concrete, architecture-safe behavior with
// alternate implementations when specific PTX instructions are missing.

#include "nerve/gpu/gpu_ptx.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nerve::gpu::advanced
{

namespace
{

__device__ inline uint32_t portableBmsk(int width, int pos)
{
    if (width <= 0 || pos < 0 || pos >= 32)
    {
        return 0u;
    }
    if (width >= 32 - pos)
    {
        return 0xFFFFFFFFu << pos;
    }
    return ((1u << width) - 1u) << pos;
}

} // namespace

__device__ void PtxMemory::streamingStore(float *ptr, float value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.global.cs.f32 [%0], %1;" : : "l"(ptr), "f"(value) : "memory");
#else
    *ptr = value;
#endif
}

__device__ float PtxMemory::ldgLoad(const float *ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

__device__ void PtxMemory::prefetchL1(const void *ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("prefetch.global.L1 [%0];" : : "l"(ptr));
#else
    (void)ptr;
#endif
}

__device__ void PtxMemory::prefetchL2(const void *ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));
#else
    (void)ptr;
#endif
}

__device__ uint32_t Ptx92MicroOps::bmsk(int width, int pos)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t result = 0u;
    asm volatile("bmsk.b32 %0, %1, %2;" : "=r"(result) : "r"(width), "r"(pos));
    return result;
#else
    return portableBmsk(width, pos);
#endif
}

__device__ int32_t Ptx92MicroOps::szext(int8_t value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    int32_t result = 0;
    asm volatile("szext.b32.b8 %0, %1;" : "=r"(result) : "r"(static_cast<int32_t>(value)));
    return result;
#else
    return static_cast<int32_t>(value);
#endif
}

__device__ void DualIssuePerfect::distanceQuadChain(const float *p1, const float *p2, int dim,
                                                    float &result)
{
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    int d = 0;
    for (; d + 3 < dim; d += 4)
    {
        const float a0 = p1[d + 0] - p2[d + 0];
        const float a1 = p1[d + 1] - p2[d + 1];
        const float a2 = p1[d + 2] - p2[d + 2];
        const float a3 = p1[d + 3] - p2[d + 3];
        sum0 = fmaf(a0, a0, sum0);
        sum1 = fmaf(a1, a1, sum1);
        sum2 = fmaf(a2, a2, sum2);
        sum3 = fmaf(a3, a3, sum3);
    }

    float tail = 0.0f;
    for (; d < dim; ++d)
    {
        const float diff = p1[d] - p2[d];
        tail = fmaf(diff, diff, tail);
    }
    result = sum0 + sum1 + sum2 + sum3 + tail;
}

__device__ void CacheLinePacker::packExact(const float *input, float *output, int count)
{
    const int lane = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + lane; i < count; i += stride)
    {
        output[i] = input[i];
    }
}

} // namespace nerve::gpu::advanced
