#include "nerve/gpu/gpu_ptx.hpp"
#include "nerve/gpu/gpu_ptx_ops.cuh"

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

// Hopper (sm90+) operations
struct PtxHopperOps
{
    __device__ static void tmaPrefetch(const void *desc, int coords[])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        asm volatile("prefetch.tensormap [%0];" : : "l"(desc));
#else
        (void)desc;
        (void)coords;
#endif
    }

    __device__ static void fenceMbarrierInit()
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
    }

    __device__ static bool isHopper()
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        return true;
#else
        return false;
#endif
    }

    __device__ static unsigned int clusterRank() { return ptx::cluster_rank(); }

    __device__ static unsigned int clusterSize() { return ptx::cluster_size(); }

    __device__ static void clusterBarrierArrive() { ptx::cluster_barrier_arrive(); }

    __device__ static void clusterBarrierWait() { ptx::cluster_barrier_wait(); }
};

// Blackwell (sm100+) operations
struct PtxBlackwellOps
{
    __device__ static void stMatrix(float *dst, const float *src, int rows, int cols)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        asm volatile("stmatrix.sync.aligned.x4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "l"(dst), "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3])
                     : "memory");
#else
        for (int i = 0; i < rows * cols; ++i)
            dst[i] = src[i];
        (void)rows;
        (void)cols;
#endif
    }

    __device__ static bool isBlackwell()
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        return true;
#else
        return false;
#endif
    }

    __device__ static bool supportsFP4() { return ptx::supports_fp4_hw(100); }
};

// Advanced warp-level reduction helpers
struct AdvancedReduction
{
    __device__ static unsigned int warpMatchAny(unsigned int mask, unsigned int value)
    {
        return ptx::match_any_sync_u32(mask, value);
    }

    __device__ static unsigned int warpMatchAny64(unsigned int mask, unsigned long long value)
    {
        return ptx::match_any_sync_u64(mask, value);
    }

    __device__ static unsigned int warpReduceOr(unsigned int value)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        for (int offset = 16; offset > 0; offset >>= 1)
            value |= __shfl_xor_sync(0xFFFFFFFF, value, offset);
#else
        for (int offset = 16; offset > 0; offset >>= 1)
            value |= __shfl_xor(value, offset);
#endif
        return value;
    }

    __device__ static unsigned long long warpReduceOr64(unsigned long long value)
    {
        int lane = threadIdx.x & 31;
        unsigned int lo = static_cast<unsigned int>(value & 0xFFFFFFFFULL);
        unsigned int hi = static_cast<unsigned int>(value >> 32);
        lo = warpReduceOr(lo);
        hi = warpReduceOr(hi);
        return (static_cast<unsigned long long>(hi) << 32) | lo;
    }

    __device__ static float warpReduceSum(float value) { return ptx::warp_reduce_sum_f32(value); }

    __device__ static double warpReduceSum64(double value)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
#else
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_xor(value, offset);
#endif
        return value;
    }
};

} // namespace nerve::gpu::advanced
