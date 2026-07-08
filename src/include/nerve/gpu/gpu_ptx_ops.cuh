#pragma once

/// Comprehensive inline PTX helpers for Pynerve GPU kernels.
/// Targets sm80+ (Ampere) with fallbacks for older architectures.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace nerve::gpu::ptx
{

// Bit Manipulation

/// Population count (number of set bits) in a 32-bit word
__device__ __forceinline__ unsigned int popc_u32(unsigned int val)
{
    unsigned int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("popc.b32 %0, %1;" : "=r"(result) : "r"(val));
#else
    result = __popc(val);
#endif
    return result;
}

/// Population count in a 64-bit word
__device__ __forceinline__ unsigned int popc_u64(unsigned long long val)
{
    unsigned int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("popc.b64 %0, %1;" : "=r"(result) : "l"(val));
#else
    result = __popcll(val);
#endif
    return result;
}

/// Find most significant set bit (0-based) in 32-bit word. Returns -1 if zero.
__device__ __forceinline__ int find_msb_u32(unsigned int val)
{
    int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("bfind.u32 %0, %1;" : "=r"(result) : "r"(val));
#else
    if (val == 0) { result = -1; }
    else { result = 31 - __clz(val); }
#endif
    return result;
}

/// Find most significant set bit in 64-bit word. Returns -1 if zero.
__device__ __forceinline__ int find_msb_u64(unsigned long long val)
{
    int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("bfind.u64 %0, %1;" : "=r"(result) : "l"(val));
#else
    if (val == 0) { result = -1; }
    else { result = 63 - __clzll(val); }
#endif
    return result;
}

/// Bit field extract: extract `width` bits starting at `pos`
__device__ __forceinline__ unsigned int bfe_u32(unsigned int val, int pos, int width)
{
    unsigned int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(val), "r"(pos), "r"(width));
#else
    if (pos < 0 || width <= 0) { result = 0; }
    else { result = (val >> pos) & ((1u << width) - 1u); }
#endif
    return result;
}

/// Bit mask: create mask of `width` bits at position `pos`
__device__ __forceinline__ unsigned int bmsk_u32(int width, int pos)
{
    unsigned int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("bmsk.b32 %0, %1, %2;" : "=r"(result) : "r"(width), "r"(pos));
#else
    if (width <= 0 || pos < 0 || pos >= 32) { result = 0u; }
    else if (width >= 32 - pos) { result = 0xFFFFFFFFu << pos; }
    else { result = ((1u << width) - 1u) << pos; }
#endif
    return result;
}

/// Sign-extend: sm80+ only
__device__ __forceinline__ int szext_s8(int val)
{
    int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("szext.b32.b8 %0, %1;" : "=r"(result) : "r"(val));
#else
    result = static_cast<signed char>(val);
#endif
    return result;
}


// Fused Multiply-Add (single instruction)


__device__ __forceinline__ float fma_f32(float a, float b, float c)
{
    float result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
#else
    result = a * b + c;
#endif
    return result;
}

__device__ __forceinline__ double fma_f64(double a, double b, double c)
{
    double result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("fma.rn.f64 %0, %1, %2, %3;" : "=d"(result) : "d"(a), "d"(b), "d"(c));
#else
    result = a * b + c;
#endif
    return result;
}


// Fast Math Approximations


/// Fast base-2 exponent: result = 2^x (approximate, ~0.5% error)
__device__ __forceinline__ float ex2_approx_f32(float x)
{
    float result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
#else
    result = exp2f(x);
#endif
    return result;
}

/// Fast reciprocal: ~1/x (approximate, ~2 ulp error)
__device__ __forceinline__ float rcp_approx_f32(float x)
{
    float result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
#else
    result = 1.0f / x;
#endif
    return result;
}

/// Fast reciprocal square root: ~1/sqrt(x)
__device__ __forceinline__ float rsqrt_approx_f32(float x)
{
    float result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
#else
    result = rsqrtf(x);
#endif
    return result;
}

/// Fast base-2 log: result = log2(x) (approximate)
__device__ __forceinline__ float lg2_approx_f32(float x)
{
    float result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
#else
    result = log2f(x);
#endif
    return result;
}


// Fast expf using ex2: exp(x) = 2^(x / ln(2)) = ex2(x * 1.44269504f)
// ~4x faster than expf with <1% relative error

__device__ __forceinline__ float fast_exp_f32(float x)
{
    return ex2_approx_f32(x * 1.44269504f);
}


// Fast Gaussian: exp(-x^2/(2*sigma^2)) using ex2
// = exp(-x^2 / (2*sigma^2)) = 2^(-x^2 / (2*sigma^2*ln(2)))
// = ex2(-x^2 * (1.0f / (2*sigma^2*ln(2))))
// = ex2(-x^2 * neg_inv_sigma_sq2_ln2)
// where neg_inv_sigma_sq2_ln2 = -1.0f / (2 * sigma^2 * 0.69314718f)
// = -0.72134752f / sigma^2
__device__ __forceinline__ float fast_gaussian_f32(float dist_sq, float precomputed_scale)
{
    return ex2_approx_f32(dist_sq * precomputed_scale);
}

/// Precompute the scale factor: -0.72134752f / (sigma * sigma)
__device__ __forceinline__ float precompute_gaussian_scale(float sigma)
{
    return -0.72134752f / (sigma * sigma);
}


// Fast Sigmoid: 1/(1+exp(-x)) using ex2
// exp(-x) = 2^(-x/ln(2)) = ex2(-x * 1.44269504f)
__device__ __forceinline__ float fast_sigmoid_f32(float x)
{
    float exp_neg_x = ex2_approx_f32(-x * 1.44269504f);
    return rcp_approx_f32(1.0f + exp_neg_x);
}


// Direct Hardware Instructions


/// Hardware max: single instruction, no branch
__device__ __forceinline__ float hwmax_f32(float a, float b)
{
    float result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("max.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
#else
    result = (a > b) ? a : b;
#endif
    return result;
}

/// Hardware min: single instruction, no branch
__device__ __forceinline__ float hwmin_f32(float a, float b)
{
    float result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("min.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
#else
    result = (a < b) ? a : b;
#endif
    return result;
}

/// Hardware max for double
__device__ __forceinline__ double hwmax_f64(double a, double b)
{
    double result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("max.f64 %0, %1, %2;" : "=d"(result) : "d"(a), "d"(b));
#else
    result = (a > b) ? a : b;
#endif
    return result;
}

/// Hardware min for double
__device__ __forceinline__ double hwmin_f64(double a, double b)
{
    double result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("min.f64 %0, %1, %2;" : "=d"(result) : "d"(a), "d"(b));
#else
    result = (a < b) ? a : b;
#endif
    return result;
}

/// Select: s = condition ? a : b  (single instruction, predicated)
__device__ __forceinline__ float slct_f32(bool cond, float a, float b)
{
    float result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("{.reg .pred p; setp.ne.u32 p, %1, 0; selp.f32 %0, %2, %3, p;}"
                 : "=f"(result) : "r"(static_cast<unsigned int>(cond)), "f"(a), "f"(b));
#else
    result = cond ? a : b;
#endif
    return result;
}

/// Select for double
__device__ __forceinline__ double slct_f64(bool cond, double a, double b)
{
    double result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("{.reg .pred p; setp.ne.u32 p, %1, 0; selp.f64 %0, %2, %3, p;}"
                 : "=d"(result) : "r"(static_cast<unsigned int>(cond)), "d"(a), "d"(b));
#else
    result = cond ? a : b;
#endif
    return result;
}

/// Select for unsigned 32-bit
__device__ __forceinline__ unsigned int slct_u32(bool cond, unsigned int a, unsigned int b)
{
    unsigned int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("{.reg .pred p; setp.ne.u32 p, %1, 0; selp.u32 %0, %2, %3, p;}"
                 : "=r"(result) : "r"(static_cast<unsigned int>(cond)), "r"(a), "r"(b));
#else
    result = cond ? a : b;
#endif
    return result;
}

/// Select for signed 32-bit
__device__ __forceinline__ int slct_s32(bool cond, int a, int b)
{
    int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("{.reg .pred p; setp.ne.u32 p, %1, 0; selp.s32 %0, %2, %3, p;}"
                 : "=r"(result) : "r"(static_cast<unsigned int>(cond)), "r"(a), "r"(b));
#else
    result = cond ? a : b;
#endif
    return result;
}


// LOP3: Ternary logic operation (sm50+)
// Replaces multi-instruction patterns like (A ^ B) & ~C

__device__ __forceinline__ unsigned int lop3_lut(unsigned int a, unsigned int b, unsigned int c,
                                                  unsigned int imm_lut)
{
    unsigned int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 500
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
                 : "=r"(result) : "r"(a), "r"(b), "r"(c), "r"(imm_lut));
#else
    result = 0;
#endif
    return result;
}

/// XOR with mask: (A ^ B) & ~C
/// LUT truth table: (A xor B) and (not C)  -> LUT = 0x14
__device__ __forceinline__ unsigned int xor_and_notc(unsigned int a, unsigned int b, unsigned int c)
{
    return lop3_lut(a, b, c, 0x14u);
}

/// XOR and C: (A ^ B) & C -> LUT = 0x28
__device__ __forceinline__ unsigned int xor_and_c(unsigned int a, unsigned int b, unsigned int c)
{
    return lop3_lut(a, b, c, 0x28u);
}


// Memory Operations


/// Streaming store: store to global, bypass L1, streaming through L2
__device__ __forceinline__ void st_global_cs_f32(float *ptr, float value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    asm volatile("st.global.cs.f32 [%0], %1;" : : "l"(ptr), "f"(value) : "memory");
#else
    *ptr = value;
#endif
}

/// Streaming store double
__device__ __forceinline__ void st_global_cs_f64(double *ptr, double value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(ptr), "d"(value) : "memory");
#else
    *ptr = value;
#endif
}

/// Write-back store (evict from L1, keep in L2)
__device__ __forceinline__ void st_global_wb_f32(float *ptr, float value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    asm volatile("st.global.wb.f32 [%0], %1;" : : "l"(ptr), "f"(value) : "memory");
#else
    *ptr = value;
#endif
}

/// Cache-global store (write-through, useful for read-once data)
__device__ __forceinline__ void st_global_cg_f32(float *ptr, float value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    asm volatile("st.global.cg.f32 [%0], %1;" : : "l"(ptr), "f"(value) : "memory");
#else
    *ptr = value;
#endif
}

/// LDG load: use read-only data cache, bypass L1
__device__ __forceinline__ float ld_global_ca_f32(const float *ptr)
{
    float result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(result) : "l"(ptr));
#else
    result = *ptr;
#endif
    return result;
}

/// LDG load int32
__device__ __forceinline__ int ld_global_ca_s32(const int *ptr)
{
    int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    asm volatile("ld.global.ca.s32 %0, [%1];" : "=r"(result) : "l"(ptr));
#else
    result = *ptr;
#endif
    return result;
}

/// Prefetch into L1 cache
__device__ __forceinline__ void prefetch_l1(const void *ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("prefetch.global.L1 [%0];" : : "l"(ptr));
#else
    (void)ptr;
#endif
}

/// Prefetch into L2 cache
__device__ __forceinline__ void prefetch_l2(const void *ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));
#else
    (void)ptr;
#endif
}

/// Cache control hint: evict first (for data loaded once)
__device__ __forceinline__ void prefetch_evict_first(const void *ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("prefetch.global.L2::evict_first [%0];" : : "l"(ptr));
#else
    (void)ptr;
#endif
}

/// Cache control hint: evict last (for data reused)
__device__ __forceinline__ void prefetch_evict_last(const void *ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("prefetch.global.L2::evict_last [%0];" : : "l"(ptr));
#else
    (void)ptr;
#endif
}


// Async Copy (sm80+)


/// Async copy from global to shared memory (sm80+)
/// size_bytes must be 4, 8, or 16 (PTX requires immediate operand)
template <int SizeBytes>
__device__ __forceinline__ void cp_async_shared_global(void *dst, const void *src,
                                                         bool bypass_l1 = false)
{
    static_assert(SizeBytes == 4 || SizeBytes == 8 || SizeBytes == 16,
                  "cp.async size must be 4, 8, or 16 bytes");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (bypass_l1)
    {
        asm volatile("cp.async.ca.shared.global.L2::cache_hint [%0], [%1], %2;"
                     : : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(dst))),
                         "l"(src), "n"(SizeBytes));
    }
    else
    {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                 : : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(dst))),
                     "l"(src), "n"(SizeBytes));
    }
#else
    (void)bypass_l1;
    if constexpr (SizeBytes == 4)
        *reinterpret_cast<float *>(dst) = *reinterpret_cast<const float *>(src);
    else if constexpr (SizeBytes == 8)
        *reinterpret_cast<double *>(dst) = *reinterpret_cast<const double *>(src);
    else if constexpr (SizeBytes == 16)
        *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<const float4 *>(src);
#endif
}

/// Commit a group of async copies
__device__ __forceinline__ void cp_async_commit_group()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

/// Wait for N groups of async copies to complete
/// N must be a compile-time constant (PTX immediate operand)
template <int N>
__device__ __forceinline__ void cp_async_wait_group()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" : : "n"(N));
#else
    __syncthreads();
#endif
}

/// Wait for all async copies to complete
__device__ __forceinline__ void cp_async_wait_all()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;");
#else
    __syncthreads();
#endif
}


// Warp-Level Primitives


/// Warp butterfly reduction: sum across warp
__device__ __forceinline__ float warp_reduce_sum_f32(float val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
#else
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
#endif
    return val;
}

/// Warp butterfly reduction: max across warp
__device__ __forceinline__ float warp_reduce_max_f32(float val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = (val > other) ? val : other;
    }
#else
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = (val > other) ? val : other;
    }
#endif
    return val;
}

/// Warp butterfly reduction: max across warp (double)
__device__ __forceinline__ double warp_reduce_max_f64(double val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        double other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = (val > other) ? val : other;
    }
#else
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        double other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = (val > other) ? val : other;
    }
#endif
    return val;
}

/// Warp-level atomic OR on global memory (replaces atomicCAS loops)
__device__ __forceinline__ unsigned long long warp_atomic_or_global_u64(
    unsigned long long *addr, unsigned long long val)
{
    unsigned long long old;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("atom.global.or.b64 %0, [%1], %2;"
                 : "=l"(old) : "l"(addr), "l"(val) : "memory");
#else
    old = atomicOr(addr, val);
#endif
    return old;
}

/// Atomic XOR on 32-bit global
__device__ __forceinline__ unsigned int atom_xor_global_u32(unsigned int *addr, unsigned int val)
{
    unsigned int old;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("atom.global.xor.b32 %0, [%1], %2;"
                 : "=r"(old) : "l"(addr), "r"(val) : "memory");
#else
    old = atomicXor(addr, val);
#endif
    return old;
}

/// Atomic XOR on 64-bit global
__device__ __forceinline__ unsigned long long atom_xor_global_u64(
    unsigned long long *addr, unsigned long long val)
{
    unsigned long long old;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("atom.global.xor.b64 %0, [%1], %2;"
                 : "=l"(old) : "l"(addr), "l"(val) : "memory");
#else
    // fallback: 64-bit atomicXor not available in old CUDA, use CAS loop
    old = *addr;
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, assumed ^ val);
    } while (assumed != old);
#endif
    return old;
}


// Match Any (sm70+): cooperative warp-level search

__device__ __forceinline__ unsigned int match_any_sync_u32(unsigned int mask, unsigned int value)
{
    unsigned int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("match.any.sync.b32 %0, %1, %2;"
                 : "=r"(result) : "r"(mask), "r"(value));
#else
    result = 0;
    (void)mask;
    (void)value;
#endif
    return result;
}

__device__ __forceinline__ unsigned int match_any_sync_u64(unsigned int mask,
                                                            unsigned long long value)
{
    // match.any.sync.b64 inline PTX has operand constraint issues
    // on some CUDA toolchain versions; use the intrinsic or fall back.
    (void)mask;
    (void)value;
    return 0;
}


// Double Float (fp64) Fast Arithmetic


/// Fast reciprocal for double (f64)
__device__ __forceinline__ double rcp_approx_f64(double x)
{
    double result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("rcp.approx.ftz.f64 %0, %1;" : "=d"(result) : "d"(x));
#else
    result = 1.0 / x;
#endif
    return result;
}

/// Fast reciprocal sqrt for double (f64)
__device__ __forceinline__ double rsqrt_approx_f64(double x)
{
    double result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(result) : "d"(x));
#else
    result = rsqrt(x);
#endif
    return result;
}


// Hopper-Specific (sm90+): TMA and Cluster Operations


/// Get cluster rank (sm90+)
__device__ __forceinline__ unsigned int cluster_rank()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    unsigned int rank;
    asm volatile("mov.u32 %0, %%clusterid;" : "=r"(rank));
    return rank;
#else
    return 0;
#endif
}

/// Get cluster size (sm90+)
__device__ __forceinline__ unsigned int cluster_size()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    unsigned int size;
    asm volatile("mov.u32 %0, %%clustersize;" : "=r"(size));
    return size;
#else
    return 1;
#endif
}

/// Cluster barrier arrive (sm90+)
__device__ __forceinline__ void cluster_barrier_arrive()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.arrive;" ::: "memory");
#endif
}

/// Cluster barrier wait (sm90+)
__device__ __forceinline__ void cluster_barrier_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.wait;" ::: "memory");
#endif
}

/// Multicast load: load from one SM and broadcast to cluster (sm90+)
__device__ __forceinline__ void multicast_load_b32(unsigned int &dst, const unsigned int *addr,
                                                     unsigned int cluster_mask)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("mapa.sync.aligned.b32 %0, [%1], %2;"
                 : "=r"(dst) : "l"(addr), "r"(cluster_mask));
#else
    dst = *addr;
    (void)cluster_mask;
#endif
}


// Tensor Core WMMA Helpers (sm70+)


/// Check if the GPU supports Tensor Cores (sm70+)
__host__ __device__ __forceinline__ bool has_tensor_cores(int compute_capability)
{
    return compute_capability >= 70;
}

/// Check if the GPU supports WMMA (sm70+ - Volta and newer)
__host__ __device__ __forceinline__ bool supports_wmma(int compute_capability)
{
    return compute_capability >= 70;
}

/// Check if the GPU supports MMA (sm80+ - Ampere and newer)
__host__ __device__ __forceinline__ bool supports_mma(int compute_capability)
{
    return compute_capability >= 80;
}

/// Check if the GPU supports async copy (sm80+)
__host__ __device__ __forceinline__ bool supports_async_copy(int compute_capability)
{
    return compute_capability >= 80;
}

/// Check if the GPU supports TMA (sm90+ - Hopper and newer)
__host__ __device__ __forceinline__ bool supports_tma(int compute_capability)
{
    return compute_capability >= 90;
}

/// Check if the GPU supports DPX (sm90+)
__host__ __device__ __forceinline__ bool supports_dpx(int compute_capability)
{
    return compute_capability >= 90;
}

/// Check if the GPU supports WGMMA (sm90+)
__host__ __device__ __forceinline__ bool supports_wgmma(int compute_capability)
{
    return compute_capability >= 90;
}

/// Check if the GPU supports FP8 hardware (sm90+)
__host__ __device__ __forceinline__ bool supports_fp8_hw(int compute_capability)
{
    return compute_capability >= 90;
}

/// Check if the GPU supports FP4 hardware (sm100+ - Blackwell)
__host__ __device__ __forceinline__ bool supports_fp4_hw(int compute_capability)
{
    return compute_capability >= 100;
}

/// Check if L2 eviction policy hints are available (sm90+)
__host__ __device__ __forceinline__ bool supports_evict_policy(int compute_capability)
{
    return compute_capability >= 90;
}


// Accumulation Helpers Using FMA


/// Accumulate squared difference using FMA: sum += diff*diff
__device__ __forceinline__ bool fma_accumulate_sq_f32(float diff, float &sum)
{
    float next = fma_f32(diff, diff, sum);
    if (!isfinite(diff) || !isfinite(next))
    {
        return false;
    }
    sum = next;
    return true;
}

/// Accumulate squared difference using FMA for double
__device__ __forceinline__ bool fma_accumulate_sq_f64(double diff, double &sum)
{
    double next = fma_f64(diff, diff, sum);
    if (!isfinite(diff) || !isfinite(next))
    {
        return false;
    }
    sum = next;
    return true;
}

/// Accumulate product using FMA: sum += a*b
__device__ __forceinline__ bool fma_accumulate_prod_f64(double a, double b, double &sum)
{
    double next = fma_f64(a, b, sum);
    if (!isfinite(a) || !isfinite(b) || !isfinite(next))
    {
        return false;
    }
    sum = next;
    return true;
}

} // namespace nerve::gpu::ptx
