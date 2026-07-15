#include "nerve/config.hpp"
#include "nerve/core_types.hpp"
#include "nerve/cpu/x86_intrinsics.hpp"
#include "nerve/platform.hpp"
#include "nerve/simd/simd_base.hpp"
#include "nerve/simd/simd_distance.hpp"

#include <cmath>
#include <cstring>

namespace nerve::algebra::geometry
{

double euclideanSse4(const double *a, const double *b, Size dim)
{
    // Delegate to the SIMD dispatch table -- selects the best ISA at runtime.
    return nerve::simd::simd_euclidean(a, b, static_cast<std::size_t>(dim));
}

float euclideanSse4Float(const float *a, const float *b, Size dim)
{
    float sum = 0.0f;
#if defined(NERVE_HAS_X86_INTRINSICS) && (defined(__SSE4_1__) || defined(__SSE4_2__))
    auto hasSse = []() -> bool {
#if defined(__SSE4_2__) || defined(__SSE4_1__)
        return true;
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
        return nerve::cpu::CpuFeatureFlags::detect().has_sse42;
#else
        return false;
#endif
    };
    if (hasSse())
    {
        Size i = 0;
        __m128 acc = _mm_setzero_ps();
        for (; i + 4 <= dim; i += 4)
        {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 d = _mm_sub_ps(va, vb);
            acc = _mm_add_ps(acc, _mm_mul_ps(d, d));
        }
        float tmp[4];
        _mm_storeu_ps(tmp, acc);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        for (; i < dim; ++i)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
#else
    (void)a;
    (void)b;
    (void)dim;
#endif
    for (Size i = 0; i < dim; ++i)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void pairwiseDistanceMatrixSse4(const double *points, Size n, Size dim, double *matrix)
{
    for (Size i = 0; i < n; ++i)
    {
        matrix[i * n + i] = 0.0;
        for (Size j = i + 1; j < n; ++j)
        {
            double d = nerve::simd::simd_euclidean(points + i * dim, points + j * dim,
                                                   static_cast<std::size_t>(dim));
            matrix[i * n + j] = d;
            matrix[j * n + i] = d;
        }
    }
}

void pairwiseDistanceMatrixSse4Float(const float *points, Size n, Size dim, float *matrix)
{
    for (Size i = 0; i < n; ++i)
    {
        matrix[i * n + i] = 0.0f;
        for (Size j = i + 1; j < n; ++j)
        {
            float d = euclideanSse4Float(points + i * dim, points + j * dim, dim);
            matrix[i * n + j] = d;
            matrix[j * n + i] = d;
        }
    }
}

double dotProductSse4(const double *a, const double *b, Size n)
{
    return nerve::simd::simd_dot(a, b, static_cast<std::size_t>(n));
}

} // namespace nerve::algebra::geometry
