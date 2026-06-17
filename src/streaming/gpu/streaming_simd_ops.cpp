#include "nerve/config.hpp"
#include "nerve/core_types.hpp"
#include "nerve/cpu/simd.hpp"

#include <cmath>
#include <cstring>

namespace nerve::streaming
{

#if defined(NERVE_HAS_X86_INTRINSICS)
static bool hasAvx2()
{
    static const bool has = cpu::simd::CPUFeatureDetector::hasAVX2();
    return has;
}

static bool hasAvx512()
{
    static const bool has = cpu::simd::CPUFeatureDetector::hasAVX512F();
    return has;
}
#endif

void batchVectorAddSimd(double *a, const double *b, Size n)
{
    Size i = 0;
#if defined(__AVX512F__)
    if (hasAvx512())
    {
        for (; i + 8 <= n; i += 8)
        {
            __m512d va = _mm512_loadu_pd(a + i);
            __m512d vb = _mm512_loadu_pd(b + i);
            _mm512_storeu_pd(a + i, _mm512_add_pd(va, vb));
        }
    }
#endif
#if defined(__AVX2__)
    if (i + 4 <= n)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(a + i, _mm256_add_pd(va, vb));
        i += 4;
    }
#endif
    for (; i < n; ++i)
        a[i] += b[i];
}

void batchScaleSimd(double *data, double alpha, Size n)
{
    Size i = 0;
#if defined(__AVX512F__)
    if (hasAvx512())
    {
        __m512d a = _mm512_set1_pd(alpha);
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(data + i);
            _mm512_storeu_pd(data + i, _mm512_mul_pd(v, a));
        }
    }
#endif
#if defined(__AVX2__)
    if (i + 4 <= n)
    {
        __m256d a = _mm256_set1_pd(alpha);
        __m256d v = _mm256_loadu_pd(data + i);
        _mm256_storeu_pd(data + i, _mm256_mul_pd(v, a));
        i += 4;
    }
#endif
    for (; i < n; ++i)
        data[i] *= alpha;
}

void batchThresholdSimd(double *data, Size n, double lo, double hi)
{
#if defined(__AVX512F__)
    if (hasAvx512())
    {
        Size i = 0;
        __m512d v_lo = _mm512_set1_pd(lo);
        __m512d v_hi = _mm512_set1_pd(hi);
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(data + i);
            v = _mm512_max_pd(v, v_lo);
            v = _mm512_min_pd(v, v_hi);
            _mm512_storeu_pd(data + i, v);
        }
        for (; i < n; ++i)
        {
            if (data[i] < lo)
                data[i] = lo;
            if (data[i] > hi)
                data[i] = hi;
        }
        return;
    }
#endif
#if defined(__AVX2__)
    if (hasAvx2())
    {
        Size i = 0;
        __m256d v_lo = _mm256_set1_pd(lo);
        __m256d v_hi = _mm256_set1_pd(hi);
        for (; i + 4 <= n; i += 4)
        {
            __m256d v = _mm256_loadu_pd(data + i);
            v = _mm256_max_pd(v, v_lo);
            v = _mm256_min_pd(v, v_hi);
            _mm256_storeu_pd(data + i, v);
        }
        for (; i < n; ++i)
        {
            if (data[i] < lo)
                data[i] = lo;
            if (data[i] > hi)
                data[i] = hi;
        }
        return;
    }
#endif
    for (Size i = 0; i < n; ++i)
    {
        if (data[i] < lo)
            data[i] = lo;
        if (data[i] > hi)
            data[i] = hi;
    }
}

} // namespace nerve::streaming
