#include "nerve/config.hpp"
#include "nerve/core_types.hpp"
#include "nerve/cpu/x86_intrinsics.hpp"

#include <cmath>
#include <cstring>

namespace nerve::algebra::geometry
{

static bool hasSse42()
{
#if defined(__SSE4_2__) || defined(__SSE4_1__)
    return true;
#elif defined(__linux__) && defined(__x86_64__)
    return __builtin_cpu_supports("sse4.2");
#else
    return false;
#endif
}

double euclideanSse4(const double *a, const double *b, Size dim)
{
    double sum = 0.0;
#if defined(NERVE_HAS_X86_INTRINSICS)
    if (hasSse42())
    {
        Size i = 0;
        __m128d acc = _mm_setzero_pd();
        for (; i + 2 <= dim; i += 2)
        {
            __m128d va = _mm_loadu_pd(a + i);
            __m128d vb = _mm_loadu_pd(b + i);
            __m128d d = _mm_sub_pd(va, vb);
            acc = _mm_add_pd(acc, _mm_mul_pd(d, d));
        }
        double tmp[2];
        _mm_storeu_pd(tmp, acc);
        sum = tmp[0] + tmp[1];

        if (i * 2 < dim)
        {
            for (; i < dim; ++i)
            {
                double diff = a[i] - b[i];
                sum += diff * diff;
            }
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
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float euclideanSse4Float(const float *a, const float *b, Size dim)
{
    float sum = 0.0f;
#if defined(NERVE_HAS_X86_INTRINSICS)
    if (hasSse42())
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
            double d = euclideanSse4(points + i * dim, points + j * dim, dim);
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
#if defined(NERVE_HAS_X86_INTRINSICS)
    if (hasSse42())
    {
        Size i = 0;
        __m128d acc = _mm_setzero_pd();
        for (; i + 2 <= n; i += 2)
        {
            __m128d va = _mm_loadu_pd(a + i);
            __m128d vb = _mm_loadu_pd(b + i);
            acc = _mm_add_pd(acc, _mm_mul_pd(va, vb));
        }
        double tmp[2];
        _mm_storeu_pd(tmp, acc);
        double sum = tmp[0] + tmp[1];
        for (; i < n; ++i)
            sum += a[i] * b[i];
        return sum;
    }
#else
    (void)a;
    (void)b;
    (void)n;
#endif
    double sum = 0.0;
    for (Size i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

} // namespace nerve::algebra::geometry
