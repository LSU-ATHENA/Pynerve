#include "nerve/config.hpp"
#include "nerve/cpu/simd.hpp"

#include <cmath>
#include <cstring>

namespace nerve::optimization
{

void simdClipGradients(double *grads, Size n, double max_norm)
{
#if defined(__AVX512F__)
    if (cpu::simd::CPUFeatureDetector::hasAVX512F())
    {
        __m512d vmax = _mm512_set1_pd(max_norm);
        __m512d vneg = _mm512_set1_pd(-max_norm);
        Size i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(grads + i);
            v = _mm512_min_pd(_mm512_max_pd(v, vneg), vmax);
            _mm512_storeu_pd(grads + i, v);
        }
        for (; i < n; ++i)
            grads[i] = std::max(-max_norm, std::min(max_norm, grads[i]));
        return;
    }
#endif
    for (Size i = 0; i < n; ++i)
        grads[i] = std::max(-max_norm, std::min(max_norm, grads[i]));
}

double simdL2Norm(const double *vec, Size n)
{
    double sum = 0.0;
#if defined(__AVX512F__)
    if (cpu::simd::CPUFeatureDetector::hasAVX512F())
    {
        __m512d acc = _mm512_setzero_pd();
        Size i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(vec + i);
            acc = _mm512_fmadd_pd(v, v, acc);
        }
        sum = _mm512_reduce_add_pd(acc);
        for (; i < n; ++i)
            sum += vec[i] * vec[i];
        return std::sqrt(sum);
    }
#elif defined(__AVX2__)
    if (cpu::simd::CPUFeatureDetector::hasAVX2())
    {
        __m256d acc = _mm256_setzero_pd();
        Size i = 0;
        for (; i + 4 <= n; i += 4)
        {
            __m256d v = _mm256_loadu_pd(vec + i);
            acc = _mm256_fmadd_pd(v, v, acc);
        }
        double tmp[4];
        _mm256_storeu_pd(tmp, acc);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        for (; i < n; ++i)
            sum += vec[i] * vec[i];
        return std::sqrt(sum);
    }
#endif
    for (Size i = 0; i < n; ++i)
        sum += vec[i] * vec[i];
    return std::sqrt(sum);
}

} // namespace nerve::optimization
