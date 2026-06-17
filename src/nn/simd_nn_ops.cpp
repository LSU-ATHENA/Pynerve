#include "nerve/config.hpp"
#include "nerve/cpu/simd.hpp"
#include "nerve/nn/simd_nn.hpp"

#include <cmath>
#include <cstring>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace nerve::nn
{

void simdReLU(double *data, Size n)
{
#if defined(__AVX512F__)
    if (cpu::simd::CPUFeatureDetector.hasAVX512F())
    {
        __m512d vzero = _mm512_setzero_pd();
        Size i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(data + i);
            _mm512_storeu_pd(data + i, _mm512_max_pd(v, vzero));
        }
        for (; i < n; ++i)
            if (data[i] < 0)
                data[i] = 0.0;
        return;
    }
#elif defined(__AVX2__)
    if (cpu::simd::CPUFeatureDetector.hasAVX2())
    {
        __m256d vzero = _mm256_setzero_pd();
        Size i = 0;
        for (; i + 4 <= n; i += 4)
        {
            __m256d v = _mm256_loadu_pd(data + i);
            _mm256_storeu_pd(data + i, _mm256_max_pd(v, vzero));
        }
        for (; i < n; ++i)
            if (data[i] < 0)
                data[i] = 0.0;
        return;
    }
#endif
    for (Size i = 0; i < n; ++i)
        if (data[i] < 0)
            data[i] = 0.0;
}

void simdSigmoid(double *data, Size n)
{
#if defined(__AVX512F__)
    if (cpu::simd::CPUFeatureDetector.hasAVX512F())
    {
        __m512d vone = _mm512_set1_pd(1.0);
        Size i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(data + i);
            __m512d neg = _mm512_sub_pd(_mm512_setzero_pd(), v);
            __m512d exp = _mm512_exp_pd(neg);
            __m512d den = _mm512_add_pd(vone, exp);
            _mm512_storeu_pd(data + i, _mm512_div_pd(vone, den));
        }
        for (; i < n; ++i)
            data[i] = 1.0 / (1.0 + std::exp(-data[i]));
        return;
    }
#endif
    for (Size i = 0; i < n; ++i)
        data[i] = 1.0 / (1.0 + std::exp(-data[i]));
}

void simdTanh(double *data, Size n)
{
#if defined(__AVX512F__)
    if (cpu::simd::CPUFeatureDetector.hasAVX512F())
    {
        __m512d vtwo = _mm512_set1_pd(2.0);
        __m512d vone = _mm512_set1_pd(1.0);
        Size i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(data + i);
            __m512d neg_two = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(vtwo, v));
            __m512d exp = _mm512_exp_pd(neg_two);
            __m512d num = _mm512_sub_pd(vone, exp);
            __m512d den = _mm512_add_pd(vone, exp);
            _mm512_storeu_pd(data + i, _mm512_div_pd(num, den));
        }
        for (; i < n; ++i)
            data[i] = std::tanh(data[i]);
        return;
    }
#endif
    for (Size i = 0; i < n; ++i)
        data[i] = std::tanh(data[i]);
}

void simdBatchNorm(double *data, Size n, double mean, double std_inv)
{
#if defined(__AVX512F__)
    if (cpu::simd::CPUFeatureDetector.hasAVX512F())
    {
        __m512d vmean = _mm512_set1_pd(mean);
        __m512d vstd = _mm512_set1_pd(std_inv);
        Size i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(data + i);
            v = _mm512_sub_pd(v, vmean);
            v = _mm512_mul_pd(v, vstd);
            _mm512_storeu_pd(data + i, v);
        }
        for (; i < n; ++i)
            data[i] = (data[i] - mean) * std_inv;
        return;
    }
#endif
    for (Size i = 0; i < n; ++i)
        data[i] = (data[i] - mean) * std_inv;
}

void simdSoftmax(double *data, Size n)
{
    double max_val = data[0];
    for (Size i = 1; i < n; ++i)
        if (data[i] > max_val)
            max_val = data[i];
    double sum = 0.0;
#if defined(__AVX512F__)
    if (cpu::simd::CPUFeatureDetector.hasAVX512F() && n >= 8)
    {
        __m512d vmax = _mm512_set1_pd(max_val);
        __m512d vsum = _mm512_setzero_pd();
        Size i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(data + i);
            v = _mm512_sub_pd(v, vmax);
            __m512d e = _mm512_exp_pd(v);
            _mm512_storeu_pd(data + i, e);
            vsum = _mm512_add_pd(vsum, e);
        }
        sum = _mm512_reduce_add_pd(vsum);
        for (; i < n; ++i)
        {
            data[i] = std::exp(data[i] - max_val);
            sum += data[i];
        }
    }
    else
#endif
    {
        for (Size i = 0; i < n; ++i)
        {
            data[i] = std::exp(data[i] - max_val);
            sum += data[i];
        }
    }
    double inv_sum = 1.0 / sum;
    for (Size i = 0; i < n; ++i)
        data[i] *= inv_sum;
}

} // namespace nerve::nn
