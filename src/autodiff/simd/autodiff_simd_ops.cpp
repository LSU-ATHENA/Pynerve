#include "nerve/config.hpp"
#include "nerve/cpu/simd.hpp"

#include <cmath>
#include <cstring>

namespace nerve::autodiff
{

void simdBackwardAdd(double *grad_a, const double *grad_out, Size n)
{
#if defined(__AVX512F__)
    if (cpu::CPUFeatureDetector::instance().hasAVX512F())
    {
        Size i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m512d g = _mm512_loadu_pd(grad_out + i);
            __m512d a = _mm512_loadu_pd(grad_a + i);
            _mm512_storeu_pd(grad_a + i, _mm512_add_pd(a, g));
        }
        for (; i < n; ++i)
            grad_a[i] += grad_out[i];
        return;
    }
#endif
    for (Size i = 0; i < n; ++i)
        grad_a[i] += grad_out[i];
}

void simdBackwardMul(double *grad_a, const double *grad_out, const double *b, Size n)
{
#if defined(__AVX512F__)
    if (cpu::CPUFeatureDetector::instance().hasAVX512F())
    {
        Size i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m512d g = _mm512_loadu_pd(grad_out + i);
            __m512d vb = _mm512_loadu_pd(b + i);
            __m512d a = _mm512_loadu_pd(grad_a + i);
            _mm512_storeu_pd(grad_a + i, _mm512_fmadd_pd(g, vb, a));
        }
        for (; i < n; ++i)
            grad_a[i] += grad_out[i] * b[i];
        return;
    }
#endif
    for (Size i = 0; i < n; ++i)
        grad_a[i] += grad_out[i] * b[i];
}

void simdBackwardRelu(double *grad_a, const double *grad_out, const double *input, Size n)
{
#if defined(__AVX512F__)
    if (cpu::CPUFeatureDetector::instance().hasAVX512F())
    {
        __m512d vzero = _mm512_setzero_pd();
        Size i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m512d g = _mm512_loadu_pd(grad_out + i);
            __m512d x = _mm512_loadu_pd(input + i);
            __mmask8 mask = _mm512_cmp_pd_mask(x, vzero, _CMP_GT_OQ);
            __m512d masked = _mm512_maskz_loadu_pd(mask, grad_a + i);
            _mm512_mask_storeu_pd(grad_a + i, mask, _mm512_add_pd(masked, g));
        }
        for (; i < n; ++i)
            if (input[i] > 0)
                grad_a[i] += grad_out[i];
        return;
    }
#endif
    for (Size i = 0; i < n; ++i)
        if (input[i] > 0)
            grad_a[i] += grad_out[i];
}

} // namespace nerve::autodiff
