#include "nerve/simd/simd_base.hpp"

#include <cmath>

namespace nerve::autodiff
{

void simdBackwardAdd(double *grad_a, const double *grad_out, std::size_t n)
{
    // grad_a[i] += grad_out[i]
    nerve::simd::simd_add(grad_a, grad_out, n);
}

void simdBackwardMul(double *grad_a, const double *grad_out, const double *b, std::size_t n)
{
    // grad_a[i] += grad_out[i] * b[i]
    // fmad computes: c[i] += a[i] * b[i]
    nerve::simd::simd_fmad(grad_out, b, grad_a, n);
}

void simdBackwardRelu(double *grad_a, const double *grad_out, const double *input, std::size_t n)
{
    // grad_a[i] += grad_out[i] if input[i] > 0
    for (std::size_t i = 0; i < n; ++i)
        if (input[i] > 0)
            grad_a[i] += grad_out[i];
}

} // namespace nerve::autodiff
