#include "nerve/simd/simd_base.hpp"

#include <cmath>

namespace nerve::streaming
{

void batchVectorAddSimd(double *a, const double *b, std::size_t n)
{
    nerve::simd::simd_add(a, b, n);
}

void batchScaleSimd(double *data, double alpha, std::size_t n)
{
    nerve::simd::simd_scale(data, alpha, n);
}

void batchThresholdSimd(double *data, std::size_t n, double lo, double hi)
{
    nerve::simd::simd_clamp(data, lo, hi, n);
}

} // namespace nerve::streaming
