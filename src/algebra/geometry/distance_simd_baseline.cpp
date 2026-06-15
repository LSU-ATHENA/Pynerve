#include "nerve/algebra/simd_distance_avx.hpp"
#include "nerve/cpu/x86_intrinsics.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace nerve::algebra
{

namespace
{
double checkedDistanceResult(double value)
{
    if (!std::isfinite(value))
    {
        throw std::overflow_error("SIMD distance overflow");
    }
    return value;
}
} // namespace

double EnhancedSIMDCalculator::euclideanSse4Simd(const double *a, const double *b, Size dim)
{
#ifdef __SSE4_1__
    __m128d sum = _mm_setzero_pd();
    Size k = 0;

    for (; k + 2 <= dim; k += 2)
    {
        __m128d va = _mm_loadu_pd(a + k);
        __m128d vb = _mm_loadu_pd(b + k);
        __m128d diff = _mm_sub_pd(va, vb);
        sum = _mm_add_pd(sum, _mm_mul_pd(diff, diff));
    }

    __m128d sum1 = _mm_hadd_pd(sum, sum);
    double total = _mm_cvtsd_f64(sum1);

    for (; k < dim; ++k)
    {
        double diff = a[k] - b[k];
        total += diff * diff;
    }

    return checkedDistanceResult(std::sqrt(total));
#else
    return euclideanScalar(a, b, dim);
#endif
}

double EnhancedSIMDCalculator::euclideanScalar(const double *a, const double *b, Size dim)
{
    double sum = 0.0;
    for (Size k = 0; k < dim; ++k)
    {
        double diff = a[k] - b[k];
        sum += diff * diff;
    }
    return checkedDistanceResult(std::sqrt(sum));
}

} // namespace nerve::algebra
