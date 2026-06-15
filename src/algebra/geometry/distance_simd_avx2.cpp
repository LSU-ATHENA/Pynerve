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

double EnhancedSIMDCalculator::euclideanAvx2Unrolled(const double *a, const double *b, Size dim)
{
#ifdef __AVX2__
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    Size k = 0;

    for (; k + 8 <= dim; k += 8)
    {
        __m256d va0 = _mm256_loadu_pd(a + k);
        __m256d vb0 = _mm256_loadu_pd(b + k);
        __m256d va1 = _mm256_loadu_pd(a + k + 4);
        __m256d vb1 = _mm256_loadu_pd(b + k + 4);

        __m256d diff0 = _mm256_sub_pd(va0, vb0);
        __m256d diff1 = _mm256_sub_pd(va1, vb1);

#ifdef __FMA__
        sum0 = _mm256_fmadd_pd(diff0, diff0, sum0);
        sum1 = _mm256_fmadd_pd(diff1, diff1, sum1);
#else
        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(diff0, diff0));
        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(diff1, diff1));
#endif
    }

    __m256d sum4 = _mm256_add_pd(sum0, sum1);
    __m128d lo = _mm256_castpd256_pd128(sum4);
    __m128d hi = _mm256_extractf128_pd(sum4, 1);
    __m128d sum2 = _mm_add_pd(lo, hi);
    __m128d sum_scalar = _mm_hadd_pd(sum2, sum2);
    double total = _mm_cvtsd_f64(sum_scalar);

    for (; k < dim; ++k)
    {
        double diff = a[k] - b[k];
        total += diff * diff;
    }

    return checkedDistanceResult(std::sqrt(total));
#else
    return euclideanSse4Simd(a, b, dim);
#endif
}

void EnhancedSIMDCalculator::batchCompute4Avx2(const double *query, const double *const *targets,
                                               double *results, Size dimension)
{
#ifdef __AVX2__
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();

    Size k = 0;

    for (; k + 4 <= dimension; k += 4)
    {
        __m256d q = _mm256_loadu_pd(query + k);

        __m256d t0 = _mm256_loadu_pd(targets[0] + k);
        __m256d t1 = _mm256_loadu_pd(targets[1] + k);
        __m256d t2 = _mm256_loadu_pd(targets[2] + k);
        __m256d t3 = _mm256_loadu_pd(targets[3] + k);

        __m256d d0 = _mm256_sub_pd(q, t0);
        __m256d d1 = _mm256_sub_pd(q, t1);
        __m256d d2 = _mm256_sub_pd(q, t2);
        __m256d d3 = _mm256_sub_pd(q, t3);

#ifdef __FMA__
        sum0 = _mm256_fmadd_pd(d0, d0, sum0);
        sum1 = _mm256_fmadd_pd(d1, d1, sum1);
        sum2 = _mm256_fmadd_pd(d2, d2, sum2);
        sum3 = _mm256_fmadd_pd(d3, d3, sum3);
#else
        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(d0, d0));
        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(d1, d1));
        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(d2, d2));
        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(d3, d3));
#endif
    }

    auto horizontalSum = [](__m256d v) {
        __m128d lo = _mm256_castpd256_pd128(v);
        __m128d hi = _mm256_extractf128_pd(v, 1);
        __m128d sum2 = _mm_add_pd(lo, hi);
        __m128d sum1 = _mm_hadd_pd(sum2, sum2);
        return _mm_cvtsd_f64(sum1);
    };

    for (int i = 0; i < 4; ++i)
    {
        double sum = horizontalSum(i == 0 ? sum0 : i == 1 ? sum1 : i == 2 ? sum2 : sum3);

        for (Size j = k; j < dimension; ++j)
        {
            double diff = query[j] - targets[i][j];
            sum += diff * diff;
        }

        results[i] = checkedDistanceResult(std::sqrt(sum));
    }
#else
    for (int i = 0; i < 4; ++i)
    {
        results[i] = distance_function_(query, targets[i], dimension);
    }
#endif
}

} // namespace nerve::algebra
