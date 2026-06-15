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

double EnhancedSIMDCalculator::euclideanAvx512Unrolled(const double *a, const double *b, Size dim)
{
#ifdef __AVX512F__
    __m512d sum = _mm512_setzero_pd();
    Size k = 0;

    for (; k + 8 <= dim; k += 8)
    {
        __m512d va = _mm512_loadu_pd(a + k);
        __m512d vb = _mm512_loadu_pd(b + k);
        __m512d diff = _mm512_sub_pd(va, vb);
#ifdef __FMA__
        sum = _mm512_fmadd_pd(diff, diff, sum);
#else
        sum = _mm512_add_pd(sum, _mm512_mul_pd(diff, diff));
#endif
    }

    double total = _mm512_reduce_add_pd(sum);

    for (; k < dim; ++k)
    {
        double diff = a[k] - b[k];
        total += diff * diff;
    }

    return checkedDistanceResult(std::sqrt(total));
#else
    return euclideanAvx2Unrolled(a, b, dim);
#endif
}

void EnhancedSIMDCalculator::batchCompute4Avx512(const double *query, const double *const *targets,
                                                 double *results, Size dimension)
{
#ifdef __AVX512F__
    alignas(64) double scratch[8];
    for (int i = 0; i < 4; ++i)
    {
        __m512d sum = _mm512_setzero_pd();
        Size k = 0;
        for (; k + 7 < dimension; k += 8)
        {
            __m512d q = _mm512_loadu_pd(query + k);
            __m512d t = _mm512_loadu_pd(targets[i] + k);
            __m512d diff = _mm512_sub_pd(q, t);
#ifdef __FMA__
            sum = _mm512_fmadd_pd(diff, diff, sum);
#else
            sum = _mm512_add_pd(sum, _mm512_mul_pd(diff, diff));
#endif
        }

        _mm512_store_pd(scratch, sum);
        double accum = 0.0;
        for (double lane : scratch)
        {
            accum += lane;
        }

        for (; k < dimension; ++k)
        {
            const double diff = query[k] - targets[i][k];
            accum += diff * diff;
        }

        results[i] = checkedDistanceResult(std::sqrt(accum));
    }
#else
    batchCompute4Avx2(query, targets, results, dimension);
#endif
}

} // namespace nerve::algebra
