#include "nerve/config.hpp"
#include "nerve/core/spectral.hpp"
#include "nerve/cpu/simd.hpp"

#include <cmath>
#include <cstring>

namespace nerve::spectral
{

#if defined(NERVE_HAS_X86_INTRINSICS)
static bool hasSimd()
{
    static const bool has = cpu::CPUFeatureDetector::instance().hasAVX2();
    return has;
}

void matVecMulSimd(const double *mat, const double *vec, Size n, double *result)
{
    std::memset(result, 0, n * sizeof(double));
    for (Size i = 0; i < n; ++i)
    {
        double sum = 0.0;
        Size j = 0;
#if defined(__AVX2__)
        if (hasSimd())
        {
            __m256d acc = _mm256_setzero_pd();
            for (; j + 4 <= n; j += 4)
            {
                __m256d m = _mm256_loadu_pd(mat + i * n + j);
                __m256d v = _mm256_loadu_pd(vec + j);
                acc = _mm256_fmadd_pd(m, v, acc);
            }
            double tmp[4];
            _mm256_storeu_pd(tmp, acc);
            sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        }
#endif
        for (; j < n; ++j)
            sum += mat[i * n + j] * vec[j];
        result[i] = sum;
    }
}

void matMatMulSimd(const double *A, const double *B, Size m, Size n, Size k, double *C)
{
    std::memset(C, 0, m * k * sizeof(double));
    for (Size i = 0; i < m; ++i)
    {
        for (Size l = 0; l < n; ++l)
        {
            double a = A[i * n + l];
#if defined(__AVX2__)
            if (hasSimd())
            {
                Size j = 0;
                __m256d a_bc = _mm256_set1_pd(a);
                for (; j + 4 <= k; j += 4)
                {
                    __m256d b = _mm256_loadu_pd(B + l * k + j);
                    __m256d c = _mm256_loadu_pd(C + i * k + j);
                    c = _mm256_fmadd_pd(a_bc, b, c);
                    _mm256_storeu_pd(C + i * k + j, c);
                }
                for (; j < k; ++j)
                    C[i * k + j] += a * B[l * k + j];
            }
            else
#endif
            {
                for (Size j = 0; j < k; ++j)
                    C[i * k + j] += a * B[l * k + j];
            }
        }
    }
}

double dotProductSimd(const double *a, const double *b, Size n)
{
    double sum = 0.0;
    Size i = 0;
#if defined(__AVX2__)
    if (hasSimd())
    {
        __m256d acc = _mm256_setzero_pd();
        for (; i + 4 <= n; i += 4)
        {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            acc = _mm256_fmadd_pd(va, vb, acc);
        }
        double tmp[4];
        _mm256_storeu_pd(tmp, acc);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
#endif
    for (; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

void axpySimd(double alpha, const double *x, Size n, double *y)
{
    Size i = 0;
#if defined(__AVX2__)
    if (hasSimd())
    {
        __m256d a = _mm256_set1_pd(alpha);
        for (; i + 4 <= n; i += 4)
        {
            __m256d xv = _mm256_loadu_pd(x + i);
            __m256d yv = _mm256_loadu_pd(y + i);
            yv = _mm256_fmadd_pd(a, xv, yv);
            _mm256_storeu_pd(y + i, yv);
        }
    }
#endif
    for (; i < n; ++i)
        y[i] += alpha * x[i];
}

void scaleSimd(double alpha, Size n, double *vec)
{
    Size i = 0;
#if defined(__AVX2__)
    if (hasSimd())
    {
        __m256d a = _mm256_set1_pd(alpha);
        for (; i + 4 <= n; i += 4)
        {
            __m256d v = _mm256_loadu_pd(vec + i);
            v = _mm256_mul_pd(v, a);
            _mm256_storeu_pd(vec + i, v);
        }
    }
#endif
    for (; i < n; ++i)
        vec[i] *= alpha;
}
#endif

} // namespace nerve::spectral
