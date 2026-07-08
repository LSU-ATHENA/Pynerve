#include "nerve/simd/simd_base.hpp"
#include "nerve/core/spectral.hpp"

#include <cmath>
#include <cstring>

namespace nerve::spectral
{

void matVecMulSimd(const double *mat, const double *vec, std::size_t n, double *result)
{
    // result = mat * vec  (mat is n x n, row-major)
    // Use GEMV with alpha=1.0, beta=0.0
    std::memset(result, 0, n * sizeof(double));
    nerve::simd::simd_gemv(1.0, mat, vec, 0.0, result, n, n);
}

void matMatMulSimd(const double *A, const double *B, std::size_t m, std::size_t n, std::size_t k, double *C)
{
    // C += A * B  (A is m x n, B is n x k, C is m x k)
    // Use gemv for each row of A
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t l = 0; l < n; ++l)
        {
            double a = A[i * n + l];
            // C[i, :] += a * B[l, :]
            nerve::simd::simd_axpy(a, B + l * k, C + i * k, k);
        }
    }
}

double dotProductSimd(const double *a, const double *b, std::size_t n)
{
    return nerve::simd::simd_dot(a, b, n);
}

void axpySimd(double alpha, const double *x, std::size_t n, double *y)
{
    nerve::simd::simd_axpy(alpha, x, y, n);
}

void scaleSimd(double alpha, std::size_t n, double *vec)
{
    nerve::simd::simd_scale(vec, alpha, n);
}

} // namespace nerve::spectral
