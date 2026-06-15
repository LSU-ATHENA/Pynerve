#include "nerve/config.hpp"
#include "nerve/cpu/simd.hpp"
#include "nerve/cpu/x86_intrinsics.hpp"
#include "nerve/sheaf/sheaf_learning.hpp"

#include <cmath>
#include <cstring>

namespace nerve::sheaf
{

namespace
{
bool useSimd()
{
    static const bool has = cpu::simd::CPUFeatureDetector::hasAVX2();
    return has;
}

} // namespace

template <typename T>
void computeSheafRestrictionSimd(const T *source_stalk, const T *restriction_matrix, Size stalk_dim,
                                 Size source_dim, T *output)
{
#if defined(NERVE_HAS_X86_INTRINSICS)
    if (useSimd() && sizeof(T) == sizeof(double))
    {
        std::memset(output, 0, stalk_dim * sizeof(T));
        for (Size i = 0; i < source_dim; ++i)
        {
            Size j = 0;
#if defined(__AVX2__)
            __m256d src = _mm256_set1_pd(static_cast<double>(source_stalk[i]));
            for (; j + 4 <= stalk_dim; j += 4)
            {
                __m256d r = _mm256_loadu_pd(restriction_matrix + i * stalk_dim + j);
                __m256d out = _mm256_loadu_pd(output + j);
                out = _mm256_fmadd_pd(src, r, out);
                _mm256_storeu_pd(output + j, out);
            }
#endif
            for (; j < stalk_dim; ++j)
                output[j] += static_cast<double>(source_stalk[i]) *
                             static_cast<double>(restriction_matrix[i * stalk_dim + j]);
        }
        return;
    }
#endif
    for (Size i = 0; i < source_dim; ++i)
        for (Size j = 0; j < stalk_dim; ++j)
            output[j] += source_stalk[i] * restriction_matrix[i * stalk_dim + j];
}

template <typename T>
void computeSheafLaplacianDiagSimd(const T *restriction_maps, const T *coboundary_maps,
                                   Size n_cells, Size max_stalk_dim, T *diagonal)
{
    (void)coboundary_maps;
    std::memset(diagonal, 0, n_cells * sizeof(T));
    for (Size c = 0; c < n_cells; ++c)
    {
        const T *R = restriction_maps + c * max_stalk_dim * max_stalk_dim;
        T sum = T{0};
        Size j = 0;
#if defined(__AVX2__)
        if (useSimd() && sizeof(T) == sizeof(double))
        {
            __m256d acc = _mm256_setzero_pd();
            for (; j + 4 <= max_stalk_dim * max_stalk_dim; j += 4)
            {
                __m256d r = _mm256_loadu_pd(R + j);
                acc = _mm256_fmadd_pd(r, r, acc);
            }
            double tmp[4];
            _mm256_storeu_pd(tmp, acc);
            sum = static_cast<T>(tmp[0] + tmp[1] + tmp[2] + tmp[3]);
        }
#endif
        for (; j < max_stalk_dim * max_stalk_dim; ++j)
            sum += R[j] * R[j];
        diagonal[c] = sum;
    }
}

template void computeSheafRestrictionSimd<float>(const float *, const float *, Size, Size, float *);
template void computeSheafRestrictionSimd<double>(const double *, const double *, Size, Size,
                                                  double *);
template void computeSheafLaplacianDiagSimd<float>(const float *, const float *, Size, Size,
                                                   float *);
template void computeSheafLaplacianDiagSimd<double>(const double *, const double *, Size, Size,
                                                    double *);

} // namespace nerve::sheaf
