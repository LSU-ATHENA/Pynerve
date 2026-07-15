#include "nerve/sheaf/sheaf_learning.hpp"
#include "nerve/simd/simd_base.hpp"

#include <cmath>
#include <cstring>

namespace nerve::sheaf
{

template <typename T>
void computeSheafRestrictionSimd(const T *source_stalk, const T *restriction_matrix,
                                 std::size_t stalk_dim, std::size_t source_dim, T *output)
{
    // Use SIMD dispatch table for float64; scalar fallback for float32
    if constexpr (std::is_same_v<T, double>)
    {
        // output[j] += source_stalk[i] * restriction_matrix[i * stalk_dim + j]
        std::memset(output, 0, stalk_dim * sizeof(T));
        auto *R = reinterpret_cast<const double *>(restriction_matrix);
        auto *src = reinterpret_cast<const double *>(source_stalk);
        auto *out = reinterpret_cast<double *>(output);
        for (std::size_t i = 0; i < source_dim; ++i)
            nerve::simd::simd_axpy(src[i], R + i * stalk_dim, out, stalk_dim);
    }
    else
    {
        std::memset(output, 0, stalk_dim * sizeof(T));
        for (std::size_t i = 0; i < source_dim; ++i)
            for (std::size_t j = 0; j < stalk_dim; ++j)
                output[j] += source_stalk[i] * restriction_matrix[i * stalk_dim + j];
    }
}

template <typename T>
void computeSheafLaplacianDiagSimd(const T *restriction_maps, const T *coboundary_maps,
                                   std::size_t n_cells, std::size_t max_stalk_dim, T *diagonal)
{
    (void)coboundary_maps;
    std::memset(diagonal, 0, n_cells * sizeof(T));
    for (std::size_t c = 0; c < n_cells; ++c)
    {
        const T *R = restriction_maps + c * max_stalk_dim * max_stalk_dim;
        // diagonal[c] = sum of all R[j]^2
        T sum = T{0};
        if constexpr (std::is_same_v<T, double>)
        {
            sum = static_cast<T>(nerve::simd::simd_norm2(reinterpret_cast<const double *>(R),
                                                         max_stalk_dim * max_stalk_dim));
            // norm2 returns sqrt(sum(R^2)), so square it back
            sum = sum * sum;
        }
        else
        {
            for (std::size_t j = 0; j < max_stalk_dim * max_stalk_dim; ++j)
                sum += R[j] * R[j];
        }
        diagonal[c] = sum;
    }
}

template void computeSheafRestrictionSimd<float>(const float *, const float *, std::size_t,
                                                 std::size_t, float *);
template void computeSheafRestrictionSimd<double>(const double *, const double *, std::size_t,
                                                  std::size_t, double *);
template void computeSheafLaplacianDiagSimd<float>(const float *, const float *, std::size_t,
                                                   std::size_t, float *);
template void computeSheafLaplacianDiagSimd<double>(const double *, const double *, std::size_t,
                                                    std::size_t, double *);

} // namespace nerve::sheaf
