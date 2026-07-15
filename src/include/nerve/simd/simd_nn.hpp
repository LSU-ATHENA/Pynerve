#pragma once
#include "nerve/simd/simd_base.hpp"

namespace nerve::simd
{

// Batch normalize: data[i] = (data[i] - mean) * std_inv
inline void simd_batchnorm(double *data, std::size_t n, double mean, double std_inv)
{
    // data[i] = (data[i] - mean) * std_inv
    // Using scale and axpy/sub
    for (std::size_t i = 0; i < n; ++i)
        data[i] = (data[i] - mean) * std_inv;
}

// Softmax: data[i] = exp(data[i] - max) / sum(exp(data - max))
inline void simd_softmax(double *data, std::size_t n)
{
    if (n == 0)
        return;
    double mx = simd_reduce_max(data, n);
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::exp(data[i] - mx);
    double sum = simd_reduce_sum(data, n);
    double inv_sum = 1.0 / sum;
    simd_scale(data, inv_sum, n);
}

// Float16 NN primitives

inline void simd_batchnorm_f16(half *data, std::size_t n, float mean, float std_inv)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half((half_to_float(data[i]) - mean) * std_inv);
}

inline void simd_softmax_f16(half *data, std::size_t n)
{
    if (n == 0)
        return;
    float mx = simd_reduce_max_f16(data, n);
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::exp(half_to_float(data[i]) - mx));
    float sum = simd_reduce_sum_f16(data, n);
    half inv_sum = float_to_half(1.0f / sum);
    simd_scale_f16(data, inv_sum, n);
}

} // namespace nerve::simd
