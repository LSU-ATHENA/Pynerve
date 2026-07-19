#include "nerve/nn/simd_nn.hpp"
#include "nerve/simd/simd_base.hpp"

#include <cmath>

namespace nerve::nn
{

void simdReLU(double *data, std::size_t n)
{
    nerve::simd::simd_relu(data, n);
}

void simdSigmoid(double *data, std::size_t n)
{
    nerve::simd::simd_sigmoid(data, n);
}

void simdTanh(double *data, std::size_t n)
{
    nerve::simd::simd_tanh(data, n);
}

void simdBatchNorm(double *data, std::size_t n, double mean, double std_inv)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = (data[i] - mean) * std_inv;
}

void simdSoftmax(double *data, std::size_t n)
{
    if (n == 0)
        return;

    double max_val = nerve::simd::simd_reduce_max(data, n);
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::exp(data[i] - max_val);

    double sum = nerve::simd::simd_reduce_sum(data, n);
    double inv_sum = 1.0 / sum;
    nerve::simd::simd_scale(data, inv_sum, n);
}

} // namespace nerve::nn
