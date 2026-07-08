#include "nerve/simd/simd_base.hpp"
#include "nerve/encoders/simd_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace nerve::encoders
{

namespace
{
void encodeScalar(const double *input, Size n, Size dim, double *output)
{
    const Size total = n * dim;
    for (Size i = 0; i < total; ++i)
        output[i] = std::tanh(input[i]);
}
} // namespace

void simdEncodeBatch(const double *input, Size n, Size dim, double *output)
{
    if (input == nullptr || output == nullptr || n == 0 || dim == 0)
        return;

    const Size total = n * dim;

    // Apply tanh via SIMD dispatch (the tanh primitive delegates
    // to the best available vectorized implementation)
    std::memcpy(output, input, static_cast<std::size_t>(total) * sizeof(double));
    nerve::simd::simd_tanh(output, static_cast<std::size_t>(total));
}

void simdDecodeBatch(const double *encoded, Size n, Size code_dim, double *output)
{
    if (encoded == nullptr || output == nullptr || n == 0 || code_dim == 0)
        return;

    // Decode is currently identity -- just copy
    std::memcpy(output, encoded,
                static_cast<std::size_t>(n) * static_cast<std::size_t>(code_dim) * sizeof(double));
}

} // namespace nerve::encoders
