#include "nerve/compression/simd_compression.hpp"
#include "nerve/simd/simd_quantize.hpp"

#include <algorithm>
#include <cmath>

namespace nerve::compression
{

void simdQuantize(const double *input, std::size_t n, int bits, std::uint8_t *output)
{
    nerve::simd::simd_quantize(input, n, bits, output);
}

void simdDequantize(const std::uint8_t *encoded, std::size_t n, int bits, double *output)
{
    nerve::simd::simd_dequantize(encoded, n, bits, output);
}

} // namespace nerve::compression
