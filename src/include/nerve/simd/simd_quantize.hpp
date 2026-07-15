#pragma once
#include "nerve/simd/simd_base.hpp"

#include <cstdint>

namespace nerve::simd
{

// Quantize double[n] to uint8[n] using [0, 1] -> [0, (1<<bits)-1]
inline void simd_quantize(const double *input, std::size_t n, int bits, std::uint8_t *output)
{
    double scale = static_cast<double>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
    {
        double clipped = std::max(0.0, std::min(input[i], 1.0));
        output[i] = static_cast<std::uint8_t>(clipped * scale + 0.5);
    }
}

// Dequantize uint8[n] to double[n]
inline void simd_dequantize(const std::uint8_t *input, std::size_t n, int bits, double *output)
{
    double inv_scale = 1.0 / static_cast<double>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
        output[i] = static_cast<double>(input[i]) * inv_scale;
}

} // namespace nerve::simd
