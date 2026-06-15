#pragma once
#include "nerve/core_types.hpp"

#include <cstddef>
#include <cstdint>

namespace nerve::compression
{

void simdQuantize(const double *input, std::size_t n, int bits, std::uint8_t *output);
void simdDequantize(const std::uint8_t *encoded, std::size_t n, int bits, double *output);

} // namespace nerve::compression
