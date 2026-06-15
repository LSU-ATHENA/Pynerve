#pragma once
#include "nerve/core_types.hpp"

#include <vector>

namespace nerve::encoders
{

void simdEncodeBatch(const double *input, Size n, Size dim, double *output);
void simdDecodeBatch(const double *encoded, Size n, Size code_dim, double *output);

} // namespace nerve::encoders
