#pragma once
#include "nerve/core_types.hpp"

#include <vector>

namespace nerve::filtration
{

void simdBatchFilterValues(double *values, Size n, Size start_dim, Size end_dim, double threshold);
void simdSortPairsByBirth(Pair *pairs, Size n);

} // namespace nerve::filtration
