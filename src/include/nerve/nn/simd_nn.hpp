#pragma once
#include "nerve/core_types.hpp"

#include <vector>

namespace nerve::nn
{

void simdReLU(double *data, Size n);
void simdSigmoid(double *data, Size n);
void simdTanh(double *data, Size n);
void simdBatchNorm(double *data, Size n, double mean, double std_inv);
void simdSoftmax(double *data, Size n);

} // namespace nerve::nn
