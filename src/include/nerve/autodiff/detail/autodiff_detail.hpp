#pragma once

#include "nerve/autodiff/autodiff.hpp"
#include "nerve/core_types.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace nerve::autodiff
{

// SIMD autodiff
namespace simd
{
void backwardAdd(const double *grad, double *out, size_t n);
void backwardMul(const double *grad, const double *a, const double *b, double *out_a, double *out_b,
                 size_t n);
void backwardRelu(const double *grad, const double *input, double *out, size_t n);
void simdBackwardAdd(const double *grad, double *out, size_t n);
void simdBackwardMul(const double *grad, const double *mid, const double *rhs, size_t n);
void simdBackwardRelu(const double *grad, const double *input, double *out, size_t n);
} // namespace simd

} // namespace nerve::autodiff
