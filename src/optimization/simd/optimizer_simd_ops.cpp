#include "nerve/simd/simd_base.hpp"

#include <cmath>

namespace nerve::optimization
{

void simdClipGradients(double *grads, std::size_t n, double max_norm)
{
    nerve::simd::simd_clamp(grads, -max_norm, max_norm, n);
}

double simdL2Norm(const double *vec, std::size_t n)
{
    return nerve::simd::simd_norm2(vec, n);
}

} // namespace nerve::optimization
