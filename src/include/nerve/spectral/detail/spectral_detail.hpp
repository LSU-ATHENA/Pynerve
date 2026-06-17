#pragma once

#include "nerve/core_types.hpp"
#include "nerve/spectral/laplacian.hpp"
#include "nerve/spectral/persistent_laplacian.hpp"

#include <vector>

namespace nerve::spectral
{

namespace simd
{
void vectorScale(const double *in, double *out, double scale, size_t n);
double vectorNorm(const double *v, size_t n);
void matrixVectorMultiply(const double *mat, const double *vec, double *out, size_t n);
void matVecMulSimd(const double *mat, const double *vec, size_t n, double *result);
double dotProductSimd(const double *a, const double *b, size_t n);
} // namespace simd

class PersistentLaplacianSupport
{
public:
    PersistentLaplacianSupport();
    std::vector<size_t> computeDimensions(const std::vector<std::vector<int>> &simplices) const;
};

} // namespace nerve::spectral
