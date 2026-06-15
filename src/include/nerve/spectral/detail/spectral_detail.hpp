#pragma once
#include "nerve/core_types.hpp"
#include "nerve/spectral/laplacian.hpp"

#include <vector>

namespace nerve::spectral
{
class SpectralAnomalyDetector
{
public:
    SpectralAnomalyDetector(size_t dim);
    bool initialize();
    bool isInitialized() const;
};

class SpectralFeatureExtractor
{
public:
    struct Config
    {
        size_t num_features = 10;
        bool normalize = true;
    };
    explicit SpectralFeatureExtractor(const Config &config = Config{});
    Config getConfig() const;
};

namespace simd
{
void vectorScale(const double *in, double *out, double scale, size_t n);
double vectorNorm(const double *v, size_t n);
void matrixVectorMultiply(const double *mat, const double *vec, double *out, size_t n);
} // namespace simd

class DiracOperator
{
public:
    DiracOperator(int dimension);
    int dimension() const;
    std::vector<std::vector<double>> getMatrix() const;
};

class PersistentLaplacianSupport
{
public:
    PersistentLaplacianSupport();
    std::vector<size_t> computeDimensions(const std::vector<std::vector<int>> &simplices) const;
};
} // namespace nerve::spectral
