#include "nerve/algebra/simd_distance.hpp"
#include "nerve/simd/simd_base.hpp"
#include "nerve/simd/simd_distance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nerve::algebra
{

namespace
{
size_t checkedPairCount(size_t n)
{
    if (n > 1 && n > std::numeric_limits<size_t>::max() / (n - 1))
        throw std::overflow_error("pairwise distance count overflows size_t");
    return n * (n - 1) / 2;
}

size_t checkedElementCount(size_t rows, size_t dim)
{
    if (dim != 0 && rows > std::numeric_limits<size_t>::max() / dim)
        throw std::overflow_error("point cloud element count overflows size_t");
    return rows * dim;
}

bool valuesAreFinite(const double *values, size_t count)
{
    return std::all_of(values, values + count, [](double value) { return std::isfinite(value); });
}

double checkedDistanceResult(double value, const char *message)
{
    if (!std::isfinite(value))
        throw std::overflow_error(message);
    return value;
}

void validateDistanceInputs(const double *a, const double *b, size_t dim)
{
    if (dim == 0)
        return;
    if (a == nullptr || b == nullptr)
        throw std::invalid_argument("distance vector inputs must not be null");
    if (!valuesAreFinite(a, dim) || !valuesAreFinite(b, dim))
        throw std::invalid_argument("distance vector inputs must be finite");
}

size_t validateBatchInputs(const double *points, size_t num_points, size_t dim)
{
    const size_t pair_count = checkedPairCount(num_points);
    const size_t element_count = checkedElementCount(num_points, dim);
    if (element_count != 0 && points == nullptr)
        throw std::invalid_argument("point cloud input must not be null");
    if (element_count != 0 && !valuesAreFinite(points, element_count))
        throw std::invalid_argument("point cloud input must be finite");
    return pair_count;
}
} // anonymous namespace

SIMDDistanceCalculator::SIMDDistanceCalculator()
{
    nerve::simd::simd_init();
}

bool SIMDDistanceCalculator::hasAvx512() const noexcept
{
    return nerve::simd::detect_simd_arch() == nerve::simd::SimdArch::AVX512;
}

bool SIMDDistanceCalculator::hasAvx2() const noexcept
{
    auto arch = nerve::simd::detect_simd_arch();
    return arch == nerve::simd::SimdArch::AVX2 || arch == nerve::simd::SimdArch::AVX512;
}

bool SIMDDistanceCalculator::hasFma() const noexcept
{
    auto arch = nerve::simd::detect_simd_arch();
    return arch == nerve::simd::SimdArch::AVX2 || arch == nerve::simd::SimdArch::AVX512;
}

double SIMDDistanceCalculator::euclideanDistance(const double *a, const double *b, size_t dim)
{
    validateDistanceInputs(a, b, dim);
    return checkedDistanceResult(nerve::simd::simd_euclidean(a, b, dim),
                                 "euclidean distance overflow");
}

double SIMDDistanceCalculator::euclideanDistanceScalar(const double *a, const double *b, size_t dim)
{
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return checkedDistanceResult(std::sqrt(sum), "euclidean distance overflow");
}

double SIMDDistanceCalculator::manhattanDistance(const double *a, const double *b, size_t dim)
{
    validateDistanceInputs(a, b, dim);
    return checkedDistanceResult(nerve::simd::simd_manhattan(a, b, dim),
                                 "manhattan distance overflow");
}

double SIMDDistanceCalculator::manhattanDistanceScalar(const double *a, const double *b, size_t dim)
{
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i)
        sum += std::abs(a[i] - b[i]);
    return checkedDistanceResult(sum, "manhattan distance overflow");
}

double SIMDDistanceCalculator::cosineDistance(const double *a, const double *b, size_t dim)
{
    validateDistanceInputs(a, b, dim);
    return checkedDistanceResult(nerve::simd::simd_cosine(a, b, dim), "cosine distance overflow");
}

double SIMDDistanceCalculator::cosineDistanceScalar(const double *a, const double *b, size_t dim)
{
    double dot_product = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    if (!std::isfinite(dot_product) || !std::isfinite(norm_a) || !std::isfinite(norm_b))
        throw std::overflow_error("cosine distance overflow");
    if (norm_a == 0.0 || norm_b == 0.0)
        return 1.0;
    double cosine_sim = dot_product / (norm_a * norm_b);
    if (!std::isfinite(cosine_sim))
        throw std::overflow_error("cosine distance overflow");
    cosine_sim = std::max(-1.0, std::min(1.0, cosine_sim));
    return 1.0 - cosine_sim;
}

std::vector<double> SIMDDistanceCalculator::batchEuclideanDistances(const double *points,
                                                                    size_t num_points, size_t dim)
{
    const size_t pair_count = validateBatchInputs(points, num_points, dim);
    std::vector<double> distances;
    distances.reserve(pair_count);
    if (pair_count == 0)
        return distances;
    if (dim == 0)
    {
        distances.assign(pair_count, 0.0);
        return distances;
    }
    for (size_t i = 0; i < num_points; ++i)
    {
        const double *pi = points + i * dim;
        for (size_t j = i + 1; j < num_points; ++j)
        {
            distances.push_back(nerve::simd::simd_euclidean(pi, points + j * dim, dim));
        }
    }
    return distances;
}

} // namespace nerve::algebra
