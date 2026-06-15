
#include "nerve/algebra/simd_distance.hpp"
#include "nerve/cpu/x86_intrinsics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nerve::algebra
{

namespace
{
#ifdef __AVX512F__
constexpr size_t kAvx512DoubleWidth = 8;
#endif
#ifdef __AVX2__
constexpr size_t kAvx2DoubleWidth = 4;
#endif

size_t checkedPairCount(size_t n)
{
    if (n > 1 && n > std::numeric_limits<size_t>::max() / (n - 1))
    {
        throw std::overflow_error("pairwise distance count overflows size_t");
    }
    return n * (n - 1) / 2;
}

size_t checkedElementCount(size_t rows, size_t dim)
{
    if (dim != 0 && rows > std::numeric_limits<size_t>::max() / dim)
    {
        throw std::overflow_error("point cloud element count overflows size_t");
    }
    return rows * dim;
}

bool valuesAreFinite(const double *values, size_t count)
{
    return std::all_of(values, values + count, [](double value) { return std::isfinite(value); });
}

double checkedDistanceResult(double value, const char *message)
{
    if (!std::isfinite(value))
    {
        throw std::overflow_error(message);
    }
    return value;
}

void validateDistanceInputs(const double *a, const double *b, size_t dim)
{
    if (dim == 0)
    {
        return;
    }
    if (a == nullptr || b == nullptr)
    {
        throw std::invalid_argument("distance vector inputs must not be null");
    }
    if (!valuesAreFinite(a, dim) || !valuesAreFinite(b, dim))
    {
        throw std::invalid_argument("distance vector inputs must be finite");
    }
}

size_t validateBatchInputs(const double *points, size_t num_points, size_t dim)
{
    const size_t pair_count = checkedPairCount(num_points);
    const size_t element_count = checkedElementCount(num_points, dim);
    if (element_count != 0 && points == nullptr)
    {
        throw std::invalid_argument("point cloud input must not be null");
    }
    if (element_count != 0 && !valuesAreFinite(points, element_count))
    {
        throw std::invalid_argument("point cloud input must be finite");
    }
    return pair_count;
}
} // namespace

SIMDDistanceCalculator::SIMDDistanceCalculator()
{
    detectCpuFeatures();
}

void SIMDDistanceCalculator::detectCpuFeatures()
{
    bool compiled_avx512 = false;
    bool compiled_avx2 = false;
    bool compiled_fma = false;
#ifdef __AVX512F__
    compiled_avx512 = true;
#endif
#ifdef __AVX2__
    compiled_avx2 = true;
#endif
#ifdef __FMA__
    compiled_fma = true;
#endif

    bool runtime_avx512 = false;
    bool runtime_avx2 = false;
    bool runtime_fma = false;

#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
    runtime_avx512 = __builtin_cpu_supports("avx512f");
    runtime_avx2 = __builtin_cpu_supports("avx2");
    runtime_fma = __builtin_cpu_supports("fma");
#endif
#endif

    has_avx512_ = compiled_avx512 && runtime_avx512;
    has_avx2_ = compiled_avx2 && runtime_avx2;
    has_fma_ = compiled_fma && runtime_fma;
}

[[nodiscard]] double SIMDDistanceCalculator::euclideanDistanceAvx512(const double *a,
                                                                     const double *b, size_t dim)
{
#ifdef __AVX512F__
    if (dim < kAvx512DoubleWidth)
        return euclideanDistanceScalar(a, b, dim);

    __m512d sum_vec = _mm512_setzero_pd();
    size_t i = 0;

    for (; i + kAvx512DoubleWidth <= dim; i += kAvx512DoubleWidth)
    {
        __m512d a_vec = _mm512_loadu_pd(a + i);
        __m512d b_vec = _mm512_loadu_pd(b + i);
        __m512d diff = _mm512_sub_pd(a_vec, b_vec);
        __m512d squared = _mm512_mul_pd(diff, diff);
        sum_vec = _mm512_add_pd(sum_vec, squared);
    }

    double sum = _mm512_reduce_add_pd(sum_vec);

    for (; i < dim; ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    return checkedDistanceResult(std::sqrt(sum), "euclidean distance overflow");
#else
    return euclideanDistanceScalar(a, b, dim);
#endif
}

double SIMDDistanceCalculator::euclideanDistanceAvx2(const double *a, const double *b, size_t dim)
{
#ifdef __AVX2__
    if (dim < kAvx2DoubleWidth)
        return euclideanDistanceScalar(a, b, dim);

    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + kAvx2DoubleWidth <= dim; i += kAvx2DoubleWidth)
    {
        __m256d a_vec = _mm256_loadu_pd(a + i);
        __m256d b_vec = _mm256_loadu_pd(b + i);
        __m256d diff = _mm256_sub_pd(a_vec, b_vec);
        __m256d squared = _mm256_mul_pd(diff, diff);
        sum_vec = _mm256_add_pd(sum_vec, squared);
    }

    __m256d sum_high_low = _mm256_hadd_pd(sum_vec, sum_vec);
    __m128d sum128 = _mm256_castpd256_pd128(sum_high_low);
    __m128d sum128_high = _mm256_extractf128_pd(sum_high_low, 1);
    sum128 = _mm_add_pd(sum128, sum128_high);
    double sum = _mm_cvtsd_f64(sum128);

    for (; i < dim; ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    return checkedDistanceResult(std::sqrt(sum), "euclidean distance overflow");
#else
    return euclideanDistanceScalar(a, b, dim);
#endif
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

double SIMDDistanceCalculator::euclideanDistance(const double *a, const double *b, size_t dim)
{
    validateDistanceInputs(a, b, dim);
    if (has_avx512_)
    {
        return euclideanDistanceAvx512(a, b, dim);
    }
    else if (has_avx2_)
    {
        return euclideanDistanceAvx2(a, b, dim);
    }
    else
    {
        return euclideanDistanceScalar(a, b, dim);
    }
}

std::vector<double> SIMDDistanceCalculator::batchEuclideanDistances(const double *points,
                                                                    size_t num_points, size_t dim)
{
    const size_t pair_count = validateBatchInputs(points, num_points, dim);
    std::vector<double> distances;
    distances.reserve(pair_count);

    if (pair_count == 0)
    {
        return distances;
    }
    if (dim == 0)
    {
        distances.assign(pair_count, 0.0);
        return distances;
    }

    if (has_avx512_)
    {
        batchEuclideanDistancesAvx512(points, num_points, dim, distances);
    }
    else if (has_avx2_)
    {
        batchEuclideanDistancesAvx2(points, num_points, dim, distances);
    }
    else
    {
        batchEuclideanDistancesScalar(points, num_points, dim, distances);
    }

    return distances;
}

void SIMDDistanceCalculator::batchEuclideanDistancesAvx512(const double *points, size_t num_points,
                                                           size_t dim,
                                                           std::vector<double> &distances)
{
#ifdef __AVX512F__
    for (size_t i = 0; i < num_points; ++i)
    {
        const double *point_i = points + i * dim;
        for (size_t j = i + 1; j < num_points; ++j)
        {
            const double *point_j = points + j * dim;
            distances.push_back(euclideanDistanceAvx512(point_i, point_j, dim));
        }
    }
#else
    batchEuclideanDistancesScalar(points, num_points, dim, distances);
#endif
}

void SIMDDistanceCalculator::batchEuclideanDistancesAvx2(const double *points, size_t num_points,
                                                         size_t dim, std::vector<double> &distances)
{
#ifdef __AVX2__
    for (size_t i = 0; i < num_points; ++i)
    {
        const double *point_i = points + i * dim;
        for (size_t j = i + 1; j < num_points; ++j)
        {
            const double *point_j = points + j * dim;
            distances.push_back(euclideanDistanceAvx2(point_i, point_j, dim));
        }
    }
#else
    batchEuclideanDistancesScalar(points, num_points, dim, distances);
#endif
}

void SIMDDistanceCalculator::batchEuclideanDistancesScalar(const double *points, size_t num_points,
                                                           size_t dim,
                                                           std::vector<double> &distances)
{
    for (size_t i = 0; i < num_points; ++i)
    {
        const double *point_i = points + i * dim;
        for (size_t j = i + 1; j < num_points; ++j)
        {
            const double *point_j = points + j * dim;
            distances.push_back(euclideanDistanceScalar(point_i, point_j, dim));
        }
    }
}

double SIMDDistanceCalculator::manhattanDistanceAvx512(const double *a, const double *b, size_t dim)
{
#ifdef __AVX512F__
    if (dim < kAvx512DoubleWidth)
        return manhattanDistanceScalar(a, b, dim);

    __m512d sum_vec = _mm512_setzero_pd();
    size_t i = 0;

    for (; i + kAvx512DoubleWidth <= dim; i += kAvx512DoubleWidth)
    {
        __m512d a_vec = _mm512_loadu_pd(a + i);
        __m512d b_vec = _mm512_loadu_pd(b + i);
        __m512d diff = _mm512_sub_pd(a_vec, b_vec);
        __m512d abs_diff = _mm512_abs_pd(diff);
        sum_vec = _mm512_add_pd(sum_vec, abs_diff);
    }

    double sum = _mm512_reduce_add_pd(sum_vec);

    for (; i < dim; ++i)
    {
        sum += std::abs(a[i] - b[i]);
    }

    return checkedDistanceResult(sum, "manhattan distance overflow");
#else
    return manhattanDistanceScalar(a, b, dim);
#endif
}

double SIMDDistanceCalculator::manhattanDistanceScalar(const double *a, const double *b, size_t dim)
{
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        sum += std::abs(a[i] - b[i]);
    }
    return checkedDistanceResult(sum, "manhattan distance overflow");
}

double SIMDDistanceCalculator::manhattanDistance(const double *a, const double *b, size_t dim)
{
    validateDistanceInputs(a, b, dim);
    if (has_avx512_)
    {
        return manhattanDistanceAvx512(a, b, dim);
    }
    else
    {
        return manhattanDistanceScalar(a, b, dim);
    }
}

double SIMDDistanceCalculator::cosineDistanceAvx512(const double *a, const double *b, size_t dim)
{
#ifdef __AVX512F__
    if (dim < kAvx512DoubleWidth)
        return cosineDistanceScalar(a, b, dim);

    __m512d dot_product_vec = _mm512_setzero_pd();
    __m512d norm_a_vec = _mm512_setzero_pd();
    __m512d norm_b_vec = _mm512_setzero_pd();

    size_t i = 0;
    for (; i + kAvx512DoubleWidth <= dim; i += kAvx512DoubleWidth)
    {
        __m512d a_vec = _mm512_loadu_pd(a + i);
        __m512d b_vec = _mm512_loadu_pd(b + i);

        __m512d mul = _mm512_mul_pd(a_vec, b_vec);
        dot_product_vec = _mm512_add_pd(dot_product_vec, mul);

        __m512d a_squared = _mm512_mul_pd(a_vec, a_vec);
        norm_a_vec = _mm512_add_pd(norm_a_vec, a_squared);

        __m512d b_squared = _mm512_mul_pd(b_vec, b_vec);
        norm_b_vec = _mm512_add_pd(norm_b_vec, b_squared);
    }

    double dot_product = _mm512_reduce_add_pd(dot_product_vec);
    double norm_a = _mm512_reduce_add_pd(norm_a_vec);
    double norm_b = _mm512_reduce_add_pd(norm_b_vec);

    for (; i < dim; ++i)
    {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);

    if (!std::isfinite(dot_product) || !std::isfinite(norm_a) || !std::isfinite(norm_b))
    {
        throw std::overflow_error("cosine distance overflow");
    }
    if (norm_a == 0.0 || norm_b == 0.0)
        return 1.0;

    double cosine_sim = dot_product / (norm_a * norm_b);
    if (!std::isfinite(cosine_sim))
    {
        throw std::overflow_error("cosine distance overflow");
    }
    cosine_sim = std::max(-1.0, std::min(1.0, cosine_sim));

    return 1.0 - cosine_sim;
#else
    return cosineDistanceScalar(a, b, dim);
#endif
}

double SIMDDistanceCalculator::cosineDistanceScalar(const double *a, const double *b, size_t dim)
{
    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    for (size_t i = 0; i < dim; ++i)
    {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);

    if (!std::isfinite(dot_product) || !std::isfinite(norm_a) || !std::isfinite(norm_b))
    {
        throw std::overflow_error("cosine distance overflow");
    }
    if (norm_a == 0.0 || norm_b == 0.0)
        return 1.0;

    double cosine_sim = dot_product / (norm_a * norm_b);
    if (!std::isfinite(cosine_sim))
    {
        throw std::overflow_error("cosine distance overflow");
    }
    cosine_sim = std::max(-1.0, std::min(1.0, cosine_sim));

    return 1.0 - cosine_sim;
}

double SIMDDistanceCalculator::cosineDistance(const double *a, const double *b, size_t dim)
{
    validateDistanceInputs(a, b, dim);
    if (has_avx512_)
    {
        return cosineDistanceAvx512(a, b, dim);
    }
    else
    {
        return cosineDistanceScalar(a, b, dim);
    }
}

} // namespace nerve::algebra
