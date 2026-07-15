#include "nerve/algorithms/distance.hpp"
#include "nerve/algorithms/distance_c.h"
#include "nerve/math/constants.hpp"
#include "nerve/simd/simd_base.hpp"
#include "nerve/simd/simd_distance.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace
{
template <nerve::algorithms::Numeric T>
long double safe_coordinate_limit(size_t dim)
{
    if (dim == 0)
    {
        return static_cast<long double>(std::numeric_limits<T>::max());
    }
    const long double max_value = static_cast<long double>(std::numeric_limits<T>::max());
    const long double max_norm_square = std::sqrt(max_value);
    return std::sqrt(max_norm_square / static_cast<long double>(dim)) / 2.0L;
}

template <nerve::algorithms::Numeric T>
void validate_distance_values(std::span<const T> values, size_t dim, std::string_view name)
{
    const long double safe_abs = safe_coordinate_limit<T>(dim);
    for (T value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(std::string(name) + " values must be finite");
        }
        if (std::abs(static_cast<long double>(value)) > safe_abs)
        {
            throw std::invalid_argument(std::string(name) + " values exceed safe distance range");
        }
    }
}

template <nerve::algorithms::Numeric T>
void validate_flat_matrix(std::span<const T> values, size_t rows, size_t dim, std::string_view name)
{
    if (dim != 0 && rows > std::numeric_limits<size_t>::max() / dim)
    {
        throw std::invalid_argument(std::string(name) + " shape overflows size_t");
    }
    const size_t required = rows * dim;
    if (values.size() < required)
    {
        throw std::invalid_argument(std::string(name) + " span is smaller than rows * dim");
    }
    validate_distance_values(values.first(required), dim, name);
}

template <nerve::algorithms::Numeric T>
void validate_equal_vectors(std::span<const T> a, std::span<const T> b)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument("distance vectors must have matching dimensions");
    }
    validate_distance_values(a, a.size(), "distance vector");
    validate_distance_values(b, b.size(), "distance vector");
}

size_t checked_product(size_t lhs, size_t rhs, std::string_view name)
{
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs)
    {
        throw std::invalid_argument(std::string(name) + " size overflows size_t");
    }
    return lhs * rhs;
}

size_t checked_square_count(size_t n, std::string_view name)
{
    return checked_product(n, n, name);
}

size_t checked_upper_triangle_count(size_t n, std::string_view name)
{
    if (n < 2)
    {
        return 0;
    }
    if (n % 2 == 0)
    {
        return checked_product(n / 2, n - 1, name);
    }
    return checked_product(n, (n - 1) / 2, name);
}

template <typename Size>
size_t checked_plus_one(Size value, std::string_view name)
{
    if (value == std::numeric_limits<size_t>::max())
    {
        throw std::invalid_argument(std::string(name) + " size overflows size_t");
    }
    return value + 1;
}

template <nerve::algorithms::Numeric T>
T checked_distance_result(T value, std::string_view name)
{
    if (!std::isfinite(value))
    {
        throw std::overflow_error(std::string(name) + " overflowed");
    }
    return value;
}

#include "detail/distance_metric_dispatch.inl"
} // namespace

namespace nerve::algorithms
{

template <Numeric T>
T EuclideanMetric<T>::compute(std::span<const T> a, std::span<const T> b) const
{
    validate_equal_vectors(a, b);
    const size_t dim = a.size();

    if constexpr (std::is_same_v<T, double>)
    {
        return checked_distance_result(nerve::simd::simd_euclidean(a.data(), b.data(), dim),
                                       "euclidean distance");
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return checked_distance_result(
            static_cast<T>(nerve::simd::simd_euclidean_f32(a.data(), b.data(), dim)),
            "euclidean distance");
    }
    else
    {
        T sum = 0;
        for (size_t i = 0; i < dim; ++i)
        {
            T diff = a[i] - b[i];
            sum += diff * diff;
        }
        return checked_distance_result(std::sqrt(sum), "euclidean distance");
    }
}

template <Numeric T>
T EuclideanMetric<T>::compute_simd(std::span<const T> a, std::span<const T> b)
{
    validate_equal_vectors(a, b);
    const size_t dim = a.size();

    if constexpr (std::is_same_v<T, double>)
    {
        return checked_distance_result(nerve::simd::simd_euclidean(a.data(), b.data(), dim),
                                       "euclidean distance");
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return checked_distance_result(
            static_cast<T>(nerve::simd::simd_euclidean_f32(a.data(), b.data(), dim)),
            "euclidean distance");
    }
    else
    {
        T sum_sq = nerve::math::Constants<T>::kZero;
        for (size_t d = 0; d < dim; ++d)
        {
            const T diff = a[d] - b[d];
            sum_sq += diff * diff;
        }
        return checked_distance_result(std::sqrt(sum_sq), "euclidean distance");
    }
}

template <Numeric T>
std::vector<T> EuclideanMetric<T>::compute_matrix(std::span<const T> points, size_t n_points,
                                                  size_t dim) const
{
    validate_flat_matrix(points, n_points, dim, "points");
    const size_t matrix_size = checked_square_count(n_points, "distance matrix");
    std::vector<T> distances(matrix_size);

#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t j = i + 1; j < n_points; ++j)
        {
            std::span<const T> a(points.data() + i * dim, dim);
            std::span<const T> b(points.data() + j * dim, dim);

            T dist = compute(a, b);
            distances[i * n_points + j] = dist;
            distances[j * n_points + i] = dist;
        }
        distances[i * n_points + i] = 0;
    }

    return distances;
}

template <Numeric T>
T ManhattanMetric<T>::compute(std::span<const T> a, std::span<const T> b) const
{
    validate_equal_vectors(a, b);
    const size_t dim = a.size();

    if constexpr (std::is_same_v<T, double>)
    {
        return checked_distance_result(nerve::simd::simd_manhattan(a.data(), b.data(), dim),
                                       "manhattan distance");
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return checked_distance_result(
            static_cast<T>(nerve::simd::simd_manhattan_f32(a.data(), b.data(), dim)),
            "manhattan distance");
    }
    else
    {
        T sum = 0;
        for (size_t i = 0; i < dim; ++i)
        {
            sum += std::abs(a[i] - b[i]);
        }
        return checked_distance_result(sum, "manhattan distance");
    }
}

template <Numeric T>
std::vector<T> ManhattanMetric<T>::compute_matrix(std::span<const T> points, size_t n_points,
                                                  size_t dim) const
{
    validate_flat_matrix(points, n_points, dim, "points");
    const size_t matrix_size = checked_square_count(n_points, "distance matrix");
    std::vector<T> distances(matrix_size);

#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t j = i + 1; j < n_points; ++j)
        {
            std::span<const T> a(points.data() + i * dim, dim);
            std::span<const T> b(points.data() + j * dim, dim);
            T dist = compute(a, b);
            distances[i * n_points + j] = dist;
            distances[j * n_points + i] = dist;
        }
        distances[i * n_points + i] = 0;
    }

    return distances;
}

template <Numeric T>
T CosineMetric<T>::compute(std::span<const T> a, std::span<const T> b) const
{
    validate_equal_vectors(a, b);
    const size_t dim = a.size();

    if constexpr (std::is_same_v<T, double>)
    {
        return checked_distance_result(nerve::simd::simd_cosine(a.data(), b.data(), dim),
                                       "cosine distance");
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return checked_distance_result(
            static_cast<T>(nerve::simd::simd_cosine_f32(a.data(), b.data(), dim)),
            "cosine distance");
    }
    else
    {
        return cosine_distance(a.data(), b.data(), dim);
    }
}

template <Numeric T>
std::vector<T> CosineMetric<T>::compute_matrix(std::span<const T> points, size_t n_points,
                                               size_t dim) const
{
    validate_flat_matrix(points, n_points, dim, "points");
    const size_t matrix_size = checked_square_count(n_points, "distance matrix");
    std::vector<T> distances(matrix_size, nerve::math::Constants<T>::kZero);

#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t j = i + 1; j < n_points; ++j)
        {
            std::span<const T> a(points.data() + i * dim, dim);
            std::span<const T> b(points.data() + j * dim, dim);
            const T dist = compute(a, b);
            distances[i * n_points + j] = dist;
            distances[j * n_points + i] = dist;
        }
    }

    return distances;
}

template <Numeric T>
std::vector<T> DistanceMatrixComputer<T>::compute(std::span<const T> points, size_t n_points,
                                                  size_t dim) const
{
    validate_flat_matrix(points, n_points, dim, "points");
    switch (config_.metric)
    {
        case Config::Metric::EUCLIDEAN:
            if (config_.use_simd)
            {
                return compute_euclidean_simd(points, n_points, dim);
            }
            if (config_.block_size > 0 && n_points > config_.block_size)
            {
                return compute_blocked(points, n_points, dim, config_.block_size);
            }
            return compute_euclidean(points, n_points, dim);

        case Config::Metric::MANHATTAN:
            return compute_manhattan(points, n_points, dim);

        case Config::Metric::COSINE:
        case Config::Metric::CHEBYSHEV:
        {
            const size_t matrix_size = checked_square_count(n_points, "distance matrix");
            std::vector<T> distances(matrix_size, nerve::math::Constants<T>::kZero);
#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic) if (config_.use_openmp)
#endif
            for (size_t i = 0; i < n_points; ++i)
            {
                for (size_t j = i + 1; j < n_points; ++j)
                {
                    const T dist = distance_from_matrix_metric(
                        points.data() + i * dim, points.data() + j * dim, dim, config_.metric);
                    distances[i * n_points + j] = dist;
                    distances[j * n_points + i] = dist;
                }
            }
            return distances;
        }
        case Config::Metric::MINKOWSKI:
        case Config::Metric::CANBERRA:
        case Config::Metric::BRAYCURTIS:
        case Config::Metric::CORRELATION:
            break;
    }
    return compute_euclidean(points, n_points, dim);
}

template <Numeric T>
std::vector<T> DistanceMatrixComputer<T>::compute_pairwise(std::span<const T> set_a, size_t n_a,
                                                           std::span<const T> set_b, size_t n_b,
                                                           size_t dim) const
{
    validate_flat_matrix(set_a, n_a, dim, "set_a");
    validate_flat_matrix(set_b, n_b, dim, "set_b");
    const size_t matrix_size = checked_product(n_a, n_b, "pairwise distance matrix");
    std::vector<T> distances(matrix_size);
#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic) if (config_.use_openmp)
#endif
    for (size_t i = 0; i < n_a; ++i)
    {
        const T *a_row = &set_a[i * dim];
        for (size_t j = 0; j < n_b; ++j)
        {
            distances[i * n_b + j] = compute_single(a_row, &set_b[j * dim], dim);
        }
    }
    return distances;
}

template <Numeric T>
std::vector<T> DistanceMatrixComputer<T>::compute_symmetric(std::span<const T> points,
                                                            size_t n_points, size_t dim) const
{
    validate_flat_matrix(points, n_points, dim, "points");
    std::vector<T> packed;
    packed.reserve(checked_upper_triangle_count(n_points, "packed distance matrix"));
    for (size_t i = 0; i < n_points; ++i)
    {
        const T *a_row = &points[i * dim];
        for (size_t j = i + 1; j < n_points; ++j)
        {
            packed.push_back(compute_single(a_row, &points[j * dim], dim));
        }
    }
    return packed;
}

template <Numeric T>
std::vector<T> DistanceMatrixComputer<T>::compute_chunked(std::span<const T> points,
                                                          size_t n_points, size_t dim,
                                                          size_t chunk_size) const
{
    validate_flat_matrix(points, n_points, dim, "points");
    if (chunk_size == 0 || chunk_size >= n_points)
    {
        return compute(points, n_points, dim);
    }

    const size_t matrix_size = checked_square_count(n_points, "distance matrix");
    std::vector<T> distances(matrix_size, nerve::math::Constants<T>::kZero);
    for (size_t i0 = 0; i0 < n_points; i0 += chunk_size)
    {
        const size_t i1 = std::min(n_points, i0 + chunk_size);
        for (size_t j0 = i0; j0 < n_points; j0 += chunk_size)
        {
            const size_t j1 = std::min(n_points, j0 + chunk_size);
            for (size_t i = i0; i < i1; ++i)
            {
                const T *a_row = &points[i * dim];
                for (size_t j = std::max(i, j0); j < j1; ++j)
                {
                    const T dist = compute_single(a_row, &points[j * dim], dim);
                    distances[i * n_points + j] = dist;
                    distances[j * n_points + i] = dist;
                }
            }
        }
    }
    return distances;
}

template <Numeric T>
std::vector<T> DistanceMatrixComputer<T>::compute_euclidean(std::span<const T> points,
                                                            size_t n_points, size_t dim) const
{
    validate_flat_matrix(points, n_points, dim, "points");
    const size_t matrix_size = checked_square_count(n_points, "distance matrix");
    std::vector<T> distances(matrix_size);

#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic) if (config_.use_openmp)
#endif
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t j = i; j < n_points; ++j)
        {
            T dist;
            if constexpr (std::is_same_v<T, double>)
            {
                dist = checked_distance_result(
                    nerve::simd::simd_euclidean(&points[i * dim], &points[j * dim], dim),
                    "euclidean distance");
            }
            else if constexpr (std::is_same_v<T, float>)
            {
                dist = checked_distance_result(static_cast<T>(nerve::simd::simd_euclidean_f32(
                                                   &points[i * dim], &points[j * dim], dim)),
                                               "euclidean distance");
            }
            else
            {
                T sum = 0;
                for (size_t d = 0; d < dim; ++d)
                {
                    T diff = points[i * dim + d] - points[j * dim + d];
                    sum += diff * diff;
                }
                dist = checked_distance_result(std::sqrt(sum), "euclidean distance");
            }

            distances[i * n_points + j] = dist;
            if (i != j)
            {
                distances[j * n_points + i] = dist;
            }
        }
    }

    return distances;
}

template <Numeric T>
std::vector<T> DistanceMatrixComputer<T>::compute_manhattan(std::span<const T> points,
                                                            size_t n_points, size_t dim) const
{
    validate_flat_matrix(points, n_points, dim, "points");
    const size_t matrix_size = checked_square_count(n_points, "distance matrix");
    std::vector<T> distances(matrix_size);

#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic) if (config_.use_openmp)
#endif
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t j = i; j < n_points; ++j)
        {
            T dist;
            if constexpr (std::is_same_v<T, double>)
            {
                dist = checked_distance_result(
                    nerve::simd::simd_manhattan(&points[i * dim], &points[j * dim], dim),
                    "manhattan distance");
            }
            else if constexpr (std::is_same_v<T, float>)
            {
                dist = checked_distance_result(static_cast<T>(nerve::simd::simd_manhattan_f32(
                                                   &points[i * dim], &points[j * dim], dim)),
                                               "manhattan distance");
            }
            else
            {
                T sum = 0;
                for (size_t d = 0; d < dim; ++d)
                {
                    sum += std::abs(points[i * dim + d] - points[j * dim + d]);
                }
                dist = checked_distance_result(sum, "manhattan distance");
            }

            distances[i * n_points + j] = dist;
            if (i != j)
            {
                distances[j * n_points + i] = dist;
            }
        }
    }

    return distances;
}

template <Numeric T>
T DistanceMatrixComputer<T>::compute_single(const T *a, const T *b, size_t dim) const
{
    if constexpr (std::is_same_v<T, double>)
    {
        switch (config_.metric)
        {
            case Config::Metric::EUCLIDEAN:
                return checked_distance_result(nerve::simd::simd_euclidean(a, b, dim),
                                               "euclidean distance");
            case Config::Metric::MANHATTAN:
                return checked_distance_result(nerve::simd::simd_manhattan(a, b, dim),
                                               "manhattan distance");
            case Config::Metric::COSINE:
                return checked_distance_result(nerve::simd::simd_cosine(a, b, dim),
                                               "cosine distance");
            default:
                return distance_from_matrix_metric(a, b, dim, config_.metric);
        }
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        switch (config_.metric)
        {
            case Config::Metric::EUCLIDEAN:
                return checked_distance_result(
                    static_cast<T>(nerve::simd::simd_euclidean_f32(a, b, dim)),
                    "euclidean distance");
            case Config::Metric::MANHATTAN:
                return checked_distance_result(
                    static_cast<T>(nerve::simd::simd_manhattan_f32(a, b, dim)),
                    "manhattan distance");
            case Config::Metric::COSINE:
                return checked_distance_result(
                    static_cast<T>(nerve::simd::simd_cosine_f32(a, b, dim)), "cosine distance");
            default:
                return distance_from_matrix_metric(a, b, dim, config_.metric);
        }
    }
    else
    {
        return distance_from_matrix_metric(a, b, dim, config_.metric);
    }
}

#include "detail/distance_c_api_instantiations.inl"

} // namespace nerve::algorithms
