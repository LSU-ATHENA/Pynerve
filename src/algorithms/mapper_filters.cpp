#include "detail/mapper_safe_arithmetic.hpp"
#include "nerve/algorithms/distance.hpp"
#include "nerve/algorithms/mapper.hpp"
#include "nerve/math/constants.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace nerve::algorithms
{

template <typename T>
std::vector<T> PCAFilter<T>::apply(std::span<const T> points, size_t n_points, size_t dim) const
{
    if (n_points == 0 || dim == 0 || n_components_ <= 0 ||
        !detail::has_flat_span(points, n_points, dim))
    {
        return {};
    }
    const size_t point_value_count = detail::checked_product_or_throw(n_points, dim, "points");
    detail::require_finite_prefix(points, point_value_count, "points");

    const size_t component_count = static_cast<size_t>(n_components_);
    size_t result_size = 0;
    size_t covariance_size = 0;
    if (!detail::checked_product(n_points, component_count, result_size) ||
        !detail::checked_product(dim, dim, covariance_size))
    {
        return {};
    }

    std::vector<T> result(result_size);
    const size_t components = std::min(component_count, dim);

    std::vector<T> mean(dim, nerve::math::Constants<T>::kZero);
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t d = 0; d < dim; ++d)
        {
            mean[d] += points[i * dim + d];
        }
    }
    for (size_t d = 0; d < dim; ++d)
    {
        mean[d] /= static_cast<T>(n_points);
    }

    std::vector<T> covariance(covariance_size, nerve::math::Constants<T>::kZero);
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t r = 0; r < dim; ++r)
        {
            const T centered_r = points[i * dim + r] - mean[r];
            for (size_t c = 0; c < dim; ++c)
            {
                const T centered_c = points[i * dim + c] - mean[c];
                covariance[r * dim + c] += centered_r * centered_c;
            }
        }
    }
    const T denom = static_cast<T>(std::max<size_t>(1, n_points - 1));
    for (T &value : covariance)
    {
        value /= denom;
    }

    std::vector<std::vector<T>> basis(components,
                                      std::vector<T>(dim, nerve::math::Constants<T>::kZero));
    constexpr int kPowerIterations = 32;

    for (size_t comp = 0; comp < components; ++comp)
    {
        auto &vector = basis[comp];
        vector[comp % dim] = nerve::math::Constants<T>::kOne;

        for (int iter = 0; iter < kPowerIterations; ++iter)
        {
            std::vector<T> next(dim, nerve::math::Constants<T>::kZero);
            for (size_t r = 0; r < dim; ++r)
            {
                T dot = nerve::math::Constants<T>::kZero;
                for (size_t c = 0; c < dim; ++c)
                {
                    dot += covariance[r * dim + c] * vector[c];
                }
                next[r] = dot;
            }

            for (size_t prev = 0; prev < comp; ++prev)
            {
                const auto &prev_vec = basis[prev];
                T projection = nerve::math::Constants<T>::kZero;
                for (size_t d = 0; d < dim; ++d)
                {
                    projection += next[d] * prev_vec[d];
                }
                for (size_t d = 0; d < dim; ++d)
                {
                    next[d] -= projection * prev_vec[d];
                }
            }

            T norm_sq = nerve::math::Constants<T>::kZero;
            for (T coord : next)
            {
                norm_sq += coord * coord;
            }
            const T norm = std::sqrt(norm_sq);
            if (norm <= nerve::math::epsilon<T>())
            {
                break;
            }
            for (size_t d = 0; d < dim; ++d)
            {
                vector[d] = next[d] / norm;
            }
        }
    }

    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t comp = 0; comp < components; ++comp)
        {
            T projection = nerve::math::Constants<T>::kZero;
            for (size_t d = 0; d < dim; ++d)
            {
                projection += (points[i * dim + d] - mean[d]) * basis[comp][d];
            }
            result[i * component_count + comp] = projection;
        }
        for (size_t comp = components; comp < component_count; ++comp)
        {
            result[i * component_count + comp] = nerve::math::Constants<T>::kZero;
        }
    }

    return result;
}

template <typename T>
std::vector<T> EccentricityFilter<T>::apply(std::span<const T> points, size_t n_points,
                                            size_t dim) const
{
    if (n_points == 0 || dim == 0 || !detail::has_flat_span(points, n_points, dim))
    {
        return {};
    }
    const size_t point_value_count = detail::checked_product_or_throw(n_points, dim, "points");
    detail::require_finite_prefix(points, point_value_count, "points");
    std::vector<T> eccentricity(n_points, nerve::math::Constants<T>::kZero);

#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < n_points; ++i)
    {
        T max_dist = nerve::math::Constants<T>::kZero;
        for (size_t j = 0; j < n_points; ++j)
        {
            if (i == j)
                continue;

            T dist_sq = nerve::math::Constants<T>::kZero;
            for (size_t d = 0; d < dim; ++d)
            {
                T diff = points[i * dim + d] - points[j * dim + d];
                dist_sq += diff * diff;
            }
            max_dist = std::max(max_dist, std::sqrt(dist_sq));
        }
        eccentricity[i] = max_dist;
    }

    return eccentricity;
}

template <typename T>
std::vector<T> DensityFilter<T>::apply(std::span<const T> points, size_t n_points, size_t dim) const
{
    if (n_points == 0 || dim == 0 || !detail::has_flat_span(points, n_points, dim))
    {
        return {};
    }
    const size_t point_value_count = detail::checked_product_or_throw(n_points, dim, "points");
    detail::require_finite_prefix(points, point_value_count, "points");
    DistanceMatrixComputer<T> dist_computer;
    auto dist_matrix = dist_computer.compute(points, n_points, dim);

    std::vector<T> density(n_points);

#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < n_points; ++i)
    {
        std::vector<std::pair<T, size_t>> dists;
        const size_t effective_k =
            std::min(static_cast<size_t>(std::max(1, k_neighbors_)), n_points - 1);
        dists.reserve(effective_k);
        const auto farther = [](const std::pair<T, size_t> &lhs, const std::pair<T, size_t> &rhs) {
            if (lhs.first != rhs.first)
            {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        };
        for (size_t j = 0; j < n_points; ++j)
        {
            if (i != j)
            {
                const std::pair<T, size_t> candidate{dist_matrix[i * n_points + j], j};
                if (dists.size() < effective_k)
                {
                    dists.push_back(candidate);
                    std::push_heap(dists.begin(), dists.end(), farther);
                }
                else if (candidate < dists.front())
                {
                    std::pop_heap(dists.begin(), dists.end(), farther);
                    dists.back() = candidate;
                    std::push_heap(dists.begin(), dists.end(), farther);
                }
            }
        }

        if (effective_k == 0)
        {
            density[i] = nerve::math::Constants<T>::kZero;
            continue;
        }

        T avg_dist = nerve::math::Constants<T>::kZero;
        for (const auto &[dist, index] : dists)
        {
            (void)index;
            avg_dist += dist;
        }
        avg_dist /= static_cast<T>(effective_k);

        density[i] = nerve::math::Constants<T>::kOne / (avg_dist + nerve::math::epsilon<T>());
    }

    return density;
}

template class PCAFilter<float>;
template class PCAFilter<double>;
template class EccentricityFilter<float>;
template class EccentricityFilter<double>;
template class DensityFilter<float>;
template class DensityFilter<double>;

} // namespace nerve::algorithms
