#include "detail/mapper_safe_arithmetic.hpp"
#include "nerve/algorithms/mapper.hpp"
#include "nerve/math/constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>

namespace nerve::algorithms
{

namespace
{

template <typename T>
T validated_overlap(T overlap)
{
    if (!detail::is_finite_value(overlap))
    {
        throw std::invalid_argument("cover overlap must be finite");
    }
    return std::clamp(overlap, static_cast<T>(0), static_cast<T>(0.95));
}

template <typename T>
void require_finite_interval(T value, const char *name)
{
    if (!detail::is_finite_value(value))
    {
        throw std::overflow_error(std::string(name) + " overflowed");
    }
}

} // namespace

template <typename T>
std::vector<std::vector<size_t>> Cover<T>::build(std::span<const T> filter_values, size_t n_points,
                                                 int n_filter_dims) const
{
    if (n_points == 0 || filter_values.empty())
    {
        return {};
    }
    if (n_filter_dims <= 0)
    {
        return {};
    }
    if (n_filter_dims == 1)
    {
        return build_1d(filter_values, n_points);
    }
    else
    {
        return build_nd(filter_values, n_points, n_filter_dims);
    }
}

template <typename T>
std::vector<std::vector<size_t>> Cover<T>::build_1d(std::span<const T> filter_values,
                                                    size_t n_points) const
{
    if (n_points == 0 || filter_values.size() < n_points)
    {
        return {};
    }

    const int resolution = std::max(1, config_.resolution);
    const T overlap = validated_overlap(config_.overlap);
    detail::require_finite_prefix(filter_values, n_points, "filter values");
    T min_val = *std::min_element(filter_values.begin(), filter_values.begin() + n_points);
    T max_val = *std::max_element(filter_values.begin(), filter_values.begin() + n_points);

    T range = max_val - min_val;
    require_finite_interval(range, "cover range");
    if (range < nerve::math::epsilon<T>())
    {
        std::vector<std::vector<size_t>> cover(1);
        for (size_t i = 0; i < n_points; ++i)
        {
            cover[0].push_back(i);
        }
        return cover;
    }

    T interval_size = range / static_cast<T>(resolution);
    T overlap_size = interval_size * overlap;
    require_finite_interval(interval_size, "cover interval size");
    require_finite_interval(overlap_size, "cover overlap size");

    std::vector<std::vector<size_t>> cover;

    for (int i = 0; i < resolution; ++i)
    {
        T start = min_val + i * interval_size - overlap_size;
        T end = min_val + (i + 1) * interval_size + overlap_size;
        require_finite_interval(start, "cover interval start");
        require_finite_interval(end, "cover interval end");

        std::vector<size_t> set;
        for (size_t j = 0; j < n_points; ++j)
        {
            if (filter_values[j] >= start && filter_values[j] <= end)
            {
                set.push_back(j);
            }
        }

        if (!set.empty())
        {
            cover.push_back(std::move(set));
        }
    }

    return cover;
}

template <typename T>
std::vector<std::vector<size_t>> Cover<T>::build_nd(std::span<const T> filter_values,
                                                    size_t n_points, int n_filter_dims) const
{
    if (n_filter_dims <= 1)
    {
        return build_1d(filter_values, n_points);
    }
    const size_t filter_dims = static_cast<size_t>(n_filter_dims);
    if (n_points == 0 || !detail::has_flat_span(filter_values.size(), n_points, filter_dims))
    {
        return {};
    }

    const int resolution = std::max(1, config_.resolution);
    const T overlap = validated_overlap(config_.overlap);
    const size_t required_values =
        detail::checked_product_or_throw(n_points, filter_dims, "filter values");
    detail::require_finite_prefix(filter_values, required_values, "filter values");

    std::vector<T> mins(filter_dims, std::numeric_limits<T>::max());
    std::vector<T> maxs(filter_dims, std::numeric_limits<T>::lowest());
    for (size_t i = 0; i < n_points; ++i)
    {
        for (int d = 0; d < n_filter_dims; ++d)
        {
            const T value =
                filter_values[i * static_cast<size_t>(n_filter_dims) + static_cast<size_t>(d)];
            mins[static_cast<size_t>(d)] = std::min(mins[static_cast<size_t>(d)], value);
            maxs[static_cast<size_t>(d)] = std::max(maxs[static_cast<size_t>(d)], value);
        }
    }

    std::vector<T> interval_sizes(filter_dims, nerve::math::Constants<T>::kZero);
    std::vector<T> overlap_sizes(filter_dims, nerve::math::Constants<T>::kZero);
    for (int d = 0; d < n_filter_dims; ++d)
    {
        const T range = maxs[static_cast<size_t>(d)] - mins[static_cast<size_t>(d)];
        require_finite_interval(range, "cover range");
        if (range > nerve::math::epsilon<T>())
        {
            interval_sizes[static_cast<size_t>(d)] = range / static_cast<T>(resolution);
            overlap_sizes[static_cast<size_t>(d)] =
                interval_sizes[static_cast<size_t>(d)] * overlap;
            require_finite_interval(interval_sizes[static_cast<size_t>(d)], "cover interval size");
            require_finite_interval(overlap_sizes[static_cast<size_t>(d)], "cover overlap size");
        }
    }

    std::map<std::vector<int>, std::vector<size_t>> buckets;
    std::vector<std::vector<int>> memberships(filter_dims);

    for (size_t point_idx = 0; point_idx < n_points; ++point_idx)
    {
        for (int d = 0; d < n_filter_dims; ++d)
        {
            auto &dim_memberships = memberships[static_cast<size_t>(d)];
            dim_memberships.clear();
            const T value = filter_values[point_idx * static_cast<size_t>(n_filter_dims) +
                                          static_cast<size_t>(d)];
            const T interval_size = interval_sizes[static_cast<size_t>(d)];

            if (interval_size <= nerve::math::epsilon<T>())
            {
                dim_memberships.push_back(0);
                continue;
            }

            for (int interval = 0; interval < resolution; ++interval)
            {
                const T start = mins[static_cast<size_t>(d)] +
                                static_cast<T>(interval) * interval_size -
                                overlap_sizes[static_cast<size_t>(d)];
                const T end = mins[static_cast<size_t>(d)] +
                              static_cast<T>(interval + 1) * interval_size +
                              overlap_sizes[static_cast<size_t>(d)];
                require_finite_interval(start, "cover interval start");
                require_finite_interval(end, "cover interval end");
                if (value >= start && value <= end)
                {
                    dim_memberships.push_back(interval);
                }
            }

            if (dim_memberships.empty())
            {
                const T normalized = (value - mins[static_cast<size_t>(d)]) /
                                     std::max(interval_size, nerve::math::epsilon<T>());
                require_finite_interval(normalized, "cover normalized coordinate");
                int default_value = 0;
                if (normalized >= static_cast<T>(resolution - 1))
                {
                    default_value = resolution - 1;
                }
                else if (normalized > static_cast<T>(0))
                {
                    default_value = static_cast<int>(normalized);
                }
                dim_memberships.push_back(default_value);
            }
        }

        std::vector<int> key;
        key.reserve(filter_dims);
        auto emit_memberships = [&](auto &&self, int dim_index) -> void {
            if (dim_index == n_filter_dims)
            {
                buckets[key].push_back(point_idx);
                return;
            }
            for (int interval : memberships[static_cast<size_t>(dim_index)])
            {
                key.push_back(interval);
                self(self, dim_index + 1);
                key.pop_back();
            }
        };
        emit_memberships(emit_memberships, 0);
    }

    std::vector<std::vector<size_t>> cover;
    cover.reserve(buckets.size());
    for (auto &[_, points] : buckets)
    {
        cover.push_back(std::move(points));
    }
    return cover;
}

template class Cover<float>;
template class Cover<double>;

} // namespace nerve::algorithms
