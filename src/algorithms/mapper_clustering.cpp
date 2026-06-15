#include "detail/mapper_safe_arithmetic.hpp"
#include "nerve/algorithms/mapper.hpp"
#include "nerve/math/constants.hpp"

#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace nerve::algorithms
{

template <typename T>
void DBSCANClustering<T>::region_query(std::span<const T> points, size_t n_points, size_t dim,
                                       size_t point_idx, std::vector<size_t> &neighbors) const
{
    neighbors.clear();

    for (size_t j = 0; j < n_points; ++j)
    {
        if (point_idx == j)
            continue;

        T dist_sq = nerve::math::Constants<T>::kZero;
        for (size_t d = 0; d < dim; ++d)
        {
            T diff = points[point_idx * dim + d] - points[j * dim + d];
            dist_sq += diff * diff;
        }

        if (std::sqrt(dist_sq) <= config_.eps)
        {
            neighbors.push_back(j);
        }
    }
}

template <typename T>
std::vector<int> DBSCANClustering<T>::cluster(std::span<const T> points, size_t n_points,
                                              size_t dim) const
{
    if (n_points == 0 || dim == 0 || !detail::has_flat_span(points, n_points, dim) ||
        !detail::fits_int(n_points))
    {
        return {};
    }
    if (!detail::is_finite_value(config_.eps))
    {
        throw std::invalid_argument("DBSCAN eps must be finite");
    }
    const size_t point_value_count = detail::checked_product_or_throw(n_points, dim, "points");
    detail::require_finite_prefix(points, point_value_count, "points");
    std::vector<int> labels(n_points, -1);
    std::vector<bool> visited(n_points, false);
    int cluster_id = 0;
    const auto min_samples = static_cast<size_t>(std::max(1, config_.min_samples));

    std::vector<size_t> neighbors;

    for (size_t i = 0; i < n_points; ++i)
    {
        if (visited[i])
            continue;

        visited[i] = true;
        region_query(points, n_points, dim, i, neighbors);

        if (neighbors.size() < min_samples)
        {
            labels[i] = -1;
            continue;
        }

        labels[i] = cluster_id;

        std::queue<size_t> seeds;
        for (auto &neighbor : neighbors)
        {
            if (labels[neighbor] == -1)
            {
                labels[neighbor] = cluster_id;
            }
            if (!visited[neighbor])
            {
                visited[neighbor] = true;
                seeds.push(neighbor);
            }
        }

        while (!seeds.empty())
        {
            size_t current = seeds.front();
            seeds.pop();

            region_query(points, n_points, dim, current, neighbors);

            if (neighbors.size() >= min_samples)
            {
                for (auto &neighbor : neighbors)
                {
                    if (labels[neighbor] == -1)
                    {
                        labels[neighbor] = cluster_id;
                    }
                    if (!visited[neighbor])
                    {
                        visited[neighbor] = true;
                        seeds.push(neighbor);
                    }
                }
            }
        }

        cluster_id++;
    }

    return labels;
}

template <typename T>
std::vector<int> SingleLinkageClustering<T>::cluster(std::span<const T> points, size_t n_points,
                                                     size_t dim) const
{
    if (n_points == 0 || dim == 0 || !detail::has_flat_span(points, n_points, dim) ||
        !detail::fits_int(n_points))
    {
        return {};
    }
    if (!detail::is_finite_value(config_.linkage_distance))
    {
        throw std::invalid_argument("single linkage distance must be finite");
    }
    const size_t point_value_count = detail::checked_product_or_throw(n_points, dim, "points");
    detail::require_finite_prefix(points, point_value_count, "points");
    std::vector<int> parent(n_points);
    std::iota(parent.begin(), parent.end(), 0);

    auto find = [&parent](int x) {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    auto unite = [&parent, &find](int x, int y) {
        x = find(x);
        y = find(y);
        if (x != y)
        {
            parent[y] = x;
        }
    };

    std::vector<std::tuple<T, size_t, size_t>> edges;

    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t j = i + 1; j < n_points; ++j)
        {
            T dist_sq = nerve::math::Constants<T>::kZero;
            for (size_t d = 0; d < dim; ++d)
            {
                T diff = points[i * dim + d] - points[j * dim + d];
                dist_sq += diff * diff;
            }
            edges.emplace_back(std::sqrt(dist_sq), i, j);
        }
    }

    std::sort(edges.begin(), edges.end());

    for (const auto &[dist, i, j] : edges)
    {
        if (dist <= config_.linkage_distance)
        {
            unite(static_cast<int>(i), static_cast<int>(j));
        }
        else
        {
            break;
        }
    }

    std::vector<int> labels(n_points);
    for (size_t i = 0; i < n_points; ++i)
    {
        labels[i] = find(static_cast<int>(i));
    }

    return labels;
}

template class DBSCANClustering<float>;
template class DBSCANClustering<double>;
template class SingleLinkageClustering<float>;
template class SingleLinkageClustering<double>;

} // namespace nerve::algorithms
