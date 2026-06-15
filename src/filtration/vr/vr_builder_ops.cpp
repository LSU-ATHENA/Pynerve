
#include "nerve/filtration/vietoris_rips.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <ranges>

namespace nerve::filtration
{
namespace
{

bool checkedSquareCount(Size count, Size &out) noexcept
{
    if (count != 0 && count > std::numeric_limits<Size>::max() / count)
    {
        return false;
    }
    out = count * count;
    return true;
}

} // namespace

std::vector<std::pair<algebra::Simplex, double>>
computeVietorisRipsFiltration(const core::ownership_utils::PointView &points, size_t dimension,
                              double max_radius, Size max_dimension)
{
    if (dimension == 0 || points.size() == 0 || points.size() % dimension != 0)
    {
        return {};
    }

    VietorisRips vr(max_radius);
    vr.setMaxDimension(max_dimension);
    const auto result = vr.buildFiltration(points, dimension);
    if (result.isError())
    {
        return {};
    }
    return result.value();
}

std::vector<std::pair<algebra::Simplex, double>>
computeWeightedVietorisRips(const core::ownership_utils::PointView &points, size_t dimension,
                            const std::vector<double> &weights, double max_radius)
{
    if (dimension == 0 || points.size() == 0 || points.size() % dimension != 0)
    {
        return {};
    }
    if (!std::isfinite(max_radius))
    {
        return {};
    }

    WeightedVietorisRips wvr(weights);
    auto result = wvr.buildFiltration(points, dimension);
    if (max_radius <= 0.0)
    {
        return result;
    }

    std::vector<std::pair<algebra::Simplex, double>> filtered;
    filtered.reserve(result.size());
    for (const auto &entry : result)
    {
        if (entry.second <= max_radius)
        {
            filtered.push_back(entry);
        }
    }
    return filtered;
}

core::ownership_utils::OwnedPointBuffer
computeAllPairDistances(const core::ownership_utils::PointView &points, size_t dimension)
{
    if (dimension == 0 || points.size() == 0 || points.size() % dimension != 0)
    {
        return core::ownership_utils::OwnedPointBuffer();
    }

    const Size n = points.size() / dimension;
    const auto *data = static_cast<const double *>(points.data());
    if (data == nullptr)
    {
        return core::ownership_utils::OwnedPointBuffer();
    }
    Size matrix_size = 0;
    if (!checkedSquareCount(n, matrix_size))
    {
        return core::ownership_utils::OwnedPointBuffer();
    }
    auto matrix_data = std::make_unique<double[]>(matrix_size);
    for (Size i = 0; i < n; ++i)
    {
        matrix_data[i * n + i] = 0.0;
        for (Size j = i + 1; j < n; ++j)
        {
            double dist_sq = 0.0;
            const double *pi = data + i * dimension;
            const double *pj = data + j * dimension;
            for (Size d = 0; d < dimension; ++d)
            {
                const double diff = pi[d] - pj[d];
                dist_sq += diff * diff;
            }
            const double dist = std::sqrt(dist_sq);
            matrix_data[i * n + j] = dist;
            matrix_data[j * n + i] = dist;
        }
    }
    return core::ownership_utils::OwnedPointBuffer(matrix_data.release(), n, n);
}

std::vector<std::pair<Index, Index>>
findKNearestNeighbors(const core::ownership_utils::PointView &points, size_t dimension,
                      Index point_index, Size k)
{
    std::vector<std::pair<Index, Index>> neighbors;
    if (dimension == 0 || points.size() == 0 || points.size() % dimension != 0)
    {
        return neighbors;
    }

    const Size n = points.size() / dimension;
    if (point_index < 0 || static_cast<Size>(point_index) >= n)
    {
        return neighbors;
    }

    const auto *data = static_cast<const double *>(points.data());
    if (data == nullptr || k == 0)
    {
        return neighbors;
    }
    const auto *p0 = data + static_cast<Size>(point_index) * dimension;
    std::vector<std::pair<double, Index>> distances;
    distances.reserve(std::min(k, n > 0 ? n - 1 : 0));
    const auto farther = [](const std::pair<double, Index> &lhs,
                            const std::pair<double, Index> &rhs) {
        if (lhs.first != rhs.first)
        {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    };
    for (Size i = 0; i < n; ++i)
    {
        if (i == static_cast<Size>(point_index))
        {
            continue;
        }

        const auto *pi = data + i * dimension;
        double dist_sq = 0.0;
        for (Size d = 0; d < dimension; ++d)
        {
            const double diff = p0[d] - pi[d];
            dist_sq += diff * diff;
        }
        const std::pair<double, Index> candidate{dist_sq, static_cast<Index>(i)};
        if (distances.size() < k)
        {
            distances.push_back(candidate);
            std::ranges::push_heap(distances, farther);
        }
        else if (candidate < distances.front())
        {
            std::ranges::pop_heap(distances, farther);
            distances.back() = candidate;
            std::ranges::push_heap(distances, farther);
        }
    }

    std::ranges::sort(distances);

    neighbors.reserve(distances.size());
    for (Size i = 0; i < distances.size(); ++i)
    {
        neighbors.emplace_back(distances[i].second, point_index);
    }

    return neighbors;
}

} // namespace nerve::filtration
