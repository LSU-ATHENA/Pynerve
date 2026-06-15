#include "detail/vr_construction_helpers.inl"
#include "nerve/filtration/vietoris_rips.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <ranges>
#include <tuple>

namespace nerve::filtration
{

double VietorisRips::simplexRadius(const std::vector<Index> &vertices) const
{
    if (vertices.size() <= 1)
    {
        return 0.0;
    }

    double max_distance = 0.0;
    for (Size i = 0; i < vertices.size(); ++i)
    {
        for (Size j = i + 1; j < vertices.size(); ++j)
        {
            const Index vi = vertices[i];
            const Index vj = vertices[j];
            if (vi < static_cast<Index>(distance_matrix_.size()) &&
                vj < static_cast<Index>(distance_matrix_[vi].size()))
            {
                max_distance = std::max(max_distance, distance_matrix_[vi][vj]);
            }
        }
    }
    return max_distance;
}

void VietorisRips::sortFiltration()
{
    std::ranges::sort(filtration_, {}, [](const auto &p) {
        return std::tuple(p.second, p.first.dimension(), p.first);
    });
}

std::vector<Index> VietorisRips::findSimplexVertices(double radius, Size k) const
{
    const Size n = points_.size();
    if (k + 1 > n)
    {
        return {};
    }

    std::vector<bool> selection(n, false);
    std::fill(
        std::next(selection.begin(), static_cast<std::vector<bool>::difference_type>(n - k - 1)),
        selection.end(), true);
    do
    {
        std::vector<Index> vertices;
        vertices.reserve(k + 1);
        for (Size i = 0; i < n; ++i)
        {
            if (selection[i])
            {
                vertices.push_back(static_cast<Index>(i));
            }
        }
        if (vertices.size() == k + 1 && isSimplexValid(vertices, radius))
        {
            return vertices;
        }
    } while (std::next_permutation(selection.begin(), selection.end()));

    return {};
}

bool VietorisRips::isSimplexValid(const std::vector<Index> &vertices, double radius) const
{
    if (vertices.size() <= 1)
    {
        return true;
    }

    for (Size i = 0; i < vertices.size(); ++i)
    {
        for (Size j = i + 1; j < vertices.size(); ++j)
        {
            const Index vi = vertices[i];
            const Index vj = vertices[j];
            if (vi < 0 || vj < 0 || static_cast<Size>(vi) >= distance_matrix_.size() ||
                static_cast<Size>(vj) >= distance_matrix_.size())
            {
                return false;
            }
            if (distance_matrix_[static_cast<Size>(vi)][static_cast<Size>(vj)] > radius)
            {
                return false;
            }
        }
    }
    return true;
}

Size VietorisRips::getPointDimension() const
{
    if (points_.empty())
    {
        return 0;
    }
    return points_[0].size();
}

} // namespace nerve::filtration
