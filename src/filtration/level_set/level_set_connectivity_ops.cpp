
#include "nerve/filtration/level_set.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace nerve::filtration
{
void LevelSet::validateScalarField(const std::vector<double> &scalar_field) const
{
    if (scalar_field.empty())
    {
        throw std::invalid_argument("Scalar field cannot be empty");
    }
    Size expected_size = getTotalGridPoints();
    if (!grid_shape_.empty() && scalar_field.size() != expected_size)
    {
        throw std::invalid_argument("Scalar field size does not match grid shape");
    }
    for (double value : scalar_field)
    {
        if (std::isnan(value) || std::isinf(value))
        {
            throw std::invalid_argument("Scalar field contains invalid values");
        }
    }
}
#ifdef USE_TORCH
void LevelSet::validateTensor(const core::Tensor &scalar_field) const
{
    if (scalar_field.size() == 0)
    {
        throw std::invalid_argument("Tensor cannot be empty");
    }
}
#endif
void LevelSet::buildGridConnectivity()
{
    use_mesh_connectivity_ = false;
    mesh_neighbors_.clear();
}
void LevelSet::buildMeshConnectivity(const std::vector<std::vector<Index>> &connectivity)
{
    use_mesh_connectivity_ = true;
    Index max_vertex = -1;
    for (const auto &simplex : connectivity)
    {
        for (const Index vertex : simplex)
        {
            max_vertex = std::max(max_vertex, vertex);
        }
    }

    if (max_vertex < 0)
    {
        mesh_neighbors_.clear();
        return;
    }

    mesh_neighbors_.assign(static_cast<Size>(max_vertex) + 1, std::vector<Index>());

    for (const auto &simplex : connectivity)
    {
        if (simplex.size() < 2)
        {
            continue;
        }
        for (Size i = 0; i < simplex.size(); ++i)
        {
            for (Size j = i + 1; j < simplex.size(); ++j)
            {
                const Index u = simplex[i];
                const Index v = simplex[j];
                if (u < 0 || v < 0 || u == v)
                {
                    continue;
                }
                auto &u_neighbors = mesh_neighbors_[static_cast<Size>(u)];
                auto &v_neighbors = mesh_neighbors_[static_cast<Size>(v)];
                u_neighbors.push_back(v);
                v_neighbors.push_back(u);
            }
        }
    }

    for (auto &neighbors : mesh_neighbors_)
    {
        std::ranges::sort(neighbors);
        const auto [first, last] = std::ranges::unique(neighbors);
        neighbors.erase(first, last);
    }
}
std::vector<Index> LevelSet::getGridNeighbors(Index point_index) const
{
    std::vector<Index> neighbors;
    if (use_mesh_connectivity_)
    {
        if (point_index >= 0 && static_cast<Size>(point_index) < mesh_neighbors_.size())
        {
            return mesh_neighbors_[static_cast<Size>(point_index)];
        }
        return neighbors;
    }

    if (point_index < 0 || grid_shape_.empty() ||
        static_cast<Size>(point_index) >= getTotalGridPoints())
    {
        return neighbors;
    }

    std::vector<Index> coords = linearIndexToGrid(point_index);
    if (coords.size() != grid_shape_.size())
    {
        return neighbors;
    }

    Size dim = grid_shape_.size();
    for (Size d = 0; d < dim; ++d)
    {
        for (int delta = -1; delta <= 1; delta += 2)
        {
            std::vector<Index> neighbor_coords = coords;
            neighbor_coords[d] += delta;
            bool valid = true;
            for (Size i = 0; i < dim; ++i)
            {
                if (neighbor_coords[i] < 0 ||
                    neighbor_coords[i] >= static_cast<Index>(grid_shape_[i]))
                {
                    valid = false;
                    break;
                }
            }
            if (valid)
            {
                Index neighbor_index = gridIndexToLinear(neighbor_coords);
                if (neighbor_index >= 0)
                {
                    neighbors.push_back(neighbor_index);
                }
            }
        }
    }
    return neighbors;
}
Index LevelSet::gridIndexToLinear(const std::vector<Index> &grid_coords) const
{
    if (grid_shape_.empty() || grid_coords.size() != grid_shape_.size())
    {
        return -1;
    }
    Size linear_index = 0;
    Size stride = 1;
    for (Size i = 0; i < grid_coords.size(); ++i)
    {
        if (grid_coords[i] < 0 || grid_coords[i] >= static_cast<Index>(grid_shape_[i]))
        {
            return -1;
        }
        linear_index += static_cast<Size>(grid_coords[i]) * stride;
        stride *= grid_shape_[i];
    }
    if (linear_index > static_cast<Size>(std::numeric_limits<Index>::max()))
    {
        return -1;
    }
    return static_cast<Index>(linear_index);
}
std::vector<Index> LevelSet::linearIndexToGrid(Index linear_index) const
{
    std::vector<Index> coords;
    if (grid_shape_.empty())
    {
        return coords;
    }

    if (linear_index < 0)
    {
        return coords;
    }

    Index remaining = linear_index;
    for (Size i = 0; i < grid_shape_.size(); ++i)
    {
        coords.push_back(remaining % static_cast<Index>(grid_shape_[i]));
        remaining /= static_cast<Index>(grid_shape_[i]);
    }
    return coords;
}
void LevelSet::buildVertexSimplices(const std::vector<double> &scalar_field)
{
    for (Size i = 0; i < scalar_field.size(); ++i)
    {
        algebra::Simplex vertex({static_cast<Index>(i)});
        double filtration_value = assignFiltrationValue(vertex, scalar_field);
        filtration_.emplace_back(vertex, filtration_value);
    }
}
void LevelSet::buildEdgeSimplices(const std::vector<double> &scalar_field)
{
    const Index n = static_cast<Index>(scalar_field.size());
    for (Index i = 0; i < n; ++i)
    {
        const std::vector<Index> neighbors = getGridNeighbors(i);
        for (const Index neighbor : neighbors)
        {
            if (neighbor <= i || neighbor < 0 || neighbor >= n)
            {
                continue;
            }
            algebra::Simplex edge({i, neighbor});
            const double filtration_value = assignFiltrationValue(edge, scalar_field);
            filtration_.emplace_back(edge, filtration_value);
        }
    }
}
void LevelSet::buildTriangleSimplices(const std::vector<double> &scalar_field)
{
    if (grid_shape_.size() < 2)
        return;
    Size width = grid_shape_[0];
    Size height = grid_shape_[1];
    if (width < 2 || height < 2)
        return;
    for (Size y = 0; y < height - 1; ++y)
    {
        for (Size x = 0; x < width - 1; ++x)
        {
            Index v0 = static_cast<Index>(y * width + x);
            Index v1 = static_cast<Index>(y * width + (x + 1));
            Index v2 = static_cast<Index>((y + 1) * width + x);
            Index v3 = static_cast<Index>((y + 1) * width + (x + 1));
            algebra::Simplex tri1({v0, v1, v2});
            double filtration_value1 = assignFiltrationValue(tri1, scalar_field);
            filtration_.emplace_back(tri1, filtration_value1);
            algebra::Simplex tri2({v1, v3, v2});
            double filtration_value2 = assignFiltrationValue(tri2, scalar_field);
            filtration_.emplace_back(tri2, filtration_value2);
        }
    }
}
void LevelSet::buildTetrahedronSimplices(const std::vector<double> &scalar_field)
{
    if (grid_shape_.size() < 3)
        return;
    Size width = grid_shape_[0];
    Size height = grid_shape_[1];
    Size depth = grid_shape_[2];
    if (width < 2 || height < 2 || depth < 2)
        return;
    for (Size z = 0; z < depth - 1; ++z)
    {
        for (Size y = 0; y < height - 1; ++y)
        {
            for (Size x = 0; x < width - 1; ++x)
            {
                Index v0 = static_cast<Index>(z * width * height + y * width + x);
                Index v1 = static_cast<Index>(z * width * height + y * width + (x + 1));
                Index v2 = static_cast<Index>(z * width * height + (y + 1) * width + x);
                Index v3 = static_cast<Index>(z * width * height + (y + 1) * width + (x + 1));
                Index v4 = static_cast<Index>((z + 1) * width * height + y * width + x);
                Index v5 = static_cast<Index>((z + 1) * width * height + y * width + (x + 1));
                Index v6 = static_cast<Index>((z + 1) * width * height + (y + 1) * width + x);
                Index v7 = static_cast<Index>((z + 1) * width * height + (y + 1) * width + (x + 1));
                std::vector<std::vector<Index>> tetrahedra = {{v0, v1, v3, v4}, {v1, v2, v3, v7},
                                                              {v1, v3, v4, v7}, {v1, v5, v3, v7},
                                                              {v3, v6, v4, v7}, {v3, v7, v6, v4}};
                for (const auto &tetra : tetrahedra)
                {
                    algebra::Simplex tet(tetra);
                    double filtration_value = assignFiltrationValue(tet, scalar_field);
                    filtration_.emplace_back(tet, filtration_value);
                }
            }
        }
    }
}
double LevelSet::assignFiltrationValue(const algebra::Simplex &simplex,
                                       const std::vector<double> &scalar_field) const
{
    std::vector<Index> vertices = getSimplexVertices(simplex);
    if (filtration_type_ == "sublevel")
    {
        double max_value = scalar_field[vertices[0]];
        for (Size i = 1; i < vertices.size(); ++i)
        {
            max_value = std::max(max_value, scalar_field[vertices[i]]);
        }
        return max_value;
    }
    else if (filtration_type_ == "superlevel")
    {
        double min_value = scalar_field[vertices[0]];
        for (Size i = 1; i < vertices.size(); ++i)
        {
            min_value = std::min(min_value, scalar_field[vertices[i]]);
        }
        return -min_value;
    }
    else
    {
        double sum = 0.0;
        for (Index vertex : vertices)
        {
            sum += scalar_field[vertex];
        }
        return sum / static_cast<double>(vertices.size());
    }
}
std::vector<Index> LevelSet::getSimplexVertices(const algebra::Simplex &simplex) const
{
    return simplex.vertices();
}
void LevelSet::sortFiltration()
{
    std::ranges::sort(filtration_, {},
                      [](const auto &f) { return std::pair(f.second, f.first.dimension()); });
}
Size LevelSet::getGridDimension() const
{
    return grid_shape_.size();
}
Size LevelSet::getTotalGridPoints() const
{
    if (grid_shape_.empty())
        return 0;
    Size total = 1;
    for (Size dim : grid_shape_)
    {
        total *= dim;
    }
    return total;
}
} // namespace nerve::filtration
