#include "nerve/persistence/cohomology/cohomology.hpp"
#include "nerve/persistence/cohomology/cohomology_ops.hpp"
#include "nerve/persistence/cohomology/cohomology_rref.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>

namespace nerve::persistence
{
namespace
{
std::vector<double> multiply_matrix_vector(const std::vector<std::vector<double>> &matrix,
                                           const std::vector<double> &vector)
{
    std::vector<double> result;
    if (matrix.empty())
    {
        return result;
    }
    result.resize(matrix.size(), 0.0);
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        for (size_t j = 0; j < matrix[i].size() && j < vector.size(); ++j)
        {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}
std::vector<std::vector<double>> submatrix(const std::vector<std::vector<double>> &matrix,
                                           const std::vector<Index> &row_indices,
                                           const std::vector<Index> &col_indices)
{
    std::vector<std::vector<double>> result;
    result.reserve(row_indices.size());
    for (Index r : row_indices)
    {
        if (r < 0 || static_cast<size_t>(r) >= matrix.size())
        {
            continue;
        }
        std::vector<double> row;
        row.reserve(col_indices.size());
        for (Index c : col_indices)
        {
            if (c < 0 || static_cast<size_t>(c) >= matrix[r].size())
            {
                row.push_back(0.0);
            }
            else
            {
                row.push_back(matrix[r][c]);
            }
        }
        result.push_back(std::move(row));
    }
    return result;
}
void validateFiniteVector(const std::vector<double> &values)
{
    for (double value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("vector entries must be finite");
        }
    }
}
} // namespace
SheafCohomology::SheafCohomology(const algebra::CellularComplex &complex)
    : complex_(complex)
{}
void SheafCohomology::assignSection(const algebra::Cell &cell, const std::vector<double> &section)
{
    validateFiniteVector(section);
    sections_[cell] = section;
}
void SheafCohomology::assignCochain(const std::vector<algebra::Cell> &cells,
                                    const std::vector<double> &values)
{
    if (cells.size() != values.size())
    {
        throw std::invalid_argument("cochain cells and values must have matching sizes");
    }
    validateFiniteVector(values);
    for (Size i = 0; i < cells.size(); ++i)
    {
        cochains_[cells[i]] = {values[i]};
    }
}
std::vector<std::vector<double>> SheafCohomology::computeSheafCohomology() const
{
    const auto sheaf_coboundary = computeSheafCoboundary();
    const int max_dim = complex_.maxDimension();
    std::vector<std::vector<double>> representatives;
    for (int dim = 0; dim <= max_dim; ++dim)
    {
        const auto cells_k = complex_.cellsOfDimension(dim);
        if (cells_k.empty())
        {
            continue;
        }
        const auto cells_k_plus_1 = complex_.cellsOfDimension(dim + 1);
        const auto cells_k_minus_1 =
            (dim > 0) ? complex_.cellsOfDimension(dim - 1) : std::vector<Index>{};
        const auto delta_k = submatrix(sheaf_coboundary, cells_k_plus_1, cells_k);
        auto kernel_basis = detail::nullspaceBasis(delta_k);
        if (kernel_basis.empty())
        {
            continue;
        }
        const auto delta_k_minus_1 = submatrix(sheaf_coboundary, cells_k, cells_k_minus_1);
        const Size image_rank = detail::matrixRank(delta_k_minus_1);
        if (image_rank >= kernel_basis.size())
        {
            continue;
        }
        const Size keep_count = kernel_basis.size() - image_rank;
        for (Size i = 0; i < keep_count; ++i)
        {
            std::vector<double> full(static_cast<Size>(complex_.numCells()), 0.0);
            for (Size j = 0; j < cells_k.size() && j < kernel_basis[i].size(); ++j)
            {
                const Index idx = cells_k[j];
                if (idx >= 0 && static_cast<Size>(idx) < full.size())
                {
                    full[static_cast<Size>(idx)] = kernel_basis[i][j];
                }
            }
            representatives.push_back(std::move(full));
        }
    }
    return representatives;
}
std::vector<std::vector<double>>
SheafCohomology::computeLocalCohomology(const algebra::Cell &cell) const
{
    const Index center = detail::findCellIndex(complex_, cell);
    if (center < 0)
    {
        return {};
    }
    std::vector<Index> neighborhood;
    neighborhood.push_back(center);
    for (const Index b : complex_.getBoundary(cell))
    {
        if (std::ranges::find(neighborhood, b) == neighborhood.end())
        {
            neighborhood.push_back(b);
        }
    }
    for (const Index c : complex_.getCoboundary(cell))
    {
        if (std::ranges::find(neighborhood, c) == neighborhood.end())
        {
            neighborhood.push_back(c);
        }
    }
    if (neighborhood.empty())
    {
        return {};
    }
    const auto sheaf_coboundary = computeSheafCoboundary();
    const auto local_matrix = submatrix(sheaf_coboundary, neighborhood, neighborhood);
    return detail::nullspaceBasis(local_matrix);
}
std::vector<std::vector<double>>
SheafCohomology::computeSheafMorphism(const std::vector<std::vector<double>> &sections) const
{
    if (sections.empty())
    {
        return {};
    }
    const auto restriction_maps = computeRestrictionMaps();
    if (restriction_maps.empty())
    {
        return sections;
    }
    std::vector<std::vector<double>> mapped;
    mapped.reserve(sections.size());
    for (const auto &section : sections)
    {
        mapped.push_back(multiply_matrix_vector(restriction_maps, section));
    }
    return mapped;
}
std::vector<double> SheafCohomology::computeStalk(const algebra::Cell &cell) const
{
    auto it = sections_.find(cell);
    if (it != sections_.end())
    {
        return it->second;
    }
    return {};
}
std::vector<std::vector<double>> SheafCohomology::computeGerms(const algebra::Cell &cell) const
{
    auto stalk = computeStalk(cell);
    std::vector<std::vector<double>> germs(1, stalk);
    return germs;
}
std::vector<std::vector<double>> SheafCohomology::computeSheafCoboundary() const
{
    return complex_.computeCoboundaryMatrix();
}
std::vector<std::vector<double>> SheafCohomology::computeRestrictionMaps() const
{
    return complex_.computeBoundaryMatrix();
}
} // namespace nerve::persistence
