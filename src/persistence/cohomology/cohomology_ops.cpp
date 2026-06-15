
#include "nerve/persistence/cohomology/cohomology_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace nerve::persistence
{

namespace
{

void validateSquareMatrix(const std::vector<std::vector<double>> &matrix, std::string_view name)
{
    const Size n = matrix.size();
    for (const auto &row : matrix)
    {
        if (row.size() != n)
        {
            throw std::invalid_argument(std::string(name) + " must be square");
        }
        for (double value : row)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument(std::string(name) + " entries must be finite");
            }
        }
    }
}

} // namespace

PersistentCohomology::PersistentCohomology(const algebra::CellularComplex &complex)
    : complex_(complex)
    , coefficient_field_(2)
    , algorithm_("standard")
    , verbose_(false)
{}

std::vector<std::vector<int>> PersistentCohomology::computeCohomology() const
{
    const std::vector<int> betti = computeBettiNumbers();
    std::vector<std::vector<int>> cohomology(betti.size());
    for (Size dim = 0; dim < betti.size(); ++dim)
    {
        cohomology[dim].reserve(static_cast<Size>(std::max(0, betti[dim])));
        for (int i = 0; i < betti[dim]; ++i)
        {
            cohomology[dim].push_back(i);
        }
    }
    return cohomology;
}

std::vector<int> PersistentCohomology::computeCohomologyGroups() const
{
    const auto cohomology = computeCohomology();
    std::vector<int> groups;
    groups.reserve(cohomology.size());
    for (const auto &group : cohomology)
    {
        groups.push_back(static_cast<int>(group.size()));
    }
    return groups;
}

std::vector<int> PersistentCohomology::computeBettiNumbers() const
{
    auto result = complex_.computeBettiNumbers();
    if (result.isError())
    {
        return {};
    }
    return result.moveValue();
}

std::vector<Pair> PersistentCohomology::computePersistentCohomology(
    const std::vector<std::pair<algebra::Cell, double>> &filtration) const
{
    auto sorted_filtration = filtration;
    for (const auto &[_cell, time] : sorted_filtration)
    {
        if (!std::isfinite(time))
        {
            throw std::invalid_argument("cohomology filtration times must be finite");
        }
    }
    std::ranges::sort(sorted_filtration, {}, &std::pair<algebra::Cell, double>::second);
    std::vector<Pair> persistence_pairs;
    persistence_pairs.reserve(sorted_filtration.size());

    std::unordered_map<Index, double> first_time_by_cell;
    first_time_by_cell.reserve(sorted_filtration.size());
    for (const auto &[cell, time] : sorted_filtration)
    {
        const Index idx = detail::findCellIndex(complex_, cell);
        if (idx < 0)
        {
            continue;
        }
        auto [it, inserted] = first_time_by_cell.emplace(idx, time);
        if (!inserted)
        {
            it->second = std::min(it->second, time);
        }
    }

    for (Size i = 0; i < sorted_filtration.size(); ++i)
    {
        const auto &[cell, birth_time] = sorted_filtration[i];
        const Index cell_index = detail::findCellIndex(complex_, cell);
        if (cell_index < 0)
        {
            continue;
        }

        double death_time = std::numeric_limits<double>::infinity();
        const int target_dim = cell.dimension() + 1;
        for (Size j = i + 1; j < sorted_filtration.size(); ++j)
        {
            const auto &[coface, t] = sorted_filtration[j];
            if (coface.dimension() != target_dim)
            {
                continue;
            }
            for (const Index boundary_index : coface.boundary())
            {
                if (boundary_index == cell_index)
                {
                    death_time = t;
                    break;
                }
            }
            if (death_time < std::numeric_limits<double>::infinity())
            {
                break;
            }
        }

        Pair pair;
        pair.dimension = cell.dimension();
        pair.birth = birth_time;
        pair.death = death_time;
        persistence_pairs.push_back(pair);
    }
    return persistence_pairs;
}

std::vector<std::vector<double>> PersistentCohomology::computeKernel(int dimension) const
{
    const auto coboundary_matrix = computeCoboundaryMatrix();
    std::vector<std::vector<double>> dim_matrix;
    for (Size i = 0; i < coboundary_matrix.size(); ++i)
    {
        if (complex_.getCell(static_cast<Index>(i)).dimension() == dimension)
        {
            dim_matrix.push_back(coboundary_matrix[i]);
        }
    }
    return computeKernelSpace(dim_matrix);
}

std::vector<std::vector<double>> PersistentCohomology::computeImage(int dimension) const
{
    const auto coboundary_matrix = computeCoboundaryMatrix();
    std::vector<std::vector<double>> dim_matrix;
    for (Size i = 0; i < coboundary_matrix.size(); ++i)
    {
        if (complex_.getCell(static_cast<Index>(i)).dimension() == dimension + 1)
        {
            dim_matrix.push_back(coboundary_matrix[i]);
        }
    }
    return dim_matrix;
}

std::vector<std::vector<double>> PersistentCohomology::computeCokernel(int dimension) const
{
    auto kernel = computeKernel(dimension);
    if (kernel.empty())
    {
        return {};
    }
    const auto image = computeImage(dimension - 1);
    const Size image_rank = detail::matrixRank(image);
    if (image_rank >= kernel.size())
    {
        return {};
    }
    return std::vector<std::vector<double>>(
        kernel.begin() + static_cast<std::ptrdiff_t>(image_rank), kernel.end());
}

std::vector<std::vector<double>>
PersistentCohomology::computeCupProduct(const std::vector<std::vector<double>> &alpha,
                                        const std::vector<std::vector<double>> &beta) const
{
    validateSquareMatrix(alpha, "alpha");
    validateSquareMatrix(beta, "beta");
    const Size n = alpha.size();
    if (beta.size() != n)
    {
        throw std::invalid_argument("cup product matrices must have matching shape");
    }
    std::vector<std::vector<double>> result(n, std::vector<double>(n, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            for (Size k = 0; k < n; ++k)
            {
                result[i][j] += alpha[i][k] * beta[k][j];
            }
        }
    }
    return result;
}

void PersistentCohomology::setCoefficientField(int p)
{
    if (p <= 1)
    {
        throw std::invalid_argument("coefficient field characteristic must be greater than one");
    }
    coefficient_field_ = p;
}
void PersistentCohomology::setAlgorithm(const std::string &algorithm)
{
    algorithm_ = algorithm;
}
void PersistentCohomology::setVerbose(bool verbose)
{
    verbose_ = verbose;
}

std::vector<std::vector<double>> PersistentCohomology::computeCoboundaryMatrix() const
{
    return complex_.computeCoboundaryMatrix();
}

std::vector<std::vector<double>>
PersistentCohomology::transposeMatrix(const std::vector<std::vector<double>> &matrix) const
{
    const Size n = matrix.size();
    const Size m = matrix.empty() ? 0 : matrix[0].size();
    std::vector<std::vector<double>> transpose(m, std::vector<double>(n, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < m; ++j)
        {
            transpose[j][i] = matrix[i][j];
        }
    }
    return transpose;
}

std::vector<std::vector<double>>
PersistentCohomology::computeKernelSpace(const std::vector<std::vector<double>> &matrix) const
{
    return detail::nullspaceBasis(matrix);
}

} // namespace nerve::persistence
