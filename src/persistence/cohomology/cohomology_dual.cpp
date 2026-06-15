#include "nerve/persistence/cohomology/cohomology.hpp"
#include "nerve/persistence/cohomology/cohomology_ops.hpp"
#include "nerve/spectral/symmetric_eigendecomposition.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>

namespace nerve::persistence
{
DualPersistentHomology::DualPersistentHomology(const algebra::CellularComplex &complex)
    : complex_(complex)
    , cohomology_(complex)
{}
std::vector<Pair> DualPersistentHomology::computeDualPersistence(
    const std::vector<std::pair<algebra::Cell, double>> &filtration) const
{
    return cohomology_.computePersistentCohomology(filtration);
}
std::vector<std::vector<double>> DualPersistentHomology::computeKernelBasis(int dimension) const
{
    return cohomology_.computeKernel(dimension);
}
std::vector<std::vector<double>> DualPersistentHomology::computeCokernelBasis(int dimension) const
{
    return cohomology_.computeCokernel(dimension);
}
std::vector<std::vector<double>>
DualPersistentHomology::computeDualMap(const std::vector<std::vector<double>> &matrix) const
{
    if (matrix.empty())
    {
        return {};
    }
    const Size n = matrix.size();
    const Size m = matrix.front().size();
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
DualPersistentHomology::computeAdjointMap(const std::vector<std::vector<double>> &matrix) const
{
    return computeDualMap(matrix);
}
std::vector<double> DualPersistentHomology::computeSpectrum(int dimension) const
{
    const auto laplacian = computeLaplacian(dimension);
    if (laplacian.empty())
    {
        return {};
    }
    try
    {
        return spectral::detail::jacobiEigendecomposition(laplacian).eigenvalues;
    }
    catch (const std::invalid_argument &)
    {
        const Size n = laplacian.size();
        std::vector<double> diagonal_values(n, 0.0);
        for (Size i = 0; i < n; ++i)
        {
            if (i < laplacian[i].size())
            {
                diagonal_values[i] = laplacian[i][i];
            }
        }
        return diagonal_values;
    }
}
std::vector<std::vector<double>> DualPersistentHomology::computeEigenvectors(int dimension) const
{
    const auto laplacian = computeLaplacian(dimension);
    if (laplacian.empty())
    {
        return {};
    }
    try
    {
        return spectral::detail::jacobiEigendecomposition(laplacian).eigenvectors;
    }
    catch (const std::invalid_argument &)
    {
        const Size n = laplacian.size();
        std::vector<std::vector<double>> identity(n, std::vector<double>(n, 0.0));
        for (Size i = 0; i < n; ++i)
        {
            identity[i][i] = 1.0;
        }
        return identity;
    }
}
std::vector<std::vector<double>> DualPersistentHomology::computeLaplacian(int /*dimension*/) const
{
    const auto boundary = complex_.computeBoundaryMatrix();
    const auto coboundary = complex_.computeCoboundaryMatrix();
    const Size n = boundary.size();
    std::vector<std::vector<double>> laplacian(n, std::vector<double>(n, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            for (Size k = 0; k < n; ++k)
            {
                laplacian[i][j] += boundary[i][k] * coboundary[k][j];
            }
        }
    }
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            for (Size k = 0; k < n; ++k)
            {
                laplacian[i][j] += coboundary[i][k] * boundary[k][j];
            }
        }
    }
    return laplacian;
}
std::vector<std::vector<double>> DualPersistentHomology::computeDiracOperator() const
{
    const auto boundary = complex_.computeBoundaryMatrix();
    const auto coboundary = complex_.computeCoboundaryMatrix();
    const Size n = boundary.size();
    const Size m = 2 * n;
    std::vector<std::vector<double>> dirac(m, std::vector<double>(m, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            dirac[i][n + j] = coboundary[i][j];
        }
    }
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            dirac[n + i][j] = boundary[i][j];
        }
    }
    return dirac;
}
} // namespace nerve::persistence
