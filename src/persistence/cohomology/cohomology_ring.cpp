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
constexpr double kCohomologyEpsilon = 1e-12;
std::vector<std::vector<double>> reduceMod2(const std::vector<std::vector<double>> &matrix)
{
    std::vector<std::vector<double>> reduced = matrix;
    for (auto &row : reduced)
    {
        for (double &value : row)
        {
            const long long rounded = static_cast<long long>(std::llround(value));
            value = (rounded & 1LL) != 0 ? 1.0 : 0.0;
        }
    }
    return reduced;
}
void validateSquareMatrix(const std::vector<std::vector<double>> &matrix)
{
    const Size n = matrix.size();
    for (const auto &row : matrix)
    {
        if (row.size() != n)
        {
            throw std::invalid_argument("matrix must be square");
        }
        for (double value : row)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument("matrix entries must be finite");
            }
        }
    }
}
} // namespace
CohomologyRing::CohomologyRing(const algebra::CellularComplex &complex)
    : complex_(complex)
{
    coboundary_matrix_ = complex_.computeCoboundaryMatrix();
}
std::vector<std::vector<double>>
CohomologyRing::cupProduct(const std::vector<std::vector<double>> &alpha,
                           const std::vector<std::vector<double>> &beta) const
{
    return computeCupProductMatrix(alpha, beta);
}
std::vector<std::vector<double>>
CohomologyRing::steenrodSquare(const std::vector<std::vector<double>> &alpha, int i) const
{
    if (alpha.empty() || i < 0)
    {
        return {};
    }
    if (i == 0)
    {
        return reduceMod2(alpha);
    }
    std::vector<std::vector<double>> result = cupProduct(alpha, alpha);
    if (i > 1)
    {
        for (int k = 2; k <= i; ++k)
        {
            result = computeCupProductMatrix(result, alpha);
        }
    }
    for (auto &row : result)
    {
        for (double &value : row)
        {
            if (std::abs(value) < kCohomologyEpsilon)
            {
                value = 0.0;
            }
        }
    }
    return reduceMod2(result);
}
std::vector<std::vector<std::vector<double>>> CohomologyRing::computeMultiplicationTable() const
{
    const int max_dim = complex_.maxDimension();
    std::vector<std::vector<std::vector<double>>> table(max_dim + 1);
    for (int i = 0; i <= max_dim; ++i)
    {
        table[i].resize(max_dim + 1);
        for (int j = 0; j <= max_dim; ++j)
        {
            const Size n = coboundary_matrix_.size();
            table[i][j] = std::vector<double>(n, 0.0);
            for (Size k = 0; k < n; ++k)
            {
                if (k < coboundary_matrix_.size() && !coboundary_matrix_[k].empty())
                {
                    const Size col =
                        static_cast<Size>((i + j) % static_cast<int>(coboundary_matrix_[k].size()));
                    table[i][j][k] = coboundary_matrix_[k][col];
                }
            }
        }
    }
    return table;
}
std::vector<std::vector<double>> CohomologyRing::computeRingGenerators() const
{
    const Size n = coboundary_matrix_.size();
    std::vector<std::vector<double>> generators;
    for (Size i = 0; i < n; ++i)
    {
        std::vector<double> generator(n, 0.0);
        generator[i] = 1.0;
        generators.push_back(generator);
    }
    return generators;
}
std::vector<int> CohomologyRing::computePoincarePolynomial() const
{
    const auto betti = computeBettiNumbers();
    std::vector<int> polynomial;
    polynomial.reserve(betti.size());
    for (const auto &value : betti)
    {
        polynomial.push_back(value.empty() ? 0 : value.front());
    }
    return polynomial;
}
std::vector<std::vector<int>> CohomologyRing::computeBettiNumbers() const
{
    auto result = complex_.computeBettiNumbers();
    if (result.isError())
    {
        return {};
    }
    const auto betti = result.moveValue();
    std::vector<std::vector<int>> out;
    out.reserve(betti.size());
    for (int value : betti)
    {
        out.push_back({value});
    }
    return out;
}
std::vector<std::vector<double>>
CohomologyRing::computeCupProductMatrix(const std::vector<std::vector<double>> &alpha,
                                        const std::vector<std::vector<double>> &beta) const
{
    validateSquareMatrix(alpha);
    validateSquareMatrix(beta);
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
} // namespace nerve::persistence
