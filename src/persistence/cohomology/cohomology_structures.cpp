#include "nerve/persistence/cohomology/cohomology.hpp"
#include "nerve/persistence/cohomology/cohomology_ops.hpp" // For SheafCohomology
#include "nerve/persistence/cohomology/cohomology_rref.hpp"
#include "nerve/spectral/symmetric_eigendecomposition.hpp"

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
} // namespace nerve::persistence
