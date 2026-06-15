
#include "nerve/algebra/simplex.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace nerve::algebra
{

constexpr double VOLUME_TOLERANCE = 1e-10;

namespace
{

bool referencedRowsAreFinite(const std::vector<Index> &vertices,
                             const std::vector<std::vector<double>> &coords)
{
    if (vertices.empty())
    {
        return false;
    }
    const auto max_vertex = *std::max_element(vertices.begin(), vertices.end());
    if (max_vertex < 0 || coords.size() <= static_cast<Size>(max_vertex))
    {
        return false;
    }
    const Size dim = coords[static_cast<Size>(vertices[0])].size();
    if (dim == 0)
    {
        return false;
    }
    for (Index vertex : vertices)
    {
        const auto &point = coords[static_cast<Size>(vertex)];
        if (point.size() != dim)
        {
            return false;
        }
        if (std::any_of(point.begin(), point.end(),
                        [](double value) { return !std::isfinite(value); }))
        {
            return false;
        }
    }
    return true;
}

double squaredDistance(const std::vector<double> &lhs, const std::vector<double> &rhs)
{
    if (lhs.size() != rhs.size())
    {
        throw std::invalid_argument("simplex coordinate dimensions must match");
    }
    double dist_sq = 0.0;
    for (Size i = 0; i < lhs.size(); ++i)
    {
        const double diff = lhs[i] - rhs[i];
        if (!std::isfinite(diff))
        {
            throw std::overflow_error("simplex volume distance overflowed");
        }
        dist_sq += diff * diff;
        if (!std::isfinite(dist_sq))
        {
            throw std::overflow_error("simplex volume distance overflowed");
        }
    }
    return dist_sq;
}

} // namespace

[[nodiscard]] double Simplex::simplexVolume(const std::vector<std::vector<double>> &coords) const
{
    Size n = vertices_.size();
    if (n == 0 || coords.empty())
    {
        return 0.0;
    }
    if (std::any_of(vertices_.begin(), vertices_.end(), [](Index vertex) { return vertex < 0; }))
    {
        return 0.0;
    }
    if (!referencedRowsAreFinite(vertices_, coords))
    {
        return 0.0;
    }
    if (n == 1)
    {
        return 0.0;
    }
    if (n == 2)
    {
        const auto &p0 = coords[vertices_[0]];
        const auto &p1 = coords[vertices_[1]];
        return std::sqrt(squaredDistance(p0, p1));
    }
    if (n == 3)
    {
        const auto &p0 = coords[vertices_[0]];
        const auto &p1 = coords[vertices_[1]];
        const auto &p2 = coords[vertices_[2]];
        std::vector<std::vector<double>> cmMatrix(4, std::vector<double>(4, 0.0));
        const std::vector<const std::vector<double> *> points = {&p0, &p1, &p2};
        for (Size i = 0; i < 3; ++i)
        {
            for (Size j = 0; j < 3; ++j)
            {
                cmMatrix[i + 1][j + 1] = squaredDistance(*points[i], *points[j]);
            }
        }
        for (Size i = 1; i < 4; ++i)
        {
            cmMatrix[0][i] = 1.0;
            cmMatrix[i][0] = 1.0;
        }
        const double det = computeDeterminant(cmMatrix);
        const double volume_squared = -det / 16.0;
        return volume_squared > 0.0 ? std::sqrt(volume_squared) : 0.0;
    }
    if (n == 4)
    {
        const auto &p0 = coords[vertices_[0]];
        const auto &p1 = coords[vertices_[1]];
        const auto &p2 = coords[vertices_[2]];
        const auto &p3 = coords[vertices_[3]];
        std::vector<std::vector<double>> cmMatrix(5, std::vector<double>(5, 0.0));
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                const auto &pi = (i == 0) ? p0 : (i == 1) ? p1 : (i == 2) ? p2 : p3;
                const auto &pj = (j == 0) ? p0 : (j == 1) ? p1 : (j == 2) ? p2 : p3;
                cmMatrix[i + 1][j + 1] = squaredDistance(pi, pj);
            }
        }
        cmMatrix[0][0] = 0.0;
        for (int i = 1; i < 5; ++i)
        {
            cmMatrix[0][i] = 1.0;
            cmMatrix[i][0] = 1.0;
        }
        double det = computeDeterminant(cmMatrix);
        double volume_squared = det / 288.0;
        if (volume_squared < 0.0)
            volume_squared = 0.0;
        return std::sqrt(volume_squared);
    }
    std::vector<std::vector<double>> cmMatrix(n + 1, std::vector<double>(n + 1, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            if (i == j)
            {
                cmMatrix[i + 1][j + 1] = 0.0;
            }
            else
            {
                const auto &pi = coords[vertices_[i]];
                const auto &pj = coords[vertices_[j]];
                cmMatrix[i + 1][j + 1] = squaredDistance(pi, pj);
            }
        }
    }
    cmMatrix[0][0] = 0.0;
    for (Size i = 1; i <= n; ++i)
    {
        cmMatrix[0][i] = 1.0;
        cmMatrix[i][0] = 1.0;
    }
    double det = computeDeterminant(cmMatrix);
    const Size simplex_dim = n - 1;
    double sign = (n % 2 == 0) ? 1.0 : -1.0;
    double denominator = std::pow(2.0, static_cast<int>(simplex_dim)) *
                         std::pow(std::tgamma(static_cast<double>(simplex_dim + 1)), 2);
    double volume_squared = sign * det / denominator;
    if (volume_squared < 0.0)
        volume_squared = 0.0;
    return std::sqrt(volume_squared);
}
double Simplex::computeDeterminant(const std::vector<std::vector<double>> &matrix) const
{
    Size n = matrix.size();
    if (n == 0)
        return 0.0;
    if (n != matrix[0].size())
        return 0.0;
    for (const auto &row : matrix)
    {
        if (row.size() != n)
            return 0.0;
    }
    if (n == 1)
        return matrix[0][0];
    if (n == 2)
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    std::vector<std::vector<double>> temp = matrix;
    double det = 1.0;
    for (Size i = 0; i < n; ++i)
    {
        Size pivot = i;
        for (Size j = i + 1; j < n; ++j)
        {
            if (std::abs(temp[j][i]) > std::abs(temp[pivot][i]))
            {
                pivot = j;
            }
        }
        if (pivot != i)
        {
            std::swap(temp[i], temp[pivot]);
            det *= -1.0;
        }
        if (std::abs(temp[i][i]) < VOLUME_TOLERANCE)
        {
            return 0.0;
        }
        det *= temp[i][i];
        for (Size j = i + 1; j < n; ++j)
        {
            double factor = temp[j][i] / temp[i][i];
            for (Size k = i; k < n; ++k)
            {
                temp[j][k] -= factor * temp[i][k];
            }
        }
    }
    return det;
}
Size Simplex::factorial(Size n) const
{
    if (n <= 1)
        return 1;
    Size result = 1;
    for (Size i = 2; i <= n; ++i)
    {
        result *= i;
    }
    return result;
}
} // namespace nerve::algebra
