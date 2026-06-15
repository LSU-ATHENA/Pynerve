
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nerve::spectral::detail
{

struct SymmetricEigenResult
{
    std::vector<double> eigenvalues;
    std::vector<std::vector<double>> eigenvectors; // row-major: eigenvectors[i][j]
};

inline SymmetricEigenResult jacobiEigendecomposition(const std::vector<std::vector<double>> &matrix,
                                                     std::size_t max_iterations = 128,
                                                     double tolerance = 1e-12)
{
    const std::size_t n = matrix.size();
    for (const auto &row : matrix)
    {
        if (row.size() != n)
        {
            throw std::invalid_argument("eigendecomposition requires a square matrix");
        }
    }

    if (n == 0)
    {
        return {};
    }

    std::vector<std::vector<double>> a = matrix;
    std::vector<std::vector<double>> v(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i)
    {
        v[i][i] = 1.0;
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = i + 1; j < n; ++j)
        {
            const double sym = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = sym;
            a[j][i] = sym;
        }
    }

    const double eps = std::max(tolerance, std::numeric_limits<double>::epsilon());
    for (std::size_t iter = 0; iter < max_iterations; ++iter)
    {
        std::size_t p = 0;
        std::size_t q = 1;
        double max_offdiag = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = i + 1; j < n; ++j)
            {
                const double val = std::fabs(a[i][j]);
                if (val > max_offdiag)
                {
                    max_offdiag = val;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_offdiag <= eps)
        {
            break;
        }

        const double app = a[p][p];
        const double aqq = a[q][q];
        const double apq = a[p][q];
        const double theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(theta);
        const double s = std::sin(theta);

        for (std::size_t k = 0; k < n; ++k)
        {
            if (k == p || k == q)
            {
                continue;
            }
            const double akp = a[k][p];
            const double akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[p][k] = a[k][p];
            a[k][q] = s * akp + c * akq;
            a[q][k] = a[k][q];
        }

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for (std::size_t k = 0; k < n; ++k)
        {
            const double vkp = v[k][p];
            const double vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }

    std::vector<std::pair<double, std::size_t>> order;
    order.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        order.emplace_back(a[i][i], i);
    }
    std::sort(order.begin(), order.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.first != rhs.first)
        {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    });

    SymmetricEigenResult result;
    result.eigenvalues.resize(n, 0.0);
    result.eigenvectors.assign(n, std::vector<double>(n, 0.0));
    for (std::size_t row = 0; row < n; ++row)
    {
        const std::size_t col = order[row].second;
        result.eigenvalues[row] = order[row].first;
        for (std::size_t comp = 0; comp < n; ++comp)
        {
            result.eigenvectors[row][comp] = v[comp][col];
        }
    }

    return result;
}

} // namespace nerve::spectral::detail
