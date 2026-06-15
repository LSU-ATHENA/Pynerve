#include "nerve/gpu/manager_compute_ops_hungarian.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace nerve::gpu
{
namespace
{

errors::ErrorResult<void> invalidInput(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT, message);
}

errors::ErrorResult<void> numericError(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN, message);
}

} // namespace

errors::ErrorResult<void>
validateSquareCostMatrix(const std::vector<std::vector<double>> &cost_matrix, bool allow_infinity)
{
    const std::size_t n = cost_matrix.size();
    for (const auto &row : cost_matrix)
    {
        if (row.size() != n)
        {
            return invalidInput("cost matrix must be square");
        }
        for (double value : row)
        {
            const bool valid = allow_infinity ? !std::isnan(value) : std::isfinite(value);
            if (!valid)
            {
                return numericError("cost matrix contains invalid numeric values");
            }
        }
    }
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<double>
solveAssignmentHungarian(const std::vector<std::vector<double>> &cost_matrix,
                         std::vector<std::pair<int, int>> &out_assignment)
{
    out_assignment.clear();
    const std::size_t n = cost_matrix.size();
    if (n == 0)
    {
        return errors::ErrorResult<double>::success(0.0);
    }
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return errors::ErrorResult<double>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                  "assignment matrix exceeds int range");
    }

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1, 0.0);
    std::vector<double> v(n + 1, 0.0);
    std::vector<int> p(n + 1, 0);
    std::vector<int> way(n + 1, 0);

    for (std::size_t i = 1; i <= n; ++i)
    {
        p[0] = static_cast<int>(i);
        std::size_t j0 = 0;
        std::vector<double> minv(n + 1, inf);
        std::vector<char> used(n + 1, false);
        do
        {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = inf;
            std::size_t j1 = 0;
            for (std::size_t j = 1; j <= n; ++j)
            {
                if (used[j])
                {
                    continue;
                }
                const double cur = cost_matrix[static_cast<std::size_t>(i0 - 1)][j - 1] -
                                   u[static_cast<std::size_t>(i0)] - v[j];
                if (cur < minv[j])
                {
                    minv[j] = cur;
                    way[j] = static_cast<int>(j0);
                }
                if (minv[j] < delta)
                {
                    delta = minv[j];
                    j1 = j;
                }
            }
            if (!std::isfinite(delta))
            {
                return errors::ErrorResult<double>::error(errors::ErrorCode::E20_NUM_NAN,
                                                          "assignment matrix is not solvable");
            }
            for (std::size_t j = 0; j <= n; ++j)
            {
                if (used[j])
                {
                    u[static_cast<std::size_t>(p[j])] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do
        {
            const std::size_t j1 = static_cast<std::size_t>(way[j0]);
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> row_to_col(n, -1);
    for (std::size_t j = 1; j <= n; ++j)
    {
        if (p[j] > 0)
        {
            row_to_col[static_cast<std::size_t>(p[j] - 1)] = static_cast<int>(j - 1);
        }
    }

    double total = 0.0;
    out_assignment.reserve(n);
    for (std::size_t row = 0; row < n; ++row)
    {
        const int col = row_to_col[row];
        if (col < 0)
        {
            out_assignment.clear();
            return errors::ErrorResult<double>::error(errors::ErrorCode::E21_NUM_NO_CONVERGE,
                                                      "assignment matrix did not fully match");
        }
        out_assignment.emplace_back(static_cast<int>(row), col);
        total += cost_matrix[row][static_cast<std::size_t>(col)];
    }
    return errors::ErrorResult<double>::success(double{total});
}

bool augmentThresholdMatching(std::size_t left, const std::vector<std::vector<double>> &costs,
                              double threshold, std::vector<char> &seen,
                              std::vector<int> &match_right)
{
    for (std::size_t right = 0; right < costs.size(); ++right)
    {
        if (seen[right] || costs[left][right] > threshold)
        {
            continue;
        }
        seen[right] = true;
        if (match_right[right] < 0 ||
            augmentThresholdMatching(static_cast<std::size_t>(match_right[right]), costs, threshold,
                                     seen, match_right))
        {
            match_right[right] = static_cast<int>(left);
            return true;
        }
    }
    return false;
}

bool thresholdPerfectMatching(const std::vector<std::vector<double>> &costs, double threshold,
                              std::vector<std::pair<int, int>> *assignment)
{
    const std::size_t n = costs.size();
    std::vector<int> match_right(n, -1);
    for (std::size_t left = 0; left < n; ++left)
    {
        std::vector<char> seen(n, false);
        if (!augmentThresholdMatching(left, costs, threshold, seen, match_right))
        {
            return false;
        }
    }
    if (assignment != nullptr)
    {
        assignment->clear();
        assignment->reserve(n);
        std::vector<int> row_to_col(n, -1);
        for (std::size_t right = 0; right < n; ++right)
        {
            if (match_right[right] >= 0)
            {
                row_to_col[static_cast<std::size_t>(match_right[right])] = static_cast<int>(right);
            }
        }
        for (std::size_t row = 0; row < n; ++row)
        {
            assignment->emplace_back(static_cast<int>(row), row_to_col[row]);
        }
    }
    return true;
}

} // namespace nerve::gpu
