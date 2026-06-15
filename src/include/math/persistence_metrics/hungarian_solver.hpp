
#pragma once

#include "nerve/error/error_registry.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace nerve::math
{

class HungarianSolver
{
public:
    struct Assignment
    {
        std::vector<std::pair<size_t, size_t>> pairs;
        double total_cost = 0.0;
        size_t num_assigned = 0;
    };

    explicit HungarianSolver(const std::vector<std::vector<double>> &cost_matrix)
        : original_cost_(cost_matrix)
        , row_count_(cost_matrix.size())
    {
        col_count_ = row_count_ == 0 ? 0 : cost_matrix.front().size();
    }

    error::Result<Assignment> solve() const
    {
        try
        {
            auto validate_result = validateInput();
            if (validate_result.isErr())
            {
                return error::Result<Assignment>::err(
                    static_cast<error::TDAErrorCode>(validate_result.error().value()),
                    std::string(validate_result.detail()));
            }

            Assignment assignment;
            if (row_count_ == 0 || col_count_ == 0)
            {
                return error::Result<Assignment>::ok(std::move(assignment));
            }

            const size_t n = std::max(row_count_, col_count_);
            const double pad_cost = computePaddingCost();
            const auto square_cost = makeSquareCostMatrix(n, pad_cost);

            std::vector<double> u(n + 1, 0.0);
            std::vector<double> v(n + 1, 0.0);
            std::vector<size_t> p(n + 1, 0);
            std::vector<size_t> way(n + 1, 0);

            for (size_t i = 1; i <= n; ++i)
            {
                p[0] = i;
                size_t j0 = 0;
                std::vector<double> minv(n + 1, std::numeric_limits<double>::infinity());
                std::vector<bool> used(n + 1, false);

                do
                {
                    used[j0] = true;
                    const size_t i0 = p[j0];
                    double delta = std::numeric_limits<double>::infinity();
                    size_t j1 = 0;

                    for (size_t j = 1; j <= n; ++j)
                    {
                        if (used[j])
                        {
                            continue;
                        }
                        const double cur = square_cost[i0 - 1][j - 1] - u[i0] - v[j];
                        if (cur < minv[j])
                        {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if (minv[j] < delta)
                        {
                            delta = minv[j];
                            j1 = j;
                        }
                    }

                    for (size_t j = 0; j <= n; ++j)
                    {
                        if (used[j])
                        {
                            u[p[j]] += delta;
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
                    const size_t j1 = way[j0];
                    p[j0] = p[j1];
                    j0 = j1;
                } while (j0 != 0);
            }

            std::vector<size_t> colForRow(n, n);
            for (size_t j = 1; j <= n; ++j)
            {
                if (p[j] != 0)
                {
                    colForRow[p[j] - 1] = j - 1;
                }
            }

            for (size_t row = 0; row < row_count_; ++row)
            {
                const size_t col = colForRow[row];
                if (col >= col_count_)
                {
                    continue;
                }
                const double c = original_cost_[row][col];
                if (!std::isfinite(c))
                {
                    continue;
                }
                assignment.pairs.emplace_back(row, col);
                assignment.total_cost += c;
                assignment.num_assigned++;
            }

            return error::Result<Assignment>::ok(std::move(assignment));
        }
        catch (const std::exception &ex)
        {
            return error::Result<Assignment>::err(error::TDAErrorCode::InvalidFieldOperation,
                                                  std::string("Hungarian solver failed: ") +
                                                      ex.what());
        }
    }

private:
    error::Result<void> validateInput() const
    {
        if (row_count_ == 0)
        {
            return error::Result<void>::ok();
        }
        if (col_count_ == 0)
        {
            return error::Result<void>::err(error::TDAErrorCode::InvalidDimension,
                                            "cost matrix has rows but no columns");
        }
        for (size_t row = 0; row < row_count_; ++row)
        {
            if (original_cost_[row].size() != col_count_)
            {
                return error::Result<void>::err(error::TDAErrorCode::InvalidDimension,
                                                "cost matrix rows have inconsistent lengths");
            }
        }
        return error::Result<void>::ok();
    }

    double computePaddingCost() const
    {
        double max_finite = 0.0;
        for (const auto &row : original_cost_)
        {
            for (double value : row)
            {
                if (std::isfinite(value))
                {
                    max_finite = std::max(max_finite, value);
                }
            }
        }
        const double base = max_finite > 0.0 ? max_finite : 1.0;
        return base * (static_cast<double>(std::max(row_count_, col_count_)) + 1.0) + 1.0;
    }

    std::vector<std::vector<double>> makeSquareCostMatrix(size_t n, double pad_cost) const
    {
        std::vector<std::vector<double>> square(n, std::vector<double>(n, pad_cost));
        for (size_t row = 0; row < row_count_; ++row)
        {
            for (size_t col = 0; col < col_count_; ++col)
            {
                const double value = original_cost_[row][col];
                square[row][col] = std::isfinite(value) ? value : pad_cost;
            }
        }
        return square;
    }

    std::vector<std::vector<double>> original_cost_;
    size_t row_count_ = 0;
    size_t col_count_ = 0;
};

} // namespace nerve::math
