#include "nerve/gpu/kernel_launcher.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>

namespace nerve::gpu
{
namespace
{

[[nodiscard]] bool validPersistencePair(const persistence::Pair &pair) noexcept
{
    if (pair.dimension < 0)
    {
        return false;
    }
    if (!std::isfinite(pair.birth))
    {
        return false;
    }
    if (pair.isInfinite())
    {
        return true;
    }
    return std::isfinite(pair.death) && pair.death >= pair.birth;
}

[[nodiscard]] bool validPersistenceDiagram(const persistence::Diagram &diagram) noexcept
{
    const auto &pairs = diagram.pairs();
    return std::all_of(pairs.begin(), pairs.end(), validPersistencePair);
}

} // namespace

errors::ErrorResult<void>
ComputeManager::computeDiagramCostMatrix(const persistence::Diagram &diagram1,
                                         const persistence::Diagram &diagram2,
                                         std::vector<std::vector<double>> &out_cost_matrix)
{
    constexpr const char *operation = "computeDiagramCostMatrix";

    const auto &pairs1 = diagram1.pairs();
    const auto &pairs2 = diagram2.pairs();
    size_t n1 = pairs1.size();
    size_t n2 = pairs2.size();
    if (!validPersistenceDiagram(diagram1) || !validPersistenceDiagram(diagram2))
    {
        out_cost_matrix.clear();
        recordFailure(operation, "Invalid persistence diagram pair");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT,
                                                "invalid persistence diagram pair");
    }
    if (n2 > std::numeric_limits<std::size_t>::max() - n1)
    {
        out_cost_matrix.clear();
        recordFailure(operation, "Diagram cost matrix size overflow");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "diagram cost matrix size overflow");
    }
    size_t n = n1 + n2;
    if (n != 0 && n > std::numeric_limits<std::size_t>::max() / n)
    {
        out_cost_matrix.clear();
        recordFailure(operation, "Diagram cost matrix area overflow");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "diagram cost matrix area overflow");
    }

    (void)selectStrategy(OperationType::kDiagramDistance, n);

    try
    {
        out_cost_matrix.assign(n, std::vector<double>(n, std::numeric_limits<double>::infinity()));
    }
    catch (const std::bad_alloc &)
    {
        out_cost_matrix.clear();
        recordFailure(operation, "Diagram cost matrix allocation failed");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM,
                                                "diagram cost matrix allocation failed");
    }

    // Compute cost matrix elements
    auto computePairDistance = [](const persistence::Pair &p1,
                                  const persistence::Pair &p2) -> double {
        if (p1.isInfinite() && p2.isInfinite())
        {
            const double distance = std::abs(p1.birth - p2.birth);
            return std::isfinite(distance) ? distance : std::numeric_limits<double>::quiet_NaN();
        }
        if (p1.isInfinite() != p2.isInfinite())
        {
            return std::numeric_limits<double>::infinity();
        }
        const double birth_diff = std::abs(p1.birth - p2.birth);
        const double death_diff = std::abs(p1.death - p2.death);
        if (!std::isfinite(birth_diff) || !std::isfinite(death_diff))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::max(birth_diff, death_diff);
    };

    auto computeDiagonalCost = [](const persistence::Pair &p) -> double {
        if (p.isInfinite())
        {
            return std::numeric_limits<double>::infinity();
        }
        const double cost = std::abs(p.death - p.birth) * 0.5;
        return std::isfinite(cost) ? cost : std::numeric_limits<double>::quiet_NaN();
    };

    double max_cost = 1.0;
    bool overflowed_cost = false;
    auto assignCost = [&](size_t row, size_t col, double cost) {
        if (std::isnan(cost))
        {
            overflowed_cost = true;
            return;
        }
        out_cost_matrix[row][col] = cost;
        if (std::isfinite(cost))
        {
            max_cost = std::max(max_cost, cost);
        }
    };

    // Fill (i < n1, j < n2) block - pair-to-pair distances
    for (size_t i = 0; i < n1; ++i)
    {
        for (size_t j = 0; j < n2; ++j)
        {
            assignCost(i, j, computePairDistance(pairs1[i], pairs2[j]));
        }
    }

    // Fill diagonal connections for pairs1
    for (size_t i = 0; i < n1; ++i)
    {
        assignCost(i, n2 + i, computeDiagonalCost(pairs1[i]));
    }

    // Fill diagonal connections for pairs2
    for (size_t j = 0; j < n2; ++j)
    {
        assignCost(n1 + j, j, computeDiagonalCost(pairs2[j]));
    }

    if (overflowed_cost)
    {
        out_cost_matrix.clear();
        recordFailure(operation, "Diagram cost matrix value overflow");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E13_PH_HIGHDIM_PRECISION,
                                                "diagram cost matrix value overflow");
    }

    for (size_t row = n1; row < n; ++row)
    {
        for (size_t col = n2; col < n; ++col)
        {
            out_cost_matrix[row][col] = 0.0;
        }
    }

    const double penalty = max_cost * (static_cast<double>(n) + 1.0) + 1.0;
    if (!std::isfinite(penalty))
    {
        out_cost_matrix.clear();
        recordFailure(operation, "Diagram cost matrix penalty overflow");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E13_PH_HIGHDIM_PRECISION,
                                                "diagram cost matrix penalty overflow");
    }

    for (auto &row : out_cost_matrix)
    {
        for (auto &cost : row)
        {
            if (!std::isfinite(cost))
            {
                cost = penalty;
            }
        }
    }

    // Normalize costs
    for (auto &row : out_cost_matrix)
    {
        for (auto &cost : row)
        {
            cost /= max_cost;
        }
    }

    recordSuccess(operation, 1.0);
    return errors::ErrorResult<void>::success();
}

} // namespace nerve::gpu
