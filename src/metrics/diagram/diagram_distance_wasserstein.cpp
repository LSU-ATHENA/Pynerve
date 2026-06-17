#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#if defined(NERVE_HAS_CUDA)
#include "nerve/gpu/bottleneck_distance.cuh"
#include "nerve/gpu/kernel_launcher.hpp"
#include "nerve/gpu/wasserstein_distance.cuh"
#include "nerve/math/persistence_metrics/point2d.hpp"

#include <future>
#endif

namespace nerve::metrics
{

struct DiagramPoint
{
    double birth = 0.0;
    double death = 0.0;
};

inline double linfDistance(const DiagramPoint &lhs, const DiagramPoint &rhs)
{
    return std::max(std::abs(lhs.birth - rhs.birth), std::abs(lhs.death - rhs.death));
}

inline double diagonalDistance(const DiagramPoint &point)
{
    return std::abs(point.death - point.birth) * 0.5;
}

inline bool hasSupportedMatchingSize(size_t n1, size_t n2)
{
    const size_t max_index = static_cast<size_t>(std::numeric_limits<int>::max());
    return n1 <= max_index && n2 <= max_index && n1 <= max_index - n2;
}

double hungarianTotalCost(const std::vector<std::vector<double>> &cost);

double computeWasserstein(const std::vector<DiagramPoint> &points1,
                          const std::vector<DiagramPoint> &points2, double p)
{
    if (points1.empty() && points2.empty())
    {
        return 0.0;
    }

    const size_t n1 = points1.size();
    const size_t n2 = points2.size();
    if (n1 > std::numeric_limits<size_t>::max() - n2)
    {
        return std::numeric_limits<double>::infinity();
    }
    const size_t n = n1 + n2;
    if (n != 0 && n > std::numeric_limits<size_t>::max() / n)
    {
        return std::numeric_limits<double>::infinity();
    }
    std::vector<std::vector<double>> cost(
        n, std::vector<double>(n, std::numeric_limits<double>::infinity()));

    bool enable_gpu = false;
#if defined(NERVE_HAS_CUDA)
    if (nerve::gpu::ComputeManager::getInstance().isAvailable() && n > 500)
    {
        std::vector<nerve::persistence::Pair> pairs1, pairs2;
        pairs1.reserve(n1);
        pairs2.reserve(n2);

        for (const auto &pt : points1)
        {
            nerve::persistence::Pair pair;
            pair.birth = pt.birth;
            pair.death = pt.death;
            pair.dimension = 0;
            pairs1.push_back(pair);
        }

        for (const auto &pt : points2)
        {
            nerve::persistence::Pair pair;
            pair.birth = pt.birth;
            pair.death = pt.death;
            pair.dimension = 0;
            pairs2.push_back(pair);
        }

        nerve::persistence::Diagram d1;
        for (const auto &p : pairs1)
            d1.addPair(p);
        nerve::persistence::Diagram d2;
        for (const auto &p : pairs2)
            d2.addPair(p);

        auto result =
            nerve::gpu::ComputeManager::getInstance().computeDiagramCostMatrix(d1, d2, cost);
        enable_gpu = result.isSuccess();
    }
#endif

    if (!enable_gpu)
    {
        double max_cost = 1.0;
        bool overflowed_cost = false;
        auto assignCost = [&](size_t row, size_t col, double value) {
            if (!std::isfinite(value))
            {
                overflowed_cost = true;
                return;
            }
            cost[row][col] = value;
            max_cost = std::max(max_cost, value);
        };

        for (size_t i = 0; i < n1; ++i)
        {
            for (size_t j = 0; j < n2; ++j)
            {
                assignCost(i, j, std::pow(linfDistance(points1[i], points2[j]), p));
            }
        }
        for (size_t i = 0; i < n1; ++i)
        {
            assignCost(i, n2 + i, std::pow(diagonalDistance(points1[i]), p));
        }
        for (size_t j = 0; j < n2; ++j)
        {
            assignCost(n1 + j, j, std::pow(diagonalDistance(points2[j]), p));
        }
        for (size_t row = n1; row < n; ++row)
        {
            for (size_t col = n2; col < n; ++col)
            {
                cost[row][col] = 0.0;
            }
        }

        if (overflowed_cost)
        {
            return std::numeric_limits<double>::infinity();
        }
        const double large_penalty = max_cost * (static_cast<double>(n) + 1.0) + 1.0;
        if (!std::isfinite(large_penalty))
        {
            return std::numeric_limits<double>::infinity();
        }
        for (auto &row : cost)
        {
            for (double &value : row)
            {
                if (!std::isfinite(value))
                {
                    value = large_penalty;
                }
            }
        }
    }

    for (const auto &row : cost)
    {
        for (double value : row)
        {
            if (!std::isfinite(value))
            {
                return std::numeric_limits<double>::infinity();
            }
        }
    }

    const double total = hungarianTotalCost(cost);
    if (!std::isfinite(total))
    {
        return std::numeric_limits<double>::infinity();
    }
    const double distance = std::pow(total, 1.0 / p);
    return std::isfinite(distance) ? distance : std::numeric_limits<double>::infinity();
}

} // namespace nerve::metrics
