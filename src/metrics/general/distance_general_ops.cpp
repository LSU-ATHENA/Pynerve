
#include "nerve/metrics/distances.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace nerve::metrics
{

namespace
{

bool isValidPair(const nerve::persistence::Pair &pair)
{
    const bool finite_death = std::isfinite(pair.death);
    return std::isfinite(pair.birth) && (finite_death || pair.isInfinite()) &&
           pair.dimension >= 0 && (!finite_death || pair.death >= pair.birth);
}

void validateDiagramPairs(const std::vector<nerve::persistence::Pair> &pairs)
{
    if (pairs.size() > static_cast<Size>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("diagram is too large for assignment indexing");
    }
    for (const auto &pair : pairs)
    {
        if (!isValidPair(pair))
        {
            throw std::invalid_argument("diagram contains invalid persistence pair values");
        }
    }
}

void validateMatchingSize(Size n)
{
    if (n > static_cast<Size>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("assignment problem is too large for indexing");
    }
    if (n != 0 && n > std::numeric_limits<Size>::max() / n)
    {
        throw std::length_error("assignment cost matrix area overflows");
    }
}

double diagonalCost(const nerve::persistence::Pair &pair)
{
    if (pair.isInfinite())
    {
        return std::numeric_limits<double>::infinity();
    }
    const double cost = std::abs(pair.death - pair.birth) * 0.5;
    if (!std::isfinite(cost))
    {
        throw std::overflow_error("bottleneck diagonal cost overflowed");
    }
    return cost;
}

bool augmentMatching(Size left, const std::vector<std::vector<Size>> &adjacency,
                     std::vector<bool> &seen, std::vector<int> &matchRight)
{
    for (const Size right : adjacency[left])
    {
        if (seen[right])
        {
            continue;
        }
        seen[right] = true;
        if (matchRight[right] < 0 ||
            augmentMatching(static_cast<Size>(matchRight[right]), adjacency, seen, matchRight))
        {
            matchRight[right] = static_cast<int>(left);
            return true;
        }
    }
    return false;
}

bool hasPerfectMatchingWithThreshold(const std::vector<std::vector<double>> &costMatrix,
                                     double threshold)
{
    const Size n = costMatrix.size();
    if (n == 0)
    {
        return true;
    }

    std::vector<std::vector<Size>> adjacency(n);
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            if (costMatrix[i][j] <= threshold)
            {
                adjacency[i].push_back(j);
            }
        }
    }

    std::vector<int> matchRight(n, -1);
    Size matched = 0;
    for (Size left = 0; left < n; ++left)
    {
        std::vector<bool> seen(n, false);
        if (augmentMatching(left, adjacency, seen, matchRight))
        {
            ++matched;
        }
    }
    return matched == n;
}

} // namespace

double BottleneckDistance::compute(const Diagram &diagram1, const Diagram &diagram2)
{
    const auto start = std::chrono::steady_clock::now();
    const auto &pairs1 = diagram1.getPairs();
    const auto &pairs2 = diagram2.getPairs();
    validateDiagramPairs(pairs1);
    validateDiagramPairs(pairs2);
    const auto costMatrix = buildCostMatrix(pairs1, pairs2);

    double value = 0.0;
    if (use_approximation_)
    {
        value = solveAssignmentGreedy(costMatrix);
    }
    else
    {
        value = solveAssignmentHungarian(costMatrix);
    }

    optimal_matching_.clear();
    const Size n = std::min(pairs1.size(), pairs2.size());
    optimal_matching_.reserve(n);
    for (Size i = 0; i < n; ++i)
    {
        optimal_matching_.emplace_back(pairs1[i], pairs2[i]);
    }

    computation_time_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return value;
}

void BottleneckDistance::setTolerance(double tolerance)
{
    if (!std::isfinite(tolerance))
    {
        throw std::invalid_argument("Bottleneck tolerance must be finite");
    }
    tolerance_ = std::max(0.0, tolerance);
}

void BottleneckDistance::setMaxIterations(Size max_iterations)
{
    max_iterations_ = max_iterations;
}

void BottleneckDistance::useApproximation(bool use_approx)
{
    use_approximation_ = use_approx;
}

std::vector<std::pair<nerve::persistence::Pair, nerve::persistence::Pair>>
BottleneckDistance::getOptimalMatching() const
{
    return optimal_matching_;
}

double BottleneckDistance::getComputationTime() const
{
    return computation_time_;
}

std::vector<std::vector<double>>
BottleneckDistance::buildCostMatrix(const std::vector<nerve::persistence::Pair> &pairs1,
                                    const std::vector<nerve::persistence::Pair> &pairs2) const
{
    const Size n1 = pairs1.size();
    const Size n2 = pairs2.size();
    const Size n = std::max(n1, n2);
    validateMatchingSize(n);
    std::vector<std::vector<double>> costMatrix(n, std::vector<double>(n, 0.0));

    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            if (i < n1 && j < n2)
            {
                costMatrix[i][j] = pairDistance(pairs1[i], pairs2[j]);
            }
            else if (i < n1)
            {
                costMatrix[i][j] = diagonalCost(pairs1[i]);
            }
            else if (j < n2)
            {
                costMatrix[i][j] = diagonalCost(pairs2[j]);
            }
        }
    }

    return costMatrix;
}

double
BottleneckDistance::solveAssignmentHungarian(const std::vector<std::vector<double>> &costMatrix)
{
    if (costMatrix.empty())
    {
        return 0.0;
    }
    validateMatchingSize(costMatrix.size());

    std::vector<double> candidates;
    candidates.reserve(costMatrix.size() * costMatrix.size());
    for (const auto &row : costMatrix)
    {
        for (double value : row)
        {
            if (std::isfinite(value))
            {
                candidates.push_back(value);
            }
        }
    }
    if (candidates.empty())
    {
        return std::numeric_limits<double>::infinity();
    }
    // Tolerance for distance metric uniqueness
    constexpr double DISTANCE_UNIQUENESS_TOLERANCE = 1e-12;

    std::ranges::sort(candidates);
    // Manual unique with tolerance instead of std::ranges::unique to avoid template constraints
    auto last = std::unique(candidates.begin(), candidates.end(), [&](double lhs, double rhs) {
        return std::abs(lhs - rhs) <= DISTANCE_UNIQUENESS_TOLERANCE;
    });
    candidates.erase(last, candidates.end());

    Size lo = 0;
    Size hi = candidates.size() - 1;
    double best = candidates[hi];
    while (lo <= hi)
    {
        const Size mid = lo + (hi - lo) / 2;
        const double threshold = candidates[mid] + tolerance_;
        if (hasPerfectMatchingWithThreshold(costMatrix, threshold))
        {
            best = candidates[mid];
            if (mid == 0)
            {
                break;
            }
            hi = mid - 1;
        }
        else
        {
            lo = mid + 1;
        }
    }
    return best;
}

double BottleneckDistance::solveAssignmentGreedy(const std::vector<std::vector<double>> &costMatrix)
{
    const Size n = costMatrix.size();
    std::vector<bool> rowUsed(n, false);
    std::vector<bool> colUsed(n, false);
    double max_cost = 0.0;

    for (Size match_count = 0; match_count < n; ++match_count)
    {
        double best_cost = std::numeric_limits<double>::infinity();
        Size best_i = 0;
        Size best_j = 0;
        for (Size i = 0; i < n; ++i)
        {
            if (rowUsed[i])
            {
                continue;
            }
            for (Size j = 0; j < n; ++j)
            {
                if (colUsed[j] || !std::isfinite(costMatrix[i][j]))
                {
                    continue;
                }
                if (costMatrix[i][j] < best_cost)
                {
                    best_cost = costMatrix[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if (!std::isfinite(best_cost))
        {
            return std::numeric_limits<double>::infinity();
        }
        rowUsed[best_i] = true;
        colUsed[best_j] = true;
        max_cost = std::max(max_cost, best_cost);
    }

    return max_cost;
}

double BottleneckDistance::pairDistance(const nerve::persistence::Pair &pair1,
                                        const nerve::persistence::Pair &pair2) const
{
    if (pair1.isInfinite() && pair2.isInfinite())
    {
        const double distance = std::abs(pair1.birth - pair2.birth);
        if (!std::isfinite(distance))
        {
            throw std::overflow_error("bottleneck pair distance overflowed");
        }
        return distance;
    }
    if (pair1.isInfinite() != pair2.isInfinite())
    {
        return std::numeric_limits<double>::infinity();
    }
    const double birth_diff = std::abs(pair1.birth - pair2.birth);
    const double death_diff = std::abs(pair1.death - pair2.death);
    if (!std::isfinite(birth_diff) || !std::isfinite(death_diff))
    {
        throw std::overflow_error("bottleneck pair distance overflowed");
    }
    return std::max(birth_diff, death_diff);
}

} // namespace nerve::metrics
