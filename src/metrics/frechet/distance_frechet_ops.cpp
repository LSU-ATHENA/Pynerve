
#include "nerve/metrics/distances.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
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
    for (const auto &pair : pairs)
    {
        if (!isValidPair(pair))
        {
            throw std::invalid_argument("diagram contains invalid persistence pair values");
        }
    }
}

Size checkedIncrement(Size value, const char *name)
{
    if (value == std::numeric_limits<Size>::max())
    {
        throw std::length_error(std::string(name) + " size overflows");
    }
    return value + 1;
}

void validateMatrixArea(Size rows, Size cols, const char *name)
{
    if (rows != 0 && cols > std::numeric_limits<Size>::max() / rows)
    {
        throw std::length_error(std::string(name) + " matrix area overflows");
    }
}

double checkedAdd(double lhs, double rhs, const char *name)
{
    const double value = lhs + rhs;
    if (!std::isfinite(value))
    {
        throw std::overflow_error(std::string(name) + " cost overflowed");
    }
    return value;
}

} // namespace

double EditDistance::compute(const SimplicialComplex &complex1, const SimplicialComplex &complex2)
{
    const auto start = std::chrono::steady_clock::now();
    std::vector<AlgebraSimplex> simplices1;
    std::vector<AlgebraSimplex> simplices2;

    for (Dimension dim = 0; dim <= complex1.maxDimension(); ++dim)
    {
        const auto per_dim = complex1.simplicesOfDimension(dim);
        simplices1.insert(simplices1.end(), per_dim.begin(), per_dim.end());
    }
    for (Dimension dim = 0; dim <= complex2.maxDimension(); ++dim)
    {
        const auto per_dim = complex2.simplicesOfDimension(dim);
        simplices2.insert(simplices2.end(), per_dim.begin(), per_dim.end());
    }

    const double value = computeEditDistanceDp(simplices1, simplices2);
    computation_time_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return value;
}

void EditDistance::setInsertionCost(double cost)
{
    if (!std::isfinite(cost))
    {
        throw std::invalid_argument("insertion cost must be finite");
    }
    insertion_cost_ = std::max(0.0, cost);
}

void EditDistance::setDeletionCost(double cost)
{
    if (!std::isfinite(cost))
    {
        throw std::invalid_argument("deletion cost must be finite");
    }
    deletion_cost_ = std::max(0.0, cost);
}

void EditDistance::setSubstitutionCost(double cost)
{
    if (!std::isfinite(cost))
    {
        throw std::invalid_argument("substitution cost must be finite");
    }
    substitution_cost_ = std::max(0.0, cost);
}

std::vector<std::string> EditDistance::getEditOperations() const
{
    return edit_operations_;
}

Size EditDistance::getNumOperations() const
{
    return edit_operations_.size();
}

double EditDistance::getComputationTime() const
{
    return computation_time_;
}

double EditDistance::computeEditDistanceDp(const std::vector<AlgebraSimplex> &simplices1,
                                           const std::vector<AlgebraSimplex> &simplices2)
{
    const Size n1 = simplices1.size();
    const Size n2 = simplices2.size();
    const Size rows = checkedIncrement(n1, "edit distance row");
    const Size cols = checkedIncrement(n2, "edit distance column");
    validateMatrixArea(rows, cols, "edit distance");
    std::vector<std::vector<double>> dp(rows, std::vector<double>(cols, 0.0));

    for (Size i = 1; i <= n1; ++i)
    {
        dp[i][0] = checkedAdd(dp[i - 1][0], deletion_cost_, "edit distance deletion");
    }
    for (Size j = 1; j <= n2; ++j)
    {
        dp[0][j] = checkedAdd(dp[0][j - 1], insertion_cost_, "edit distance insertion");
    }

    for (Size i = 1; i <= n1; ++i)
    {
        for (Size j = 1; j <= n2; ++j)
        {
            const double sub_cost = simplexDistance(simplices1[i - 1], simplices2[j - 1]);
            dp[i][j] =
                std::min({checkedAdd(dp[i - 1][j], deletion_cost_, "edit distance deletion"),
                          checkedAdd(dp[i][j - 1], insertion_cost_, "edit distance insertion"),
                          checkedAdd(dp[i - 1][j - 1], sub_cost, "edit distance substitution")});
        }
    }
    return dp[n1][n2];
}

double EditDistance::simplexDistance(const AlgebraSimplex &simplex1,
                                     const AlgebraSimplex &simplex2) const
{
    if (simplex1 == simplex2)
    {
        return 0.0;
    }
    if (simplex1.dimension() != simplex2.dimension())
    {
        return substitution_cost_;
    }

    const auto &vertices1 = simplex1.vertices();
    const auto &vertices2 = simplex2.vertices();
    std::set<Index> set1(vertices1.begin(), vertices1.end());
    std::set<Index> set2(vertices2.begin(), vertices2.end());
    std::vector<Index> intersection;
    std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                          std::back_inserter(intersection));
    const double overlap = set1.empty() && set2.empty()
                               ? 1.0
                               : static_cast<double>(intersection.size()) /
                                     static_cast<double>(std::max(set1.size(), set2.size()));
    const double distance = substitution_cost_ * (1.0 - overlap);
    if (!std::isfinite(distance))
    {
        throw std::overflow_error("edit distance substitution cost overflowed");
    }
    return distance;
}

double FrechetDistance::compute(const Diagram &diagram1, const Diagram &diagram2)
{
    const auto &pairs1 = diagram1.getPairs();
    const auto &pairs2 = diagram2.getPairs();
    validateDiagramPairs(pairs1);
    validateDiagramPairs(pairs2);
    return computeFrechetDistanceDp(pairs1, pairs2);
}

void FrechetDistance::setParameterization(const std::string &param)
{
    parameterization_ = param;
}

void FrechetDistance::setTolerance(double tolerance)
{
    if (!std::isfinite(tolerance))
    {
        throw std::invalid_argument("Frechet tolerance must be finite");
    }
    tolerance_ = std::max(0.0, tolerance);
}

double
FrechetDistance::computeFrechetDistanceDp(const std::vector<nerve::persistence::Pair> &pairs1,
                                          const std::vector<nerve::persistence::Pair> &pairs2)
{
    const Size n1 = pairs1.size();
    const Size n2 = pairs2.size();
    if (n1 == 0 || n2 == 0)
    {
        return std::numeric_limits<double>::infinity();
    }
    validateMatrixArea(n1, n2, "Frechet distance");

    std::vector<std::vector<double>> dp(n1, std::vector<double>(n2, 0.0));
    dp[0][0] = curveDistance(pairs1[0], pairs2[0]);

    for (Size i = 1; i < n1; ++i)
    {
        dp[i][0] = std::max(dp[i - 1][0], curveDistance(pairs1[i], pairs2[0]));
    }
    for (Size j = 1; j < n2; ++j)
    {
        dp[0][j] = std::max(dp[0][j - 1], curveDistance(pairs1[0], pairs2[j]));
    }
    for (Size i = 1; i < n1; ++i)
    {
        for (Size j = 1; j < n2; ++j)
        {
            const double local = curveDistance(pairs1[i], pairs2[j]);
            dp[i][j] = std::max(local, std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}));
        }
    }
    return dp[n1 - 1][n2 - 1];
}

double FrechetDistance::curveDistance(const nerve::persistence::Pair &pair1,
                                      const nerve::persistence::Pair &pair2) const
{
    if (pair1.isInfinite() && pair2.isInfinite())
    {
        const double distance = std::abs(pair1.birth - pair2.birth);
        if (!std::isfinite(distance))
        {
            throw std::overflow_error("Frechet curve distance overflowed");
        }
        return distance;
    }
    if (pair1.isInfinite() != pair2.isInfinite())
    {
        return std::numeric_limits<double>::infinity();
    }
    const double dx = pair1.birth - pair2.birth;
    const double dy = pair1.death - pair2.death;
    const double distance = std::hypot(dx, dy);
    if (!std::isfinite(distance))
    {
        throw std::overflow_error("Frechet curve distance overflowed");
    }
    return distance;
}

} // namespace nerve::metrics
