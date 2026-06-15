// Distance transport detail operations  --  Gromov-Hausdorff and Interleaving distance
// implementations.

#include "nerve/metrics/distances.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::metrics
{

namespace
{

void validateFiltrationValues(const std::vector<std::pair<AlgebraSimplex, double>> &filtration)
{
    for (const auto &[simplex, value] : filtration)
    {
        (void)simplex;
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("filtration values must be finite");
        }
    }
}

double checkedAbsoluteDifference(double lhs, double rhs)
{
    const double diff = lhs - rhs;
    if (!std::isfinite(diff))
    {
        throw std::overflow_error("interleaving filtration value overflowed");
    }
    return std::abs(diff);
}

} // namespace

std::vector<std::vector<double>>
GromovHausdorffDistance::embedComplex(const SimplicialComplex &complex) const
{
    std::vector<std::vector<double>> points;
    points.reserve(complex.size());

    const auto simplices = complex.getSimplices();
    for (const auto &simplex : simplices)
    {
        const auto &vertices = simplex.vertices();
        std::vector<double> point(embedding_dimension_, 0.0);
        for (Size i = 0; i < std::min(embedding_dimension_, vertices.size()); ++i)
        {
            point[i] = static_cast<double>(vertices[i]);
        }
        points.push_back(std::move(point));
    }
    return points;
}

double GromovHausdorffDistance::computeHausdorffDistance(
    const std::vector<std::vector<double>> &points1,
    const std::vector<std::vector<double>> &points2) const
{
    return hausdorffDistance(points1, points2);
}

double
InterleavingDistance::compute(const std::vector<std::pair<AlgebraSimplex, double>> &filtration1,
                              const std::vector<std::pair<AlgebraSimplex, double>> &filtration2)
{
    validateFiltrationValues(filtration1);
    validateFiltrationValues(filtration2);
    return use_approximate_algorithm_ ? computeInterleavingDistanceApprox(filtration1, filtration2)
                                      : computeInterleavingDistanceExact(filtration1, filtration2);
}

void InterleavingDistance::setMaxDimension(Size max_dim)
{
    max_dimension_ = max_dim;
}

void InterleavingDistance::setTolerance(double tolerance)
{
    if (!std::isfinite(tolerance))
    {
        throw std::invalid_argument("Interleaving tolerance must be finite");
    }
    tolerance_ = std::max(0.0, tolerance);
}

void InterleavingDistance::useApproximateAlgorithm(bool use_approx)
{
    use_approximate_algorithm_ = use_approx;
}

double InterleavingDistance::computeInterleavingDistanceExact(
    const std::vector<std::pair<AlgebraSimplex, double>> &filtration1,
    const std::vector<std::pair<AlgebraSimplex, double>> &filtration2)
{
    std::unordered_map<std::string, double> values1;
    std::unordered_map<std::string, double> values2;
    for (const auto &[simplex, value] : filtration1)
    {
        if (static_cast<Size>(simplex.dimension()) <= max_dimension_)
        {
            values1.emplace(simplex.toString(), value);
        }
    }
    for (const auto &[simplex, value] : filtration2)
    {
        if (static_cast<Size>(simplex.dimension()) <= max_dimension_)
        {
            values2.emplace(simplex.toString(), value);
        }
    }

    double max_diff = 0.0;
    for (const auto &[key, value1] : values1)
    {
        auto iter = values2.find(key);
        if (iter != values2.end())
        {
            max_diff = std::max(max_diff, checkedAbsoluteDifference(value1, iter->second));
        }
        else
        {
            max_diff = std::max(max_diff, std::abs(value1));
        }
    }
    for (const auto &[key, value2] : values2)
    {
        if (!values1.contains(key))
        {
            max_diff = std::max(max_diff, std::abs(value2));
        }
    }
    return max_diff;
}

double InterleavingDistance::computeInterleavingDistanceApprox(
    const std::vector<std::pair<AlgebraSimplex, double>> &filtration1,
    const std::vector<std::pair<AlgebraSimplex, double>> &filtration2)
{
    return computeInterleavingDistanceExact(filtration1, filtration2);
}

} // namespace nerve::metrics
