#include "nerve/gpu/manager_compute_ops_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace nerve::gpu
{
namespace
{

constexpr double kGeometryTolerance = 1e-12;

} // namespace

double euclideanDistance(const std::vector<double> &lhs, const std::vector<double> &rhs)
{
    double distance_sq = 0.0;
    for (std::size_t dim = 0; dim < lhs.size(); ++dim)
    {
        const double diff = lhs[dim] - rhs[dim];
        const double contribution = diff * diff;
        const double next_distance_sq = distance_sq + contribution;
        if (!std::isfinite(diff) || !std::isfinite(contribution) ||
            !std::isfinite(next_distance_sq))
        {
            return std::numeric_limits<double>::infinity();
        }
        distance_sq = next_distance_sq;
    }
    return std::sqrt(distance_sq);
}

double triangleCircumradius(double a, double b, double c)
{
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c))
    {
        return std::numeric_limits<double>::infinity();
    }
    const double max_side = std::max({a, b, c});
    if (max_side == 0.0)
    {
        return 0.0;
    }

    const double s = (a + b + c) * 0.5;
    const double area_product = s * (s - a) * (s - b) * (s - c);
    if (std::isfinite(area_product))
    {
        const double area_sq = std::max(0.0, area_product);
        if (area_sq <= kGeometryTolerance)
        {
            return max_side * 0.5;
        }
        const double radius = (a * b * c) / (4.0 * std::sqrt(area_sq));
        if (std::isfinite(radius))
        {
            return radius;
        }
    }

    const double scaled_a = a / max_side;
    const double scaled_b = b / max_side;
    const double scaled_c = c / max_side;
    const double scaled_s = (scaled_a + scaled_b + scaled_c) * 0.5;
    const double scaled_area_product =
        scaled_s * (scaled_s - scaled_a) * (scaled_s - scaled_b) * (scaled_s - scaled_c);
    if (!std::isfinite(scaled_area_product))
    {
        return std::numeric_limits<double>::infinity();
    }
    const double scaled_area_sq = std::max(0.0, scaled_area_product);
    if (scaled_area_sq <= kGeometryTolerance)
    {
        return max_side * 0.5;
    }
    const double scaled_radius =
        (scaled_a * scaled_b * scaled_c) / (4.0 * std::sqrt(scaled_area_sq));
    const double radius = max_side * scaled_radius;
    return std::isfinite(radius) ? radius : std::numeric_limits<double>::infinity();
}

double enclosingRadiusEstimate(const std::vector<std::vector<double>> &points,
                               const std::vector<int> &vertices)
{
    if (vertices.size() <= 1)
    {
        return 0.0;
    }

    double radius = 0.0;
    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
        for (std::size_t j = i + 1; j < vertices.size(); ++j)
        {
            const double distance =
                euclideanDistance(points[static_cast<std::size_t>(vertices[i])],
                                  points[static_cast<std::size_t>(vertices[j])]);
            radius = std::max(radius, distance * 0.5);
        }
    }

    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
        for (std::size_t j = i + 1; j < vertices.size(); ++j)
        {
            for (std::size_t k = j + 1; k < vertices.size(); ++k)
            {
                const double a = euclideanDistance(points[static_cast<std::size_t>(vertices[j])],
                                                   points[static_cast<std::size_t>(vertices[k])]);
                const double b = euclideanDistance(points[static_cast<std::size_t>(vertices[i])],
                                                   points[static_cast<std::size_t>(vertices[k])]);
                const double c = euclideanDistance(points[static_cast<std::size_t>(vertices[i])],
                                                   points[static_cast<std::size_t>(vertices[j])]);
                radius = std::max(radius, triangleCircumradius(a, b, c));
            }
        }
    }

    return radius;
}

} // namespace nerve::gpu
