
#pragma once

#include <algorithm>
#include <cmath>

namespace nerve::math
{

struct Point2D
{
    double x = 0.0;
    double y = 0.0;

    Point2D() = default;
    Point2D(double x_val, double y_val)
        : x(x_val)
        , y(y_val)
    {}

    double distanceTo(const Point2D &other) const
    {
        const double dx = std::abs(x - other.x);
        const double dy = std::abs(y - other.y);
        return std::max(dx, dy);
    }

    double euclideanDistanceTo(const Point2D &other) const
    {
        const double dx = x - other.x;
        const double dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    double diagonalDistance() const { return std::abs(y - x) * 0.5; }
};

inline double linfDistance(const Point2D &lhs, const Point2D &rhs)
{
    return lhs.distanceTo(rhs);
}

inline double diagonalDistance(const Point2D &point)
{
    return point.diagonalDistance();
}

} // namespace nerve::math
