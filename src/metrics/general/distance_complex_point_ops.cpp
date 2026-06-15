
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/metrics/distances.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace nerve::metrics
{

namespace
{

double euclideanDistance(const std::vector<double> &lhs, const std::vector<double> &rhs)
{
    if (lhs.size() != rhs.size())
    {
        return std::numeric_limits<double>::infinity();
    }
    double sum = 0.0;
    for (Size i = 0; i < lhs.size(); ++i)
    {
        if (!std::isfinite(lhs[i]) || !std::isfinite(rhs[i]))
        {
            return std::numeric_limits<double>::infinity();
        }
        const double diff = lhs[i] - rhs[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double pointSetHausdorff(const std::vector<std::vector<double>> &points1,
                         const std::vector<std::vector<double>> &points2)
{
    if (points1.empty() || points2.empty())
    {
        return std::numeric_limits<double>::infinity();
    }

    double max_dist = 0.0;
    for (const auto &point1 : points1)
    {
        double min_dist = std::numeric_limits<double>::infinity();
        for (const auto &point2 : points2)
        {
            min_dist = std::min(min_dist, euclideanDistance(point1, point2));
        }
        max_dist = std::max(max_dist, min_dist);
    }

    for (const auto &point2 : points2)
    {
        double min_dist = std::numeric_limits<double>::infinity();
        for (const auto &point1 : points1)
        {
            min_dist = std::min(min_dist, euclideanDistance(point2, point1));
        }
        max_dist = std::max(max_dist, min_dist);
    }

    return max_dist;
}

std::vector<std::vector<double>> complexToPoints(const SimplicialComplex &complex)
{
    std::vector<std::vector<double>> points;
    for (Dimension dim = 0; dim <= complex.maxDimension(); ++dim)
    {
        const auto simplices = complex.simplicesOfDimension(dim);
        for (const auto &simplex : simplices)
        {
            const auto vertices = simplex.vertices();
            std::vector<double> point(3, 0.0);
            for (Size i = 0; i < std::min(static_cast<Size>(vertices.size()), static_cast<Size>(3));
                 ++i)
            {
                point[i] = static_cast<double>(vertices[i]);
            }
            points.push_back(std::move(point));
        }
    }
    return points;
}

std::vector<std::vector<double>> pointBufferToPoints(const core::PointBuffer &points)
{
    std::vector<std::vector<double>> converted;
    converted.reserve(points.size());
    for (size_t index = 0; index < points.size(); ++index)
    {
        const auto data = points.getPoint(index);
        std::vector<double> point(points.dimension(), 0.0);
        std::copy(data, data + points.dimension(), point.begin());
        converted.push_back(std::move(point));
    }
    return converted;
}

} // namespace

double gromovHausdorffDistance(const SimplicialComplex &complex1, const SimplicialComplex &complex2)
{
    return pointSetHausdorff(complexToPoints(complex1), complexToPoints(complex2));
}

double hausdorffDistance(const SimplicialComplex &complex1, const SimplicialComplex &complex2)
{
    return gromovHausdorffDistance(complex1, complex2);
}

double editDistance(const SimplicialComplex &complex1, const SimplicialComplex &complex2)
{
    Size count1 = 0;
    Size count2 = 0;
    for (Dimension dim = 0; dim <= complex1.maxDimension(); ++dim)
    {
        count1 += complex1.simplicesOfDimension(dim).size();
    }
    for (Dimension dim = 0; dim <= complex2.maxDimension(); ++dim)
    {
        count2 += complex2.simplicesOfDimension(dim).size();
    }
    return static_cast<double>(
        std::abs(static_cast<long long>(count1) - static_cast<long long>(count2)));
}

double interleavingDistance(const SimplicialComplex &complex1, const SimplicialComplex &complex2)
{
    return editDistance(complex1, complex2);
}

double hausdorffDistance(const std::vector<std::vector<double>> &points1,
                         const std::vector<std::vector<double>> &points2)
{
    return pointSetHausdorff(points1, points2);
}

double chamferDistance(const std::vector<std::vector<double>> &points1,
                       const std::vector<std::vector<double>> &points2)
{
    if (points1.empty() || points2.empty())
    {
        return std::numeric_limits<double>::infinity();
    }

    double total = 0.0;
    for (const auto &point1 : points1)
    {
        double min_dist = std::numeric_limits<double>::infinity();
        for (const auto &point2 : points2)
        {
            min_dist = std::min(min_dist, euclideanDistance(point1, point2));
        }
        total += min_dist;
    }
    for (const auto &point2 : points2)
    {
        double min_dist = std::numeric_limits<double>::infinity();
        for (const auto &point1 : points1)
        {
            min_dist = std::min(min_dist, euclideanDistance(point2, point1));
        }
        total += min_dist;
    }

    return total / static_cast<double>(points1.size() + points2.size());
}

double earthMoversDistance(const std::vector<std::vector<double>> &points1,
                           const std::vector<std::vector<double>> &points2)
{
    return chamferDistance(points1, points2);
}

double hausdorffDistance(const core::PointBuffer &points1, const core::PointBuffer &points2)
{
    return pointSetHausdorff(pointBufferToPoints(points1), pointBufferToPoints(points2));
}

double chamferDistance(const core::PointBuffer &points1, const core::PointBuffer &points2)
{
    return chamferDistance(pointBufferToPoints(points1), pointBufferToPoints(points2));
}

double earthMoversDistance(const core::PointBuffer &points1, const core::PointBuffer &points2)
{
    return earthMoversDistance(pointBufferToPoints(points1), pointBufferToPoints(points2));
}

} // namespace nerve::metrics
