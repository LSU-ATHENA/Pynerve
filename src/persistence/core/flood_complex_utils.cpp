#include "nerve/persistence/core/flood_complex.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nerve::persistence
{

namespace
{

constexpr double CAYLEY_MENGER_COEFFICIENT = 288.0;

} // namespace

std::vector<int> farthestPointSampling(const std::vector<double> &points, size_t point_dim,
                                       size_t n_points, size_t subset_size)
{
    std::vector<int> selected;
    if (point_dim == 0 || n_points == 0 || subset_size == 0 || n_points > points.size() / point_dim)
    {
        return selected;
    }
    subset_size = std::min(subset_size, n_points);
    selected.reserve(subset_size);

    int first = 0;
    selected.push_back(first);

    std::vector<double> min_distances(n_points, std::numeric_limits<double>::infinity());

    for (size_t i = 0; i < n_points; ++i)
    {
        double dist_sq = 0.0;
        for (size_t d = 0; d < point_dim; ++d)
        {
            double diff = points[i * point_dim + d] - points[first * point_dim + d];
            dist_sq += diff * diff;
        }
        min_distances[i] = dist_sq;
    }

    for (size_t k = 1; k < subset_size; ++k)
    {
        int farthest = -1;
        double max_min_dist = -1.0;

        for (size_t i = 0; i < n_points; ++i)
        {
            if (min_distances[i] > max_min_dist)
            {
                max_min_dist = min_distances[i];
                farthest = static_cast<int>(i);
            }
        }

        if (farthest < 0)
            break;
        selected.push_back(farthest);

        for (size_t i = 0; i < n_points; ++i)
        {
            double dist_sq = 0.0;
            for (size_t d = 0; d < point_dim; ++d)
            {
                double diff = points[i * point_dim + d] - points[farthest * point_dim + d];
                dist_sq += diff * diff;
            }
            min_distances[i] = std::min(min_distances[i], dist_sq);
        }
    }

    return selected;
}

bool isSimplexFlooded(const std::vector<int> &simplex_vertices,
                      const std::vector<int> &candidate_points, const std::vector<double> &points,
                      size_t point_dim, double radius)
{
    double radius_sq = radius * radius;

    for (int idx : candidate_points)
    {
        const double *pt = &points[idx * point_dim];
        bool covered = false;

        for (int v_idx : simplex_vertices)
        {
            const double *v_pt = &points[v_idx * point_dim];

            double dist_sq = 0.0;
            for (size_t d = 0; d < point_dim; ++d)
            {
                double diff = pt[d] - v_pt[d];
                dist_sq += diff * diff;
            }

            if (dist_sq <= radius_sq)
            {
                covered = true;
                break;
            }
        }

        if (!covered)
        {
            return false;
        }
    }

    return true;
}

double simplexCircumradius(const std::vector<int> &vertices, const std::vector<double> &points,
                           size_t point_dim)
{
    if (vertices.size() == 2)
    {
        const double *p1 = &points[vertices[0] * point_dim];
        const double *p2 = &points[vertices[1] * point_dim];

        double dist_sq = 0.0;
        for (size_t d = 0; d < point_dim; ++d)
        {
            double diff = p1[d] - p2[d];
            dist_sq += diff * diff;
        }

        return std::sqrt(dist_sq) / 2.0;
    }
    else if (vertices.size() == 3 && point_dim >= 2)
    {
        const double *p0 = &points[vertices[0] * point_dim];
        const double *p1 = &points[vertices[1] * point_dim];
        const double *p2 = &points[vertices[2] * point_dim];

        double a_sq = 0.0, b_sq = 0.0, c_sq = 0.0;
        for (size_t d = 0; d < point_dim; ++d)
        {
            double d0 = p1[d] - p0[d];
            double d1 = p2[d] - p1[d];
            double d2 = p0[d] - p2[d];
            a_sq += d0 * d0;
            b_sq += d1 * d1;
            c_sq += d2 * d2;
        }

        double a = std::sqrt(a_sq);
        double b = std::sqrt(b_sq);
        double c = std::sqrt(c_sq);

        double area = 0.0;
        if (point_dim == 2)
        {
            area = 0.5 *
                   std::abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]));
        }
        else
        {
            double cx = (p1[1] - p0[1]) * (p2[2] - p0[2]) - (p1[2] - p0[2]) * (p2[1] - p0[1]);
            double cy = (p1[2] - p0[2]) * (p2[0] - p0[0]) - (p1[0] - p0[0]) * (p2[2] - p0[2]);
            double cz = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
            area = 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
        }

        if (area < 1e-10)
        {
            return std::max({a, b, c}) / 2.0;
        }

        return (a * b * c) / (4.0 * area);
    }
    else if (vertices.size() == 4 && point_dim >= 3)
    {
        const double *p0 = &points[vertices[0] * point_dim];
        const double *p1 = &points[vertices[1] * point_dim];
        const double *p2 = &points[vertices[2] * point_dim];
        const double *p3 = &points[vertices[3] * point_dim];

        double d01 = 0.0, d02 = 0.0, d03 = 0.0, d12 = 0.0, d13 = 0.0, d23 = 0.0;
        for (size_t d = 0; d < point_dim; ++d)
        {
            double diff01 = p1[d] - p0[d];
            double diff02 = p2[d] - p0[d];
            double diff03 = p3[d] - p0[d];
            double diff12 = p2[d] - p1[d];
            double diff13 = p3[d] - p1[d];
            double diff23 = p3[d] - p2[d];
            d01 += diff01 * diff01;
            d02 += diff02 * diff02;
            d03 += diff03 * diff03;
            d12 += diff12 * diff12;
            d13 += diff13 * diff13;
            d23 += diff23 * diff23;
        }

        d01 = std::sqrt(d01);
        d02 = std::sqrt(d02);
        d03 = std::sqrt(d03);
        d12 = std::sqrt(d12);
        d13 = std::sqrt(d13);
        d23 = std::sqrt(d23);

        double vol = 0.0;
        if (point_dim == 3)
        {
            double ax = p1[0] - p0[0], ay = p1[1] - p0[1], az = p1[2] - p0[2];
            double bx = p2[0] - p0[0], by = p2[1] - p0[1], bz = p2[2] - p0[2];
            double cx = p3[0] - p0[0], cy = p3[1] - p0[1], cz = p3[2] - p0[2];

            vol = std::abs(ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) +
                           az * (bx * cy - by * cx)) /
                  6.0;
        }

        if (vol < 1e-10)
        {
            return std::max({d01, d02, d03, d12, d13, d23}) / 2.0;
        }

        double cm_det = CAYLEY_MENGER_COEFFICIENT * vol * vol;
        double product = d01 * d02 * d03 * d12 * d13 * d23;

        return product / std::sqrt(cm_det);
    }

    double max_edge = 0.0;
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        for (size_t j = i + 1; j < vertices.size(); ++j)
        {
            const double *p1 = &points[vertices[i] * point_dim];
            const double *p2 = &points[vertices[j] * point_dim];

            double dist_sq = 0.0;
            for (size_t d = 0; d < point_dim; ++d)
            {
                double diff = p1[d] - p2[d];
                dist_sq += diff * diff;
            }

            max_edge = std::max(max_edge, std::sqrt(dist_sq));
        }
    }

    return max_edge / 2.0;
}

} // namespace nerve::persistence
