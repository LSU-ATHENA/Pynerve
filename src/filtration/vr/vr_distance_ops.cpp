
#include "nerve/filtration/vietoris_rips.hpp"

#include <cmath>
#include <stdexcept>

namespace nerve::filtration
{

double VietorisRips::euclideanDistance(const std::vector<double> &p1,
                                       const std::vector<double> &p2) const
{
    if (p1.size() != p2.size())
    {
        throw std::invalid_argument("Points must have the same dimension");
    }

    double sum = 0.0;
    for (Size i = 0; i < p1.size(); ++i)
    {
        const double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double VietorisRips::manhattanDistance(const std::vector<double> &p1,
                                       const std::vector<double> &p2) const
{
    if (p1.size() != p2.size())
    {
        throw std::invalid_argument("Points must have the same dimension");
    }

    double sum = 0.0;
    for (Size i = 0; i < p1.size(); ++i)
    {
        sum += std::abs(p1[i] - p2[i]);
    }
    return sum;
}

double VietorisRips::chebyshevDistance(const std::vector<double> &p1,
                                       const std::vector<double> &p2) const
{
    if (p1.size() != p2.size())
    {
        throw std::invalid_argument("Points must have the same dimension");
    }

    double max_diff = 0.0;
    for (Size i = 0; i < p1.size(); ++i)
    {
        max_diff = std::max(max_diff, std::abs(p1[i] - p2[i]));
    }
    return max_diff;
}

double VietorisRips::cosineDistance(const std::vector<double> &p1,
                                    const std::vector<double> &p2) const
{
    if (p1.size() != p2.size())
    {
        throw std::invalid_argument("Points must have the same dimension");
    }

    double dot_product = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    for (Size i = 0; i < p1.size(); ++i)
    {
        dot_product += p1[i] * p2[i];
        norm1 += p1[i] * p1[i];
        norm2 += p2[i] * p2[i];
    }

    if (norm1 == 0.0 || norm2 == 0.0)
    {
        return 1.0;
    }

    const double cosine_similarity = dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
    return 1.0 - cosine_similarity;
}

void VietorisRips::validatePoints(const std::vector<std::vector<double>> &points) const
{
    if (points.empty())
    {
        throw std::invalid_argument("Point set cannot be empty");
    }

    const Size dim = points[0].size();
    for (const auto &point : points)
    {
        if (point.size() != dim)
        {
            throw std::invalid_argument("All points must have the same dimension");
        }
        for (const double coord : point)
        {
            if (std::isnan(coord) || std::isinf(coord))
            {
                throw std::invalid_argument("Point coordinates must be finite");
            }
        }
    }
}

} // namespace nerve::filtration
