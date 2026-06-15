#include "nerve/persistence/streaming/streaming_reducer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <vector>

namespace nerve::persistence::streaming
{
namespace
{

constexpr int kMaxStreamingSimplexDimension = 5;

bool squareCountExceedsCapacity(size_t n)
{
    if (n != 0 && n > std::numeric_limits<size_t>::max() / n)
    {
        return true;
    }
    return n * n > std::vector<double>().max_size();
}

} // namespace

StreamingColumnGenerator::StreamingColumnGenerator(std::span<const double> points, size_t n_points,
                                                   size_t point_dim, double max_distance)
    : points_(points)
    , n_points_(n_points)
    , point_dim_(point_dim)
    , max_distance_(max_distance)
    , total_simplices_(0)
{
    if (point_dim_ == 0 || n_points_ > points_.size() / point_dim_ ||
        n_points_ > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        squareCountExceedsCapacity(n_points_) || !std::isfinite(max_distance_) ||
        max_distance_ < 0.0)
    {
        n_points_ = 0;
        point_dim_ = 0;
        return;
    }
    const long double safe_abs =
        std::sqrt(static_cast<long double>(std::numeric_limits<double>::max()) /
                  static_cast<long double>(point_dim_)) /
        4.0L;
    if (!std::ranges::all_of(points_.first(n_points_ * point_dim_), [&](double value) {
            return std::isfinite(value) && std::abs(static_cast<long double>(value)) <= safe_abs;
        }))
    {
        n_points_ = 0;
        point_dim_ = 0;
        return;
    }

    buildAdjacency();
    enumerateAllSimplices();
}

std::vector<int> StreamingColumnGenerator::generateColumn(size_t simplex_idx) const
{
    std::vector<int> boundary;

    auto vertices = simplexToVertices(simplex_idx);
    int dim = static_cast<int>(vertices.size()) - 1;

    if (dim == 0)
    {
        return boundary;
    }

    for (size_t i = 0; i < vertices.size(); ++i)
    {
        std::vector<int> face;
        face.reserve(vertices.size() - 1);

        for (size_t j = 0; j < vertices.size(); ++j)
        {
            if (j != i)
            {
                face.push_back(vertices[j]);
            }
        }

        int face_idx = verticesToSimplexIndex(face);
        if (face_idx >= 0)
        {
            boundary.push_back(face_idx);
        }
    }

    std::sort(boundary.begin(), boundary.end());

    return boundary;
}

size_t StreamingColumnGenerator::getNumSimplices() const
{
    return total_simplices_;
}

double StreamingColumnGenerator::getFiltrationValue(size_t simplex_idx) const
{
    if (simplex_idx < all_simplices_.size())
    {
        return all_simplices_[simplex_idx].filtration_value;
    }
    return 0.0;
}

int StreamingColumnGenerator::getSimplexDimension(size_t simplex_idx) const
{
    if (simplex_idx < all_simplices_.size())
    {
        return static_cast<int>(all_simplices_[simplex_idx].vertices.size()) - 1;
    }
    return -1;
}

std::vector<int> StreamingColumnGenerator::simplexToVertices(size_t simplex_idx) const
{
    if (simplex_idx < all_simplices_.size())
    {
        return all_simplices_[simplex_idx].vertices;
    }
    return {};
}

int StreamingColumnGenerator::verticesToSimplexIndex(const std::vector<int> &vertices) const
{
    auto it = simplex_index_map_.find(vertices);
    return it == simplex_index_map_.end() ? -1 : static_cast<int>(it->second);
}

void StreamingColumnGenerator::buildAdjacency()
{
    adjacency_.resize(n_points_);
    if (squareCountExceedsCapacity(n_points_))
    {
        n_points_ = 0;
        point_dim_ = 0;
        adjacency_.clear();
        edge_weights_.clear();
        return;
    }
    const size_t edge_weight_count = n_points_ * n_points_;
    edge_weights_.assign(edge_weight_count, std::numeric_limits<double>::infinity());

    for (size_t i = 0; i < n_points_; ++i)
    {
        edge_weights_[i * n_points_ + i] = 0.0;
        for (size_t j = i + 1; j < n_points_; ++j)
        {
            double dist = computeDistance(i, j);
            if (std::isfinite(dist) && dist <= max_distance_)
            {
                adjacency_[i].push_back(static_cast<int>(j));
                adjacency_[j].push_back(static_cast<int>(i));
                edge_weights_[i * n_points_ + j] = dist;
                edge_weights_[j * n_points_ + i] = dist;
            }
        }
    }
}

void StreamingColumnGenerator::enumerateAllSimplices()
{
    BronKerboschEnumerator enumerator(adjacency_);
    all_simplices_ = enumerator.enumerateCliques(std::span<const double>(edge_weights_), n_points_,
                                                 kMaxStreamingSimplexDimension);

    for (auto &simplex : all_simplices_)
    {
        simplex_index_map_[simplex.vertices] = simplex.simplex_index;
    }

    edge_simplices_.clear();
    triangle_list_.clear();
    tetra_list_.clear();
    four_simplices_.clear();
    for (const auto &simplex : all_simplices_)
    {
        int dim = static_cast<int>(simplex.vertices.size()) - 1;
        switch (dim)
        {
            case 0:
                break;
            case 1:
                edge_simplices_.push_back(simplex);
                break;
            case 2:
                triangle_list_.push_back(simplex);
                break;
            case 3:
                tetra_list_.push_back(simplex);
                break;
            case 4:
                four_simplices_.push_back(simplex);
                break;
        }
    }

    total_simplices_ = all_simplices_.size();
}

double StreamingColumnGenerator::computeDistance(size_t i, size_t j) const
{
    double sum = 0.0;
    for (size_t d = 0; d < point_dim_; ++d)
    {
        double diff = points_[i * point_dim_ + d] - points_[j * point_dim_ + d];
        const double contribution = diff * diff;
        if (!std::isfinite(contribution) || sum > std::numeric_limits<double>::max() - contribution)
        {
            return std::numeric_limits<double>::infinity();
        }
        sum += contribution;
    }
    const double distance = std::sqrt(sum);
    return std::isfinite(distance) ? distance : std::numeric_limits<double>::infinity();
}

} // namespace nerve::persistence::streaming
