
#include "nerve/filtration/vietoris_rips.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace nerve::filtration
{
namespace
{

Size checkedSquareCount(Size count, const char *context)
{
    if (count != 0 && count > std::numeric_limits<Size>::max() / count)
    {
        throw std::length_error(context);
    }
    return count * count;
}

} // namespace

SparseVietorisRips::SparseVietorisRips(Size k_neighbors)
{
    setKNeighbors(k_neighbors);
}

void SparseVietorisRips::setKNeighbors(Size k)
{
    k_neighbors_ = k;
}

void SparseVietorisRips::setApproximationFactor(double factor)
{
    if (!std::isfinite(factor) || factor <= 0.0)
    {
        throw std::invalid_argument("Approximation factor must be positive and finite");
    }
    approximation_factor_ = factor;
}

void SparseVietorisRips::setBatchSize(Size batch_size)
{
    if (batch_size == 0)
    {
        throw std::invalid_argument("Batch size must be positive");
    }
    batch_size_ = batch_size;
}

std::vector<std::pair<algebra::Simplex, double>>
SparseVietorisRips::buildFiltration(const core::ownership_utils::PointView &points,
                                    size_t dimension, const core::DeterminismContract &contract)
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        throw std::runtime_error("Cannot satisfy strict determinism contract");
    }

    if (dimension == 0 || points.size() == 0 || points.size() % dimension != 0)
    {
        throw std::invalid_argument("Invalid point buffer or dimension");
    }

    const size_t num_points = points.size() / dimension;
    checkedSquareCount(num_points, "sparse VR distance matrix size overflows size_t");
    std::vector<std::vector<double>> points_vector;
    points_vector.reserve(num_points);

    const auto *data_ptr = static_cast<const double *>(points.data());
    if (data_ptr == nullptr)
    {
        throw std::invalid_argument("Point buffer data is null");
    }
    for (size_t i = 0; i < num_points; ++i)
    {
        points_vector.emplace_back(data_ptr + i * dimension, data_ptr + (i + 1) * dimension);
        if (!std::ranges::all_of(points_vector.back(),
                                 [](double value) { return std::isfinite(value); }))
        {
            throw std::invalid_argument("Point coordinates must be finite");
        }
    }

    buildNeighborGraph(points_vector);
    buildApproximateFiltration();
    return filtration_;
}

std::vector<Index> SparseVietorisRips::findApproximateNeighbors(Index point_index) const
{
    if (point_index >= 0 && point_index < static_cast<Index>(neighbor_graph_.size()))
    {
        return neighbor_graph_[point_index];
    }
    return {};
}

double SparseVietorisRips::approximateDistance(Index i, Index j) const
{
    if (i >= 0 && j >= 0 && i < static_cast<Index>(neighbor_distances_.size()) &&
        j < static_cast<Index>(neighbor_distances_[i].size()))
    {
        return neighbor_distances_[i][j];
    }
    return std::numeric_limits<double>::infinity();
}

void SparseVietorisRips::buildNeighborGraph(const std::vector<std::vector<double>> &points)
{
    const Size n = points.size();
    checkedSquareCount(n, "sparse VR distance matrix size overflows size_t");
    neighbor_graph_.resize(n);
    neighbor_distances_.resize(n, std::vector<double>(n, std::numeric_limits<double>::infinity()));

    for (Size i = 0; i < n; ++i)
    {
        neighbor_graph_[i].clear();
        std::vector<std::pair<double, Index>> distances;
        const Size k_limit = std::min(k_neighbors_, n > 0 ? n - 1 : 0);
        distances.reserve(k_limit);
        const auto farther = [](const std::pair<double, Index> &lhs,
                                const std::pair<double, Index> &rhs) {
            if (lhs.first != rhs.first)
            {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        };
        for (Size j = 0; j < n; ++j)
        {
            if (i == j)
            {
                continue;
            }
            double dist = 0.0;
            for (Size d = 0; d < std::min(points[i].size(), points[j].size()); ++d)
            {
                const double diff = points[i][d] - points[j][d];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);
            neighbor_distances_[i][j] = dist;
            if (k_limit == 0)
            {
                continue;
            }

            const std::pair<double, Index> candidate{dist, static_cast<Index>(j)};
            if (distances.size() < k_limit)
            {
                distances.push_back(candidate);
                std::ranges::push_heap(distances, farther);
            }
            else if (candidate < distances.front())
            {
                std::ranges::pop_heap(distances, farther);
                distances.back() = candidate;
                std::ranges::push_heap(distances, farther);
            }
        }

        std::ranges::sort(distances);
        for (const auto &[distance, neighbor] : distances)
        {
            (void)distance;
            neighbor_graph_[i].push_back(neighbor);
        }
    }
}

void SparseVietorisRips::buildApproximateFiltration()
{
    filtration_.clear();
    const Size n = neighbor_graph_.size();
    filtration_.reserve(n);

    for (Size i = 0; i < n; ++i)
    {
        const algebra::Simplex vertex({static_cast<Index>(i)});
        filtration_.emplace_back(vertex, 0.0);
    }

    std::vector<std::vector<double>> edgeWeight(
        n, std::vector<double>(n, std::numeric_limits<double>::infinity()));

    for (Size i = 0; i < n; ++i)
    {
        for (const Index neighbor : neighbor_graph_[i])
        {
            if (neighbor < 0 || static_cast<Size>(neighbor) >= n)
            {
                continue;
            }
            const double dist = neighbor_distances_[i][static_cast<Size>(neighbor)];
            if (std::isinf(dist))
            {
                continue;
            }
            const double weight = dist * approximation_factor_;
            const Size j = static_cast<Size>(neighbor);
            if (weight < edgeWeight[i][j])
            {
                edgeWeight[i][j] = weight;
                edgeWeight[j][i] = weight;
            }
        }
    }

    for (Size i = 0; i < n; ++i)
    {
        for (Size j = i + 1; j < n; ++j)
        {
            if (!std::isinf(edgeWeight[i][j]))
            {
                const algebra::Simplex edge({static_cast<Index>(i), static_cast<Index>(j)});
                filtration_.emplace_back(edge, edgeWeight[i][j]);
            }
        }
    }

    constexpr Size kMaxDimension = 2;
    if (kMaxDimension >= 2)
    {
        for (Size i = 0; i < n; ++i)
        {
            for (Size j = i + 1; j < n; ++j)
            {
                if (std::isinf(edgeWeight[i][j]))
                {
                    continue;
                }
                for (Size k = j + 1; k < n; ++k)
                {
                    if (std::isinf(edgeWeight[i][k]) || std::isinf(edgeWeight[j][k]))
                    {
                        continue;
                    }
                    const double radius =
                        std::max({edgeWeight[i][j], edgeWeight[i][k], edgeWeight[j][k]});
                    const algebra::Simplex triangle(
                        {static_cast<Index>(i), static_cast<Index>(j), static_cast<Index>(k)});
                    filtration_.emplace_back(triangle, radius);
                }
            }
        }
    }

    std::ranges::sort(filtration_, {}, [](const auto &p) {
        return std::tuple(p.second, p.first.dimension(), p.first);
    });
}

} // namespace nerve::filtration
