
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

bool isSupportedWeightFunction(const std::string &function)
{
    return function == "inverse_distance" || function == "exponential" || function == "linear";
}

void validateWeights(const std::vector<double> &weights)
{
    for (double weight : weights)
    {
        if (!std::isfinite(weight) || weight < 0.0)
        {
            throw std::invalid_argument("Weights must be finite and non-negative");
        }
    }
}

Size checkedSquareCount(Size count, const char *context)
{
    if (count != 0 && count > std::numeric_limits<Size>::max() / count)
    {
        throw std::length_error(context);
    }
    return count * count;
}

} // namespace

WeightedVietorisRips::WeightedVietorisRips(const std::vector<double> &weights)
{
    setWeights(weights);
}

void WeightedVietorisRips::setWeights(const std::vector<double> &weights)
{
    validateWeights(weights);
    weights_ = weights;
}

void WeightedVietorisRips::setAdaptiveRadius(bool adaptive)
{
    adaptive_radius_ = adaptive;
}

void WeightedVietorisRips::setWeightFunction(const std::string &function)
{
    if (!isSupportedWeightFunction(function))
    {
        throw std::invalid_argument("Unknown weight function: " + function);
    }
    weight_function_ = function;
}

std::vector<std::pair<algebra::Simplex, double>>
WeightedVietorisRips::buildFiltration(const core::ownership_utils::PointView &points,
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
    if (weights_.size() != num_points)
    {
        throw std::invalid_argument("Number of weights must match number of points");
    }
    checkedSquareCount(num_points, "weighted VR distance matrix size overflows size_t");

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

    std::vector<std::vector<double>> distances(num_points, std::vector<double>(num_points, 0.0));
    for (Size i = 0; i < num_points; ++i)
    {
        for (Size j = i + 1; j < num_points; ++j)
        {
            double dist_sq = 0.0;
            for (Size d = 0; d < dimension; ++d)
            {
                const double diff = points_vector[i][d] - points_vector[j][d];
                const double contribution = diff * diff;
                const double next_dist_sq = dist_sq + contribution;
                if (!std::isfinite(diff) || !std::isfinite(contribution) ||
                    !std::isfinite(next_dist_sq))
                {
                    throw std::invalid_argument("Weighted VR point distance overflow");
                }
                dist_sq = next_dist_sq;
            }
            const double dist = std::sqrt(dist_sq);
            if (!std::isfinite(dist))
            {
                throw std::invalid_argument("Weighted VR point distance overflow");
            }
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    std::vector<double> vertexRadii(num_points, 0.0);
    if (adaptive_radius_)
    {
        for (Size i = 0; i < num_points; ++i)
        {
            vertexRadii[i] = computeAdaptiveRadius(static_cast<Index>(i), 0);
            if (!std::isfinite(vertexRadii[i]))
            {
                throw std::invalid_argument("Weighted VR vertex radius must be finite");
            }
        }
    }

    std::vector<std::pair<algebra::Simplex, double>> filtration;
    filtration.reserve(num_points);
    for (Size i = 0; i < num_points; ++i)
    {
        const algebra::Simplex vertex({static_cast<Index>(i)});
        filtration.emplace_back(vertex, vertexRadii[i]);
    }

    constexpr Size kMaxDimension = 2;
    std::vector<std::vector<double>> edgeRadius(num_points, std::vector<double>(num_points, 0.0));
    for (Size i = 0; i < num_points; ++i)
    {
        for (Size j = i + 1; j < num_points; ++j)
        {
            const double combined_weight = 0.5 * weights_[i] + 0.5 * weights_[j];
            if (!std::isfinite(combined_weight))
            {
                throw std::invalid_argument("Weighted VR combined weight must be finite");
            }
            const double weighted_dist = applyWeightFunction(distances[i][j], combined_weight);
            const double radius = std::max({weighted_dist, vertexRadii[i], vertexRadii[j]});
            if (!std::isfinite(weighted_dist) || !std::isfinite(radius))
            {
                throw std::invalid_argument("Weighted VR filtration radius must be finite");
            }
            edgeRadius[i][j] = radius;
            edgeRadius[j][i] = radius;
            const algebra::Simplex edge({static_cast<Index>(i), static_cast<Index>(j)});
            filtration.emplace_back(edge, radius);
        }
    }

    if (kMaxDimension >= 2)
    {
        for (Size i = 0; i < num_points; ++i)
        {
            for (Size j = i + 1; j < num_points; ++j)
            {
                for (Size k = j + 1; k < num_points; ++k)
                {
                    const double radius =
                        std::max({edgeRadius[i][j], edgeRadius[i][k], edgeRadius[j][k]});
                    if (!std::isfinite(radius))
                    {
                        throw std::invalid_argument("Weighted VR filtration radius must be finite");
                    }
                    const algebra::Simplex triangle(
                        {static_cast<Index>(i), static_cast<Index>(j), static_cast<Index>(k)});
                    filtration.emplace_back(triangle, radius);
                }
            }
        }
    }

    std::ranges::sort(filtration, {}, [](const auto &p) {
        return std::tuple(p.second, p.first.dimension(), p.first);
    });

    return filtration;
}

std::vector<double>
WeightedVietorisRips::computeLocalWeights(const core::ownership_utils::PointView &points,
                                          size_t dimension) const
{
    if (dimension == 0 || points.size() % dimension != 0)
    {
        throw std::invalid_argument("Invalid point buffer or dimension");
    }
    if (weights_.size() != points.size() / dimension)
    {
        throw std::invalid_argument("Number of weights must match number of points");
    }
    return weights_;
}

double WeightedVietorisRips::computeAdaptiveRadius(Index point_index, Size k) const
{
    if (k != 0)
    {
        throw std::invalid_argument(
            "Weighted adaptive radius only supports per-vertex radius queries");
    }
    if (point_index < 0 || point_index >= static_cast<Index>(weights_.size()))
    {
        return 1.0;
    }

    const double weight = weights_[point_index];
    if (weight_function_ == "inverse_distance")
    {
        return 1.0 / (1.0 + weight);
    }
    if (weight_function_ == "exponential")
    {
        return std::exp(-weight);
    }
    return weight;
}

double WeightedVietorisRips::computeWeightedRadius(Index i, Index j) const
{
    if (i < 0 || j < 0 || i >= static_cast<Index>(weights_.size()) ||
        j >= static_cast<Index>(weights_.size()))
    {
        return 1.0;
    }
    const double combined_weight = 0.5 * weights_[i] + 0.5 * weights_[j];
    return applyWeightFunction(1.0, combined_weight);
}

double WeightedVietorisRips::applyWeightFunction(double distance, double weight) const
{
    if (weight_function_ == "inverse_distance")
    {
        return distance / (1.0 + weight);
    }
    if (weight_function_ == "exponential")
    {
        return distance * std::exp(-weight);
    }
    if (weight_function_ == "linear")
    {
        return distance * std::max(0.1, 1.0 - weight);
    }
    return distance;
}

} // namespace nerve::filtration
