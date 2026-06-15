
#include "nerve/filtration/vietoris_rips.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace nerve::filtration
{
namespace
{

bool isSupportedMetric(const std::string &metric)
{
    return metric == "euclidean" || metric == "manhattan" || metric == "chebyshev" ||
           metric == "cosine";
}

Size checkedProduct(Size lhs, Size rhs, const char *context)
{
    if (lhs != 0 && rhs > std::numeric_limits<Size>::max() / lhs)
    {
        throw std::length_error(context);
    }
    return lhs * rhs;
}

Size checkedSquareCount(Size count, const char *context)
{
    return checkedProduct(count, count, context);
}

std::vector<std::vector<double>> pointRows(const core::ownership_utils::PointView &points,
                                           size_t dimension)
{
    if (dimension == 0 || points.size() == 0 || points.size() % dimension != 0)
    {
        throw std::invalid_argument("Invalid point buffer shape");
    }
    const auto *data_ptr = static_cast<const double *>(points.data());
    if (data_ptr == nullptr)
    {
        throw std::invalid_argument("Point buffer data is null");
    }
    const size_t num_points = points.size() / dimension;
    checkedSquareCount(num_points, "VR distance matrix size overflows size_t");
    std::vector<std::vector<double>> rows;
    rows.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i)
    {
        rows.emplace_back(data_ptr + i * dimension, data_ptr + (i + 1) * dimension);
    }
    return rows;
}

} // namespace

VietorisRips::VietorisRips(double max_radius)
{
    setMaxRadius(max_radius);
}

void VietorisRips::setMaxRadius(double radius)
{
    if (!std::isfinite(radius) || radius < 0.0)
    {
        throw std::invalid_argument("Max radius must be finite and non-negative");
    }
    max_radius_ = radius;
}

void VietorisRips::setMaxDimension(Size dimension)
{
    max_dimension_ = dimension;
}

void VietorisRips::setDistanceMetric(const std::string &metric)
{
    if (!isSupportedMetric(metric))
    {
        throw std::invalid_argument("Unknown distance metric: " + metric);
    }
    distance_metric_ = metric;
}

errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>
VietorisRips::buildFiltration(const core::ownership_utils::PointView &points, size_t dimension,
                              const core::DeterminismContract &contract)
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            errors::ErrorCode::E30_DET_MISMATCH);
    }
    if (dimension == 0 || points.size() == 0 || points.size() % dimension != 0)
    {
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }

    const auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double>> points_vector;
    try
    {
        points_vector = pointRows(points, dimension);
        validatePoints(points_vector);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }

    points_ = std::move(points_vector);
    filtration_.clear();

    buildDistanceMatrix();
    buildSimplices();
    sortFiltration();

    const auto end = std::chrono::high_resolution_clock::now();
    computation_time_ = std::chrono::duration<double>(end - start).count();
    return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::success(
        std::vector<std::pair<algebra::Simplex, double>>(filtration_));
}

errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>
VietorisRips::buildFiltration(const core::ownership_utils::PointView &points, size_t dimension,
                              const core::ownership_utils::PointView &custom_radii,
                              const core::DeterminismContract &contract)
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            errors::ErrorCode::E30_DET_MISMATCH);
    }
    if (dimension == 0 || points.size() == 0 || points.size() % dimension != 0)
    {
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }

    const auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double>> points_vector;
    try
    {
        points_vector = pointRows(points, dimension);
        validatePoints(points_vector);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }

    const size_t num_points = points_vector.size();
    if (custom_radii.size() < num_points)
    {
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }
    const auto *radii_ptr = static_cast<const double *>(custom_radii.data());
    if (radii_ptr == nullptr)
    {
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }
    std::vector<double> radiiVector(radii_ptr, radii_ptr + num_points);
    if (std::ranges::any_of(radiiVector,
                            [](double radius) { return !std::isfinite(radius) || radius < 0.0; }))
    {
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }

    points_ = std::move(points_vector);
    filtration_.clear();

    buildDistanceMatrix();

    for (Size i = 0; i < points_.size(); ++i)
    {
        const algebra::Simplex vertex({static_cast<Index>(i)});
        const double radius = (i < radiiVector.size()) ? radiiVector[i] : max_radius_;
        filtration_.emplace_back(vertex, radius);
    }

    for (Size k = 1; k <= max_dimension_; ++k)
    {
        buildKSimplicesWithCustomRadii(k, radiiVector, 1.0);
    }

    sortFiltration();

    const auto end = std::chrono::high_resolution_clock::now();
    computation_time_ = std::chrono::duration<double>(end - start).count();
    return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::success(
        std::vector<std::pair<algebra::Simplex, double>>(filtration_));
}

void VietorisRips::addPoint(const core::ownership_utils::PointView &point, size_t dimension)
{
    auto rows = pointRows(point, dimension);
    if (rows.size() != 1 || (!points_.empty() && rows.front().size() != points_.front().size()))
    {
        throw std::invalid_argument("Point dimension does not match existing filtration");
    }
    validatePoints(rows);
    std::vector<double> pointVector = std::move(rows.front());
    points_.push_back(std::move(pointVector));
    buildDistanceMatrix();
    filtration_.clear();
    buildSimplices();
    sortFiltration();
}

void VietorisRips::addPoints(const core::ownership_utils::PointView &points, size_t dimension)
{
    std::vector<std::vector<double>> points_vector = pointRows(points, dimension);
    if (!points_.empty() && !points_vector.empty() &&
        points_vector.front().size() != points_.front().size())
    {
        throw std::invalid_argument("Point dimension does not match existing filtration");
    }

    validatePoints(points_vector);
    points_.insert(points_.end(), points_vector.begin(), points_vector.end());
    buildDistanceMatrix();
    filtration_.clear();
    buildSimplices();
    sortFiltration();
}

std::vector<std::pair<algebra::Simplex, double>> VietorisRips::getCurrentFiltration()
{
    return filtration_;
}

core::ownership_utils::OwnedPointBuffer
VietorisRips::computeDistanceMatrix(const core::ownership_utils::PointView &points,
                                    size_t dimension) const
{
    std::vector<std::vector<double>> points_vector = pointRows(points, dimension);
    validatePoints(points_vector);

    const Size n = points_vector.size();
    const Size matrix_size = checkedSquareCount(n, "VR distance matrix size overflows size_t");
    auto matrix_data = std::make_unique<double[]>(matrix_size);

    for (Size i = 0; i < n; ++i)
    {
        for (Size j = i + 1; j < n; ++j)
        {
            const double *p1_data = points_vector[i].data();
            const double *p2_data = points_vector[j].data();
            const core::ownership_utils::PointView p1View(p1_data, points_vector[i].size());
            const core::ownership_utils::PointView p2View(p2_data, points_vector[j].size());
            const double dist = computeDistance(p1View, p2View, dimension);
            matrix_data[i * n + j] = dist;
            matrix_data[j * n + i] = dist;
        }
    }

    return core::PointBuffer(matrix_data.release(), n, n);
}

double VietorisRips::computeDistance(const core::ownership_utils::PointView &p1,
                                     const core::ownership_utils::PointView &p2,
                                     size_t dimension) const
{
    if (dimension == 0 || p1.size() < dimension || p2.size() < dimension)
    {
        throw std::invalid_argument("Point buffers must contain the requested dimension");
    }
    const auto *p1_data = static_cast<const double *>(p1.data());
    const auto *p2_data = static_cast<const double *>(p2.data());
    if (p1_data == nullptr || p2_data == nullptr)
    {
        throw std::invalid_argument("Point buffer data is null");
    }

    std::vector<double> p1Vec(p1_data, p1_data + dimension);
    std::vector<double> p2Vec(p2_data, p2_data + dimension);

    if (distance_metric_ == "euclidean")
    {
        return euclideanDistance(p1Vec, p2Vec);
    }
    if (distance_metric_ == "manhattan")
    {
        return manhattanDistance(p1Vec, p2Vec);
    }
    if (distance_metric_ == "chebyshev")
    {
        return chebyshevDistance(p1Vec, p2Vec);
    }
    if (distance_metric_ == "cosine")
    {
        return cosineDistance(p1Vec, p2Vec);
    }
    throw std::invalid_argument("Unknown distance metric: " + distance_metric_);
}

Size VietorisRips::getNumSimplices() const
{
    return filtration_.size();
}

Size VietorisRips::getNumSimplicesOfDimension(Size dim) const
{
    Size count = 0;
    for (const auto &entry : filtration_)
    {
        if (entry.first.dimension() == dim)
        {
            ++count;
        }
    }
    return count;
}

std::vector<double> VietorisRips::getFiltrationValues() const
{
    std::vector<double> values;
    values.reserve(filtration_.size());
    for (const auto &entry : filtration_)
    {
        values.push_back(entry.second);
    }
    return values;
}

std::vector<Size> VietorisRips::getSimplexDimensions() const
{
    std::vector<Size> dimensions;
    dimensions.reserve(filtration_.size());
    for (const auto &entry : filtration_)
    {
        dimensions.push_back(entry.first.dimension());
    }
    return dimensions;
}

double VietorisRips::getComputationTime() const
{
    return computation_time_;
}

void VietorisRips::buildDistanceMatrix()
{
    if (points_.empty())
    {
        distance_matrix_.clear();
        return;
    }
    const size_t num_points = points_.size();
    checkedSquareCount(num_points, "VR distance matrix size overflows size_t");
    const Size point_dimension = points_.front().size();
    distance_matrix_.assign(num_points, std::vector<double>(num_points, 0.0));
    for (size_t i = 0; i < num_points; ++i)
    {
        for (size_t j = i + 1; j < num_points; ++j)
        {
            const core::ownership_utils::PointView p1View(points_[i].data(), point_dimension);
            const core::ownership_utils::PointView p2View(points_[j].data(), point_dimension);
            const double dist = computeDistance(p1View, p2View, point_dimension);
            distance_matrix_[i][j] = dist;
            distance_matrix_[j][i] = dist;
        }
    }
}

} // namespace nerve::filtration
