#include "nerve/filtration/vietoris_rips.hpp"
#include "nerve/gpu/manager_compute_ops_filtration.hpp"
#include "nerve/gpu/manager_compute_ops_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace nerve::gpu
{
constexpr double kGeometryTolerance = 1e-12;

namespace
{

errors::ErrorResult<void> invalidInput(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT, message);
}

errors::ErrorResult<void> resourceLimit(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT, message);
}

errors::ErrorResult<void> numericError(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN, message);
}

bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

errors::ErrorResult<void> flattenPointRows(const std::vector<std::vector<double>> &points,
                                           std::vector<double> &flat, std::size_t &point_dim)
{
    flat.clear();
    point_dim = 0;
    if (points.empty())
    {
        return errors::ErrorResult<void>::success();
    }
    point_dim = points.front().size();
    if (point_dim == 0)
    {
        return invalidInput("points must have positive dimension");
    }
    std::size_t total_values = 0;
    if (!checkedProduct(points.size(), point_dim, total_values))
    {
        return resourceLimit("point buffer size overflows size_t");
    }
    flat.reserve(total_values);
    for (const auto &point : points)
    {
        if (point.size() != point_dim)
        {
            flat.clear();
            return invalidInput("point dimensions must be uniform");
        }
        for (double value : point)
        {
            if (!std::isfinite(value))
            {
                flat.clear();
                return numericError("point coordinates must be finite");
            }
            flat.push_back(value);
        }
    }
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> simplexVerticesAsInt(const algebra::Simplex &simplex,
                                               std::vector<int> &vertices)
{
    vertices.clear();
    vertices.reserve(simplex.vertices().size());
    for (Index vertex : simplex.vertices())
    {
        if (vertex < 0 || vertex > static_cast<Index>(std::numeric_limits<int>::max()))
        {
            return resourceLimit("simplex vertex index exceeds int range");
        }
        vertices.push_back(static_cast<int>(vertex));
    }
    return errors::ErrorResult<void>::success();
}

} // namespace

errors::ErrorResult<void>
buildVietorisRipsFiltration(const std::vector<std::vector<double>> &points, double max_radius,
                            int max_dimension,
                            std::vector<std::pair<algebra::Simplex, double>> &filtration)
{
    filtration.clear();
    if (!std::isfinite(max_radius) || max_radius < 0.0 || max_dimension < 0)
    {
        return invalidInput("invalid filtration radius or dimension");
    }
    if (points.empty())
    {
        return errors::ErrorResult<void>::success();
    }

    std::vector<double> flat_points;
    std::size_t point_dim = 0;
    auto point_status = flattenPointRows(points, flat_points, point_dim);
    if (point_status.isError())
    {
        return point_status;
    }

    filtration::VietorisRips vr(max_radius);
    vr.setMaxDimension(static_cast<Size>(max_dimension));
    const core::ownership_utils::PointView point_view(flat_points.data(), flat_points.size());
    auto filtration_result = vr.buildFiltration(point_view, point_dim);
    if (filtration_result.isError())
    {
        return errors::ErrorResult<void>::error(filtration_result.errorCode(),
                                                filtration_result.error().message);
    }
    filtration = filtration_result.moveValue();
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void>
convertVrFiltration(const std::vector<std::pair<algebra::Simplex, double>> &filtration,
                    std::vector<ComputeManager::VRSimplex> &out_simplices)
{
    out_simplices.clear();
    out_simplices.reserve(filtration.size());
    for (const auto &[simplex, value] : filtration)
    {
        ComputeManager::VRSimplex out;
        auto vertex_status = simplexVerticesAsInt(simplex, out.vertices);
        if (vertex_status.isError())
        {
            out_simplices.clear();
            return vertex_status;
        }
        if (simplex.dimension() > static_cast<Size>(std::numeric_limits<int>::max()))
        {
            out_simplices.clear();
            return resourceLimit("simplex dimension exceeds int range");
        }
        out.filtration_value = value;
        out.dimension = static_cast<int>(simplex.dimension());
        out_simplices.push_back(std::move(out));
    }
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> buildCechLikeComplex(const std::vector<std::vector<double>> &points,
                                               double max_radius, int max_dimension,
                                               std::vector<ComputeManager::CechSimplex> &out)
{
    out.clear();
    if (!std::isfinite(max_radius) || max_radius < 0.0 || max_dimension < 0)
    {
        return invalidInput("invalid complex radius or dimension");
    }
    if (max_radius > std::numeric_limits<double>::max() * 0.5)
    {
        return resourceLimit("complex radius exceeds supported range");
    }

    std::vector<std::pair<algebra::Simplex, double>> candidates;
    auto filtration_status =
        buildVietorisRipsFiltration(points, max_radius * 2.0, max_dimension, candidates);
    if (filtration_status.isError())
    {
        return filtration_status;
    }

    out.reserve(candidates.size());
    for (const auto &[simplex, ignored_value] : candidates)
    {
        (void)ignored_value;
        ComputeManager::CechSimplex out_simplex;
        auto vertex_status = simplexVerticesAsInt(simplex, out_simplex.vertices);
        if (vertex_status.isError())
        {
            out.clear();
            return vertex_status;
        }
        if (simplex.dimension() > static_cast<Size>(std::numeric_limits<int>::max()))
        {
            out.clear();
            return resourceLimit("simplex dimension exceeds int range");
        }

        const double radius = enclosingRadiusEstimate(points, out_simplex.vertices);
        if (!std::isfinite(radius) || radius > max_radius + kGeometryTolerance)
        {
            continue;
        }
        out_simplex.filtration_value = radius;
        out_simplex.alpha_value = radius;
        out_simplex.dimension = static_cast<int>(simplex.dimension());
        out_simplex.isValid = true;
        out.push_back(std::move(out_simplex));
    }

    std::sort(out.begin(), out.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.dimension != rhs.dimension)
        {
            return lhs.dimension < rhs.dimension;
        }
        if (lhs.filtration_value != rhs.filtration_value)
        {
            return lhs.filtration_value < rhs.filtration_value;
        }
        return lhs.vertices < rhs.vertices;
    });
    return errors::ErrorResult<void>::success();
}

} // namespace nerve::gpu
