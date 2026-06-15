#include "nerve/gpu/kernel_launcher.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace nerve::gpu
{
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

bool checkedIntIndex(std::size_t value, int &out) noexcept
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

errors::ErrorResult<void> validateRadius(double radius)
{
    if (!std::isfinite(radius) || radius < 0.0)
    {
        return invalidInput("streaming radius must be finite and non-negative");
    }
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> validatePointRows(const std::vector<std::vector<double>> &points,
                                            std::size_t &point_dim)
{
    point_dim = 0;
    if (points.empty())
    {
        return errors::ErrorResult<void>::success();
    }
    point_dim = points.front().size();
    if (point_dim == 0)
    {
        return invalidInput("streaming points must have positive dimension");
    }
    for (const auto &point : points)
    {
        if (point.size() != point_dim)
        {
            return invalidInput("streaming point dimensions must be uniform");
        }
        for (double value : point)
        {
            if (!std::isfinite(value))
            {
                return numericError("streaming point coordinates must be finite");
            }
        }
    }
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> validateSlidePoint(const std::vector<double> &point,
                                             std::size_t expected_dim, std::string_view label)
{
    if (point.empty())
    {
        return errors::ErrorResult<void>::success();
    }
    if (expected_dim != 0 && point.size() != expected_dim)
    {
        return invalidInput(std::string(label) + " dimension does not match streaming window");
    }
    for (double value : point)
    {
        if (!std::isfinite(value))
        {
            return numericError(std::string(label) + " coordinates must be finite");
        }
    }
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<double> squaredDistance(const std::vector<double> &lhs,
                                            const std::vector<double> &rhs)
{
    double distance = 0.0;
    for (std::size_t dim = 0; dim < lhs.size(); ++dim)
    {
        const double diff = lhs[dim] - rhs[dim];
        if (!std::isfinite(diff))
        {
            return errors::ErrorResult<double>::error(errors::ErrorCode::E20_NUM_NAN,
                                                      "streaming point distance overflow");
        }
        const double contribution = diff * diff;
        if (!std::isfinite(contribution) ||
            distance > std::numeric_limits<double>::max() - contribution)
        {
            return errors::ErrorResult<double>::error(errors::ErrorCode::E20_NUM_NAN,
                                                      "streaming point distance overflow");
        }
        distance += contribution;
    }
    return errors::ErrorResult<double>::success(double{distance});
}

bool pointMatches(const std::vector<double> &lhs, const std::vector<double> &rhs)
{
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

errors::ErrorResult<void> applySlide(streaming::Window &window, const streaming::Slide &slide,
                                     std::size_t point_dim)
{
    auto old_status = validateSlidePoint(slide.old_point, point_dim, "old streaming point");
    if (old_status.isError())
    {
        return old_status;
    }
    auto new_status = validateSlidePoint(slide.new_point, point_dim, "new streaming point");
    if (new_status.isError())
    {
        return new_status;
    }

    if (!slide.old_point.empty())
    {
        const auto old_it = std::ranges::find_if(
            window.points, [&](const auto &point) { return pointMatches(point, slide.old_point); });
        if (old_it == window.points.end())
        {
            return invalidInput("old streaming point is not present in the window");
        }
        window.points.erase(old_it);
    }

    if (!slide.new_point.empty())
    {
        window.points.push_back(slide.new_point);
    }

    if (window.max_size > 0)
    {
        while (window.points.size() > window.max_size)
        {
            window.points.erase(window.points.begin());
        }
    }

    if (slide.max_radius.has_value())
    {
        window.max_radius = *slide.max_radius;
    }

    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> buildDiagramFromWindow(ComputeManager &manager,
                                                 const streaming::Window &window,
                                                 persistence::Diagram &out_diagram)
{
    out_diagram.clear();
    if (window.points.empty())
    {
        return errors::ErrorResult<void>::success();
    }
    if (window.max_dimension > static_cast<Size>(std::numeric_limits<int>::max()))
    {
        return resourceLimit("streaming max dimension exceeds int range");
    }

    std::vector<ComputeManager::VRSimplex> simplices;
    auto build_status = manager.buildVRComplex(window.points, window.max_radius,
                                               static_cast<int>(window.max_dimension), simplices);
    if (build_status.isError())
    {
        return build_status;
    }

    algebra::SimplicialComplex complex;
    try
    {
        for (const auto &simplex : simplices)
        {
            std::vector<Index> vertices;
            vertices.reserve(simplex.vertices.size());
            for (int vertex : simplex.vertices)
            {
                if (vertex < 0)
                {
                    return invalidInput("streaming simplex vertex must be non-negative");
                }
                vertices.push_back(static_cast<Index>(vertex));
            }
            complex.addSimplexWithFiltration(algebra::Simplex(vertices), simplex.filtration_value);
        }
        const auto result = persistence::computeExactPersistenceZ2(complex, window.max_dimension);
        for (const auto &pair : result.pairs)
        {
            out_diagram.addPair(pair);
        }
    }
    catch (const std::invalid_argument &ex)
    {
        return invalidInput(ex.what());
    }
    catch (const std::length_error &ex)
    {
        return resourceLimit(ex.what());
    }
    catch (const std::exception &ex)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT, ex.what());
    }

    return errors::ErrorResult<void>::success();
}

} // namespace

errors::ErrorResult<void> ComputeManager::processWindowSlide(streaming::Window &window,
                                                             persistence::Diagram &out_diagram)
{
    constexpr const char *operation = "processWindowSlide";

    auto radius_status = validateRadius(window.max_radius);
    if (radius_status.isError())
    {
        recordFailure(operation, radius_status.error().message);
        return radius_status;
    }

    std::size_t point_dim = 0;
    auto point_status = validatePointRows(window.points, point_dim);
    if (point_status.isError())
    {
        recordFailure(operation, point_status.error().message);
        return point_status;
    }

    if (window.pending_slide.has_value())
    {
        const auto &slide = *window.pending_slide;
        auto affected_status = detectAffectedRegion(window, slide, window.last_affected_indices);
        if (affected_status.isError())
        {
            recordFailure(operation, affected_status.error().message);
            return affected_status;
        }
        auto apply_status = applySlide(window, slide, point_dim);
        if (apply_status.isError())
        {
            recordFailure(operation, apply_status.error().message);
            return apply_status;
        }
        window.pending_slide.reset();
    }
    else
    {
        window.last_affected_indices.clear();
    }

    auto diagram_status = buildDiagramFromWindow(*this, window, out_diagram);
    if (diagram_status.isError())
    {
        recordFailure(operation, diagram_status.error().message);
        return diagram_status;
    }

    recordSuccess(operation, 1.0);
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> ComputeManager::detectAffectedRegion(const streaming::Window &window,
                                                               const streaming::Slide &slide,
                                                               std::vector<int> &affected_indices)
{
    constexpr const char *operation = "detectAffectedRegion";
    affected_indices.clear();

    const double radius = slide.max_radius.value_or(window.max_radius);
    auto radius_status = validateRadius(radius);
    if (radius_status.isError())
    {
        recordFailure(operation, radius_status.error().message);
        return radius_status;
    }

    std::size_t point_dim = 0;
    auto point_status = validatePointRows(window.points, point_dim);
    if (point_status.isError())
    {
        recordFailure(operation, point_status.error().message);
        return point_status;
    }
    auto old_status = validateSlidePoint(slide.old_point, point_dim, "old streaming point");
    if (old_status.isError())
    {
        recordFailure(operation, old_status.error().message);
        return old_status;
    }
    auto new_status = validateSlidePoint(slide.new_point, point_dim, "new streaming point");
    if (new_status.isError())
    {
        recordFailure(operation, new_status.error().message);
        return new_status;
    }
    if (window.points.empty() || (slide.old_point.empty() && slide.new_point.empty()))
    {
        recordSuccess(operation, 1.0);
        return errors::ErrorResult<void>::success();
    }

    const double radius_sq = radius * radius;
    if (!std::isfinite(radius_sq))
    {
        const auto error = numericError("streaming radius squared overflow");
        recordFailure(operation, error.error().message);
        return error;
    }
    for (std::size_t index = 0; index < window.points.size(); ++index)
    {
        const auto &point = window.points[index];
        bool affected_by_old = false;
        if (!slide.old_point.empty())
        {
            const auto distance = squaredDistance(point, slide.old_point);
            if (distance.isError())
            {
                affected_indices.clear();
                recordFailure(operation, distance.error().message);
                return errors::ErrorResult<void>::error(distance.errorCode(),
                                                        distance.error().message);
            }
            affected_by_old = distance.value() <= radius_sq;
        }
        bool affected_by_new = false;
        if (!slide.new_point.empty())
        {
            const auto distance = squaredDistance(point, slide.new_point);
            if (distance.isError())
            {
                affected_indices.clear();
                recordFailure(operation, distance.error().message);
                return errors::ErrorResult<void>::error(distance.errorCode(),
                                                        distance.error().message);
            }
            affected_by_new = distance.value() <= radius_sq;
        }
        if (!affected_by_old && !affected_by_new)
        {
            continue;
        }
        int affected_index = 0;
        if (!checkedIntIndex(index, affected_index))
        {
            affected_indices.clear();
            const auto error = resourceLimit("streaming affected index exceeds int range");
            recordFailure(operation, error.error().message);
            return error;
        }
        affected_indices.push_back(affected_index);
    }

    recordSuccess(operation, 1.0);
    return errors::ErrorResult<void>::success();
}

} // namespace nerve::gpu
