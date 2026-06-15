#include "nerve/gpu/kernel_launcher.hpp"
#include "nerve/gpu/manager_compute_ops_filtration.hpp"
#include "nerve/gpu/manager_compute_ops_hungarian.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <utility>
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

} // namespace

errors::ErrorResult<void>
ComputeManager::constructLaplacian(const std::vector<std::vector<double>> &points,
                                   const std::vector<std::vector<int>> &edges,
                                   std::vector<std::vector<double>> &out_laplacian)
{
    constexpr const char *operation = "constructLaplacian";
    std::vector<double> flat_points;
    std::size_t point_dim = 0;
    auto point_status = flattenPointRows(points, flat_points, point_dim);
    if (point_status.isError())
    {
        out_laplacian.clear();
        recordFailure(operation, point_status.error().message);
        return point_status;
    }
    (void)flat_points;
    (void)point_dim;

    const std::size_t n = points.size();
    try
    {
        out_laplacian.assign(n, std::vector<double>(n, 0.0));
    }
    catch (const std::bad_alloc &)
    {
        out_laplacian.clear();
        recordFailure(operation, "Laplacian allocation failed");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM,
                                                "Laplacian allocation failed");
    }

    for (const auto &edge : edges)
    {
        if (edge.size() < 2 || edge[0] < 0 || edge[1] < 0 ||
            static_cast<std::size_t>(edge[0]) >= n || static_cast<std::size_t>(edge[1]) >= n)
        {
            out_laplacian.clear();
            recordFailure(operation, "Invalid Laplacian edge endpoint");
            return invalidInput("invalid Laplacian edge endpoint");
        }
        const std::size_t u = static_cast<std::size_t>(edge[0]);
        const std::size_t v = static_cast<std::size_t>(edge[1]);
        if (u == v)
        {
            continue;
        }
        out_laplacian[u][u] += 1.0;
        out_laplacian[v][v] += 1.0;
        out_laplacian[u][v] -= 1.0;
        out_laplacian[v][u] -= 1.0;
    }

    recordSuccess(operation, 1.0);
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void>
ComputeManager::buildVRComplex(const std::vector<std::vector<double>> &points, double max_radius,
                               int max_dimension, std::vector<VRSimplex> &out_simplices)
{
    constexpr const char *operation = "buildVRComplex";
    std::vector<std::pair<algebra::Simplex, double>> filtration;
    auto filtration_status =
        buildVietorisRipsFiltration(points, max_radius, max_dimension, filtration);
    if (filtration_status.isError())
    {
        out_simplices.clear();
        recordFailure(operation, filtration_status.error().message);
        return filtration_status;
    }
    auto convert_status = convertVrFiltration(filtration, out_simplices);
    if (convert_status.isError())
    {
        recordFailure(operation, convert_status.error().message);
        return convert_status;
    }
    recordSuccess(operation, 1.0);
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void>
ComputeManager::buildCechComplex(const std::vector<std::vector<double>> &points, double max_radius,
                                 int max_dimension, std::vector<CechSimplex> &out_simplices)
{
    constexpr const char *operation = "buildCechComplex";
    auto result = buildCechLikeComplex(points, max_radius, max_dimension, out_simplices);
    if (result.isError())
    {
        recordFailure(operation, result.error().message);
        return result;
    }
    recordSuccess(operation, 1.0);
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void>
ComputeManager::buildAlphaComplex(const std::vector<std::vector<double>> &points, double max_alpha,
                                  int max_dimension, std::vector<CechSimplex> &out_simplices)
{
    constexpr const char *operation = "buildAlphaComplex";
    auto result = buildCechLikeComplex(points, max_alpha, max_dimension, out_simplices);
    if (result.isError())
    {
        recordFailure(operation, result.error().message);
        return result;
    }
    recordSuccess(operation, 1.0);
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<double>
ComputeManager::solveAssignment(const std::vector<std::vector<double>> &cost_matrix,
                                std::vector<std::pair<int, int>> &out_assignment)
{
    constexpr const char *operation = "solveAssignment";
    auto validation = validateSquareCostMatrix(cost_matrix, false);
    if (validation.isError())
    {
        out_assignment.clear();
        recordFailure(operation, validation.error().message);
        return errors::ErrorResult<double>::error(validation.errorCode(),
                                                  validation.error().message);
    }
    auto result = solveAssignmentHungarian(cost_matrix, out_assignment);
    if (result.isError())
    {
        recordFailure(operation, result.error().message);
        return result;
    }
    recordSuccess(operation, 1.0);
    return result;
}

errors::ErrorResult<double>
ComputeManager::solveBottleneck(const std::vector<std::vector<double>> &cost_matrix,
                                std::vector<std::pair<int, int>> &out_assignment)
{
    constexpr const char *operation = "solveBottleneck";
    auto validation = validateSquareCostMatrix(cost_matrix, true);
    if (validation.isError())
    {
        out_assignment.clear();
        recordFailure(operation, validation.error().message);
        return errors::ErrorResult<double>::error(validation.errorCode(),
                                                  validation.error().message);
    }
    if (cost_matrix.empty())
    {
        out_assignment.clear();
        recordSuccess(operation, 1.0);
        return errors::ErrorResult<double>::success(0.0);
    }

    std::vector<double> candidates;
    candidates.reserve(cost_matrix.size() * cost_matrix.size());
    for (const auto &row : cost_matrix)
    {
        for (double value : row)
        {
            if (std::isfinite(value))
            {
                candidates.push_back(value);
            }
        }
    }
    if (candidates.empty())
    {
        out_assignment.clear();
        recordSuccess(operation, 1.0);
        return errors::ErrorResult<double>::success(std::numeric_limits<double>::infinity());
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    std::size_t lo = 0;
    std::size_t hi = candidates.size() - 1;
    double best = candidates.back();
    while (lo <= hi)
    {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (thresholdPerfectMatching(cost_matrix, candidates[mid], nullptr))
        {
            best = candidates[mid];
            if (mid == 0)
            {
                break;
            }
            hi = mid - 1;
        }
        else
        {
            lo = mid + 1;
        }
    }
    if (!thresholdPerfectMatching(cost_matrix, best, &out_assignment))
    {
        out_assignment.clear();
        recordFailure(operation, "No bottleneck perfect matching found");
        return errors::ErrorResult<double>::error(errors::ErrorCode::E21_NUM_NO_CONVERGE,
                                                  "no bottleneck perfect matching found");
    }
    recordSuccess(operation, 1.0);
    return errors::ErrorResult<double>::success(double{best});
}

errors::ErrorResult<void>
ComputeManager::applyClearing(const algebra::BoundaryMatrix &boundary_matrix,
                              const std::vector<int> &simplex_dimensions,
                              const std::vector<double> &filtration_values, int target_dimension,
                              double max_filtration, ClearingResult &out_result)
{
    constexpr const char *operation = "applyClearing";
    out_result = ClearingResult{};
    if (target_dimension < 0 || !std::isfinite(max_filtration))
    {
        recordFailure(operation, "Invalid clearing dimension or filtration bound");
        return invalidInput("invalid clearing dimension or filtration bound");
    }
    if (simplex_dimensions.size() != boundary_matrix.cols())
    {
        recordFailure(operation, "Clearing dimension count must match boundary matrix columns");
        return invalidInput("clearing dimension count must match boundary matrix columns");
    }
    if (!filtration_values.empty() && filtration_values.size() != simplex_dimensions.size())
    {
        recordFailure(operation, "Clearing filtration count must match simplex dimensions");
        return invalidInput("clearing filtration count must match simplex dimensions");
    }
    for (double value : filtration_values)
    {
        if (!std::isfinite(value))
        {
            recordFailure(operation, "Clearing filtration values must be finite");
            return numericError("clearing filtration values must be finite");
        }
    }

    const std::size_t cols = simplex_dimensions.size();
    out_result.positive_simplices.assign(cols, false);
    out_result.columns_to_clear.assign(cols, false);
    out_result.operations_saved = 0;

    for (std::size_t col = 0; col < cols; ++col)
    {
        bool has_boundary = false;
        std::size_t nonzeros = 0;
        for (Size row = 0; row < boundary_matrix.rows(); ++row)
        {
            if (boundary_matrix.getMatrixEntry(row, col) != 0.0)
            {
                has_boundary = true;
                ++nonzeros;
            }
        }
        const bool positive = simplex_dimensions[col] == 0 || !has_boundary;
        const bool within_filtration =
            filtration_values.empty() || filtration_values[col] <= max_filtration;
        const bool clear_column =
            within_filtration && positive && simplex_dimensions[col] == target_dimension;
        out_result.positive_simplices[col] = within_filtration && positive;
        out_result.columns_to_clear[col] = clear_column;
        if (clear_column)
        {
            const std::size_t saved = nonzeros <= std::numeric_limits<std::size_t>::max() /
                                                      std::max<std::size_t>(1, nonzeros)
                                          ? nonzeros * nonzeros
                                          : std::numeric_limits<std::size_t>::max();
            out_result.operations_saved =
                saved > std::numeric_limits<std::size_t>::max() - out_result.operations_saved
                    ? std::numeric_limits<std::size_t>::max()
                    : out_result.operations_saved + saved;
        }
    }

    recordSuccess(operation, 1.0);
    return errors::ErrorResult<void>::success();
}

} // namespace nerve::gpu
