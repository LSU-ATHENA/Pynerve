// Distance matrix computation for ComputeManager API.
// This implementation is intentionally deterministic and uses symmetric
// accumulation plus optional OpenMP row parallelism.

#include "nerve/gpu/kernel_launcher.hpp"

#include <atomic>
#include <cmath>
#include <limits>
#include <new>
#include <vector>

namespace nerve::gpu
{

errors::ErrorResult<void>
ComputeManager::computeDistanceMatrix(const std::vector<std::vector<double>> &points,
                                      std::vector<std::vector<double>> &out_distances)
{
    constexpr const char *operation = "computeDistanceMatrix";

    const size_t n_points = points.size();
    if (n_points != 0 && n_points > std::numeric_limits<size_t>::max() / n_points)
    {
        out_distances.clear();
        recordFailure(operation, "Distance matrix area overflows size_t");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "distance matrix area overflows size_t");
    }

    (void)selectStrategy(OperationType::kDistanceMatrix, n_points);

    if (n_points == 0)
    {
        out_distances.clear();
        recordSuccess(operation, 1.0);
        return errors::ErrorResult<void>::success();
    }

    const size_t point_dim = points.front().size();
    if (point_dim == 0)
    {
        out_distances.clear();
        recordFailure(operation, "Points must have positive dimension");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT);
    }
    for (const auto &point : points)
    {
        if (point.size() != point_dim)
        {
            out_distances.clear();
            recordFailure(operation, "Point dimensions must be uniform");
            return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT);
        }
        for (double value : point)
        {
            if (!std::isfinite(value))
            {
                out_distances.clear();
                recordFailure(operation, "Point coordinates must be finite");
                return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN,
                                                        "point coordinates must be finite");
            }
        }
    }

    try
    {
        out_distances.assign(n_points, std::vector<double>(n_points, 0.0));
    }
    catch (const std::bad_alloc &)
    {
        out_distances.clear();
        recordFailure(operation, "Distance matrix allocation failed");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM,
                                                "distance matrix allocation failed");
    }

    std::atomic_bool overflowed_distance{false};

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n_points); ++i)
    {
        if (overflowed_distance.load(std::memory_order_relaxed))
        {
            continue;
        }
        out_distances[i][i] = 0.0;
        for (size_t j = i + 1; j < n_points; ++j)
        {
            if (overflowed_distance.load(std::memory_order_relaxed))
            {
                continue;
            }
            double dist_sq = 0.0;
            bool pair_overflowed = false;
            for (size_t d = 0; d < point_dim; ++d)
            {
                const double diff = points[i][d] - points[j][d];
                if (!std::isfinite(diff))
                {
                    pair_overflowed = true;
                    break;
                }
                const double contribution = diff * diff;
                if (!std::isfinite(contribution) ||
                    dist_sq > std::numeric_limits<double>::max() - contribution)
                {
                    pair_overflowed = true;
                    break;
                }
                dist_sq += contribution;
            }
            const double dist = std::sqrt(dist_sq);
            if (pair_overflowed || !std::isfinite(dist))
            {
                overflowed_distance.store(true, std::memory_order_relaxed);
                continue;
            }
            out_distances[i][j] = dist;
            out_distances[j][i] = dist;
        }
    }

    if (overflowed_distance.load(std::memory_order_relaxed))
    {
        out_distances.clear();
        recordFailure(operation, "Distance matrix value overflow");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN,
                                                "distance matrix value overflow");
    }

    recordSuccess(operation, 1.0);
    return errors::ErrorResult<void>::success();
}

} // namespace nerve::gpu
