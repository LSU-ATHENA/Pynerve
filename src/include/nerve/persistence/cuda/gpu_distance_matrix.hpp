
#pragma once

#include "nerve/core.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/gpu/cuda_tile_api.hpp"
#include "nerve/persistence/cuda/cuda_edge_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nerve::persistence::accelerated
{

class GPUDistanceMatrix
{
public:
    struct Config
    {
        size_t max_points = 200000;
        size_t max_dimension = 4096;
    };

    static errors::ErrorResult<std::unique_ptr<GPUDistanceMatrix>> create()
    {
        return create(Config{});
    }

    static errors::ErrorResult<std::unique_ptr<GPUDistanceMatrix>> create(const Config &config)
    {
        if (config.max_points == 0 || config.max_dimension == 0)
        {
            return errors::ErrorResult<std::unique_ptr<GPUDistanceMatrix>>::error(
                errors::ErrorCode::E52_PH_CONFIG);
        }
        return errors::ErrorResult<std::unique_ptr<GPUDistanceMatrix>>::success(
            std::unique_ptr<GPUDistanceMatrix>(new GPUDistanceMatrix(config)));
    }

    errors::ErrorResult<void> computeDistances(const core::BufferView<const double> &points,
                                               size_t n_points, size_t point_dim, double max_radius,
                                               std::vector<double> &distances) const
    {
        distances.clear();

        size_t point_values = 0;
        size_t matrix_values = 0;
        if (point_dim == 0 || n_points == 0 || points.data() == nullptr ||
            !std::isfinite(max_radius) || !(max_radius > 0.0))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT);
        }
        if (!checkedProduct(n_points, point_dim, point_values) ||
            !checkedProduct(n_points, n_points, matrix_values))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        if (points.size() != point_values)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT);
        }
        if (n_points > config_.max_points || point_dim > config_.max_dimension)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        if (n_points > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            point_dim > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            matrix_values > std::numeric_limits<size_t>::max() / sizeof(double))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        if (!std::all_of(points.begin(), points.end(),
                         [](double value) { return std::isfinite(value); }))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN);
        }
        if (!nerve::gpu::tile::tileApiAvailable())
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        try
        {
            distances.assign(matrix_values, 0.0);
        }
        catch (const std::bad_alloc &)
        {
            distances.clear();
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }
        catch (const std::length_error &)
        {
            distances.clear();
            return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
        }

        try
        {
            nerve::gpu::tile::TileBuffer<double> device_points(static_cast<int>(n_points),
                                                               static_cast<int>(point_dim));
            nerve::gpu::tile::TileBuffer<double> device_distances(static_cast<int>(n_points),
                                                                  static_cast<int>(n_points));
            if (device_points.data() == nullptr || device_distances.data() == nullptr)
            {
                distances.clear();
                return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
            }
            device_points.copyFromHost(points.data());
            const cudaError_t launch_status =
                nerve::gpu::tile::detail::launchDistanceMatrixDeviceBackend<double>(
                    device_points.data(), static_cast<int>(n_points), static_cast<int>(point_dim),
                    device_distances.data(), static_cast<float>(max_radius), nullptr);
            if (launch_status != cudaSuccess || cudaStreamSynchronize(nullptr) != cudaSuccess)
            {
                distances.clear();
                return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
            }
            device_distances.copyToHost(distances.data());
        }
        catch (const std::runtime_error &)
        {
            distances.clear();
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
        return errors::ErrorResult<void>::success();
    }

private:
    explicit GPUDistanceMatrix(Config config)
        : config_(std::move(config))
    {}

    static bool checkedProduct(size_t lhs, size_t rhs, size_t &out) noexcept
    {
        if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        {
            return false;
        }
        out = lhs * rhs;
        return true;
    }

    Config config_;
};

} // namespace nerve::persistence::accelerated
