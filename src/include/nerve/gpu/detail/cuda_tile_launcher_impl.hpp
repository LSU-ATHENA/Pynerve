#pragma once

#include "nerve/gpu/detail/cuda_tile_detail_impl.hpp"

namespace nerve::gpu::tile
{

// Inline Implementation: TileLauncher

inline TileLauncher::TileLauncher(const TileConfig &config)
    : config_(config)
    , kernelArgs_{}
{}

template <typename T>
inline cudaError_t TileLauncher::launchDistanceMatrix(const TileBuffer<T> &points,
                                                      TileBuffer<T> &distances, float maxRadius,
                                                      cudaStream_t stream)
{
    const int n_points = points.rows();
    const int point_dim = points.cols();
    if (n_points <= 0 || point_dim <= 0 || points.data() == nullptr || distances.data() == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    if (distances.rows() != n_points || distances.cols() != n_points)
    {
        return cudaErrorInvalidValue;
    }
    if (!std::isfinite(maxRadius))
    {
        return cudaErrorInvalidValue;
    }

    // Non-full precision modes are implemented via host-side quantization.
    if (config_.precision == TilePrecision::kFull)
    {
        const cudaError_t device_status = detail::launchDistanceMatrixDeviceBackend<T>(
            points.data(), n_points, point_dim, distances.data(), maxRadius, stream);
        if (device_status == cudaSuccess)
        {
            return cudaStreamSynchronize(stream);
        }
    }

    std::vector<T> host_points(static_cast<size_t>(n_points) * static_cast<size_t>(point_dim));
    std::vector<T> host_distances(static_cast<size_t>(n_points) * static_cast<size_t>(n_points),
                                  T{0});
    if (cudaMemcpyAsync(host_points.data(), points.data(), host_points.size() * sizeof(T),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    (void)cudaStreamSynchronize(stream);

    detail::computeSymmetricDistancesHostTiled(
        host_points.data(), static_cast<uint32_t>(n_points), static_cast<uint32_t>(point_dim),
        maxRadius, config_.precision, config_.tileSizeM, config_.tileSizeN, host_distances);

    if (cudaMemcpyAsync(distances.data(), host_distances.data(), host_distances.size() * sizeof(T),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    return cudaStreamSynchronize(stream);
}

template <typename T>
inline cudaError_t TileLauncher::launchReduction(const TileBuffer<T> &input, TileBuffer<T> &output,
                                                 cudaStream_t stream)
{
    if (input.data() == nullptr || output.data() == nullptr || output.size() == 0)
    {
        return cudaErrorInvalidValue;
    }
    std::vector<T> host_input(input.size());
    if (cudaMemcpyAsync(host_input.data(), input.data(), host_input.size() * sizeof(T),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    (void)cudaStreamSynchronize(stream);
    T sum = T{0};
    for (const T value : host_input)
    {
        sum += value;
    }
    if (cudaMemcpyAsync(output.data(), &sum, sizeof(T), cudaMemcpyHostToDevice, stream) !=
        cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    return cudaStreamSynchronize(stream);
}

template <typename T>
inline cudaError_t TileLauncher::launchTranspose(const TileBuffer<T> &input, TileBuffer<T> &output,
                                                 cudaStream_t stream)
{
    if (input.data() == nullptr || output.data() == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    if (output.rows() != input.cols() || output.cols() != input.rows())
    {
        return cudaErrorInvalidValue;
    }
    std::vector<T> host_input(input.size());
    std::vector<T> host_output(output.size(), T{0});
    if (cudaMemcpyAsync(host_input.data(), input.data(), host_input.size() * sizeof(T),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    (void)cudaStreamSynchronize(stream);
    for (int r = 0; r < input.rows(); ++r)
    {
        for (int c = 0; c < input.cols(); ++c)
        {
            host_output[static_cast<size_t>(c) * static_cast<size_t>(output.cols()) +
                        static_cast<size_t>(r)] =
                host_input[static_cast<size_t>(r) * static_cast<size_t>(input.cols()) +
                           static_cast<size_t>(c)];
        }
    }
    if (cudaMemcpyAsync(output.data(), host_output.data(), host_output.size() * sizeof(T),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    return cudaStreamSynchronize(stream);
}

template <typename T, typename Func>
inline cudaError_t TileLauncher::launchTileOp(const TileBuffer<T> &input, TileBuffer<T> &output,
                                              Func operation, cudaStream_t stream)
{
    if (input.data() == nullptr || output.data() == nullptr || input.size() != output.size())
    {
        return cudaErrorInvalidValue;
    }
    std::vector<T> host_input(input.size());
    std::vector<T> host_output(output.size());
    if (cudaMemcpyAsync(host_input.data(), input.data(), host_input.size() * sizeof(T),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    (void)cudaStreamSynchronize(stream);
    for (size_t i = 0; i < host_input.size(); ++i)
    {
        host_output[i] = operation(host_input[i]);
    }
    if (cudaMemcpyAsync(output.data(), host_output.data(), host_output.size() * sizeof(T),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    return cudaStreamSynchronize(stream);
}

inline size_t TileLauncher::getSharedMemSize() const
{
    return config_.sharedMemSize();
}

inline dim3 TileLauncher::getGridDim(int rows, int cols) const
{
    if (rows <= 0 || cols <= 0 || config_.tileSizeM <= 0 || config_.tileSizeN <= 0)
    {
        return dim3(0, 0, 0);
    }
    const auto ceil_div = [](int value, int tile) -> unsigned {
        const size_t numerator = static_cast<size_t>(value) + static_cast<size_t>(tile) - 1U;
        const size_t result = numerator / static_cast<size_t>(tile);
        return result > std::numeric_limits<unsigned>::max() ? std::numeric_limits<unsigned>::max()
                                                             : static_cast<unsigned>(result);
    };
    const unsigned gx = ceil_div(cols, config_.tileSizeN);
    const unsigned gy = ceil_div(rows, config_.tileSizeM);
    return dim3(gx, gy, 1);
}

inline dim3 TileLauncher::getBlockDim() const
{
    return detail::selectThreadBlockDim(config_);
}

} // namespace nerve::gpu::tile
