#pragma once

#define NERVE_CUDA_TILE_API_DECLARATIONS_ONLY
#include "nerve/gpu/cuda_tile_api.hpp"
#undef NERVE_CUDA_TILE_API_DECLARATIONS_ONLY

namespace nerve::gpu::tile
{

namespace detail
{
inline std::unordered_map<std::string, TileConfig> &tileConfigCache()
{
    static std::unordered_map<std::string, TileConfig> cache;
    return cache;
}

inline int selectDevice(int deviceId)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
    {
        return 0;
    }
    const int selected = std::max(0, std::min(deviceId, device_count - 1));
    (void)cudaSetDevice(selected);
    return selected;
}

inline cudaDeviceProp queryDeviceProperties(int deviceId)
{
    cudaDeviceProp properties{};
    const int selected = selectDevice(deviceId);
    if (cudaGetDeviceProperties(&properties, selected) != cudaSuccess)
    {
        properties.major = 0;
        properties.minor = 0;
        properties.multiProcessorCount = 1;
        properties.warpSize = 32;
        properties.maxThreadsPerBlock = 1024;
        properties.totalGlobalMem = 0;
    }
    if (properties.warpSize <= 0)
    {
        properties.warpSize = 32;
    }
    if (properties.maxThreadsPerBlock <= 0)
    {
        properties.maxThreadsPerBlock = 1024;
    }
    if (properties.multiProcessorCount <= 0)
    {
        properties.multiProcessorCount = 1;
    }
    return properties;
}

inline dim3 selectThreadBlockDim(const TileConfig &config)
{
    const unsigned tile_cols = static_cast<unsigned>(std::max(1, config.tileSizeN));
    const unsigned tile_rows = static_cast<unsigned>(std::max(1, config.tileSizeM));
    const unsigned block_x = std::min(32U, tile_cols);
    const unsigned max_y = std::max(1U, 256U / block_x);
    const unsigned block_y = std::min(tile_rows, max_y);
    return dim3(block_x, block_y, 1);
}

inline float estimateOccupancy(const TileConfig &config, const cudaDeviceProp &prop)
{
    const dim3 block = selectThreadBlockDim(config);
    const unsigned threads = std::max(1U, block.x * block.y * block.z);
    const unsigned max_threads = static_cast<unsigned>(std::max(1, prop.maxThreadsPerBlock));
    const float thread_fraction =
        std::min(1.0f, static_cast<float>(threads) / static_cast<float>(max_threads));
    const float cluster_fraction =
        1.0f / static_cast<float>(std::max(1, config.totalClusterSize()));
    return std::max(0.05f, std::min(1.0f, thread_fraction * cluster_fraction));
}

template <typename T>
inline T clipDistanceByRadius(T distance, float maxRadius)
{
    if (!std::isfinite(static_cast<double>(distance)))
    {
        if constexpr (std::numeric_limits<T>::has_infinity)
        {
            return std::numeric_limits<T>::infinity();
        }
        return T{};
    }
    if (!(maxRadius > 0.0f))
    {
        return distance;
    }
    if (static_cast<double>(distance) <= static_cast<double>(maxRadius))
    {
        return distance;
    }
    if constexpr (std::numeric_limits<T>::has_infinity)
    {
        return std::numeric_limits<T>::infinity();
    }
    return static_cast<T>(maxRadius);
}

inline double quantizeDifference(double diff, TilePrecision precision)
{
    switch (precision)
    {
        case TilePrecision::kUltraLow:
            return std::round(diff * 64.0) / 64.0;
        case TilePrecision::kLow:
            return std::round(diff * 256.0) / 256.0;
        case TilePrecision::kMixed:
            // cppcheck-suppress suspiciousFloatingPointCast
            return static_cast<double>(static_cast<float>(diff));
        case TilePrecision::kFull:
        default:
            return diff;
    }
}

inline bool accumulateQuantizedDifference(double diff, TilePrecision precision, double &sum)
{
    const double quantized = quantizeDifference(diff, precision);
    const double contribution = quantized * quantized;
    const double next = sum + contribution;
    if (!std::isfinite(quantized) || !std::isfinite(contribution) || !std::isfinite(next))
    {
        return false;
    }
    sum = next;
    return true;
}

inline bool checkedTileOffset(size_t pitch, int rowStart, int colStart, size_t &offset)
{
    if (rowStart < 0 || colStart < 0 || pitch == 0)
    {
        return false;
    }

    const size_t row = static_cast<size_t>(rowStart);
    const size_t col = static_cast<size_t>(colStart);
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (row != 0 && pitch > (max_size - col) / row)
    {
        return false;
    }

    offset = row * pitch + col;
    return true;
}

inline bool checkedTileRegion(int totalRows, int totalCols, size_t pitch, int rowStart,
                              int colStart, int rows, int cols, size_t &offset)
{
    if (totalRows <= 0 || totalCols <= 0 || rows <= 0 || cols <= 0)
    {
        return false;
    }
    if (rowStart < 0 || colStart < 0 || rowStart >= totalRows || colStart >= totalCols)
    {
        return false;
    }
    if (rows > totalRows - rowStart || cols > totalCols - colStart)
    {
        return false;
    }
    return checkedTileOffset(pitch, rowStart, colStart, offset);
}

template <typename T>
inline void computeSymmetricDistancesHostTiled(const T *points, uint32_t nPoints, uint32_t pointDim,
                                               float maxRadius, TilePrecision precision,
                                               int tileRows, int tileCols,
                                               std::vector<T> &hostDistances)
{
    hostDistances.assign(static_cast<size_t>(nPoints) * static_cast<size_t>(nPoints), T{0});
    const uint32_t row_tile = std::max<uint32_t>(1U, static_cast<uint32_t>(tileRows));
    const uint32_t col_tile = std::max<uint32_t>(1U, static_cast<uint32_t>(tileCols));
    const int tile_count = static_cast<int>((nPoints + row_tile - 1U) / row_tile);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int tile_i = 0; tile_i < tile_count; ++tile_i)
    {
        const uint32_t i0 = static_cast<uint32_t>(tile_i) * row_tile;
        const uint32_t i_end = std::min(nPoints, i0 + row_tile);
        for (uint32_t j0 = i0; j0 < nPoints; j0 += col_tile)
        {
            const uint32_t j_end = std::min(nPoints, j0 + col_tile);
            for (uint32_t i = i0; i < i_end; ++i)
            {
                const uint32_t j_start = std::max(i, j0);
                for (uint32_t j = j_start; j < j_end; ++j)
                {
                    const size_t off_i = static_cast<size_t>(i) * pointDim;
                    const size_t off_j = static_cast<size_t>(j) * pointDim;
                    double sum = 0.0;
                    bool valid_distance = true;

                    if (pointDim <= 4U)
                    {
#if defined(__CUDACC__)
#pragma unroll
#endif
                        for (uint32_t k = 0; k < pointDim; ++k)
                        {
                            const double diff =
                                static_cast<double>(points[off_i + static_cast<size_t>(k)] -
                                                    points[off_j + static_cast<size_t>(k)]);
                            if (!accumulateQuantizedDifference(diff, precision, sum))
                            {
                                valid_distance = false;
                                break;
                            }
                        }
                    }
                    else
                    {
                        for (uint32_t k = 0; k < pointDim; ++k)
                        {
                            const double diff =
                                static_cast<double>(points[off_i + static_cast<size_t>(k)] -
                                                    points[off_j + static_cast<size_t>(k)]);
                            if (!accumulateQuantizedDifference(diff, precision, sum))
                            {
                                valid_distance = false;
                                break;
                            }
                        }
                    }

                    const T raw_distance = valid_distance
                                               ? static_cast<T>(std::sqrt(sum))
                                               : (std::numeric_limits<T>::has_infinity
                                                      ? std::numeric_limits<T>::infinity()
                                                      : T{});
                    const T d = clipDistanceByRadius(raw_distance, maxRadius);
                    hostDistances[static_cast<size_t>(i) * nPoints + static_cast<size_t>(j)] = d;
                    hostDistances[static_cast<size_t>(j) * nPoints + static_cast<size_t>(i)] = d;
                }
            }
        }
    }
}

template <typename T>
inline cudaError_t launchDistanceMatrixDeviceBackend(const T *, int, int, T *, float, cudaStream_t)
{
    return cudaErrorInvalidValue;
}

template <>
inline cudaError_t launchDistanceMatrixDeviceBackend<float>(const float *d_points, int n_points,
                                                            int point_dim, float *d_distances,
                                                            float max_radius, cudaStream_t stream)
{
    return launch_pairwise_distance_radius_f32(d_points, point_dim, d_distances, n_points, n_points,
                                               point_dim, max_radius, stream);
}

template <>
inline cudaError_t launchDistanceMatrixDeviceBackend<double>(const double *d_points, int n_points,
                                                             int point_dim, double *d_distances,
                                                             float max_radius, cudaStream_t stream)
{
    return launch_pairwise_distance_radius_f64(d_points, point_dim, d_distances, n_points, n_points,
                                               point_dim, static_cast<double>(max_radius), stream);
}
} // namespace detail

} // namespace nerve::gpu::tile
