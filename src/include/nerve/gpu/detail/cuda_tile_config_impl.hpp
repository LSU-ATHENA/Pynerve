#pragma once

#define NERVE_CUDA_TILE_API_DECLARATIONS_ONLY
#include "nerve/gpu/cuda_tile_api.hpp"
#undef NERVE_CUDA_TILE_API_DECLARATIONS_ONLY

namespace nerve::gpu::tile
{

// Inline Implementation: TileConfig

inline bool TileConfig::isValid() const
{
    if (tileSizeM <= 0 || tileSizeN <= 0 || tileSizeK <= 0)
    {
        return false;
    }
    if (clusterSizeX <= 0 || clusterSizeY <= 0 || clusterSizeZ <= 0)
    {
        return false;
    }

    const int clusterSize = totalClusterSize();
    if (clusterSize <= 0 || clusterSize > 16)
    {
        return false;
    }
    return true;
}

inline size_t TileConfig::sharedMemSize() const
{
    if (tileSizeM <= 0 || tileSizeN <= 0 || tileSizeK <= 0)
    {
        return 0;
    }

    const size_t m = static_cast<size_t>(tileSizeM);
    const size_t n = static_cast<size_t>(tileSizeN);
    const size_t k = static_cast<size_t>(tileSizeK);
    const size_t max_elements = std::numeric_limits<size_t>::max() / sizeof(float);
    if (m > max_elements / k || k > max_elements / n || m > max_elements / n)
    {
        return std::numeric_limits<size_t>::max();
    }

    const size_t elements = (m * k) + (k * n) + (m * n);
    if (elements > max_elements)
    {
        return std::numeric_limits<size_t>::max();
    }
    return elements * sizeof(float);
}

// Inline Implementation: DistanceTileConfig

inline TileConfig DistanceTileConfig::toTileConfig(TileDataType dtype) const
{
    TileConfig config;
    config.tileSizeM = pointTileSize;
    config.tileSizeN = pointTileSize;
    config.tileSizeK = pointDim;
    config.clusterSizeX = useClustering ? clusterSize : 1;
    config.dataType = dtype;
    config.useTMA = false;
    return config;
}

} // namespace nerve::gpu::tile
