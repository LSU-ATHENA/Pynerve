#pragma once
#include "nerve/core_types.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace nerve::persistence::vram
{

enum class Algorithm
{
    FULL_GPU,
    CHUNKED,
    STREAMING,
    HYBRID,
    UNIFIED_MEMORY
};

struct VRAMConfig
{
    std::size_t total_vram_bytes = 0;
    std::size_t available_vram_bytes = 0;
    double safety_fraction = 0.8;

    [[nodiscard]] inline std::size_t safeBytes() const noexcept
    {
        const std::size_t capped = total_vram_bytes == 0
                                       ? available_vram_bytes
                                       : std::min(available_vram_bytes, total_vram_bytes);
        if (capped == 0 || safety_fraction <= 0.0)
            return 0;
        if (safety_fraction >= 1.0)
            return capped;
        return static_cast<std::size_t>(static_cast<long double>(capped) *
                                        static_cast<long double>(safety_fraction));
    }
    [[nodiscard]] inline Algorithm select(std::size_t n_points,
                                          std::size_t point_dim) const noexcept
    {
        if (n_points == 0 || point_dim == 0)
            return Algorithm::HYBRID;
        const std::size_t usable = safeBytes();
        constexpr std::size_t gb = 1024ULL * 1024ULL * 1024ULL;
        constexpr std::size_t mb_256 = 256ULL * 1024ULL * 1024ULL;
        if (usable >= gb)
            return Algorithm::FULL_GPU;
        if (usable >= mb_256)
            return Algorithm::CHUNKED;
        return Algorithm::STREAMING;
    }
};

[[nodiscard]] inline Algorithm selectAlgorithm(std::size_t n_points, std::size_t point_dim,
                                               std::size_t available_vram_bytes) noexcept
{
    VRAMConfig config{};
    config.total_vram_bytes = available_vram_bytes;
    config.available_vram_bytes = available_vram_bytes;
    return config.select(n_points, point_dim);
}

[[nodiscard]] PersistenceDiagram computeVRAMAware(std::span<const double> points,
                                                  std::size_t n_points, std::size_t point_dim,
                                                  double max_distance,
                                                  std::size_t available_vram_bytes);

template <typename T = double>
[[nodiscard]] inline PersistenceDiagram
computeVRAMAware(std::span<const T> points, std::size_t n_points, std::size_t point_dim,
                 T max_distance, std::size_t available_vram_bytes)
{
    if constexpr (std::is_same_v<T, double>)
    {
        return computeVRAMAware(std::span<const double>(points.data(), points.size()), n_points,
                                point_dim, max_distance, available_vram_bytes);
    }
    else
    {
        std::vector<double> converted(points.size());
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            converted[i] = static_cast<double>(points[i]);
        }
        return computeVRAMAware(std::span<const double>(converted.data(), converted.size()),
                                n_points, point_dim, static_cast<double>(max_distance),
                                available_vram_bytes);
    }
}

} // namespace nerve::persistence::vram
