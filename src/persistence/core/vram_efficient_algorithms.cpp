#include "nerve/cpu/simd.hpp"
#include "nerve/persistence/memory/vram_algorithms.hpp"

#include <algorithm>
#include <limits>

namespace nerve::persistence::vram
{
namespace
{

[[nodiscard]] bool hasPointBuffer(std::span<const double> points, const std::size_t n_points,
                                  const std::size_t point_dim) noexcept
{
    if (n_points == 0 || point_dim == 0)
    {
        return false;
    }
    if (n_points > std::numeric_limits<std::size_t>::max() / point_dim)
    {
        return false;
    }
    return points.size() >= n_points * point_dim;
}

} // namespace

std::size_t VRAMConfig::safeBytes() const noexcept
{
    const std::size_t capped_available = total_vram_bytes == 0
                                             ? available_vram_bytes
                                             : std::min(available_vram_bytes, total_vram_bytes);
    if (capped_available == 0 || safety_fraction <= 0.0)
    {
        return 0;
    }
    if (safety_fraction >= 1.0)
    {
        return capped_available;
    }
    return static_cast<std::size_t>(static_cast<long double>(capped_available) *
                                    static_cast<long double>(safety_fraction));
}

Algorithm VRAMConfig::select(const std::size_t n_points, const std::size_t point_dim) const noexcept
{
    if (n_points == 0 || point_dim == 0)
    {
        return Algorithm::HYBRID;
    }

    const std::size_t usable = safeBytes();
    constexpr std::size_t gb = 1024ULL * 1024ULL * 1024ULL;
    constexpr std::size_t mb_256 = 256ULL * 1024ULL * 1024ULL;

#ifdef NERVE_HAS_CUDA
    if (usable >= gb)
    {
        return Algorithm::FULL_GPU;
    }
    if (usable >= mb_256)
    {
        return Algorithm::CHUNKED;
    }
    return Algorithm::STREAMING;
#else
    if (usable >= mb_256)
    {
        return Algorithm::CHUNKED;
    }
    return Algorithm::STREAMING;
#endif
}

Algorithm selectAlgorithm(const std::size_t n_points, const std::size_t point_dim,
                          const std::size_t available_vram_bytes) noexcept
{
    VRAMConfig config{};
    config.total_vram_bytes = available_vram_bytes;
    config.available_vram_bytes = available_vram_bytes;
    return config.select(n_points, point_dim);
}

PersistenceDiagram computeVRAMAware(std::span<const double> points, const std::size_t n_points,
                                    const std::size_t point_dim, const double max_distance,
                                    const std::size_t available_vram_bytes)
{
    if (!hasPointBuffer(points, n_points, point_dim))
    {
        return {};
    }

    (void)available_vram_bytes;
    return cpu::simd::computeCPUOptimized(points, n_points, point_dim, max_distance);
}

} // namespace nerve::persistence::vram
