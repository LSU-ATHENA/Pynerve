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
