#pragma once

#include "nerve/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nerve::persistence::accelerated::detail
{

inline bool checkedSizeProduct(Size lhs, Size rhs, Size &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<Size>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

inline bool checkedCompleteGraphEdgeCount(Size n_points, Size &out) noexcept
{
    if (n_points < 2)
    {
        out = 0;
        return true;
    }

    Size lhs = n_points;
    Size rhs = n_points - 1;
    if ((lhs % 2) == 0)
    {
        lhs /= 2;
    }
    else
    {
        rhs /= 2;
    }
    return checkedSizeProduct(lhs, rhs, out);
}

inline Size saturatingAdd(Size lhs, Size rhs) noexcept
{
    return rhs > std::numeric_limits<Size>::max() - lhs ? std::numeric_limits<Size>::max()
                                                        : lhs + rhs;
}

inline Size saturatingProduct(Size lhs, Size rhs) noexcept
{
    Size out = 0;
    return checkedSizeProduct(lhs, rhs, out) ? out : std::numeric_limits<Size>::max();
}

inline Size saturatingCompleteGraphEdgeCount(Size n_points) noexcept
{
    Size out = 0;
    return checkedCompleteGraphEdgeCount(n_points, out) ? out : std::numeric_limits<Size>::max();
}

inline Size saturatingScale(Size value, double factor) noexcept
{
    if (!std::isfinite(factor) || factor <= 0.0 || value == 0)
    {
        return 0;
    }
    const long double scaled = static_cast<long double>(value) * static_cast<long double>(factor);
    const long double max_size = static_cast<long double>(std::numeric_limits<Size>::max());
    if (scaled >= max_size)
    {
        return std::numeric_limits<Size>::max();
    }
    return static_cast<Size>(std::ceil(scaled));
}

inline Size ceilDiv(Size total, Size block) noexcept
{
    if (block == 0)
    {
        return 0;
    }
    return (total / block) + ((total % block) == 0 ? 0 : 1);
}

} // namespace nerve::persistence::accelerated::detail

namespace nerve::persistence::accelerated::utils
{

inline Size getOptimalBlockSize(Size work_items, Size max_threads = 1024)
{
    const Size capped_threads = std::min(std::max<Size>(max_threads, 1), Size(1024));
    if (work_items < 1000)
        return std::min(capped_threads, Size(128));
    if (work_items < 10000)
        return std::min(capped_threads, Size(256));
    if (work_items < 100000)
        return std::min(capped_threads, Size(512));
    return capped_threads;
}

inline Size getOptimalGridSize(Size total_elements, Size block_size)
{
    return detail::ceilDiv(total_elements, block_size);
}

inline bool shouldUseStreaming(Size memory_usage, Size available_gpu_memory)
{
    return available_gpu_memory > 0 && memory_usage > available_gpu_memory / 2;
}

} // namespace nerve::persistence::accelerated::utils
