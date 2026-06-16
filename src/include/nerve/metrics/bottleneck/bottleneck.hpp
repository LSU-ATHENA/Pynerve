#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nerve::metrics::bottleneck
{

[[nodiscard]] inline double
adaptiveBottleneckDistance(const std::vector<std::pair<float, float>> &diagram_a,
                           const std::vector<std::pair<float, float>> &diagram_b)
{
    for (const auto &[birth, death] : diagram_a)
    {
        if (!std::isfinite(birth) || !std::isfinite(death) || birth > death)
        {
            throw std::invalid_argument(
                "adaptiveBottleneckDistance diagram_a contains non-finite or inverted pairs");
        }
    }
    for (const auto &[birth, death] : diagram_b)
    {
        if (!std::isfinite(birth) || !std::isfinite(death) || birth > death)
        {
            throw std::invalid_argument(
                "adaptiveBottleneckDistance diagram_b contains non-finite or inverted pairs");
        }
    }
    if (diagram_a.empty() && diagram_b.empty())
        return 0.0;
    if (diagram_a.empty() || diagram_b.empty())
        return std::numeric_limits<double>::infinity();
    double max_diff = 0.0;
    for (const auto &[birth_a, death_a] : diagram_a)
    {
        double closest = std::numeric_limits<double>::max();
        for (const auto &[birth_b, death_b] : diagram_b)
            closest = std::min(closest, std::max(static_cast<double>(std::abs(birth_a - birth_b)),
                                                 static_cast<double>(std::abs(death_a - death_b))));
        if (closest > max_diff)
            max_diff = closest;
    }
    for (const auto &[birth_b, death_b] : diagram_b)
    {
        double closest = std::numeric_limits<double>::max();
        for (const auto &[birth_a, death_a] : diagram_a)
            closest = std::min(closest, std::max(static_cast<double>(std::abs(birth_a - birth_b)),
                                                 static_cast<double>(std::abs(death_a - death_b))));
        if (closest > max_diff)
            max_diff = closest;
    }
    return max_diff;
}

[[nodiscard]] inline std::vector<double>
parallelBottleneckDistances(const std::vector<std::vector<std::pair<float, float>>> &diagrams_a,
                            const std::vector<std::vector<std::pair<float, float>>> &diagrams_b,
                            int num_threads = 1)
{
    (void)num_threads;
    if (num_threads < 1)
        throw std::invalid_argument("num_threads must be >= 1");
    auto n = std::min(diagrams_a.size(), diagrams_b.size());
    std::vector<double> results;
    results.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        results.push_back(adaptiveBottleneckDistance(diagrams_a[i], diagrams_b[i]));
    return results;
}

} // namespace nerve::metrics::bottleneck
