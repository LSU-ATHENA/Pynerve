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
    {
        return 0.0;
    }
    if (diagram_a.empty() || diagram_b.empty())
    {
        return std::numeric_limits<double>::infinity();
    }
    double max_diff = 0.0;
    for (const auto &[birth_a, death_a] : diagram_a)
    {
        double closest_a = std::numeric_limits<double>::max();
        for (const auto &[birth_b, death_b] : diagram_b)
        {
            const double diff = std::max(std::abs(birth_a - birth_b), std::abs(death_a - death_b));
            if (diff < closest_a)
            {
                closest_a = diff;
            }
        }
        if (closest_a > max_diff)
        {
            max_diff = closest_a;
        }
    }
    for (const auto &[birth_b, death_b] : diagram_b)
    {
        double closest_b = std::numeric_limits<double>::max();
        for (const auto &[birth_a, death_a] : diagram_a)
        {
            const double diff = std::max(std::abs(birth_a - birth_b), std::abs(death_a - death_b));
            if (diff < closest_b)
            {
                closest_b = diff;
            }
        }
        if (closest_b > max_diff)
        {
            max_diff = closest_b;
        }
    }
    return max_diff;
}

[[nodiscard]] inline std::vector<double>
parallelBottleneckDistances(const std::vector<std::vector<std::pair<float, float>>> &diagrams_a,
                            const std::vector<std::vector<std::pair<float, float>>> &diagrams_b,
                            int num_threads)
{
    if (num_threads < 1)
    {
        throw std::invalid_argument("parallelBottleneckDistances num_threads must be >= 1");
    }
    const size_t n = std::min(diagrams_a.size(), diagrams_b.size());
    std::vector<double> results;
    results.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        results.push_back(adaptiveBottleneckDistance(diagrams_a[i], diagrams_b[i]));
    }
    return results;
}

} // namespace nerve::metrics::bottleneck
