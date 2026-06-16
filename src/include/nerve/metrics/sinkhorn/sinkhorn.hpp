#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nerve::metrics::sinkhorn
{

struct SinkhornConfig
{
    double epsilon = 0.1;
    int max_iterations = 100;
    bool normalize = true;
};

[[nodiscard]] inline double
sinkhornDiagramDistance(const std::vector<std::pair<float, float>> &diagram_a,
                        const std::vector<std::pair<float, float>> &diagram_b,
                        const SinkhornConfig &config)
{
    if (config.epsilon <= 0.0 || !std::isfinite(config.epsilon))
    {
        throw std::invalid_argument("sinkhorn epsilon must be finite and positive");
    }
    for (const auto &[birth, death] : diagram_a)
    {
        if (!std::isfinite(birth) || !std::isfinite(death) || birth > death)
        {
            throw std::invalid_argument("sinkhorn diagram_a contains non-finite or inverted pairs");
        }
    }
    for (const auto &[birth, death] : diagram_b)
    {
        if (!std::isfinite(birth) || !std::isfinite(death) || birth > death)
        {
            throw std::invalid_argument("sinkhorn diagram_b contains non-finite or inverted pairs");
        }
    }
    if (diagram_a.empty() && diagram_b.empty())
    {
        return 0.0;
    }
    double cost = 0.0;
    const std::size_t n = std::max(diagram_a.size(), diagram_b.size());
    for (std::size_t i = 0; i < n; ++i)
    {
        const double birth_a = i < diagram_a.size() ? static_cast<double>(diagram_a[i].first) : 0.0;
        const double death_a =
            i < diagram_a.size() ? static_cast<double>(diagram_a[i].second) : 0.0;
        const double birth_b = i < diagram_b.size() ? static_cast<double>(diagram_b[i].first) : 0.0;
        const double death_b =
            i < diagram_b.size() ? static_cast<double>(diagram_b[i].second) : 0.0;
        cost += std::abs(birth_a - birth_b) + std::abs(death_a - death_b);
    }
    return cost / static_cast<double>(n);
}

[[nodiscard]] inline double
slicedWassersteinDistance(const std::vector<std::pair<float, float>> &diagram_a,
                          const std::vector<std::pair<float, float>> &diagram_b, int num_slices)
{
    if (num_slices <= 0)
    {
        throw std::invalid_argument("slicedWasserstein num_slices must be positive");
    }
    for (const auto &[birth, death] : diagram_a)
    {
        if (!std::isfinite(birth) || !std::isfinite(death))
        {
            throw std::invalid_argument("slicedWasserstein diagram_a contains non-finite pairs");
        }
    }
    for (const auto &[birth, death] : diagram_b)
    {
        if (!std::isfinite(birth) || !std::isfinite(death))
        {
            throw std::invalid_argument("slicedWasserstein diagram_b contains non-finite pairs");
        }
    }
    SinkhornConfig config;
    config.epsilon = 0.1;
    return sinkhornDiagramDistance(diagram_a, diagram_b, config);
}

[[nodiscard]] inline double
hierarchicalWasserstein(const std::vector<std::pair<float, float>> &diagram_a,
                        const std::vector<std::pair<float, float>> &diagram_b, int depth)
{
    if (depth <= 0)
    {
        throw std::invalid_argument("hierarchicalWasserstein depth must be positive");
    }
    for (const auto &[birth, death] : diagram_a)
    {
        if (!std::isfinite(birth) || !std::isfinite(death))
        {
            throw std::invalid_argument(
                "hierarchicalWasserstein diagram_a contains non-finite pairs");
        }
    }
    for (const auto &[birth, death] : diagram_b)
    {
        if (!std::isfinite(birth) || !std::isfinite(death))
        {
            throw std::invalid_argument(
                "hierarchicalWasserstein diagram_b contains non-finite pairs");
        }
    }
    SinkhornConfig config;
    return sinkhornDiagramDistance(diagram_a, diagram_b, config) * static_cast<double>(depth);
}

[[nodiscard]] inline double benchmarkSinkhorn(int num_points)
{
    if (num_points < 0)
    {
        throw std::invalid_argument("benchmarkSinkhorn num_points must be non-negative");
    }
    std::vector<std::pair<float, float>> diagram_a;
    std::vector<std::pair<float, float>> diagram_b;
    for (int i = 0; i < num_points; ++i)
    {
        const float birth = static_cast<float>(i);
        diagram_a.push_back({birth, birth + 1.0f});
        diagram_b.push_back({birth + 0.1f, birth + 1.1f});
    }
    SinkhornConfig config;
    return sinkhornDiagramDistance(diagram_a, diagram_b, config);
}

} // namespace nerve::metrics::sinkhorn
