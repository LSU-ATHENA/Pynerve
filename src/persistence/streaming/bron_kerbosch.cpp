#include "nerve/persistence/streaming/streaming_reducer.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <ranges>
#include <tuple>

namespace nerve::persistence::streaming
{
namespace
{

constexpr int kMaxIndexedVertex = std::numeric_limits<int>::max();

} // namespace

BronKerboschEnumerator::BronKerboschEnumerator(const std::vector<std::vector<int>> &adjacency)
    : adjacency_(adjacency)
{}

std::vector<Clique> BronKerboschEnumerator::enumerateCliques(std::span<const double> edge_weights,
                                                             size_t n_points, int max_dim)
{
    std::vector<Clique> result;
    if (max_dim < 0 || n_points > static_cast<size_t>(kMaxIndexedVertex))
    {
        return result;
    }

    auto filtrationValue = [&](const std::vector<int> &vertices) {
        double value = 0.0;
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            for (size_t j = i + 1; j < vertices.size(); ++j)
            {
                const size_t u = static_cast<size_t>(vertices[i]);
                const size_t v = static_cast<size_t>(vertices[j]);
                const size_t idx = u * n_points + v;
                if (u < n_points && v < n_points && idx < edge_weights.size())
                {
                    value = std::max(value, edge_weights[idx]);
                }
            }
        }
        return value;
    };

    const size_t max_vertices = static_cast<size_t>(max_dim) + 1;
    std::function<void(std::vector<int> &, const std::vector<int> &)> expand =
        [&](std::vector<int> &current, const std::vector<int> &candidates) {
            if (!current.empty())
            {
                result.push_back({current, filtrationValue(current), 0});
            }
            if (current.size() >= max_vertices)
            {
                return;
            }

            for (size_t pos = 0; pos < candidates.size(); ++pos)
            {
                const int v = candidates[pos];
                current.push_back(v);
                std::vector<int> new_excluded;
                new_excluded.reserve(candidates.size() - pos - 1);
                for (size_t next = pos + 1; next < candidates.size(); ++next)
                {
                    const int u = candidates[next];
                    if (isAdjacent(v, u))
                    {
                        new_excluded.push_back(u);
                    }
                }
                expand(current, new_excluded);
                current.pop_back();
            }
        };

    std::vector<int> all_vertices(n_points);
    std::iota(all_vertices.begin(), all_vertices.end(), 0);
    std::vector<int> current;
    expand(current, all_vertices);

    std::ranges::sort(result, {}, [](const Clique &clique) {
        return std::tuple(clique.filtration_value, clique.vertices.size(), clique.vertices);
    });
    for (size_t i = 0; i < result.size(); ++i)
    {
        result[i].simplex_index = i;
    }

    return result;
}

bool BronKerboschEnumerator::isAdjacent(int u, int v) const
{
    if (u < 0 || v < 0 || static_cast<size_t>(u) >= adjacency_.size() ||
        static_cast<size_t>(v) >= adjacency_.size())
    {
        return false;
    }
    const auto &neighbors = adjacency_[u];
    return std::binary_search(neighbors.begin(), neighbors.end(), v);
}

} // namespace nerve::persistence::streaming
