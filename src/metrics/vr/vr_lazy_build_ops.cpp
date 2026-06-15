#include "nerve/metrics/lazy_distance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace nerve::metrics::lazy
{
namespace
{

void validateLazyVrInputs(size_t n_points, double max_distance, int max_dim)
{
    if (!std::isfinite(max_distance) || max_distance < 0.0)
    {
        throw std::invalid_argument("lazy VR max distance must be finite and non-negative");
    }
    if (max_dim < 0)
    {
        throw std::invalid_argument("lazy VR max dimension must be non-negative");
    }
    if (n_points > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("lazy VR point count exceeds int range");
    }
}

size_t checkedReserveHint(size_t n_points)
{
    constexpr size_t kReserveFactor = 10;
    if (n_points > std::numeric_limits<size_t>::max() / kReserveFactor)
    {
        throw std::length_error("lazy VR simplex reserve size overflows");
    }
    return n_points * kReserveFactor;
}

} // namespace

std::vector<std::vector<int>> buildVRLazy(std::span<const double> points, size_t n_points,
                                          size_t point_dim, double max_distance, int max_dim)
{
    validateLazyVrInputs(n_points, max_distance, max_dim);
    LazyDistanceMatrix lazy_dist(points, n_points, point_dim, "euclidean");

    std::vector<std::vector<int>> simplices;
    simplices.reserve(checkedReserveHint(n_points));

    // Add 0-simplices (vertices)
    for (size_t i = 0; i < n_points; ++i)
    {
        simplices.push_back({static_cast<int>(i)});
    }

    // Add edges within max_distance
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t j = i + 1; j < n_points; ++j)
        {
            if (lazy_dist.isWithinRadius(i, j, max_distance))
            {
                simplices.push_back({static_cast<int>(i), static_cast<int>(j)});
            }
        }
    }

    // Build adjacency list for higher dimensions
    if (max_dim >= 2)
    {
        std::vector<std::vector<int>> adjacency(n_points);
        for (size_t i = 0; i < n_points; ++i)
        {
            for (size_t j = i + 1; j < n_points; ++j)
            {
                if (lazy_dist.isWithinRadius(i, j, max_distance))
                {
                    adjacency[i].push_back(static_cast<int>(j));
                    adjacency[j].push_back(static_cast<int>(i));
                }
            }
        }

        // Sort adjacency lists
        for (auto &neighbors : adjacency)
        {
            std::ranges::sort(neighbors);
        }

        // Generate triangles (2-simplices)
        if (max_dim >= 2)
        {
            for (size_t i = 0; i < n_points; ++i)
            {
                for (size_t ni = 0; ni < adjacency[i].size(); ++ni)
                {
                    int j = adjacency[i][ni];
                    if (j <= static_cast<int>(i))
                        continue;

                    for (size_t nj = ni + 1; nj < adjacency[i].size(); ++nj)
                    {
                        int k = adjacency[i][nj];
                        if (k <= j)
                            continue;

                        if (std::binary_search(adjacency[j].begin(), adjacency[j].end(), k))
                        {
                            simplices.push_back({static_cast<int>(i), j, k});
                        }
                    }
                }
            }
        }

        // Generate tetrahedra (3-simplices)
        if (max_dim >= 3)
        {
            for (size_t i = 0; i < n_points; ++i)
            {
                for (int j : adjacency[i])
                {
                    if (j <= static_cast<int>(i))
                        continue;

                    for (int k : adjacency[i])
                    {
                        if (k <= j)
                            continue;

                        if (!std::binary_search(adjacency[j].begin(), adjacency[j].end(), k))
                        {
                            continue;
                        }

                        for (int l : adjacency[i])
                        {
                            if (l <= k)
                                continue;

                            if (std::binary_search(adjacency[j].begin(), adjacency[j].end(), l) &&
                                std::binary_search(adjacency[k].begin(), adjacency[k].end(), l))
                            {
                                simplices.push_back({static_cast<int>(i), j, k, l});
                            }
                        }
                    }
                }
            }
        }

        // Higher dimensions
        if (max_dim >= 4)
        {
            const size_t max_simplices = checkedReserveHint(simplices.size());

            for (size_t i = 0; i < n_points && simplices.size() < max_simplices; ++i)
            {
                expandCliquesRecursive(adjacency, {static_cast<int>(i)}, adjacency[i], 4, max_dim,
                                       simplices, max_simplices);
            }
        }
    }

    return simplices;
}

void expandCliquesRecursive(const std::vector<std::vector<int>> &adjacency,
                            const std::vector<int> &current_clique,
                            const std::vector<int> &candidates, int current_dim, int max_dim,
                            std::vector<std::vector<int>> &simplices, size_t max_simplices)
{
    if (simplices.size() >= max_simplices)
        return;
    if (current_dim > max_dim)
        return;

    for (size_t i = 0; i < candidates.size() && simplices.size() < max_simplices; ++i)
    {
        int v = candidates[i];

        bool connected = true;
        for (int u : current_clique)
        {
            if (!std::binary_search(adjacency[u].begin(), adjacency[u].end(), v))
            {
                connected = false;
                break;
            }
        }

        if (!connected)
            continue;

        std::vector<int> new_clique = current_clique;
        new_clique.push_back(v);
        std::ranges::sort(new_clique);

        simplices.push_back(new_clique);

        std::vector<int> new_candidates;
        for (size_t j = i + 1; j < candidates.size(); ++j)
        {
            int u = candidates[j];
            if (u > v && std::binary_search(adjacency[v].begin(), adjacency[v].end(), u))
            {
                new_candidates.push_back(u);
            }
        }

        if (!new_candidates.empty())
        {
            expandCliquesRecursive(adjacency, new_clique, new_candidates, current_dim + 1, max_dim,
                                   simplices, max_simplices);
        }
    }
}

} // namespace nerve::metrics::lazy
