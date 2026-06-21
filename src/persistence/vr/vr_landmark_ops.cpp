#include "nerve/core/rng/random.hpp"
#include "nerve/persistence/vr/vr_landmark_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nerve::persistence
{

namespace
{

// Landmark selection constants
constexpr size_t WITNESS_K_NEAREST_NEIGHBORS = 10;
constexpr uint64_t WITNESS_HASH_MULTIPLIER_1 = 73856093u;
constexpr uint64_t WITNESS_HASH_MULTIPLIER_2 = 19349663u;

void validateLandmarkInput(const std::vector<double> &points, size_t point_dim, size_t num_points)
{
    if (point_dim == 0 || num_points == 0)
    {
        return;
    }
    if (num_points > std::numeric_limits<size_t>::max() / point_dim)
    {
        throw std::length_error("landmark point buffer size overflows");
    }
    const size_t required_values = num_points * point_dim;
    if (points.size() != required_values)
    {
        throw std::invalid_argument("landmark point buffer size does not match point count");
    }
    if (!std::all_of(points.begin(), points.begin() + static_cast<std::ptrdiff_t>(required_values),
                     [](double value) { return std::isfinite(value); }))
    {
        throw std::invalid_argument("landmark points must contain only finite coordinates");
    }
}

} // namespace

std::vector<size_t> LandmarkSelector::selectLandmarks(const std::vector<double> &points,
                                                      size_t point_dim, size_t num_points,
                                                      size_t num_landmarks, Strategy strategy)
{
    if (point_dim == 0 || num_points == 0 || num_landmarks == 0)
    {
        return {};
    }
    validateLandmarkInput(points, point_dim, num_points);
    num_landmarks = std::min(num_landmarks, num_points);
    switch (strategy)
    {
        case Strategy::MAXMIN:
            return selectMaxmin(points, point_dim, num_points, num_landmarks);
        case Strategy::RANDOM:
            return selectRandom(num_points, num_landmarks);
        case Strategy::GRID:
            return selectGrid(points, point_dim, num_points, num_landmarks);
        case Strategy::DENSITY:
            return selectDensityWeighted(points, point_dim, num_points, num_landmarks);
        default:
            return selectMaxmin(points, point_dim, num_points, num_landmarks);
    }
}

std::vector<size_t> LandmarkSelector::selectMaxmin(const std::vector<double> &points,
                                                   size_t point_dim, size_t num_points,
                                                   size_t num_landmarks)
{
    std::vector<size_t> landmarks;
    landmarks.reserve(num_landmarks);

    // Start with random point
    std::mt19937 rng(42); // Fixed seed for determinism
    std::uniform_int_distribution<size_t> dist(0, num_points - 1);
    landmarks.push_back(dist(rng));

    // Distances from each point to nearest landmark
    std::vector<double> min_dists(num_points, std::numeric_limits<double>::infinity());

    for (size_t i = 1; i < num_landmarks; ++i)
    {
        // Update distances for new landmark
        size_t new_landmark = landmarks.back();
        min_dists[new_landmark] = -std::numeric_limits<double>::infinity();
        const double *new_pt = &points[new_landmark * point_dim];

        for (size_t j = 0; j < num_points; ++j)
        {
            if (std::ranges::find(landmarks, j) != landmarks.end())
            {
                continue;
            }

            const double *pt = &points[j * point_dim];
            double dist_sq = 0.0;
            for (size_t d = 0; d < point_dim; ++d)
            {
                double diff = new_pt[d] - pt[d];
                const double contribution = diff * diff;
                if (!std::isfinite(contribution) ||
                    dist_sq > std::numeric_limits<double>::max() - contribution)
                {
                    dist_sq = std::numeric_limits<double>::infinity();
                    break;
                }
                dist_sq += contribution;
            }

            const double distance = std::sqrt(dist_sq);
            min_dists[j] = std::min(min_dists[j], distance);
        }

        // Find point with maximum distance to nearest landmark
        auto max_it = std::ranges::max_element(min_dists);
        size_t farthest = std::distance(min_dists.begin(), max_it);
        landmarks.push_back(farthest);
    }

    return landmarks;
}

std::vector<size_t> LandmarkSelector::selectRandom(size_t num_points, size_t num_landmarks)
{
    std::vector<size_t> indices(num_points);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    indices.resize(std::min(num_landmarks, num_points));
    return indices;
}

std::vector<size_t> LandmarkSelector::selectGrid(const std::vector<double> &points,
                                                 size_t point_dim, size_t num_points,
                                                 size_t num_landmarks)
{
    // Compute bounding box
    std::vector<double> min_coords(point_dim, std::numeric_limits<double>::infinity());
    std::vector<double> max_coords(point_dim, -std::numeric_limits<double>::infinity());

    for (size_t i = 0; i < num_points; ++i)
    {
        const double *pt = &points[i * point_dim];
        for (size_t d = 0; d < point_dim; ++d)
        {
            min_coords[d] = std::min(min_coords[d], pt[d]);
            max_coords[d] = std::max(max_coords[d], pt[d]);
        }
    }

    // Grid-based selection using spatial hashing
    std::vector<double> cell_sizes(point_dim);
    for (size_t d = 0; d < point_dim; ++d)
    {
        cell_sizes[d] =
            (max_coords[d] - min_coords[d]) / std::sqrt(static_cast<double>(num_landmarks));
        if (cell_sizes[d] == 0.0)
            cell_sizes[d] = 1.0;
    }

    std::unordered_map<size_t, std::vector<size_t>> grid_cells;
    const size_t grid_reserve =
        num_landmarks > grid_cells.max_size() / 2 ? num_landmarks : num_landmarks * 2;
    if (grid_reserve <= grid_cells.max_size())
    {
        grid_cells.reserve(grid_reserve);
    }

    for (size_t i = 0; i < num_points; ++i)
    {
        const double *pt = &points[i * point_dim];

        // Compute cell hash
        size_t hash = 0;
        for (size_t d = 0; d < point_dim; ++d)
        {
            size_t cell_idx = static_cast<size_t>((pt[d] - min_coords[d]) / cell_sizes[d]);
            hash = hash * WITNESS_HASH_MULTIPLIER_1 + cell_idx * WITNESS_HASH_MULTIPLIER_2;
        }

        grid_cells[hash].push_back(i);
    }

    // Select one point from each cell (up to num_landmarks)
    std::vector<size_t> selected;
    selected.reserve(std::min(num_landmarks, grid_cells.size()));

    for (const auto &[hash, indices] : grid_cells)
    {
        if (selected.size() >= num_landmarks)
            break;

        // Pick the middle point in the cell (good representative)
        size_t idx = indices[indices.size() / 2];
        selected.push_back(idx);
    }

    // If we need more, fall back to random selection
    if (selected.size() < num_landmarks)
    {
        std::vector<size_t> remaining;
        for (size_t i = 0; i < num_points; ++i)
        {
            if (std::ranges::find(selected, i) == selected.end())
            {
                remaining.push_back(i);
            }
        }

        std::mt19937 g(static_cast<uint32_t>(num_points * 2654435761ULL ^
                                             num_landmarks * 11400714819323198485ULL));
        std::shuffle(remaining.begin(), remaining.end(), g);

        size_t needed = num_landmarks - selected.size();
        for (size_t i = 0; i < needed && i < remaining.size(); ++i)
        {
            selected.push_back(remaining[i]);
        }
    }

    return selected;
}

std::vector<size_t> LandmarkSelector::selectDensityWeighted(const std::vector<double> &points,
                                                            size_t point_dim, size_t num_points,
                                                            size_t num_landmarks)
{
    if (num_points < 2)
    {
        return selectRandom(num_points, num_landmarks);
    }

    // Estimate density at each point using k-nearest neighbors
    const size_t k = WITNESS_K_NEAREST_NEIGHBORS;
    std::vector<double> densities(num_points);

#pragma omp parallel for
    for (size_t i = 0; i < num_points; ++i)
    {
        const double *p1 = &points[i * point_dim];

        // Find k nearest neighbors
        std::priority_queue<std::pair<double, size_t>> max_heap;

        for (size_t j = 0; j < num_points; ++j)
        {
            if (i == j)
                continue;

            const double *p2 = &points[j * point_dim];
            double dist_sq = 0.0;
            for (size_t d = 0; d < point_dim; ++d)
            {
                double diff = p1[d] - p2[d];
                const double contribution = diff * diff;
                if (!std::isfinite(contribution) ||
                    dist_sq > std::numeric_limits<double>::max() - contribution)
                {
                    dist_sq = std::numeric_limits<double>::infinity();
                    break;
                }
                dist_sq += contribution;
            }

            if (std::isfinite(dist_sq))
            {
                max_heap.push({dist_sq, j});
            }
            if (max_heap.size() > k)
            {
                max_heap.pop();
            }
        }

        // Density inversely proportional to k-th nearest neighbor distance
        if (max_heap.empty())
        {
            densities[i] = 0.0;
        }
        else
        {
            double k_dist = std::sqrt(max_heap.top().first);
            densities[i] = std::isfinite(k_dist) ? 1.0 / (k_dist + 1e-10) : 0.0;
        }
    }

    // Weighted random sampling
    std::discrete_distribution<size_t> dist(densities.begin(), densities.end());
    std::mt19937 rng(42);

    std::vector<size_t> landmarks;
    landmarks.reserve(num_landmarks);
    std::unordered_set<size_t> selected;

    while (landmarks.size() < num_landmarks && landmarks.size() < num_points)
    {
        size_t idx = dist(rng);
        if (selected.insert(idx).second)
        {
            landmarks.push_back(idx);
        }
    }

    return landmarks;
}

} // namespace nerve::persistence
