#include "nerve/algebra/complex.hpp"
#include "nerve/persistence/vr/vr_landmark_ops.hpp"
#include "nerve/persistence/vr/vr_lazy_witness_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <vector>

namespace nerve::persistence
{

namespace
{

// Hash constants for witness complex
constexpr size_t WITNESS_SEEN_RESERVE_FACTOR = 8;

} // namespace

LazyWitnessComplex::LazyWitnessComplex(const std::vector<double> &all_points, size_t point_dim,
                                       const std::vector<size_t> &landmarks, size_t max_dim,
                                       double max_radius)
    : all_points_(all_points)
    , point_dim_(point_dim)
    , landmarks_(landmarks)
    , max_dim_(max_dim)
    , max_radius_(max_radius)
{}

bool LazyWitnessComplex::hasValidInput() const
{
    if (point_dim_ == 0 || !std::isfinite(max_radius_) || max_radius_ < 0.0)
    {
        return false;
    }
    if (max_dim_ > static_cast<size_t>(std::numeric_limits<Dimension>::max()))
    {
        return false;
    }
    if ((all_points_.size() % point_dim_) != 0)
    {
        return false;
    }
    const long double safe_abs =
        std::sqrt(static_cast<long double>(std::numeric_limits<double>::max()) /
                  static_cast<long double>(point_dim_)) /
        4.0L;
    const size_t num_points = all_points_.size() / point_dim_;
    for (const double value : all_points_)
    {
        if (!std::isfinite(value) || std::abs(static_cast<long double>(value)) > safe_abs)
        {
            return false;
        }
    }
    for (const size_t landmark : landmarks_)
    {
        if (landmark >= num_points ||
            landmark > static_cast<size_t>(std::numeric_limits<Index>::max()))
        {
            return false;
        }
    }
    return true;
}

double LazyWitnessComplex::landmarkDistance(size_t i, size_t j) const
{
    const double *p1 = &all_points_[i * point_dim_];
    const double *p2 = &all_points_[j * point_dim_];

    double dist_sq = 0.0;
    for (size_t d = 0; d < point_dim_; ++d)
    {
        double diff = p1[d] - p2[d];
        const double contribution = diff * diff;
        if (!std::isfinite(contribution) ||
            dist_sq > std::numeric_limits<double>::max() - contribution)
        {
            return std::numeric_limits<double>::infinity();
        }
        dist_sq += contribution;
    }
    const double distance = std::sqrt(dist_sq);
    return std::isfinite(distance) ? distance : std::numeric_limits<double>::infinity();
}

void LazyWitnessComplex::buildComplex(algebra::SimplicialComplex &complex)
{
    if (!hasValidInput())
    {
        return;
    }

    // Build 1-skeleton: edges between landmarks within max_radius
    std::vector<std::vector<size_t>> landmark_neighbors(landmarks_.size());

    for (size_t i = 0; i < landmarks_.size(); ++i)
    {
        for (size_t j = i + 1; j < landmarks_.size(); ++j)
        {
            double dist = landmarkDistance(landmarks_[i], landmarks_[j]);
            if (std::isfinite(dist) && dist <= max_radius_)
            {
                landmark_neighbors[i].push_back(j);
                landmark_neighbors[j].push_back(i);
            }
        }
    }

    // Sort neighbor lists for efficient clique finding
    for (auto &neighbors : landmark_neighbors)
    {
        std::ranges::sort(neighbors);
    }

    // Add vertices (0-simplices)
    for (size_t i = 0; i < landmarks_.size(); ++i)
    {
        complex.addSimplexWithFiltration(algebra::Simplex({static_cast<Index>(landmarks_[i])}),
                                         0.0);
    }

    // Add edges (1-simplices)
    for (size_t i = 0; i < landmarks_.size(); ++i)
    {
        for (size_t j : landmark_neighbors[i])
        {
            if (j > i)
            {
                double dist = landmarkDistance(landmarks_[i], landmarks_[j]);
                if (!std::isfinite(dist))
                {
                    continue;
                }
                complex.addSimplexWithFiltration(
                    algebra::Simplex(
                        {static_cast<Index>(landmarks_[i]), static_cast<Index>(landmarks_[j])}),
                    dist);
            }
        }
    }

    // Build higher-dimensional simplices via clique enumeration
    if (landmarks_.size() >= 3)
    {
        buildHigherSimplices(landmark_neighbors, complex);
    }
}

void LazyWitnessComplex::buildHigherSimplices(
    const std::vector<std::vector<size_t>> &landmark_neighbors, algebra::SimplicialComplex &complex)
{
    std::unordered_set<std::vector<size_t>, SimplexKeyHash> seen;
    const size_t seen_reserve = landmarks_.size() > seen.max_size() / WITNESS_SEEN_RESERVE_FACTOR
                                    ? landmarks_.size()
                                    : landmarks_.size() * WITNESS_SEEN_RESERVE_FACTOR;
    if (seen_reserve <= seen.max_size())
    {
        seen.reserve(seen_reserve);
    }

    // To compute H_k correctly, include simplices up to dimension k + 1.
    const size_t max_simplex_size = std::min(landmarks_.size(), max_dim_ + 2);
    for (size_t simplex_size = 3; simplex_size <= max_simplex_size; ++simplex_size)
    {
        std::vector<size_t> current;
        current.reserve(simplex_size);

        // Find all simplex_size-cliques in the 1-skeleton.
        for (size_t i = 0; i < landmarks_.size(); ++i)
        {
            current.push_back(i);

            std::vector<size_t> candidates = landmark_neighbors[i];

            // Filter candidates to those > i
            candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                            [i](size_t x) { return x <= i; }),
                             candidates.end());

            expandCliques(current, candidates, simplex_size, landmark_neighbors, complex, seen);
            current.pop_back();
        }
    }
}

void LazyWitnessComplex::expandCliques(
    std::vector<size_t> &current, std::vector<size_t> &candidates, size_t target_size,
    const std::vector<std::vector<size_t>> &neighbors, algebra::SimplicialComplex &complex,
    std::unordered_set<std::vector<size_t>, SimplexKeyHash> &seen)
{
    if (current.size() == target_size)
    {
        // Compute filtration value
        double max_edge = 0.0;
        for (size_t i = 0; i < current.size(); ++i)
        {
            for (size_t j = i + 1; j < current.size(); ++j)
            {
                max_edge = std::max(
                    max_edge, landmarkDistance(landmarks_[current[i]], landmarks_[current[j]]));
            }
        }

        if (std::isfinite(max_edge) && max_edge <= max_radius_)
        {
            std::vector<size_t> key = current;
            std::ranges::sort(key);

            if (seen.insert(key).second)
            {
                std::vector<Index> verts;
                verts.reserve(current.size());
                for (size_t v : current)
                {
                    verts.push_back(static_cast<Index>(landmarks_[v]));
                }
                complex.addSimplexWithFiltration(algebra::Simplex(verts), max_edge);
            }
        }
        return;
    }

    while (!candidates.empty())
    {
        size_t v = candidates.back();
        candidates.pop_back();

        std::vector<size_t> new_candidates;
        new_candidates.reserve(candidates.size());

        for (size_t u : candidates)
        {
            if (std::binary_search(neighbors[v].begin(), neighbors[v].end(), u))
            {
                new_candidates.push_back(u);
            }
        }

        current.push_back(v);
        expandCliques(current, new_candidates, target_size, neighbors, complex, seen);
        current.pop_back();
    }
}

} // namespace nerve::persistence
