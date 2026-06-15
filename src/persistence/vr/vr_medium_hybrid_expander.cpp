#include "detail/vr_medium_hybrid_helpers.inl"
#include "nerve/algebra/complex.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace nerve::persistence
{

using namespace detail;

class ParallelCliqueExpander
{
public:
    ParallelCliqueExpander(const std::vector<std::vector<int>> &neighbors,
                           const std::vector<std::vector<double>> &distance_matrix, size_t max_dim,
                           double max_radius)
        : neighbors_(neighbors)
        , distance_matrix_(distance_matrix)
        , max_dim_(max_dim)
        , max_radius_(max_radius)
    {}

    void expand(size_t num_points, algebra::SimplicialComplex &complex, SimplexSet &seen)
    {
        for (size_t i = 0; i < num_points; ++i)
        {
            std::vector<Index> v{static_cast<Index>(i)};
            complex.addSimplexWithFiltration(algebra::Simplex(v), 0.0);
        }

        for (size_t i = 0; i < num_points; ++i)
        {
            for (int j : neighbors_[i])
            {
                if (static_cast<size_t>(j) <= i)
                {
                    continue;
                }
                double d = distance_matrix_[i][j];
                std::vector<Index> edge{static_cast<Index>(i), static_cast<Index>(j)};
                complex.addSimplexWithFiltration(algebra::Simplex(edge), d);
            }
        }

        if (num_points < 3)
        {
            return;
        }
        const size_t max_simplex_size = max_dim_ > num_points - 2 ? num_points : max_dim_ + 2;

        for (size_t simplex_size = 3; simplex_size <= max_simplex_size; ++simplex_size)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
            for (size_t i = 0; i < num_points; ++i)
            {
                std::vector<int> current{static_cast<int>(i)};
                std::vector<int> candidates;
                for (int j : neighbors_[i])
                {
                    if (j > static_cast<int>(i))
                    {
                        candidates.push_back(j);
                    }
                }
                expandCliquesThread(current, candidates, simplex_size, complex, seen);
            }
        }
    }

private:
    const std::vector<std::vector<int>> &neighbors_;
    const std::vector<std::vector<double>> &distance_matrix_;
    size_t max_dim_;
    double max_radius_;

    double simplexFiltration(const std::vector<int> &verts)
    {
        double w = 0.0;
        for (size_t i = 0; i < verts.size(); ++i)
        {
            for (size_t j = i + 1; j < verts.size(); ++j)
            {
                w = std::max(w, distance_matrix_[verts[i]][verts[j]]);
            }
        }
        return w;
    }

    void expandCliquesThread(std::vector<int> &current, std::vector<int> &candidates,
                             size_t target_size, algebra::SimplicialComplex &complex,
                             SimplexSet &seen)
    {
        if (current.size() == target_size)
        {
            double filt = simplexFiltration(current);
            if (!std::isfinite(filt) || filt > max_radius_)
            {
                return;
            }

            std::vector<int> key = current;
            std::ranges::sort(key);

#ifdef _OPENMP
#pragma omp critical
#endif
            {
                const bool inserted = seen.insert(key).second;
                if (inserted)
                {
                    std::vector<Index> verts;
                    verts.reserve(key.size());
                    for (int v : key)
                    {
                        verts.push_back(static_cast<Index>(v));
                    }
                    complex.addSimplexWithFiltration(algebra::Simplex(verts), filt);
                }
            }
            return;
        }

        while (!candidates.empty())
        {
            int v = candidates.back();
            candidates.pop_back();
            std::vector<int> new_candidates;
            new_candidates.reserve(candidates.size());
            for (int u : candidates)
            {
                if (std::binary_search(neighbors_[v].begin(), neighbors_[v].end(), u))
                {
                    new_candidates.push_back(u);
                }
            }

            current.push_back(v);
            expandCliquesThread(current, new_candidates, target_size, complex, seen);
            current.pop_back();
        }
    }
};

} // namespace nerve::persistence
