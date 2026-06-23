#include "nerve/algebra/complex.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"
#include "nerve/persistence/utils/exact_engine_fast.hpp"
#include "nerve/persistence/vr/vr_cohomology_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nerve::persistence
{

std::vector<Pair> computeVrPersistenceCohomology(const core::BufferView<const double> &points,
                                                 Size point_dim, const VRConfig &config)
{
    const Size n = point_dim == 0 ? 0 : points.size() / point_dim;
    if (n == 0)
        return {};

    // Build VR complex exactly the same way as computeVrPersistenceExact
    // but apply cohomology reduction instead of homology reduction.
    algebra::SimplicialComplex complex;

    // 0-simplices
    for (Size i = 0; i < n; ++i)
    {
        complex.addSimplex({static_cast<Index>(i)}, {});
    }

    // Edges within max_radius
    const double max_radius_sq = config.max_radius * config.max_radius;
    const double *pt = points.data();

    for (Size i = 0; i < n; ++i)
    {
        for (Size j = i + 1; j < n; ++j)
        {
            double dist_sq = 0.0;
            for (Size d = 0; d < point_dim; ++d)
            {
                double diff = pt[i * point_dim + d] - pt[j * point_dim + d];
                dist_sq += diff * diff;
            }
            if (dist_sq <= max_radius_sq)
            {
                complex.addSimplexWithFiltration({static_cast<Index>(i), static_cast<Index>(j)},
                                                 std::sqrt(dist_sq), {});
            }
        }
    }

    // Use cohomology reduction on the built complex
    auto result = computeExactCohomologyZ2(
        complex, static_cast<Size>(config.max_dim > 0 ? config.max_dim : 2));
    return result.pairs;
}

ExactPersistenceResult computeCohomologyVR(const std::vector<double> &points, Size n, Size dim,
                                           double max_radius)
{
    if (n == 0 || dim == 0 || points.empty())
        return {};

    std::vector<std::vector<int>> neighbors(n);
    std::unordered_map<std::uint64_t, double> edge_weights;

    const double max_radius_sq = max_radius * max_radius;

    for (Size i = 0; i < n; ++i)
    {
        for (Size j = i + 1; j < n; ++j)
        {
            double dist_sq = 0.0;
            for (Size d = 0; d < dim; ++d)
            {
                double diff = points[i * dim + d] - points[j * dim + d];
                dist_sq += diff * diff;
            }
            if (dist_sq <= max_radius_sq)
            {
                neighbors[i].push_back(static_cast<int>(j));
                neighbors[j].push_back(static_cast<int>(i));
                std::uint64_t key =
                    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(i)) << 32) |
                    static_cast<std::uint32_t>(j);
                edge_weights[key] = std::sqrt(dist_sq);
            }
        }
    }

    for (auto &nb : neighbors)
    {
        std::sort(nb.begin(), nb.end());
    }

    Size max_dim = dim > 0 ? static_cast<Size>(dim) : 2;
    return computeExactCohomologyZ2Fast(static_cast<int>(n), static_cast<int>(max_dim), max_radius,
                                        neighbors, edge_weights);
}

} // namespace nerve::persistence
