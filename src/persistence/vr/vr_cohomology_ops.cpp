#include "nerve/algebra/complex.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"
#include "nerve/persistence/vr/vr_cohomology_ops.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
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

    algebra::SimplicialComplex complex;

    for (Size i = 0; i < n; ++i)
    {
        complex.addSimplex({static_cast<Index>(i)}, {});
    }

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
                complex.addSimplexWithFiltration(
                    algebra::Simplex{static_cast<Index>(i), static_cast<Index>(j)},
                    std::sqrt(dist_sq), {});
            }
        }
    }

    Size max_dim = dim > 0 ? static_cast<Size>(dim) : 2;
    const Size max_simplex_size = std::min(n, max_dim + 1);
    for (Size simplex_size = 3; simplex_size <= max_simplex_size; ++simplex_size)
    {
        std::function<void(std::vector<Index>, Size)> enumerate;
        enumerate = [&](std::vector<Index> current, Size start) {
            if (current.size() == simplex_size)
            {
                double max_filt = 0.0;
                for (Size a = 0; a < current.size(); ++a)
                {
                    for (Size b = a + 1; b < current.size(); ++b)
                    {
                        double dist_sq = 0.0;
                        for (Size d = 0; d < dim; ++d)
                        {
                            double diff =
                                points[current[a] * dim + d] - points[current[b] * dim + d];
                            dist_sq += diff * diff;
                        }
                        if (dist_sq > max_filt * max_filt)
                            max_filt = std::sqrt(dist_sq);
                    }
                }
                if (max_filt <= max_radius)
                {
                    complex.addSimplexWithFiltration(algebra::Simplex(current), max_filt, {});
                }
                return;
            }
            for (Size i = start; i < n; ++i)
            {
                current.push_back(static_cast<Index>(i));
                enumerate(current, i + 1);
                current.pop_back();
            }
        };
        enumerate({}, 0);
    }

    return computeExactCohomologyZ2(complex, max_dim);
}

} // namespace nerve::persistence
