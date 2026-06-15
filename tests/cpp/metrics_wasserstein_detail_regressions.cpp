#include "nerve/core_types.hpp"
#include "nerve/metrics/distances.hpp"
#include "nerve/metrics/gpu_distances.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Field;
using nerve::Size;
using nerve::metrics::Diagram;
using nerve::metrics::WassersteinDistance;
using nerve::persistence::Pair;

constexpr double kTol = 1e-10;

Diagram make_diagram(const std::vector<Pair> &pairs)
{
    Diagram d;
    for (const auto &p : pairs)
        d.addPair(p);
    return d;
}

std::vector<std::pair<float, float>> to_float_pairs(const std::vector<Pair> &pairs)
{
    std::vector<std::pair<float, float>> result;
    for (const auto &p : pairs)
        result.emplace_back(static_cast<float>(p.birth), static_cast<float>(p.death));
    return result;
}

bool check_sinkhorn_basic()
{
    auto d1 = to_float_pairs({{0.0, 1.0, 0}, {0.2, 0.7, 0}});
    auto d2 = to_float_pairs({{0.0, 1.5, 0}, {0.5, 2.5, 0}});
    nerve::metrics::sinkhorn::SinkhornConfig cfg;
    cfg.epsilon = 0.1;
    cfg.max_iterations = 50;
    cfg.gpu_accelerated = false;
    double dist = nerve::metrics::sinkhorn::sinkhornDiagramDistance(d1, d2, cfg);
    if (dist < 0.0 || !std::isfinite(dist))
    {
        std::cerr << "sinkhorn distance invalid: " << dist << "\n";
        return false;
    }
    return true;
}

bool check_sinkhorn_symmetry()
{
    auto d1 = to_float_pairs({{0.0, 1.0, 0}});
    auto d2 = to_float_pairs({{0.0, 2.0, 0}});
    nerve::metrics::sinkhorn::SinkhornConfig cfg;
    cfg.epsilon = 0.1;
    cfg.max_iterations = 50;
    cfg.gpu_accelerated = false;
    double d12 = nerve::metrics::sinkhorn::sinkhornDiagramDistance(d1, d2, cfg);
    double d21 = nerve::metrics::sinkhorn::sinkhornDiagramDistance(d2, d1, cfg);
    if (std::abs(d12 - d21) > 1e-6)
    {
        std::cerr << "sinkhorn not symmetric: " << d12 << " vs " << d21 << "\n";
        return false;
    }
    return true;
}

bool check_sinkhorn_non_negative()
{
    auto d1 = to_float_pairs({{0.0, 1.0, 0}});
    auto d2 = to_float_pairs({{0.0, 2.0, 0}});
    nerve::metrics::sinkhorn::SinkhornConfig cfg;
    cfg.epsilon = 0.1;
    cfg.gpu_accelerated = false;
    double dist = nerve::metrics::sinkhorn::sinkhornDiagramDistance(d1, d2, cfg);
    if (dist < 0.0)
    {
        std::cerr << "sinkhorn negative: " << dist << "\n";
        return false;
    }
    return true;
}

bool check_sinkhorn_self_zero()
{
    auto d1 = to_float_pairs({{0.0, 1.0, 0}, {0.2, 0.7, 0}});
    nerve::metrics::sinkhorn::SinkhornConfig cfg;
    cfg.epsilon = 0.1;
    cfg.gpu_accelerated = false;
    double dist = nerve::metrics::sinkhorn::sinkhornDiagramDistance(d1, d1, cfg);
    if (dist > 1e-6)
    {
        std::cerr << "sinkhorn(D,D) should be near 0, got " << dist << "\n";
        return false;
    }
    return true;
}

bool check_sinkhorn_different_epsilon()
{
    auto d1 = to_float_pairs({{0.0, 1.0, 0}});
    auto d2 = to_float_pairs({{0.0, 2.0, 0}});
    nerve::metrics::sinkhorn::SinkhornConfig cfg1, cfg2;
    cfg1.epsilon = 0.01;
    cfg1.gpu_accelerated = false;
    cfg2.epsilon = 0.5;
    cfg2.gpu_accelerated = false;
    double d_eps1 = nerve::metrics::sinkhorn::sinkhornDiagramDistance(d1, d2, cfg1);
    double d_eps2 = nerve::metrics::sinkhorn::sinkhornDiagramDistance(d1, d2, cfg2);
    if (!std::isfinite(d_eps1) || !std::isfinite(d_eps2))
    {
        std::cerr << "non-finite sinkhorn with different epsilons\n";
        return false;
    }
    return true;
}

bool check_wasserstein_class_basic()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}});
    WassersteinDistance wd;
    double dist = wd.compute(d1, d2);
    if (dist < 0.0)
    {
        std::cerr << "Wasserstein class returned negative\n";
        return false;
    }
    return true;
}

bool check_wasserstein_class_sinkhorn()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}});
    WassersteinDistance wd;
    wd.useSinkhorn(true);
    wd.setRegularization(0.1);
    double dist = wd.compute(d1, d2);
    if (dist < 0.0)
    {
        std::cerr << "Wasserstein class with sinkhorn returned negative\n";
        return false;
    }
    return true;
}

bool check_wasserstein_class_with_order()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}});
    WassersteinDistance wd(1.0);
    double dist = wd.computeWithOrder(d1, d2, 1.0);
    if (dist < 0.0)
    {
        std::cerr << "Wasserstein class computeWithOrder failed\n";
        return false;
    }
    return true;
}

bool check_sinkhorn_sliced_wasserstein()
{
    auto d1 = to_float_pairs({{0.0, 1.0, 0}});
    auto d2 = to_float_pairs({{0.0, 2.0, 0}});
    double dist = nerve::metrics::sinkhorn::slicedWassersteinDistance(d1, d2, 50);
    if (dist < 0.0 || !std::isfinite(dist))
    {
        std::cerr << "sliced wasserstein invalid: " << dist << "\n";
        return false;
    }
    return true;
}

bool check_sinkhorn_hierarchical()
{
    auto d1 = to_float_pairs({{0.0, 1.0, 0}});
    auto d2 = to_float_pairs({{0.0, 2.0, 0}});
    double dist = nerve::metrics::sinkhorn::hierarchicalWasserstein(d1, d2, 3);
    if (dist < 0.0 || !std::isfinite(dist))
    {
        std::cerr << "hierarchical wasserstein invalid: " << dist << "\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_sinkhorn_basic())
    {
        std::cerr << "FAIL: sinkhorn basic\n";
        return 1;
    }
    if (!check_sinkhorn_symmetry())
    {
        std::cerr << "FAIL: sinkhorn symmetry\n";
        return 1;
    }
    if (!check_sinkhorn_non_negative())
    {
        std::cerr << "FAIL: sinkhorn non negative\n";
        return 1;
    }
    if (!check_sinkhorn_self_zero())
    {
        std::cerr << "FAIL: sinkhorn self zero\n";
        return 1;
    }
    if (!check_sinkhorn_different_epsilon())
    {
        std::cerr << "FAIL: sinkhorn epsilon\n";
        return 1;
    }
    if (!check_wasserstein_class_basic())
    {
        std::cerr << "FAIL: wasserstein class\n";
        return 1;
    }
    if (!check_wasserstein_class_sinkhorn())
    {
        std::cerr << "FAIL: wasserstein sinkhorn\n";
        return 1;
    }
    if (!check_wasserstein_class_with_order())
    {
        std::cerr << "FAIL: wasserstein order\n";
        return 1;
    }
    if (!check_sinkhorn_sliced_wasserstein())
    {
        std::cerr << "FAIL: sliced wasserstein\n";
        return 1;
    }
    if (!check_sinkhorn_hierarchical())
    {
        std::cerr << "FAIL: hierarchical wasserstein\n";
        return 1;
    }
    return 0;
}
