#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/metrics/distances.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::core::BufferView;
using nerve::metrics::Diagram;
using nerve::persistence::Pair;

constexpr double kTol = 1e-10;

Diagram make_diagram(const std::vector<Pair> &pairs)
{
    Diagram d;
    for (const auto &p : pairs)
        d.addPair(p);
    return d;
}

bool check_bottleneck_self_distance_zero()
{
    Diagram d = make_diagram(
        {{0.0, 1.0, 0}, {0.5, 2.0, 0}, {1.0, std::numeric_limits<double>::infinity(), 1}});
    double dist = nerve::metrics::bottleneckDistance(d, d);
    if (std::abs(dist) > kTol)
    {
        std::cerr << "bottleneck(D,D) should be 0, got " << dist << "\n";
        return false;
    }
    return true;
}

bool check_bottleneck_symmetry()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}, {0.0, 2.0, 0}});
    Diagram d2 = make_diagram({{0.0, 1.5, 0}, {0.5, 2.5, 0}});

    double d12 = nerve::metrics::bottleneckDistance(d1, d2);
    double d21 = nerve::metrics::bottleneckDistance(d2, d1);

    if (std::abs(d12 - d21) > kTol)
    {
        std::cerr << "bottleneck not symmetric: " << d12 << " vs " << d21 << "\n";
        return false;
    }
    return true;
}

bool check_bottleneck_non_negative()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}, {0.0, 3.0, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}, {1.0, 4.0, 0}});

    double dist = nerve::metrics::bottleneckDistance(d1, d2);
    if (dist < 0.0)
    {
        std::cerr << "bottleneck distance negative: " << dist << "\n";
        return false;
    }
    return true;
}

bool check_bottleneck_known_value()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}});

    double dist = nerve::metrics::bottleneckDistance(d1, d2);

    if (dist <= 0.0)
    {
        std::cerr << "expected positive bottleneck, got " << dist << "\n";
        return false;
    }
    return true;
}

bool check_bottleneck_triangle_inequality()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}});
    Diagram d2 = make_diagram({{0.5, 2.0, 0}});
    Diagram d3 = make_diagram({{0.0, 3.0, 0}});

    double d12 = nerve::metrics::bottleneckDistance(d1, d2);
    double d23 = nerve::metrics::bottleneckDistance(d2, d3);
    double d13 = nerve::metrics::bottleneckDistance(d1, d3);

    if (d13 > d12 + d23 + kTol)
    {
        std::cerr << "triangle inequality violated: " << d13 << " > " << d12 + d23 << "\n";
        return false;
    }
    return true;
}

bool check_bottleneck_empty_diagram()
{
    Diagram empty;
    Diagram d = make_diagram({{0.0, 1.0, 0}});

    double dist = nerve::metrics::bottleneckDistance(empty, d);
    if (dist < 0.0)
    {
        std::cerr << "bottleneck with empty should be >= 0\n";
        return false;
    }

    double self_empty = nerve::metrics::bottleneckDistance(empty, empty);
    if (std::abs(self_empty) > kTol)
    {
        std::cerr << "bottleneck(empty,empty) should be 0\n";
        return false;
    }

    return true;
}

bool check_bottleneck_class_api()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}});

    nerve::metrics::BottleneckDistance bd;
    bd.setTolerance(1e-8);
    bd.setMaxIterations(500);

    double dist = bd.compute(d1, d2);
    if (dist <= 0.0)
    {
        std::cerr << "BottleneckDistance class returned non-positive\n";
        return false;
    }
    if (bd.getComputationTime() < 0.0)
    {
        std::cerr << "invalid computation time\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_bottleneck_self_distance_zero())
    {
        std::cerr << "FAIL: bottleneck self distance zero\n";
        return 1;
    }
    if (!check_bottleneck_symmetry())
    {
        std::cerr << "FAIL: bottleneck symmetry\n";
        return 1;
    }
    if (!check_bottleneck_non_negative())
    {
        std::cerr << "FAIL: bottleneck non negative\n";
        return 1;
    }
    if (!check_bottleneck_known_value())
    {
        std::cerr << "FAIL: bottleneck known value\n";
        return 1;
    }
    if (!check_bottleneck_triangle_inequality())
    {
        std::cerr << "FAIL: bottleneck triangle inequality\n";
        return 1;
    }
    if (!check_bottleneck_empty_diagram())
    {
        std::cerr << "FAIL: bottleneck empty diagram\n";
        return 1;
    }
    if (!check_bottleneck_class_api())
    {
        std::cerr << "FAIL: bottleneck class api\n";
        return 1;
    }
    return 0;
}
