#include "nerve/core_types.hpp"
#include "nerve/metrics/distances.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Field;
using nerve::Size;
using nerve::metrics::Diagram;
using nerve::metrics::FrechetDistance;
using nerve::persistence::Pair;

constexpr double kTol = 1e-10;

Diagram make_diagram(const std::vector<Pair> &pairs)
{
    Diagram d;
    for (const auto &p : pairs)
        d.addPair(p);
    return d;
}

bool check_frechet_self_distance_zero()
{
    Diagram d = make_diagram({{0.0, 1.0, 0}, {0.5, 2.0, 0}, {1.0, 3.0, 0}});
    double dist = nerve::metrics::frechetDistance(d, d);
    if (std::abs(dist) > kTol)
    {
        std::cerr << "frechet(D,D) should be 0, got " << dist << "\n";
        return false;
    }
    return true;
}

bool check_frechet_symmetry()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}, {0.2, 0.7, 0}});
    Diagram d2 = make_diagram({{0.0, 1.5, 0}, {0.5, 2.5, 0}});
    double d12 = nerve::metrics::frechetDistance(d1, d2);
    double d21 = nerve::metrics::frechetDistance(d2, d1);
    if (std::abs(d12 - d21) > kTol)
    {
        std::cerr << "frechet not symmetric: " << d12 << " vs " << d21 << "\n";
        return false;
    }
    return true;
}

bool check_frechet_non_negative()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}, {0.0, 3.0, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}, {1.0, 4.0, 0}});
    double dist = nerve::metrics::frechetDistance(d1, d2);
    if (dist < 0.0)
    {
        std::cerr << "frechet distance negative: " << dist << "\n";
        return false;
    }
    return true;
}

bool check_frechet_line_self_zero()
{
    Diagram line = make_diagram({{0.0, 0.0, 0}, {1.0, 1.0, 0}, {2.0, 2.0, 0}});
    double dist = nerve::metrics::frechetDistance(line, line);
    if (std::abs(dist) > kTol)
    {
        std::cerr << "frechet(line,line) should be 0, got " << dist << "\n";
        return false;
    }
    return true;
}

bool check_frechet_class_api()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}});
    FrechetDistance fd;
    fd.setTolerance(1e-8);
    fd.setParameterization("linear");
    double dist = fd.compute(d1, d2);
    if (dist < 0.0)
    {
        std::cerr << "FrechetDistance class returned negative\n";
        return false;
    }
    return true;
}

bool check_frechet_empty_with_nonempty()
{
    Diagram empty;
    Diagram d = make_diagram({{0.0, 1.0, 0}});
    double dist = nerve::metrics::frechetDistance(empty, d);
    if (dist < 0.0)
    {
        std::cerr << "frechet(empty, d) should be >= 0\n";
        return false;
    }
    double self_empty = nerve::metrics::frechetDistance(empty, empty);
    if (std::abs(self_empty) > kTol)
    {
        std::cerr << "frechet(empty,empty) should be 0\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_frechet_self_distance_zero())
    {
        std::cerr << "FAIL: frechet self distance zero\n";
        return 1;
    }
    if (!check_frechet_symmetry())
    {
        std::cerr << "FAIL: frechet symmetry\n";
        return 1;
    }
    if (!check_frechet_non_negative())
    {
        std::cerr << "FAIL: frechet non negative\n";
        return 1;
    }
    if (!check_frechet_line_self_zero())
    {
        std::cerr << "FAIL: frechet line self zero\n";
        return 1;
    }
    if (!check_frechet_class_api())
    {
        std::cerr << "FAIL: frechet class api\n";
        return 1;
    }
    if (!check_frechet_empty_with_nonempty())
    {
        std::cerr << "FAIL: frechet empty\n";
        return 1;
    }
    return 0;
}
