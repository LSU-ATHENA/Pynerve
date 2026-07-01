#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/metrics/distances.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace
{

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

bool check_wasserstein_self_distance_zero()
{
    Diagram d = make_diagram({{0.0, 1.0, 0}, {0.0, 2.0, 0}});
    double w1 = nerve::metrics::wassersteinDistance(d, d, 1.0);
    double w2 = nerve::metrics::wassersteinDistance(d, d, 2.0);

    if (std::abs(w1) > kTol)
    {
        std::cerr << "W_1(D,D) should be 0, got " << w1 << "\n";
        return false;
    }
    if (std::abs(w2) > kTol)
    {
        std::cerr << "W_2(D,D) should be 0, got " << w2 << "\n";
        return false;
    }
    return true;
}

bool check_wasserstein_non_negative()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 2.0);

    std::vector<Pair> pairs1, pairs2;
    for (int i = 0; i < 5; ++i)
    {
        double b1 = dist(rng), d1 = b1 + dist(rng);
        double b2 = dist(rng), d2 = b2 + dist(rng);
        pairs1.push_back({b1, d1, 0});
        pairs2.push_back({b2, d2, 0});
    }

    Diagram d1 = make_diagram(pairs1);
    Diagram d2 = make_diagram(pairs2);

    double w = nerve::metrics::wassersteinDistance(d1, d2, 2.0);
    if (w < 0.0)
    {
        std::cerr << "Wasserstein distance negative: " << w << "\n";
        return false;
    }
    return true;
}

bool check_wasserstein_symmetry()
{
    Diagram d1 = make_diagram({{0.0, 1.5, 0}});
    Diagram d2 = make_diagram({{0.0, 2.5, 0}});

    double w12 = nerve::metrics::wassersteinDistance(d1, d2, 2.0);
    double w21 = nerve::metrics::wassersteinDistance(d2, d1, 2.0);

    if (std::abs(w12 - w21) > kTol)
    {
        std::cerr << "Wasserstein not symmetric: " << w12 << " vs " << w21 << "\n";
        return false;
    }
    return true;
}

bool check_wasserstein_known_values()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}});
    Diagram d2 = make_diagram({{0.0, 3.0, 0}});

    double w1 = nerve::metrics::wassersteinDistance(d1, d2, 1.0);
    double w2 = nerve::metrics::wassersteinDistance(d1, d2, 2.0);

    if (w1 <= 0.0)
    {
        std::cerr << "expected positive W_1\n";
        return false;
    }
    if (w2 <= 0.0)
    {
        std::cerr << "expected positive W_2\n";
        return false;
    }

    return true;
}

bool check_wasserstein_empty_diagram()
{
    Diagram empty;
    Diagram d = make_diagram({{0.0, 1.0, 0}});

    double w = nerve::metrics::wassersteinDistance(empty, d, 2.0);
    if (w < 0.0)
    {
        std::cerr << "Wasserstein with empty negative\n";
        return false;
    }

    double w_empty = nerve::metrics::wassersteinDistance(empty, empty, 2.0);
    if (std::abs(w_empty) > kTol)
    {
        std::cerr << "Wasserstein(empty, empty) should be 0\n";
        return false;
    }

    return true;
}

bool check_wasserstein_class_api()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}, {0.0, 2.0, 0}});
    Diagram d2 = make_diagram({{0.0, 1.5, 0}, {0.5, 2.5, 0}});

    nerve::metrics::WassersteinDistance wd(2.0);
    double dist = wd.compute(d1, d2);
    if (dist < 0.0)
    {
        std::cerr << "WassersteinDistance class returned negative\n";
        return false;
    }
    if (wd.getComputationTime() < 0.0)
    {
        std::cerr << "invalid computation time\n";
        return false;
    }

    wd.setOrder(1.0);
    double dist1 = wd.compute(d1, d2);
    if (dist1 < 0.0)
    {
        std::cerr << "WassersteinDistance class (p=1) returned negative\n";
        return false;
    }

    return true;
}

bool check_wasserstein_different_p_orders()
{
    Diagram d1 = make_diagram({{0.0, 2.0, 0}});
    Diagram d2 = make_diagram({{0.0, 1.0, 0}});

    double w1 = nerve::metrics::wassersteinDistance(d1, d2, 1.0);
    double w2 = nerve::metrics::wassersteinDistance(d1, d2, 2.0);
    double w_inf = nerve::metrics::wassersteinDistance(d1, d2, 10.0);

    if (w1 < 0.0 || w2 < 0.0 || w_inf < 0.0)
    {
        std::cerr << "negative Wasserstein distance\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_wasserstein_self_distance_zero())
    {
        std::cerr << "FAIL: wasserstein self distance zero\n";
        return 1;
    }
    if (!check_wasserstein_non_negative())
    {
        std::cerr << "FAIL: wasserstein non negative\n";
        return 1;
    }
    if (!check_wasserstein_symmetry())
    {
        std::cerr << "FAIL: wasserstein symmetry\n";
        return 1;
    }
    if (!check_wasserstein_known_values())
    {
        std::cerr << "FAIL: wasserstein known values\n";
        return 1;
    }
    if (!check_wasserstein_empty_diagram())
    {
        std::cerr << "FAIL: wasserstein empty diagram\n";
        return 1;
    }
    if (!check_wasserstein_class_api())
    {
        std::cerr << "FAIL: wasserstein class api\n";
        return 1;
    }
    if (!check_wasserstein_different_p_orders())
    {
        std::cerr << "FAIL: wasserstein different p orders\n";
        return 1;
    }
    return 0;
}
