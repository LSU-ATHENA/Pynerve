#include "nerve/core_types.hpp"
#include "nerve/probabilistic/probabilistic.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::persistence::Diagram;
using nerve::persistence::Pair;

constexpr double kTol = 1e-10;

Diagram make_diagram(const std::vector<Pair> &pairs)
{
    Diagram d;
    for (const auto &p : pairs)
        d.addPair(p);
    return d;
}

bool check_bootstrap_construction()
{
    Diagram d = make_diagram({{0.0, 1.0, 0}, {0.2, 0.7, 0}});
    nerve::probabilistic::BootstrapPersistence bp(d);
    bp.setRandomSeed(42);
    auto mean = bp.computeMeanPersistence();
    if (mean.empty())
    {
        std::cerr << "bootstrap mean empty\n";
        return false;
    }
    for (double v : mean)
    {
        if (!std::isfinite(v))
        {
            std::cerr << "bootstrap mean non-finite\n";
            return false;
        }
    }
    return true;
}

bool check_bootstrap_resample()
{
    Diagram d = make_diagram({{0.0, 1.0, 0}, {0.2, 0.7, 0}, {0.5, 2.0, 1}});
    nerve::probabilistic::BootstrapPersistence bp(d);
    bp.setRandomSeed(42);
    bp.setBootstrapMethod("with_replacement");
    bp.setNumBootstrapSamples(10);
    auto samples = bp.bootstrapResample(5);
    if (samples.empty())
    {
        std::cerr << "bootstrap resample empty\n";
        return false;
    }
    for (const auto &s : samples)
    {
        if (s.count() == 0)
        {
            std::cerr << "bootstrap sample has zero pairs\n";
            return false;
        }
    }
    return true;
}

bool check_bootstrap_confidence_intervals()
{
    Diagram d = make_diagram({{0.0, 1.0, 0}, {0.2, 0.7, 0}});
    nerve::probabilistic::BootstrapPersistence bp(d);
    bp.setRandomSeed(42);
    auto ci = bp.computeConfidenceIntervals(0.95);
    if (ci.empty())
    {
        std::cerr << "bootstrap CI empty\n";
        return false;
    }
    for (double v : ci)
    {
        if (!std::isfinite(v))
        {
            std::cerr << "bootstrap CI non-finite\n";
            return false;
        }
    }
    return true;
}

bool check_bootstrap_std()
{
    Diagram d = make_diagram({{0.0, 1.0, 0}, {0.2, 0.7, 0}});
    nerve::probabilistic::BootstrapPersistence bp(d);
    bp.setRandomSeed(42);
    auto stddev = bp.computeStdPersistence();
    if (stddev.empty())
    {
        std::cerr << "bootstrap std empty\n";
        return false;
    }
    for (double v : stddev)
    {
        if (v < 0.0 || !std::isfinite(v))
        {
            std::cerr << "bootstrap std invalid\n";
            return false;
        }
    }
    return true;
}

bool check_bootstrap_intervals()
{
    Diagram d = make_diagram({{0.0, 1.0, 0}});
    nerve::probabilistic::BootstrapPersistence bp(d);
    bp.setRandomSeed(42);
    auto intervals = bp.bootstrapPersistenceIntervals(0.95);
    if (intervals.empty())
    {
        std::cerr << "bootstrap persistence intervals empty\n";
        return false;
    }
    return true;
}

bool check_statistical_permutation_test()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}, {0.2, 0.7, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}, {0.5, 2.5, 0}});
    nerve::probabilistic::StatisticalPersistence sp;
    double pval = sp.permutationTest(d1, d2, 50);
    if (pval < 0.0 || pval > 1.0)
    {
        std::cerr << "permutation test p-value out of range: " << pval << "\n";
        return false;
    }
    return true;
}

bool check_statistical_ks_test()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}, {0.2, 0.7, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}, {0.5, 2.5, 0}});
    nerve::probabilistic::StatisticalPersistence sp;
    double ks = sp.kolmogorovSmirnovTest(d1, d2);
    if (ks < 0.0 || ks > 1.0)
    {
        std::cerr << "KS test out of range: " << ks << "\n";
        return false;
    }
    return true;
}

bool check_statistical_effect_size()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}});
    nerve::probabilistic::StatisticalPersistence sp;
    double d = sp.computeCohensD(d1, d2);
    if (!std::isfinite(d))
    {
        std::cerr << "Cohen's d non-finite\n";
        return false;
    }
    double es = sp.computeEffectSize(d1, d2);
    if (!std::isfinite(es))
    {
        std::cerr << "effect size non-finite\n";
        return false;
    }
    return true;
}

bool check_probabilistic_tda_bootstrap_ci()
{
    std::vector<double> sample{1.0, 2.0, 3.0, 4.0, 5.0};
    auto ci = nerve::probabilistic::ProbabilisticTDA::computeBootstrapCi(sample, 0.95);
    if (ci.size() != 2)
    {
        std::cerr << "bootstrap CI should have 2 values\n";
        return false;
    }
    for (double v : ci)
    {
        if (!std::isfinite(v))
        {
            std::cerr << "ProbabilisticTDA CI non-finite\n";
            return false;
        }
    }
    return true;
}

bool check_probabilistic_tda_effect_size()
{
    std::vector<double> s1{1.0, 2.0, 3.0};
    std::vector<double> s2{4.0, 5.0, 6.0};
    double es = nerve::probabilistic::ProbabilisticTDA::computeEffectSize(s1, s2);
    if (!std::isfinite(es))
    {
        std::cerr << "ProbabilisticTDA effect size non-finite\n";
        return false;
    }
    return true;
}

bool check_empty_diagram()
{
    Diagram empty;
    Diagram d = make_diagram({{0.0, 1.0, 0}});
    nerve::probabilistic::StatisticalPersistence sp;
    double ks = sp.kolmogorovSmirnovTest(empty, d);
    if (!std::isfinite(ks))
    {
        std::cerr << "KS test with empty failed\n";
        return false;
    }
    return true;
}

bool check_single_point_diagram()
{
    Diagram single = make_diagram({{0.0, 1.0, 0}});
    nerve::probabilistic::BootstrapPersistence bp(single);
    bp.setRandomSeed(42);
    auto mean = bp.computeMeanPersistence();
    if (mean.empty())
    {
        std::cerr << "single point bootstrap mean empty\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_bootstrap_construction())
    {
        std::cerr << "FAIL: bootstrap construction\n";
        return 1;
    }
    if (!check_bootstrap_resample())
    {
        std::cerr << "FAIL: bootstrap resample\n";
        return 1;
    }
    if (!check_bootstrap_confidence_intervals())
    {
        std::cerr << "FAIL: bootstrap CI\n";
        return 1;
    }
    if (!check_bootstrap_std())
    {
        std::cerr << "FAIL: bootstrap std\n";
        return 1;
    }
    if (!check_bootstrap_intervals())
    {
        std::cerr << "FAIL: bootstrap intervals\n";
        return 1;
    }
    if (!check_statistical_permutation_test())
    {
        std::cerr << "FAIL: permutation test\n";
        return 1;
    }
    if (!check_statistical_ks_test())
    {
        std::cerr << "FAIL: KS test\n";
        return 1;
    }
    if (!check_statistical_effect_size())
    {
        std::cerr << "FAIL: effect size\n";
        return 1;
    }
    if (!check_probabilistic_tda_bootstrap_ci())
    {
        std::cerr << "FAIL: TDA bootstrap CI\n";
        return 1;
    }
    if (!check_probabilistic_tda_effect_size())
    {
        std::cerr << "FAIL: TDA effect size\n";
        return 1;
    }
    if (!check_empty_diagram())
    {
        std::cerr << "FAIL: empty diagram\n";
        return 1;
    }
    if (!check_single_point_diagram())
    {
        std::cerr << "FAIL: single point\n";
        return 1;
    }
    return 0;
}
