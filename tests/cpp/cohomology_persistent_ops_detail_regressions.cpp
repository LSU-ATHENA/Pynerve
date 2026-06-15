
#include "nerve/persistence/cohomology/cohomology_persistent_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::persistence::CohomologyConfig;
using nerve::persistence::CohomologyResult;
using nerve::persistence::CohomologySpeedupEstimate;
using nerve::persistence::computePersistentCohomology;
using nerve::persistence::estimateCohomologySpeedup;
using nerve::persistence::extractHomologyFromCohomology;
using nerve::persistence::getOptimalCohomologyConfig;
using nerve::persistence::Pair;
using nerve::persistence::shouldUseCohomology;

std::vector<Pair> canonical(std::vector<Pair> pairs)
{
    std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) {
        return std::tuple(a.dimension, a.birth, a.death) <
               std::tuple(b.dimension, b.birth, b.death);
    });
    return pairs;
}

bool check_persistent_cohomology_small_cloud()
{
    std::vector<std::vector<int>> simplices = {{0},    {1},    {2},    {3},    {0, 1},
                                               {1, 2}, {2, 3}, {3, 0}, {0, 2}, {1, 3}};
    std::vector<double> filt = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2};
    std::vector<int> dims = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};

    CohomologyConfig cfg;
    auto res = computePersistentCohomology(simplices, filt, dims, 1, cfg);

    if (res.all_pairs.empty())
    {
        std::cerr << "expected non-empty pairs\n";
        return false;
    }
    return true;
}

bool check_birth_less_than_death()
{
    std::vector<std::vector<int>> simplices = {{0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}, {0, 1, 2}};
    std::vector<double> filt = {0, 0, 0, 1, 1, 1, 2};
    std::vector<int> dims = {0, 0, 0, 1, 1, 1, 2};

    CohomologyConfig cfg;
    auto res = computePersistentCohomology(simplices, filt, dims, 2, cfg);

    for (const auto &p : res.all_pairs)
    {
        if (!p.isInfinite() && !(p.birth <= p.death + 1e-12))
        {
            std::cerr << "birth<=death violated\n";
            return false;
        }
    }
    return true;
}

bool check_nonnegative_persistence()
{
    std::vector<std::vector<int>> simplices = {{0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}, {0, 1, 2}};
    std::vector<double> filt = {0, 0, 0, 1, 1, 1, 2};
    std::vector<int> dims = {0, 0, 0, 1, 1, 1, 2};

    CohomologyConfig cfg;
    auto res = computePersistentCohomology(simplices, filt, dims, 2, cfg);

    for (const auto &p : res.all_pairs)
    {
        if (!p.isInfinite() && p.lifetime() < -1e-12)
        {
            std::cerr << "negative persistence\n";
            return false;
        }
    }
    return true;
}

bool check_extract_homology_matches()
{
    std::vector<std::vector<int>> simplices = {{0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}, {0, 1, 2}};
    std::vector<double> filt = {0, 0, 0, 1, 1, 1, 2};
    std::vector<int> dims = {0, 0, 0, 1, 1, 1, 2};

    CohomologyConfig cfg;
    auto res = computePersistentCohomology(simplices, filt, dims, 2, cfg);
    auto extracted = extractHomologyFromCohomology(res);

    auto canon_res = canonical(res.all_pairs);
    auto canon_ext = canonical(extracted);

    if (canon_res.size() != canon_ext.size())
        return false;
    for (size_t i = 0; i < canon_res.size(); ++i)
    {
        if (canon_res[i].dimension != canon_ext[i].dimension)
            return false;
    }
    return true;
}

bool check_cohomology_config_customization()
{
    CohomologyConfig cfg;
    cfg.max_dimension = 4;
    cfg.use_clearing = true;
    cfg.use_apparent_pairs = true;
    cfg.num_threads = 2;

    std::vector<std::vector<int>> simplices = {{0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}};
    std::vector<double> filt = {0, 0, 0, 1, 1, 1};
    std::vector<int> dims = {0, 0, 0, 1, 1, 1};

    auto res = computePersistentCohomology(simplices, filt, dims, 1, cfg);
    if (res.all_pairs.empty())
        return false;
    if (res.max_dim < 0)
        return false;
    return true;
}

bool check_optimal_config()
{
    auto cfg = getOptimalCohomologyConfig(4, 1000);
    if (cfg.max_dimension < 0)
        return false;
    return true;
}

bool check_speedup_estimate()
{
    auto est = estimateCohomologySpeedup(4, 1000, true, true);
    if (est.total_speedup < 0.1)
        return false;
    return true;
}

bool check_should_use_cohomology()
{
    if (!shouldUseCohomology(4, 1000, false))
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_persistent_cohomology_small_cloud())
    {
        std::cerr << "FAIL: persistent cohomology small cloud\n";
        return 1;
    }
    if (!check_birth_less_than_death())
    {
        std::cerr << "FAIL: birth less than death\n";
        return 1;
    }
    if (!check_nonnegative_persistence())
    {
        std::cerr << "FAIL: nonnegative persistence\n";
        return 1;
    }
    if (!check_extract_homology_matches())
    {
        std::cerr << "FAIL: extract homology matches\n";
        return 1;
    }
    if (!check_cohomology_config_customization())
    {
        std::cerr << "FAIL: cohomology config customization\n";
        return 1;
    }
    if (!check_optimal_config())
    {
        std::cerr << "FAIL: optimal config\n";
        return 1;
    }
    if (!check_speedup_estimate())
    {
        std::cerr << "FAIL: speedup estimate\n";
        return 1;
    }
    if (!check_should_use_cohomology())
    {
        std::cerr << "FAIL: should use cohomology\n";
        return 1;
    }
    return 0;
}
