
#include "nerve/persistence/cohomology/cohomology_involuted_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::persistence::computeInvolutedHomology;
using nerve::persistence::estimateInvolutedSpeedup;
using nerve::persistence::getOptimalInvolutedConfig;
using nerve::persistence::InvolutedConfig;
using nerve::persistence::InvolutedResult;
using nerve::persistence::InvolutedSpeedupEstimate;
using nerve::persistence::Pair;
using nerve::persistence::shouldUseInvolution;

std::vector<Pair> canonical(std::vector<Pair> pairs)
{
    std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) {
        return std::tuple(a.dimension, a.birth, a.death) <
               std::tuple(b.dimension, b.birth, b.death);
    });
    return pairs;
}

bool check_involuted_config_defaults()
{
    InvolutedConfig cfg;
    if (cfg.max_dim != 6)
        return false;
    if (!cfg.use_involution)
        return false;
    if (cfg.involution_threshold_dim != 3)
        return false;
    return true;
}

bool check_involuted_result_stores_pairs()
{
    InvolutedResult res;
    Pair p1{0.0, 1.0, 0, 0, 1};
    Pair p2{1.0, 2.0, 1, 1, 2};
    res.all_pairs.push_back(p1);
    res.all_pairs.push_back(p2);
    res.pairs_by_dimension[0].push_back(p1);
    res.pairs_by_dimension[1].push_back(p2);
    if (res.all_pairs.size() != 2)
        return false;
    if (res.pairs_by_dimension[0].size() != 1)
        return false;
    if (res.pairs_by_dimension[1].size() != 1)
        return false;
    return true;
}

bool check_speedup_estimate()
{
    auto est_dim2 = estimateInvolutedSpeedup(2, 100);
    auto est_dim4 = estimateInvolutedSpeedup(4, 100);
    if (est_dim2.involution_speedup < 0.1)
        return false;
    if (est_dim4.involution_speedup < 0.1)
        return false;
    return true;
}

bool check_involution_same_as_standard()
{
    std::vector<std::vector<int>> simplices = {{0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}, {0, 1, 2}};
    std::vector<double> filt = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0};
    std::vector<int> dims = {0, 0, 0, 1, 1, 1, 2};

    InvolutedConfig no_inv;
    no_inv.use_involution = false;
    auto res_no = computeInvolutedHomology(simplices, filt, dims, 2, no_inv);

    InvolutedConfig with_inv;
    with_inv.use_involution = true;
    auto res_inv = computeInvolutedHomology(simplices, filt, dims, 2, with_inv);

    auto canon_no = canonical(res_no.all_pairs);
    auto canon_inv = canonical(res_inv.all_pairs);

    if (canon_no.size() != canon_inv.size())
        return false;
    for (size_t i = 0; i < canon_no.size(); ++i)
    {
        if (canon_no[i].dimension != canon_inv[i].dimension)
            return false;
        if (std::abs(canon_no[i].birth - canon_inv[i].birth) > 1e-12)
            return false;
        if (canon_no[i].isInfinite() != canon_inv[i].isInfinite())
            return false;
        if (!canon_no[i].isInfinite() && std::abs(canon_no[i].death - canon_inv[i].death) > 1e-12)
            return false;
    }
    return true;
}

bool check_optimal_config()
{
    auto cfg = getOptimalInvolutedConfig(4, 1000);
    if (cfg.max_dim < 0)
        return false;
    return true;
}

bool check_should_use_involution()
{
    if (shouldUseInvolution(2))
        return false;
    if (!shouldUseInvolution(3))
        return false;
    if (!shouldUseInvolution(6))
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_involuted_config_defaults())
    {
        std::cerr << "FAIL: involuted config defaults\n";
        return 1;
    }
    if (!check_involuted_result_stores_pairs())
    {
        std::cerr << "FAIL: involuted result stores pairs\n";
        return 1;
    }
    if (!check_speedup_estimate())
    {
        std::cerr << "FAIL: speedup estimate\n";
        return 1;
    }
    if (!check_involution_same_as_standard())
    {
        std::cerr << "FAIL: involution same as standard\n";
        return 1;
    }
    if (!check_optimal_config())
    {
        std::cerr << "FAIL: optimal config\n";
        return 1;
    }
    if (!check_should_use_involution())
    {
        std::cerr << "FAIL: should use involution\n";
        return 1;
    }
    return 0;
}
