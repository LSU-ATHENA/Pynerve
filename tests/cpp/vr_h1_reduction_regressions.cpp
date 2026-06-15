
#include "nerve/persistence/vr/detail/vr_detail.hpp"
#include "nerve/persistence/vr/vr_h1_reduction_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::persistence::reduced::buildReducedVRForH1;
using nerve::persistence::reduced::computeH1ReducedVR;
using nerve::persistence::reduced::Edge;
using nerve::persistence::reduced::estimateReducedVRSpeedup;
using nerve::persistence::reduced::getOptimalReducedVRH1Config;
using nerve::persistence::reduced::H1Pair;
using nerve::persistence::reduced::H1Result;
using nerve::persistence::reduced::Point;
using nerve::persistence::reduced::ReducedVRH1Config;
using nerve::persistence::reduced::ReducedVRH1Result;
using nerve::persistence::reduced::ReducedVRSpeedup;
using nerve::persistence::reduced::shouldUseReducedVRH1;
using nerve::persistence::reduced::Triangle;

bool check_config_defaults()
{
    ReducedVRH1Config cfg;
    if (std::abs(cfg.max_radius - 1.0) > 1e-12)
        return false;
    if (!cfg.use_cycle_filter)
        return false;
    if (!cfg.use_connectivity_pruning)
        return false;
    if (!cfg.use_triangle_filter)
        return false;
    if (!cfg.preserve_connectivity)
        return false;
    if (cfg.num_threads != 0)
        return false;
    return true;
}

bool check_point_distance()
{
    Point p1;
    p1.id = 0;
    p1.coords = {0.0, 0.0};
    Point p2;
    p2.id = 1;
    p2.coords = {3.0, 4.0};
    double d = p1.distance(p2);
    if (std::abs(d - 5.0) > 1e-12)
        return false;
    return true;
}

bool check_build_reduced_vr_square()
{
    std::vector<double> points = {0, 0, 1, 0, 1, 1, 0, 1};
    ReducedVRH1Config cfg;
    cfg.max_radius = 2.0;
    auto result = buildReducedVRForH1(points, 2, 4, cfg);
    if (result.edges.empty())
    {
        std::cerr << "expected edges in reduced VR\n";
        return false;
    }
    if (result.original_edge_count <= 0)
    {
        std::cerr << "expected positive original edge count\n";
        return false;
    }
    if (result.reduced_edge_count <= 0)
    {
        std::cerr << "expected positive reduced edge count\n";
        return false;
    }
    if (result.edge_reduction_ratio < 0.0)
    {
        std::cerr << "negative edge reduction ratio\n";
        return false;
    }
    return true;
}

bool check_h1_compute_cycle()
{
    std::vector<double> points = {0, 0, 1, 0, 1, 1, 0, 1};
    ReducedVRH1Config cfg;
    cfg.max_radius = 2.0;
    auto vr = buildReducedVRForH1(points, 2, 4, cfg);
    auto h1 = computeH1ReducedVR(vr, cfg);
    if (h1.pairs.empty())
    {
        std::cerr << "expected H1 pairs for square\n";
        return false;
    }
    for (const auto &p : h1.pairs)
    {
        if (p.birth_time > p.death_time + 1e-12)
        {
            std::cerr << "birth <= death violated\n";
            return false;
        }
        if (p.birth_index < 0)
        {
            std::cerr << "negative birth index\n";
            return false;
        }
    }
    return true;
}

bool check_h1_cycle_detection()
{
    std::vector<double> points = {0, 0, 1, 0, 1, 1, 0, 1};
    ReducedVRH1Config cfg;
    cfg.max_radius = 1.5;
    auto vr = buildReducedVRForH1(points, 2, 4, cfg);
    if (vr.edges.size() < 4)
    {
        std::cerr << "expected at least 4 edges for square\n";
        return false;
    }
    bool has_cycle_edges = false;
    for (const auto &e : vr.edges)
    {
        if (e.v1 >= 0 && e.v2 >= 0 && e.v1 < 4 && e.v2 < 4)
        {
            has_cycle_edges = true;
        }
    }
    if (!has_cycle_edges)
    {
        std::cerr << "expected valid cycle edges\n";
        return false;
    }
    return true;
}

bool check_optimal_config()
{
    auto cfg = getOptimalReducedVRH1Config(100, 1.0);
    if (cfg.max_radius < 0)
        return false;
    return true;
}

bool check_speedup_estimate()
{
    auto est = estimateReducedVRSpeedup(100, 1.0);
    if (est.edge_reduction < 0.0)
        return false;
    if (est.total_speedup < 0.0)
        return false;
    return true;
}

bool check_should_use_reduced_vr()
{
    if (shouldUseReducedVRH1(10))
        return false;
    if (!shouldUseReducedVRH1(1000))
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_config_defaults())
    {
        std::cerr << "FAIL: config defaults\n";
        return 1;
    }
    if (!check_point_distance())
    {
        std::cerr << "FAIL: point distance\n";
        return 1;
    }
    if (!check_build_reduced_vr_square())
    {
        std::cerr << "FAIL: build reduced vr square\n";
        return 1;
    }
    if (!check_h1_compute_cycle())
    {
        std::cerr << "FAIL: h1 compute cycle\n";
        return 1;
    }
    if (!check_h1_cycle_detection())
    {
        std::cerr << "FAIL: h1 cycle detection\n";
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
    if (!check_should_use_reduced_vr())
    {
        std::cerr << "FAIL: should use reduced vr\n";
        return 1;
    }
    return 0;
}
