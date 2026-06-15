
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_h1_reduction_ops.hpp"

#include <chrono>
#include <unordered_set>
#include <vector>

namespace nerve::persistence::h1
{

struct H1Config
{
    double max_radius = 1.0;
    bool use_cycle_filter = true;
    bool use_connectivity_pruning = true;
    bool use_triangle_filter = true;
    bool use_bit_parallel = false;
    bool use_clearing = true;
};

struct H1Pair
{
    int birth_index;
    int death_index;
    double birth_time;
    double death_time;
};

struct H1Result
{
    std::vector<H1Pair> pairs;
    std::unordered_set<int> cleared_for_h2; // Triangles to skip in H2

    // Metadata
    int num_edges;
    int num_triangles;
    double edge_reduction_ratio;
    int num_reductions;
    int num_cleared;
    std::string algorithm_used;
    H1Config config;

    // Timing
    double build_time_ms;
    double coboundary_time_ms;
    double computation_time_ms;
    double total_time_ms;
};

struct H1SpeedupEstimate
{
    double edge_reduction;
    double layout_speedup;
    double clearing_speedup;
    double bit_parallel_speedup;
    double total_speedup;
};

H1Result computeH1Persistence(const std::vector<double> &points, size_t point_dim,
                              size_t num_points, const H1Config &config);

H1Config getOptimalH1Config(size_t num_points, double max_radius);
H1SpeedupEstimate estimateH1Speedup(size_t num_points);

} // namespace nerve::persistence::h1
