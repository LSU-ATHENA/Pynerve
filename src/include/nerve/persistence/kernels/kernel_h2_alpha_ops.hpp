
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace nerve::persistence::h2
{

struct H2Config
{
    double max_radius = 1.0;
    bool use_alpha_complex = true;
    bool force_alpha_complex = false; // Allow non-2D (projects to 2D)
    bool use_bit_parallel = false;
};

struct H2Pair
{
    int birth_index = -1;
    int death_index = -1;
    double birth_time = 0.0;
    double death_time = 0.0;
};

struct Simplex
{
    std::vector<int> vertices;
    int dimension = 0;
    double filtration_value = 0.0;
};

struct AlphaComplex
{
    std::vector<Simplex> simplices;
    double max_radius = 0.0;
    int num_edges = 0;
    int num_triangles = 0;
};

struct H2Result
{
    std::vector<H2Pair> pairs;
    std::string error;

    // Metadata
    int num_delaunay_triangles = 0;
    int num_alpha_simplices = 0;
    double sparsification_ratio = 0.0; // VR / Alpha simplices
    H2Config config;

    // Timing
    double delaunay_time_ms = 0.0;
    double alpha_build_time_ms = 0.0;
    double computation_time_ms = 0.0;
    double total_time_ms = 0.0;
};

struct H2SpeedupEstimate
{
    double sparsification = 1.0;
    double delaunay_speedup = 1.0;
    double total_speedup = 1.0;
};

H2Result computeH2AlphaComplex(const std::vector<double> &points_data, size_t point_dim,
                               size_t num_points, const H2Config &config);

H2Config getOptimalH2Config(size_t num_points, size_t point_dim, double max_radius);
H2SpeedupEstimate estimateH2Speedup(size_t num_points, size_t point_dim);

} // namespace nerve::persistence::h2
