#pragma once

#include "nerve/algebra/boundary.hpp"

#include <tuple>
#include <utility>
#include <vector>

namespace nerve::filtration::vr::parallel
{

struct Edge
{
    int u = 0;
    int v = 0;
    float distance = 0.0f;

    bool operator<(const Edge &other) const
    {
        return std::tie(distance, u, v) < std::tie(other.distance, other.u, other.v);
    }
};

struct VRComplex
{
    std::vector<Edge> edges;
    std::vector<std::vector<int>> triangles;
    std::vector<std::vector<int>> tetrahedra;
};

struct VRBenchmark
{
    double sequential_time_ms = 0.0;
    double parallel_time_ms = 0.0;
    double speedup = 1.0;
    size_t num_points = 0;
    size_t num_edges = 0;
    int num_threads = 0;
};

std::vector<Edge> parallelEdgeDetection(const std::vector<nerve::algebra::Point> &points,
                                        float threshold, int num_threads = 0);
VRComplex buildParallelVRComplex(const std::vector<nerve::algebra::Point> &points,
                                 float max_distance, int max_dim = 2);
VRBenchmark benchmarkParallelVR(int n_points, float threshold, int num_threads = 0);

} // namespace nerve::filtration::vr::parallel

namespace nerve::filtration::vr::ann
{

struct ANNBenchmark
{
    double exact_time_ms = 0.0;
    double ann_time_ms = 0.0;
    double recall = 0.0;
    double speedup = 1.0;
    size_t num_points = 0;
    size_t num_neighbors_exact = 0;
    size_t num_neighbors_ann = 0;
};

std::vector<std::pair<int, int>> buildVRWithANN(const std::vector<nerve::algebra::Point> &points,
                                                float threshold, int k_neighbors = 50,
                                                bool use_ann = true);
ANNBenchmark benchmarkANN(int n_points, float threshold, int k = 50);

} // namespace nerve::filtration::vr::ann
