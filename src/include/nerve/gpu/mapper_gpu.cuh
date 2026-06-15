#pragma once

#include <functional>
#include <utility>
#include <vector>

namespace nerve::gpu::mapper
{

void compute_density_filter_gpu(const std::vector<float> &points, int n_points, int dim,
                                int k_neighbors, std::function<void(std::vector<float>)> callback);

void compute_eccentricity_filter_gpu(const std::vector<float> &points, int n_points, int dim,
                                     std::function<void(std::vector<float>)> callback);

void compute_kmeans_clustering_gpu(const std::vector<float> &points, int n_points, int dim, int k,
                                   int max_iterations,
                                   std::function<void(std::vector<int>)> callback);

void compute_nerve_graph_gpu(const std::vector<std::vector<int>> &nodes_cover_sets,
                             std::function<void(std::vector<std::pair<int, int>>)> callback);

} // namespace nerve::gpu::mapper
