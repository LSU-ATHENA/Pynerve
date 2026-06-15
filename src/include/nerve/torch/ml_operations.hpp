// ML Operations for Persistence Diagrams
// C++20 header for vectorization, statistics, and kernel methods

#pragma once

#include <torch/torch.h>

#include <string>
#include <utility>
#include <vector>

namespace nerve::torch
{

// Vectorization Operations

at::Tensor ml_persistence_image(const at::Tensor &diagram, int64_t resolution_birth = 20,
                                int64_t resolution_death = 20, double sigma = 0.5,
                                double birth_min = 0.0, double birth_max = 0.0,
                                double death_min = 0.0, double death_max = 0.0,
                                const std::string &weight_fn = "persistence");

at::Tensor ml_persistence_landscape(const at::Tensor &diagram, int64_t k = 5,
                                    int64_t num_samples = 100, double x_min = 0.0,
                                    double x_max = 0.0);

at::Tensor ml_persistence_silhouette(const at::Tensor &diagram, int64_t num_samples = 100,
                                     double x_min = 0.0, double x_max = 0.0,
                                     const std::string &weight_fn = "persistence");

at::Tensor ml_heat_kernel_signature(const at::Tensor &diagram, int64_t num_samples = 100,
                                    double sigma = 0.1,
                                    const at::Tensor &t_values = at::logspace(-2, 0, 10));

at::Tensor ml_birth_death_curve(const at::Tensor &diagram, int64_t num_bins = 50,
                                const std::string &statistic = "count");

// Statistical Operations

double ml_total_persistence(const at::Tensor &diagram, int64_t dim = -1, double p = 1.0);

double ml_mean_persistence(const at::Tensor &diagram, int64_t dim = -1);

double ml_max_persistence(const at::Tensor &diagram, int64_t dim = -1);

double ml_persistence_variance(const at::Tensor &diagram, int64_t dim = -1);

double ml_persistence_entropy(const at::Tensor &diagram, int64_t dim = -1,
                              double base = std::exp(1));

int64_t ml_number_of_features(const at::Tensor &diagram, int64_t dim = -1,
                              double min_persistence = 0.0);

at::Tensor ml_betti_curve(const at::Tensor &diagram, int64_t num_samples = 100, int64_t dim = -1);

double ml_amplitude(const at::Tensor &diagram, const std::string &metric = "persistence",
                    double p = 1.0, int64_t dim = -1);

std::vector<std::pair<std::string, double>>
ml_all_statistics(const at::Tensor &diagram, const std::vector<int64_t> &dims = {0, 1});

at::Tensor ml_extract_features(const at::Tensor &diagram,
                               const std::vector<int64_t> &dims = {0, 1});

// Kernel Operations

double ml_gaussian_kernel(const at::Tensor &d1, const at::Tensor &d2, double sigma = 0.5,
                          const std::string &distance_metric = "euclidean");

double ml_persistence_scale_space_kernel(const at::Tensor &d1, const at::Tensor &d2,
                                         double sigma = 0.5, double weight = 0.5);

double ml_sliced_wasserstein_kernel(const at::Tensor &d1, const at::Tensor &d2,
                                    int64_t num_slices = 10, double sigma = 0.5);

double ml_persistence_fisher_kernel(const at::Tensor &d1, const at::Tensor &d2, double sigma = 0.5);

double ml_linear_kernel(const at::Tensor &d1, const at::Tensor &d2, int64_t num_samples = 100);

at::Tensor ml_compute_kernel_matrix(const std::vector<at::Tensor> &diagrams,
                                    const std::string &kernel = "gaussian", double sigma = 0.5,
                                    int64_t num_slices = 10);

at::Tensor ml_normalize_kernel_matrix(const at::Tensor &K);

at::Tensor ml_center_kernel_matrix(const at::Tensor &K);

} // namespace nerve::torch
