//
// Based on: "Learning Sheaf Laplacian" (arXiv:2501.19207, 2025)

#pragma once
#include "nerve/config.hpp"
#include "nerve/sheaf/sheaf_laplacian.hpp"

#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <concepts>
#include <span>
#include <vector>

namespace nerve
{
namespace sheaf
{
namespace learning
{

constinit const double DEFAULT_LEARNING_RATE = 0.01;
constinit const int DEFAULT_MAX_ITERATIONS = 1000;
constinit const double DEFAULT_TOLERANCE = 1e-6;

template <typename T>
concept LearnableSheaf = requires(T s) {
    { s.num_nodes } -> std::convertible_to<size_t>;
};

/**
 * @brief Sheaf learning configuration
 */
struct SheafLearningConfig
{
    double learning_rate = DEFAULT_LEARNING_RATE;
    int max_iterations = DEFAULT_MAX_ITERATIONS;
    double convergence_tolerance = DEFAULT_TOLERANCE;
    bool use_closed_form = true; // O(n^3) vs O(n^6) SDP
    bool optimize_locally = true;
    bool verbose = false;
};

/**
 * @brief Learning result with learned sheaf
 */
struct SheafLearningResult
{
    Eigen::SparseMatrix<double> learned_sheaf_laplacian;
    std::vector<Eigen::MatrixXd> restriction_maps;
    double final_total_variation;
    int iterations_used;
    bool converged = false;
    double compute_time_ms;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] double getDirichletEnergy() const;
};

/**
 * @brief Sheaf Learner with closed-form optimization
 */
class SheafLearner
{
public:
    explicit SheafLearner(const SheafLearningConfig &config = {});

    // Closed-form learning (O(n^3))
    [[nodiscard]] SheafLearningResult
    learnSheafClosedForm(const Eigen::SparseMatrix<double> &graph_adjacency,
                         std::span<const Eigen::MatrixXd> node_data,
                         std::span<const size_t> node_dimensions);

    // Local per-edge optimization
    [[nodiscard]] SheafLearningResult
    learnSheafLocal(const Eigen::SparseMatrix<double> &graph_adjacency,
                    std::span<const Eigen::MatrixXd> node_data,
                    std::span<const size_t> node_dimensions);

    // Compute optimal restriction map for single edge
    // F_ij = (x_i x_j^T)(x_j x_j^T)^{-1}
    [[nodiscard]] Eigen::MatrixXd computeOptimalRestrictionMap(const Eigen::MatrixXd &data_i,
                                                               const Eigen::MatrixXd &data_j) const;

    // Compute total variation: TV = sum ||x_i - F_ij x_j||^2
    [[nodiscard]] double
    computeTotalVariation(const Eigen::SparseMatrix<double> &graph_adjacency,
                          std::span<const Eigen::MatrixXd> node_data,
                          std::span<const Eigen::MatrixXd> restriction_maps) const;

    // Construct sheaf Laplacian from learned maps
    [[nodiscard]] Eigen::SparseMatrix<double>
    constructSheafLaplacian(const Eigen::SparseMatrix<double> &graph_adjacency,
                            std::span<const size_t> node_dimensions,
                            std::span<const Eigen::MatrixXd> restriction_maps) const;

    void setConfig(const SheafLearningConfig &config);
    [[nodiscard]] SheafLearningConfig getConfig() const { return config_; }

private:
    SheafLearningConfig config_;

    [[nodiscard]] bool validateInputs(const Eigen::SparseMatrix<double> &graph_adjacency,
                                      std::span<const Eigen::MatrixXd> node_data,
                                      std::span<const size_t> node_dimensions) const;
};

/**
 * @brief High-level API: Learn sheaf from graph signal
 */
[[nodiscard]] std::unique_ptr<SheafLaplacianRuntime>
learnSheafFromGraphSignal(std::span<const std::pair<uint32_t, uint32_t>> graph_edges,
                          std::span<const std::vector<double>> node_signals,
                          const SheafLearningConfig &config = {});

/**
 * @brief Benchmark sheaf learning methods
 */
struct SheafLearningBenchmark
{
    double closed_form_time_ms;
    double sdp_time_ms;
    double speedup_factor;
    double tv_closed_form;
    double tv_sdp;
    double accuracy_ratio;
};

[[nodiscard]] SheafLearningBenchmark
benchmarkSheafLearning(const Eigen::SparseMatrix<double> &graph,
                       std::span<const Eigen::MatrixXd> node_data,
                       std::span<const size_t> dimensions);

// Closed-form and SDP benchmarks are workload-dependent.

} // namespace learning
} // namespace sheaf
} // namespace nerve

#else
#include <cstddef>

namespace nerve::sheaf::learning
{

struct SheafLearningConfig
{
    double learning_rate = 0.01;
    int max_iterations = 1000;
    double convergence_tolerance = 1e-6;
    bool use_closed_form = true;
    bool optimize_locally = true;
    bool verbose = false;
};

struct SheafLearningResult;
struct SheafLearningBenchmark;
class SheafLearner;

} // namespace nerve::sheaf::learning

#endif
