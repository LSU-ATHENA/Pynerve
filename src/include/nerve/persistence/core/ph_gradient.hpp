
#pragma once

#include "nerve/config.hpp"

#if HAS_EIGEN && __has_include(<Eigen/Core>)
#include "nerve/core_types.hpp"

#include <Eigen/Core>
#include <functional>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace nerve
{
namespace persistence
{
namespace gradient
{

// Data Structures

struct DifferentiableDiagram
{
    std::vector<std::pair<double, double>> persistence_pairs;
    std::vector<int> dimensions;
    std::vector<Eigen::VectorXi> birth_simplices;
    std::vector<Eigen::VectorXi> death_simplices;

    [[nodiscard]] size_t size() const { return persistence_pairs.size(); }
    [[nodiscard]] bool empty() const { return persistence_pairs.empty(); }
};

// Gradient Computation Classes

class PersistenceGradient
{
public:
    PersistenceGradient() = default;

    [[nodiscard]] Eigen::MatrixXd
    computeGradient(const Eigen::MatrixXd &points, const DifferentiableDiagram &diagram,
                    const std::vector<std::pair<double, double>> &loss_gradients);

    [[nodiscard]] Eigen::MatrixXd
    computeTopologyOptimizationGradient(const Eigen::MatrixXd &points,
                                        const DifferentiableDiagram &current_diagram,
                                        const DifferentiableDiagram &target_diagram);

private:
    [[nodiscard]] Eigen::MatrixXd computeSimplexGradient(const Eigen::MatrixXd &points,
                                                         const Eigen::VectorXi &simplex_vertices,
                                                         double filtration_value);

    [[nodiscard]] std::vector<std::pair<int, int>>
    matchPersistencePairs(const DifferentiableDiagram &current,
                          const DifferentiableDiagram &target);

    [[nodiscard]] std::vector<std::pair<int, int>>
    hungarianAlgorithm(const std::vector<std::vector<double>> &cost, size_t n_current,
                       size_t n_target);
};

class StochasticPersistenceGradient
{
public:
    explicit StochasticPersistenceGradient(int batch_size = 32, double learning_rate = 0.01);

    [[nodiscard]] Eigen::MatrixXd computeStochasticGradient(
        const Eigen::MatrixXd &points,
        const std::function<DifferentiableDiagram(const std::vector<int> &indices)> &ph_function,
        const std::vector<std::pair<double, double>> &loss_gradients);

private:
    int batch_size_;
    double learning_rate_;
    std::mt19937_64 rng_{42};

    [[nodiscard]] std::vector<int> sampleBatch(int n_points);
};

class TopologyOptimizer
{
public:
    struct Config
    {
        double learning_rate = 0.01;
        int max_iterations = 1000;
        double convergence_threshold = 1e-6;
        bool use_momentum = true;
        double momentum_beta = 0.9;
    };

    TopologyOptimizer();
    explicit TopologyOptimizer(const Config &config);

    [[nodiscard]] Eigen::MatrixXd
    optimize(Eigen::MatrixXd points, const DifferentiableDiagram &target_diagram,
             const std::function<DifferentiableDiagram(const Eigen::MatrixXd &)> &ph_function);

private:
    Config config_;
};

// High-level API

[[nodiscard]] DifferentiableDiagram computeDifferentiable(std::span<const double> points,
                                                          size_t n_points, size_t point_dim,
                                                          double max_distance, int max_dim);

[[nodiscard]] std::vector<double> persistenceBackward(std::span<const double> points,
                                                      size_t n_points, size_t point_dim,
                                                      const DifferentiableDiagram &diagram,
                                                      std::span<const double> grad_birth,
                                                      std::span<const double> grad_death);

} // namespace gradient
} // namespace persistence
} // namespace nerve

#else
#include "nerve/persistence/core/ph_gradient_basic.hpp"
#endif
