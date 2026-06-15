// This basic implementation provides Eigen-compatible data structures without requiring Eigen3.

#pragma once
#include "nerve/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace nerve::persistence::gradient
{

// Data Structures

class VectorXi
{
    std::vector<int> data_;

public:
    VectorXi() = default;
    explicit VectorXi(int size)
        : data_(size)
    {}
    explicit VectorXi(std::initializer_list<int> init)
        : data_(init)
    {}
    int &operator[](int i) { return data_[i]; }
    const int &operator[](int i) const { return data_[i]; }
    [[nodiscard]] int size() const { return static_cast<int>(data_.size()); }
    [[nodiscard]] bool empty() const { return data_.empty(); }
    void resize(int n) { data_.resize(n); }
    void push_back(int val) { data_.push_back(val); }
    int *data() { return data_.data(); }
    const int *data() const { return data_.data(); }
};

class MatrixXd
{
    std::vector<std::vector<double>> data_;
    int rows_, cols_;

public:
    MatrixXd()
        : rows_(0)
        , cols_(0)
    {}
    MatrixXd(int rows, int cols)
        : data_(rows, std::vector<double>(cols, 0.0))
        , rows_(rows)
        , cols_(cols)
    {}
    static MatrixXd Zero(int rows, int cols) { return MatrixXd(rows, cols); }
    double &operator()(int i, int j) { return data_[i][j]; }
    const double &operator()(int i, int j) const { return data_[i][j]; }
    [[nodiscard]] int rows() const { return rows_; }
    [[nodiscard]] int cols() const { return cols_; }
    [[nodiscard]] int size() const { return rows_ * cols_; }
    MatrixXd &operator+=(const MatrixXd &other);
    MatrixXd operator*(double scalar) const;
    MatrixXd operator+(const MatrixXd &other) const;
    MatrixXd operator-(const MatrixXd &other) const;
    double norm() const;
    // Row access
    std::vector<double> &row(int i) { return data_[i]; }
    const std::vector<double> &row(int i) const { return data_[i]; }
};

struct DifferentiableDiagram
{
    std::vector<std::pair<double, double>> persistence_pairs;
    std::vector<int> dimensions;
    std::vector<VectorXi> birth_simplices;
    std::vector<VectorXi> death_simplices;

    [[nodiscard]] size_t size() const { return persistence_pairs.size(); }
    [[nodiscard]] bool empty() const { return persistence_pairs.empty(); }
};

// Gradient Computation Classes

class PersistenceGradient
{
public:
    PersistenceGradient() = default;

    [[nodiscard]] MatrixXd
    computeGradient(const MatrixXd &points, const DifferentiableDiagram &diagram,
                    const std::vector<std::pair<double, double>> &loss_gradients);

    [[nodiscard]] MatrixXd
    computeTopologyOptimizationGradient(const MatrixXd &points,
                                        const DifferentiableDiagram &current_diagram,
                                        const DifferentiableDiagram &target_diagram);

private:
    [[nodiscard]] MatrixXd computeSimplexGradient(const MatrixXd &points,
                                                  const VectorXi &simplex_vertices,
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

    [[nodiscard]] MatrixXd computeStochasticGradient(
        const MatrixXd &points,
        const std::function<DifferentiableDiagram(const std::vector<int> &indices)> &ph_function,
        const std::vector<std::pair<double, double>> &loss_gradients);

private:
    int batch_size_;
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

    TopologyOptimizer()
        : config_(Config{})
    {}
    explicit TopologyOptimizer(const Config &config);

    [[nodiscard]] MatrixXd
    optimize(MatrixXd points, const DifferentiableDiagram &target_diagram,
             const std::function<DifferentiableDiagram(const MatrixXd &)> &ph_function);

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

} // namespace nerve::persistence::gradient
