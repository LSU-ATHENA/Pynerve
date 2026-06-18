
#include "nerve/config.hpp"
#include "nerve/persistence/core/ph_gradient.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#if NERVE_HAS_EIGEN

namespace
{

using nerve::persistence::gradient::DifferentiableDiagram;
using nerve::persistence::gradient::PersistenceGradient;
using nerve::persistence::gradient::StochasticPersistenceGradient;
using nerve::persistence::gradient::TopologyOptimizer;

std::mt19937_64 make_rng()
{
    std::mt19937_64 rng(42);
    return rng;
}

bool check_differentiable_diagram_default()
{
    DifferentiableDiagram diag;
    if (!diag.persistence_pairs.empty())
        return false;
    if (!diag.dimensions.empty())
        return false;
    if (!diag.birth_simplices.empty())
        return false;
    if (!diag.death_simplices.empty())
        return false;
    if (!diag.empty())
        return false;
    if (diag.size() != 0)
        return false;
    return true;
}

bool check_differentiable_diagram_with_pairs()
{
    DifferentiableDiagram diag;
    diag.persistence_pairs = {{0.0, 1.0}, {0.5, 2.0}};
    diag.dimensions = {0, 1};
    diag.birth_simplices.resize(2);
    diag.death_simplices.resize(2);
    if (diag.empty())
        return false;
    if (diag.size() != 2)
        return false;
    return true;
}

bool check_persistence_gradient_default_construction()
{
    PersistenceGradient pg;
    return true;
}

bool check_persistence_gradient_compute_empty()
{
    PersistenceGradient pg;
    Eigen::MatrixXd points(3, 2);
    points << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;
    DifferentiableDiagram diag;
    std::vector<std::pair<double, double>> loss_gradients;
    auto grad = pg.computeGradient(points, diag, loss_gradients);
    if (grad.rows() != 3)
        return false;
    if (grad.cols() != 2)
        return false;
    if (grad.norm() != 0.0)
        return false;
    return true;
}

bool check_persistence_gradient_compute_with_pairs()
{
    PersistenceGradient pg;
    Eigen::MatrixXd points(2, 2);
    points << 0.0, 0.0, 1.0, 0.0;
    DifferentiableDiagram diag;
    diag.persistence_pairs = {{0.0, 1.0}};
    diag.dimensions = {0};
    Eigen::VectorXi birth(1);
    birth(0) = 0;
    Eigen::VectorXi death(1);
    death(0) = 1;
    diag.birth_simplices.push_back(birth);
    diag.death_simplices.push_back(death);
    std::vector<std::pair<double, double>> loss_gradients = {{1.0, 1.0}};
    auto grad = pg.computeGradient(points, diag, loss_gradients);
    if (grad.rows() != 2)
        return false;
    if (grad.cols() != 2)
        return false;
    if (!std::isfinite(grad.norm()))
        return false;
    return true;
}

bool check_persistence_gradient_topology_optimization()
{
    PersistenceGradient pg;
    Eigen::MatrixXd points(3, 2);
    points << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;
    DifferentiableDiagram current;
    current.persistence_pairs = {{0.0, 1.0}, {0.0, 0.5}};
    current.dimensions = {0, 1};
    current.birth_simplices.resize(2);
    current.death_simplices.resize(2);
    DifferentiableDiagram target;
    target.persistence_pairs = {{0.0, 0.8}};
    target.dimensions = {0};
    target.birth_simplices.resize(1);
    target.death_simplices.resize(1);
    auto grad = pg.computeTopologyOptimizationGradient(points, current, target);
    if (grad.rows() != 3)
        return false;
    if (grad.cols() != 2)
        return false;
    if (!std::isfinite(grad.norm()))
        return false;
    return true;
}

bool check_persistence_gradient_no_matching_pairs()
{
    PersistenceGradient pg;
    Eigen::MatrixXd points(2, 2);
    points << 0.0, 0.0, 1.0, 0.0;
    DifferentiableDiagram diag;
    diag.persistence_pairs = {{0.0, 1.0}};
    diag.dimensions = {0};
    Eigen::VectorXi simplex(1);
    simplex(0) = 0;
    diag.birth_simplices.push_back(simplex);
    diag.death_simplices.push_back(simplex);
    std::vector<std::pair<double, double>> loss_gradients;
    auto grad = pg.computeGradient(points, diag, loss_gradients);
    if (grad.norm() != 0.0)
        return false;
    return true;
}

bool check_stochastic_gradient_construction()
{
    StochasticPersistenceGradient sg(16, 0.01);
    return true;
}

bool check_topology_optimizer_construction()
{
    TopologyOptimizer opt;
    return true;
}

bool check_topology_optimizer_with_config()
{
    TopologyOptimizer::Config cfg;
    cfg.learning_rate = 0.1;
    cfg.max_iterations = 10;
    cfg.convergence_threshold = 1e-4;
    cfg.use_momentum = true;
    cfg.momentum_beta = 0.9;
    TopologyOptimizer opt(cfg);
    return true;
}

bool check_gradient_all_finite()
{
    PersistenceGradient pg;
    Eigen::MatrixXd points(4, 3);
    points << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    DifferentiableDiagram diag;
    diag.persistence_pairs = {{0.0, 1.0}, {0.0, 1.414}};
    diag.dimensions = {0, 1};
    Eigen::VectorXi b0(1);
    b0(0) = 0;
    Eigen::VectorXi d0(1);
    d0(0) = 1;
    Eigen::VectorXi b1(3);
    b1 << 1, 2, 3;
    Eigen::VectorXi d1(1);
    d1(0) = 0;
    diag.birth_simplices.push_back(b0);
    diag.death_simplices.push_back(d0);
    diag.birth_simplices.push_back(b1);
    diag.death_simplices.push_back(d1);
    std::vector<std::pair<double, double>> loss_gradients = {{1.0, 1.0}, {0.5, 0.5}};
    auto grad = pg.computeGradient(points, diag, loss_gradients);
    for (Eigen::Index i = 0; i < grad.size(); ++i)
    {
        if (!std::isfinite(grad(i)))
            return false;
    }
    return true;
}

bool check_gradient_zero_for_unchanged()
{
    PersistenceGradient pg;
    Eigen::MatrixXd points(3, 2);
    points << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;
    DifferentiableDiagram diag;
    diag.persistence_pairs = {{0.0, 1.0}};
    diag.dimensions = {0};
    Eigen::VectorXi simplex(1);
    simplex(0) = 0;
    diag.birth_simplices.push_back(simplex);
    diag.death_simplices.push_back(simplex);
    std::vector<std::pair<double, double>> loss_gradients = {{0.0, 0.0}};
    auto grad = pg.computeGradient(points, diag, loss_gradients);
    if (grad.norm() != 0.0)
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_differentiable_diagram_default())
    {
        std::cerr << "FAIL: differentiable diagram default\n";
        return 1;
    }
    if (!check_differentiable_diagram_with_pairs())
    {
        std::cerr << "FAIL: differentiable diagram with pairs\n";
        return 1;
    }
    if (!check_persistence_gradient_default_construction())
    {
        std::cerr << "FAIL: persistence gradient default construction\n";
        return 1;
    }
    if (!check_persistence_gradient_compute_empty())
    {
        std::cerr << "FAIL: persistence gradient compute empty\n";
        return 1;
    }
    if (!check_persistence_gradient_compute_with_pairs())
    {
        std::cerr << "FAIL: persistence gradient compute with pairs\n";
        return 1;
    }
    if (!check_persistence_gradient_topology_optimization())
    {
        std::cerr << "FAIL: persistence gradient topology optimization\n";
        return 1;
    }
    if (!check_persistence_gradient_no_matching_pairs())
    {
        std::cerr << "FAIL: persistence gradient no matching pairs\n";
        return 1;
    }
    if (!check_stochastic_gradient_construction())
    {
        std::cerr << "FAIL: stochastic gradient construction\n";
        return 1;
    }
    if (!check_topology_optimizer_construction())
    {
        std::cerr << "FAIL: topology optimizer construction\n";
        return 1;
    }
    if (!check_topology_optimizer_with_config())
    {
        std::cerr << "FAIL: topology optimizer with config\n";
        return 1;
    }
    if (!check_gradient_all_finite())
    {
        std::cerr << "FAIL: gradient all finite\n";
        return 1;
    }
    if (!check_gradient_zero_for_unchanged())
    {
        std::cerr << "FAIL: gradient zero for unchanged\n";
        return 1;
    }
    return 0;
}

#else

int main()
{
    return 0;
}

#endif // NERVE_HAS_EIGEN3
