#include "nerve/autodiff/autodiff.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/budget.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/differentiable/differentiable_persistence.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::PersistenceBudget;
using nerve::autodiff::Tensor;
using nerve::persistence::DifferentiableConfig;
using nerve::persistence::DifferentiablePersistenceManager;
using nerve::persistence::OptimizationResult;
using nerve::persistence::OptimizationTarget;
using namespace nerve::test;

std::vector<nerve::Pair> canonical(std::vector<nerve::Pair> pairs)
{
    std::sort(pairs.begin(), pairs.end(), [](const nerve::Pair &a, const nerve::Pair &b) {
        return std::tuple(a.dimension, a.birth, a.death) <
               std::tuple(b.dimension, b.birth, b.death);
    });
    return pairs;
}

bool check_differentiable_config_construction()
{
    DifferentiableConfig config;
    config.max_dimension = 1;
    config.landmark_ratio = 0.1;
    config.computeGradients = true;
    config.gradient_epsilon = 1e-6;

    if (config.max_dimension != 1)
    {
        std::cerr << "DifferentiableConfig max_dimension not set\n";
        return false;
    }
    return true;
}

bool check_differentiable_persistence_manager_construction()
{
    PersistenceBudget budget;
    budget.memory_limit_mb = 256;
    budget.time_limit_ms = 500;

    DifferentiablePersistenceManager manager(budget);
    (void)manager;
    return true;
}

bool check_manager_select_algorithm()
{
    DifferentiablePersistenceManager manager;
    auto algo = manager.selectOptimalAlgorithm(10, 1, PersistenceBudget{});
    (void)algo;
    return true;
}

bool check_manager_set_preferred_algorithm()
{
    DifferentiablePersistenceManager manager;
    manager.setPreferredAlgorithm(nerve::persistence::DifferentiableAlgorithm::AUTO_SELECT);
    auto algo = manager.getPreferredAlgorithm();
    if (algo != nerve::persistence::DifferentiableAlgorithm::AUTO_SELECT)
    {
        std::cerr << "preferred algorithm mismatch\n";
        return false;
    }
    return true;
}

bool check_optimize_persistence_no_crash()
{
    PersistenceBudget budget;
    budget.memory_limit_mb = 256;
    budget.time_limit_ms = 100;

    DifferentiablePersistenceManager manager(budget);
    DifferentiableConfig config;
    config.max_dimension = 1;

    Tensor points({0.0, 0.0, 1.0, 0.0}, {2, 2});

    OptimizationTarget target;
    target.target_type = "total_persistence";
    target.target_dimension = 0;
    target.weight = 1.0;

    OptimizationResult result;
    try
    {
        result = manager.optimizePersistence(points, target, 1);
        (void)result;
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool check_compare_algorithms()
{
    PersistenceBudget budget;
    DifferentiablePersistenceManager manager(budget);

    Tensor points({0.0, 0.0, 1.0, 0.0, 0.5, 0.866}, {3, 2});

    auto comparison = manager.compareAlgorithms(points, 1);
    (void)comparison;
    return true;
}

} // namespace

int main()
{
    if (!check_differentiable_config_construction())
    {
        std::cerr << "FAIL: DifferentiableConfig construction\n";
        return 1;
    }
    if (!check_differentiable_persistence_manager_construction())
    {
        std::cerr << "FAIL: DifferentiablePersistenceManager construction\n";
        return 1;
    }
    if (!check_manager_select_algorithm())
    {
        std::cerr << "FAIL: selectOptimalAlgorithm\n";
        return 1;
    }
    if (!check_manager_set_preferred_algorithm())
    {
        std::cerr << "FAIL: setPreferredAlgorithm\n";
        return 1;
    }
    if (!check_optimize_persistence_no_crash())
    {
        std::cerr << "FAIL: optimizePersistence no crash\n";
        return 1;
    }
    if (!check_compare_algorithms())
    {
        std::cerr << "FAIL: compareAlgorithms\n";
        return 1;
    }
    return 0;
}
