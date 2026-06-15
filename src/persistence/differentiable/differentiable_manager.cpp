
#include "nerve/persistence/detail/differentiable_persistence_internal.hpp"
#include "nerve/persistence/differentiable/differentiable_persistence.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace nerve::persistence
{

using namespace detail;
DifferentiablePersistenceManager::DifferentiablePersistenceManager(const PersistenceBudget &budget)
    : budget_(budget)
    , preferred_algorithm_(DifferentiableAlgorithm::AUTO_SELECT)
    , ph5_(std::make_unique<DifferentiablePH5>(budget))
    , ph6_(std::make_unique<DifferentiablePH6>(budget))
{}

autodiff::Tensor
DifferentiablePersistenceManager::computePersistenceAutodiff(const autodiff::Tensor &points,
                                                             const DifferentiableConfig &config)
{
    DifferentiableAlgorithm algorithm = config.algorithm;
    if (algorithm == DifferentiableAlgorithm::AUTO_SELECT)
    {
        algorithm = preferred_algorithm_ == DifferentiableAlgorithm::AUTO_SELECT
                        ? selectOptimalAlgorithm(tensorRows(points), config.max_dimension, budget_)
                        : preferred_algorithm_;
    }
    if (algorithm == DifferentiableAlgorithm::PH5_COHOMOLOGY)
    {
        return computeWithPh5(points, config);
    }
    return computeWithPh6(points, config);
}

void DifferentiablePersistenceManager::setPreferredAlgorithm(DifferentiableAlgorithm algorithm)
{
    preferred_algorithm_ = algorithm;
}

DifferentiableAlgorithm DifferentiablePersistenceManager::getPreferredAlgorithm() const
{
    return preferred_algorithm_;
}

DifferentiableAlgorithm
DifferentiablePersistenceManager::selectOptimalAlgorithm(size_t num_points, size_t max_dimension,
                                                         const PersistenceBudget &budget)
{
    const double dimension_scale = static_cast<double>(std::max<size_t>(1, max_dimension + 1));
    const double dense_memory_mb = static_cast<double>(num_points) *
                                   static_cast<double>(num_points) * dimension_scale *
                                   sizeof(double) / (1024.0 * 1024.0);
    const bool ph5_exceeds_budget = budget.strict_budget_enforcement &&
                                    dense_memory_mb > static_cast<double>(budget.memory_limit_mb);
    if (ph5_exceeds_budget)
    {
        return DifferentiableAlgorithm::PH6_WITNESS;
    }
    if (budget.time_limit_ms < 250 && num_points >= 256)
    {
        return DifferentiableAlgorithm::PH6_WITNESS;
    }
    if (max_dimension <= 2 && num_points >= 512)
    {
        return DifferentiableAlgorithm::PH5_COHOMOLOGY;
    }
    if (max_dimension >= 3 || num_points < 512)
    {
        return DifferentiableAlgorithm::PH6_WITNESS;
    }
    return DifferentiableAlgorithm::PH5_COHOMOLOGY;
}

AlgorithmComparison
DifferentiablePersistenceManager::compareAlgorithms(const autodiff::Tensor &points,
                                                    size_t max_dimension)
{
    AlgorithmComparison comparison;
    DifferentiableConfig config{};
    config.max_dimension = max_dimension;
    config.landmark_ratio = 0.1;
    config.computeGradients = false;
    config.verifyGradients = false;
    config.gradient_epsilon = 1e-6;
    config.gradient_tolerance = 1e-4;

    const auto start_ph5 = std::chrono::high_resolution_clock::now();
    config.algorithm = DifferentiableAlgorithm::PH5_COHOMOLOGY;
    const autodiff::Tensor ph5_result = computePersistenceAutodiff(points, config);
    const auto end_ph5 = std::chrono::high_resolution_clock::now();

    const auto start_ph6 = std::chrono::high_resolution_clock::now();
    config.algorithm = DifferentiableAlgorithm::PH6_WITNESS;
    const autodiff::Tensor ph6_result = computePersistenceAutodiff(points, config);
    const auto end_ph6 = std::chrono::high_resolution_clock::now();

    const double ph5_ms = std::chrono::duration<double, std::milli>(end_ph5 - start_ph5).count();
    const double ph6_ms = std::chrono::duration<double, std::milli>(end_ph6 - start_ph6).count();
    comparison.computation_times[DifferentiableAlgorithm::PH5_COHOMOLOGY] = ph5_ms;
    comparison.computation_times[DifferentiableAlgorithm::PH6_WITNESS] = ph6_ms;
    comparison.memory_usage[DifferentiableAlgorithm::PH5_COHOMOLOGY] =
        static_cast<double>(ph5_result.size() * sizeof(double));
    comparison.memory_usage[DifferentiableAlgorithm::PH6_WITNESS] =
        static_cast<double>(ph6_result.size() * sizeof(double));
    comparison.gradient_accuracies[DifferentiableAlgorithm::PH5_COHOMOLOGY] =
        std::max(0.0, 1.0 - std::abs(tensorMeanAbs(ph5_result)));
    comparison.gradient_accuracies[DifferentiableAlgorithm::PH6_WITNESS] =
        std::max(0.0, 1.0 - std::abs(tensorMeanAbs(ph6_result)));
    comparison.recommended_algorithm = ph5_ms <= ph6_ms ? DifferentiableAlgorithm::PH5_COHOMOLOGY
                                                        : DifferentiableAlgorithm::PH6_WITNESS;
    comparison.recommendation_reason = "Lower observed runtime";
    return comparison;
}

std::vector<autodiff::Tensor> DifferentiablePersistenceManager::computeBatchPersistence(
    const std::vector<autodiff::Tensor> &point_clouds, const DifferentiableConfig &config)
{
    std::vector<autodiff::Tensor> output;
    output.reserve(point_clouds.size());
    for (const auto &points : point_clouds)
    {
        output.push_back(computePersistenceAutodiff(points, config));
    }
    return output;
}

OptimizationResult DifferentiablePersistenceManager::optimizePersistence(
    const autodiff::Tensor &points, const OptimizationTarget &target, size_t optimization_steps)
{
    OptimizationResult result{};
    result.accelerated_points = points;
    result.optimization_steps = optimization_steps;
    result.converged = false;
    result.convergence_reason = "max_steps_reached";

    const auto start = std::chrono::high_resolution_clock::now();
    DifferentiableConfig config{};
    config.max_dimension = target.target_dimension;
    config.landmark_ratio = 0.1;
    config.computeGradients = false;
    config.verifyGradients = false;
    config.gradient_epsilon = 1e-6;
    config.gradient_tolerance = 1e-4;
    config.algorithm = DifferentiableAlgorithm::AUTO_SELECT;

    const double epsilon = 1e-4;
    const double learning_rate = 1e-2;
    for (size_t step = 0; step < optimization_steps; ++step)
    {
        const autodiff::Tensor output =
            computePersistenceAutodiff(result.accelerated_points, config);
        const double objective = evaluateTargetObjective(output, target);
        result.objective_history.push_back(objective);
        if (objective <= 1e-10)
        {
            result.converged = true;
            result.convergence_reason = "objective_tolerance";
            break;
        }

        autodiff::Tensor gradients = autodiff::Tensor::zeros(result.accelerated_points.shape());
        for (Size i = 0; i < result.accelerated_points.data().size(); ++i)
        {
            autodiff::Tensor plus = result.accelerated_points;
            autodiff::Tensor minus = result.accelerated_points;
            plus.data()[i] += epsilon;
            minus.data()[i] -= epsilon;
            const double objective_plus =
                evaluateTargetObjective(computePersistenceAutodiff(plus, config), target);
            const double objective_minus =
                evaluateTargetObjective(computePersistenceAutodiff(minus, config), target);
            gradients.data()[i] = (objective_plus - objective_minus) / (2.0 * epsilon);
        }

        for (Size i = 0; i < result.accelerated_points.data().size(); ++i)
        {
            result.accelerated_points.data()[i] -= learning_rate * gradients.data()[i];
        }
    }

    const autodiff::Tensor final_output =
        computePersistenceAutodiff(result.accelerated_points, config);
    result.final_objective_value = evaluateTargetObjective(final_output, target);
    const auto end = std::chrono::high_resolution_clock::now();
    result.optimization_time = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

autodiff::Tensor
DifferentiablePersistenceManager::computeWithPh5(const autodiff::Tensor &points,
                                                 const DifferentiableConfig &config)
{
    ph5_->setAutodiffConfig(config.autodiff_config);
    return ph5_->computePersistenceCohomologyAutodiff(points, config.max_dimension,
                                                      config.computeGradients);
}

autodiff::Tensor
DifferentiablePersistenceManager::computeWithPh6(const autodiff::Tensor &points,
                                                 const DifferentiableConfig &config)
{
    ph6_->setAutodiffConfig(config.autodiff_config);
    return ph6_->computePersistenceWitnessAutodiff(points, config.max_dimension,
                                                   config.landmark_ratio, config.computeGradients);
}

} // namespace nerve::persistence
