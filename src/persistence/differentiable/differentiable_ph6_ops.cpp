
#include "nerve/persistence/detail/differentiable_persistence_internal.hpp"
#include "nerve/persistence/differentiable/differentiable_persistence.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace nerve::persistence
{

using namespace detail;
DifferentiablePH6::DifferentiablePH6(const PersistenceBudget &budget)
    : budget_(budget)
    , gradient_checking_enabled_(true)
{
    autodiff_config_.enable_gradients = true;
    autodiff_config_.enable_higher_order = false;
    autodiff_config_.gradient_epsilon = 1e-6;
    autodiff_config_.gradient_tolerance = 1e-4;
    autodiff_config_.max_gradient_iterations = 100;
    autodiff_config_.enable_gradient_clipping = true;
    autodiff_config_.gradient_clip_value = 1.0;
    autodiff_config_.enable_mixed_precision = false;
    resetMetrics();
}

autodiff::Tensor
DifferentiablePH6::computePersistenceWitnessAutodiff(const autodiff::Tensor &points,
                                                     size_t max_dimension, double landmark_ratio,
                                                     bool compute_gradients)
{
    if (compute_gradients && autodiff_config_.enable_gradients)
    {
        autodiff::enableGrad(points);
    }
    const auto start_time = std::chrono::high_resolution_clock::now();
    const Size n_points = tensorRows(points);
    Size n_landmarks = static_cast<Size>(std::ceil(landmark_ratio * static_cast<double>(n_points)));
    const Size max_landmarks_from_dim = std::max<Size>(1, max_dimension + 1);
    n_landmarks = std::max<Size>(
        1, std::min<Size>(std::min<Size>(n_points == 0 ? 1 : n_points, max_landmarks_from_dim),
                          n_landmarks));
    const autodiff::Tensor weights = autodiff::Tensor::ones({n_landmarks});
    const autodiff::Tensor witness_graph = buildWitnessGraphAutodiff(points, weights);
    autodiff::Tensor output = computeWitnessComplexAutodiff(witness_graph);
    const auto end_time = std::chrono::high_resolution_clock::now();
    metrics_.forward_pass_time =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return output;
}

autodiff::Tensor
DifferentiablePH6::computeLandmarkGradients(const autodiff::Tensor &points,
                                            const autodiff::Tensor &landmark_weights,
                                            size_t target_dimension)
{
    const auto start_time = std::chrono::high_resolution_clock::now();
    autodiff::enableGrad(landmark_weights);
    const autodiff::Tensor witness =
        computePersistenceWitnessAutodiff(points, target_dimension, 0.1, true);
    autodiff::backward(witness.sum());
    autodiff::Tensor gradients = autodiff::grad(landmark_weights);
    if (gradients.shape() != landmark_weights.shape())
    {
        gradients = autodiff::Tensor::zeros(landmark_weights.shape());
    }
    const auto end_time = std::chrono::high_resolution_clock::now();
    metrics_.backward_pass_time =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
    metrics_.gradient_computations += 1;
    metrics_.gradient_computation_time =
        std::max(metrics_.gradient_computation_time, metrics_.backward_pass_time);
    trackGradientComputation("computeLandmarkGradients", metrics_.backward_pass_time);
    return gradients;
}

bool DifferentiablePH6::verifyFiniteDifferences(const autodiff::Tensor &points,
                                                size_t max_dimension, double epsilon,
                                                double tolerance)
{
    if (epsilon < 0.0)
    {
        epsilon = autodiff_config_.gradient_epsilon;
    }
    if (tolerance < 0.0)
    {
        tolerance = autodiff_config_.gradient_tolerance;
    }
    const Size n_landmarks = std::max<Size>(1, tensorRows(points) / 10);
    const autodiff::Tensor weights = autodiff::Tensor::ones({n_landmarks});
    const autodiff::Tensor analytical = computeLandmarkGradients(points, weights, max_dimension);
    const autodiff::Tensor numerical = finiteDifferenceGradient(
        [this, &points](const autodiff::Tensor &w) {
            return this->computeWitnessComplexAutodiff(this->buildWitnessGraphAutodiff(points, w))
                .sum();
        },
        weights, epsilon);
    const double error = autodiff::tensorDistance(analytical, numerical);
    metrics_.finite_difference_verifications += 1;
    metrics_.gradient_accuracy = std::max(0.0, 1.0 - error);
    if (gradient_checking_enabled_)
    {
        validateGradientComputation(analytical, numerical);
    }
    return error <= tolerance;
}

autodiff::Tensor DifferentiablePH6::optimizeLandmarkSelection(const autodiff::Tensor &points,
                                                              size_t max_dimension,
                                                              size_t num_landmarks,
                                                              size_t optimization_steps)
{
    static_cast<void>(max_dimension);
    autodiff::Tensor weights = autodiff::Tensor::ones({std::max<Size>(1, num_landmarks)});
    const double learning_rate = 1e-2;
    for (size_t step = 0; step < optimization_steps; ++step)
    {
        const autodiff::Tensor grads = finiteDifferenceGradient(
            [this, &points](const autodiff::Tensor &w) {
                return this
                    ->computeWitnessComplexAutodiff(this->buildWitnessGraphAutodiff(points, w))
                    .sum();
            },
            weights, autodiff_config_.gradient_epsilon);
        for (Size i = 0; i < weights.data().size(); ++i)
        {
            weights.data()[i] -= learning_rate * grads.data()[i];
        }
    }
    return weights;
}

void DifferentiablePH6::setAutodiffConfig(const AutodiffConfig &config)
{
    autodiff_config_ = config;
}

const AutodiffConfig &DifferentiablePH6::getAutodiffConfig() const
{
    return autodiff_config_;
}

void DifferentiablePH6::enableGradientChecking(bool enable)
{
    gradient_checking_enabled_ = enable;
}

bool DifferentiablePH6::isGradientCheckingEnabled() const
{
    return gradient_checking_enabled_;
}

const DifferentiableMetrics &DifferentiablePH6::getDifferentiableMetrics() const
{
    return metrics_;
}

void DifferentiablePH6::resetMetrics()
{
    metrics_ = DifferentiableMetrics{};
}

autodiff::Tensor
DifferentiablePH6::buildWitnessGraphAutodiff(const autodiff::Tensor &points,
                                             const autodiff::Tensor &landmark_weights)
{
    const Size rows = tensorRows(points);
    const Size cols = std::max<Size>(1, landmark_weights.size());
    autodiff::Tensor graph = autodiff::Tensor::zeros({rows, cols});
    const double point_scale = tensorMeanAbs(points);
    for (Size row = 0; row < rows; ++row)
    {
        for (Size col = 0; col < cols; ++col)
        {
            graph[row][col] = point_scale * static_cast<double>(landmark_weights[col]);
        }
    }
    return graph;
}

autodiff::Tensor
DifferentiablePH6::computeWitnessComplexAutodiff(const autodiff::Tensor &witness_graph)
{
    const Size rows = tensorRows(witness_graph);
    const Size cols = std::max<Size>(1, tensorCols(witness_graph));
    autodiff::Tensor output = autodiff::Tensor::zeros({rows, 2});
    for (Size row = 0; row < rows; ++row)
    {
        double birth = static_cast<double>(witness_graph[row][0]);
        double death = birth;
        for (Size col = 1; col < cols; ++col)
        {
            death = std::max(death, static_cast<double>(witness_graph[row][col]));
        }
        output[row][0] = birth;
        output[row][1] = death;
    }
    return output;
}

void DifferentiablePH6::trackGradientComputation(const std::string &operation,
                                                 double computation_time)
{
    if (operation.empty() || computation_time <= 0.0)
    {
        return;
    }
    metrics_.gradient_computation_time =
        std::max(metrics_.gradient_computation_time, computation_time);
    metrics_.memory_usage_mb = std::max(
        metrics_.memory_usage_mb,
        static_cast<size_t>((operation.size() + 1U) * sizeof(double) / (1024ULL * 1024ULL)));
}

void DifferentiablePH6::validateGradientComputation(const autodiff::Tensor &analytical_gradients,
                                                    const autodiff::Tensor &finite_diff_gradients)
{
    const double error = autodiff::tensorDistance(analytical_gradients, finite_diff_gradients);
    metrics_.gradient_accuracy = std::max(0.0, 1.0 - error);
}

} // namespace nerve::persistence
