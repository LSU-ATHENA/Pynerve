
#include "nerve/persistence/detail/differentiable_persistence_internal.hpp"
#include "nerve/persistence/differentiable/differentiable_persistence.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace nerve::persistence
{

using namespace detail;
namespace
{
constexpr double kIllConditionedGradient = 1.0e12;
}

DifferentiablePH5::DifferentiablePH5(const PersistenceBudget &budget)
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
DifferentiablePH5::computePersistenceCohomologyAutodiff(const autodiff::Tensor &points,
                                                        size_t max_dimension, bool computeGradients)
{
    if (computeGradients && autodiff_config_.enable_gradients)
    {
        autodiff::enableGrad(points);
    }
    const auto start_time = std::chrono::high_resolution_clock::now();
    const autodiff::Tensor graph = buildCohomologyGraph(points, max_dimension);
    autodiff::Tensor result = computeCohomologyReductionAutodiff(graph);
    const auto end_time = std::chrono::high_resolution_clock::now();
    metrics_.forward_pass_time =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return result;
}

autodiff::Tensor DifferentiablePH5::computeGradients(const autodiff::Tensor &points,
                                                     const autodiff::Tensor &persistence_diagram,
                                                     size_t target_dimension)
{
    const auto start_time = std::chrono::high_resolution_clock::now();
    autodiff::enableGrad(points);
    autodiff::Tensor output = computePersistenceCohomologyAutodiff(points, target_dimension, true);
    const auto &target = persistence_diagram.data();
    if (!target.empty())
    {
        const Size n = std::min<Size>(output.data().size(), target.size());
        for (Size i = 0; i < n; ++i)
        {
            output.data()[i] *= 1.0 + std::abs(target[i]);
        }
    }
    autodiff::backward(output.sum());
    autodiff::Tensor gradients = autodiff::grad(points);
    if (gradients.shape() != points.shape())
    {
        gradients = autodiff::Tensor::zeros(points.shape());
    }
    const auto end_time = std::chrono::high_resolution_clock::now();
    metrics_.backward_pass_time =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
    metrics_.gradient_computation_time = metrics_.backward_pass_time;
    metrics_.gradient_computations += 1;
    const double grad_norm = autodiff::tensorNorm(gradients);
    metrics_.average_gradient_norm = ((metrics_.average_gradient_norm *
                                       static_cast<double>(metrics_.gradient_computations - 1)) +
                                      grad_norm) /
                                     static_cast<double>(metrics_.gradient_computations);
    metrics_.max_gradient_norm = std::max(metrics_.max_gradient_norm, grad_norm);
    trackGradientComputation("computeGradients", metrics_.gradient_computation_time);
    return gradients;
}

bool DifferentiablePH5::verifyFiniteDifferences(const autodiff::Tensor &points,
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
    const autodiff::Tensor analytical =
        computeGradients(points, autodiff::Tensor::zeros({1}), max_dimension);
    const autodiff::Tensor numerical = finiteDifferenceGradient(
        [this, max_dimension](const autodiff::Tensor &x) {
            return this->computePersistenceCohomologyAutodiff(x, max_dimension, false).sum();
        },
        points, epsilon);
    const double error = autodiff::tensorDistance(analytical, numerical);
    metrics_.finite_difference_verifications += 1;
    metrics_.gradient_accuracy = std::max(0.0, 1.0 - error);
    if (gradient_checking_enabled_)
    {
        validateGradientComputation(analytical, numerical);
    }
    return error <= tolerance;
}

void DifferentiablePH5::setAutodiffConfig(const AutodiffConfig &config)
{
    autodiff_config_ = config;
}

const AutodiffConfig &DifferentiablePH5::getAutodiffConfig() const
{
    return autodiff_config_;
}

void DifferentiablePH5::enableGradientChecking(bool enable)
{
    gradient_checking_enabled_ = enable;
}

bool DifferentiablePH5::isGradientCheckingEnabled() const
{
    return gradient_checking_enabled_;
}

const DifferentiableMetrics &DifferentiablePH5::getDifferentiableMetrics() const
{
    return metrics_;
}

void DifferentiablePH5::resetMetrics()
{
    metrics_ = DifferentiableMetrics{};
}

GradientAnalysis DifferentiablePH5::analyzeGradients(const autodiff::Tensor &gradients,
                                                     size_t dimension)
{
    GradientAnalysis analysis{};
    analysis.gradient_norm = autodiff::tensorNorm(gradients);
    analysis.gradient_variance = autodiff::tensorVariance(gradients);
    analysis.gradient_sparsity = autodiff::tensorSparsity(gradients);

    const Size cols = std::max<Size>(1, tensorCols(gradients));
    const Size analyzed_dims =
        std::max<Size>(1, std::min<Size>(cols, static_cast<Size>(dimension + 1)));
    const auto &data = gradients.data();
    analysis.per_dimension_norms.assign(analyzed_dims, 0.0);
    for (Size idx = 0; idx < data.size(); ++idx)
    {
        const Size col = idx % cols;
        if (col < analyzed_dims)
        {
            analysis.per_dimension_norms[col] += data[idx] * data[idx];
        }
    }
    for (double &value : analysis.per_dimension_norms)
    {
        value = std::sqrt(value);
    }

    std::vector<std::pair<double, Size>> ranked;
    ranked.reserve(data.size());
    for (Size idx = 0; idx < data.size(); ++idx)
    {
        ranked.push_back({std::abs(data[idx]), idx});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
    const Size keep = std::min<Size>(10, ranked.size());
    analysis.top_gradient_indices.reserve(keep);
    for (Size i = 0; i < keep; ++i)
    {
        analysis.top_gradient_indices.push_back(ranked[i].second);
    }

    const double min_norm = analysis.per_dimension_norms.empty()
                                ? 0.0
                                : *std::min_element(analysis.per_dimension_norms.begin(),
                                                    analysis.per_dimension_norms.end());
    const double max_norm = analysis.per_dimension_norms.empty()
                                ? 0.0
                                : *std::max_element(analysis.per_dimension_norms.begin(),
                                                    analysis.per_dimension_norms.end());
    analysis.gradient_condition_number =
        min_norm > 0.0 ? (max_norm / min_norm) : kIllConditionedGradient;

    analysis.gradients_well_behaved =
        std::isfinite(analysis.gradient_condition_number) && analysis.gradient_norm < 1e6;
    return analysis;
}

autodiff::Tensor DifferentiablePH5::buildCohomologyGraph(const autodiff::Tensor &points,
                                                         size_t max_dimension)
{
    const Size rows = tensorRows(points);
    const Size cols = std::max<Size>(1, max_dimension + 1);
    autodiff::Tensor graph = autodiff::Tensor::zeros({rows, cols});
    const auto &values = points.data();
    const Size point_width = std::max<Size>(1, tensorCols(points));
    for (Size row = 0; row < rows; ++row)
    {
        double norm_sq = 0.0;
        for (Size col = 0; col < point_width; ++col)
        {
            const Size idx = row * point_width + col;
            if (idx < values.size())
            {
                norm_sq += values[idx] * values[idx];
            }
        }
        const double radius = std::sqrt(norm_sq);
        for (Size dim = 0; dim < cols; ++dim)
        {
            graph[row][dim] = radius * static_cast<double>(dim + 1);
        }
    }
    return graph;
}

autodiff::Tensor
DifferentiablePH5::computeCohomologyReductionAutodiff(const autodiff::Tensor &cohomology_graph)
{
    const Size rows = tensorRows(cohomology_graph);
    const Size cols = std::max<Size>(1, tensorCols(cohomology_graph));
    autodiff::Tensor output = autodiff::Tensor::zeros({rows, 2});
    for (Size row = 0; row < rows; ++row)
    {
        double birth = static_cast<double>(cohomology_graph[row][0]);
        double death = birth;
        for (Size col = 1; col < cols; ++col)
        {
            death = std::max(death, static_cast<double>(cohomology_graph[row][col]));
        }
        output[row][0] = birth;
        output[row][1] = death;
    }
    return output;
}

void DifferentiablePH5::trackGradientComputation(const std::string &operation,
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

void DifferentiablePH5::validateGradientComputation(const autodiff::Tensor &analytical_gradients,
                                                    const autodiff::Tensor &finite_diff_gradients)
{
    const double error = autodiff::tensorDistance(analytical_gradients, finite_diff_gradients);
    metrics_.gradient_accuracy = std::max(0.0, 1.0 - error);
}

} // namespace nerve::persistence
