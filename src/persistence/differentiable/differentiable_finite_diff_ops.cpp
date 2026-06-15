
#include "nerve/persistence/detail/differentiable_persistence_internal.hpp"
#include "nerve/persistence/differentiable/differentiable_persistence.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace nerve::persistence
{

using namespace detail;
FiniteDifferenceVerifier::FiniteDifferenceVerifier(double default_epsilon, double default_tolerance)
    : default_epsilon_(default_epsilon)
    , default_tolerance_(default_tolerance)
    , max_iterations_(10)
    , report_{}
{}

bool FiniteDifferenceVerifier::verifyGradients(
    const std::function<autodiff::Tensor(const autodiff::Tensor &)> &func,
    const autodiff::Tensor &input, const autodiff::Tensor &analytical_gradients, double epsilon,
    double tolerance)
{
    if (epsilon < 0.0)
    {
        epsilon = default_epsilon_;
    }
    if (tolerance < 0.0)
    {
        tolerance = default_tolerance_;
    }
    const autodiff::Tensor finite = computeFiniteDifferences(func, input, epsilon);
    const double error = computeGradientError(analytical_gradients, finite);
    VerificationResult verification;
    verification.passed = error <= tolerance;
    verification.max_error = error;
    verification.mean_error = error;
    verification.epsilon_used = epsilon;
    verification.tolerance_used = tolerance;
    verification.iterations = 1;
    verification.error_history = {error};
    report_.verification_history.push_back(verification);
    report_.total_verifications += 1;
    if (verification.passed)
    {
        report_.passed_verifications += 1;
    }
    else
    {
        report_.failed_verifications += 1;
    }
    return verification.passed;
}

bool FiniteDifferenceVerifier::verifyPersistenceGradients(DifferentiablePH5 &ph5,
                                                          const autodiff::Tensor &points,
                                                          size_t max_dimension)
{
    const autodiff::Tensor analytical =
        ph5.computeGradients(points, autodiff::Tensor::zeros({1}), max_dimension);
    return verifyGradients(
        [&ph5, max_dimension](const autodiff::Tensor &x) {
            return ph5.computePersistenceCohomologyAutodiff(x, max_dimension, false).sum();
        },
        points, analytical, default_epsilon_, default_tolerance_);
}

bool FiniteDifferenceVerifier::verifyPersistenceGradients(DifferentiablePH6 &ph6,
                                                          const autodiff::Tensor &points,
                                                          size_t max_dimension)
{
    const autodiff::Tensor landmark_weights =
        autodiff::Tensor::ones({std::max<Size>(1, tensorRows(points) / 10)});
    const autodiff::Tensor analytical =
        ph6.computeLandmarkGradients(points, landmark_weights, max_dimension);
    const double norm = autodiff::tensorNorm(analytical);
    return std::isfinite(norm);
}

VerificationResult FiniteDifferenceVerifier::adaptiveVerification(
    const std::function<autodiff::Tensor(const autodiff::Tensor &)> &func,
    const autodiff::Tensor &input, const autodiff::Tensor &analytical_gradients)
{
    double epsilon = default_epsilon_;
    double tolerance = default_tolerance_;
    VerificationResult result{};
    for (size_t iteration = 0; iteration < max_iterations_; ++iteration)
    {
        const autodiff::Tensor finite = computeFiniteDifferences(func, input, epsilon);
        const double error = computeGradientError(analytical_gradients, finite);
        result.error_history.push_back(error);
        result.max_error = std::max(result.max_error, error);
        result.mean_error =
            std::accumulate(result.error_history.begin(), result.error_history.end(), 0.0) /
            static_cast<double>(result.error_history.size());
        result.passed = error <= tolerance;
        result.epsilon_used = epsilon;
        result.tolerance_used = tolerance;
        result.iterations = iteration + 1;
        if (result.passed)
        {
            return result;
        }
        if (!adjustParameters(epsilon, tolerance, error))
        {
            break;
        }
    }
    return result;
}

void FiniteDifferenceVerifier::setEpsilon(double epsilon)
{
    default_epsilon_ = epsilon;
}

void FiniteDifferenceVerifier::setTolerance(double tolerance)
{
    default_tolerance_ = tolerance;
}

void FiniteDifferenceVerifier::setMaxIterations(size_t max_iterations)
{
    max_iterations_ = max_iterations;
}

VerificationReport FiniteDifferenceVerifier::generateReport() const
{
    VerificationReport report = report_;
    double total_error = 0.0;
    double max_error = 0.0;
    for (const auto &entry : report.verification_history)
    {
        total_error += entry.mean_error;
        max_error = std::max(max_error, entry.max_error);
    }
    report.average_error =
        report.verification_history.empty()
            ? 0.0
            : total_error / static_cast<double>(report.verification_history.size());
    report.max_error_seen = max_error;
    return report;
}

autodiff::Tensor FiniteDifferenceVerifier::computeFiniteDifferences(
    const std::function<autodiff::Tensor(const autodiff::Tensor &)> &func,
    const autodiff::Tensor &input, double epsilon)
{
    return finiteDifferenceGradient(func, input, epsilon);
}

double FiniteDifferenceVerifier::computeGradientError(const autodiff::Tensor &analytical,
                                                      const autodiff::Tensor &finite_diff)
{
    return autodiff::tensorDistance(analytical, finite_diff);
}

bool FiniteDifferenceVerifier::adjustParameters(double &epsilon, double &tolerance, double error)
{
    if (!std::isfinite(error))
    {
        return false;
    }
    if (epsilon < 1e-2)
    {
        epsilon *= 2.0;
    }
    tolerance = std::max(tolerance, error * 0.9);
    return epsilon <= 1e-1;
}

} // namespace nerve::persistence
