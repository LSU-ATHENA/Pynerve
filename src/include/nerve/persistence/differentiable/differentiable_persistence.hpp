
#pragma once

#include "nerve/autodiff/autodiff.hpp"
#include "nerve/core/budget.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::persistence
{

enum class DifferentiableAlgorithm
{
    PH5_COHOMOLOGY,
    PH6_WITNESS,
    AUTO_SELECT
};

struct AutodiffConfig
{
    bool enable_gradients = true;
    bool enable_higher_order = false;
    double gradient_epsilon = 1e-6;
    double gradient_tolerance = 1e-4;
    size_t max_gradient_iterations = 100;
    bool enable_gradient_clipping = true;
    double gradient_clip_value = 1.0;
    bool enable_mixed_precision = false;
};

struct DifferentiableMetrics
{
    double forward_pass_time = 0.0;
    double backward_pass_time = 0.0;
    double gradient_computation_time = 0.0;
    double finite_difference_time = 0.0;
    size_t gradient_computations = 0;
    size_t finite_difference_verifications = 0;
    double average_gradient_norm = 0.0;
    double max_gradient_norm = 0.0;
    double gradient_accuracy = 0.0;
    size_t memory_usage_mb = 0;
};

struct GradientAnalysis
{
    double gradient_norm = 0.0;
    double gradient_variance = 0.0;
    double gradient_sparsity = 0.0;
    std::vector<double> per_dimension_norms;
    std::vector<size_t> top_gradient_indices;
    double gradient_condition_number = 0.0;
    bool gradients_well_behaved = false;
};

struct DifferentiableConfig
{
    size_t max_dimension = 1;
    double landmark_ratio = 0.1;
    bool computeGradients = true;
    bool verifyGradients = false;
    double gradient_epsilon = 1e-6;
    double gradient_tolerance = 1e-4;
    DifferentiableAlgorithm algorithm = DifferentiableAlgorithm::AUTO_SELECT;
    AutodiffConfig autodiff_config;
};

struct OptimizationTarget
{
    std::string target_type;
    size_t target_dimension = 0;
    double target_value = 0.0;
    double weight = 1.0;
    std::unordered_map<std::string, double> constraints;
};

struct OptimizationResult
{
    autodiff::Tensor accelerated_points;
    double final_objective_value = 0.0;
    std::vector<double> objective_history;
    size_t optimization_steps = 0;
    double optimization_time = 0.0;
    bool converged = false;
    std::string convergence_reason;
};

struct AlgorithmComparison
{
    std::unordered_map<DifferentiableAlgorithm, double> computation_times;
    std::unordered_map<DifferentiableAlgorithm, double> memory_usage;
    std::unordered_map<DifferentiableAlgorithm, double> gradient_accuracies;
    DifferentiableAlgorithm recommended_algorithm = DifferentiableAlgorithm::AUTO_SELECT;
    std::string recommendation_reason;
};

struct VerificationResult
{
    bool passed = false;
    double max_error = 0.0;
    double mean_error = 0.0;
    double epsilon_used = 0.0;
    double tolerance_used = 0.0;
    size_t iterations = 0;
    std::vector<double> error_history;
};

struct VerificationReport
{
    size_t total_verifications = 0;
    size_t passed_verifications = 0;
    size_t failed_verifications = 0;
    double average_error = 0.0;
    double max_error_seen = 0.0;
    std::vector<VerificationResult> verification_history;
    std::unordered_map<std::string, double> error_statistics;
};

class DifferentiablePH5
{
public:
    explicit DifferentiablePH5(const PersistenceBudget &budget = PersistenceBudget{});

    autodiff::Tensor computePersistenceCohomologyAutodiff(const autodiff::Tensor &points,
                                                          size_t max_dimension,
                                                          bool computeGradients = true);

    autodiff::Tensor computeGradients(const autodiff::Tensor &points,
                                      const autodiff::Tensor &persistence_diagram,
                                      size_t target_dimension);

    bool verifyFiniteDifferences(const autodiff::Tensor &points, size_t max_dimension,
                                 double epsilon = 1e-6, double tolerance = 1e-4);

    void setAutodiffConfig(const AutodiffConfig &config);
    const AutodiffConfig &getAutodiffConfig() const;
    void enableGradientChecking(bool enable);
    bool isGradientCheckingEnabled() const;
    const DifferentiableMetrics &getDifferentiableMetrics() const;
    void resetMetrics();
    GradientAnalysis analyzeGradients(const autodiff::Tensor &gradients, size_t dimension);

private:
    PersistenceBudget budget_;
    AutodiffConfig autodiff_config_;
    bool gradient_checking_enabled_ = true;
    DifferentiableMetrics metrics_;

    autodiff::Tensor buildCohomologyGraph(const autodiff::Tensor &points, size_t max_dimension);
    autodiff::Tensor computeCohomologyReductionAutodiff(const autodiff::Tensor &cohomology_graph);
    void trackGradientComputation(const std::string &operation, double computation_time);
    void validateGradientComputation(const autodiff::Tensor &analytical_gradients,
                                     const autodiff::Tensor &finite_diff_gradients);
};

class DifferentiablePH6
{
public:
    explicit DifferentiablePH6(const PersistenceBudget &budget = PersistenceBudget{});

    autodiff::Tensor computePersistenceWitnessAutodiff(const autodiff::Tensor &points,
                                                       size_t max_dimension,
                                                       double landmark_ratio = 0.1,
                                                       bool computeGradients = true);

    autodiff::Tensor computeLandmarkGradients(const autodiff::Tensor &points,
                                              const autodiff::Tensor &landmark_weights,
                                              size_t target_dimension);

    bool verifyFiniteDifferences(const autodiff::Tensor &points, size_t max_dimension,
                                 double epsilon = 1e-6, double tolerance = 1e-4);

    autodiff::Tensor optimizeLandmarkSelection(const autodiff::Tensor &points, size_t max_dimension,
                                               size_t num_landmarks,
                                               size_t optimization_steps = 100);

    void setAutodiffConfig(const AutodiffConfig &config);
    const AutodiffConfig &getAutodiffConfig() const;
    void enableGradientChecking(bool enable);
    bool isGradientCheckingEnabled() const;
    const DifferentiableMetrics &getDifferentiableMetrics() const;
    void resetMetrics();

private:
    PersistenceBudget budget_;
    AutodiffConfig autodiff_config_;
    bool gradient_checking_enabled_ = true;
    DifferentiableMetrics metrics_;

    autodiff::Tensor buildWitnessGraphAutodiff(const autodiff::Tensor &points,
                                               const autodiff::Tensor &landmark_weights);
    autodiff::Tensor computeWitnessComplexAutodiff(const autodiff::Tensor &witness_graph);
    void trackGradientComputation(const std::string &operation, double computation_time);
    void validateGradientComputation(const autodiff::Tensor &analytical_gradients,
                                     const autodiff::Tensor &finite_diff_gradients);
};

class DifferentiablePersistenceManager
{
public:
    explicit DifferentiablePersistenceManager(
        const PersistenceBudget &budget = PersistenceBudget{});

    autodiff::Tensor computePersistenceAutodiff(const autodiff::Tensor &points,
                                                const DifferentiableConfig &config);

    void setPreferredAlgorithm(DifferentiableAlgorithm algorithm);
    DifferentiableAlgorithm getPreferredAlgorithm() const;
    DifferentiableAlgorithm selectOptimalAlgorithm(size_t num_points, size_t max_dimension,
                                                   const PersistenceBudget &budget);

    AlgorithmComparison compareAlgorithms(const autodiff::Tensor &points, size_t max_dimension);

    std::vector<autodiff::Tensor>
    computeBatchPersistence(const std::vector<autodiff::Tensor> &point_clouds,
                            const DifferentiableConfig &config);

    OptimizationResult optimizePersistence(const autodiff::Tensor &points,
                                           const OptimizationTarget &target,
                                           size_t optimization_steps = 100);

private:
    PersistenceBudget budget_;
    DifferentiableAlgorithm preferred_algorithm_ = DifferentiableAlgorithm::AUTO_SELECT;
    std::unique_ptr<DifferentiablePH5> ph5_;
    std::unique_ptr<DifferentiablePH6> ph6_;

    autodiff::Tensor computeWithPh5(const autodiff::Tensor &points,
                                    const DifferentiableConfig &config);
    autodiff::Tensor computeWithPh6(const autodiff::Tensor &points,
                                    const DifferentiableConfig &config);
};

class FiniteDifferenceVerifier
{
public:
    FiniteDifferenceVerifier(double default_epsilon = 1e-6, double default_tolerance = 1e-4);

    bool verifyGradients(const std::function<autodiff::Tensor(const autodiff::Tensor &)> &func,
                         const autodiff::Tensor &input,
                         const autodiff::Tensor &analytical_gradients, double epsilon = -1.0,
                         double tolerance = -1.0);

    bool verifyPersistenceGradients(DifferentiablePH5 &ph5, const autodiff::Tensor &points,
                                    size_t max_dimension);

    bool verifyPersistenceGradients(DifferentiablePH6 &ph6, const autodiff::Tensor &points,
                                    size_t max_dimension);

    VerificationResult
    adaptiveVerification(const std::function<autodiff::Tensor(const autodiff::Tensor &)> &func,
                         const autodiff::Tensor &input,
                         const autodiff::Tensor &analytical_gradients);

    void setEpsilon(double epsilon);
    void setTolerance(double tolerance);
    void setMaxIterations(size_t max_iterations);
    VerificationReport generateReport() const;

private:
    double default_epsilon_;
    double default_tolerance_;
    size_t max_iterations_;
    VerificationReport report_;

    autodiff::Tensor
    computeFiniteDifferences(const std::function<autodiff::Tensor(const autodiff::Tensor &)> &func,
                             const autodiff::Tensor &input, double epsilon);
    double computeGradientError(const autodiff::Tensor &analytical,
                                const autodiff::Tensor &finite_diff);
    bool adjustParameters(double &epsilon, double &tolerance, double error);
};

} // namespace nerve::persistence
