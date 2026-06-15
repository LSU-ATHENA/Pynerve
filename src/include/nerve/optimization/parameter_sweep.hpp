
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
namespace nerve
{
namespace optimization
{
struct Parameter
{
    std::string name;
    double min_value;
    double max_value;
    size_t num_steps;
    std::string scale_type;
    std::vector<double> custom_values;
    std::vector<double> generateValues() const;
    bool isValid() const;
};
struct ParameterCombination
{
    std::unordered_map<std::string, double> values;
    std::array<uint8_t, 32> params_hash;
    uint64_t combination_id;
    std::vector<std::string> tags;
    std::vector<uint8_t> serialize() const;
    bool deserialize(const std::vector<uint8_t> &data);
    std::array<uint8_t, 32> computeHash() const;
    std::string toString() const;
};
struct SweepConfig
{
    std::vector<Parameter> parameters;
    size_t max_concurrent_evaluations = 4;
    bool enable_memoization = true;
    bool enableIntermediateReuse = true;
    size_t memoization_cache_size = 1000;
    std::string sweep_strategy = "grid";
    size_t max_total_combinations = 10000;
    bool enableEarlyStopping = true;
    double early_stopping_threshold = 0.01;
};
struct EvaluationResult
{
    ParameterCombination combination;
    std::vector<float> feature_vector;
    double evaluation_score;
    double computation_time_ms;
    uint64_t memory_usage_mb;
    bool success;
    std::string error_message;
    std::vector<std::string> intermediate_results;
    double feature_quality_score;
    double stability_score;
    double efficiency_score;
    std::vector<uint8_t> serialize() const;
    bool deserialize(const std::vector<uint8_t> &data);
};
using FeatureComputationFunction = std::function<EvaluationResult(
    const ParameterCombination &params, const std::vector<std::vector<float>> &input_data,
    const std::unordered_map<std::string, std::vector<uint8_t>> &intermediate_cache)>;
class ParameterSweepEngine
{
public:
    explicit ParameterSweepEngine(const SweepConfig &config);
    ~ParameterSweepEngine();
    std::vector<EvaluationResult> runSweep(const std::vector<std::vector<float>> &input_data,
                                           FeatureComputationFunction compute_function);
    std::future<std::vector<EvaluationResult>>
    runSweepAsync(const std::vector<std::vector<float>> &input_data,
                  FeatureComputationFunction compute_function);
    EvaluationResult evaluateCombination(const ParameterCombination &combination,
                                         const std::vector<std::vector<float>> &input_data,
                                         FeatureComputationFunction compute_function);
    std::vector<ParameterCombination> generateCombinations();
    std::vector<ParameterCombination> generateRandomCombinations(size_t count);
    std::vector<ParameterCombination>
    generateAdaptiveCombinations(const std::vector<EvaluationResult> &previous_results);
    void setMemoizationCache(size_t cache_size);
    void clearMemoizationCache();
    std::unordered_map<std::string, std::vector<uint8_t>> getMemoizationCache() const;
    void enableIntermediateReuse(bool enable);
    std::unordered_map<std::string, std::vector<uint8_t>> getIntermediateCache() const;
    struct SweepProgress
    {
        size_t total_combinations;
        size_t completed_combinations;
        size_t failed_combinations;
        double completion_percentage;
        double average_score;
        double best_score;
        ParameterCombination best_combination;
    };
    SweepProgress getProgress() const;
    void resetProgress();
    void enableEarlyStopping(bool enable);
    bool shouldStopEarly() const;

private:
    SweepConfig config_;
    std::unordered_map<std::string, std::vector<uint8_t>> memoization_cache_;
    mutable std::shared_mutex cache_mutex_;
    std::unordered_map<std::string, std::vector<uint8_t>> intermediate_cache_;
    mutable std::shared_mutex intermediate_mutex_;
    mutable std::mutex progress_mutex_;
    SweepProgress progress_;
    bool early_stop_triggered_ = false;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> sweep_running_{false};
    std::vector<ParameterCombination> generateGridCombinations();
    std::vector<ParameterCombination>
    generateBayesianCombinations(const std::vector<EvaluationResult> &previous_results);
    EvaluationResult evaluateWithMemoization(const ParameterCombination &combination,
                                             const std::vector<std::vector<float>> &input_data,
                                             FeatureComputationFunction compute_function);
    std::string getCacheKey(const ParameterCombination &combination) const;
    void updateProgress(const EvaluationResult &result);
    void checkEarlyStopping(const std::vector<EvaluationResult> &recent_results);
};
class SensitivityAnalyzer
{
public:
    struct SensitivityConfig
    {
        double sensitivity_threshold = 0.1;
        size_t num_perturbation_samples = 100;
        double perturbation_magnitude = 0.05;
        bool enable_global_sensitivity = true;
        bool enable_local_sensitivity = true;
    };
    explicit SensitivityAnalyzer(const SensitivityConfig &config);
    struct SensitivityResult
    {
        std::string parameter_name;
        double global_sensitivity;
        double local_sensitivity;
        std::vector<double> sensitivity_profile;
        double parameter_importance;
    };
    std::vector<SensitivityResult>
    analyzeSensitivity(const std::vector<Parameter> &parameters,
                       const std::vector<EvaluationResult> &evaluation_results);
    std::vector<std::pair<std::string, double>>
    rankParameterImportance(const std::vector<SensitivityResult> &sensitivity_results);
    std::string
    generateSensitivityReport(const std::vector<SensitivityResult> &sensitivity_results);

private:
    SensitivityConfig config_;
    double computeGlobalSensitivity(const std::string &parameter_name,
                                    const std::vector<EvaluationResult> &results);
    double computeLocalSensitivity(const std::string &parameter_name,
                                   const std::vector<EvaluationResult> &results);
    std::vector<double> computeSensitivityProfile(const std::string &parameter_name,
                                                  const std::vector<EvaluationResult> &results);
};
class OptimizationStrategies
{
public:
    enum class Strategy
    {
        GRID_SEARCH,
        RANDOM_SEARCH,
        BAYESIAN_OPTIMIZATION,
        GENETIC_ALGORITHM,
        SIMULATED_ANNEALING,
        ADAPTIVE_SAMPLING
    };
    struct OptimizationConfig
    {
        Strategy strategy = Strategy::GRID_SEARCH;
        size_t max_iterations = 100;
        double convergence_threshold = 1e-6;
        size_t population_size = 50;
        double mutation_rate = 0.1;
        double crossover_rate = 0.8;
        double temperature = 1.0;
    };
    explicit OptimizationStrategies(const OptimizationConfig &config);
    std::vector<ParameterCombination> optimize(const std::vector<Parameter> &parameters,
                                               FeatureComputationFunction objective_function,
                                               const std::vector<std::vector<float>> &input_data);
    std::vector<ParameterCombination>
    gridSearchOptimize(const std::vector<Parameter> &parameters,
                       FeatureComputationFunction objective_function,
                       const std::vector<std::vector<float>> &input_data);
    std::vector<ParameterCombination>
    bayesianOptimize(const std::vector<Parameter> &parameters,
                     FeatureComputationFunction objective_function,
                     const std::vector<std::vector<float>> &input_data);
    std::vector<ParameterCombination>
    geneticAlgorithmOptimize(const std::vector<Parameter> &parameters,
                             FeatureComputationFunction objective_function,
                             const std::vector<std::vector<float>> &input_data);

private:
    OptimizationConfig config_;
    class GaussianProcess;
    class AcquisitionFunction;
    std::unique_ptr<GaussianProcess> gp_;
    std::unique_ptr<AcquisitionFunction> acquisition_;
    struct Individual
    {
        ParameterCombination params;
        double fitness;
    };
    std::vector<Individual> initializePopulation(const std::vector<Parameter> &parameters);
    Individual crossover(const Individual &parent1, const Individual &parent2);
    Individual mutate(const Individual &individual, const std::vector<Parameter> &parameters);
    std::vector<Individual> selectParents(const std::vector<Individual> &population);
};
class SweepManager
{
public:
    static SweepManager &instance();
    void createSweep(const std::string &sweep_id, const SweepConfig &config);
    std::shared_ptr<ParameterSweepEngine> getSweep(const std::string &sweep_id);
    void removeSweep(const std::string &sweep_id);
    std::vector<std::string> getSweepIds() const;
    std::vector<EvaluationResult>
    runMultipleSweeps(const std::vector<std::string> &sweep_ids,
                      const std::vector<std::vector<float>> &input_data,
                      FeatureComputationFunction compute_function);
    std::vector<SensitivityAnalyzer::SensitivityResult>
    analyzeSensitivity(const std::string &sweep_id,
                       const std::vector<EvaluationResult> &evaluation_results);
    std::vector<ParameterCombination>
    optimizeParameters(const std::vector<Parameter> &parameters,
                       OptimizationStrategies::Strategy strategy,
                       FeatureComputationFunction objective_function,
                       const std::vector<std::vector<float>> &input_data);
    void storeResults(const std::string &sweep_id, const std::vector<EvaluationResult> &results);
    std::vector<EvaluationResult> getResults(const std::string &sweep_id);
    void exportResults(const std::string &sweep_id, const std::string &filename);
    std::string generateSweepReport(const std::string &sweep_id);
    std::string generateComparisonReport(const std::vector<std::string> &sweep_ids);

private:
    SweepManager() = default;
    std::unordered_map<std::string, std::shared_ptr<ParameterSweepEngine>> sweeps_;
    std::unordered_map<std::string, std::vector<EvaluationResult>> results_storage_;
    std::unique_ptr<SensitivityAnalyzer> sensitivity_analyzer_;
    std::unique_ptr<OptimizationStrategies> optimization_strategies_;
    mutable std::shared_mutex mutex_;
};
} // namespace optimization
} // namespace nerve
