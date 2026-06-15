
#pragma once

#include "nerve/core.hpp"

#include <memory>
#include <string>
#include <vector>

namespace nerve::persistence
{
namespace gpu
{

#ifdef __CUDACC__
struct AlgorithmInfo
{
    std::string name;
    float base_complexity;
    float memory_overhead;
    int max_dimension;
    bool gpu_capable;
    float accuracy_score;
};

struct SelectorConfig
{
    int num_algorithms = 5;
    int feature_dim = 20;
    int hidden_dim = 64;
};

struct Prediction
{
    int selected_algorithm;
    float confidence;
    std::vector<float> algorithm_scores;
};

class GPUAdaptiveSelector
{
public:
    explicit GPUAdaptiveSelector(const SelectorConfig &config);
    ~GPUAdaptiveSelector();

    void registerAlgorithm(const AlgorithmInfo &info);

    void
    extractFeaturesFromPersistence(const std::vector<std::pair<float, float>> &persistence_pairs,
                                   const std::vector<int> &dimensions);

    Prediction predict();
    std::vector<Prediction> batchPredict(const std::vector<std::vector<float>> &feature_batches);

    void trainStep(const std::vector<float> &features, int correct_algorithm, float learning_rate);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

struct SelectorBenchmark
{
    double cpu_feature_ms;
    double gpu_feature_ms;
    double cpu_predict_ms;
    double gpu_predict_ms;
    double speedup_feature;
    double speedup_predict;
    int num_pairs;
    int num_algorithms;
};

SelectorBenchmark benchmarkSelector(int num_pairs, int num_algorithms);

} // namespace gpu
} // namespace nerve::persistence
