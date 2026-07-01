
#include "nerve/persistence/adaptive_acceleration/adaptive_algorithm_selector.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_selector_calibration.hpp"

#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{

using nerve::persistence::adaptive_acceleration::adaptiveAlgorithmKey;
using nerve::persistence::adaptive_acceleration::AdaptiveAlgorithmType;
using nerve::persistence::adaptive_acceleration::PerformancePredictor;
using nerve::persistence::adaptive_acceleration::problemBucketId;
using nerve::persistence::adaptive_acceleration::ProblemCharacteristics;

bool check_adaptive_algorithm_key_matrix_multiplication()
{
    auto key = adaptiveAlgorithmKey(AdaptiveAlgorithmType::MATRIX_MULTIPLICATION_CPU);
    if (key != "matrix_multiplication_cpu")
        return false;
    return true;
}

bool check_adaptive_algorithm_key_sparsified()
{
    auto key = adaptiveAlgorithmKey(AdaptiveAlgorithmType::SPARSIFIED_REDUCTION_CPU);
    if (key != "sparsified_reduction_cpu")
        return false;
    return true;
}

bool check_adaptive_algorithm_key_lockfree()
{
    auto key = adaptiveAlgorithmKey(AdaptiveAlgorithmType::LOCKFREE_MULTICORE_CPU);
    if (key != "lockfree_multicore_cpu")
        return false;
    return true;
}

bool check_adaptive_algorithm_key_gpu()
{
    auto key = adaptiveAlgorithmKey(AdaptiveAlgorithmType::GPU_ACCELERATED);
    if (key != "gpu_accelerated")
        return false;
    return true;
}

bool check_adaptive_algorithm_key_hybrid()
{
    auto key = adaptiveAlgorithmKey(AdaptiveAlgorithmType::HYBRID_GPU_CPU);
    if (key != "hybrid_gpu_cpu")
        return false;
    return true;
}

bool check_adaptive_algorithm_key_standard()
{
    auto key = adaptiveAlgorithmKey(AdaptiveAlgorithmType::STANDARD_CPU);
    if (key != "standard_cpu")
        return false;
    return true;
}

bool check_problem_bucket_id()
{
    ProblemCharacteristics prob;
    prob.num_points = 100;
    prob.point_dim = 3;
    prob.max_simplex_size = 10;
    prob.estimated_columns = 500;
    auto id = problemBucketId(prob);
    if (id.empty())
        return false;
    if (id.find("100pts_500cols") == std::string::npos)
        return false;
    return true;
}

bool check_problem_bucket_id_zero_values()
{
    ProblemCharacteristics prob;
    prob.num_points = 0;
    prob.estimated_columns = 0;
    auto id = problemBucketId(prob);
    if (id.empty())
        return false;
    return true;
}

bool check_blend_prediction_no_crash()
{
    PerformancePredictor::Prediction pred;
    pred.algorithm_type = AdaptiveAlgorithmType::STANDARD_CPU;
    pred.estimated_time_ms = 100.0;
    pred.estimated_memory_mb = 256.0;
    ProblemCharacteristics prob;
    prob.num_points = 1000;
    prob.estimated_columns = 5000;
    blendPredictionWithCalibration(pred, prob, "test_fp");
    return true;
}

bool check_record_observation_no_crash()
{
    PerformancePredictor::Prediction pred;
    pred.algorithm_type = AdaptiveAlgorithmType::GPU_ACCELERATED;
    pred.estimated_time_ms = 50.0;
    pred.estimated_memory_mb = 1024.0;
    pred.confidence_score = 0.85;
    ProblemCharacteristics prob;
    prob.num_points = 500;
    prob.estimated_columns = 2000;
    recordAdaptiveCalibrationObservation(prob, AdaptiveAlgorithmType::GPU_ACCELERATED, &pred, 123.4,
                                         500 * 3);
    return true;
}

bool check_record_observation_null_prediction()
{
    ProblemCharacteristics prob;
    prob.num_points = 100;
    prob.estimated_columns = 500;
    recordAdaptiveCalibrationObservation(prob, AdaptiveAlgorithmType::STANDARD_CPU, nullptr, 10.0,
                                         300);
    return true;
}

} // namespace

int main()
{
    if (!check_adaptive_algorithm_key_matrix_multiplication())
    {
        std::cerr << "FAIL: adaptive algorithm key matrix multiplication\n";
        return 1;
    }
    if (!check_adaptive_algorithm_key_sparsified())
    {
        std::cerr << "FAIL: adaptive algorithm key sparsified\n";
        return 1;
    }
    if (!check_adaptive_algorithm_key_lockfree())
    {
        std::cerr << "FAIL: adaptive algorithm key lockfree\n";
        return 1;
    }
    if (!check_adaptive_algorithm_key_gpu())
    {
        std::cerr << "FAIL: adaptive algorithm key gpu\n";
        return 1;
    }
    if (!check_adaptive_algorithm_key_hybrid())
    {
        std::cerr << "FAIL: adaptive algorithm key hybrid\n";
        return 1;
    }
    if (!check_adaptive_algorithm_key_standard())
    {
        std::cerr << "FAIL: adaptive algorithm key standard\n";
        return 1;
    }
    if (!check_problem_bucket_id())
    {
        std::cerr << "FAIL: problem bucket id\n";
        return 1;
    }
    if (!check_problem_bucket_id_zero_values())
    {
        std::cerr << "FAIL: problem bucket id zero values\n";
        return 1;
    }
    if (!check_blend_prediction_no_crash())
    {
        std::cerr << "FAIL: blend prediction no crash\n";
        return 1;
    }
    if (!check_record_observation_no_crash())
    {
        std::cerr << "FAIL: record observation no crash\n";
        return 1;
    }
    if (!check_record_observation_null_prediction())
    {
        std::cerr << "FAIL: record observation null prediction\n";
        return 1;
    }
    return 0;
}
