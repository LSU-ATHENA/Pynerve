
#include "nerve/persistence/adaptive_acceleration/adaptive_selector_calibration.hpp"
#include "nerve/runtime/calibration_model.hpp"
#include "nerve/runtime/hardware_probe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>

namespace nerve::persistence::adaptive_acceleration
{

std::string adaptiveAlgorithmKey(AdaptiveAlgorithmType type)
{
    switch (type)
    {
        case AdaptiveAlgorithmType::MATRIX_MULTIPLICATION_CPU:
            return "matrix_multiplication_cpu";
        case AdaptiveAlgorithmType::SPARSIFIED_REDUCTION_CPU:
            return "sparsified_reduction_cpu";
        case AdaptiveAlgorithmType::LOCKFREE_MULTICORE_CPU:
            return "lockfree_multicore_cpu";
        case AdaptiveAlgorithmType::GPU_ACCELERATED:
            return "gpu_accelerated";
        case AdaptiveAlgorithmType::HYBRID_GPU_CPU:
            return "hybrid_gpu_cpu";
        case AdaptiveAlgorithmType::STANDARD_CPU:
        default:
            return "standard_cpu";
    }
}

std::string problemBucketId(const ProblemCharacteristics &problem)
{
    return std::to_string(problem.num_points) + "pts_" + std::to_string(problem.estimated_columns) +
           "cols";
}

void blendPredictionWithCalibration(PerformancePredictor::Prediction &prediction,
                                    const ProblemCharacteristics &problem,
                                    const std::string &hardware_fingerprint)
{
    const auto load_status = runtime::loadCalibrationModel();
    if (load_status.isError())
    {
        return;
    }
    runtime::PredictionKey key;
    key.hardware_fingerprint = hardware_fingerprint;
    key.problem_bucket = problemBucketId(problem);
    key.algorithm = adaptiveAlgorithmKey(prediction.algorithm_type);
    const runtime::PredictionWithBounds cal = runtime::queryCalibratedPrediction(key);
    if (!cal.available || cal.sample_count == 0)
    {
        return;
    }
    const double w = std::min(0.85, 0.25 * std::sqrt(static_cast<double>(cal.sample_count)));
    prediction.estimated_time_ms =
        w * cal.predicted_time_ms + (1.0 - w) * prediction.estimated_time_ms;
    prediction.estimated_memory_mb =
        w * cal.predicted_memory_mb + (1.0 - w) * prediction.estimated_memory_mb;
}

void recordAdaptiveCalibrationObservation(
    const ProblemCharacteristics &problem, AdaptiveAlgorithmType selected,
    const PerformancePredictor::Prediction *selected_prediction, double elapsed_ms,
    std::size_t point_buffer_elements)
{
    const runtime::HardwareSnapshot snap = runtime::collectHardwareSnapshot();
    const std::string fp = runtime::getHardwareFingerprint(snap);
    runtime::CalibrationSample sample;
    sample.hardware_fingerprint = fp;
    sample.problem_bucket = problemBucketId(problem);
    sample.algorithm = adaptiveAlgorithmKey(selected);
    if (selected_prediction != nullptr)
    {
        sample.predicted_time_ms = selected_prediction->estimated_time_ms;
        sample.predicted_memory_mb = selected_prediction->estimated_memory_mb;
        sample.confidence = selected_prediction->confidence_score;
    }
    sample.observed_time_ms = elapsed_ms;
    sample.observed_memory_mb =
        static_cast<double>(point_buffer_elements * sizeof(double)) / (1024.0 * 1024.0);
    sample.relative_error =
        elapsed_ms > 1e-9 ? std::abs(sample.predicted_time_ms - elapsed_ms) / elapsed_ms : 0.0;
    sample.timestamp_unix_ms =
        static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count());
    const auto record_status = runtime::recordCalibrationSample(sample);
    if (record_status.isError())
    {
        return;
    }
}

} // namespace nerve::persistence::adaptive_acceleration
