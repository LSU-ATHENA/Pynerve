
#pragma once

#include "adaptive_algorithm_selector.hpp"

#include <string>

namespace nerve::persistence::adaptive_acceleration
{

std::string adaptiveAlgorithmKey(AdaptiveAlgorithmType type);

std::string problemBucketId(const ProblemCharacteristics &problem);

void blendPredictionWithCalibration(PerformancePredictor::Prediction &prediction,
                                    const ProblemCharacteristics &problem,
                                    const std::string &hardware_fingerprint);

void recordAdaptiveCalibrationObservation(
    const ProblemCharacteristics &problem, AdaptiveAlgorithmType selected,
    const PerformancePredictor::Prediction *selected_prediction, double elapsed_ms,
    std::size_t point_buffer_elements);

} // namespace nerve::persistence::adaptive_acceleration
