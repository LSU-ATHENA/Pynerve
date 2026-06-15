
#pragma once

#include "nerve/errors/errors.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace nerve::runtime
{

struct CalibrationSample
{
    std::string schema_version = "v1";
    std::string hardware_fingerprint;
    std::string problem_bucket;
    std::string algorithm;

    double predicted_time_ms = 0.0;
    double predicted_memory_mb = 0.0;
    double observed_time_ms = 0.0;
    double observed_memory_mb = 0.0;

    double confidence = 0.0;
    double relative_error = 1.0;
    std::int64_t timestamp_unix_ms = 0;
};

struct PredictionKey
{
    std::string hardware_fingerprint;
    std::string problem_bucket;
    std::string algorithm;
};

struct PredictionWithBounds
{
    bool available = false;
    double predicted_time_ms = 0.0;
    double predicted_memory_mb = 0.0;
    double confidence = 0.0;
    double error_bound = 1.0;
    std::size_t sample_count = 0;
    std::string calibration_bucket_id;
};

struct SelectionGateDiagnostics
{
    bool gate_passed = false;
    std::string gate_reason;
    double confidence = 0.0;
    double error_bound = 1.0;
    std::size_t sample_count = 0;
    std::string calibration_bucket_id;
};

errors::ErrorResult<void> loadCalibrationModel();

errors::ErrorResult<void> recordCalibrationSample(const CalibrationSample &sample);

PredictionWithBounds queryCalibratedPrediction(const PredictionKey &key);

SelectionGateDiagnostics evaluatePredictionGate(const PredictionWithBounds &prediction);

std::string defaultCalibrationStorePath();

void setCalibrationStorePathForTesting(const std::string &path);

} // namespace nerve::runtime
